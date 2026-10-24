# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""One persistent GPU worker: a survey's target image from FITS bytes to a
scored catalog.

The worker is created by cuPhoton's shared executor after it has bound the
process to one
GPU. A survey adapter (``transient_chain.surveys``) turns the selection into
device-resident
target planes and a template provider; from there the chain is survey-free:
bad pixels to NaN,
star-based affine refinement, compact-source fit mask, constant-kernel XPOIS,
hotspot
detection, stamp cutting, xFit dipole fits, and XScan inference through
DLPack. The only host
traffic is FITS headers on the way in, small star lists and scalars for
control, and one
packed catalog on the way out. The only files written are that catalog and a
summary.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import importlib
import os
import socket
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from . import ops
from .catalog import empty_table, finish_table
from .config import config_sha256, validate_config
from .fitsio import PlaneReader
from .surveys import make_adapter

OPTIONS_SCHEMA = "transient-chain.worker-options/v2"
RESULT_SCHEMA = "transient-chain.item-result/v2"
STAGES = (
    "read",
    "template",
    "refine",
    "mask",
    "xpois",
    "detect",
    "stamps",
    "xfit",
    "uncertainty",
    "xscan",
    "terminal_d2h",
    "write",
)


class ChainWorker:
    """Worker for cuPhoton's shared executor. See the module docstring."""

    def __init__(self, options: Mapping[str, Any]) -> None:
        if options.get("schema") != OPTIONS_SCHEMA:
            raise ValueError(
                f"unsupported transient-chain worker options schema "
                f"{options.get('schema')!r}; expected {OPTIONS_SCHEMA}"
            )
        self.cfg = validate_config(options["config"])
        self.config_sha256 = config_sha256(self.cfg)
        torch = importlib.import_module(
            "torch"
        )  # Torch first: it fixes the CUDA library load order
        cp = importlib.import_module("cupy")
        torch.cuda.set_device(0)
        cp.cuda.Device(0).use()
        self.cp, self.torch = cp, torch
        from cuphoton.core.bulk import collect_gpu_identity
        from cuphoton.xfit import GaussianDipoleModel, LMConfig

        self.gpu_identity = collect_gpu_identity("cupy")
        self.device = torch.device("cuda:0")
        self.adapter = make_adapter(self.cfg["survey"])
        self.io = PlaneReader(self.cfg["io"]["reader"], device=True)
        size = int(self.cfg["stamps"]["stamp_size"])
        self.compute_dtype = np.dtype(self.cfg["xfit"]["compute_dtype"])
        self.dipole_model = GaussianDipoleModel(
            (size, size), backend="cupy", dtype=self.compute_dtype
        )  # built once, on this GPU
        self.lm_config = LMConfig(**self.cfg["xfit"]["solver"])
        self.parameter_names = tuple(self.dipole_model.parameter_names)
        self.model = None
        self.performance = None
        self.inference_policy = None
        self.checkpoint_sha256 = options.get("checkpoint_sha256")
        if self.cfg["classify"]["checkpoint_dir"]:
            self._load_model(Path(self.cfg["classify"]["checkpoint_dir"]))
        self.consumer_stream = torch.cuda.Stream(device=self.device)
        self.closed = False
        self.items_done = 0

    def _load_model(self, checkpoint_dir: Path) -> None:
        from cuphoton.xscan.config import PerformanceConfig
        from cuphoton.xscan.device_pipeline import (
            _strict_inference_policy_payload,
        )
        from cuphoton.xscan.training import (
            _validate_model_device,
            load_model_from_checkpoint,
        )

        ckpt = checkpoint_dir / "checkpoint.pt"
        digest = hashlib.sha256(ckpt.read_bytes()).hexdigest()
        if self.checkpoint_sha256 and digest != self.checkpoint_sha256:
            raise RuntimeError("XScan checkpoint changed before worker load")
        policy = self.cfg["classify"]["inference_policy"]
        override = (
            PerformanceConfig(**_strict_inference_policy_payload())
            if policy == "strict"
            else None
        )
        model, checkpoint, performance = load_model_from_checkpoint(
            checkpoint_dir, device=self.device, performance_override=override
        )
        self.inference_policy = {
            "requested": policy,
            "amp_dtype": str(getattr(performance, "amp_dtype", None)),
            "allow_tf32": bool(getattr(performance, "allow_tf32", False)),
            "cudnn_benchmark": bool(
                getattr(performance, "cudnn_benchmark", False)
            ),
        }
        mc = checkpoint.get("model_config", {})
        stamp = int(self.cfg["stamps"]["stamp_size"])
        if mc.get("input_mode") != "triplet":
            raise ValueError(
                "this chain scores stamp triplets; the checkpoint's "
                "input_mode is "
                f"{mc.get('input_mode')!r}. Train or pick a triplet image-"
                f"only checkpoint."
            )
        if int(mc.get("image_size", -1)) != stamp:
            raise ValueError(
                f"checkpoint image_size {mc.get('image_size')} != "
                f"stamps.stamp_size {stamp}; "
                "set stamps.stamp_size to the checkpoint's image_size or use "
                "a matching checkpoint"
            )
        if mc.get("xfit_feature_names"):
            raise ValueError(
                "this chain scores image-only checkpoints; this one expects "
                "xFit fusion features"
            )
        _validate_model_device(model, device=self.device)
        self.model, self.performance, self.checkpoint_sha256 = (
            model,
            performance,
            digest,
        )

    # ------------------------------------------------------------------ item
    def run_item(self, item, output_dir: Path, /) -> Mapping[str, Any]:
        if self.closed:
            raise RuntimeError("worker is closed")
        cp, torch, cfg = self.cp, self.torch, self.cfg
        sel = dict(item.payload)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        t_item = time.perf_counter()
        timings: dict[str, float] = {}
        diag: dict[str, Any] = {}
        xfer = {
            "terminal_bytes": 0,
            "control_bytes": 0,
            "diagnostic_bytes": 0,
        }
        identity = self.adapter.identity(sel)
        a_cfg, s_cfg, d_cfg, c_cfg = (
            cfg["align"],
            cfg["subtract"],
            cfg["detect"],
            cfg["classify"],
        )
        size = int(cfg["stamps"]["stamp_size"])
        half = size // 2

        from cuphoton.xfit import (
            DipoleQualityFlag,
            derived_dipole_quantities,
            dipole_quality_flags,
            fit_dipoles_device,
            forced_psf_photometry,
        )
        from cuphoton.xpois import (
            GaussianBasisComponent,
            solve_constant_kernel_device,
        )

        timer = _StageTimer(timings, cp)

        # 1. the survey adapter reads the target planes through the selected
        # FITS reader
        self.io.receipts = []
        with timer("read"):
            target = self.adapter.load_target(sel, self.io)
        ny, nx = target.shape
        vimg = target.image
        diag["read"] = {
            "shape": [ny, nx],
            "image_dtype": str(vimg.dtype),
            "mask_bits": target.mask_bits,
            "has_mask": target.bad is not None,
            "has_variance": target.variance is not None,
            "receipts": list(self.io.receipts),
        }

        # 2. template on the target grid (mosaic, reprojected reference, or
        # supplied grid image)
        with timer("template"):
            provider = self.adapter.template(
                sel,
                target,
                self.io,
                mapping_grid_step=int(a_cfg["mapping_grid_step"]),
            )
            timg, tbad_raw, tcnt = provider.build()
        w = int(a_cfg["xcorr_window_px"])
        y0, x0 = ny // 2 - w // 2, nx // 2 - w // 2
        win = (slice(y0, y0 + w), slice(x0, x0 + w))
        diag["read"]["receipts"] = list(self.io.receipts)
        diag["align"] = {
            "footprint_fraction": float(cp.isfinite(timg).mean()),
            "xcorr_visit_minus_template_px": ops.xcorr_offset(
                vimg[win], timg[win]
            ),
            "template_refinable": bool(provider.refinable),
            "template_coverage_max": None
            if tcnt is None
            else int(tcnt.max()),
        }
        xfer["diagnostic_bytes"] += w * w * 8

        # 3. star-based affine refinement: only two small star lists cross to
        # the host
        with timer("refine"):
            ref_diag: dict[str, Any] = {
                "mode": "off"
                if not a_cfg["refine"]
                else (
                    "template not refinable"
                    if not provider.refinable
                    else "skipped"
                )
            }
            if a_cfg["refine"] and provider.refinable:
                stars = dict(
                    fwhm=float(a_cfg["star_fwhm_px"]),
                    nsigma=float(a_cfg["star_threshold_sigma"]),
                    max_n=int(a_cfg["max_stars"]),
                )
                finder_name = a_cfg["star_finder"]
                if finder_name == "daofind":
                    finder = ops.find_stars_daofind
                else:
                    finder = functools.partial(
                        ops.find_stars,
                        box=int(a_cfg["star_box_px"]),
                        iterations=int(a_cfg["centroid_iterations"]),
                    )
                sv, st = finder(vimg, **stars), finder(timg, **stars)
                xfer["control_bytes"] += sv.nbytes + st.nbytes
                if finder_name == "daofind":
                    # DAOStarFinder ran on host copies of both images (and
                    # the rebuilt one)
                    xfer["control_bytes"] += 3 * int(vimg.nbytes)
                mv, mt = ops.match_stars(
                    sv, st, float(a_cfg["match_radius_px"])
                )
                A = None
                if len(mv) >= int(a_cfg["min_affine_matches"]):
                    A, b, rms = ops.fit_affine(mv, mt)
                    mode = "affine"
                elif len(mv) >= int(a_cfg["min_shift_matches"]):
                    A, b, rms, mode = (
                        np.eye(2),
                        np.median(mt - mv, axis=0),
                        float(ops.robust_sigma(mt - mv)),
                        "shift",
                    )
                if A is None:
                    ref_diag = {
                        "mode": "skipped",
                        "matches_before": int(len(mv)),
                        "stars_visit": int(len(sv)),
                        "stars_template": int(len(st)),
                    }
                else:
                    timg, tbad_raw, tcnt = provider.build((A, b))
                    mv1, mt1 = ops.match_stars(
                        sv,
                        finder(timg, **stars),
                        float(a_cfg["verify_radius_px"]),
                    )
                    before, after = mv - mt, mv1 - mt1
                    ref_diag = {
                        "mode": mode,
                        "matches_before": int(len(mv)),
                        "matches_after": int(len(mv1)),
                        "affine": {
                            "A": A.tolist(),
                            "b": b.tolist(),
                            "fit_rms_px": rms,
                        },
                        "star_offset_before_px": _offset_stats(before),
                        "star_residual_after_px": _offset_stats(after)
                        if len(after)
                        else None,
                        "xcorr_visit_minus_template_px_after": (
                            ops.xcorr_offset(vimg[win], timg[win])
                        ),
                    }
                    xfer["diagnostic_bytes"] += w * w * 8
            ref_diag["sign_convention"] = "visit minus template, pixels"
            diag["align"]["refine"] = ref_diag
            diag["align"]["star_finder"] = (
                (
                    "daofind-host"
                    if a_cfg["star_finder"] == "daofind"
                    else "gpu-windowed-centroid"
                )
                if a_cfg["refine"]
                else None
            )

        # 4. bad pixels of either image become NaN; the target's variance
        # follows its mask
        with timer("mask"):
            from .surveys.base import dilate_bool

            ref_bad = ~cp.isfinite(timg)
            if tbad_raw is not None:
                tbad = tbad_raw if tbad_raw.dtype == bool else (tbad_raw != 0)
                ref_bad |= dilate_bool(
                    tbad, int(s_cfg["template_bad_dilation_px"]), cp
                )
            tgt_bad = ~cp.isfinite(vimg)
            if target.bad is not None:
                tgt_bad |= target.bad
            reference = cp.where(ref_bad, cp.nan, timg).astype(cp.float64)
            search = cp.where(tgt_bad, cp.nan, vimg).astype(cp.float64)
            variance = (
                None
                if target.variance is None
                else cp.where(tgt_bad, cp.nan, target.variance)
            )
            fit_mask = None
            stamp_diag = None
            if s_cfg["auto_stamp_mask"]:
                both = cp.isfinite(reference) & cp.isfinite(search)
                selection = cp.where(both, 0.5 * (reference + search), cp.nan)
                fit_mask, stamp_diag = ops.compact_source_stamp_mask(
                    selection,
                    variance=variance,
                    stamp_size=int(s_cfg["auto_stamp_size"]),
                    max_stamps=int(s_cfg["auto_stamp_count"]),
                    peak_percentile=float(s_cfg["auto_peak_percentile"]),
                    background_filter_size=int(
                        s_cfg["auto_background_filter_size"]
                    ),
                    peak_filter_size=int(s_cfg["auto_peak_filter_size"]),
                    compact_core_size=int(s_cfg["auto_compact_core_size"]),
                )
            diag["mask"] = {
                "reference_bad_fraction": float(ref_bad.mean()),
                "target_bad_fraction": float(tgt_bad.mean()),
                "template_bad_dilation_px": int(
                    s_cfg["template_bad_dilation_px"]
                ),
                "auto_stamp_mask": stamp_diag,
            }
            del ref_bad

        # 5. constant-kernel XPOIS on the device
        with timer("xpois"):
            components = [
                GaussianBasisComponent(sigma=float(s), degree=int(d))
                for s, d in zip(
                    s_cfg["basis_sigmas"], s_cfg["basis_degrees"], strict=True
                )
            ]
            fit = solve_constant_kernel_device(
                reference,
                search,
                components,
                kernel_shape=(
                    int(s_cfg["kernel_height"]),
                    int(s_cfg["kernel_width"]),
                ),
                variance=variance,
                fit_mask=fit_mask,
                background_degree=int(s_cfg["background_degree"]),
                flux_conserve=bool(s_cfg["flux_conserve"]),
            )
            residual = fit.residual
            rsigma = ops.robust_sigma(residual[fit.fit_mask])
            diag["xpois"] = {
                "chi2": float(fit.chi2),
                "dof": int(fit.dof),
                "fit_pixel_count": int(fit.fit_pixel_count),
                "kernel_sum": float(fit.kernel.sum()),
                "robust_sigma": rsigma,
                "residual_finite_fraction": float(
                    cp.isfinite(residual).mean()
                ),
            }

        # 6. detection: hotspot rule, then the filters (size, edge margin, bad
        # target pixel, finite residual)
        with timer("detect"):
            spots = ops.hotspots(
                residual,
                robust_sigma=rsigma,
                threshold_sigma=float(d_cfg["threshold_sigma"]),
            )
            n_spots = int(spots["pixel_count"].size)
            xfer["control_bytes"] += sum(
                int(v.nbytes) for v in spots.values()
            )
            if n_spots:
                cy = np.rint(spots["centroid_y"]).astype(int)
                cx = np.rint(spots["centroid_x"]).astype(int)
                keep = (
                    (spots["pixel_count"] >= int(d_cfg["min_pixels"]))
                    & (cy >= half)
                    & (cy <= ny - half - 1)
                    & (cx >= half)
                    & (cx <= nx - half - 1)
                )
                cy_d, cx_d = cp.asarray(cy[keep]), cp.asarray(cx[keep])
                ok = cp.isfinite(residual[cy_d, cx_d]) & ~tgt_bad[cy_d, cx_d]
                ok_h = ops.to_host(ok)
                xfer["control_bytes"] += int(ok_h.nbytes)
                idx = np.flatnonzero(keep)[ok_h]
            else:
                idx = np.empty(0, int)
            cand = {k: v[idx] for k, v in spots.items()}
            cand_x, cand_y = (
                np.rint(cand["centroid_x"]).astype(int),
                np.rint(cand["centroid_y"]).astype(int),
            )
            n_cand = int(len(idx))
            diag["detect"] = {
                "hotspots": n_spots,
                "candidates": n_cand,
                "threshold_sigma": float(d_cfg["threshold_sigma"]),
                "min_pixels": int(d_cfg["min_pixels"]),
                "edge_margin_px": half,
            }
            del tgt_bad

        table: dict[str, np.ndarray] = empty_table(self.parameter_names)
        if n_cand:
            # 7. stamps: search, search minus difference, difference; plus the
            # noise-match factor
            with timer("stamps"):
                stamps, difference, fit_valid, fit_variance = (
                    ops.extract_stamps(
                        vimg, residual, cand_x, cand_y, size, variance
                    )
                )
                if c_cfg["apply_noise_match"]:
                    factor, n_replaced = ops.noise_match_factors(
                        stamps[:, 0], float(c_cfg["noise_reference_sigma"])
                    )
                else:
                    factor, n_replaced = cp.ones(n_cand, cp.float32), 0
                diag["stamps"] = {"noise_factors_replaced": int(n_replaced)}
            # 8. xFit Gaussian dipoles on the float64 difference stamps,
            # weighted by the target variance; masked and nonfinite pixels
            # are excluded
            with timer("xfit"):
                xres = fit_dipoles_device(
                    difference.astype(self.compute_dtype, copy=False),
                    model=self.dipole_model,
                    mode=cfg["xfit"]["mode"],
                    mask=fit_valid,
                    variance=None
                    if fit_variance is None
                    else fit_variance.astype(self.compute_dtype, copy=False),
                    config=self.lm_config,
                )
            # 8b. quality flags, catalog quantities with uncertainties
            # propagated through the full covariance, and forced PSF
            # photometry at the source when the PSF width is configured
            with timer("uncertainty"):
                psf_sigma = cfg["xfit"]["psf_sigma_px"]
                qflags = dipole_quality_flags(
                    xres,
                    difference,
                    variance=fit_variance,
                    mask=fit_valid,
                    **(
                        {}
                        if psf_sigma is None
                        else {
                            "width_range": (0.5 * psf_sigma, 2.5 * psf_sigma)
                        }
                    ),
                )
                derived = derived_dipole_quantities(xres)
                photometry = None
                # only with an independent PSF width: the fitted dipole
                # width would make psf_flux_err far too small
                if fit_variance is not None and psf_sigma is not None:
                    positive = int(
                        DipoleQualityFlag.POSITIVE_LOBE_UNSUPPORTED
                    )
                    negative = int(
                        DipoleQualityFlag.NEGATIVE_LOBE_UNSUPPORTED
                    )
                    fading = ((qflags & positive) != 0) & (
                        (qflags & negative) == 0
                    )
                    source = cp.stack(
                        [
                            cp.where(fading, 2.0 * c - p, p)
                            for c, p in (
                                (derived.center_x, derived.positive_x),
                                (derived.center_y, derived.positive_y),
                            )
                        ],
                        axis=1,
                    )
                    photometry = forced_psf_photometry(
                        difference,
                        fit_variance,
                        source,
                        psf_sigma,
                        mask=fit_valid,
                    )
            # 9. XScan through DLPack: the scaled stamps never leave the
            # device
            logits_t = probs_t = view = scaled = None
            if self.model is not None:
                with timer("xscan"):
                    from cuphoton.xscan.training import (
                        cupy_to_torch,
                        predict_tensors,
                    )

                    scaled = cp.ascontiguousarray(
                        stamps * factor[:, None, None, None]
                    )
                    with torch.cuda.stream(self.consumer_stream):
                        with _timer(timings, "xscan_handoff"):
                            view = cupy_to_torch(scaled, device=self.device)
                        bs = int(c_cfg["batch_size"])
                        with _timer(timings, "xscan_forward"):
                            outs = [
                                predict_tensors(
                                    model=self.model,
                                    images=view.tensor[
                                        i : i + bs
                                    ].contiguous(),
                                    device=self.device,
                                    performance=self.performance,
                                )
                                for i in range(0, n_cand, bs)
                            ]
                            logits_t = torch.cat([o["logits"] for o in outs])
                            probs_t = torch.cat(
                                [o["probabilities"] for o in outs]
                            )
                        with _timer(timings, "xscan_sync"):
                            self.consumer_stream.synchronize()
            # 10. the one terminal transfer: catalog columns
            with timer("terminal_d2h"):
                table = {
                    "x": cand_x,
                    "y": cand_y,
                    "peak_abs_sigma": cand["peak_abs_sigma"],
                    "pixel_count": cand["pixel_count"].astype(int),
                    "peak_value": cand["peak_residual"],
                    "sign": np.sign(cand["peak_residual"]).astype(int),
                    "noise_match_factor": ops.to_host(factor),
                }
                params, errs = (
                    ops.to_host(xres.parameters),
                    ops.to_host(xres.standard_errors),
                )
                for j, name in enumerate(xres.parameter_names):
                    table[name] = params[:, j]
                    table[f"{name}_err"] = errs[:, j]
                for name in (
                    "converged",
                    "chi_square",
                    "reduced_chi_square",
                    "uncertainty_valid",
                    "evaluations",
                    "valid_pixel_count",
                ):
                    table[name] = ops.to_host(getattr(xres, name))
                table["quality_flags"] = ops.to_host(qflags)
                for name in ("center", "positive"):
                    for axis in ("x", "y"):
                        table[f"{name}_{axis}"] = ops.to_host(
                            getattr(derived, f"{name}_{axis}")
                        )
                        table[f"{name}_{axis}_err"] = ops.to_host(
                            getattr(derived, f"{name}_{axis}_error")
                        )
                    table[f"{name}_xy_cov"] = ops.to_host(
                        getattr(derived, f"{name}_xy_covariance")
                    )
                for name in ("separation", "lobe_flux"):
                    table[name] = ops.to_host(getattr(derived, name))
                    table[f"{name}_err"] = ops.to_host(
                        getattr(derived, f"{name}_error")
                    )
                if photometry is not None:
                    table["psf_flux"] = ops.to_host(photometry.flux)
                    table["psf_flux_err"] = ops.to_host(photometry.flux_error)
                if self.model is not None:
                    table["logit"] = logits_t.detach().cpu().numpy()
                    table["probability"] = probs_t.detach().cpu().numpy()
                    table["decision"] = table["probability"] >= float(
                        c_cfg["decision_threshold"]
                    )
                    del view
                sky = target.wcs.all_pix2world(
                    np.column_stack([cand_x, cand_y]).astype(float), 0
                )
                table["ra"], table["dec"] = sky[:, 0], sky[:, 1]
                xfer["terminal_bytes"] += sum(
                    int(np.asarray(v).nbytes) for v in table.values()
                )
                del xres, stamps, difference, scaled, derived, photometry
        else:
            for k in (
                "stamps",
                "xfit",
                "uncertainty",
                "xscan",
                "terminal_d2h",
            ):
                timings[k] = 0.0

        # 11. write: the catalog and the summary are the only files this item
        # produces
        with timer("write"):
            df = finish_table(
                table,
                n_cand,
                identity.columns(),
                self.adapter.astrometry_label,
                identity.exposure_id,
                identity.ccd_id,
                self.parameter_names,
            )
            catalog_path = output_dir / "catalog.parquet"
            df.to_parquet(catalog_path, index=False)
            del (
                fit,
                residual,
                reference,
                search,
                variance,
                vimg,
                timg,
                target,
                provider,
            )
            cp.get_default_memory_pool().free_all_blocks()
        # Snapshot the completed catalog write; the summary cannot time its
        # own serialization.
        wall = time.perf_counter() - t_item
        summary = {
            "schema": RESULT_SCHEMA,
            "item_id": item.item_id,
            "identity": identity.columns(),
            "survey": self.adapter.name,
            "reader_policy": self.io.reader,
            "candidates": n_cand,
            "catalog": str(catalog_path),
            "diagnostics": diag,
            "timings_sec": {k: float(timings.get(k, 0.0)) for k in STAGES},
            "wall_sec": wall,
            "timings_detail_sec": {
                k: float(v) for k, v in timings.items() if k not in STAGES
            },
            "transfers": {
                **{k: int(v) for k, v in xfer.items()},
                "intermediate_files": 0,
            },
            "config_sha256": self.config_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "inference_policy": self.inference_policy,
            "gpu": self.gpu_identity,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
        }
        summary_path = output_dir / "summary.json"
        _write_json(summary_path, summary)
        self.items_done += 1
        return {
            "run_dir": str(output_dir),
            "summary_path": str(summary_path),
            "backend": "cupy",
            "device": self.gpu_identity.get("name", "cuda:0"),
            "runtime": {"cupy": cp.__version__, "torch": torch.__version__},
            "timings_sec": summary["timings_sec"],
            "wall_sec": {"item": wall},
            "candidates": n_cand,
        }

    def close(self) -> None:
        self.closed = True
        self.cp.cuda.get_current_stream().synchronize()
        self.consumer_stream.synchronize()


class _StageTimer:
    """Per-stage wall time including the device work issued in the stage
    (synchronizes on exit).
    """

    def __init__(self, timings: dict[str, float], cp) -> None:
        self.timings, self.cp = timings, cp

    @contextlib.contextmanager
    def __call__(self, name: str):
        t = time.perf_counter()
        try:
            yield
        finally:
            self.cp.cuda.get_current_stream().synchronize()
            self.timings[name] = (
                self.timings.get(name, 0.0) + time.perf_counter() - t
            )


@contextlib.contextmanager
def _timer(timings: dict[str, float], name: str):
    t = time.perf_counter()
    try:
        yield
    finally:
        timings[name] = timings.get(name, 0.0) + time.perf_counter() - t


def _offset_stats(d: np.ndarray) -> dict[str, float]:
    return {
        "median_dx": float(np.median(d[:, 0])),
        "median_dy": float(np.median(d[:, 1])),
        "rms": float(np.sqrt(np.mean(np.sum(d**2, axis=1)))),
    }


def _jsonable(o):
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        v = float(o)
        return None if not np.isfinite(v) else v
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(
        f"summary value of type {type(o).__name__} is not JSON-serializable"
    )


def _sanitize(o):
    if isinstance(o, Mapping):
        return {str(k): _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    if isinstance(o, (str, bool, int)) or o is None:
        return o
    return (
        _jsonable(o)
        if not isinstance(o, float)
        else (o if np.isfinite(o) else None)
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Durable JSON through the library's atomic writer; NaN and inf become
    null first.
    """
    from cuphoton.core.bulk import atomic_write_json

    atomic_write_json(path, _sanitize(payload))


def create_worker(options: Mapping[str, Any]) -> ChainWorker:
    """Executor entry point: package-importable factory called inside the
    bound worker process.
    """
    return ChainWorker(options)
