# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Torch backend implementation."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import DTypeLike

from cuphoton.ximage.backends.torch_gpu_preprocess import (
    prepare_torch_reconstruction_inputs,
)
from cuphoton.ximage.data import XImageArray, XImageBundle
from cuphoton.ximage.psf import gaussian_psf as gaussian_psf_fn
from cuphoton.ximage.robust import irls_weights, rho_values


class TorchBackend:
    name = "torch"

    def is_available(self) -> bool:
        try:
            import torch  # noqa: F401
        except Exception:
            return False
        return True

    def to_torch_images(self, images: np.ndarray):
        torch = _load_torch()
        return (
            torch.from_numpy(np.asarray(images))
            .permute(3, 0, 1, 2)
            .contiguous()
        )

    def to_torch_psfs(self, psfs: np.ndarray):
        torch = _load_torch()
        return (
            torch.from_numpy(np.asarray(psfs))
            .permute(3, 2, 0, 1)
            .contiguous()
        )

    def to_torch_masks(self, masks: np.ndarray):
        torch = _load_torch()
        arr = np.asarray(masks)
        if arr.ndim != 4:
            raise ValueError(
                f"Expected a rank-4 good-pixel mask, saw rank {arr.ndim}"
            )
        return torch.from_numpy(arr).permute(3, 0, 1, 2).contiguous()

    def solve_psf(self, bundle: Any, config: Any) -> dict[str, Any]:
        torch = _load_torch()
        F = torch.nn.functional

        f_np = np.asarray(bundle[config.psf_artifact], dtype=np.float64)
        if f_np.ndim != 4:
            raise ValueError(
                f"Expected rank-4 PSF stack, saw shape {f_np.shape}"
            )
        if f_np.shape[0] != f_np.shape[1] or f_np.shape[0] % 2 == 0:
            raise ValueError(
                "PSF targets must be square with odd spatial dimensions; "
                f"saw shape {f_np.shape[:2]}."
            )
        if f_np.shape[2] != 1:
            raise NotImplementedError(
                "Torch PSF solving currently supports "
                "single-channel PSFs only, saw "
                f"channels={f_np.shape[2]}."
            )

        num_psfs = f_np.shape[3]
        f_size = f_np.shape[0]
        h_size = _round_down_to_odd(config.srf * f_size)
        g_size = config.g_size or h_size

        device = _resolve_device(torch, getattr(config, "device", "auto"))
        dtype = _resolve_dtype(torch, getattr(config, "dtype", "float64"))
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        started = _start_timed_section(torch, device)
        if config.method == "mm":
            if config.loss != "L2":
                raise ValueError("method='mm' supports only loss='L2'")
            result = _solve_psf_mm(
                torch,
                F,
                f_np,
                config,
                device=device,
                dtype=dtype,
                h_size=h_size,
                g_size=g_size,
            )
            result["performance"] = _psf_performance_summary(
                torch,
                device,
                config,
                elapsed_sec=_stop_timed_section(torch, device, started),
                num_psfs=num_psfs,
                target_shape=f_np.shape,
            )
            return result
        if config.method != "adam":
            raise ValueError(f"Unsupported PSF solve method: {config.method}")

        g_total_size = (
            g_size + h_size - 1 if config.padding == "VALID" else g_size
        )

        gaussian_psf = np.expand_dims(
            gaussian_psf_fn(
                g_total_size,
                sigma=config.sigma_g * config.srf,
                offset=(0.0, 0.0),
            ),
            axis=(0, -1),
        )

        target = (
            torch.from_numpy(f_np[:, :, 0, :])
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device=device, dtype=dtype)
        )
        g_input = (
            torch.from_numpy(gaussian_psf)
            .permute(0, 3, 1, 2)
            .to(device=device, dtype=dtype)
        )

        h_param = torch.nn.Parameter(
            _initialize_h(
                config.h_initialization,
                num_psfs=num_psfs,
                h_size=h_size,
                dtype=np.float64,
                seed=config.seed,
            ).to(device=device, dtype=dtype)
        )

        optimizer = torch.optim.Adam([h_param], lr=config.lr)
        trace_loss: list[float] = []
        trace_iterations: list[int] = []

        initial_pred = _forward_psf_bank(
            F, h_param, g_input, config.srf, config.padding
        )
        if initial_pred.shape != target.shape:
            raise RuntimeError(
                "Torch PSF solve produced shape "
                f"{tuple(initial_pred.shape)} but expected "
                f"{tuple(target.shape)}."
            )

        for step in range(int(config.num_steps)):
            optimizer.zero_grad()
            pred = _forward_psf_bank(
                F, h_param, g_input, config.srf, config.padding
            )

            if config.loss == "L2":
                loss = torch.sum((target - pred) ** 2)
            elif config.loss == "L1":
                loss = torch.sum(torch.abs(target - pred))
            else:
                raise ValueError(
                    f"Unsupported loss for torch PSF solve: {config.loss}"
                )

            if config.grad_reg_coeff:
                loss = loss + config.grad_reg_coeff * _gradient_penalty(
                    F, h_param
                )

            loss.backward()
            optimizer.step()

            trace_iterations.append(step)
            trace_loss.append(float(loss.detach().cpu().item()))

        # Adam fits correlation kernels; public PSFs are point-source images.
        h_out = (
            h_param.detach()
            .flip((-2, -1))
            .cpu()
            .squeeze(1)
            .permute(1, 2, 0)
            .unsqueeze(2)
            .numpy()
        )
        result = {
            "psfs": h_out,
            "gaussian_psf": gaussian_psf,
            "trace": {
                "iterations": np.asarray(trace_iterations, dtype=np.int64),
                "loss": np.asarray(trace_loss, dtype=np.float64),
            },
        }
        result["performance"] = _psf_performance_summary(
            torch,
            device,
            config,
            elapsed_sec=_stop_timed_section(torch, device, started),
            num_psfs=num_psfs,
            target_shape=f_np.shape,
        )
        return result

    def reconstruct(self, bundle: Any, config: Any) -> dict[str, Any]:
        if config.algorithm not in {"l2", "rho"}:
            raise NotImplementedError(
                "Torch backend currently implements `l2` and "
                "`rho` reconstruction only."
            )
        if config.background_model not in {"none", "scalar_per_exposure"}:
            raise NotImplementedError(
                "Torch reconstruction supports `background_model='none'` "
                "and `background_model='scalar_per_exposure'` only."
            )
        if config.tiling.enabled:
            return self._reconstruct_tiled(bundle, config)

        torch = _load_torch()
        F = torch.nn.functional
        device = _resolve_device(torch, getattr(config, "device", "auto"))
        dtype = _resolve_dtype(torch, getattr(config, "dtype", "float64"))
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        total_start = _start_timed_section(torch, device)
        psf_start = _start_timed_section(torch, device)
        psfs = _load_psfs(
            bundle,
            config.psf_path,
            exposure_indices=config.data.exposure_indices,
            max_exposures=config.data.max_exposures,
            target_exposure_count=bundle.images.shape[3],
        )
        timings = {"psf_load": _stop_timed_section(torch, device, psf_start)}
        prepared = prepare_torch_reconstruction_inputs(
            bundle,
            config,
            device=device,
            dtype=dtype,
            psfs=psfs,
        )
        timings.update(prepared.metadata["timings_sec"])

        x = prepared.x0
        y = prepared.y
        m = prepared.m
        v = prepared.v
        w = prepared.w
        f = prepared.f
        total_exposures = y.shape[1]
        exposure_batch_size = _resolve_exposure_batch_size(
            config.exposure_batch_size,
            total_exposures=total_exposures,
        )

        m_normalization = float(torch.sum(m).detach().cpu().item())
        if m_normalization == 0:
            raise RuntimeError("All pixels are masked; cannot reconstruct.")

        start = _start_timed_section(torch, device)
        if config.algorithm == "l2" and config.background_model == "none":
            u_num = _accumulate_transpose_sum(
                F,
                f,
                w * y,
                srf=config.srf,
                padding=config.padding,
                exposure_batch_size=exposure_batch_size,
            )
        effective = _accumulate_transpose_sum(
            F,
            f,
            m,
            srf=config.srf,
            padding=config.padding,
            exposure_batch_size=exposure_batch_size,
        )
        effective = effective / total_exposures
        m_effective = (effective > config.effective_mask_mu).to(dtype)
        m_effective_normalization = torch.sum(m_effective)
        if float(m_effective_normalization.detach().cpu().item()) == 0:
            m_effective_normalization = torch.tensor(
                1.0, device=device, dtype=dtype
            )
        timings["initialize"] = timings.get(
            "initialize", 0.0
        ) + _stop_timed_section(torch, device, start)

        trace: dict[str, list[float | np.ndarray]] = {
            "iterations": [],
            "loss": [],
            "u_score": [],
        }
        if config.background_model != "none":
            trace["background"] = []
        u_score = abs(10 * config.stop_tol)
        delta_schedule = np.linspace(
            config.delta_rho_start, config.delta_rho_end, config.max_iter
        )
        previous_u_clipped = None
        timings["ximage_iterations"] = 0.0
        timings["loss_evaluation"] = 0.0

        for step in range(int(config.max_iter)):
            if u_score < config.stop_tol:
                break

            trace["iterations"].append(step)
            iteration_start = _start_timed_section(torch, device)
            if config.algorithm == "l2":
                if config.background_model == "none":
                    u_denom = _accumulate_l2_denom(
                        F,
                        f,
                        w,
                        x,
                        srf=config.srf,
                        padding=config.padding,
                        exposure_batch_size=exposure_batch_size,
                    )
                else:
                    u_num, u_denom, background = _accumulate_l2_terms(
                        F,
                        f,
                        y,
                        w,
                        x,
                        background_model=config.background_model,
                        srf=config.srf,
                        padding=config.padding,
                        exposure_batch_size=exposure_batch_size,
                    )
                    trace["background"].append(
                        _to_numpy_background(background)
                    )
            else:
                delta_value = float(delta_schedule[step])
                u_num, u_denom, background = _accumulate_rho_terms(
                    F,
                    f,
                    y,
                    w,
                    v,
                    x,
                    background_model=config.background_model,
                    rho_loss=config.rho_loss,
                    delta_value=delta_value,
                    srf=config.srf,
                    padding=config.padding,
                    exposure_batch_size=exposure_batch_size,
                )
                if background is not None:
                    trace["background"].append(
                        _to_numpy_background(background)
                    )
            safe_denom = torch.where(
                torch.abs(u_denom) > 1e-12, u_denom, torch.ones_like(u_denom)
            )
            u = torch.where(
                torch.abs(u_denom) > 1e-12,
                u_num / safe_denom,
                torch.ones_like(u_num),
            )
            u_clipped = torch.pow(
                torch.clamp(u, min=1.0 / config.ucf, max=config.ucf),
                config.learning_rate,
            )
            x = x * u_clipped
            timings["ximage_iterations"] += _stop_timed_section(
                torch, device, iteration_start
            )

            loss_start = _start_timed_section(torch, device)
            loss = _accumulate_loss(
                F,
                f,
                y,
                w,
                m,
                v,
                x,
                algorithm=config.algorithm,
                rho_loss=config.rho_loss,
                background_model=config.background_model,
                delta_value=float(delta_schedule[step]),
                srf=config.srf,
                padding=config.padding,
                exposure_batch_size=exposure_batch_size,
                m_normalization=m_normalization,
            )
            trace["loss"].append(float(loss.detach().cpu().item()))
            timings["loss_evaluation"] += _stop_timed_section(
                torch, device, loss_start
            )

            if step > 1 and previous_u_clipped is not None:
                ratio = u_clipped / previous_u_clipped
                u_score_tensor = torch.abs(
                    (
                        torch.sum(m_effective * ratio)
                        / m_effective_normalization
                    )
                    - 1
                )
                u_score = float(u_score_tensor.detach().cpu().item())
                trace["u_score"].append(u_score)
            previous_u_clipped = u_clipped

        completed_trace: dict[str, np.ndarray | list[float | np.ndarray]] = {
            **trace,
            "iterations": np.asarray(trace["iterations"], dtype=np.int64),
            "loss": np.asarray(trace["loss"], dtype=np.float64),
            "u_score": np.asarray(trace["u_score"], dtype=np.float64),
        }
        if "background" in trace:
            completed_trace["background"] = np.asarray(
                trace["background"], dtype=np.float64
            )
        start = _start_timed_section(torch, device)
        image = _to_numpy_nhwc(x)
        timings["device_to_host_result"] = _stop_timed_section(
            torch, device, start
        )
        timings["total"] = _stop_timed_section(torch, device, total_start)
        performance = _reconstruction_performance_summary(
            torch,
            device,
            config,
            prepared.metadata,
            timings,
            completed_trace,
        )
        return {
            "image": image,
            "trace": completed_trace,
            "performance": performance,
        }

    def _reconstruct_tiled(self, bundle: Any, config: Any) -> dict[str, Any]:
        tiled_start = time.perf_counter()
        _validate_tiling_config(bundle, config)
        height = int(config.roi.height)
        width = int(config.roi.width)
        tile_height = int(config.tiling.tile_height)
        tile_width = int(config.tiling.tile_width)
        halo = int(config.tiling.halo)
        origin_y, origin_x = _roi_origin(
            config.roi.center,
            height=height,
            width=width,
            image_shape=bundle.images.shape[1:3],
        )

        merged = np.zeros(
            (bundle.images.shape[0], height, width, 1),
            dtype=np.dtype(config.dtype),
        )
        coverage = np.zeros((height, width), dtype=np.uint8)
        tile_records: list[dict[str, Any]] = []
        tile_config = config.model_copy(
            update={
                "roi": config.roi.model_copy(
                    update={"center": None, "height": None, "width": None}
                ),
                "tiling": config.tiling.model_copy(update={"enabled": False}),
            }
        )

        for core_y, core_x, expanded_y, expanded_x in _iter_tiles(
            height,
            width,
            tile_height=tile_height,
            tile_width=tile_width,
            halo=halo,
        ):
            source_y = slice(
                origin_y + expanded_y.start,
                origin_y + expanded_y.stop,
            )
            source_x = slice(
                origin_x + expanded_x.start,
                origin_x + expanded_x.stop,
            )
            tile_bundle = _slice_bundle_spatial(
                bundle,
                source_y=source_y,
                source_x=source_x,
                tile_record={
                    "core_yx": [
                        core_y.start,
                        core_y.stop,
                        core_x.start,
                        core_x.stop,
                    ],
                    "expanded_yx": [
                        expanded_y.start,
                        expanded_y.stop,
                        expanded_x.start,
                        expanded_x.stop,
                    ],
                },
            )
            tile_result = self.reconstruct(tile_bundle, tile_config)
            tile_image = np.asarray(
                tile_result["image"], dtype=np.dtype(config.dtype)
            )
            expected_shape = (
                bundle.images.shape[0],
                expanded_y.stop - expanded_y.start,
                expanded_x.stop - expanded_x.start,
                1,
            )
            if tile_image.shape != expected_shape:
                raise RuntimeError(
                    "Tiled reconstruction produced tile image shape "
                    f"{tile_image.shape}, expected {expected_shape}."
                )

            crop_y = slice(
                core_y.start - expanded_y.start,
                core_y.stop - expanded_y.start,
            )
            crop_x = slice(
                core_x.start - expanded_x.start,
                core_x.stop - expanded_x.start,
            )
            merged[:, core_y, core_x, :] = tile_image[:, crop_y, crop_x, :]
            coverage[core_y, core_x] += 1
            tile_record = {
                "core_yx": [
                    core_y.start,
                    core_y.stop,
                    core_x.start,
                    core_x.stop,
                ],
                "expanded_yx": [
                    expanded_y.start,
                    expanded_y.stop,
                    expanded_x.start,
                    expanded_x.stop,
                ],
                "source_yx": [
                    source_y.start,
                    source_y.stop,
                    source_x.start,
                    source_x.stop,
                ],
                "trace_summary": _tile_trace_summary(
                    tile_result.get("trace")
                ),
            }
            tile_performance = tile_result.get("performance")
            if isinstance(tile_performance, dict):
                tile_record["performance"] = {
                    "device": tile_performance.get("device"),
                    "timings_sec": tile_performance.get("timings_sec", {}),
                    "iteration_metrics": tile_performance.get(
                        "iteration_metrics", {}
                    ),
                }
            tile_records.append(tile_record)

        if not np.all(coverage == 1):
            raise RuntimeError(
                "Tiled reconstruction coverage must be exactly one for "
                "every output pixel."
            )
        tiling_summary = {
            "enabled": True,
            "tile_count": len(tile_records),
            "tile_height": tile_height,
            "tile_width": tile_width,
            "halo": halo,
            "roi_origin_yx": [origin_y, origin_x],
        }
        performance = _tiled_performance_summary(
            config,
            tile_records,
            tiling_summary=tiling_summary,
            elapsed_sec=time.perf_counter() - tiled_start,
        )
        return {
            "image": merged,
            "trace": {
                "tiling": tiling_summary,
                "tiles": tile_records,
            },
            "performance": performance,
        }


def _solve_psf_mm(
    torch,
    F,
    f_np: np.ndarray,
    config: Any,
    *,
    device,
    dtype,
    h_size: int,
    g_size: int,
) -> dict[str, Any]:
    """Solve an entire PSF bank with vectorized multiplicative updates."""

    num_psfs = int(f_np.shape[-1])
    h_pad_size = g_size - 1 if config.padding == "VALID" else 0
    padded_size = h_size + h_pad_size
    gaussian = gaussian_psf_fn(
        g_size,
        sigma=config.sigma_g * config.srf,
        offset=(0.0, 0.0),
    )
    gaussian_psf = np.expand_dims(gaussian, axis=(0, -1))
    kernel = (
        torch.from_numpy(gaussian)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device=device, dtype=dtype)
    )
    target = (
        torch.from_numpy(f_np[:, :, 0, :])
        .permute(2, 0, 1)
        .unsqueeze(1)
        .to(device=device, dtype=dtype)
    )
    h = _initialize_h(
        config.h_initialization,
        num_psfs=num_psfs,
        h_size=padded_size,
        dtype=np.float64,
        seed=config.seed,
    ).to(device=device, dtype=dtype)
    if config.h_initialization == "zeros":
        raise ValueError(
            "h_initialization='zeros' is not valid for multiplicative "
            "updates; use 'ones', 'random', 'gaussian', or 'dirac'."
        )
    if config.h_initialization == "dirac":
        h = h + torch.as_tensor(1e-6, device=device, dtype=dtype)
        h[:, :, padded_size // 2, padded_size // 2] = 1.0

    upsampled_target = _resize_psf(F, target, h_size)
    u_num = _conv_transpose_single(kernel, upsampled_target, config.padding)
    traces: list[dict[str, list[float]]] = [
        {"iterations": [], "loss": [], "u_score": []} for _ in range(num_psfs)
    ]
    active = torch.ones(num_psfs, device=device, dtype=torch.bool)
    previous_u = None
    previous_previous_u = None

    for step in range(int(config.max_iter)):
        if not bool(torch.any(active).item()):
            break
        next_active = active
        if (
            step >= int(config.min_iter)
            and previous_u is not None
            and previous_previous_u is not None
        ):
            current_sum = torch.sum(previous_u, dim=(1, 2, 3))
            previous_sum = torch.sum(previous_previous_u, dim=(1, 2, 3))
            scores = torch.abs(current_sum / previous_sum - 1.0)
            scores_cpu = scores.detach().cpu().numpy()
            active_cpu = active.detach().cpu().numpy()
            for index in range(num_psfs):
                if active_cpu[index]:
                    traces[index]["u_score"].append(float(scores_cpu[index]))
            next_active = active & (scores >= config.tol)
        prediction = _conv_single(F, kernel, h, config.srf, config.padding)
        upsampled_prediction = _resize_psf(F, prediction, h_size)
        u_denom = _conv_transpose_single(
            kernel, upsampled_prediction, config.padding
        )
        safe = torch.where(
            torch.abs(u_denom) > 1e-12,
            u_denom,
            torch.ones_like(u_denom),
        )
        update = torch.where(
            torch.abs(u_denom) > 1e-12,
            u_num / safe,
            torch.ones_like(u_num),
        )
        update = torch.clamp(update, 1.0 / config.ucf, config.ucf)
        update = torch.where(
            active[:, None, None, None], update, torch.ones_like(update)
        )
        h = h * update

        prediction = _conv_single(F, kernel, h, config.srf, config.padding)
        losses = torch.sum((target - prediction).square(), dim=(1, 2, 3))
        losses_cpu = losses.detach().cpu().numpy()
        active_cpu = active.detach().cpu().numpy()
        for index in range(num_psfs):
            if not active_cpu[index]:
                continue
            traces[index]["iterations"].append(step)
            traces[index]["loss"].append(float(losses_cpu[index]))

        active = next_active
        previous_previous_u = previous_u
        previous_u = update

    # MM optimizes h as an image filtered by a symmetric Gaussian, so its
    # orientation already matches the target point-source image.
    h_padded = (
        h.detach().cpu().squeeze(1).permute(1, 2, 0).unsqueeze(2).numpy()
    )
    crop = h_pad_size // 2
    h_out = h_padded if crop == 0 else h_padded[crop:-crop, crop:-crop, :, :]
    completed_traces = [
        {
            **trace,
            "iterations": np.asarray(trace["iterations"], dtype=np.int64),
            "loss": np.asarray(trace["loss"], dtype=np.float64),
            "u_score": np.asarray(trace["u_score"], dtype=np.float64),
        }
        for trace in traces
    ]
    return {
        "psfs": h_out,
        "padded_psfs": h_padded,
        "gaussian_psf": gaussian_psf,
        "trace": completed_traces,
    }


def _conv_single(F, kernel, images, srf: int, padding: str):
    conv_padding = "same" if padding == "SAME" else 0
    result = F.conv2d(images, kernel, padding=conv_padding)
    if srf > 1:
        result = _avg_pool2d_same(F, result, kernel_size=srf, stride=srf)
    return result


def _conv_transpose_single(kernel, values, padding: str):
    torch = _load_torch()
    F = torch.nn.functional
    if padding == "SAME":
        pad = ((kernel.shape[-2] - 1) // 2, (kernel.shape[-1] - 1) // 2)
    elif padding == "VALID":
        pad = (0, 0)
    else:
        raise ValueError("padding must be 'SAME' or 'VALID'")
    return F.conv_transpose2d(values, kernel, padding=pad)


def _resize_psf(F, value, size: int):
    if value.shape[-2:] == (size, size):
        return value
    return F.interpolate(value, size=(size, size), mode="nearest")


def _validate_tiling_config(bundle: Any, config: Any) -> None:
    if config.padding != "SAME":
        raise NotImplementedError(
            "Torch tiled reconstruction currently supports padding='SAME' "
            "only."
        )
    if int(config.srf) != 1:
        raise NotImplementedError(
            "Torch tiled reconstruction currently supports srf=1 only."
        )
    if config.background_model != "none":
        raise NotImplementedError(
            "Torch tiled reconstruction currently supports "
            "background_model='none' only."
        )
    if (
        config.roi.center is None
        or config.roi.height is None
        or config.roi.width is None
    ):
        raise ValueError(
            "Torch tiled reconstruction requires an explicit ROI center, "
            "height, and width."
        )
    if config.tiling.tile_height is None or config.tiling.tile_width is None:
        raise ValueError(
            "Torch tiled reconstruction requires tile_height and tile_width."
        )

    height = int(config.roi.height)
    width = int(config.roi.width)
    tile_height = int(config.tiling.tile_height)
    tile_width = int(config.tiling.tile_width)
    halo = int(config.tiling.halo)
    _require_even("roi.height", height)
    _require_even("roi.width", width)
    _require_even("tiling.tile_height", tile_height)
    _require_even("tiling.tile_width", tile_width)
    _require_even("tiling.halo", halo)
    if height % tile_height or width % tile_width:
        raise ValueError(
            "Torch tiled reconstruction requires ROI dimensions to be "
            "exact multiples of tile dimensions."
        )
    _roi_origin(
        config.roi.center,
        height=height,
        width=width,
        image_shape=bundle.images.shape[1:3],
    )


def _require_even(name: str, value: int) -> None:
    if value <= 0 and name != "tiling.halo":
        raise ValueError(f"{name} must be positive")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    if value % 2:
        raise ValueError(f"{name} must be even for tiled reconstruction")


def _roi_origin(
    center: tuple[int, int],
    *,
    height: int,
    width: int,
    image_shape: tuple[int, int],
) -> tuple[int, int]:
    origin_y = int(center[0]) - height // 2
    origin_x = int(center[1]) - width // 2
    if origin_y < 0 or origin_x < 0:
        raise ValueError("Tiled reconstruction ROI starts outside the image.")
    if (
        origin_y + height > image_shape[0]
        or origin_x + width > image_shape[1]
    ):
        raise ValueError(
            "Tiled reconstruction ROI extends outside the image."
        )
    return origin_y, origin_x


def _iter_tiles(
    height: int,
    width: int,
    *,
    tile_height: int,
    tile_width: int,
    halo: int,
):
    for y0 in range(0, height, tile_height):
        y1 = y0 + tile_height
        for x0 in range(0, width, tile_width):
            x1 = x0 + tile_width
            core_y = slice(y0, y1)
            core_x = slice(x0, x1)
            expanded_y = slice(max(0, y0 - halo), min(height, y1 + halo))
            expanded_x = slice(max(0, x0 - halo), min(width, x1 + halo))
            yield core_y, core_x, expanded_y, expanded_x


def _slice_bundle_spatial(
    bundle: Any,
    *,
    source_y: slice,
    source_x: slice,
    tile_record: dict[str, Any],
) -> XImageBundle:
    image_shape = tuple(bundle.images.shape[1:3])
    arrays: dict[str, XImageArray] = {}
    for key, arr in bundle.arrays.items():
        if arr.ndim >= 3 and tuple(arr.shape[1:3]) == image_shape:
            arrays[key] = arr[:, source_y, source_x, ...]
        else:
            arrays[key] = arr
    metadata = dict(bundle.metadata)
    metadata["spatial_tile"] = tile_record
    return XImageBundle(
        store_format=bundle.store_format,
        store_path=bundle.store_path,
        arrays=arrays,
        metadata=metadata,
        resources=bundle.resources,
    )


def _tile_trace_summary(trace: Any) -> dict[str, Any]:
    if not isinstance(trace, dict):
        return {"trace_type": type(trace).__name__}
    summary: dict[str, Any] = {}
    for key in ("iterations", "loss", "u_score"):
        value = trace.get(key)
        if value is None:
            continue
        arr = np.asarray(value)
        if not np.issubdtype(arr.dtype, np.number):
            continue
        summary[f"{key}_count"] = int(arr.size)
        if arr.size:
            summary[f"{key}_last"] = float(arr.reshape(-1)[-1])
    return summary


def _tiled_performance_summary(
    config: Any,
    tile_records: list[dict[str, Any]],
    *,
    tiling_summary: dict[str, Any],
    elapsed_sec: float,
) -> dict[str, Any]:
    tile_performance = [
        record["performance"]
        for record in tile_records
        if isinstance(record.get("performance"), dict)
    ]
    tile_totals = [
        float(performance.get("timings_sec", {}).get("total", 0.0))
        for performance in tile_performance
    ]
    device = tile_performance[0].get("device") if tile_performance else None
    return {
        "algorithm": config.algorithm,
        "backend": "torch",
        "dtype": config.dtype,
        "device": device,
        "input": {
            "srf": int(config.srf),
            "padding": config.padding,
            "background_model": config.background_model,
            "tiling": tiling_summary,
        },
        "timings_sec": {
            "tiles_total": float(sum(tile_totals)),
            "maximum_tile_total": max(tile_totals, default=0.0),
            "total": float(elapsed_sec),
        },
        "tile_count": len(tile_records),
    }


def _load_torch():
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            "xImage requires PyTorch. Install the cuPhoton torch extra "
            "with a PyTorch build appropriate for your device."
        ) from exc
    return torch


def _resolve_device(torch, requested: str):
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "cuda":
        raise RuntimeError(
            "CUDA was requested but Torch cannot access a CUDA device. "
            "Install the `gpu` extra and verify the NVIDIA driver."
        )
    return torch.device("cpu")


def _resolve_dtype(torch, requested: str):
    if requested == "float32":
        return torch.float32
    if requested == "float64":
        return torch.float64
    raise ValueError("dtype must be 'float32' or 'float64'")


def _start_timed_section(torch, device) -> float:
    _sync_if_cuda(torch, device)
    return time.perf_counter()


def _stop_timed_section(torch, device, start: float) -> float:
    _sync_if_cuda(torch, device)
    return time.perf_counter() - start


def _sync_if_cuda(torch, device) -> None:
    if getattr(device, "type", None) == "cuda":
        torch.cuda.synchronize(device)


def _reconstruction_performance_summary(
    torch,
    device,
    config: Any,
    preprocess_metadata: dict[str, Any],
    timings: dict[str, float],
    trace: dict[str, Any],
) -> dict[str, Any]:
    tensor_shapes = preprocess_metadata["tensor_shapes"]
    y_shape = tensor_shapes["y"]
    f_shape = tensor_shapes["f"]
    iterations = int(len(trace.get("iterations", [])))
    iteration_seconds = float(timings.get("ximage_iterations", 0.0))
    exposure_pixels = int(y_shape[1] * y_shape[2] * y_shape[3])
    exposure_pixels_per_second = (
        exposure_pixels * iterations / iteration_seconds
        if iteration_seconds > 0
        else None
    )
    seconds_per_iteration = (
        iteration_seconds / iterations if iterations > 0 else None
    )
    return {
        "algorithm": config.algorithm,
        "backend": "torch",
        "dtype": str(getattr(config, "dtype", "float64")),
        "device": _torch_device_summary(torch, device),
        "input": {
            "exposures": int(y_shape[1]),
            "roi_shape": [int(y_shape[2]), int(y_shape[3])],
            "psf_shape": list(f_shape),
            "mask_mode": config.mask.mode,
            "srf": int(config.srf),
            "padding": config.padding,
            "background_model": config.background_model,
            "exposure_batch_size": _resolve_exposure_batch_size(
                config.exposure_batch_size,
                total_exposures=int(y_shape[1]),
            ),
        },
        "preprocess": preprocess_metadata,
        "timings_sec": {
            key: float(value) for key, value in sorted(timings.items())
        },
        "iteration_metrics": {
            "iterations": iterations,
            "seconds_per_iteration": seconds_per_iteration,
            "exposure_pixels_per_second": exposure_pixels_per_second,
        },
    }


def _psf_performance_summary(
    torch,
    device,
    config: Any,
    *,
    elapsed_sec: float,
    num_psfs: int,
    target_shape: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "algorithm": f"psf-{config.method}",
        "backend": "torch",
        "dtype": str(getattr(config, "dtype", "float64")),
        "device": _torch_device_summary(torch, device),
        "input": {
            "psfs": int(num_psfs),
            "target_shape": [int(value) for value in target_shape],
            "srf": int(config.srf),
            "padding": config.padding,
        },
        "timings_sec": {"total": float(elapsed_sec)},
    }


def _torch_device_summary(torch, device) -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    benchmark_limit = torch.backends.cudnn.benchmark_limit
    summary = {
        "torch": getattr(torch, "__version__", None),
        "torch_cuda": getattr(torch.version, "cuda", None),
        "cuda_available": cuda_available,
        "device": str(device),
        "device_name": None,
        "max_memory_allocated_bytes": 0,
        "max_memory_reserved_bytes": 0,
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_benchmark_limit": (
            int(benchmark_limit) if benchmark_limit is not None else None
        ),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
    }
    if getattr(device, "type", None) != "cuda":
        return summary
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    summary["device_name"] = torch.cuda.get_device_name(index)
    summary["max_memory_allocated_bytes"] = int(
        torch.cuda.max_memory_allocated(device)
    )
    summary["max_memory_reserved_bytes"] = int(
        torch.cuda.max_memory_reserved(device)
    )
    return summary


def _round_down_to_odd(n: float) -> int:
    if n <= 0:
        raise ValueError(f"Expected positive size, got {n}")
    if (n % 2) == 1:
        return int(n)
    return int(np.ceil(n) // 2 * 2 - 1)


def _initialize_h(
    initialization: str,
    *,
    num_psfs: int,
    h_size: int,
    dtype: DTypeLike,
    seed: int,
):
    torch = _load_torch()
    if initialization == "ones":
        return torch.ones((num_psfs, 1, h_size, h_size), dtype=torch.float64)
    if initialization == "zeros":
        return torch.zeros((num_psfs, 1, h_size, h_size), dtype=torch.float64)
    if initialization == "random":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        return torch.rand(
            (num_psfs, 1, h_size, h_size),
            dtype=torch.float64,
            generator=generator,
        )
    if initialization == "gaussian":
        h0 = gaussian_psf_fn(h_size, sigma=1.0, offset=(0.0, 0.0))
        return (
            torch.from_numpy(h0.astype(dtype))
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(num_psfs, 1, 1, 1)
        )
    if initialization == "dirac":
        h0 = torch.zeros((num_psfs, 1, h_size, h_size), dtype=torch.float64)
        h0[:, :, h_size // 2, h_size // 2] = 1.0
        return h0
    raise ValueError(
        f"Unsupported h_initialization for torch PSF solve: {initialization}"
    )


def _resolve_exposure_batch_size(
    exposure_batch_size: int | None, *, total_exposures: int
) -> int:
    if exposure_batch_size is None or exposure_batch_size >= total_exposures:
        return total_exposures
    return exposure_batch_size


def _iter_exposure_slices(total_exposures: int, exposure_batch_size: int):
    for start in range(0, total_exposures, exposure_batch_size):
        stop = min(start + exposure_batch_size, total_exposures)
        yield slice(start, stop)


def _gradient_penalty(F, h_param):
    torch = _load_torch()
    kernel_row = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=h_param.device,
        dtype=h_param.dtype,
    ).reshape(1, 1, 3, 3)
    kernel_column = kernel_row.transpose(-1, -2)
    padded = F.pad(h_param, (1, 1, 1, 1), mode="reflect")
    row_gradient = F.conv2d(padded, kernel_row)
    column_gradient = F.conv2d(padded, kernel_column)
    return row_gradient.square().sum() + column_gradient.square().sum()


def _upsample_tensor(F, tensor, srf: int):
    if srf == 1:
        return tensor
    return F.interpolate(tensor, scale_factor=srf, mode="nearest")


def _conv_bank(F, f, x, srf: int, padding: str):
    conv_padding = "same" if padding == "SAME" else 0
    y = F.conv2d(x, f, padding=conv_padding)
    if srf > 1:
        y = _avg_pool2d_same(F, y, kernel_size=srf, stride=srf)
    return y


def _conv_transpose_bank(F, f, y, padding: str):
    if padding == "SAME":
        pad_h = (f.shape[-2] - 1) // 2
        pad_w = (f.shape[-1] - 1) // 2
    elif padding == "VALID":
        pad_h = 0
        pad_w = 0
    else:
        raise ValueError("padding must be 'SAME' or 'VALID'")
    return F.conv_transpose2d(y, f, padding=(pad_h, pad_w))


def _accumulate_transpose_sum(
    F,
    f,
    values,
    *,
    srf: int,
    padding: str,
    exposure_batch_size: int,
):
    total = None
    for exposure_slice in _iter_exposure_slices(
        f.shape[0], exposure_batch_size
    ):
        partial = _conv_transpose_bank(
            F,
            f[exposure_slice],
            _upsample_tensor(F, values[:, exposure_slice], srf),
            padding=padding,
        )
        total = partial if total is None else total + partial
    if total is None:
        raise RuntimeError(
            "No exposures available for transpose accumulation."
        )
    return total


def _accumulate_l2_denom(
    F,
    f,
    w,
    x,
    *,
    srf: int,
    padding: str,
    exposure_batch_size: int,
):
    total = None
    for exposure_slice in _iter_exposure_slices(
        f.shape[0], exposure_batch_size
    ):
        current = _conv_bank(
            F,
            f[exposure_slice],
            x,
            srf=srf,
            padding=padding,
        )
        partial = _conv_transpose_bank(
            F,
            f[exposure_slice],
            _upsample_tensor(F, w[:, exposure_slice] * current, srf),
            padding=padding,
        )
        total = partial if total is None else total + partial
    if total is None:
        raise RuntimeError("No exposures available for l2 accumulation.")
    return total


def _accumulate_l2_terms(
    F,
    f,
    y,
    w,
    x,
    *,
    background_model: str,
    srf: int,
    padding: str,
    exposure_batch_size: int,
):
    total_num = None
    total_denom = None
    background_chunks = []
    for exposure_slice in _iter_exposure_slices(
        f.shape[0], exposure_batch_size
    ):
        current = _conv_bank(
            F,
            f[exposure_slice],
            x,
            srf=srf,
            padding=padding,
        )
        y_slice = y[:, exposure_slice]
        w_slice = w[:, exposure_slice]
        residual = y_slice - current
        background = _estimate_scalar_background(
            residual, w_slice, background_model=background_model
        )
        adjusted_y = y_slice - background
        background_chunks.append(background)

        partial_num = _conv_transpose_bank(
            F,
            f[exposure_slice],
            _upsample_tensor(F, w_slice * adjusted_y, srf),
            padding=padding,
        )
        partial_denom = _conv_transpose_bank(
            F,
            f[exposure_slice],
            _upsample_tensor(F, w_slice * current, srf),
            padding=padding,
        )
        total_num = (
            partial_num if total_num is None else total_num + partial_num
        )
        total_denom = (
            partial_denom
            if total_denom is None
            else total_denom + partial_denom
        )
    if total_num is None or total_denom is None:
        raise RuntimeError("No exposures available for l2 accumulation.")
    return total_num, total_denom, _cat_backgrounds(background_chunks)


def _accumulate_rho_terms(
    F,
    f,
    y,
    w,
    v,
    x,
    *,
    background_model: str,
    rho_loss: str,
    delta_value: float,
    srf: int,
    padding: str,
    exposure_batch_size: int,
):
    torch = _load_torch()
    total_num = None
    total_denom = None
    background_chunks = []
    for exposure_slice in _iter_exposure_slices(
        f.shape[0], exposure_batch_size
    ):
        current = _conv_bank(
            F,
            f[exposure_slice],
            x,
            srf=srf,
            padding=padding,
        )
        y_slice = y[:, exposure_slice]
        w_slice = w[:, exposure_slice]
        v_slice = v[:, exposure_slice]
        sqrt_v = torch.sqrt(v_slice)
        raw_residual = y_slice - current
        delta_tensor = torch.as_tensor(
            delta_value,
            device=raw_residual.device,
            dtype=raw_residual.dtype,
        )
        background = _estimate_rho_background(
            raw_residual,
            w_slice,
            v_slice,
            delta_tensor=delta_tensor,
            rho_loss=rho_loss,
            background_model=background_model,
        )
        adjusted_y = y_slice if background is None else y_slice - background
        residual = (adjusted_y - current) / sqrt_v
        psi_x = irls_weights(
            torch,
            residual,
            loss=rho_loss,
            delta=delta_tensor,
            variance=v_slice,
        )
        w_x = w_slice * psi_x
        if background is not None:
            background_chunks.append(background)

        partial_num = _conv_transpose_bank(
            F,
            f[exposure_slice],
            _upsample_tensor(F, w_x * adjusted_y, srf),
            padding=padding,
        )
        partial_denom = _conv_transpose_bank(
            F,
            f[exposure_slice],
            _upsample_tensor(F, w_x * current, srf),
            padding=padding,
        )
        total_num = (
            partial_num if total_num is None else total_num + partial_num
        )
        total_denom = (
            partial_denom
            if total_denom is None
            else total_denom + partial_denom
        )

    if total_num is None or total_denom is None:
        raise RuntimeError("No exposures available for rho accumulation.")
    background = (
        _cat_backgrounds(background_chunks) if background_chunks else None
    )
    return total_num, total_denom, background


def _accumulate_loss(
    F,
    f,
    y,
    w,
    m,
    v,
    x,
    *,
    algorithm: str,
    rho_loss: str,
    background_model: str,
    delta_value: float,
    srf: int,
    padding: str,
    exposure_batch_size: int,
    m_normalization: float,
):
    torch = _load_torch()
    total = None
    for exposure_slice in _iter_exposure_slices(
        f.shape[0], exposure_batch_size
    ):
        current = _conv_bank(
            F,
            f[exposure_slice],
            x,
            srf=srf,
            padding=padding,
        )
        y_slice = y[:, exposure_slice]
        w_slice = w[:, exposure_slice]
        v_slice = v[:, exposure_slice]
        raw_residual = y_slice - current
        if algorithm == "l2":
            background = _estimate_scalar_background(
                raw_residual,
                w_slice,
                background_model=background_model,
            )
            residual = raw_residual - background
            partial = torch.sum(w_slice * residual.square())
        else:
            sqrt_v = torch.sqrt(v_slice)
            delta_tensor = torch.as_tensor(
                delta_value,
                device=raw_residual.device,
                dtype=raw_residual.dtype,
            )
            background = _estimate_rho_background(
                raw_residual,
                w_slice,
                v_slice,
                delta_tensor=delta_tensor,
                rho_loss=rho_loss,
                background_model=background_model,
            )
            residual = (
                raw_residual
                if background is None
                else raw_residual - background
            )
            normalized = residual / sqrt_v
            partial = torch.sum(
                m[:, exposure_slice]
                * rho_values(
                    torch,
                    normalized,
                    loss=rho_loss,
                    delta=delta_tensor,
                    variance=v_slice,
                )
            )
        total = partial if total is None else total + partial
    if total is None:
        raise RuntimeError("No exposures available for loss accumulation.")
    return total / m_normalization


def _estimate_scalar_background(residual, weights, *, background_model: str):
    torch = _load_torch()
    if background_model == "none":
        return torch.zeros(
            residual.shape[:2] + (1, 1),
            device=residual.device,
            dtype=residual.dtype,
        )
    if background_model != "scalar_per_exposure":
        raise ValueError(f"Unsupported background model: {background_model}")

    denom = torch.sum(weights, dim=(-2, -1), keepdim=True)
    numer = torch.sum(weights * residual, dim=(-2, -1), keepdim=True)
    return torch.where(
        denom > 1e-12,
        numer / denom.clamp_min(1e-12),
        torch.zeros_like(numer),
    )


def _estimate_rho_background(
    residual,
    weights,
    variances,
    *,
    delta_tensor,
    rho_loss: str,
    background_model: str,
):
    if background_model == "none":
        return None
    normalized = residual / _load_torch().sqrt(variances)
    torch = _load_torch()
    rho_weights = weights * irls_weights(
        torch,
        normalized,
        loss=rho_loss,
        delta=delta_tensor,
        variance=variances,
    )
    return _estimate_scalar_background(
        residual, rho_weights, background_model=background_model
    )


def _cat_backgrounds(backgrounds):
    torch = _load_torch()
    if not backgrounds:
        raise RuntimeError("No background estimates available.")
    return torch.cat(backgrounds, dim=1)


def _to_numpy_nhwc(value):
    arr = (
        value.detach().cpu().numpy()
        if hasattr(value, "detach")
        else np.asarray(value)
    )
    return np.transpose(arr, (0, 2, 3, 1))


def _to_numpy_background(value):
    arr = value.detach().cpu().numpy()
    return np.squeeze(arr, axis=(-1, -2))


def _forward_psf_bank(F, h_param, g_input, srf: int, padding: str):
    conv_padding = "same" if padding == "SAME" else 0
    pred = F.conv2d(g_input, h_param, padding=conv_padding)
    if srf > 1:
        pred = _avg_pool2d_same(F, pred, kernel_size=srf, stride=srf)
    return pred


def _avg_pool2d_same(F, tensor, kernel_size: int, stride: int):
    torch = _load_torch()
    height = tensor.shape[-2]
    width = tensor.shape[-1]
    out_h = (height + stride - 1) // stride
    out_w = (width + stride - 1) // stride
    pad_h = max((out_h - 1) * stride + kernel_size - height, 0)
    pad_w = max((out_w - 1) * stride + kernel_size - width, 0)
    mask = torch.ones_like(tensor)
    if pad_h or pad_w:
        pads = (
            pad_w // 2,
            pad_w - pad_w // 2,
            pad_h // 2,
            pad_h - pad_h // 2,
        )
        tensor = F.pad(tensor, pads)
        mask = F.pad(mask, pads)
    summed = F.avg_pool2d(
        tensor,
        kernel_size=kernel_size,
        stride=stride,
        divisor_override=1,
    )
    counts = F.avg_pool2d(
        mask,
        kernel_size=kernel_size,
        stride=stride,
        divisor_override=1,
    )
    return summed / counts.clamp_min(1)


def _load_psfs(
    bundle: Any,
    psf_path: Path | None,
    *,
    exposure_indices: list[int] | None = None,
    max_exposures: int | None = None,
    target_exposure_count: int | None = None,
) -> np.ndarray:
    if psf_path is None:
        psfs = np.asarray(bundle.psfs)
        if (
            target_exposure_count is not None
            and psfs.shape[-1] != target_exposure_count
        ):
            raise RuntimeError(
                "Bundle PSFs are not aligned with the selected "
                "exposure count."
            )
        return psfs

    psfs = np.load(psf_path, allow_pickle=False)
    if psfs.ndim != 4:
        raise ValueError(
            "External PSFs must have shape "
            "(height, width, channel, exposure); "
            f"received {psfs.shape}."
        )
    if exposure_indices is not None and max_exposures is not None:
        raise RuntimeError(
            "Specify either `exposure_indices` or "
            "`max_exposures` for PSF selection, not both."
        )
    saved_exposures = _load_saved_selected_exposures(psf_path)
    if saved_exposures is not None:
        if (
            len(saved_exposures) != psfs.shape[-1]
            or len(set(saved_exposures)) != len(saved_exposures)
            or any(
                index < 0 or isinstance(index, bool)
                for index in saved_exposures
            )
        ):
            raise RuntimeError(
                "PSF artifact exposure metadata is not aligned "
                "with its arrays."
            )
        requested = exposure_indices
        if requested is None:
            requested = bundle.metadata.get("selected_exposures")
        if requested is None:
            count = target_exposure_count
            if count is None:
                count = psfs.shape[-1]
                if max_exposures is not None:
                    count = min(count, max_exposures)
            requested = list(range(count))
        try:
            positions = [saved_exposures.index(index) for index in requested]
        except ValueError as exc:
            raise RuntimeError(
                "PSF artifact exposure identities do not match the selected "
                f"input exposures {requested}."
            ) from exc
        return psfs[:, :, :, positions]
    if exposure_indices is not None:
        if not exposure_indices:
            return psfs[:, :, :, :0]
        if (
            target_exposure_count is not None
            and psfs.shape[-1] == target_exposure_count
        ):
            identity = list(range(target_exposure_count))
            if exposure_indices == identity:
                return psfs
            raise RuntimeError(
                "External PSF artifact already matches the selected "
                "exposure count, but non-identity "
                "`exposure_indices` require package-generated "
                "summary metadata to prove alignment."
            )
        if max(exposure_indices) >= psfs.shape[-1]:
            raise RuntimeError(
                "PSF artifact exposure count "
                f"{psfs.shape[-1]} is incompatible with requested "
                f"indices {exposure_indices}."
            )
        return psfs[:, :, :, exposure_indices]
    if max_exposures is not None:
        psfs = psfs[:, :, :, :max_exposures]
    if (
        target_exposure_count is not None
        and psfs.shape[-1] != target_exposure_count
    ):
        raise RuntimeError(
            "External PSFs are not aligned with the selected exposure "
            f"count: expected {target_exposure_count}, got {psfs.shape[-1]}."
        )
    return psfs


def _load_saved_selected_exposures(psf_path: Path | None) -> list[int] | None:
    if psf_path is None:
        return None
    summary_path = psf_path.parent.parent / "summary.json"
    if not summary_path.exists():
        return None
    try:
        payload = json.loads(summary_path.read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            "Cannot verify PSF exposure alignment: failed to read "
            f"saved summary metadata at {summary_path}."
        ) from exc
    selected = payload.get("selected_exposures")
    if isinstance(selected, list) and all(
        isinstance(item, int) for item in selected
    ):
        return selected
    return None
