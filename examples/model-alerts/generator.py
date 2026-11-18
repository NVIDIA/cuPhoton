# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent electron-count exposures and measured xPois candidates.

No catalog, image, calibration file, checkpoint or network input is accepted.
Truth is returned separately and used only after detection for association.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.special import erf


@dataclass(frozen=True)
class Design:
    size: int = 128
    stamp_size: int = 31
    epochs: int = 3
    stars: int = 22
    changes: int = 5
    detection_threshold: float = 4.0
    association_radius: float = 3.0
    read_noise: float = 4.0
    calibration_size: int = 64
    star_flux_min: float = 400.0
    star_flux_max: float = 18000.0
    hot_flux_min: float = 300.0
    hot_flux_max: float = 1500.0
    cosmic_flux_min: float = 200.0
    cosmic_flux_max: float = 900.0
    moving_flux: float = 1800.0
    global_offset_sigma: float = 0.2
    local_offset_sigma: float = 0.35
    mismatch_factor: float = 1.25
    column_flux_min: float = 100.0
    column_flux_max: float = 400.0
    column_half_length: int = 5
    root_seed: int = 20261115

    def __post_init__(self):
        if self.calibration_size < 2:
            raise ValueError(
                "calibration frames must contain at least 2x2 pixels"
            )
        if self.stamp_size < 3 or self.stamp_size % 2 != 1:
            raise ValueError("stamp size must be odd and at least 3")
        if self.size <= self.stamp_size + 20:
            raise ValueError(
                "field must leave a complete stamp/kernel margin"
            )
        if min(self.epochs, self.stars, self.changes) < 1:
            raise ValueError(
                "positive epoch, star and change counts required"
            )
        if self.changes > self.stars:
            raise ValueError("brightening positions require enough stars")
        if (
            min(
                self.detection_threshold,
                self.association_radius,
                self.read_noise,
            )
            <= 0
        ):
            raise ValueError(
                "positive threshold, radius and read noise required"
            )


DEFAULT_DESIGN = Design()


def rng_for(design, field, stream):
    """Disjoint deterministic streams; field identity owns every epoch."""
    return np.random.default_rng(
        np.random.SeedSequence([design.root_seed, int(field), int(stream)])
    )


def psf(shape, x, y, sigma):
    """Pixel-integrated circular Gaussian with unit infinite-plane flux."""
    if not np.isfinite([x, y, sigma]).all() or sigma <= 0:
        raise ValueError("finite position and positive sigma required")
    yy, xx = np.indices(shape)
    scale = np.sqrt(2) * sigma
    px = 0.5 * (erf((xx + 0.5 - x) / scale) - erf((xx - 0.5 - x) / scale))
    py = 0.5 * (erf((yy + 0.5 - y) / scale) - erf((yy - 0.5 - y) / scale))
    return px * py


def expose(expected, rng, read_noise):
    """Poisson electrons plus independent Gaussian readout noise, gain=1."""
    if not np.isfinite(expected).all() or np.any(expected < 0):
        raise ValueError("expected electrons must be finite and nonnegative")
    return rng.poisson(expected).astype(float) + rng.normal(
        0, read_noise, expected.shape
    )


def calibration_sample(background, read_noise, rng, *, size=64):
    """Estimate sky/read variance from independent calibration frames."""
    shape = (size, size)
    sky = expose(np.full(shape, background), rng, read_noise)
    dark_a = rng.normal(0, read_noise, shape)
    dark_b = rng.normal(0, read_noise, shape)
    return dict(
        background=float(sky.mean()),
        read_variance=float(np.var(dark_a - dark_b, ddof=1) / 2),
        pixels=size * size,
    )


def input_variances(exposure):
    """Measured source intensity plus independently estimated calibration."""
    values = []
    for name, prefix in (("search", "search"), ("template", "reference")):
        background = exposure[prefix + "_background"]
        values.append(
            np.maximum(exposure[name] - background, 0)
            + background
            + exposure[prefix + "_read_variance"]
        )
    return tuple(values)


def field_exposures(field, *, design=DEFAULT_DESIGN, shifted=False):
    """One shared template and independent search noise per epoch.

    Teaching parameters: uniform backgrounds 60--140 electrons/pixel;
    reference sigma 0.9--1.2 pixels; search sigma 1.2--1.9 pixels. Shift uses
    background 180--300 and search sigma 1.9--2.6, plus unseen column bleed.
    Expected change SNR is log-uniform 4--35 under a background-only Gaussian
    matched-filter approximation. It is an injection parameter, never a
    measured detection significance or a model feature.
    """
    rng = rng_for(design, field, 0)
    shape = (design.size, design.size)
    margin = design.stamp_size // 2 + 9
    positions = rng.uniform(margin, design.size - margin, (design.stars, 2))
    fluxes = np.exp(
        rng.uniform(
            np.log(design.star_flux_min),
            np.log(design.star_flux_max),
            design.stars,
        )
    )
    reference_sigma = rng.uniform(0.9, 1.2)
    background = rng.uniform(60, 140)
    reference_calibration = calibration_sample(
        background,
        design.read_noise,
        rng_for(design, field, 1000),
        size=design.calibration_size,
    )
    reference_mean = np.full(shape, background)
    for (x, y), flux in zip(positions, fluxes, strict=False):
        reference_mean += flux * psf(shape, x, y, reference_sigma)
    # Detector dark-current defect is present in both reference and search.
    hot_x, hot_y = rng.integers(margin, design.size - margin, 2)
    hot_flux = rng.uniform(design.hot_flux_min, design.hot_flux_max)
    reference_mean[hot_y, hot_x] += hot_flux
    template = expose(
        reference_mean, rng_for(design, field, 1), design.read_noise
    )
    for epoch in range(design.epochs):
        erng = rng_for(design, field, 10 + epoch)
        sigma = erng.uniform(1.9, 2.6) if shifted else erng.uniform(1.2, 1.9)
        sky = erng.uniform(180, 300) if shifted else erng.uniform(60, 140)
        expected = np.full(shape, sky)
        causes = []
        # A single constant kernel can absorb a common translation, but not
        # independent local astrometric errors for different static sources.
        offsets = erng.normal(0, design.global_offset_sigma, 2) + erng.normal(
            0, design.local_offset_sigma, (design.stars, 2)
        )
        mismatch = np.arange(design.stars) % 7 == 0
        host_pool = np.flatnonzero(~mismatch)
        host_count = (design.changes + 1) // 2
        if host_count > len(host_pool):
            raise ValueError(
                "not enough non-mismatch stars for brightening hosts"
            )
        hosts = erng.choice(host_pool, host_count, replace=False)
        for index, ((x, y), flux) in enumerate(
            zip(positions, fluxes, strict=False)
        ):
            dx, dy = offsets[index]
            local_sigma = sigma * (
                design.mismatch_factor if mismatch[index] else 1
            )
            expected += flux * psf(shape, x + dx, y + dy, local_sigma)
            causes.append(
                dict(
                    x=float(x + dx),
                    y=float(y + dy),
                    family="psf_mismatch"
                    if index % 7 == 0
                    else "misregistration",
                    positive=False,
                    challenge=False,
                )
            )
        changes = erng.uniform(
            margin, design.size - margin, (design.changes, 2)
        )
        for index, (x, y) in enumerate(changes):
            snr = np.exp(erng.uniform(np.log(4), np.log(35)))
            flux = snr * np.sqrt(
                4
                * np.pi
                * sigma**2
                * (sky + background + 2 * design.read_noise**2)
            )
            # Half brighten an existing source; half appear on blank sky.
            if index % 2 == 0:
                host = hosts[index // 2]
                x, y = positions[host] + offsets[host]
            expected += flux * psf(shape, x, y, sigma)
            causes.append(
                dict(
                    x=float(x),
                    y=float(y),
                    family="brightening" if index % 2 == 0 else "appearance",
                    positive=True,
                    challenge=False,
                    expected_snr=float(snr),
                    injection_id=f"{field}:{epoch}:change-{index}",
                )
            )
        # One untrailed mover is a held-out astrophysical challenge per epoch.
        x, y = erng.uniform(margin, design.size - margin, 2)
        expected += design.moving_flux * psf(shape, x, y, sigma)
        causes.append(
            dict(
                x=float(x),
                y=float(y),
                family="moving_source",
                positive=True,
                challenge=True,
            )
        )
        expected[hot_y, hot_x] += hot_flux
        search = expose(expected, erng, design.read_noise)
        causes.append(
            dict(
                x=float(hot_x),
                y=float(hot_y),
                family="bad_pixel",
                positive=False,
                challenge=False,
            )
        )
        x, y = erng.integers(margin, design.size - margin, 2)
        search[y, x : x + 3] += erng.uniform(
            design.cosmic_flux_min, design.cosmic_flux_max, 3
        )
        causes.append(
            dict(
                x=float(x + 1),
                y=float(y),
                family="cosmic_ray",
                positive=False,
                challenge=False,
            )
        )
        if shifted:
            x, y = erng.integers(margin, design.size - margin, 2)
            extent = design.column_half_length
            search[y - extent : y + extent + 1, x] += erng.uniform(
                design.column_flux_min, design.column_flux_max
            )
            causes.append(
                dict(
                    x=float(x),
                    y=float(y),
                    family="column_bleed",
                    segment_half_length=extent,
                    positive=False,
                    challenge=False,
                )
            )
        search_calibration = calibration_sample(
            sky,
            design.read_noise,
            rng_for(design, field, 1100 + epoch),
            size=design.calibration_size,
        )
        yield dict(
            template=template.copy(),
            search=search,
            reference_background=reference_calibration["background"],
            reference_read_variance=reference_calibration["read_variance"],
            search_background=search_calibration["background"],
            search_read_variance=search_calibration["read_variance"],
            field=int(field),
            epoch=epoch,
            causes=causes,
        )


def detect(exposure, *, design=DEFAULT_DESIGN, backend="cpu"):
    """Fit xPois without truth masks, then find positive filtered maxima.

    Variances use measured images, assumed unit gain and independently
    estimated read-noise/sky calibration. Kernel uncertainty and spatial
    covariance are omitted;
    resulting SNR is a diagnostic rather than a calibrated significance.
    """
    from cuphoton.xpois import (
        GaussianBasisComponent,
        solve_constant_kernel,
        standardize_constant_kernel_residual,
    )

    search, template = exposure["search"], exposure["template"]
    variance, ref_variance = input_variances(exposure)
    fit = solve_constant_kernel(
        template,
        search,
        [
            GaussianBasisComponent(sigma=0.7, degree=2),
            GaussianBasisComponent(sigma=1.5, degree=2),
            GaussianBasisComponent(sigma=2.5, degree=0),
        ],
        variance=variance,
        kernel_shape=(15, 15),
        backend=backend,
    )
    noise = standardize_constant_kernel_residual(
        fit.residual,
        kernel=fit.kernel,
        target_variance=variance,
        reference_variance=ref_variance,
    )
    # Fixed detection filter, independent of the injection seeing truth.
    axis = np.arange(-4, 5)
    filt = np.exp(-(axis[:, None] ** 2 + axis[None, :] ** 2) / (2 * 1.4**2))
    filt /= filt.sum()
    filtered = ndimage.convolve(fit.residual, filt, mode="constant")
    vfiltered = ndimage.convolve(
        np.nan_to_num(noise.marginal_residual_variance),
        filt**2,
        mode="constant",
    )
    snr = filtered / np.sqrt(np.maximum(vfiltered, 1e-12))
    valid = ndimage.binary_erosion(
        noise.valid_mask, iterations=design.stamp_size // 2
    )
    half = design.stamp_size // 2
    valid[:half] = valid[-half:] = False
    valid[:, :half] = valid[:, -half:] = False
    selected = (
        valid
        & (snr >= design.detection_threshold)
        & (snr == ndimage.maximum_filter(snr, size=7))
    )
    locations = np.column_stack(np.where(selected))
    stamps, metadata = [], []
    for y, x in locations:
        cut = np.s_[y - half : y + half + 1, x - half : x + half + 1]
        stamps.append(
            np.stack([search[cut], template[cut], fit.residual[cut]])
        )
        metadata.append(dict(x=int(x), y=int(y), snr=float(snr[y, x])))
    return stamps, metadata, fit, noise


def associate(candidate, causes, radius):
    """Associate truth after detection and flag coverage limits."""

    def distance(cause):
        dx = candidate["x"] - cause["x"]
        dy = max(
            abs(candidate["y"] - cause["y"])
            - cause.get("segment_half_length", 0),
            0,
        )
        return np.hypot(dx, dy)

    near = [c for c in causes if distance(c) <= radius]
    near.sort(key=distance)
    real = [c for c in near if c["positive"]]
    artifacts = [
        c
        for c in near
        if c["family"] in ("cosmic_ray", "bad_pixel", "column_bleed")
    ]
    ambiguous = len(real) > 1 or bool(real and artifacts)
    return dict(
        associated_injections=[
            c["injection_id"] for c in real if "injection_id" in c
        ],
        label=int(bool(real)),
        family=real[0]["family"]
        if real
        else (near[0]["family"] if near else "noise"),
        ambiguous=ambiguous,
        overlapping_causes=len(near) > 1,
        challenge=any(c["challenge"] for c in real),
    )


def write_dataset(
    output, fields, *, design=DEFAULT_DESIGN, shifted=False, backend="cpu"
):
    """Create native xScan arrays; refuse occupied paths before any writes.

    ``fields`` is an explicit mapping of field identifier to split name. The
    caller owns the development/freeze/test schedule; this function never
    chooses or extends seed blocks on the basis of classification outcomes.
    """
    from cuphoton.xscan.dataset import validate_dataset_dir
    from cuphoton.xscan.training import require_training_label_provenance

    output = Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("dataset output must be absent or empty")
    if not fields or not set(fields.values()) <= {"train", "val", "test"}:
        raise ValueError("explicit field-to-split mapping required")
    if design.stamp_size % 2 != 1 or design.stamp_size < 3:
        raise ValueError("stamp size must be odd and at least 3")
    if any(type(field) is not int or field < 0 for field in fields):
        raise ValueError("field identifiers must be nonnegative integers")
    images, rows, recovery, variances = [], [], [], []
    for field, split in fields.items():
        for exposure in field_exposures(
            field, design=design, shifted=shifted
        ):
            stamps, candidates, fit, noise = detect(
                exposure, design=design, backend=backend
            )
            positives = [
                c
                for c in exposure["causes"]
                if c["positive"] and not c["challenge"]
            ]
            recovered = sum(
                any(
                    np.hypot(c["x"] - t["x"], c["y"] - t["y"])
                    <= design.association_radius
                    for c in candidates
                )
                for t in positives
            )
            recovery.append(
                dict(
                    field=field,
                    epoch=exposure["epoch"],
                    injected=len(positives),
                    injection_families={
                        c["injection_id"]: c["family"] for c in positives
                    },
                    calibration=dict(
                        reference_background=exposure["reference_background"],
                        search_background=exposure["search_background"],
                        reference_read_variance=exposure[
                            "reference_read_variance"
                        ],
                        search_read_variance=exposure["search_read_variance"],
                        pixels=design.calibration_size**2,
                    ),
                    detected=recovered,
                )
            )
            for stamp, candidate in zip(stamps, candidates, strict=False):
                truth = associate(
                    candidate, exposure["causes"], design.association_radius
                )
                index = len(rows)
                rows.append(
                    dict(
                        candidate_id=index,
                        band="modeled_broadband",
                        ccd_id=f"detector-{field}",
                        exposure_id=f"field-{field}-epoch-{exposure['epoch']}",
                        sample_index=index,
                        field=field,
                        epoch=exposure["epoch"],
                        split=split,
                        split_group=f"field-{field}",
                        label_source="analytic_injection",
                        target_label_available=True,
                        **candidate,
                        **truth,
                    )
                )
                images.append(stamp)
                half = design.stamp_size // 2
                x, y = candidate["x"], candidate["y"]
                variances.append(
                    noise.marginal_residual_variance[
                        y - half : y + half + 1, x - half : x + half + 1
                    ]
                )
    if not images:
        raise ValueError("design produced no detected candidates")
    images = np.asarray(images, dtype=np.float32)
    digests = [hashlib.sha256(s.tobytes()).hexdigest() for s in images]
    if len(set(digests)) != len(digests):
        raise ValueError("duplicate candidate content detected")
    output.mkdir(parents=True, exist_ok=True)
    for index, name in enumerate(("search", "template", "difference")):
        np.save(output / f"{name}.npy", images[:, index], allow_pickle=False)
    np.save(output / "variance.npy", np.asarray(variances, dtype=np.float64))
    np.save(
        output / "labels.npy",
        np.array([r["label"] for r in rows], dtype=np.int64),
    )
    np.save(
        output / "split.npy",
        np.array(
            [("train", "val", "test").index(r["split"]) for r in rows],
            dtype=np.int64,
        ),
    )
    (output / "metadata.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    (output / "generation.json").write_text(
        json.dumps(
            dict(
                design=asdict(design),
                fields=fields,
                shifted=shifted,
                backend=backend,
                recovery=recovery,
                candidates=len(rows),
                unique_stamp_hashes=len(
                    {hashlib.sha256(s.tobytes()).hexdigest() for s in images}
                ),
            ),
            indent=2,
        )
        + "\n"
    )
    validate_dataset_dir(output, dataset_kind="prepared")
    require_training_label_provenance(output)
    return rows
