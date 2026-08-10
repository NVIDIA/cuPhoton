# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Opt-in xFit checks against caller-supplied observational data."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from astropy.io import fits
from scipy.ndimage import shift, zoom
from scipy.stats import rankdata

from cuphoton.xfit import (
    LMConfig,
    StampDipoleModel,
    fit_dipoles,
)
from cuphoton.xrep import BBox, ReprojectionSpec, reproject_array

pytestmark = pytest.mark.real_data

_ZTF_DIFFERENCE_SHA256 = (
    "6429aa5e1e2706ebc6a49689d16b220cbc1e3c0efd5babd118cca07d799b552a"
)
_ZTF_PSF_SHA256 = (
    "218b039c7f6a97ebb246aed3d453f3ded92cd5e5dd56b0919081a358d9f8493f"
)
_HARD_MASK_PLANES = (
    "BAD",
    "SAT",
    "INTRP",
    "CR",
    "EDGE",
    "SUSPECT",
    "NO_DATA",
    "CROSSTALK",
    "UNMASKEDNAN",
    "VIGNETTED",
    "STREAK",
)


def _environment_path(name: str) -> Path:
    value = os.environ.get(name)
    if value is None:
        pytest.skip(f"set {name} to run this observational-data check")
    path = Path(value).expanduser().resolve()
    if not path.exists():
        pytest.fail(f"{name} does not exist: {path}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@pytest.fixture(scope="module")
def _ztf_quiet_backgrounds() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the pinned ZTF PSF and 16 quiet real background stamps."""

    root = _environment_path("CUPHOTON_XFIT_ZTF_DIR")
    difference_path = root / "difference.fits.fz"
    psf_path = root / "difference-psf.fits"
    assert _sha256(difference_path) == _ZTF_DIFFERENCE_SHA256
    assert _sha256(psf_path) == _ZTF_PSF_SHA256

    with fits.open(difference_path, memmap=True) as hdus:
        difference = np.asarray(hdus[1].data, dtype=np.float64)
    with fits.open(psf_path, memmap=True) as hdus:
        psf = np.asarray(hdus[0].data, dtype=np.float64)
    psf /= psf.sum()

    size = 51
    half = size // 2
    quiet: list[tuple[float, int, int, np.ndarray]] = []
    for y_center in range(100, difference.shape[0] - 100, 100):
        for x_center in range(100, difference.shape[1] - 100, 100):
            image = difference[
                y_center - half : y_center + half + 1,
                x_center - half : x_center + half + 1,
            ]
            if image.shape != (size, size) or not np.isfinite(image).all():
                continue
            median = np.median(image)
            sigma = 1.4826 * np.median(np.abs(image - median))
            if sigma <= 0 or np.max(np.abs(image - median)) / sigma > 6:
                continue
            quiet.append((sigma, y_center, x_center, image - median))
    quiet.sort(key=lambda item: (item[0], item[1], item[2]))
    quiet = quiet[:16]
    assert len(quiet) == 16
    return (
        psf,
        np.stack([item[3] for item in quiet]),
        np.asarray([item[0] for item in quiet]),
    )


def _ztf_injections(
    quiet_backgrounds: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, ...]:
    psf, images, sigma = quiet_backgrounds
    images = images.copy()
    size = images.shape[-1]
    canvas = np.zeros((size, size), dtype=np.float64)
    psf_start = (size - psf.shape[0]) // 2
    canvas[
        psf_start : psf_start + psf.shape[0],
        psf_start : psf_start + psf.shape[1],
    ] = psf

    peak_snr = np.resize(np.asarray([10.0, 15.0, 20.0, 30.0]), len(images))
    truth = np.empty((len(images), 5), dtype=np.float64)
    for row in range(len(images)):
        x_pos, y_pos, x_neg, y_neg = (-2.3, -1.1, 2.0, 1.2)
        flux = peak_snr[row] * sigma[row] / psf.max()
        positive = shift(
            canvas,
            (y_pos, x_pos),
            order=3,
            mode="constant",
            cval=0,
            prefilter=True,
        )
        negative = shift(
            canvas,
            (y_neg, x_neg),
            order=3,
            mode="constant",
            cval=0,
            prefilter=True,
        )
        images[row] += flux * (positive - negative)
        truth[row] = (x_pos, y_pos, x_neg, y_neg, flux)
    return images, psf, sigma, truth


def _assert_ztf_recovery(result, truth: np.ndarray) -> dict[str, float | str]:
    error = result.parameters - truth
    centroid_error = np.maximum(
        np.hypot(error[:, 0], error[:, 1]),
        np.hypot(error[:, 2], error[:, 3]),
    )
    relative_flux_error = np.abs(error[:, 4] / truth[:, 4])

    assert result.converged.all()
    assert result.uncertainty_valid.all()
    assert np.max(centroid_error) < 0.2
    assert np.quantile(relative_flux_error, 0.9) < 0.1
    assert 0.8 < np.median(result.reduced_chi_square) < 1.2
    return {
        "maximum_centroid_error_pixels": float(np.max(centroid_error)),
        "median_centroid_error_pixels": float(np.median(centroid_error)),
        "maximum_relative_flux_error": float(np.max(relative_flux_error)),
        "median_reduced_chi_square": float(
            np.median(result.reduced_chi_square)
        ),
    }


def test_ztf_real_background_and_psf_injection_recovery(
    _ztf_quiet_backgrounds: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    images, psf, sigma, truth = _ztf_injections(_ztf_quiet_backgrounds)
    model = StampDipoleModel(
        psf,
        image_shape=images.shape[-2:],
        evaluation="bilinear",
        dtype=np.float64,
    )
    config = LMConfig(
        max_evaluations=500,
        finite_difference_step=1.0e-4,
    )
    cpu = fit_dipoles(
        images,
        model=model,
        variance=sigma[:, None, None] ** 2,
        backend="numpy",
        config=config,
    )
    metrics = _assert_ztf_recovery(cpu, truth)
    metrics["cpu_device"] = cpu.device

    if os.environ.get("CUPHOTON_XFIT_REAL_GPU") == "1":
        gpu = fit_dipoles(
            images,
            model=model,
            variance=sigma[:, None, None] ** 2,
            backend="cupy",
            config=config,
        )
        _assert_ztf_recovery(gpu, truth)
        assert gpu.backend == "cupy"
        assert np.allclose(
            gpu.parameters, cpu.parameters, rtol=1e-7, atol=1e-7
        )
        assert np.allclose(
            gpu.chi_square,
            cpu.chi_square,
            rtol=1e-10,
            atol=1e-10,
        )
        metrics["gpu_maximum_parameter_delta"] = float(
            np.max(np.abs(gpu.parameters - cpu.parameters))
        )
        metrics["gpu_device"] = gpu.device
    print(json.dumps(metrics, sort_keys=True))


_ASTROMETRIC_OVERSAMPLE = 4
_ASTROMETRIC_STARTS = (0.6, 1.8)
_ASTROMETRIC_CONFIG = LMConfig(
    max_evaluations=3000,
    finite_difference_step=1.0e-2,
)


def _xrep_shifted(
    canvas: np.ndarray, delta: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    offset = np.asarray(delta, dtype=np.float64)

    def mapping(coordinates: np.ndarray) -> np.ndarray:
        return coordinates - offset

    spec = ReprojectionSpec(
        mapping=mapping,
        output_bbox=BBox(0, 0, canvas.shape[1], canvas.shape[0]),
        interpolation="lanczos3",
        mapping_grid_step=10,
        area_scaling=False,
    )
    resampled = reproject_array(canvas, spec, backend="cpu").image
    valid = np.isfinite(resampled)
    return np.where(valid, resampled, 0.0), valid


def _astrometric_valley_initial(
    image: np.ndarray, size: int, separation: float
) -> tuple[float, float, float, float, float]:
    half = size // 2
    y_grid, x_grid = np.mgrid[-half : half + 1, -half : half + 1]
    windowed = np.where(x_grid * x_grid + y_grid * y_grid <= 100, image, 0.0)
    moment_x = float(np.sum(x_grid * windowed))
    moment_y = float(np.sum(y_grid * windowed))
    moment = float(np.hypot(moment_x, moment_y))
    if moment <= 0.0:
        return (-separation / 2.0, 0.0, separation / 2.0, 0.0, 1.0)
    unit_x = -moment_x / moment
    unit_y = -moment_y / moment
    return (
        -unit_x * separation / 2.0,
        -unit_y * separation / 2.0,
        unit_x * separation / 2.0,
        unit_y * separation / 2.0,
        moment / separation,
    )


def _astrometric_fit(
    model: StampDipoleModel,
    images: np.ndarray,
    masks: np.ndarray,
    variance: np.ndarray,
    separation: float,
    backend: str,
):
    size = images.shape[-1]
    initials = np.asarray(
        [
            _astrometric_valley_initial(image * mask, size, separation)
            for image, mask in zip(images, masks, strict=True)
        ],
        dtype=np.float64,
    )
    return fit_dipoles(
        images,
        model=model,
        mask=masks,
        variance=variance,
        initial=initials,
        backend=backend,
        config=_ASTROMETRIC_CONFIG,
    )


def _astrometric_best(
    model: StampDipoleModel,
    images: np.ndarray,
    masks: np.ndarray,
    variance: np.ndarray,
    backend: str,
) -> tuple[np.ndarray, np.ndarray]:
    parameters: np.ndarray | None = None
    converged: np.ndarray | None = None
    best_chi_square: np.ndarray | None = None
    for separation in _ASTROMETRIC_STARTS:
        result = _astrometric_fit(
            model, images, masks, variance, separation, backend
        )
        chi_square = np.where(result.converged, result.chi_square, np.inf)
        if parameters is None:
            parameters = result.parameters.copy()
            converged = result.converged.copy()
            best_chi_square = chi_square
            continue
        better = chi_square < best_chi_square
        parameters[better] = result.parameters[better]
        converged[better] = result.converged[better]
        best_chi_square = np.minimum(best_chi_square, chi_square)
    return parameters, converged


def test_ztf_astrometric_offset_recovery_through_xrep(
    _ztf_quiet_backgrounds: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Recover known science/template offsets on real ZTF backgrounds.

    A sub-pixel astrometric offset between science and template creates the
    dipole ``flux * (PSF(x) - PSF(x - delta))``. The template lobe is built
    with the real xRep Lanczos-3 resampling path, and the fitted
    lobe-separation vector estimates ``delta``. In the unresolved regime the
    flux-separation degeneracy leaves only the dipole moment
    ``flux * delta`` and the offset direction well constrained, so the
    assertions follow the regime.
    """

    psf, backgrounds, sigma = _ztf_quiet_backgrounds
    size = backgrounds.shape[-1]
    canvas = np.zeros((size, size), dtype=np.float64)
    psf_start = (size - psf.shape[0]) // 2
    canvas[
        psf_start : psf_start + psf.shape[0],
        psf_start : psf_start + psf.shape[1],
    ] = psf
    psf_fine = zoom(
        psf,
        _ASTROMETRIC_OVERSAMPLE,
        order=3,
        prefilter=True,
        grid_mode=True,
        mode="grid-constant",
    )
    model = StampDipoleModel(
        psf_fine,
        image_shape=(size, size),
        scale=1.0 / _ASTROMETRIC_OVERSAMPLE,
        dtype=np.float64,
    )

    peak_snr = 100.0
    magnitudes = (0.25, 1.0)
    angles = (0.0, 45.0, 90.0, 135.0)
    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    variance: list[np.ndarray] = []
    truths: list[tuple[float, float, float, float]] = []
    for magnitude in magnitudes:
        for angle in angles:
            delta = (
                magnitude * float(np.cos(np.radians(angle))),
                magnitude * float(np.sin(np.radians(angle))),
            )
            template, valid = _xrep_shifted(canvas, delta)
            for row in range(len(backgrounds)):
                flux = peak_snr * sigma[row] / psf.max()
                images.append(backgrounds[row] + flux * (canvas - template))
                masks.append(valid)
                variance.append(np.full((size, size), float(sigma[row]) ** 2))
                truths.append((delta[0], delta[1], flux, magnitude))
    stacked = np.stack(images)
    stacked_masks = np.stack(masks)
    stacked_variance = np.stack(variance)
    truth = np.asarray(truths, dtype=np.float64)

    parameters, converged = _astrometric_best(
        model, stacked, stacked_masks, stacked_variance, "numpy"
    )
    assert float(converged.mean()) >= 0.85

    recovered = np.stack(
        [
            parameters[:, 2] - parameters[:, 0],
            parameters[:, 3] - parameters[:, 1],
        ],
        axis=1,
    )
    offset_error = np.hypot(*(recovered - truth[:, :2]).T)
    moment_error = np.hypot(
        *((parameters[:, 4:5] * recovered) - truth[:, 2:3] * truth[:, :2]).T
    ) / np.hypot(*(truth[:, 2:3] * truth[:, :2]).T)
    alignment = np.sum(recovered * truth[:, :2], axis=1) / (
        np.hypot(*recovered.T) * np.hypot(*truth[:, :2].T)
    )
    direction_error = np.degrees(np.arccos(np.clip(alignment, -1.0, 1.0)))

    resolved = (truth[:, 3] == 1.0) & converged
    unresolved = (truth[:, 3] == 0.25) & converged
    assert float(np.median(offset_error[resolved])) < 0.25
    assert float(np.median(direction_error[resolved])) < 1.5
    assert float(np.median(moment_error[resolved])) < 0.05
    assert float(np.median(moment_error[unresolved])) < 0.15
    assert float(np.median(direction_error[unresolved])) < 5.0

    metrics = {
        "converged_fraction": float(converged.mean()),
        "resolved_offset_error_median_pixels": float(
            np.median(offset_error[resolved])
        ),
        "resolved_direction_error_median_degrees": float(
            np.median(direction_error[resolved])
        ),
        "resolved_moment_error_median": float(
            np.median(moment_error[resolved])
        ),
        "unresolved_moment_error_median": float(
            np.median(moment_error[unresolved])
        ),
        "unresolved_direction_error_median_degrees": float(
            np.median(direction_error[unresolved])
        ),
    }

    if os.environ.get("CUPHOTON_XFIT_REAL_GPU") == "1":
        for separation in _ASTROMETRIC_STARTS:
            cpu = _astrometric_fit(
                model,
                stacked,
                stacked_masks,
                stacked_variance,
                separation,
                "numpy",
            )
            gpu = _astrometric_fit(
                model,
                stacked,
                stacked_masks,
                stacked_variance,
                separation,
                "cupy",
            )
            assert gpu.backend == "cupy"
            assert float((cpu.status == gpu.status).mean()) >= 0.9
            both = cpu.converged & gpu.converged
            assert float(both.mean()) >= 0.85
            assert np.allclose(
                cpu.parameters[both],
                gpu.parameters[both],
                rtol=1.0e-7,
                atol=1.0e-7,
            )
        metrics["gpu_device"] = gpu.device
    print(json.dumps(metrics, sort_keys=True))


def _load_rubin_candidates(
    metadata_path: Path,
) -> tuple[np.ndarray, ...]:
    metadata = pq.read_table(metadata_path).to_pylist()
    assert metadata
    stamp_sizes = {int(row["stamp_size"]) for row in metadata}
    assert len(stamp_sizes) == 1
    size = stamp_sizes.pop()
    half = size // 2
    images = np.empty((len(metadata), size, size), dtype=np.float64)
    variance = np.empty_like(images)
    valid = np.empty_like(images, dtype=bool)

    rows_by_path: dict[str, list[tuple[int, dict[str, object]]]] = {}
    for index, row in enumerate(metadata):
        rows_by_path.setdefault(str(row["difference_image_path"]), []).append(
            (index, row)
        )
    for path, rows in rows_by_path.items():
        with fits.open(path, memmap=True, lazy_load_hdus=True) as hdus:
            mask_header = hdus["MASK"].header
            hard_mask = sum(
                1 << int(mask_header[f"MP_{name}"])
                for name in _HARD_MASK_PLANES
                if f"MP_{name}" in mask_header
            )
            for index, row in rows:
                y_center = int(row["y"])
                x_center = int(row["x"])
                section = np.s_[
                    y_center - half : y_center + half + 1,
                    x_center - half : x_center + half + 1,
                ]
                images[index] = hdus["IMAGE"].section[section]
                variance[index] = hdus["VARIANCE"].section[section]
                mask = np.asarray(hdus["MASK"].section[section])
                valid[index] = (
                    np.isfinite(images[index])
                    & np.isfinite(variance[index])
                    & (variance[index] > 0)
                    & ((mask & hard_mask) == 0)
                )

    y_grid, x_grid = np.mgrid[-half : half + 1, -half : half + 1]
    radius_squared = x_grid * x_grid + y_grid * y_grid
    fit_region = radius_squared <= 12**2
    background_annulus = (radius_squared >= 10**2) & fit_region
    valid &= fit_region
    retained = np.sum(valid, axis=(1, 2)) >= 100
    metadata = [
        row for row, keep in zip(metadata, retained, strict=True) if keep
    ]
    images = images[retained]
    variance = variance[retained]
    valid = valid[retained]
    initial = np.empty((len(metadata), 8), dtype=np.float64)
    for index in range(len(metadata)):
        background_pixels = valid[index] & background_annulus
        if not background_pixels.any():
            background_pixels = valid[index] & fit_region
        images[index] -= np.median(images[index][background_pixels])
        signal_to_noise = np.full((size, size), np.nan, dtype=np.float64)
        np.divide(
            images[index],
            np.sqrt(np.where(valid[index], variance[index], 1.0)),
            out=signal_to_noise,
            where=valid[index],
        )
        positive = np.nanargmax(signal_to_noise)
        negative = np.nanargmin(signal_to_noise)
        y_pos, x_pos = np.unravel_index(positive, (size, size))
        y_neg, x_neg = np.unravel_index(negative, (size, size))
        amplitude = max(
            images[index, y_pos, x_pos],
            -images[index, y_neg, x_neg],
        )
        initial[index] = (
            amplitude,
            2.0,
            2.0,
            0.0,
            x_pos - half,
            y_pos - half,
            x_neg - half,
            y_neg - half,
        )
    labels = np.asarray([bool(row["candidate_isDipole"]) for row in metadata])
    return images, variance, valid, initial, labels


def test_rubin_real_candidate_pipeline_dipole_smoke() -> None:
    metadata_path = _environment_path("CUPHOTON_XFIT_RUBIN_METADATA")
    images, variance, valid, initial, labels = _load_rubin_candidates(
        metadata_path
    )
    assert labels.sum() >= 8
    assert (~labels).sum() >= 8

    result = fit_dipoles(
        images,
        model="gaussian",
        initial=initial,
        mask=valid,
        variance=variance,
        backend="numpy",
        config=LMConfig(max_evaluations=500),
    )
    parameters = result.parameters
    plausible = (
        result.converged
        & np.isfinite(parameters).all(axis=1)
        & (parameters[:, 0] > 0)
        & (parameters[:, 1:3] >= 0.3).all(axis=1)
        & (parameters[:, 1:3] <= 8.0).all(axis=1)
        & (np.abs(parameters[:, 4:8]) <= 12.0).all(axis=1)
    )
    weighted_images = np.zeros_like(images)
    np.divide(
        images,
        np.sqrt(np.where(valid, variance, 1.0)),
        out=weighted_images,
        where=valid,
    )
    null_norm = np.sqrt(
        np.sum(weighted_images * weighted_images, axis=(1, 2))
    )
    improvement = 1.0 - result.residual_norm / null_norm
    score = np.where(plausible, improvement, -1.0)
    ranks = rankdata(score)
    positive_count = int(labels.sum())
    negative_count = int((~labels).sum())
    auc = (
        ranks[labels].sum() - positive_count * (positive_count + 1) / 2
    ) / (positive_count * negative_count)

    assert plausible[labels].mean() >= 0.8
    assert np.median(improvement[labels & plausible]) > np.median(
        improvement[(~labels) & plausible]
    )
    assert auc >= 0.8
    print(
        json.dumps(
            {
                "candidate_count": len(labels),
                "pipeline_is_dipole_count": positive_count,
                "plausible_fit_count": int(plausible.sum()),
                "pipeline_is_dipole_plausible_rate": float(
                    plausible[labels].mean()
                ),
                "fit_improvement_auc": float(auc),
            },
            sort_keys=True,
        )
    )
