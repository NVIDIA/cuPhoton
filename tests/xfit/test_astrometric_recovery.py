# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Recover known astrometric offsets from xRep-resampled dipole templates.

A science/template astrometric offset ``delta`` turns one point source into
the exact dipole ``flux * (PSF(x - c) - PSF(x - c - delta))``, so the fitted
lobe-separation vector of a :class:`~cuphoton.xfit.StampDipoleModel` directly
estimates ``delta``. These checks build the shifted template with the real
``cuphoton.xrep`` resampling path and verify that the fit recovers the
injected offset, that Lanczos-3 resampling stays astrometrically transparent
against an analytically evaluated template, and that NumPy and CuPy agree.
"""

from __future__ import annotations

import numpy as np
import pytest

from cuphoton.xfit import LMConfig, StampDipoleModel, fit_dipoles
from cuphoton.xrep import BBox, ReprojectionSpec, reproject_array

SIZE = 41
HALF = SIZE // 2
OVERSAMPLE = 4
PSF_SIGMA = 1.8
PSF_RADIUS = 10
NOISE_SIGMA = 1.0
PEAK_SNR = 100.0
FIT_CONFIG = LMConfig(max_evaluations=3000, finite_difference_step=1.0e-2)
OFFSETS = (
    (0.6, 25.0),
    (1.4, 110.0),
)
SEEDS = (20260807, 20260808, 20260809, 20260810)
START_SEPARATIONS = (0.6, 1.8)

_Y_GRID, _X_GRID = np.mgrid[-HALF : HALF + 1, -HALF : HALF + 1]
_CORE = (_X_GRID * _X_GRID + _Y_GRID * _Y_GRID) <= 10**2


def _gaussian(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    normalization = 1.0 / (2.0 * np.pi * PSF_SIGMA * PSF_SIGMA)
    return normalization * np.exp(
        -0.5 * (x * x + y * y) / (PSF_SIGMA * PSF_SIGMA)
    )


def _fine_basis() -> np.ndarray:
    width = 2 * PSF_RADIUS * OVERSAMPLE + 1
    offsets = (np.arange(width) - (width - 1) / 2) / OVERSAMPLE
    return _gaussian(offsets[None, :], offsets[:, None])


def _science_canvas() -> np.ndarray:
    return _gaussian(_X_GRID.astype(np.float64), _Y_GRID.astype(np.float64))


def _xrep_template(
    delta: tuple[float, float], interpolation: str
) -> tuple[np.ndarray, np.ndarray]:
    shift = np.asarray(delta, dtype=np.float64)

    def mapping(coordinates: np.ndarray) -> np.ndarray:
        return coordinates - shift

    spec = ReprojectionSpec(
        mapping=mapping,
        output_bbox=BBox(0, 0, SIZE, SIZE),
        interpolation=interpolation,
        mapping_grid_step=10,
        area_scaling=False,
    )
    resampled = reproject_array(_science_canvas(), spec, backend="cpu").image
    valid = np.isfinite(resampled)
    return np.where(valid, resampled, 0.0), valid


def _valley_initial(
    image: np.ndarray, separation: float
) -> tuple[float, float, float, float, float]:
    """Start on the flux-separation degeneracy valley.

    The windowed first moment of the dipole image estimates the dipole
    moment ``flux * delta`` directly, so the initial point fixes the lobe
    separation at ``separation`` along the moment direction and assigns the
    matching flux.
    """

    windowed = np.where(_CORE, image, 0.0)
    moment_x = float(np.sum(_X_GRID * windowed))
    moment_y = float(np.sum(_Y_GRID * windowed))
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


def _fourier_template(delta: tuple[float, float]) -> np.ndarray:
    """Shift the sampled science canvas exactly in Fourier space."""

    canvas = _science_canvas()
    frequency_y = np.fft.fftfreq(SIZE)[:, None]
    frequency_x = np.fft.fftfreq(SIZE)[None, :]
    phase = np.exp(
        -2j * np.pi * (frequency_x * delta[0] + frequency_y * delta[1])
    )
    return np.real(np.fft.ifft2(np.fft.fft2(canvas) * phase))


def _dipole_batch(
    template_kind: str,
    *,
    offsets: tuple[tuple[float, float], ...] = OFFSETS,
    peak_snr: float = PEAK_SNR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    canvas = _science_canvas()
    peak_value = float(canvas.max())
    flux = peak_snr * NOISE_SIGMA / peak_value
    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    truths: list[tuple[float, float, float]] = []
    for magnitude, angle in offsets:
        delta = (
            magnitude * float(np.cos(np.radians(angle))),
            magnitude * float(np.sin(np.radians(angle))),
        )
        if template_kind == "fourier":
            template = _fourier_template(delta)
            valid = np.ones((SIZE, SIZE), dtype=bool)
        else:
            template, valid = _xrep_template(delta, template_kind)
        for seed in SEEDS:
            noise = np.random.default_rng(seed).normal(
                0.0, NOISE_SIGMA, (SIZE, SIZE)
            )
            images.append(flux * (canvas - template) + noise)
            masks.append(valid)
            truths.append((delta[0], delta[1], flux))
    return (
        np.stack(images),
        np.stack(masks),
        np.asarray(truths, dtype=np.float64),
    )


def _fit_start(
    images: np.ndarray,
    masks: np.ndarray,
    separation: float,
    backend: str,
):
    model = StampDipoleModel(
        _fine_basis(),
        image_shape=(SIZE, SIZE),
        scale=1.0 / OVERSAMPLE,
        dtype=np.float64,
    )
    initials = np.asarray(
        [
            _valley_initial(image * mask, separation)
            for image, mask in zip(images, masks, strict=True)
        ],
        dtype=np.float64,
    )
    return fit_dipoles(
        images,
        model=model,
        mask=masks,
        variance=np.full_like(images, NOISE_SIGMA * NOISE_SIGMA),
        initial=initials,
        backend=backend,
        config=FIT_CONFIG,
    )


def _fit_best(
    images: np.ndarray, masks: np.ndarray, backend: str
) -> tuple[np.ndarray, np.ndarray]:
    """Fit from both valley starts and keep the better chi-square row."""

    parameters: np.ndarray | None = None
    converged: np.ndarray | None = None
    best_chi_square: np.ndarray | None = None
    for separation in START_SEPARATIONS:
        result = _fit_start(images, masks, separation, backend)
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


def _recovered_offsets(parameters: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            parameters[:, 2] - parameters[:, 0],
            parameters[:, 3] - parameters[:, 1],
        ],
        axis=1,
    )


def test_lanczos3_template_offset_recovery() -> None:
    images, masks, truths = _dipole_batch("lanczos3")
    parameters, converged = _fit_best(images, masks, "numpy")
    assert converged.all()

    recovered = _recovered_offsets(parameters)
    offset_error = np.hypot(*(recovered - truths[:, :2]).T)
    assert float(np.median(offset_error)) < 0.30
    assert float(np.max(offset_error)) < 0.6

    alignment = np.sum(recovered * truths[:, :2], axis=1) / (
        np.hypot(*recovered.T) * np.hypot(*truths[:, :2].T)
    )
    direction_error = np.degrees(np.arccos(np.clip(alignment, -1.0, 1.0)))
    assert float(np.max(direction_error)) < 10.0

    recovered_moment = parameters[:, 4:5] * recovered
    true_moment = truths[:, 2:3] * truths[:, :2]
    moment_error = np.hypot(*(recovered_moment - true_moment).T) / np.hypot(
        *true_moment.T
    )
    assert float(np.max(moment_error)) < 0.10
    assert float(np.median(moment_error)) < 0.05


def test_lanczos3_resampling_is_astrometrically_transparent() -> None:
    """Compare xRep-resampled and exactly shifted templates when resolved.

    In the resolved high signal-to-noise regime the flux-separation
    degeneracy is broken, so the paired difference between the two template
    constructions isolates the astrometric error contributed by Lanczos-3
    resampling itself.
    """

    resolved = ((2.2, 110.0), (3.0, 35.0))
    images, masks, _ = _dipole_batch(
        "lanczos3", offsets=resolved, peak_snr=300.0
    )
    resampled_parameters, resampled_converged = _fit_best(
        images, masks, "numpy"
    )
    exact_images, exact_masks, _ = _dipole_batch(
        "fourier", offsets=resolved, peak_snr=300.0
    )
    exact_parameters, exact_converged = _fit_best(
        exact_images, exact_masks, "numpy"
    )

    assert resampled_converged.all()
    assert exact_converged.all()
    difference = np.hypot(
        *(
            _recovered_offsets(resampled_parameters)
            - _recovered_offsets(exact_parameters)
        ).T
    )
    assert float(np.median(difference)) < 0.05
    assert float(np.max(difference)) < 0.10


def test_offset_recovery_matches_between_numpy_and_cupy() -> None:
    """NumPy and CuPy agree at round-off on jointly converged rows.

    A row on the flux-separation degeneracy valley may sit exactly at a
    convergence threshold, so one backend can report ``no_progress`` where
    the other converges; the parity contract is round-off agreement on the
    rows both backends converge, with at most one threshold flip.
    """

    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device is unavailable")
    except Exception:
        pytest.skip("CUDA runtime is unavailable")
    images, masks, _ = _dipole_batch("lanczos3")
    for separation in START_SEPARATIONS:
        cpu = _fit_start(images, masks, separation, "numpy")
        gpu = _fit_start(images, masks, separation, "cupy")
        assert gpu.backend == "cupy"
        assert int((cpu.status == gpu.status).sum()) >= len(images) - 1
        both = cpu.converged & gpu.converged
        assert int(both.sum()) >= len(images) - 1
        assert np.allclose(
            cpu.parameters[both],
            gpu.parameters[both],
            rtol=1.0e-7,
            atol=1.0e-7,
        )
