# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""CuPy metric kernels, imported only when GPU evaluation is requested."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def load_cupy() -> tuple[Any, Callable[..., Any]]:
    """Load optional GPU libraries and initialize the current CUDA device."""

    message = (
        "CuPy evaluation requires CuPy and a usable CUDA device. "
        "Install cuphoton[ximage,gpu] for the optional GPU dependencies."
    )
    try:
        import cupy as cp
        from cupyx.scipy.ndimage import gaussian_filter
    except (ImportError, OSError) as exc:
        raise RuntimeError(message) from exc
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            raise RuntimeError(message)
        cp.cuda.Device().use()
    except cp.cuda.runtime.CUDARuntimeError as exc:
        raise RuntimeError(message) from exc
    return cp, gaussian_filter


def compute_metrics(
    images: list[np.ndarray],
    background_mask: np.ndarray | None,
    exposures: list[np.ndarray],
    cp: Any,
    gaussian_filter: Callable[..., Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute image summaries and comparisons with one scalar-array download.

    Image summaries share a batched FFT and robust reductions. Pair moments
    stay on the device; exposure comparisons reuse the uploaded reference and
    do not calculate unused exposure FFT or sky statistics. Processing pairs
    sequentially bounds scratch memory independently of the exposure count.
    """

    device_images = cp.asarray(np.stack(images))
    finite = cp.isfinite(device_images)
    coverage = cp.mean(finite, axis=(1, 2))
    complete = cp.all(finite, axis=(1, 2))
    cleaned = cp.where(finite, device_images, 0.0)
    spectrum = cp.fft.fft2(cleaned, axes=(-2, -1))
    sharpness = cp.mean(cp.log(cp.abs(spectrum) + 1e-12), axis=(1, 2))
    sharpness = cp.where(complete, sharpness, cp.nan)

    background = (
        True if background_mask is None else cp.asarray(background_mask)
    )
    values = cp.where(finite & background, device_images, cp.nan).reshape(
        len(images), -1
    )
    median = cp.nanmedian(values, axis=1, keepdims=True)
    scale = 1.4826 * cp.nanmedian(
        cp.abs(values - median), axis=1, keepdims=True
    )
    clip_mask = cp.abs(values - median) <= 3.0 * scale
    keep_original = (scale == 0) | (
        cp.sum(clip_mask, axis=1, keepdims=True) == 0
    )
    sigma = cp.nanstd(
        cp.where(clip_mask | keep_original, values, cp.nan), axis=1
    )
    summaries = cp.stack((sharpness, sigma, coverage), axis=1)

    reference, reference_finite = device_images[0], finite[0]
    pairs = [
        _pair_metrics(reference, image, reference_finite, cp, gaussian_filter)
        for image in device_images[1:]
    ]
    for exposure in exposures:
        pairs.append(
            _pair_metrics(
                reference,
                cp.asarray(exposure, dtype=cp.float64),
                reference_finite,
                cp,
                gaussian_filter,
            )
        )
    # One packed scalar download completes all metric kernels.
    packed = cp.asnumpy(
        cp.concatenate((summaries.ravel(), cp.stack(pairs).ravel()))
    )
    split = len(images) * 3
    return packed[:split].reshape(-1, 3), packed[split:].reshape(-1, 3)


def _pair_metrics(
    reference: Any,
    candidate: Any,
    reference_finite: Any,
    cp: Any,
    gaussian_filter: Callable[..., Any],
) -> Any:
    valid = reference_finite & cp.isfinite(candidate)
    count = cp.sum(valid)
    denominator_count = cp.maximum(count, 1)
    ref = cp.where(valid, reference, 0.0)
    cand = cp.where(valid, candidate, 0.0)
    minimum = cp.minimum(
        cp.min(cp.where(valid, ref, cp.inf)),
        cp.min(cp.where(valid, cand, cp.inf)),
    )
    maximum = cp.maximum(
        cp.max(cp.where(valid, ref, -cp.inf)),
        cp.max(cp.where(valid, cand, -cp.inf)),
    )
    data_range = cp.maximum(maximum - minimum, 1e-12)
    mse = cp.sum(cp.where(valid, (ref - cand) ** 2, 0.0)) / denominator_count
    psnr = 20.0 * cp.log10(data_range) - 10.0 * cp.log10(mse)
    psnr = cp.where(count > 0, psnr, cp.nan)

    planes = cp.stack((ref, cand, ref * ref, cand * cand, ref * cand, valid))
    moments = gaussian_filter(planes, sigma=(0.0, 1.5, 1.5), mode="reflect")
    weight = cp.where(moments[5] > 0.0, moments[5], 1.0)
    mu_ref, mu_cand = moments[0] / weight, moments[1] / weight
    var_ref = cp.maximum(moments[2] / weight - mu_ref * mu_ref, 0.0)
    var_cand = cp.maximum(moments[3] / weight - mu_cand * mu_cand, 0.0)
    covariance = moments[4] / weight - mu_ref * mu_cand
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    numerator = (2.0 * mu_ref * mu_cand + c1) * (2.0 * covariance + c2)
    denominator = (mu_ref**2 + mu_cand**2 + c1) * (var_ref + var_cand + c2)
    score = numerator / cp.maximum(denominator, 1e-12)
    ssim = cp.sum(cp.where(valid, score, 0.0)) / denominator_count
    ssim = cp.where(count > 0, ssim, cp.nan)
    return cp.stack((psnr, ssim, count / valid.size))
