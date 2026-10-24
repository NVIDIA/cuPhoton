# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Image similarity, background statistics, and observation-grid helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter


def image_metrics(
    image: np.ndarray,
    background_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Report finite coverage; Fourier sharpness needs a complete image."""

    finite = np.isfinite(image)
    complete = bool(np.all(finite))
    result: dict[str, Any] = {
        "sharpness_sf": fourier_sharpness(image) if complete else None,
        "sky_sigma": sky_sigma(image, background_mask=background_mask),
        "finite_pixel_fraction": float(np.mean(finite)),
    }
    if not complete:
        result["sharpness_sf_unavailable_reason"] = (
            "Fourier sharpness requires a complete finite image."
        )
    return result


def fourier_sharpness(image: np.ndarray) -> float:
    image_2d = np.asarray(image, dtype=np.float64)
    if not np.all(np.isfinite(image_2d)):
        return float("nan")
    spectrum = np.fft.fft2(image_2d)
    return float(np.mean(np.log(np.abs(spectrum) + 1e-12)))


def sky_sigma(
    image: np.ndarray,
    *,
    background_mask: np.ndarray | None = None,
) -> float:
    arr = np.asarray(image, dtype=np.float64)
    valid = np.isfinite(arr)
    if background_mask is not None:
        valid &= background_mask
    values = arr[valid]
    if values.size == 0:
        return float("nan")
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    scale = 1.4826 * mad
    if scale == 0:
        return float(np.std(values))
    clipped = values[np.abs(values - median) <= 3.0 * scale]
    if clipped.size == 0:
        clipped = values
    return float(np.std(clipped))


def psnr(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Compare values and dynamic range on jointly finite pixels."""

    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    valid = np.isfinite(ref) & np.isfinite(cand)
    if not np.any(valid):
        return float("nan")
    ref = ref[valid]
    cand = cand[valid]
    mse = np.mean((ref - cand) ** 2)
    if mse == 0:
        return float("inf")
    data_min = min(float(ref.min()), float(cand.min()))
    data_max = max(float(ref.max()), float(cand.max()))
    data_range = max(data_max - data_min, 1e-12)
    return float(20.0 * np.log10(data_range) - 10.0 * np.log10(mse))


def ssim(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Average SSIM on joint finite support with normalized local moments."""

    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    valid = np.isfinite(ref) & np.isfinite(cand)
    if not np.any(valid):
        return float("nan")
    data_min = min(float(ref[valid].min()), float(cand[valid].min()))
    data_max = max(float(ref[valid].max()), float(cand[valid].max()))
    data_range = max(data_max - data_min, 1e-12)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    weight: float | np.ndarray = 1.0
    if not np.all(valid):
        ref = np.where(valid, ref, 0.0)
        cand = np.where(valid, cand, 0.0)
        weight = gaussian_filter(
            valid.astype(float), sigma=1.5, mode="reflect"
        )
        weight = np.where(weight > 0.0, weight, 1.0)

    mu_ref = gaussian_filter(ref, sigma=1.5, mode="reflect") / weight
    mu_cand = gaussian_filter(cand, sigma=1.5, mode="reflect") / weight
    mu_ref_sq = mu_ref * mu_ref
    mu_cand_sq = mu_cand * mu_cand
    mu_ref_cand = mu_ref * mu_cand

    sigma_ref_sq = (
        gaussian_filter(ref * ref, sigma=1.5, mode="reflect") / weight
        - mu_ref_sq
    )
    sigma_cand_sq = (
        gaussian_filter(cand * cand, sigma=1.5, mode="reflect") / weight
        - mu_cand_sq
    )
    sigma_ref_cand = (
        gaussian_filter(ref * cand, sigma=1.5, mode="reflect") / weight
        - mu_ref_cand
    )
    sigma_ref_sq = np.maximum(sigma_ref_sq, 0.0)
    sigma_cand_sq = np.maximum(sigma_cand_sq, 0.0)

    numerator = (2 * mu_ref_cand + c1) * (2 * sigma_ref_cand + c2)
    denominator = (mu_ref_sq + mu_cand_sq + c1) * (
        sigma_ref_sq + sigma_cand_sq + c2
    )
    score = numerator / np.maximum(denominator, 1e-12)
    return float(np.mean(score[valid]))


def build_background_mask(
    masks: np.ndarray | None,
    *,
    min_fraction: float,
) -> np.ndarray:
    if masks is None:
        raise ValueError("Masks are required to construct a background mask.")
    mask_stack = squeeze_exposure_stack(masks)
    return np.mean(mask_stack, axis=0) >= min_fraction


def project_restoration_to_observation_grid(
    image: np.ndarray,
    *,
    target_shape: tuple[int, int],
    srf: int,
) -> np.ndarray:
    arr = squeeze_single_image(image)
    if srf > 1:
        arr = average_downsample(arr, factor=srf)
    return center_crop(arr, target_shape)


def average_downsample(image: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return image
    height = (image.shape[0] // factor) * factor
    width = (image.shape[1] // factor) * factor
    trimmed = image[:height, :width]
    return trimmed.reshape(
        height // factor,
        factor,
        width // factor,
        factor,
    ).mean(axis=(1, 3))


def center_crop(
    image: np.ndarray,
    target_shape: tuple[int, int],
) -> np.ndarray:
    target_h, target_w = target_shape
    source_h, source_w = image.shape
    if target_h > source_h or target_w > source_w:
        raise ValueError(
            f"Cannot center-crop {image.shape} to {target_shape}."
        )
    start_h = (source_h - target_h) // 2
    start_w = (source_w - target_w) // 2
    return image[
        start_h : start_h + target_h,
        start_w : start_w + target_w,
    ]


def squeeze_single_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 4:
        if arr.shape[0] != 1 or arr.shape[-1] != 1:
            raise ValueError(
                f"Expected a single image, saw shape {arr.shape}."
            )
        return arr[0, :, :, 0]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Expected rank-2 or rank-4 image, saw {arr.shape}.")


def squeeze_exposure_stack(images: np.ndarray) -> np.ndarray:
    arr = np.asarray(images)
    if arr.ndim != 4 or arr.shape[0] != 1:
        raise ValueError(
            "Expected exposure stack with shape (1, H, W, N), "
            f"saw {arr.shape}."
        )
    return np.transpose(arr[0], (2, 0, 1))


def crop_about_center(
    image: np.ndarray,
    *,
    center: tuple[int, int],
    height: int,
    width: int,
) -> np.ndarray:
    start_h = center[0] - height // 2
    start_w = center[1] - width // 2
    stop_h = start_h + height
    stop_w = start_w + width
    if (
        start_h < 0
        or start_w < 0
        or stop_h > image.shape[0]
        or stop_w > image.shape[1]
    ):
        raise ValueError(
            f"Requested ROI {(height, width)} around {center} exceeds "
            f"image bounds {image.shape}."
        )
    cropped = image[start_h:stop_h, start_w:stop_w]
    if cropped.shape != (height, width):
        raise ValueError(
            f"Expected cropped shape {(height, width)}, saw {cropped.shape}."
        )
    return cropped


def crop_bounds_about_center(
    *,
    center: tuple[int, int],
    height: int,
    width: int,
    image_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    start_h = center[0] - height // 2
    start_w = center[1] - width // 2
    stop_h = start_h + height
    stop_w = start_w + width
    if (
        start_h < 0
        or start_w < 0
        or stop_h > image_shape[0]
        or stop_w > image_shape[1]
    ):
        raise ValueError(
            f"Requested ROI {(height, width)} around {center} exceeds "
            f"image bounds {image_shape}."
        )
    return start_h, start_w, stop_h, stop_w


def evaluate_image_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    background_mask: np.ndarray | None = None,
    backend: str = "auto",
) -> dict[str, Any]:
    """Evaluate two equally shaped, nonempty 2-D images in float64.

    ``auto`` uses CuPy when it can initialize a CUDA device, otherwise the
    NumPy/SciPy reference. ``cpu`` never imports CuPy; ``cupy`` requires it.
    GPU computation errors propagate after backend selection. GPU reductions
    and Gaussian filtering can differ from CPU results by floating-point
    rounding. Fourier sharpness's logarithm amplifies FFT roundoff at zero
    or nearly zero coefficients; such spectra need not closely match between
    backends. One packed result transfer completes all GPU metrics.

    Fourier sharpness is unavailable for incomplete images. Sky statistics
    use finite pixels in the optional shared background mask. PSNR and SSIM
    use jointly finite support; no common pixels yields NaN. Identical valid
    values yield infinite PSNR. Returns both image summaries, pair metrics,
    joint finite coverage, and the selected ``backend``.
    """

    metrics = _evaluate_metric_group(
        [reference, candidate],
        background_mask=background_mask,
        backend=backend,
    )
    return {
        "backend": metrics["backend"],
        "reference": metrics["images"][0],
        "candidate": metrics["images"][1],
        **metrics["comparisons"][0],
    }


def _evaluate_metric_group(
    images: list[np.ndarray],
    *,
    background_mask: np.ndarray | None,
    backend: str,
    exposures: np.ndarray | None = None,
) -> dict[str, Any]:
    if backend not in {"auto", "cpu", "cupy"}:
        raise ValueError("metrics backend must be 'auto', 'cpu', or 'cupy'")
    arrays = [np.asarray(image, dtype=np.float64) for image in images]
    shape = arrays[0].shape
    if (
        len(shape) != 2
        or not all(shape)
        or any(image.shape != shape for image in arrays)
    ):
        raise ValueError(
            "Metric images must be nonempty 2-D arrays of equal shape"
        )
    if background_mask is not None:
        background_mask = np.asarray(background_mask, dtype=bool)
        if background_mask.shape != shape:
            raise ValueError("background_mask must match the image shape")
    exposure_images = [] if exposures is None else list(exposures)
    if any(image.shape != shape for image in exposure_images):
        raise ValueError("Exposure images must match the reference shape")
    if backend != "cpu":
        from cuphoton.ximage._metrics_cupy import (
            compute_metrics,
            load_cupy,
        )

        try:
            cp, gpu_gaussian_filter = load_cupy()
        except RuntimeError:
            if backend == "cupy":
                raise
        else:
            summaries, comparisons = compute_metrics(
                arrays,
                background_mask,
                exposure_images,
                cp,
                gpu_gaussian_filter,
            )
            image_results = []
            for sharpness, sigma, coverage in summaries:
                image_result: dict[str, Any] = {
                    "sharpness_sf": float(sharpness)
                    if coverage == 1.0
                    else None,
                    "sky_sigma": float(sigma),
                    "finite_pixel_fraction": float(coverage),
                }
                if coverage != 1.0:
                    image_result["sharpness_sf_unavailable_reason"] = (
                        "Fourier sharpness requires a complete finite image."
                    )
                image_results.append(image_result)
            pair_results = [
                {
                    "psnr": float(p),
                    "ssim": float(s),
                    "comparison_valid_fraction": float(coverage),
                }
                for p, s, coverage in comparisons
            ]
            return {
                "backend": "cupy",
                "images": image_results,
                "comparisons": pair_results[: len(arrays) - 1],
                "exposures": pair_results[len(arrays) - 1 :],
            }

    reference = arrays[0]
    pair_results = [
        {
            "psnr": psnr(reference, image),
            "ssim": ssim(reference, image),
            "comparison_valid_fraction": float(
                np.mean(np.isfinite(reference) & np.isfinite(image))
            ),
        }
        for image in [*arrays[1:], *exposure_images]
    ]
    return {
        "backend": "cpu",
        "images": [image_metrics(image, background_mask) for image in arrays],
        "comparisons": pair_results[: len(arrays) - 1],
        "exposures": pair_results[len(arrays) - 1 :],
    }
