# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""NumPy preprocessing helpers for xImage reconstruction."""

from __future__ import annotations

import numpy as np
from numpy.typing import DTypeLike
from scipy.ndimage import maximum_filter

from cuphoton.ximage.data import XImageBundle


def round_down_to_odd(n: float) -> int:
    if n <= 0:
        raise ValueError(f"round_down_to_odd: expected positive n, got {n}")
    if (n % 2) == 1:
        return int(n)
    result = int(np.ceil(n) // 2 * 2 - 1)
    if result < 1:
        raise ValueError(f"round_down_to_odd: no positive odd integer <= {n}")
    return result


def create_object_masks(
    object_masks: np.ndarray, df: int = 1, dtype: DTypeLike = np.uint8
) -> tuple[np.ndarray, np.ndarray]:
    """Return good-pixel masks from the xImage twelve-plane mask schema.

    The input shape is ``(batch, height, width, plane, exposure)``. Planes
    0, 4, and 8 mark chip gaps and borders. Conservative masking additionally
    excludes stripe planes 1, 7, and 9 and dilates large-source plane 10 by
    a ``df``-pixel square. Other planes do not affect these mask policies.
    """

    object_dict = {
        "chip_gap_border_1": 0,
        "stripes_1": 1,
        "chip_gap_border_2": 2,
        "small_sources": 3,
        "chip_gap_border_3": 4,
        "various_sources": 5,
        "null_1": 6,
        "stripes_2": 7,
        "chip_gap": 8,
        "stripes_3": 9,
        "large_sources": 10,
        "null_2": 11,
    }

    exclusions_noncon = ["chip_gap_border_1", "chip_gap_border_3", "chip_gap"]
    excluded_objects_noncon = object_masks[
        :, :, :, object_dict[exclusions_noncon[0]], :
    ]
    for key in exclusions_noncon[1:]:
        excluded_objects_noncon = np.logical_or(
            excluded_objects_noncon,
            object_masks[:, :, :, object_dict[key], :],
        )
    non_conservative = np.logical_not(excluded_objects_noncon).astype(dtype)

    large_sources = np.array(
        object_masks[:, :, :, object_dict["large_sources"], :], copy=True
    )
    for batch in range(large_sources.shape[0]):
        for img in range(large_sources.shape[-1]):
            large_sources[batch, :, :, img] = maximum_filter(
                large_sources[batch, :, :, img].astype(dtype),
                size=(df, df),
                mode="constant",
                cval=0,
            )
    excluded_objects_con = np.logical_or(
        excluded_objects_noncon, large_sources
    )

    for key in ["stripes_1", "stripes_2", "stripes_3"]:
        excluded_objects_con = np.logical_or(
            excluded_objects_con,
            object_masks[:, :, :, object_dict[key], :],
        )
    conservative = np.logical_not(excluded_objects_con).astype(dtype)
    return conservative, non_conservative


def _exposure_times(bundle: XImageBundle) -> np.ndarray:
    exp_times = np.asarray(bundle.exp_times, dtype=np.float64)
    n_exposures = bundle.images.shape[3]
    if exp_times.shape != (n_exposures,):
        raise ValueError(
            f"Expected {n_exposures} exposure times, saw shape "
            f"{exp_times.shape}"
        )
    if not np.all(np.isfinite(exp_times) & (exp_times > 0)):
        raise ValueError(
            "exposure times must be finite and strictly positive"
        )
    return exp_times.reshape(1, 1, 1, -1)


def prepare_reconstruction_inputs(
    bundle: XImageBundle, mask_mode: str, dilation_factor: int
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    exp_times = _exposure_times(bundle)
    images = np.asarray(bundle.images) / exp_times
    variances = np.asarray(bundle.variances) / (exp_times**2)

    if mask_mode == "none":
        masks = None
    else:
        conservative, non_conservative = create_object_masks(
            np.asarray(bundle.masks), df=dilation_factor
        )
        if mask_mode == "conservative":
            masks = conservative
        elif mask_mode == "non_conservative":
            masks = non_conservative
        else:
            raise ValueError(f"Unsupported mask mode: {mask_mode}")
    return images, masks, variances


def prepare_reconstruction_inputs_roi(
    bundle: XImageBundle,
    *,
    mask_mode: str,
    dilation_factor: int,
    center_roi: tuple[int, int] | list[int] | None = None,
    height_roi: int | None = None,
    width_roi: int | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    slice_h, slice_w = spatial_roi_slices(
        (bundle.images.shape[1], bundle.images.shape[2]),
        center_roi=center_roi,
        height_roi=height_roi,
        width_roi=width_roi,
    )
    exp_times = _exposure_times(bundle)
    images = np.asarray(bundle.images[:, slice_h, slice_w, :]) / exp_times
    variances = np.asarray(bundle.variances[:, slice_h, slice_w, :]) / (
        exp_times**2
    )

    if mask_mode == "none":
        masks = None
    else:
        mask_roi = np.asarray(bundle.masks[:, slice_h, slice_w, :, :])
        conservative, non_conservative = create_object_masks(
            mask_roi, df=dilation_factor
        )
        if mask_mode == "conservative":
            masks = conservative
        elif mask_mode == "non_conservative":
            masks = non_conservative
        else:
            raise ValueError(f"Unsupported mask mode: {mask_mode}")
    return images, masks, variances


def select_cutout(
    images: np.ndarray,
    masks: np.ndarray | None = None,
    variances: np.ndarray | None = None,
    center_roi: tuple[int, int] | list[int] | None = None,
    height_roi: int | None = None,
    width_roi: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if masks is None:
        masks = np.ones(images.shape, dtype=np.float64)
    if variances is None:
        variances = np.ones(images.shape, dtype=np.float64)
    slice_h, slice_w = spatial_roi_slices(
        images.shape[1:3],
        center_roi=center_roi,
        height_roi=height_roi,
        width_roi=width_roi,
    )
    return (
        images[:, slice_h, slice_w, :],
        masks[:, slice_h, slice_w, :],
        variances[:, slice_h, slice_w, :],
    )


def spatial_roi_slices(
    image_shape: tuple[int, int],
    *,
    center_roi: tuple[int, int] | list[int] | None = None,
    height_roi: int | None = None,
    width_roi: int | None = None,
) -> tuple[slice, slice]:
    if center_roi is None:
        center_roi = [image_shape[0] // 2, image_shape[1] // 2]
    if height_roi is None:
        height_roi = image_shape[0]
    if width_roi is None:
        width_roi = image_shape[1]

    if height_roi <= 0 or width_roi <= 0:
        raise ValueError("ROI height and width must be positive.")

    start_y = center_roi[0] - height_roi // 2
    start_x = center_roi[1] - width_roi // 2
    stop_y = start_y + height_roi
    stop_x = start_x + width_roi
    if (
        start_y < 0
        or start_x < 0
        or stop_y > image_shape[0]
        or stop_x > image_shape[1]
    ):
        raise ValueError(
            f"ROI exceeds image bounds {image_shape}: "
            f"requested y=[{start_y}, {stop_y}), "
            f"x=[{start_x}, {stop_x})."
        )
    slice_h = slice(start_y, stop_y)
    slice_w = slice(start_x, stop_x)
    return slice_h, slice_w


def upsample_arrays(
    images: np.ndarray,
    masks: np.ndarray | None = None,
    variances: np.ndarray | None = None,
    srf: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if masks is None:
        masks = np.ones(images.shape, dtype=np.float64)
    if variances is None:
        variances = np.ones(images.shape, dtype=np.float64)
    if srf == 1:
        return images, masks, variances
    return (
        np.repeat(np.repeat(images, srf, axis=1), srf, axis=2),
        np.repeat(np.repeat(masks, srf, axis=1), srf, axis=2),
        np.repeat(np.repeat(variances, srf, axis=1), srf, axis=2),
    )


def pad_with_data(
    images: np.ndarray,
    masks: np.ndarray | None = None,
    variances: np.ndarray | None = None,
    center_roi: tuple[int, int] | list[int] | None = None,
    height_roi: int | None = None,
    width_roi: int | None = None,
    srf: int = 1,
    psf_size: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pad an upsampled ROI using existing surrounding image pixels.

    Each edge requires ``ceil((psf_size // 2) / srf)`` available input
    pixels beyond the ROI. Raise ``ValueError`` when that region extends
    outside the input; no pixels are synthesized at image boundaries.
    """

    if masks is None:
        masks = np.ones(images.shape, dtype=np.float64)
    if variances is None:
        variances = np.ones(images.shape, dtype=np.float64)
    if center_roi is None:
        center_roi = [images.shape[1] // 2, images.shape[2] // 2]
    if height_roi is None:
        height_roi = images.shape[1]
    if width_roi is None:
        width_roi = images.shape[2]

    if srf < 1 or psf_size < 1:
        raise ValueError("srf and psf_size must be positive")
    half_pad_size = psf_size // 2
    source_padding = (half_pad_size + srf - 1) // srf
    start_y = center_roi[0] - height_roi // 2
    start_x = center_roi[1] - width_roi // 2
    if (
        start_y < source_padding
        or start_x < source_padding
        or start_y + height_roi + source_padding > images.shape[1]
        or start_x + width_roi + source_padding > images.shape[2]
    ):
        raise ValueError(
            "Padded ROI exceeds the input image bounds; "
            f"the requested PSF and srf require {source_padding} "
            "input pixels beyond each ROI edge."
        )

    y_cutout, m_cutout, v_cutout = select_cutout(
        images,
        masks=masks,
        variances=variances,
        center_roi=center_roi,
        height_roi=height_roi + 2 * source_padding,
        width_roi=width_roi + 2 * source_padding,
    )
    y_plus, m_plus, v_plus = upsample_arrays(
        y_cutout, m_cutout, v_cutout, srf=srf
    )
    # Preserve the ROI origin within the enlarged, upsampled source cutout.
    offset_y = srf * source_padding
    offset_x = srf * source_padding
    slice_h = slice(
        offset_y - half_pad_size,
        offset_y + srf * height_roi + half_pad_size,
    )
    slice_w = slice(
        offset_x - half_pad_size,
        offset_x + srf * width_roi + half_pad_size,
    )
    return (
        y_plus[:, slice_h, slice_w, :],
        m_plus[:, slice_h, slice_w, :],
        v_plus[:, slice_h, slice_w, :],
    )


def coadd(
    images: np.ndarray,
    masks: np.ndarray | float = 1,
    variances: np.ndarray | float = 1,
    method: str = "median",
    positive: bool = False,
) -> np.ndarray:
    """Combine valid exposures in their input units with equal weights.

    Masks select contributors; variances exclude invalid samples without
    rescaling their flux. Pixels with no valid contributors are zero.
    """

    values = np.asarray(images, dtype=np.result_type(images, np.float64))
    valid = (
        np.isfinite(values)
        & np.isfinite(masks)
        & (np.asarray(masks) > 0)
        & np.isfinite(variances)
        & (np.asarray(variances) > 0)
    )
    count = np.sum(valid, axis=-1, keepdims=True)
    if method == "median":
        samples = np.where(valid, values, np.nan)
        samples = np.where(count > 0, samples, 0.0)
        result = np.nanmedian(samples, axis=-1, keepdims=True)
    elif method == "mean":
        total = np.sum(np.where(valid, values, 0.0), axis=-1, keepdims=True)
        result = np.divide(
            total, count, out=np.zeros_like(total), where=count > 0
        )
    else:
        raise ValueError("method must be 'median' or 'mean'")

    if positive:
        min_value = np.min(result)
        if min_value <= 0:
            scale = np.max(np.abs(result), axis=(1, 2), keepdims=True)
            epsilon = 1e-6 * np.where(scale > 0, scale, 1.0)
            result = result - min_value + epsilon
    return result
