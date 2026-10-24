# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Torch-native reconstruction preprocessing helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..preprocess import spatial_roi_slices


@dataclass
class TorchReconstructionInputs:
    x0: Any
    y: Any
    m: Any
    v: Any
    w: Any
    f: Any
    metadata: dict[str, Any]


def prepare_torch_reconstruction_inputs(
    bundle: Any,
    config: Any,
    *,
    device: Any,
    dtype: Any,
    psfs: np.ndarray | None = None,
) -> TorchReconstructionInputs:
    """Transfer the requested ROI and prepare reconstruction tensors.

    VALID padding also transfers the surrounding source pixels required to
    initialize the padded reconstruction. Coordinates and raw-shape metadata
    refer to the original input bundle.
    """

    if config.padding == "SAME" and config.srf != 1:
        raise ValueError("SAME padding requires srf=1 for reconstruction.")

    torch = _load_torch()
    timings: dict[str, float] = {}

    start = _start_timer(torch, device)
    psf_values = bundle.psfs if psfs is None else psfs
    if len(psf_values.shape) != 4 or psf_values.shape[2] != 1:
        raise ValueError(
            "Torch reconstruction requires PSFs with shape "
            "(height, width, 1, exposure); "
            f"received {psf_values.shape}."
        )
    if (
        psf_values.shape[0] != psf_values.shape[1]
        or psf_values.shape[0] % 2 != 1
    ):
        raise ValueError("Reconstruction PSF kernels must be square and odd.")

    image_shape = bundle.images.shape[1:3]
    slice_h, slice_w = spatial_roi_slices(
        image_shape,
        center_roi=config.roi.center,
        height_roi=config.roi.height,
        width_roi=config.roi.width,
    )
    height_roi = slice_h.stop - slice_h.start
    width_roi = slice_w.stop - slice_w.start
    source_padding = 0
    if config.padding == "VALID":
        srf = int(config.srf)
        if srf < 1:
            raise ValueError("srf and psf_size must be positive")
        source_padding = (psf_values.shape[0] // 2 + srf - 1) // srf
        if (
            slice_h.start < source_padding
            or slice_w.start < source_padding
            or slice_h.stop + source_padding > image_shape[0]
            or slice_w.stop + source_padding > image_shape[1]
        ):
            raise ValueError(
                "Padded ROI exceeds the input image bounds; "
                f"the requested PSF and srf require {source_padding} "
                "input pixels beyond each ROI edge."
            )
    source_h = slice(
        slice_h.start - source_padding, slice_h.stop + source_padding
    )
    source_w = slice(
        slice_w.start - source_padding, slice_w.stop + source_padding
    )
    center_roi = (
        source_padding + height_roi // 2,
        source_padding + width_roi // 2,
    )
    images = _as_device_tensor(
        torch,
        bundle.images[:, source_h, source_w, :],
        device=device,
        dtype=dtype,
    )
    variances = _as_device_tensor(
        torch,
        bundle.variances[:, source_h, source_w, :],
        device=device,
        dtype=dtype,
    )
    object_masks = _as_device_tensor(
        torch,
        bundle.masks[:, source_h, source_w, :, :],
        device=device,
        dtype=torch.uint8,
    )
    exp_times = _as_device_tensor(
        torch, bundle.exp_times, device=device, dtype=dtype
    )
    psfs_bhwc = _as_device_tensor(
        torch, psf_values, device=device, dtype=dtype
    )
    timings["host_to_device"] = _stop_timer(torch, device, start)

    start = _start_timer(torch, device)
    exp_scale = exp_times.reshape(1, 1, 1, -1)
    images = images / exp_scale
    variances = variances / exp_scale.square()
    masks = _create_object_masks_torch(
        torch,
        object_masks,
        mask_mode=config.mask.mode,
        dilation_factor=int(config.mask.dilation_factor),
        dtype=dtype,
    )
    y_cutout, m_cutout, v_cutout = _select_cutout(
        torch,
        images,
        masks=masks,
        variances=variances,
        center_roi=center_roi,
        height_roi=height_roi,
        width_roi=width_roi,
    )
    if bool(torch.any(v_cutout <= 0).detach().cpu().item()):
        raise RuntimeError(
            "Variance images must be strictly positive for Torch "
            "reconstruction."
        )
    timings["preprocess"] = _stop_timer(torch, device, start)

    start = _start_timer(torch, device)
    if config.padding == "VALID":
        y0_cutout, _, _ = _pad_with_data(
            torch,
            images,
            masks=masks,
            variances=variances,
            center_roi=center_roi,
            height_roi=height_roi,
            width_roi=width_roi,
            srf=int(config.srf),
            psf_size=int(psfs_bhwc.shape[0]),
        )
    else:
        y0_cutout = y_cutout

    if config.x_initialization == "constant":
        initial_bhwe = 0.1 * torch.ones_like(y0_cutout[:, :, :, [0]])
    else:
        initial_bhwe = _coadd(
            torch,
            y0_cutout,
            method=config.x_initialization,
            positive=True,
        )
    timings["initialize"] = _stop_timer(torch, device, start)

    weights_bhwe = m_cutout / torch.clamp(v_cutout, min=1e-30)
    x0 = _bhwe_to_behw(initial_bhwe)
    y = _bhwe_to_behw(y_cutout)
    m = _bhwe_to_behw(m_cutout)
    v = _bhwe_to_behw(v_cutout)
    w = _bhwe_to_behw(weights_bhwe)
    f = psfs_bhwc.permute(3, 2, 0, 1).contiguous()

    metadata = {
        "timings_sec": timings,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "raw_shapes": {
            "images": list(bundle.images.shape),
            "masks": list(bundle.masks.shape),
            "variances": list(bundle.variances.shape),
            "psfs": list(psfs_bhwc.shape),
        },
        "tensor_shapes": {
            "x0": list(x0.shape),
            "y": list(y.shape),
            "m": list(m.shape),
            "v": list(v.shape),
            "w": list(w.shape),
            "f": list(f.shape),
        },
        "mask_summary": _mask_summary(torch, m_cutout),
        "finite_summary": _finite_summary(torch, y_cutout, v_cutout),
        "preprocess_level": (
            "torch_device_from_host_store"
            if device.type == "cuda"
            else "torch_cpu_from_host_store"
        ),
    }
    return TorchReconstructionInputs(
        x0=x0,
        y=y,
        m=m,
        v=v,
        w=w,
        f=f,
        metadata=metadata,
    )


def _load_torch():
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            "Torch GPU preprocessing requires the torch package."
        ) from exc
    return torch


def _as_device_tensor(torch: Any, value: Any, *, device: Any, dtype: Any):
    return torch.tensor(np.asarray(value), device=device, dtype=dtype)


def _start_timer(torch: Any, device: Any) -> float:
    _sync_if_cuda(torch, device)
    return time.perf_counter()


def _stop_timer(torch: Any, device: Any, start: float) -> float:
    _sync_if_cuda(torch, device)
    return time.perf_counter() - start


def _sync_if_cuda(torch: Any, device: Any) -> None:
    if getattr(device, "type", None) == "cuda":
        torch.cuda.synchronize(device)


def _create_object_masks_torch(
    torch: Any,
    object_masks: Any,
    *,
    mask_mode: str,
    dilation_factor: int,
    dtype: Any,
):
    if mask_mode == "none":
        shape = (
            object_masks.shape[0],
            object_masks.shape[1],
            object_masks.shape[2],
            object_masks.shape[4],
        )
        return torch.ones(shape, device=object_masks.device, dtype=dtype)

    object_index = {
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
    mask_bool = object_masks.bool()
    excluded_noncon = (
        mask_bool[:, :, :, object_index["chip_gap_border_1"], :]
        | mask_bool[:, :, :, object_index["chip_gap_border_3"], :]
        | mask_bool[:, :, :, object_index["chip_gap"], :]
    )
    if mask_mode == "non_conservative":
        return (~excluded_noncon).to(dtype=dtype)

    if mask_mode != "conservative":
        raise ValueError(f"Unsupported mask mode: {mask_mode}")
    if dilation_factor != 1:
        raise NotImplementedError(
            "Torch conservative mask preprocessing currently supports "
            "dilation_factor=1 only."
        )

    large_sources = mask_bool[:, :, :, object_index["large_sources"], :]
    excluded_con = excluded_noncon | large_sources
    for key in ("stripes_1", "stripes_2", "stripes_3"):
        excluded_con = excluded_con | mask_bool[:, :, :, object_index[key], :]
    return (~excluded_con).to(dtype=dtype)


def _select_cutout(
    torch: Any,
    images: Any,
    *,
    masks: Any | None = None,
    variances: Any | None = None,
    center_roi: list[int] | tuple[int, int] | None = None,
    height_roi: int | None = None,
    width_roi: int | None = None,
):
    if masks is None:
        masks = torch.ones_like(images)
    if variances is None:
        variances = torch.ones_like(images)
    slice_h, slice_w = spatial_roi_slices(
        (int(images.shape[1]), int(images.shape[2])),
        center_roi=center_roi,
        height_roi=height_roi,
        width_roi=width_roi,
    )
    return (
        images[:, slice_h, slice_w, :],
        masks[:, slice_h, slice_w, :],
        variances[:, slice_h, slice_w, :],
    )


def _upsample_arrays(
    torch: Any, images: Any, masks: Any, variances: Any, srf: int
):
    if srf == 1:
        return images, masks, variances
    return (
        torch.repeat_interleave(
            torch.repeat_interleave(images, srf, dim=1), srf, dim=2
        ),
        torch.repeat_interleave(
            torch.repeat_interleave(masks, srf, dim=1), srf, dim=2
        ),
        torch.repeat_interleave(
            torch.repeat_interleave(variances, srf, dim=1), srf, dim=2
        ),
    )


def _pad_with_data(
    torch: Any,
    images: Any,
    *,
    masks: Any | None = None,
    variances: Any | None = None,
    center_roi: list[int] | tuple[int, int] | None = None,
    height_roi: int | None = None,
    width_roi: int | None = None,
    srf: int = 1,
    psf_size: int = 1,
):
    if masks is None:
        masks = torch.ones_like(images)
    if variances is None:
        variances = torch.ones_like(images)
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

    y_cutout, m_cutout, v_cutout = _select_cutout(
        torch,
        images,
        masks=masks,
        variances=variances,
        center_roi=center_roi,
        height_roi=height_roi + 2 * source_padding,
        width_roi=width_roi + 2 * source_padding,
    )
    y_plus, m_plus, v_plus = _upsample_arrays(
        torch, y_cutout, m_cutout, v_cutout, srf=srf
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


def _coadd(torch: Any, images: Any, *, method: str, positive: bool):
    if method == "median":
        result = torch.quantile(images, 0.5, dim=-1, keepdim=True)
    elif method == "mean":
        result = torch.mean(images, dim=-1, keepdim=True)
    else:
        raise ValueError("method must be 'median' or 'mean'")

    if positive:
        min_value = torch.min(result)
        if bool((min_value <= 0).detach().cpu().item()):
            scale = torch.amax(torch.abs(result), dim=(1, 2), keepdim=True)
            epsilon = 1e-6 * torch.where(
                scale > 0, scale, torch.ones_like(scale)
            )
            result = result - min_value + epsilon
    return result


def _bhwe_to_behw(value: Any):
    return value.permute(0, 3, 1, 2).contiguous()


def _mask_summary(torch: Any, mask: Any) -> dict[str, Any]:
    mask_float = mask.to(dtype=torch.float64)
    per_exposure = torch.mean(mask_float, dim=(0, 1, 2))
    return {
        "unmasked_fraction": float(
            torch.mean(mask_float).detach().cpu().item()
        ),
        "per_exposure_unmasked_fraction": [
            float(value) for value in per_exposure.detach().cpu().tolist()
        ],
        "zero_weight_pixel_count": int(
            torch.count_nonzero(mask == 0).detach().cpu().item()
        ),
    }


def _finite_summary(
    torch: Any, images: Any, variances: Any
) -> dict[str, Any]:
    return {
        "image_nan_count": int(
            torch.isnan(images).sum().detach().cpu().item()
        ),
        "image_inf_count": int(
            torch.isinf(images).sum().detach().cpu().item()
        ),
        "variance_nan_count": int(
            torch.isnan(variances).sum().detach().cpu().item()
        ),
        "variance_inf_count": int(
            torch.isinf(variances).sum().detach().cpu().item()
        ),
    }
