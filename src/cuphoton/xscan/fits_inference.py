# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Direct inference from caller-aligned FITS planes and candidate centers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from cuphoton.core.fits_io import inspect_fits_image, read_fits_images

if TYPE_CHECKING:
    import torch

    from .config import PerformanceConfig

# A failed CUDA synchronization cannot prove that borrowed input storage is
# idle. Retain its owners until process exit and reject reuse of that device.
_FAILED_OWNERS: list[tuple[int, list[Any]]] = []


@dataclass(frozen=True)
class FitsPlane:
    """One explicit, zero-based image HDU in a local FITS file."""

    path: str | Path
    hdu: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.hdu, bool)
            or not isinstance(self.hdu, (int, np.integer))
            or self.hdu < 0
        ):
            raise ValueError("FITS plane HDU must be a nonnegative integer")
        object.__setattr__(
            self, "path", Path(self.path).expanduser().resolve()
        )
        object.__setattr__(self, "hdu", int(self.hdu))


@dataclass(frozen=True)
class FitsInferenceResult:
    """Ordered predictions copied to the host after inference completes."""

    candidate_ids: tuple[int | str, ...]
    logits: np.ndarray
    probabilities: np.ndarray
    fits_reads: tuple[dict[str, Any], ...]


def _plan_inputs(planes, centers_yx, candidate_ids, stamp_shape):
    if not all(isinstance(plane, FitsPlane) for plane in planes):
        raise TypeError("search, template and difference must be FitsPlane")
    shape = tuple(stamp_shape)
    if len(shape) != 2 or any(
        isinstance(size, bool)
        or not isinstance(size, (int, np.integer))
        or size <= 0
        or size % 2 != 1
        for size in shape
    ):
        raise ValueError("stamp_shape must contain two positive odd integers")
    shape = tuple(int(size) for size in shape)
    ids = tuple(candidate_ids)
    if not ids or any(
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer, str))
        or (isinstance(value, str) and not value)
        for value in ids
    ):
        raise ValueError("candidate_ids must be nonempty integers or strings")
    ids = tuple(
        int(value) if isinstance(value, np.integer) else value
        for value in ids
    )
    if len({(type(value), value) for value in ids}) != len(ids):
        raise ValueError("candidate_ids must be unique")
    centers = tuple(tuple(center) for center in centers_yx)
    if len(centers) != len(ids):
        raise ValueError(
            "candidate_ids and centers_yx must have equal length"
        )
    if any(
        len(center) != 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            for value in center
        )
        for center in centers
    ):
        raise ValueError("centers_yx must contain integer (y, x) pairs")
    infos = tuple(
        inspect_fits_image(plane.path, plane.hdu) for plane in planes
    )
    image_shape = infos[0].shape
    if any(info.shape != image_shape for info in infos):
        raise ValueError("FITS planes must be aligned with matching shapes")
    if any(info.dtype.kind not in "iuf" for info in infos):
        raise ValueError("FITS image planes must be real numeric arrays")
    half_y, half_x = (size // 2 for size in shape)
    slices = []
    for center_y, center_x in centers:
        y0, x0 = int(center_y) - half_y, int(center_x) - half_x
        y1, x1 = y0 + shape[0], x0 + shape[1]
        if y0 < 0 or x0 < 0 or y1 > image_shape[0] or x1 > image_shape[1]:
            raise ValueError("candidate stamp exceeds image bounds")
        slices.append((slice(y0, y1), slice(x0, x1)))
    return ids, shape, tuple(slices)


def _read_planes(planes, *, reader, stream):
    groups: dict[Path, list[int]] = {}
    for plane in planes:
        groups.setdefault(plane.path, []).append(plane.hdu)
    arrays = {}
    receipts = []
    for path, hdus in groups.items():
        selected = tuple(dict.fromkeys(hdus))
        result = read_fits_images(
            path, selected, reader=reader, device=True, stream=stream
        )
        arrays.update(
            ((path, hdu), array)
            for hdu, array in zip(selected, result.arrays, strict=True)
        )
        receipts.append({"path": str(path), **result.metadata()})
    return tuple(arrays[(plane.path, plane.hdu)] for plane in planes), tuple(
        receipts
    )


def _crop_planes(cp, planes, slices, stamp_shape, owners):
    stamps = cp.empty(
        (len(slices), len(planes), *stamp_shape), dtype=cp.float32
    )
    owners.append(stamps)
    for row, section in enumerate(slices):
        for channel, plane in enumerate(planes):
            stamps[row, channel] = plane[section]
    if not bool(cp.isfinite(stamps).all().item()):
        raise ValueError("candidate stamps must contain only finite values")
    return stamps


def _complete_failed_work(streams, owners, device_id, error):
    failed = False
    for stream in streams:
        try:
            stream.synchronize()
        except BaseException as cleanup_error:
            failed = True
            error.add_note(
                "FITS inference cleanup could not confirm CUDA completion: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
    if failed:
        _FAILED_OWNERS.append((device_id, owners))


def predict_fits(
    *,
    model: torch.nn.Module,
    search: FitsPlane,
    template: FitsPlane,
    centers_yx: Sequence[tuple[int, int]],
    candidate_ids: Sequence[int | str],
    stamp_shape: tuple[int, int],
    device: torch.device,
    performance: PerformanceConfig,
    difference: FitsPlane | None = None,
    xfit_features: torch.Tensor | None = None,
    fits_reader: str = "auto",
) -> FitsInferenceResult:
    """Crop aligned FITS planes on one GPU and run the existing tensor model.

    Channels are search, template and optional difference. Centers are integer
    zero-based ``(y, x)`` coordinates, and prediction order follows
    ``candidate_ids``. This function does not align images, compute a
    difference, interpret masks or normalize pixels. Nonfinite values are
    rejected within selected stamps; pixels outside them are not classified.

    ``model`` must already be loaded on an explicit CUDA device with its
    desired runtime/compilation policy. Optional float32 Torch
    ``xfit_features`` must have one row per candidate on that same device and
    be ready on its current Torch stream. The performance argument is passed
    unchanged to ``predict_tensors``. FITS decoding uses ``auto``, ``astropy``
    or strict ``xdr``; each read reports its actual implementation.

    The function blocks until inference and compact output transfer complete.
    CuPy arrays and DLPack owners remain alive through that boundary. If CUDA
    completion fails, retained owners protect unfinished work and this device
    requires process restart before another call.
    """
    import torch

    from . import training
    from .config import PerformanceConfig

    planes = (search, template) + (
        () if difference is None else (difference,)
    )
    ids, shape, slices = _plan_inputs(
        planes, centers_yx, candidate_ids, stamp_shape
    )
    if fits_reader not in {"auto", "astropy", "xdr"}:
        raise ValueError("fits_reader must be auto, astropy, or xdr")
    if not isinstance(performance, PerformanceConfig):
        raise TypeError("performance must be a PerformanceConfig")
    device = training._require_explicit_cuda_device(device)
    if any(index == device.index for index, _ in _FAILED_OWNERS):
        raise RuntimeError(
            "FITS inference CUDA completion failed; restart this process"
        )
    training._validate_model_device(model, device=device)
    if xfit_features is not None:
        training._validate_cuda_tensor(
            xfit_features, name="xfit_features", device=device, ndim=2
        )
        if xfit_features.shape[0] != len(ids):
            raise ValueError("xfit_features must have one row per candidate")
    cp = training._load_cupy_for_dlpack()
    owners: list[Any] = [model, xfit_features]
    with cp.cuda.Device(device.index), torch.cuda.device(device):
        producer = cp.cuda.Stream(non_blocking=True)
        consumer = torch.cuda.current_stream(device)
        owners.extend((producer, consumer))
        try:
            with producer, torch.cuda.stream(consumer):
                if xfit_features is not None and not bool(
                    torch.isfinite(xfit_features).all().item()
                ):
                    raise ValueError(
                        "xfit_features must contain only finite values"
                    )
                arrays, receipts = _read_planes(
                    planes, reader=fits_reader, stream=producer
                )
                owners.extend(arrays)
                stamps = _crop_planes(cp, arrays, slices, shape, owners)
                image_view = training.cupy_to_torch(stamps, device=device)
                owners.append(image_view)
                prediction = training.predict_tensors(
                    model=model,
                    images=image_view.tensor,
                    device=device,
                    xfit_features=xfit_features,
                    performance=performance,
                )
                owners.append(prediction)
                packed = torch.stack(
                    (prediction["logits"], prediction["probabilities"])
                )
                owners.append(packed)
                host = packed.detach().cpu().numpy()
        except BaseException as error:
            _complete_failed_work(
                (producer, consumer), owners, device.index, error
            )
            raise
    if not np.isfinite(host).all():
        raise ValueError(
            "FITS inference predictions contain nonfinite values"
        )
    return FitsInferenceResult(
        ids,
        np.array(host[0], copy=True),
        np.array(host[1], copy=True),
        receipts,
    )


__all__ = ["FitsPlane", "FitsInferenceResult", "predict_fits"]
