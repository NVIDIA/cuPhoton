# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Explicit aligned FITS planes and candidate stamps for standalone xFit."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cuphoton.core.artifacts import file_sha256
from cuphoton.core.fits_io import inspect_fits_image, read_fits_images

from ._types import FIT_MODES


@dataclass(frozen=True)
class FitsInputPlan:
    path: Path
    sha256: str
    mode: str
    stamp_shape: tuple[int, int]
    image_shape: tuple[int, int]
    dtype: np.dtype
    candidate_id: np.ndarray
    centers: tuple[tuple[int, int], ...]
    planes: tuple[dict[str, Any], ...]
    sources: tuple[dict[str, Any], ...]
    initial: np.ndarray | None
    stamp_basis: dict[str, Any] | None

    @property
    def batch_size(self) -> int:
        return len(self.centers)

    @property
    def images_shape(self) -> tuple[int, ...]:
        channels = (3,) if self.mode == "split" else ()
        return (self.batch_size, *channels, *self.stamp_shape)


def _mapping(value, allowed, *, required=(), name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    if set(value) - set(allowed) or set(required) - set(value):
        raise ValueError(f"{name} has missing or unsupported fields")


def _integer(value, name, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _plane(value, root, *, name, auxiliary=True):
    keys = {"path", "hdu"}
    if auxiliary:
        keys |= {"mask_hdu", "variance_hdu", "bad_mask_bits"}
    _mapping(value, keys, required=("path", "hdu"), name=name)
    if not isinstance(value["path"], str) or not value["path"]:
        raise ValueError(f"{name}.path must be a nonempty string")
    path = Path(value["path"]).expanduser()
    path = (root / path).resolve()
    result = {**value, "path": str(path)}
    image = inspect_fits_image(path, _integer(value["hdu"], "hdu"))
    for key in ("mask_hdu", "variance_hdu"):
        if key not in value:
            continue
        info = inspect_fits_image(path, _integer(value[key], key))
        if info.shape != image.shape:
            raise ValueError(f"{name} auxiliary HDUs must match image shape")
        if key == "mask_hdu" and info.dtype.kind not in "iu":
            raise ValueError("FITS masks must contain integer bit fields")
    if "bad_mask_bits" in value:
        if "mask_hdu" not in value:
            raise ValueError("bad_mask_bits requires mask_hdu")
        _integer(value["bad_mask_bits"], "bad_mask_bits")
        mask_info = inspect_fits_image(path, value["mask_hdu"])
        if value["bad_mask_bits"] >= 1 << (8 * mask_info.dtype.itemsize):
            raise ValueError("bad_mask_bits exceeds the mask dtype width")
    return result, image


def plan_fits_input(path, *, mode=None, model=None) -> FitsInputPlan:
    """Validate headers/candidates and hash sources without pixel decode."""
    from .io import PARAMETER_COUNTS, _validate_candidate_id

    path = Path(path).expanduser().resolve()
    manifest_hash = file_sha256(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    _mapping(
        data,
        {
            "schema",
            "mode",
            "stamp_shape",
            "images",
            "candidates",
            "initial",
            "stamp_basis",
        },
        required=("schema", "mode", "stamp_shape", "images", "candidates"),
        name="FITS manifest",
    )
    if data["schema"] != "cuphoton.xfit.fits-input/v1":
        raise ValueError("unsupported xFit FITS manifest schema")
    if data["mode"] not in FIT_MODES or mode not in (None, data["mode"]):
        raise ValueError("FITS manifest mode differs from requested mode")
    shape = data["stamp_shape"]
    if not isinstance(shape, list) or len(shape) != 2:
        raise ValueError("stamp_shape must contain height and width")
    for value in shape:
        if _integer(value, "stamp dimension", minimum=1) % 2 != 1:
            raise ValueError("stamp dimensions must be odd")
    images = data["images"]
    count = 3 if data["mode"] == "split" else 1
    if not isinstance(images, list) or len(images) != count:
        raise ValueError(
            f"{data['mode']} mode requires {count} FITS image planes"
        )
    planes, infos = zip(
        *(_plane(value, path.parent, name="image") for value in images),
        strict=True,
    )
    if len({info.shape for info in infos}) != 1:
        raise ValueError(
            "FITS planes must be aligned with matching dimensions"
        )
    variance_present = ["variance_hdu" in plane for plane in planes]
    if any(variance_present) and not all(variance_present):
        raise ValueError(
            "variance_hdu is required for every split plane or none"
        )
    candidates = data["candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidates must be a nonempty list")
    ids, centers = [], []
    for row in candidates:
        _mapping(
            row,
            {"candidate_id", "x", "y"},
            required=("candidate_id", "x", "y"),
            name="candidate",
        )
        identifier = row["candidate_id"]
        if type(identifier) not in (int, str):
            raise ValueError("candidate_id must be an integer or string")
        ids.append(identifier)
        x, y = (_integer(row[key], key) for key in ("x", "y"))
        h, w = shape
        if not (
            w // 2 <= x < infos[0].shape[1] - w // 2
            and h // 2 <= y < infos[0].shape[0] - h // 2
        ):
            raise ValueError("candidate stamp extends outside a FITS image")
        centers.append((x, y))
    if len({type(value) for value in ids}) != 1:
        raise ValueError("candidate IDs must be all integers or all strings")
    if type(ids[0]) is int and any(
        not -(1 << 63) <= value < (1 << 63) for value in ids
    ):
        raise ValueError("integer candidate_id values must fit signed 64-bit")
    candidate_id = np.asarray(ids)
    _validate_candidate_id(candidate_id, len(centers))
    initial = None
    if "initial" in data:
        initial = np.asarray(data["initial"])
        if (
            initial.dtype.kind not in "fiu"
            or not np.isfinite(initial).all()
            or initial.ndim != 2
            or initial.shape[0] != len(ids)
        ):
            raise ValueError(
                "initial must contain finite per-candidate parameters"
            )
        if (
            model in PARAMETER_COUNTS
            and initial.shape[1] != PARAMETER_COUNTS[model]
        ):
            raise ValueError(
                f"initial must contain {PARAMETER_COUNTS[model]} "
                f"parameters for {model} model"
            )
    basis = None
    if "stamp_basis" in data:
        basis, _ = _plane(
            data["stamp_basis"],
            path.parent,
            name="stamp_basis",
            auxiliary=False,
        )
    if model == "stamp" and basis is None:
        raise ValueError("stamp model requires a stamp_basis array")
    if model == "gaussian" and basis is not None:
        raise ValueError("stamp_basis is not used by the gaussian model")
    sources = []
    paths = {Path(plane["path"]) for plane in planes}
    if basis is not None:
        paths.add(Path(basis["path"]))
    for source in sorted(paths):
        before = source.stat()
        digest = file_sha256(source)
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError("FITS input changed while planning")
        sources.append(
            {
                "path": str(source),
                "sha256": digest,
                "size": after.st_size,
                "mtime_ns": after.st_mtime_ns,
            }
        )
    if file_sha256(path) != manifest_hash:
        raise ValueError("FITS manifest changed while planning")
    return FitsInputPlan(
        path,
        manifest_hash,
        data["mode"],
        tuple(shape),
        infos[0].shape,
        np.result_type(*(info.dtype for info in infos)),
        candidate_id,
        tuple(centers),
        tuple(planes),
        tuple(sources),
        initial,
        basis,
    )


def check_fits_sources(sources, *, hashes=False):
    """Check the frozen referenced files as well as the manifest itself."""
    for source in sources:
        path = Path(source["path"])
        stat = path.stat()
        if (
            stat.st_size != source["size"]
            or stat.st_mtime_ns != source["mtime_ns"]
            or (hashes and file_sha256(path) != source["sha256"])
        ):
            raise ValueError("xFit referenced FITS input changed")


def _candidate_stamps(result, keys, plan, *, offset, ap):
    """Retain decoded owners until copies finish, including on errors."""
    h, w = plan.stamp_shape
    ox, oy = offset
    arrays = {}

    def copy_stamps():
        for key, array in zip(keys, result.arrays, strict=True):
            arrays[key] = ap.stack(
                [
                    array[
                        y - h // 2 - oy : y + h // 2 + 1 - oy,
                        x - w // 2 - ox : x + w // 2 + 1 - ox,
                    ]
                    for x, y in plan.centers
                ]
            )

    if ap is np:
        copy_stamps()
    else:
        from cuphoton.xdr.prefetch import (
            _gpu_submission_guard,
            _GpuBatchHandle,
            _synchronize_gpu_batch,
        )

        with _gpu_submission_guard():
            handle = _GpuBatchHandle(
                event=None,
                keepalive=[result, arrays],
                stream=ap.cuda.get_current_stream(),
                device_id=ap.cuda.runtime.getDevice(),
            )
            try:
                copy_stamps()
            finally:
                _synchronize_gpu_batch(handle, synchronize_stream=True)
    return arrays


def load_fits_input(
    path, *, mode=None, model=None, reader="auto", device=False
):
    """Read the candidate bounding region and keep GPU stamps on device."""
    from .io import XFitDataset, _validate_dataset_contract

    plan = plan_fits_input(path, mode=mode, model=model)
    ap = np
    if device:
        import cupy as ap
    h, w = plan.stamp_shape
    x0 = min(x for x, _ in plan.centers) - w // 2
    y0 = min(y for _, y in plan.centers) - h // 2
    x1 = max(x for x, _ in plan.centers) + w // 2 + 1
    y1 = max(y for _, y in plan.centers) + h // 2 + 1
    section = (slice(y0, y1), slice(x0, x1))
    images, masks, variances, metadata = [], [], [], []
    roles = ("difference", "positive", "negative")
    for plane_index, plane in enumerate(plan.planes):
        keys = [
            key for key in ("hdu", "mask_hdu", "variance_hdu") if key in plane
        ]
        hdus = [plane[key] for key in keys]
        # xDR currently supports sections for compressed images. An explicit
        # xDR request may read an uncompressed plane in full before cropping.
        full = section == (
            slice(0, plan.image_shape[0]),
            slice(0, plan.image_shape[1]),
        ) or (
            reader == "xdr"
            and any(
                inspect_fits_image(plane["path"], index).compression is None
                for index in hdus
            )
        )
        result = read_fits_images(
            plane["path"],
            hdus,
            reader=reader,
            device=device,
            section=None if full else section,
        )
        arrays = _candidate_stamps(
            result,
            keys,
            plan,
            offset=(0, 0) if full else (x0, y0),
            ap=ap,
        )
        images.append(arrays["hdu"])
        if "mask_hdu" in arrays:
            mask = arrays["mask_hdu"]
            if "bad_mask_bits" in plane:
                mask = mask.astype(ap.uint64) & np.uint64(
                    plane["bad_mask_bits"]
                )
            masks.append(mask == 0)
        else:
            masks.append(ap.ones(arrays["hdu"].shape, dtype=bool))
        if "variance_hdu" in arrays:
            variances.append(arrays["variance_hdu"])
        metadata.append(
            {
                **result.metadata(),
                "path": plane["path"],
                "role": roles[plane_index],
                "section": None if full else [y0, y1, x0, x1],
            }
        )

    def combine(values):
        return ap.stack(values, axis=1) if plan.mode == "split" else values[0]

    basis = None
    if plan.stamp_basis is not None:
        result = read_fits_images(
            plan.stamp_basis["path"],
            [plan.stamp_basis["hdu"]],
            reader=reader,
            device=False,
        )
        basis = result.arrays[0]
        metadata.append(
            {
                **result.metadata(),
                "path": plan.stamp_basis["path"],
                "role": "stamp_basis",
            }
        )
    dataset = XFitDataset(
        path=plan.path,
        candidate_id=plan.candidate_id,
        images=combine(images),
        input_archive_sha256=plan.sha256,
        initial=plan.initial,
        mask=combine(masks)
        if any("mask_hdu" in plane for plane in plan.planes)
        else None,
        variance=combine(variances) if variances else None,
        stamp_basis=basis,
        input_sources=plan.sources,
        reader_metadata=tuple(metadata),
    )
    _validate_dataset_contract(dataset, mode=mode, model=model)
    check_fits_sources(plan.sources, hashes=True)
    if file_sha256(plan.path) != plan.sha256:
        raise ValueError("FITS manifest changed while loading")
    return dataset
