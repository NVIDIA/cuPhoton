# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Rubin-to-xRep masked-plane bridge.

The xRep dependency is imported only when a warp is requested. This keeps
base xImage imports lightweight while failing explicitly at the integration
seam.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from importlib import import_module
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np

from .image import (
    RubinImageBounds,
    RubinImageError,
    RubinImageFormatError,
    RubinImageView,
)


class RubinWarpDependencyError(RubinImageError):
    """Raised when cuPhoton lacks the required xRep API."""


@dataclass(frozen=True, slots=True)
class RubinWarpPlanes:
    """Bounded Rubin planes and source metadata prepared for xRep.

    ``requirements_met`` preserves the source reader's metadata-presence
    result. It does not certify the scientific quality of the reprojection.
    """

    image: np.ndarray
    variance: np.ndarray
    mask: np.ndarray
    source_path: Path
    source_format: str
    bounds: RubinImageBounds
    wcs: Any
    wcs_fidelity: str
    unit: str | None
    variance_unit: str | None
    mask_plane_names: tuple[str, ...]
    selected_bad_mask_plane_names: tuple[str, ...]
    mask_plane_schema: dict[str, dict[str, int]]
    mask_representation: str
    requirements_met: bool
    product_role: str
    reconstruction_ready: bool
    cell_key: dict[str, Any] | None
    cell_diagnostics: dict[str, Any] | None
    cell_metadata: dict[str, Any] | None
    psf: Any | None
    aperture_corrections: dict[str, Any]
    provenance: Any | None
    archive_info: dict[str, Any] | None
    archive_metadata: dict[str, Any] | None

    @property
    def source_origin_xy(self) -> tuple[int, int]:
        """Return the absolute ``(x, y)`` origin of the bounded planes.

        Use this as ``ReprojectionSpec.source_origin`` only when the mapping
        returns absolute pixels from the full-product geometry WCS.
        ``self.wcs`` is already sliced into the bounded array's local pixel
        frame.
        """

        return (self.bounds.x_start, self.bounds.y_start)

    @property
    def local_wcs_source_origin_xy(self) -> tuple[int, int]:
        """Return the xRep origin for a mapping built from ``self.wcs``."""

        return (0, 0)

    def metadata(self) -> dict[str, Any]:
        """Return structured context without paths or free-form metadata."""

        return {
            "source_format": self.source_format,
            "absolute_bounds": self.bounds.to_dict(),
            "absolute_source_origin_xy": list(self.source_origin_xy),
            "local_wcs_source_origin_xy": list(
                self.local_wcs_source_origin_xy
            ),
            "wcs_fidelity": self.wcs_fidelity,
            "unit": self.unit,
            "variance_unit": self.variance_unit,
            "mask_plane_names": list(self.mask_plane_names),
            "selected_bad_mask_plane_names": list(
                self.selected_bad_mask_plane_names
            ),
            "mask_plane_schema": deepcopy(self.mask_plane_schema),
            "mask_representation": self.mask_representation,
            "requirements_met": self.requirements_met,
            "product_role": self.product_role,
            "reconstruction_ready": self.reconstruction_ready,
            "cell_key": deepcopy(self.cell_key),
            "cell_diagnostics": deepcopy(self.cell_diagnostics),
            "cell_metadata": deepcopy(self.cell_metadata),
            "psf_available": self.psf is not None,
            "aperture_corrections_available": bool(self.aperture_corrections),
            "aperture_correction_names": sorted(self.aperture_corrections),
            "provenance_available": self.provenance is not None,
            "archive_info": deepcopy(self.archive_info),
        }


@dataclass(frozen=True, slots=True)
class RubinMaskedWarpResult:
    """xRep result with its Rubin source context."""

    xrep_result: Any
    source: RubinWarpPlanes

    @property
    def image(self) -> np.ndarray:
        return self.xrep_result.image

    @property
    def variance(self) -> np.ndarray:
        return self.xrep_result.variance

    @property
    def mask(self) -> np.ndarray:
        return self.xrep_result.mask

    @property
    def bbox(self) -> Any:
        return self.xrep_result.bbox

    @property
    def backend(self) -> str:
        return self.xrep_result.backend

    @property
    def interpolation(self) -> str:
        return self.xrep_result.interpolation

    @property
    def metadata(self) -> dict[str, Any]:
        metadata = dict(self.xrep_result.metadata)
        metadata["rubin_image"] = self.source.metadata()
        return metadata


@dataclass(frozen=True, slots=True)
class _XRepAPI:
    masked_reprojection_result_cls: type
    reproject_masked_array: Any


def rubin_view_to_warp_planes(
    view: RubinImageView,
    *,
    diagnostic_position_xy: tuple[float, float] | None = None,
) -> RubinWarpPlanes:
    """Prepare one normalized Rubin view for the generic xRep API.

    The view's WCS is retained by identity.  In particular, a full CellCoadd
    archive is never downgraded to an IMAGE-header approximation.
    ``diagnostic_position_xy`` is an absolute coadd pixel used to bind the
    retained cell metadata; it defaults to the center of the bounded view.
    """

    image = np.asarray(view.image)
    if image.ndim != 2:
        raise RubinImageFormatError(
            f"Rubin image plane must be two-dimensional, got {image.shape}."
        )
    if view.variance is None:
        raise RubinImageFormatError(
            "Rubin image view has no variance plane for masked-plane warping."
        )
    variance = np.asarray(view.variance)
    if variance.ndim != 2 or variance.shape != image.shape:
        raise RubinImageFormatError(
            "Rubin variance plane must be two-dimensional and match the "
            f"image shape, got {variance.shape} and {image.shape}."
        )
    if view.wcs is None:
        raise RubinImageFormatError(
            "Rubin image view has no authoritative WCS for xRep."
        )
    if (
        view.source_format == "lsst-images-archive"
        and view.wcs_fidelity != "exact-lsst-images"
    ):
        raise RubinImageFormatError(
            "CellCoadd xRep input requires the exact native lsst.images WCS."
        )

    mask = _pack_native_mask(view)
    if mask.shape != image.shape:
        raise RubinImageFormatError(
            "Rubin mask plane does not match the bounded image shape."
        )

    x, y = diagnostic_position_xy or (
        (view.bounds.x_start + view.bounds.x_stop - 1) / 2.0,
        (view.bounds.y_start + view.bounds.y_stop - 1) / 2.0,
    )
    diagnostics = view.cell_diagnostics_at(x=float(x), y=float(y))
    cell_key = _cell_key(view, diagnostics)

    return RubinWarpPlanes(
        image=image,
        variance=variance,
        mask=mask,
        source_path=view.source_path,
        source_format=view.source_format,
        bounds=view.bounds,
        wcs=view.wcs,
        wcs_fidelity=view.wcs_fidelity,
        unit=view.unit,
        variance_unit=view.variance_unit,
        mask_plane_names=view.mask_plane_names,
        selected_bad_mask_plane_names=view.selected_bad_mask_plane_names,
        mask_plane_schema=deepcopy(view.mask_plane_schema),
        mask_representation="packed-global-bit-index",
        requirements_met=bool(view.requirements_met),
        product_role=view.product_role,
        reconstruction_ready=bool(view.reconstruction_ready),
        cell_key=cell_key,
        cell_diagnostics=deepcopy(diagnostics),
        cell_metadata=deepcopy(view.cell_metadata),
        psf=view.psf,
        aperture_corrections=dict(view.aperture_corrections or {}),
        provenance=view.provenance,
        archive_info=deepcopy(view.archive_info),
        archive_metadata=deepcopy(view.archive_metadata),
    )


def warp_rubin_image_view(
    view: RubinImageView,
    spec: Any,
    *,
    backend: str | None = None,
    prepared: Any | None = None,
    diagnostic_position_xy: tuple[float, float] | None = None,
) -> RubinMaskedWarpResult:
    """Warp a normalized Rubin image through the xRep masked-plane API."""

    planes = rubin_view_to_warp_planes(
        view,
        diagnostic_position_xy=diagnostic_position_xy,
    )
    api = _load_xrep_api()
    result = api.reproject_masked_array(
        planes.image,
        planes.variance,
        planes.mask,
        spec,
        backend=backend,
        prepared=prepared,
    )
    if not isinstance(result, api.masked_reprojection_result_cls):
        raise RubinWarpDependencyError(
            "cuphoton.xrep.reproject_masked_array returned an unexpected "
            f"result type: {type(result).__name__}."
        )
    return RubinMaskedWarpResult(xrep_result=result, source=planes)


def _pack_native_mask(view: RubinImageView) -> np.ndarray:
    if view.source_format == "legacy-fits":
        raw = np.asarray(view.native_mask)
        if raw.ndim != 2:
            raise RubinImageFormatError(
                "Legacy Rubin native mask must be two-dimensional."
            )
        if raw.dtype.kind == "b":
            _validate_mask_plane_schema(view, capacity_bits=1)
            return raw
        if raw.dtype.kind not in {"i", "u"}:
            raise RubinImageFormatError(
                "Legacy Rubin native mask must have an integer dtype."
            )
        capacity_bits = raw.dtype.itemsize * 8
        _validate_mask_plane_schema(view, capacity_bits=capacity_bits)
        # FITS signed integers retain mask bits, including the sign bit.
        return raw.astype(_unsigned_dtype(capacity_bits), copy=False)

    native_array = getattr(view.native_mask, "array", None)
    if native_array is None:
        raise RubinImageFormatError(
            "lsst.images Rubin view has no native mask word array."
        )
    words = np.asarray(native_array)
    if words.ndim != 3 or words.dtype.kind != "u":
        raise RubinImageFormatError(
            "lsst.images native mask must use unsigned word planes."
        )
    if words.shape[:2] != view.image.shape:
        raise RubinImageFormatError(
            "lsst.images native mask words do not match the image shape."
        )
    bits_per_word = int(np.iinfo(words.dtype).bits)
    total_bits = int(words.shape[-1]) * bits_per_word
    if total_bits > 64:
        raise RubinImageFormatError(
            "lsst.images native mask needs "
            f"{total_bits} bits; xRep supports exact packed masks up to "
            "64 bits."
        )
    _validate_mask_plane_schema(
        view,
        capacity_bits=total_bits,
        bits_per_word=bits_per_word,
        word_count=int(words.shape[-1]),
    )
    dtype = _unsigned_dtype(max(total_bits, 1))
    packed = np.zeros(words.shape[:2], dtype=dtype)
    for word_index in range(words.shape[-1]):
        shift = word_index * bits_per_word
        packed |= words[..., word_index].astype(dtype) << shift
    return packed


def _validate_mask_plane_schema(
    view: RubinImageView,
    *,
    capacity_bits: int,
    bits_per_word: int | None = None,
    word_count: int | None = None,
) -> None:
    """Fail closed unless the declared schema fits the packed native mask."""

    if capacity_bits < 1 or capacity_bits > 64:
        raise RubinImageFormatError(
            "Rubin mask schema capacity must be between 1 and 64 bits."
        )
    schema = view.mask_plane_schema
    if not isinstance(schema, dict):
        raise RubinImageFormatError("Rubin mask schema must be a dictionary.")
    names = tuple(view.mask_plane_names)
    if set(schema) != set(names):
        raise RubinImageFormatError(
            "Rubin mask schema names do not match the declared mask planes."
        )

    used_indices: set[int] = set()
    for name in names:
        entry = schema[name]
        if not isinstance(entry, dict):
            raise RubinImageFormatError(
                "Rubin mask schema entries must be dictionaries."
            )
        if view.source_format == "legacy-fits":
            global_index = _schema_integer(entry, "bit")
        else:
            global_index = _schema_integer(entry, "index")
            mask = _schema_integer(entry, "mask")
            word = _schema_integer(entry, "word")
            word_mask = _schema_integer(entry, "word_mask")
            local_bit = _schema_integer(entry, "bit")
            if bits_per_word is None or word_count is None:
                raise RubinImageFormatError(
                    "Rubin mask schema lacks native word geometry."
                )
            if (
                word < 0
                or word >= word_count
                or local_bit < 0
                or local_bit >= bits_per_word
                or global_index != word * bits_per_word + local_bit
                or mask != 1 << global_index
                or word_mask != 1 << local_bit
            ):
                raise RubinImageFormatError(
                    "Rubin mask schema is inconsistent with native word "
                    "geometry."
                )
        if global_index < 0 or global_index >= capacity_bits:
            raise RubinImageFormatError(
                "Rubin mask schema bit is outside packed native capacity."
            )
        if global_index in used_indices:
            raise RubinImageFormatError(
                "Rubin mask schema contains duplicate packed bit indices."
            )
        used_indices.add(global_index)


def _schema_integer(entry: dict[str, Any], key: str) -> int:
    value = entry.get(key)
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise RubinImageFormatError(
            "Rubin mask schema bit fields must be integers."
        )
    return int(value)


def _unsigned_dtype(bits: int) -> np.dtype:
    if bits <= 8:
        return np.dtype(np.uint8)
    if bits <= 16:
        return np.dtype(np.uint16)
    if bits <= 32:
        return np.dtype(np.uint32)
    if bits <= 64:
        return np.dtype(np.uint64)
    raise RubinImageFormatError(
        f"xRep supports exact packed Rubin masks up to 64 bits, got {bits}."
    )


def _cell_key(
    view: RubinImageView,
    diagnostics: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if diagnostics is None:
        return None
    metadata = view.cell_metadata or {}
    return {
        "skymap": metadata.get("skymap"),
        "tract": metadata.get("tract"),
        "patch": deepcopy(metadata.get("patch")),
        "band": metadata.get("band"),
        "cell_index_ij": deepcopy(diagnostics.get("cell_index_ij")),
    }


def _load_xrep_api() -> _XRepAPI:
    try:
        xrep = import_module("cuphoton.xrep")
    except (ImportError, ModuleNotFoundError) as exc:
        raise RubinWarpDependencyError(
            "Rubin masked-plane warping requires cuphoton.xrep."
        ) from exc

    result_cls = getattr(xrep, "MaskedReprojectionResult", None)
    reproject_masked_array = getattr(xrep, "reproject_masked_array", None)
    if result_cls is None or not callable(reproject_masked_array):
        raise RubinWarpDependencyError(
            "The installed cuPhoton does not provide "
            "MaskedReprojectionResult and reproject_masked_array."
        )
    return _XRepAPI(
        masked_reprojection_result_cls=result_cls,
        reproject_masked_array=reproject_masked_array,
    )


__all__ = [
    "RubinMaskedWarpResult",
    "RubinWarpDependencyError",
    "RubinWarpPlanes",
    "rubin_view_to_warp_planes",
    "warp_rubin_image_view",
]
