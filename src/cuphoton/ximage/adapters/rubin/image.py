# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Normalized Rubin image access across FITS and CellCoadd archives.

The ``lsst.images`` dependency is optional and imported only when a CellCoadd
backend is selected.  Callers must name the source format explicitly; this
module never tries another parser after a backend fails.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

RUBIN_IMAGE_FORMATS = frozenset(
    {
        "legacy-fits",
        "lsst-images-archive",
        "lsst-images-cutdown",
    }
)
LSST_IMAGES_VERSION = "30.0.10"

_RUBIN_BAD_MASK_PLANES = (
    "BAD",
    "SAT",
    "INTRP",
    "CR",
    "EDGE",
    "NO_DATA",
    "SUSPECT",
    "UNMASKEDNAN",
    "VIGNETTED",
    "STREAK",
    "CROSSTALK",
    "CLIPPED",
    "REJECTED",
    "SENSOR_EDGE",
    "SAT_TEMPLATE",
    "INJECTED",
    "INJECTED_TEMPLATE",
    "ITL_DIP",
)
_LSST_BAD_MASK_PLANES = (
    "BAD",
    "SATURATED",
    "INTERPOLATED",
    "COSMIC_RAY",
    "DETECTION_EDGE",
    "NO_DATA",
    "SUSPECT",
    "UNMASKED_NAN",
    "VIGNETTED",
    "STREAK",
    "CROSSTALK",
    "CLIPPED",
    "REJECTED",
    "SENSOR_EDGE",
    "SAT_TEMPLATE",
    "INJECTED",
    "INJECTED_TEMPLATE",
    "ITL_DIP",
)
_MASK_POLICIES = {"rubin-bad-planes", "nonzero-bad", "none-good"}
_EXPOSURE_SUMMARY_KEYS = (
    "expTime",
    "meanVar",
    "nPsfStar",
    "pixelScale",
    "psfArea",
    "psfIxx",
    "psfIxy",
    "psfIyy",
    "psfSigma",
    "skyBg",
    "skyNoise",
    "zeroPoint",
)


class RubinImageError(RuntimeError):
    """Base error for normalized Rubin image access."""


class RubinImageFormatError(RubinImageError):
    """Raised when the selected backend does not match the input."""


class RubinImageDependencyError(RubinImageError):
    """Raised when the optional Rubin archive dependency is unavailable."""


@dataclass(frozen=True)
class RubinImageBounds:
    """Half-open absolute pixel bounds in ``(y, x)`` order."""

    y_start: int
    y_stop: int
    x_start: int
    x_stop: int

    @property
    def shape(self) -> tuple[int, int]:
        return (
            self.y_stop - self.y_start,
            self.x_stop - self.x_start,
        )

    def to_dict(self) -> dict[str, int | list[int]]:
        return {
            "y_start": self.y_start,
            "y_stop": self.y_stop,
            "x_start": self.x_start,
            "x_stop": self.x_stop,
            "shape_yx": list(self.shape),
        }


def local_pixel_is_inside(
    bounds: Any,
    *,
    x: float,
    y: float,
) -> bool:
    """Return whether a local WCS pixel center is inside image bounds."""

    width = int(bounds.x_stop) - int(bounds.x_start)
    height = int(bounds.y_stop) - int(bounds.y_start)
    return bool(
        math.isfinite(x)
        and math.isfinite(y)
        and -0.5 <= x < width - 0.5
        and -0.5 <= y < height - 0.5
    )


def local_pixel_is_addressable(
    bounds: Any,
    *,
    x: float,
    y: float,
) -> bool:
    """Return whether local WCS pixels map to an addressable array element."""

    width = int(bounds.x_stop) - int(bounds.x_start)
    height = int(bounds.y_stop) - int(bounds.y_start)
    return bool(
        math.isfinite(x)
        and math.isfinite(y)
        and 0.0 <= x < width
        and 0.0 <= y < height
    )


def local_to_absolute_pixel(
    bounds: Any,
    *,
    x: float,
    y: float,
) -> tuple[float, float]:
    """Translate local WCS pixels to the absolute bbox/cell frame."""

    return (
        float(x) + int(bounds.x_start),
        float(y) + int(bounds.y_start),
    )


@dataclass(frozen=True)
class RubinImageGeometry:
    """Metadata-only image bounds and authoritative pixel-to-sky mapping."""

    source_path: Path
    source_format: str
    bounds: RubinImageBounds
    wcs: Any | None
    wcs_fidelity: str
    archive_info: dict[str, Any] | None = None

    def pixel_to_skycoord(self, x: Any, y: Any) -> SkyCoord:
        """Convert local array pixels with the WCS high-level API."""

        return _pixel_to_skycoord(self.wcs, x=x, y=y)

    def skycoord_to_pixel(self, sky: SkyCoord) -> tuple[Any, Any]:
        """Convert a unit-aware sky coordinate to local array pixels."""

        return _skycoord_to_pixel(self.wcs, sky=sky)

    def world_degrees_to_pixel(
        self,
        ra: Any,
        dec: Any,
    ) -> tuple[Any, Any]:
        """Convert ICRS coordinates expressed explicitly in degrees."""

        return _world_degrees_to_pixel(self.wcs, ra=ra, dec=dec)


@dataclass
class RubinImageView:
    """Normalized planes and metadata from one Rubin image product.

    ``requirements_met`` records metadata presence for the selected reader.
    FITS inputs require units, an exposure summary, and WCS; CellCoadd
    archives require WCS, a PSF, and input provenance. Transport-only
    cutdowns do not meet these requirements. This flag does not validate
    scientific accuracy or suitability for a measurement.
    """

    source_path: Path
    source_format: str
    image: np.ndarray
    variance: np.ndarray
    bad_mask: np.ndarray
    native_mask: Any
    mask_plane_names: tuple[str, ...]
    selected_bad_mask_plane_names: tuple[str, ...]
    mask_plane_schema: dict[str, dict[str, int]]
    unit: str | None
    variance_unit: str | None
    bounds: RubinImageBounds
    wcs: Any | None
    wcs_fidelity: str
    requirements_met: bool
    product_role: str
    reconstruction_ready: bool
    exposure_time: float
    exposure_time_source: str
    photo_calib: dict[str, Any] | None = None
    exposure_summary: dict[str, Any] | None = None
    psf: Any | None = None
    aperture_corrections: dict[str, Any] | None = None
    provenance: Any | None = None
    cell_metadata: dict[str, Any] | None = None
    archive_info: dict[str, Any] | None = None
    archive_metadata: dict[str, Any] | None = None
    native: Any | None = None

    @property
    def mask_nonzero_fraction(self) -> float:
        if not self.bad_mask.size:
            return 0.0
        if self.source_format == "legacy-fits":
            nonzero = np.asarray(self.native_mask) != 0
        else:
            array = np.asarray(self.native_mask.array)
            nonzero = np.any(array != 0, axis=-1)
        return float(np.count_nonzero(nonzero) / nonzero.size)

    def get_mask_plane(self, name: str) -> np.ndarray:
        """Return one exact mask plane as a 2-D boolean array."""

        if name not in self.mask_plane_names:
            raise KeyError(f"Mask plane {name!r} is not present.")
        if self.source_format == "legacy-fits":
            schema = self.mask_plane_schema[name]
            bit = schema["bit"]
            raw = np.asarray(self.native_mask, dtype=np.uint64)
            return (raw & (1 << bit)) != 0
        return np.asarray(self.native_mask.get(name), dtype=bool)

    def pixel_to_skycoord(self, x: Any, y: Any) -> SkyCoord:
        """Convert local array pixels with the WCS high-level API."""

        return _pixel_to_skycoord(self.wcs, x=x, y=y)

    def skycoord_to_pixel(self, sky: SkyCoord) -> tuple[Any, Any]:
        """Convert a unit-aware sky coordinate to local array pixels."""

        return _skycoord_to_pixel(self.wcs, sky=sky)

    def world_degrees_to_pixel(
        self,
        ra: Any,
        dec: Any,
    ) -> tuple[Any, Any]:
        """Convert ICRS coordinates expressed explicitly in degrees.

        Native ``lsst.images`` WCS objects declare low-level world units in
        radians.  This high-level path prevents callers from accidentally
        passing degree-valued catalog coordinates to
        ``world_to_pixel_values``.
        """

        return _world_degrees_to_pixel(self.wcs, ra=ra, dec=dec)

    def sample_psf_kernel_at(
        self,
        *,
        x: float,
        y: float,
        size: int,
    ) -> np.ndarray | None:
        """Sample and normalize the native PSF at an absolute pixel.

        ``lsst.images`` cell PSFs are indexed in the coadd's absolute pixel
        frame, not in local array coordinates.  Missing cells therefore fail
        explicitly instead of falling back to an approximate or delta PSF.
        """

        if self.psf is None:
            return None
        if size <= 0 or size % 2 == 0:
            raise ValueError("PSF size must be a positive odd integer.")
        self._check_absolute_position(x=x, y=y)
        try:
            kernel_image = self.psf.compute_kernel_image(x=x, y=y)
        except Exception as exc:
            raise RubinImageFormatError(
                "Native lsst.images PSF is unavailable at "
                f"absolute pixel ({x}, {y}); the cell may be missing."
            ) from exc
        kernel = np.asarray(kernel_image.array, dtype=np.float64)
        if kernel.ndim != 2:
            raise RubinImageFormatError(
                "lsst.images PSF kernel is not a two-dimensional array."
            )
        kernel = _center_fit_array(kernel, size)
        if not np.all(np.isfinite(kernel)):
            raise RubinImageFormatError(
                "lsst.images PSF kernel contains non-finite values."
            )
        total = float(np.sum(kernel))
        if not math.isfinite(total) or total <= 0:
            raise RubinImageFormatError(
                "lsst.images PSF kernel has a non-positive sum."
            )
        return kernel / total

    def sample_psf_kernel(self, size: int) -> np.ndarray | None:
        """Sample the native spatial PSF at the center of this view."""

        x = (self.bounds.x_start + self.bounds.x_stop - 1) / 2.0
        y = (self.bounds.y_start + self.bounds.y_stop - 1) / 2.0
        return self.sample_psf_kernel_at(x=x, y=y, size=size)

    def sample_aperture_corrections_at(
        self,
        *,
        x: float,
        y: float,
    ) -> dict[str, dict[str, Any]]:
        """Evaluate native cell aperture corrections at an absolute pixel."""

        self._check_absolute_position(x=x, y=y)
        result: dict[str, dict[str, Any]] = {}
        for name, field in sorted((self.aperture_corrections or {}).items()):
            try:
                raw_value = float(field(x=x, y=y))
            except Exception as exc:
                raise RubinImageFormatError(
                    "Native lsst.images aperture correction "
                    f"{name!r} could not be evaluated at absolute pixel "
                    f"({x}, {y})."
                ) from exc
            value = raw_value if math.isfinite(raw_value) else None
            result[str(name)] = {
                "value": value,
                "unit": _unit_to_string(getattr(field, "unit", None)),
                "available": value is not None,
            }
        return result

    def cell_diagnostics_at(
        self,
        *,
        x: float,
        y: float,
    ) -> dict[str, Any] | None:
        """Return cell, missing-data, aperture, and provenance diagnostics."""

        self._check_absolute_position(x=x, y=y)
        if self.psf is None:
            return None
        bounds = getattr(self.psf, "bounds", None)
        grid = getattr(bounds, "grid", None)
        if bounds is None or grid is None:
            raise RubinImageFormatError(
                "Native lsst.images PSF does not expose cell-grid bounds."
            )
        cell = grid.index_of(x=round(x), y=round(y))
        missing = cell in bounds.missing
        cell_bbox = grid.bbox_of(cell)
        provenance_counts = _provenance_counts_for_cell(
            self.provenance,
            cell,
            missing=missing,
        )
        return {
            "cell_index_ij": [int(cell.i), int(cell.j)],
            "cell_bbox": _bounds_from_lsst_box(cell_bbox).to_dict(),
            "missing": bool(missing),
            "psf_available": not missing,
            "aperture_corrections": self.sample_aperture_corrections_at(
                x=x,
                y=y,
            ),
            "provenance": provenance_counts,
        }

    def _check_absolute_position(self, *, x: float, y: float) -> None:
        if (
            not math.isfinite(x)
            or not math.isfinite(y)
            or x < self.bounds.x_start
            or x >= self.bounds.x_stop
            or y < self.bounds.y_start
            or y >= self.bounds.y_stop
        ):
            raise ValueError(
                "Absolute pixel lies outside the normalized image bounds: "
                f"({x}, {y}) not in {self.bounds.to_dict()}."
            )


@dataclass(frozen=True)
class _LsstImagesAPI:
    images: Any
    cell_coadd_cls: type
    open_archive: Any


def open_rubin_image(
    path: str | Path,
    *,
    source_format: str,
    section: tuple[slice, slice] | None = None,
    mask_policy: str = "rubin-bad-planes",
    include_provenance: bool = True,
) -> RubinImageView:
    """Open one Rubin image with the explicitly selected backend."""

    source_format = _validate_source_format(source_format)
    section = _validate_section(section)
    mask_policy = _validate_mask_policy(mask_policy)
    source_path = Path(path).expanduser()
    if not source_path.is_file():
        raise FileNotFoundError(f"Rubin image file not found: {source_path}")

    if source_format == "legacy-fits":
        return _open_legacy_fits(
            source_path,
            section=section,
            mask_policy=mask_policy,
        )
    if source_format == "lsst-images-archive":
        return _open_lsst_images_archive(
            source_path,
            section=section,
            mask_policy=mask_policy,
            include_provenance=include_provenance,
        )
    return _open_lsst_images_cutdown(
        source_path,
        section=section,
        mask_policy=mask_policy,
    )


def read_rubin_image_bounds(
    path: str | Path,
    *,
    source_format: str,
) -> RubinImageBounds:
    """Read image bounds without loading science pixel arrays."""

    source_format = _validate_source_format(source_format)
    source_path = Path(path).expanduser()
    if not source_path.is_file():
        raise FileNotFoundError(f"Rubin image file not found: {source_path}")

    if source_format == "legacy-fits":
        with fits.open(source_path, memmap=True) as hdul:
            _reject_wrong_fits_backend(hdul, source_format)
            image_hdu = _get_named_hdu(hdul, "IMAGE")
            shape = tuple(int(value) for value in image_hdu.shape)
        if len(shape) != 2:
            raise RubinImageFormatError(
                f"IMAGE HDU must be two-dimensional, got {shape}."
            )
        return RubinImageBounds(0, shape[0], 0, shape[1])

    if source_format == "lsst-images-cutdown":
        with fits.open(source_path, memmap=True) as hdul:
            _reject_wrong_fits_backend(hdul, source_format)
            image_hdu = _get_named_hdu(hdul, "IMAGE")
            shape = tuple(int(value) for value in image_hdu.shape)
            if len(shape) != 2:
                raise RubinImageFormatError(
                    f"IMAGE HDU must be two-dimensional, got {shape}."
                )
            y_start, x_start = _cutdown_origin(image_hdu.header)
        return RubinImageBounds(
            y_start,
            y_start + shape[0],
            x_start,
            x_start + shape[1],
        )

    api = _load_lsst_images_api()
    try:
        with api.open_archive(
            source_path,
            api.cell_coadd_cls,
            partial=True,
        ) as reader:
            _validate_cell_coadd_archive_info(reader.info)
            return _bounds_from_lsst_box(reader.get_component("bbox"))
    except RubinImageError:
        raise
    except Exception as exc:
        raise RubinImageFormatError(
            f"Failed to inspect lsst.images CellCoadd archive {source_path}: "
            f"{exc}"
        ) from exc


def inspect_rubin_image_geometry(
    path: str | Path,
    *,
    source_format: str,
) -> RubinImageGeometry:
    """Read bounds and WCS without retrieving science pixel arrays."""

    source_format = _validate_source_format(source_format)
    source_path = Path(path).expanduser()
    if not source_path.is_file():
        raise FileNotFoundError(f"Rubin image file not found: {source_path}")

    if source_format in {"legacy-fits", "lsst-images-cutdown"}:
        with fits.open(source_path, memmap=True) as hdul:
            _reject_wrong_fits_backend(hdul, source_format)
            image_hdu = _get_named_hdu(hdul, "IMAGE")
            shape = tuple(int(value) for value in image_hdu.shape)
            if len(shape) != 2:
                raise RubinImageFormatError(
                    f"IMAGE HDU must be two-dimensional, got {shape}."
                )
            if source_format == "lsst-images-cutdown":
                y_start, x_start = _cutdown_origin(image_hdu.header)
            else:
                y_start, x_start = 0, 0
            wcs = WCS(image_hdu.header)
        return RubinImageGeometry(
            source_path=source_path,
            source_format=source_format,
            bounds=RubinImageBounds(
                y_start,
                y_start + shape[0],
                x_start,
                x_start + shape[1],
            ),
            wcs=wcs,
            wcs_fidelity=(
                "fits-cutdown"
                if source_format == "lsst-images-cutdown"
                else "fits"
            ),
            archive_info=None,
        )

    api = _load_lsst_images_api()
    try:
        with api.open_archive(
            source_path,
            api.cell_coadd_cls,
            partial=True,
        ) as reader:
            archive_info = _validate_cell_coadd_archive_info(reader.info)
            bbox = reader.get_component("bbox")
            sky_projection = reader.get_component("sky_projection")
            if sky_projection is None:
                raise RubinImageFormatError(
                    "CellCoadd archive has no native sky projection."
                )
            wcs = sky_projection.as_astropy(bbox)
    except RubinImageError:
        raise
    except Exception as exc:
        raise RubinImageFormatError(
            "Failed to inspect lsst.images CellCoadd geometry "
            f"{source_path}: {exc}"
        ) from exc
    return RubinImageGeometry(
        source_path=source_path,
        source_format=source_format,
        bounds=_bounds_from_lsst_box(bbox),
        wcs=wcs,
        wcs_fidelity="exact-lsst-images",
        archive_info=archive_info,
    )


def _open_legacy_fits(
    path: Path,
    *,
    section: tuple[slice, slice] | None,
    mask_policy: str,
) -> RubinImageView:
    try:
        with fits.open(path, memmap=True) as hdul:
            _reject_wrong_fits_backend(hdul, "legacy-fits")
            image_hdu = _get_named_hdu(hdul, "IMAGE")
            mask_hdu = _get_named_hdu(hdul, "MASK")
            variance_hdu = _get_named_hdu(hdul, "VARIANCE")
            raw_shape = tuple(int(value) for value in image_hdu.shape)
            if len(raw_shape) != 2:
                raise RubinImageFormatError(
                    f"IMAGE HDU must be two-dimensional, got {raw_shape}."
                )
            shape = (raw_shape[0], raw_shape[1])
            active_section = section or (
                slice(0, shape[0]),
                slice(0, shape[1]),
            )
            _check_section_within_shape(active_section, shape)
            image = np.array(
                _read_hdu_section(image_hdu, active_section),
                dtype=np.float32,
                copy=True,
            )
            variance = np.array(
                _read_hdu_section(variance_hdu, active_section),
                dtype=np.float32,
                copy=True,
            )
            raw_mask = np.array(
                _read_hdu_section(mask_hdu, active_section),
                copy=True,
            )
            mask_bits = _extract_mask_plane_bits(mask_hdu.header)
            bad_mask, selected = _legacy_bad_mask(
                raw_mask,
                mask_bits,
                mask_policy,
            )
            full_wcs = WCS(image_hdu.header)
            wcs = (
                full_wcs
                if section is None
                else full_wcs.slice(active_section)
            )
            exposure_time, exposure_time_source = _extract_exposure_time(hdul)
            photo_calib = _extract_photo_calib(hdul)
            exposure_summary = _extract_exposure_summary(hdul)
            unit = _extract_bunit(image_hdu)
            variance_unit = _extract_bunit(variance_hdu)
    except RubinImageError:
        raise
    except Exception as exc:
        raise RubinImageFormatError(
            f"Failed to read legacy Rubin FITS {path}: {exc}"
        ) from exc

    bounds = RubinImageBounds(
        active_section[0].start,
        active_section[0].stop,
        active_section[1].start,
        active_section[1].stop,
    )
    return RubinImageView(
        source_path=path,
        source_format="legacy-fits",
        image=image,
        variance=variance,
        bad_mask=bad_mask,
        native_mask=raw_mask,
        mask_plane_names=tuple(sorted(mask_bits)),
        selected_bad_mask_plane_names=selected,
        mask_plane_schema={
            name: {"bit": bit} for name, bit in sorted(mask_bits.items())
        },
        unit=unit,
        variance_unit=variance_unit,
        bounds=bounds,
        wcs=wcs,
        wcs_fidelity="fits",
        requirements_met=(
            unit is not None
            and exposure_summary is not None
            and wcs is not None
        ),
        product_role="legacy-image-unspecified",
        reconstruction_ready=True,
        exposure_time=exposure_time,
        exposure_time_source=exposure_time_source,
        photo_calib=photo_calib,
        exposure_summary=exposure_summary,
        archive_info={
            "schema_name": "legacy_fits",
            "schema_version": None,
            "format_version": None,
        },
    )


def _open_lsst_images_archive(
    path: Path,
    *,
    section: tuple[slice, slice] | None,
    mask_policy: str,
    include_provenance: bool,
) -> RubinImageView:
    api = _load_lsst_images_api()
    try:
        with api.open_archive(
            path,
            api.cell_coadd_cls,
            partial=True,
        ) as reader:
            archive_info = _validate_cell_coadd_archive_info(reader.info)
            full_bbox = reader.get_component("bbox")
            bbox = _section_to_lsst_box(section, full_bbox, api.images.Box)
            read_provenance = include_provenance
            provenance_skipped_missing_only = False
            provenance_override = None
            if include_provenance and bbox is not None:
                bounded_psf = reader.get_component("psf", bbox=bbox)
                cell_indices = tuple(bounded_psf.bounds.cell_indices())
                if not cell_indices:
                    read_provenance = False
                    provenance_skipped_missing_only = True
                else:
                    full_provenance = reader.get_component("provenance")
                    if (
                        full_provenance is not None
                        and _provenance_contribution_count(
                            full_provenance,
                            cell_indices,
                        )
                        == 0
                    ):
                        read_provenance = False
                        try:
                            provenance_override = type(full_provenance)(
                                inputs=full_provenance.inputs[:0],
                                contributions=(
                                    full_provenance.contributions[:0]
                                ),
                            )
                        except Exception as exc:
                            raise RubinImageFormatError(
                                "CellCoadd empty provenance could not be "
                                "represented."
                            ) from exc
            coadd = reader.read(
                bbox=bbox,
                provenance=read_provenance,
            )
            archive_metadata = dict(reader.metadata)
    except RubinImageError:
        raise
    except Exception as exc:
        raise RubinImageFormatError(
            f"Failed to read lsst.images CellCoadd archive {path}: {exc}"
        ) from exc

    if not isinstance(coadd, api.cell_coadd_cls):
        raise RubinImageFormatError(
            "lsst.images archive did not deserialize to CellCoadd."
        )
    provenance = (
        provenance_override
        if provenance_override is not None
        else _optional_attribute(coadd, "provenance")
    )
    psf = _optional_attribute(coadd, "psf")
    aperture_corrections = dict(
        _optional_attribute(coadd, "aperture_corrections") or {}
    )
    view = _view_from_lsst_masked_image(
        path=path,
        source_format="lsst-images-archive",
        masked_image=coadd,
        mask_policy=mask_policy,
        wcs_fidelity="exact-lsst-images",
        requirements_met=(
            _optional_attribute(coadd, "astropy_wcs") is not None
            and psf is not None
            and provenance is not None
        ),
        product_role="reference-evaluation-only",
        reconstruction_ready=False,
        exposure_time=1.0,
        exposure_time_source="calibrated_archive_unit",
        psf=psf,
        aperture_corrections=aperture_corrections,
        provenance=provenance,
        cell_metadata=_cell_metadata(
            coadd,
            provenance=provenance,
            aperture_corrections=aperture_corrections,
            provenance_requested=include_provenance,
            provenance_skipped_missing_only=(provenance_skipped_missing_only),
        ),
        archive_info=archive_info,
        archive_metadata=archive_metadata,
    )
    return view


def _open_lsst_images_cutdown(
    path: Path,
    *,
    section: tuple[slice, slice] | None,
    mask_policy: str,
) -> RubinImageView:
    api = _load_lsst_images_api()
    try:
        with fits.open(path, memmap=False) as hdul:
            _reject_wrong_fits_backend(hdul, "lsst-images-cutdown")
            exposure_time, exposure_time_source = _extract_exposure_time(hdul)
            masked_image = api.images.MaskedImage.from_hdu_list(hdul)
        if section is not None:
            bbox = _section_to_lsst_box(
                section,
                masked_image.bbox,
                api.images.Box,
            )
            masked_image = masked_image[bbox]
    except RubinImageError:
        raise
    except Exception as exc:
        raise RubinImageFormatError(
            f"Failed to reconstruct lsst.images cutdown {path}: {exc}"
        ) from exc

    return _view_from_lsst_masked_image(
        path=path,
        source_format="lsst-images-cutdown",
        masked_image=masked_image,
        mask_policy=mask_policy,
        wcs_fidelity="fits-cutdown",
        requirements_met=False,
        product_role="pixel-transport-only",
        reconstruction_ready=False,
        exposure_time=exposure_time,
        exposure_time_source=exposure_time_source,
        psf=None,
        aperture_corrections={},
        provenance=None,
        cell_metadata={
            "psf_present": False,
            "aperture_corrections_present": False,
            "aperture_correction_names": [],
            "provenance_present": False,
            "provenance_input_count": None,
            "provenance_contribution_count": None,
            "cutdown_metadata_complete": False,
        },
        archive_info={
            "schema_name": "masked_image_cutdown",
            "schema_version": None,
            "format_version": None,
        },
        archive_metadata={},
    )


def _view_from_lsst_masked_image(
    *,
    path: Path,
    source_format: str,
    masked_image: Any,
    mask_policy: str,
    wcs_fidelity: str,
    requirements_met: bool,
    product_role: str,
    reconstruction_ready: bool,
    exposure_time: float,
    exposure_time_source: str,
    psf: Any | None,
    aperture_corrections: dict[str, Any],
    provenance: Any | None,
    cell_metadata: dict[str, Any],
    archive_info: dict[str, Any],
    archive_metadata: dict[str, Any],
) -> RubinImageView:
    image = np.array(masked_image.image.array, dtype=np.float32, copy=True)
    variance = np.array(
        masked_image.variance.array,
        dtype=np.float32,
        copy=True,
    )
    if image.ndim != 2 or variance.shape != image.shape:
        raise RubinImageFormatError(
            "lsst.images image and variance planes must be matching 2-D "
            f"arrays, got {image.shape} and {variance.shape}."
        )
    native_mask = masked_image.mask
    names = tuple(sorted(str(name) for name in native_mask.schema.names))
    bad_mask, selected = _lsst_bad_mask(
        native_mask,
        names,
        mask_policy,
    )
    if bad_mask.shape != image.shape:
        raise RubinImageFormatError(
            "lsst.images mask shape does not match the image plane."
        )
    mask_array = np.asarray(native_mask.array)
    if mask_array.ndim != 3 or mask_array.dtype.kind != "u":
        raise RubinImageFormatError(
            "lsst.images mask storage must be unsigned word planes."
        )
    bits_per_word = int(np.iinfo(mask_array.dtype).bits)
    schema = {}
    global_indices = set()
    for name in names:
        bit = native_mask.schema.bit(name)
        word = int(bit.index)
        word_mask = int(bit.mask)
        if (
            word < 0
            or word >= mask_array.shape[-1]
            or word_mask <= 0
            or word_mask & (word_mask - 1)
        ):
            raise RubinImageFormatError(
                "lsst.images mask schema contains an invalid word bit."
            )
        local_bit = word_mask.bit_length() - 1
        if local_bit >= bits_per_word:
            raise RubinImageFormatError(
                "lsst.images mask bit exceeds its storage word."
            )
        global_index = word * bits_per_word + local_bit
        if global_index in global_indices:
            raise RubinImageFormatError(
                "lsst.images mask schema contains duplicate bits."
            )
        global_indices.add(global_index)
        schema[name] = {
            "index": global_index,
            "mask": 1 << global_index,
            "word": word,
            "word_mask": word_mask,
            "bit": local_bit,
        }
    unit = _unit_to_string(masked_image.image.unit)
    variance_unit = _unit_to_string(masked_image.variance.unit)
    return RubinImageView(
        source_path=path,
        source_format=source_format,
        image=image,
        variance=variance,
        bad_mask=bad_mask,
        native_mask=native_mask,
        mask_plane_names=names,
        selected_bad_mask_plane_names=selected,
        mask_plane_schema=schema,
        unit=unit,
        variance_unit=variance_unit,
        bounds=_bounds_from_lsst_box(masked_image.bbox),
        wcs=masked_image.astropy_wcs,
        wcs_fidelity=wcs_fidelity,
        requirements_met=requirements_met,
        product_role=product_role,
        reconstruction_ready=reconstruction_ready,
        exposure_time=exposure_time,
        exposure_time_source=exposure_time_source,
        psf=psf,
        aperture_corrections=aperture_corrections,
        provenance=provenance,
        cell_metadata=cell_metadata,
        archive_info=archive_info,
        archive_metadata=archive_metadata,
        native=masked_image,
    )


def _load_lsst_images_api() -> _LsstImagesAPI:
    try:
        images = import_module("lsst.images")
        cells = import_module("lsst.images.cells")
        serialization = import_module("lsst.images.serialization")
    except (ImportError, ModuleNotFoundError) as exc:
        raise RubinImageDependencyError(
            "Rubin archive access requires cuphoton[rubin] with "
            f"lsst-images=={LSST_IMAGES_VERSION}."
        ) from exc

    version = getattr(images, "__version__", None)
    if version != LSST_IMAGES_VERSION:
        raise RubinImageDependencyError(
            "Rubin archive access requires exactly "
            f"lsst-images=={LSST_IMAGES_VERSION}; found {version!r}."
        )
    cell_coadd_cls = getattr(cells, "CellCoadd", None)
    open_archive = getattr(serialization, "open_archive", None)
    if cell_coadd_cls is None or open_archive is None:
        raise RubinImageDependencyError(
            f"lsst-images=={LSST_IMAGES_VERSION} is missing the released "
            "CellCoadd archive API."
        )
    return _LsstImagesAPI(
        images=images,
        cell_coadd_cls=cell_coadd_cls,
        open_archive=open_archive,
    )


def _validate_cell_coadd_archive_info(info: Any) -> dict[str, Any]:
    schema_name = str(info.schema_name)
    if schema_name != "cell_coadd":
        raise RubinImageFormatError(
            "Expected an lsst.images CellCoadd archive with schema "
            f"'cell_coadd', found {schema_name!r}."
        )
    return {
        "schema_url": str(info.schema_url),
        "schema_name": schema_name,
        "schema_version": str(info.schema_version),
        "format_version": (
            None if info.format_version is None else int(info.format_version)
        ),
        "lsst_images_version": LSST_IMAGES_VERSION,
    }


def _reject_wrong_fits_backend(
    hdul: fits.HDUList,
    source_format: str,
) -> None:
    data_model = hdul[0].header.get("DATAMODL")
    names = {str(hdu.name).upper() for hdu in hdul}
    if source_format == "legacy-fits" and data_model is not None:
        raise RubinImageFormatError(
            "A DATAMODL-bearing lsst.images archive was passed to the "
            "legacy-fits backend."
        )
    if source_format == "lsst-images-cutdown":
        has_index_hdus = {"JSON", "INDEX"}.issubset(names)
        data_model_text = (
            "" if data_model is None else str(data_model).casefold()
        )
        indexed_masked_image = (
            has_index_hdus and "/masked_image-" in data_model_text
        )
        if has_index_hdus and not indexed_masked_image:
            raise RubinImageFormatError(
                "An indexed lsst.images product that is not declared as a "
                "MaskedImage was passed to the cutdown backend."
            )


def _cutdown_origin(header: fits.Header) -> tuple[int, int]:
    """Read the absolute ``(y, x)`` origin of a valid cut-down image."""

    has_ltv1 = "LTV1" in header
    has_ltv2 = "LTV2" in header
    if has_ltv1 or has_ltv2:
        if not (has_ltv1 and has_ltv2):
            raise RubinImageFormatError(
                "Cut-down IMAGE HDU has an incomplete LTV1/LTV2 origin."
            )
        return -round(header["LTV2"]), -round(header["LTV1"])
    if header.get("CTYPE1A") == "LINEAR":
        required = {"CRVAL1A", "CRVAL2A"}
        if not required.issubset(header):
            raise RubinImageFormatError(
                "Cut-down IMAGE HDU has an incomplete linear offset WCS."
            )
        x_start = round(
            header["CRVAL1A"] - (header.get("CRPIX1A", 1.0) - 1.0)
        )
        y_start = round(
            header["CRVAL2A"] - (header.get("CRPIX2A", 1.0) - 1.0)
        )
        return y_start, x_start
    raise RubinImageFormatError(
        "Cut-down IMAGE HDU records no absolute pixel origin via "
        "LTV1/LTV2 or the lsst.images linear offset WCS."
    )


def _legacy_bad_mask(
    raw_mask: np.ndarray,
    mask_bits: dict[str, int],
    mask_policy: str,
) -> tuple[np.ndarray, tuple[str, ...]]:
    if mask_policy == "none-good":
        return np.zeros(raw_mask.shape, dtype=bool), ()
    if mask_policy == "nonzero-bad":
        return np.asarray(raw_mask) != 0, ("ANY_NONZERO",)
    selected = tuple(
        name for name in _RUBIN_BAD_MASK_PLANES if name in mask_bits
    )
    bitmask = 0
    for name in selected:
        bit = mask_bits[name]
        if bit >= 0:
            bitmask |= 1 << bit
    raw = np.asarray(raw_mask, dtype=np.uint64)
    return (raw & bitmask) != 0, selected


def _lsst_bad_mask(
    mask: Any,
    names: tuple[str, ...],
    mask_policy: str,
) -> tuple[np.ndarray, tuple[str, ...]]:
    shape = tuple(int(value) for value in mask.bbox.shape)
    if mask_policy == "none-good":
        return np.zeros(shape, dtype=bool), ()
    if mask_policy == "nonzero-bad":
        bad = np.any(np.asarray(mask.array) != 0, axis=-1)
        return np.asarray(bad, dtype=bool), ("ANY_NONZERO",)
    by_casefold = {name.casefold(): name for name in names}
    selected = tuple(
        by_casefold[name.casefold()]
        for name in _LSST_BAD_MASK_PLANES
        if name.casefold() in by_casefold
    )
    bad = np.zeros(shape, dtype=bool)
    for name in selected:
        bad |= np.asarray(mask.get(name), dtype=bool)
    return bad, selected


def _section_to_lsst_box(
    section: tuple[slice, slice] | None,
    full_bbox: Any,
    box_cls: Any,
) -> Any | None:
    if section is None:
        return None
    height, width = full_bbox.shape
    _check_section_within_shape(section, (int(height), int(width)))
    y_slice, x_slice = section
    return box_cls.from_shape(
        (
            y_slice.stop - y_slice.start,
            x_slice.stop - x_slice.start,
        ),
        start=(
            int(full_bbox.y.start) + y_slice.start,
            int(full_bbox.x.start) + x_slice.start,
        ),
    )


def _bounds_from_lsst_box(box: Any) -> RubinImageBounds:
    return RubinImageBounds(
        y_start=int(box.y.start),
        y_stop=int(box.y.stop),
        x_start=int(box.x.start),
        x_stop=int(box.x.stop),
    )


def _cell_metadata(
    coadd: Any,
    *,
    provenance: Any | None,
    aperture_corrections: dict[str, Any],
    provenance_requested: bool,
    provenance_skipped_missing_only: bool,
) -> dict[str, Any]:
    patch = _optional_attribute(coadd, "patch")
    grid = _optional_attribute(coadd, "grid")
    bounds = _optional_attribute(coadd, "bounds")
    result: dict[str, Any] = {
        "band": _optional_attribute(coadd, "band"),
        "skymap": _optional_attribute(coadd, "skymap"),
        "tract": _optional_attribute(coadd, "tract"),
        "patch_present": patch is not None,
        "patch": _model_dump_json(patch),
        "psf_present": _optional_attribute(coadd, "psf") is not None,
        "aperture_corrections_present": bool(aperture_corrections),
        "aperture_correction_names": sorted(aperture_corrections),
        "provenance_present": provenance is not None,
        "provenance_requested": provenance_requested,
        "provenance_skipped_missing_only": (provenance_skipped_missing_only),
        "provenance_type": (
            None if provenance is None else type(provenance).__name__
        ),
        "provenance_input_count": _table_length(
            _optional_attribute(provenance, "inputs")
            if provenance is not None
            else None
        ),
        "provenance_contribution_count": _table_length(
            _optional_attribute(provenance, "contributions")
            if provenance is not None
            else None
        ),
        "mask_fraction_planes": sorted(
            str(name) for name in getattr(coadd, "mask_fractions", {}).keys()
        ),
    }
    if grid is not None:
        result["grid"] = {
            "bbox": _bounds_from_lsst_box(grid.bbox).to_dict(),
            "cell_shape_yx": [
                int(grid.cell_shape.y),
                int(grid.cell_shape.x),
            ],
        }
    if bounds is not None:
        result["missing_cells"] = sorted(
            [int(cell.i), int(cell.j)] for cell in bounds.missing
        )
    return result


def _provenance_counts_for_cell(
    provenance: Any | None,
    cell: Any,
    *,
    missing: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "present": provenance is not None,
        "available_for_cell": False,
        "input_count": None,
        "contribution_count": None,
    }
    if provenance is None or missing:
        return result
    if _provenance_contribution_count(provenance, (cell,)) == 0:
        result.update(
            {
                "available_for_cell": True,
                "input_count": 0,
                "contribution_count": 0,
            }
        )
        return result
    try:
        cell_provenance = provenance[cell]
    except Exception as exc:
        raise RubinImageFormatError(
            "CellCoadd provenance could not be selected for "
            f"cell ({cell.i}, {cell.j})."
        ) from exc
    result.update(
        {
            "available_for_cell": True,
            "input_count": _table_length(
                _optional_attribute(cell_provenance, "inputs")
            ),
            "contribution_count": _table_length(
                _optional_attribute(cell_provenance, "contributions")
            ),
        }
    )
    return result


def _provenance_contribution_count(
    provenance: Any,
    cells: tuple[Any, ...],
) -> int:
    contributions = _optional_attribute(provenance, "contributions")
    if contributions is None:
        raise RubinImageFormatError(
            "CellCoadd provenance has no contribution table."
        )
    try:
        cell_i = np.asarray(contributions["cell_i"])
        cell_j = np.asarray(contributions["cell_j"])
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise RubinImageFormatError(
            "CellCoadd provenance contributions are invalid."
        ) from exc
    if cell_i.ndim != 1 or cell_i.shape != cell_j.shape:
        raise RubinImageFormatError(
            "CellCoadd provenance contribution indices differ."
        )
    matches = np.zeros(cell_i.shape, dtype=bool)
    for cell in cells:
        matches |= (cell_i == int(cell.i)) & (cell_j == int(cell.j))
    return int(np.count_nonzero(matches))


def _table_length(value: Any | None) -> int | None:
    if value is None:
        return None
    try:
        return int(len(value))
    except TypeError:
        return None


def _model_dump_json(value: Any | None) -> Any | None:
    if value is None:
        return None
    model_dump = getattr(value, "model_dump", None)
    if model_dump is None:
        return {"type": type(value).__name__}
    return model_dump(mode="json", exclude_none=True)


def _optional_attribute(value: Any, name: str) -> Any | None:
    try:
        return getattr(value, name)
    except AttributeError:
        return None


def _validate_source_format(source_format: str) -> str:
    if source_format not in RUBIN_IMAGE_FORMATS:
        raise RubinImageFormatError(
            "source_format must be one of "
            + ", ".join(sorted(RUBIN_IMAGE_FORMATS))
            + f"; got {source_format!r}."
        )
    return source_format


def _validate_mask_policy(mask_policy: str) -> str:
    if mask_policy not in _MASK_POLICIES:
        raise ValueError(
            "mask_policy must be one of " + ", ".join(sorted(_MASK_POLICIES))
        )
    return mask_policy


def _pixel_to_skycoord(wcs: Any | None, *, x: Any, y: Any) -> SkyCoord:
    if wcs is None:
        raise RubinImageFormatError("Rubin image has no sky WCS.")
    try:
        sky = wcs.pixel_to_world(x, y)
    except Exception as exc:
        raise RubinImageFormatError(
            "Rubin image WCS could not convert pixels to sky coordinates."
        ) from exc
    if not isinstance(sky, SkyCoord):
        raise RubinImageFormatError(
            "Rubin image WCS is not a two-dimensional celestial WCS."
        )
    return sky


def _skycoord_to_pixel(
    wcs: Any | None,
    *,
    sky: SkyCoord,
) -> tuple[Any, Any]:
    if wcs is None:
        raise RubinImageFormatError("Rubin image has no sky WCS.")
    if not isinstance(sky, SkyCoord):
        raise TypeError("sky must be an astropy.coordinates.SkyCoord.")
    try:
        pixels = wcs.world_to_pixel(sky)
    except Exception as exc:
        raise RubinImageFormatError(
            "Rubin image WCS could not convert sky coordinates to pixels."
        ) from exc
    if not isinstance(pixels, tuple) or len(pixels) != 2:
        raise RubinImageFormatError(
            "Rubin image WCS did not return two-dimensional pixels."
        )
    return pixels[0], pixels[1]


def _world_degrees_to_pixel(
    wcs: Any | None,
    *,
    ra: Any,
    dec: Any,
) -> tuple[Any, Any]:
    sky = SkyCoord(ra=ra, dec=dec, unit="deg", frame="icrs")
    return _skycoord_to_pixel(wcs, sky=sky)


def _validate_section(
    section: tuple[slice, slice] | None,
) -> tuple[slice, slice] | None:
    if section is None:
        return None
    if len(section) != 2:
        raise ValueError("section must contain y and x slices.")
    for item in section:
        if (
            item.start is None
            or item.stop is None
            or item.step not in (None, 1)
            or item.start < 0
            or item.stop <= item.start
        ):
            raise ValueError(
                "section slices require 0 <= start < stop and unit step."
            )
    return section


def _check_section_within_shape(
    section: tuple[slice, slice],
    shape: tuple[int, int],
) -> None:
    y_slice, x_slice = section
    if y_slice.stop > shape[0] or x_slice.stop > shape[1]:
        raise RubinImageFormatError(
            f"Requested section exceeds image shape {shape}: "
            f"{y_slice.start}:{y_slice.stop},"
            f"{x_slice.start}:{x_slice.stop}."
        )


def _get_named_hdu(hdul: fits.HDUList, name: str) -> Any:
    normalized = name.casefold()
    for hdu in hdul:
        if hdu.name.casefold() == normalized:
            return hdu
    raise RubinImageFormatError(
        f"FITS file does not contain HDU named {name!r}."
    )


def _read_hdu_section(
    hdu: Any,
    section: tuple[slice, slice],
) -> np.ndarray:
    if hasattr(hdu, "section"):
        return np.asarray(hdu.section[section])
    return np.asarray(hdu.data[section])


def _extract_exposure_time(
    hdul: fits.HDUList,
) -> tuple[float, str]:
    for hdu in hdul:
        value = hdu.header.get("EXPTIME")
        if value is None:
            continue
        try:
            exposure_time = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(exposure_time) and exposure_time > 0:
            return exposure_time, f"{hdu.name}.header.EXPTIME"
    return 1.0, "default_unit"


def _extract_bunit(hdu: Any) -> str | None:
    value = hdu.header.get("BUNIT")
    if value is None:
        return None
    return str(value).strip() or None


def _extract_photo_calib(
    hdul: fits.HDUList,
) -> dict[str, Any] | None:
    try:
        hdu = _get_named_hdu(hdul, "PhotoCalib")
    except RubinImageFormatError:
        return None
    data = getattr(hdu, "data", None)
    if data is None or len(data) == 0:
        return None
    row = data[0]
    result: dict[str, Any] = {}
    for key in ("calibrationMean", "calibrationErr"):
        if key in hdu.columns.names:
            result[key] = _safe_float_or_none(row[key])
    for key in ("PHOTOCALIB_ID", "ARCHIVE_ID_PHOTOCALIB"):
        value = hdul[0].header.get(key)
        if value is not None:
            result[key.lower()] = int(value)
    return result or None


def _extract_exposure_summary(
    hdul: fits.HDUList,
) -> dict[str, Any] | None:
    try:
        hdu = _get_named_hdu(hdul, "ExposureSummaryStats")
    except RubinImageFormatError:
        return None
    data = getattr(hdu, "data", None)
    names = getattr(getattr(hdu, "columns", None), "names", None) or ()
    if data is None or "bytes" not in names:
        return None
    for row in data:
        raw = row["bytes"]
        payload = bytes(np.asarray(raw, dtype=np.uint8).tolist())
        text = payload.decode("utf-8", errors="ignore")
        if "psfSigma:" not in text and "skyBg:" not in text:
            continue
        parsed = yaml.safe_load(text)
        if not isinstance(parsed, dict):
            continue
        summary = {
            key: _safe_float_or_none(parsed.get(key))
            for key in _EXPOSURE_SUMMARY_KEYS
            if key in parsed
        }
        return summary or None
    return None


def _extract_mask_plane_bits(header: fits.Header) -> dict[str, int]:
    bits: dict[str, int] = {}
    for key, value in header.items():
        if not key.startswith("MP_"):
            continue
        try:
            bits[key[3:]] = int(value)
        except (TypeError, ValueError):
            continue
    return bits


def _safe_float_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _unit_to_string(unit: Any | None) -> str | None:
    if unit is None:
        return None
    to_string = getattr(unit, "to_string", None)
    if to_string is not None:
        return str(to_string())
    return str(unit)


def _center_fit_array(array: np.ndarray, size: int) -> np.ndarray:
    output = np.zeros((size, size), dtype=np.float64)
    source_y0 = max((array.shape[0] - size) // 2, 0)
    source_x0 = max((array.shape[1] - size) // 2, 0)
    source_y1 = min(source_y0 + size, array.shape[0])
    source_x1 = min(source_x0 + size, array.shape[1])
    height = source_y1 - source_y0
    width = source_x1 - source_x0
    target_y0 = (size - height) // 2
    target_x0 = (size - width) // 2
    output[
        target_y0 : target_y0 + height,
        target_x0 : target_x0 + width,
    ] = array[source_y0:source_y1, source_x0:source_x1]
    return output


__all__ = [
    "LSST_IMAGES_VERSION",
    "RUBIN_IMAGE_FORMATS",
    "RubinImageBounds",
    "RubinImageDependencyError",
    "RubinImageError",
    "RubinImageFormatError",
    "RubinImageGeometry",
    "RubinImageView",
    "inspect_rubin_image_geometry",
    "local_pixel_is_addressable",
    "local_pixel_is_inside",
    "local_to_absolute_pixel",
    "open_rubin_image",
    "read_rubin_image_bounds",
]
