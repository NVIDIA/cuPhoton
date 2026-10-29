# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from cuphoton.ximage.adapters.rubin import image as rubin_image
from cuphoton.ximage.adapters.rubin import warp as rubin_warp
from cuphoton.ximage.adapters.rubin.bundle import (
    build_rubin_store,
    read_manifest_stack,
    summarize_rubin_coadd_semantics,
)
from cuphoton.ximage.adapters.rubin.image import (
    RubinImageDependencyError,
    RubinImageFormatError,
    inspect_rubin_image_geometry,
    open_rubin_image,
    read_rubin_image_bounds,
)


class _YX(NamedTuple):
    y: int
    x: int


@dataclass(frozen=True)
class _Interval:
    start: int
    stop: int


@dataclass(frozen=True)
class _CellIndex:
    i: int
    j: int


class _Box:
    def __init__(
        self,
        *,
        y_start: int,
        y_stop: int,
        x_start: int,
        x_stop: int,
    ) -> None:
        self.y = _Interval(y_start, y_stop)
        self.x = _Interval(x_start, x_stop)

    @property
    def shape(self) -> _YX:
        return _YX(
            self.y.stop - self.y.start,
            self.x.stop - self.x.start,
        )

    @classmethod
    def from_shape(cls, shape, start=None):
        y_start, x_start = (0, 0) if start is None else start
        return cls(
            y_start=y_start,
            y_stop=y_start + shape[0],
            x_start=x_start,
            x_stop=x_start + shape[1],
        )


class _Unit:
    def __init__(self, value: str) -> None:
        self.value = value

    def to_string(self) -> str:
        return self.value


class _Bit:
    def __init__(self, index: int, mask: int) -> None:
        self.index = index
        self.mask = mask


class _MaskSchema:
    names = frozenset({"BAD", "DETECTED", "SATURATED"})

    def bit(self, name: str) -> _Bit:
        return {
            "BAD": _Bit(0, 1),
            "DETECTED": _Bit(1, 1),
            "SATURATED": _Bit(0, 4),
        }[name]


class _Mask:
    def __init__(self, bbox: _Box) -> None:
        self.bbox = bbox
        self.schema = _MaskSchema()
        self.array = np.zeros(bbox.shape + (2,), dtype=np.uint8)
        self.array[0, 0, 0] = 1
        self.array[-1, -1, 1] = 1
        self.array[0, -1, 0] = 4

    def get(self, name: str) -> np.ndarray:
        bit = self.schema.bit(name)
        return (self.array[..., bit.index] & bit.mask) != 0


class _Image:
    def __init__(
        self,
        array: np.ndarray,
        *,
        unit: str,
    ) -> None:
        self.array = array
        self.unit = _Unit(unit)


class _Psf:
    def __init__(self, bounds) -> None:
        self.samples = []
        self.bounds = bounds

    def compute_kernel_image(self, *, x: float, y: float):
        self.samples.append((x, y))
        cell = self.bounds.grid.index_of(x=round(x), y=round(y))
        if cell in self.bounds.missing:
            raise ValueError("missing cell")
        kernel = np.arange(9, dtype=np.float64).reshape(3, 3) + 1.0
        return SimpleNamespace(array=kernel)


class _Grid:
    def __init__(self, bbox: _Box, cell_shape: _YX) -> None:
        self.bbox = bbox
        self.cell_shape = cell_shape

    def index_of(self, *, x: int, y: int) -> _CellIndex:
        return _CellIndex(
            i=(y - self.bbox.y.start) // self.cell_shape.y,
            j=(x - self.bbox.x.start) // self.cell_shape.x,
        )

    def bbox_of(self, cell: _CellIndex) -> _Box:
        y_start = self.bbox.y.start + cell.i * self.cell_shape.y
        x_start = self.bbox.x.start + cell.j * self.cell_shape.x
        return _Box(
            y_start=y_start,
            y_stop=y_start + self.cell_shape.y,
            x_start=x_start,
            x_stop=x_start + self.cell_shape.x,
        )


class _CellField:
    unit = _Unit("dimensionless")

    def __call__(self, *, x: float, y: float) -> float:
        return 1.25


class _Contributions:
    def __init__(self) -> None:
        self.cell_i = np.zeros(3, dtype=np.uint16)
        self.cell_j = np.zeros(3, dtype=np.uint16)

    def __getitem__(self, name: str) -> np.ndarray:
        return getattr(self, name)

    def __len__(self) -> int:
        return len(self.cell_i)


class _Provenance:
    inputs = [object(), object()]
    contributions = _Contributions()

    def __getitem__(self, _cell):
        return self


class _SkyProjection:
    def __init__(self) -> None:
        self.calls = []

    def as_astropy(self, bbox):
        self.calls.append(bbox)
        return SimpleNamespace(bbox=bbox)


class _Patch:
    def model_dump(self, **kwargs):
        return {"index": {"i": 2, "j": 3}}


class _CellCoadd:
    def __init__(self, bbox: _Box) -> None:
        self.bbox = bbox
        grid_bbox = _Box(
            y_start=100,
            y_stop=120,
            x_start=200,
            x_stop=240,
        )
        self.grid = _Grid(grid_bbox, _YX(10, 10))
        self.bounds = SimpleNamespace(
            grid=self.grid,
            missing=frozenset({_CellIndex(i=1, j=2)}),
        )
        shape = tuple(bbox.shape)
        self.image = _Image(
            np.arange(np.prod(shape), dtype=np.float32).reshape(shape),
            unit="nJy",
        )
        self.variance = _Image(
            np.full(shape, 2.0, dtype=np.float32),
            unit="nJy2",
        )
        self.mask = _Mask(bbox)
        self.astropy_wcs = object()
        self.psf = _Psf(self.bounds)
        self.aperture_corrections = {"base_PsfFlux": _CellField()}
        self.provenance = _Provenance()
        self.band = "r"
        self.skymap = "synthetic_v1"
        self.tract = 42
        self.patch = _Patch()
        self.mask_fractions = {"SAT": object()}


class _MaskedImage:
    def __init__(self, bbox: _Box) -> None:
        self.bbox = bbox
        shape = tuple(bbox.shape)
        self.image = _Image(np.ones(shape, dtype=np.float32), unit="nJy")
        self.variance = _Image(
            np.full(shape, 3.0, dtype=np.float32),
            unit="nJy2",
        )
        self.mask = _Mask(bbox)
        self.astropy_wcs = object()

    def __getitem__(self, bbox: _Box):
        return type(self)(bbox)


class _Reader:
    def __init__(
        self,
        calls: dict,
        *,
        schema_name: str = "cell_coadd",
    ) -> None:
        self.calls = calls
        self.full_bbox = _Box(
            y_start=100,
            y_stop=105,
            x_start=200,
            x_stop=206,
        )
        self.info = SimpleNamespace(
            schema_url=(
                f"https://images.lsst.io/schemas/{schema_name}-1.0.0"
            ),
            schema_name=schema_name,
            schema_version="1.0.0",
            format_version=2,
        )
        self.metadata = {"instrument": "synthetic"}
        self.sky_projection = _SkyProjection()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def get_component(self, name: str, **kwargs):
        if name == "bbox":
            return self.full_bbox
        if name == "sky_projection":
            return self.sky_projection
        if name == "psf":
            return SimpleNamespace(
                bounds=SimpleNamespace(
                    cell_indices=lambda: iter([_CellIndex(i=0, j=0)])
                )
            )
        if name == "provenance":
            return _Provenance()
        raise AssertionError(name)

    def read(self, **kwargs):
        self.calls["read"] = kwargs
        return _CellCoadd(kwargs["bbox"] or self.full_bbox)


def _fake_archive_api(calls: dict, *, schema_name="cell_coadd"):
    def open_archive(path, cls, *, partial):
        calls["open"] = {
            "path": path,
            "cls": cls,
            "partial": partial,
        }
        return _Reader(calls, schema_name=schema_name)

    return rubin_image._LsstImagesAPI(
        images=SimpleNamespace(Box=_Box),
        cell_coadd_cls=_CellCoadd,
        open_archive=open_archive,
    )


def _simple_wcs_header() -> fits.Header:
    header = fits.Header()
    header["WCSAXES"] = 2
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CRVAL1"] = 10.0
    header["CRVAL2"] = -5.0
    header["CRPIX1"] = 1.0
    header["CRPIX2"] = 1.0
    header["CD1_1"] = 1.0e-3
    header["CD1_2"] = 0.0
    header["CD2_1"] = 0.0
    header["CD2_2"] = 1.0e-3
    return header


def _write_legacy_fits(
    path: Path,
    *,
    data_model=None,
    origin_yx: tuple[int, int] | None = None,
    mask=None,
) -> None:
    image = np.arange(30, dtype=np.float32).reshape(5, 6)
    variance = np.full(image.shape, 2.0, dtype=np.float32)
    if mask is None:
        mask = np.zeros(image.shape, dtype=np.int32)
        mask[1, 2] = 1
        mask[2, 3] = 1 << 5
    primary = fits.PrimaryHDU()
    primary.header["EXPTIME"] = 30.0
    if data_model is not None:
        primary.header["DATAMODL"] = data_model
    image_header = _simple_wcs_header()
    image_header["BUNIT"] = "nJy"
    variance_header = fits.Header()
    variance_header["BUNIT"] = "nJy2"
    mask_header = fits.Header()
    mask_header["MP_BAD"] = 0
    mask_header["MP_DETECTED"] = 5
    if origin_yx is not None:
        for header in (image_header, variance_header, mask_header):
            header["LTV1"] = -origin_yx[1]
            header["LTV2"] = -origin_yx[0]
    fits.HDUList(
        [
            primary,
            fits.ImageHDU(image, header=image_header, name="IMAGE"),
            fits.ImageHDU(mask, header=mask_header, name="MASK"),
            fits.ImageHDU(
                variance,
                header=variance_header,
                name="VARIANCE",
            ),
        ]
    ).writeto(path)


def test_legacy_view_preserves_planes_mask_bounds_units_and_wcs(
    tmp_path,
) -> None:
    path = tmp_path / "legacy.fits"
    _write_legacy_fits(path)
    section = (slice(1, 4), slice(2, 6))

    view = open_rubin_image(
        path,
        source_format="legacy-fits",
        section=section,
    )

    assert view.source_format == "legacy-fits"
    assert view.image.shape == (3, 4)
    assert view.variance.shape == (3, 4)
    assert view.bounds.to_dict() == {
        "y_start": 1,
        "y_stop": 4,
        "x_start": 2,
        "x_stop": 6,
        "shape_yx": [3, 4],
    }
    assert view.unit == "nJy"
    assert view.variance_unit == "nJy2"
    assert view.mask_plane_names == ("BAD", "DETECTED")
    assert view.selected_bad_mask_plane_names == ("BAD",)
    assert view.bad_mask[0, 0]
    assert not view.bad_mask[1, 1]
    assert view.get_mask_plane("DETECTED")[1, 1]
    assert view.mask_plane_schema["BAD"] == {"bit": 0}
    assert view.wcs_fidelity == "fits"

    full_wcs = WCS(_simple_wcs_header())
    expected = full_wcs.pixel_to_world_values(2.0, 1.0)
    actual = view.wcs.pixel_to_world_values(0.0, 0.0)
    assert np.allclose(actual, expected)
    assert read_rubin_image_bounds(
        path,
        source_format="legacy-fits",
    ).shape == (5, 6)


def test_legacy_view_rejects_non_two_dimensional_image(tmp_path) -> None:
    path = tmp_path / "rank-3.fits"
    image = np.ones((2, 3, 4), dtype=np.float32)
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(image, name="IMAGE"),
            fits.ImageHDU(
                np.zeros(image.shape, dtype=np.int16),
                name="MASK",
            ),
            fits.ImageHDU(image, name="VARIANCE"),
        ]
    ).writeto(path)

    with pytest.raises(
        RubinImageFormatError,
        match=r"IMAGE HDU must be two-dimensional, got \(2, 3, 4\)",
    ):
        open_rubin_image(path, source_format="legacy-fits")


def test_backend_choice_is_explicit_and_full_archive_does_not_fallback(
    tmp_path,
) -> None:
    path = tmp_path / "full-archive.fits"
    _write_legacy_fits(
        path,
        data_model=("https://images.lsst.io/schemas/cell_coadd-1.0.0"),
    )

    with pytest.raises(
        RubinImageFormatError,
        match="passed to the legacy-fits backend",
    ):
        open_rubin_image(path, source_format="legacy-fits")
    with pytest.raises(
        RubinImageFormatError,
        match="source_format must be one of",
    ):
        open_rubin_image(path, source_format="auto")


def test_full_archive_uses_cell_coadd_bbox_mask_wcs_and_psf(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "cell-coadd.fits"
    path.write_bytes(b"not opened by fake API")
    calls = {}
    monkeypatch.setattr(
        rubin_image,
        "_load_lsst_images_api",
        lambda: _fake_archive_api(calls),
    )

    view = open_rubin_image(
        path,
        source_format="lsst-images-archive",
        section=(slice(1, 4), slice(2, 6)),
    )

    assert calls["open"] == {
        "path": path,
        "cls": _CellCoadd,
        "partial": True,
    }
    read_bbox = calls["read"]["bbox"]
    assert (
        read_bbox.y.start,
        read_bbox.y.stop,
        read_bbox.x.start,
        read_bbox.x.stop,
    ) == (101, 104, 202, 206)
    assert calls["read"]["provenance"] is True
    assert view.source_format == "lsst-images-archive"
    assert view.bounds.shape == (3, 4)
    assert view.unit == "nJy"
    assert view.variance_unit == "nJy2"
    assert view.wcs_fidelity == "exact-lsst-images"
    assert view.requirements_met is True
    assert view.product_role == "reference-evaluation-only"
    assert view.reconstruction_ready is False
    assert view.archive_info == {
        "schema_url": ("https://images.lsst.io/schemas/cell_coadd-1.0.0"),
        "schema_name": "cell_coadd",
        "schema_version": "1.0.0",
        "format_version": 2,
        "lsst_images_version": "30.0.10",
    }
    assert view.archive_metadata == {"instrument": "synthetic"}
    assert view.selected_bad_mask_plane_names == ("BAD", "SATURATED")
    assert view.bad_mask[0, 0]
    assert view.bad_mask[0, -1]
    assert view.get_mask_plane("DETECTED")[-1, -1]
    assert view.mask_plane_schema["BAD"] == {
        "index": 0,
        "mask": 1,
        "word": 0,
        "word_mask": 1,
        "bit": 0,
    }
    assert view.mask_plane_schema["DETECTED"] == {
        "index": 8,
        "mask": 256,
        "word": 1,
        "word_mask": 1,
        "bit": 0,
    }
    assert view.provenance is view.native.provenance
    assert view.cell_metadata["tract"] == 42
    assert view.cell_metadata["provenance_present"] is True
    assert view.cell_metadata["provenance_input_count"] == 2
    assert view.cell_metadata["provenance_contribution_count"] == 3
    assert view.cell_metadata["aperture_correction_names"] == ["base_PsfFlux"]

    kernel = view.sample_psf_kernel(5)
    assert kernel.shape == (5, 5)
    assert np.isclose(kernel.sum(), 1.0)
    assert view.psf.samples == [(203.5, 102.0)]
    assert view.sample_aperture_corrections_at(
        x=203.5,
        y=102.0,
    ) == {
        "base_PsfFlux": {
            "value": 1.25,
            "unit": "dimensionless",
            "available": True,
        }
    }
    cell = view.cell_diagnostics_at(x=203.5, y=102.0)
    assert cell["cell_index_ij"] == [0, 0]
    assert cell["missing"] is False
    assert cell["provenance"] == {
        "present": True,
        "available_for_cell": True,
        "input_count": 2,
        "contribution_count": 3,
    }


def test_full_archive_warp_bridge_preserves_bounded_planes_and_context(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "cell-coadd.fits"
    path.write_bytes(b"not opened by fake API")
    calls = {}
    monkeypatch.setattr(
        rubin_image,
        "_load_lsst_images_api",
        lambda: _fake_archive_api(calls),
    )
    view = open_rubin_image(
        path,
        source_format="lsst-images-archive",
        section=(slice(1, 4), slice(2, 6)),
    )

    planes = rubin_warp.rubin_view_to_warp_planes(view)

    assert planes.image is view.image
    assert planes.variance is view.variance
    assert planes.image.shape == planes.variance.shape == planes.mask.shape
    assert planes.mask.dtype == np.uint16
    assert planes.mask[0, 0] == 1
    assert planes.mask[0, -1] == 4
    assert planes.mask[-1, -1] == 256
    assert planes.bounds is view.bounds
    assert planes.wcs is view.wcs
    assert planes.wcs_fidelity == "exact-lsst-images"
    assert planes.source_origin_xy == (202, 101)
    assert planes.local_wcs_source_origin_xy == (0, 0)
    assert planes.mask_plane_schema == view.mask_plane_schema
    assert planes.mask_plane_schema is not view.mask_plane_schema
    assert planes.cell_key == {
        "skymap": "synthetic_v1",
        "tract": 42,
        "patch": {"index": {"i": 2, "j": 3}},
        "band": "r",
        "cell_index_ij": [0, 0],
    }
    assert planes.cell_diagnostics["missing"] is False
    assert planes.psf is view.psf
    assert (
        planes.aperture_corrections["base_PsfFlux"]
        is (view.aperture_corrections["base_PsfFlux"])
    )
    assert planes.provenance is view.provenance
    assert planes.product_role == "reference-evaluation-only"
    assert planes.reconstruction_ready is False
    assert planes.archive_metadata == {"instrument": "synthetic"}
    planes.archive_metadata["note"] = "unrelated-archive-note"
    metadata = planes.metadata()
    assert "archive_metadata" not in metadata
    assert "unrelated-archive-note" not in str(metadata)
    assert planes.archive_metadata["note"] == "unrelated-archive-note"
    assert metadata["archive_info"] == view.archive_info
    assert metadata["cell_metadata"] == view.cell_metadata


def test_warp_bridge_delegates_once_and_keeps_rubin_metadata(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "legacy.fits"
    _write_legacy_fits(path)
    view = open_rubin_image(
        path,
        source_format="legacy-fits",
        section=(slice(1, 4), slice(2, 6)),
    )
    calls = {}
    expected_prepared = object()

    def fake_warp(image, variance, mask, spec, *, backend, prepared):
        calls.update(
            {
                "image": image,
                "variance": variance,
                "mask": mask,
                "spec": spec,
                "backend": backend,
                "prepared": prepared,
            }
        )
        return SimpleNamespace(
            image=image + 1,
            variance=variance + 2,
            mask=mask,
            bbox="output-bbox",
            backend=backend,
            interpolation="bilinear",
            metadata={"variance_propagation": "diagonal"},
        )

    monkeypatch.setattr(
        rubin_warp,
        "_load_xrep_api",
        lambda: rubin_warp._XRepAPI(
            masked_reprojection_result_cls=SimpleNamespace,
            reproject_masked_array=fake_warp,
        ),
    )

    result = rubin_warp.warp_rubin_image_view(
        view,
        "warp-spec",
        backend="cpu",
        prepared=expected_prepared,
    )

    assert calls["image"] is view.image
    assert calls["variance"] is view.variance
    assert calls["spec"] == "warp-spec"
    assert calls["backend"] == "cpu"
    assert calls["prepared"] is expected_prepared
    assert np.array_equal(calls["mask"], np.asarray(view.native_mask))
    assert np.array_equal(result.image, view.image + 1)
    assert np.array_equal(result.variance, view.variance + 2)
    assert result.bbox == "output-bbox"
    assert result.metadata["variance_propagation"] == "diagonal"
    rubin_metadata = result.metadata["rubin_image"]
    assert "source_path" not in rubin_metadata
    assert rubin_metadata["wcs_fidelity"] == "fits"
    assert rubin_metadata["mask_plane_schema"] == view.mask_plane_schema
    assert rubin_metadata["product_role"] == "legacy-image-unspecified"
    assert rubin_metadata["reconstruction_ready"] is True


@pytest.mark.parametrize("backend", ["cpu", "torch"])
def test_warp_bridge_matches_xrep_array_api(
    tmp_path,
    backend,
) -> None:
    if backend == "torch":
        pytest.importorskip("torch")

    from cuphoton.xrep import (
        BBox,
        MaskedReprojectionResult,
        ReprojectionSpec,
    )

    path = tmp_path / "legacy.fits"
    _write_legacy_fits(path)
    view = open_rubin_image(
        path,
        source_format="legacy-fits",
        section=(slice(1, 4), slice(2, 6)),
    )
    spec = ReprojectionSpec(
        mapping=lambda coords: np.asarray(coords, dtype=np.float64),
        output_bbox=BBox(
            min_x=0,
            min_y=0,
            width=view.image.shape[1],
            height=view.image.shape[0],
        ),
        interpolation="bilinear",
        mapping_grid_step=1,
        area_scaling=False,
    )

    result = rubin_warp.warp_rubin_image_view(view, spec, backend=backend)

    assert isinstance(result.xrep_result, MaskedReprojectionResult)
    assert result.source.image is view.image
    assert result.source.variance is view.variance
    assert result.backend == backend
    assert np.array_equal(result.image, view.image)
    assert np.array_equal(result.variance, view.variance)
    assert np.array_equal(result.mask, np.asarray(view.native_mask))
    assert result.mask.dtype.kind == "u"
    assert (
        result.mask.dtype.itemsize
        == np.asarray(view.native_mask).dtype.itemsize
    )


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        (
            {
                "index": 0,
                "mask": 1,
                "word": 0,
                "word_mask": 1,
                "bit": 0,
            },
            "duplicate packed bit indices",
        ),
        (
            {
                "index": 16,
                "mask": 1 << 16,
                "word": 2,
                "word_mask": 1,
                "bit": 0,
            },
            "native word geometry",
        ),
        (
            {
                "index": 9,
                "mask": 1 << 9,
                "word": 1,
                "word_mask": 1,
                "bit": 0,
            },
            "native word geometry",
        ),
    ],
)
def test_warp_bridge_validates_authoritative_native_mask_schema(
    tmp_path,
    monkeypatch,
    replacement,
    message,
) -> None:
    path = tmp_path / "cell-coadd.fits"
    path.write_bytes(b"not opened by fake API")
    calls = {}
    monkeypatch.setattr(
        rubin_image,
        "_load_lsst_images_api",
        lambda: _fake_archive_api(calls),
    )
    view = open_rubin_image(
        path,
        source_format="lsst-images-archive",
        section=(slice(1, 4), slice(2, 6)),
    )
    schema = {
        name: dict(entry) for name, entry in view.mask_plane_schema.items()
    }
    schema["DETECTED"] = replacement
    invalid = replace(view, mask_plane_schema=schema)

    with pytest.raises(RubinImageFormatError, match=message):
        rubin_warp.rubin_view_to_warp_planes(invalid)


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ({"BAD": {"bit": 0}, "DETECTED": {"bit": 0}}, "duplicate"),
        ({"BAD": {"bit": 0}, "DETECTED": {"bit": 32}}, "capacity"),
    ],
)
def test_warp_bridge_validates_legacy_mask_schema(
    tmp_path,
    replacement,
    message,
) -> None:
    path = tmp_path / "legacy.fits"
    _write_legacy_fits(path)
    view = open_rubin_image(path, source_format="legacy-fits")
    invalid = replace(view, mask_plane_schema=replacement)

    with pytest.raises(RubinImageFormatError, match=message):
        rubin_warp.rubin_view_to_warp_planes(invalid)


@pytest.mark.parametrize("dtype", [np.int16, np.int32, np.int64])
def test_warp_bridge_preserves_signed_fits_mask_bits(tmp_path, dtype) -> None:
    mask = np.zeros((5, 6), dtype=dtype)
    mask[1, 2] = np.iinfo(dtype).min
    mask[2, 3] = -1
    path = tmp_path / "legacy.fits"
    _write_legacy_fits(path, mask=mask)
    view = open_rubin_image(path, source_format="legacy-fits")
    assert view.native_mask.dtype.kind == "i"
    assert view.native_mask[1, 2] < 0

    planes = rubin_warp.rubin_view_to_warp_planes(view)

    bits = mask.dtype.itemsize * 8
    assert planes.mask.dtype.kind == "u"
    assert planes.mask.dtype.itemsize == mask.dtype.itemsize
    assert planes.mask[1, 2] == 1 << (bits - 1)
    assert planes.mask[2, 3] == (1 << bits) - 1
    assert np.count_nonzero(planes.mask) == 2


def test_warp_bridge_rejects_floating_point_masks(tmp_path) -> None:
    path = tmp_path / "legacy.fits"
    _write_legacy_fits(path)
    view = open_rubin_image(path, source_format="legacy-fits")
    invalid = replace(view, native_mask=view.native_mask.astype(np.float32))

    with pytest.raises(RubinImageFormatError, match="integer dtype"):
        rubin_warp.rubin_view_to_warp_planes(invalid)


def test_cutdown_warp_bridge_is_bounded_and_transport_only(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "cutdown.fits"
    _write_legacy_fits(path)

    class MaskedImageFactory:
        @classmethod
        def from_hdu_list(cls, _hdul):
            return _MaskedImage(
                _Box(
                    y_start=10,
                    y_stop=15,
                    x_start=20,
                    x_stop=26,
                )
            )

    monkeypatch.setattr(
        rubin_image,
        "_load_lsst_images_api",
        lambda: rubin_image._LsstImagesAPI(
            images=SimpleNamespace(
                Box=_Box,
                MaskedImage=MaskedImageFactory,
            ),
            cell_coadd_cls=_CellCoadd,
            open_archive=None,
        ),
    )
    view = open_rubin_image(
        path,
        source_format="lsst-images-cutdown",
        section=(slice(1, 4), slice(2, 5)),
    )

    planes = rubin_warp.rubin_view_to_warp_planes(view)

    assert planes.image.shape == planes.variance.shape == (3, 3)
    assert planes.mask.shape == (3, 3)
    assert planes.bounds.to_dict()["shape_yx"] == [3, 3]
    assert planes.wcs is view.wcs
    assert planes.wcs_fidelity == "fits-cutdown"
    assert planes.cell_key is None
    assert planes.psf is None
    assert planes.aperture_corrections == {}
    assert planes.provenance is None
    assert planes.product_role == "pixel-transport-only"
    assert planes.reconstruction_ready is False


@pytest.mark.parametrize("missing", ["variance", "wcs"])
def test_warp_bridge_fails_closed_without_variance_or_wcs(
    tmp_path,
    missing,
) -> None:
    path = tmp_path / "legacy.fits"
    _write_legacy_fits(path)
    view = open_rubin_image(path, source_format="legacy-fits")
    broken = replace(view, **{missing: None})

    with pytest.raises(
        RubinImageFormatError,
        match=("variance plane" if missing == "variance" else "WCS"),
    ):
        rubin_warp.rubin_view_to_warp_planes(broken)


def test_warp_bridge_rejects_native_mask_larger_than_uint64(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "cutdown.fits"
    _write_legacy_fits(path)

    class OversizedMaskedImage(_MaskedImage):
        def __init__(self, bbox):
            super().__init__(bbox)
            self.mask.array = np.zeros(
                tuple(bbox.shape) + (9,),
                dtype=np.uint8,
            )

    class MaskedImageFactory:
        @classmethod
        def from_hdu_list(cls, _hdul):
            return OversizedMaskedImage(
                _Box(
                    y_start=10,
                    y_stop=15,
                    x_start=20,
                    x_stop=26,
                )
            )

    monkeypatch.setattr(
        rubin_image,
        "_load_lsst_images_api",
        lambda: rubin_image._LsstImagesAPI(
            images=SimpleNamespace(
                Box=_Box,
                MaskedImage=MaskedImageFactory,
            ),
            cell_coadd_cls=_CellCoadd,
            open_archive=None,
        ),
    )
    view = open_rubin_image(
        path,
        source_format="lsst-images-cutdown",
    )

    with pytest.raises(RubinImageFormatError, match="72 bits"):
        rubin_warp.rubin_view_to_warp_planes(view)


def test_warp_bridge_fails_lazily_when_xrep_api_is_missing(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        rubin_warp,
        "import_module",
        lambda _name: SimpleNamespace(),
    )

    with pytest.raises(
        rubin_warp.RubinWarpDependencyError,
        match="MaskedReprojectionResult and reproject_masked_array",
    ):
        rubin_warp._load_xrep_api()


def test_full_archive_geometry_is_metadata_only_and_native(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "cell-coadd.fits"
    path.write_bytes(b"not opened by fake API")
    calls = {}
    api = _fake_archive_api(calls)
    monkeypatch.setattr(rubin_image, "_load_lsst_images_api", lambda: api)

    geometry = inspect_rubin_image_geometry(
        path,
        source_format="lsst-images-archive",
    )

    assert geometry.bounds.to_dict() == {
        "y_start": 100,
        "y_stop": 105,
        "x_start": 200,
        "x_stop": 206,
        "shape_yx": [5, 6],
    }
    assert geometry.wcs_fidelity == "exact-lsst-images"
    assert geometry.wcs.bbox is not None
    assert "read" not in calls


def test_full_archive_rejects_non_cell_coadd_schema(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "masked-image.fits"
    path.write_bytes(b"not opened by fake API")
    calls = {}
    monkeypatch.setattr(
        rubin_image,
        "_load_lsst_images_api",
        lambda: _fake_archive_api(calls, schema_name="masked_image"),
    )

    with pytest.raises(
        RubinImageFormatError,
        match="Expected an lsst.images CellCoadd archive",
    ):
        open_rubin_image(
            path,
            source_format="lsst-images-archive",
        )
    assert "read" not in calls


def test_full_archive_store_rejects_coadd_as_exposure_stack(
    tmp_path,
    monkeypatch,
) -> None:
    root = tmp_path / "products"
    root.mkdir()
    relative_path = "deep_coadd/cell-coadd.fits"
    path = root / relative_path
    path.parent.mkdir()
    path.write_bytes(b"not opened by fake API")
    calls = {}
    monkeypatch.setattr(
        rubin_image,
        "_load_lsst_images_api",
        lambda: _fake_archive_api(calls),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "source_kind": "rubin-local-image-products",
                "rubin_image_format": "lsst-images-archive",
                "file_count": 1,
                "files": [
                    {
                        "group": "stack",
                        "product": "deep_coadd",
                        "rubin_image_format": "lsst-images-archive",
                        "relative_path": relative_path,
                        "band": "r",
                    }
                ],
            }
        )
    )
    output = tmp_path / "store"

    with pytest.raises(
        ValueError,
        match="reference/evaluation products",
    ):
        build_rubin_store(
            manifest_path=manifest_path,
            input_root=root,
            output_dir=output,
            group="stack",
            product="deep_coadd",
            psf_size=5,
        )
    assert not output.exists()
    assert "read" not in calls


def test_direct_stack_reader_rejects_legacy_coadd_rows(tmp_path) -> None:
    with pytest.raises(
        ValueError,
        match="reference/evaluation products",
    ):
        read_manifest_stack(
            [
                {
                    "product": "deep_coadd",
                    "relative_path": "deep_coadd/coadd.fits",
                }
            ],
            input_root=tmp_path,
            section=None,
            mask_policy="rubin-bad-planes",
            psf_size=5,
        )


def test_full_archive_coadd_semantics_use_native_adapter_components(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "cell-coadd.fits"
    path.write_bytes(b"not opened by fake API")
    calls = {}
    monkeypatch.setattr(
        rubin_image,
        "_load_lsst_images_api",
        lambda: _fake_archive_api(calls),
    )

    result = summarize_rubin_coadd_semantics(
        path,
        rubin_image_format="lsst-images-archive",
        require_coadd_psf=True,
        require_coadd_inputs=True,
    )

    assert result["ok"] is True
    assert result["requirements_met"] is True
    assert result["product_role"] == "reference-evaluation-only"
    assert result["reconstruction_contract"]["reconstruction_ready"] is False
    assert result["coadd_inputs"] == {
        "present": True,
        "source": "lsst.images.CellCoadd.provenance",
    }
    assert result["coadd_psf"]["native_cell_psf"] is True
    assert result["coadd_psf"]["kernel_normalized"] is True
    assert result["cell_metadata"]["aperture_correction_names"] == [
        "base_PsfFlux"
    ]


def test_cutdown_is_bounded_but_never_requirements_met(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "cutdown.fits"
    _write_legacy_fits(path)
    calls = {}

    class MaskedImageFactory:
        @classmethod
        def from_hdu_list(cls, hdul):
            calls["hdu_names"] = [hdu.name for hdu in hdul]
            return _MaskedImage(
                _Box(
                    y_start=10,
                    y_stop=15,
                    x_start=20,
                    x_stop=26,
                )
            )

    api = rubin_image._LsstImagesAPI(
        images=SimpleNamespace(
            Box=_Box,
            MaskedImage=MaskedImageFactory,
        ),
        cell_coadd_cls=_CellCoadd,
        open_archive=None,
    )
    monkeypatch.setattr(
        rubin_image,
        "_load_lsst_images_api",
        lambda: api,
    )

    view = open_rubin_image(
        path,
        source_format="lsst-images-cutdown",
        section=(slice(1, 4), slice(2, 5)),
    )

    assert calls["hdu_names"] == [
        "PRIMARY",
        "IMAGE",
        "MASK",
        "VARIANCE",
    ]
    assert view.bounds.to_dict() == {
        "y_start": 11,
        "y_stop": 14,
        "x_start": 22,
        "x_stop": 25,
        "shape_yx": [3, 3],
    }
    assert view.wcs_fidelity == "fits-cutdown"
    assert view.requirements_met is False
    assert view.product_role == "pixel-transport-only"
    assert view.reconstruction_ready is False
    assert view.psf is None
    assert view.provenance is None
    assert view.cell_metadata == {
        "psf_present": False,
        "aperture_corrections_present": False,
        "aperture_correction_names": [],
        "provenance_present": False,
        "provenance_input_count": None,
        "provenance_contribution_count": None,
        "cutdown_metadata_complete": False,
    }


def test_cutdown_accepts_lsst_images_datamodel_header(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "cutdown.fits"
    _write_legacy_fits(
        path,
        data_model="https://images.lsst.io/schemas/cell_coadd-1.0.0",
    )

    class MaskedImageFactory:
        @classmethod
        def from_hdu_list(cls, _hdul):
            return _MaskedImage(
                _Box(
                    y_start=10,
                    y_stop=15,
                    x_start=20,
                    x_stop=26,
                )
            )

    monkeypatch.setattr(
        rubin_image,
        "_load_lsst_images_api",
        lambda: rubin_image._LsstImagesAPI(
            images=SimpleNamespace(
                Box=_Box,
                MaskedImage=MaskedImageFactory,
            ),
            cell_coadd_cls=_CellCoadd,
            open_archive=None,
        ),
    )

    view = open_rubin_image(
        path,
        source_format="lsst-images-cutdown",
    )

    assert view.source_format == "lsst-images-cutdown"
    assert view.requirements_met is False


def test_cutdown_bounds_preserve_nonzero_absolute_origin(tmp_path) -> None:
    path = tmp_path / "cutdown.fits"
    _write_legacy_fits(
        path,
        data_model="https://images.lsst.io/schemas/cell_coadd-1.0.0",
        origin_yx=(1234, 5678),
    )

    bounds = read_rubin_image_bounds(
        path,
        source_format="lsst-images-cutdown",
    )
    geometry = inspect_rubin_image_geometry(
        path,
        source_format="lsst-images-cutdown",
    )

    assert bounds.to_dict() == {
        "y_start": 1234,
        "y_stop": 1239,
        "x_start": 5678,
        "x_stop": 5684,
        "shape_yx": [5, 6],
    }
    assert geometry.bounds == bounds
    assert geometry.wcs_fidelity == "fits-cutdown"


def test_cutdown_cannot_enter_exposure_reconstruction(tmp_path) -> None:
    root = tmp_path / "products"
    root.mkdir()
    relative_path = "visit_image/cutdown.fits"
    path = root / relative_path
    path.parent.mkdir()
    _write_legacy_fits(
        path,
        data_model="https://images.lsst.io/schemas/cell_coadd-1.0.0",
        origin_yx=(10, 20),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "rubin_image_format": "lsst-images-cutdown",
                "files": [
                    {
                        "group": "stack",
                        "product": "visit_image",
                        "relative_path": relative_path,
                    }
                ],
            }
        )
    )
    output = tmp_path / "store"

    with pytest.raises(ValueError, match="pixel-transport-only"):
        build_rubin_store(
            manifest_path=manifest_path,
            input_root=root,
            output_dir=output,
            group="stack",
            product="visit_image",
        )
    assert not output.exists()


def test_cutdown_rejects_full_indexed_archive(tmp_path, monkeypatch) -> None:
    path = tmp_path / "archive.fits"
    primary = fits.PrimaryHDU()
    primary.header["DATAMODL"] = (
        "https://images.lsst.io/schemas/cell_coadd-1.0.0"
    )
    fits.HDUList(
        [
            primary,
            fits.BinTableHDU(name="JSON"),
            fits.BinTableHDU(name="INDEX"),
        ]
    ).writeto(path)
    monkeypatch.setattr(
        rubin_image,
        "_load_lsst_images_api",
        lambda: rubin_image._LsstImagesAPI(
            images=SimpleNamespace(MaskedImage=object),
            cell_coadd_cls=_CellCoadd,
            open_archive=None,
        ),
    )

    with pytest.raises(
        RubinImageFormatError,
        match="not declared as a MaskedImage",
    ):
        open_rubin_image(
            path,
            source_format="lsst-images-cutdown",
        )


def test_cutdown_accepts_stale_container_cards_when_index_hdus_are_removed(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "archive.fits"
    primary = fits.PrimaryHDU()
    primary.header["DATAMODL"] = (
        "https://images.lsst.io/schemas/cell_coadd-1.0.0"
    )
    for name in ("INDXADDR", "INDXSIZE", "JSONADDR", "JSONSIZE"):
        primary.header[name] = 1
    fits.HDUList(
        [
            primary,
            fits.ImageHDU(np.ones((2, 2)), name="IMAGE"),
            fits.ImageHDU(np.zeros((2, 2), dtype=np.int16), name="MASK"),
            fits.ImageHDU(np.ones((2, 2)), name="VARIANCE"),
        ]
    ).writeto(path)

    class MaskedImageFactory:
        @classmethod
        def from_hdu_list(cls, _hdul):
            return _MaskedImage(
                _Box(
                    y_start=10,
                    y_stop=12,
                    x_start=20,
                    x_stop=22,
                )
            )

    monkeypatch.setattr(
        rubin_image,
        "_load_lsst_images_api",
        lambda: rubin_image._LsstImagesAPI(
            images=SimpleNamespace(MaskedImage=MaskedImageFactory),
            cell_coadd_cls=_CellCoadd,
            open_archive=None,
        ),
    )

    view = open_rubin_image(
        path,
        source_format="lsst-images-cutdown",
    )
    assert view.bounds.shape == (2, 2)
    assert view.product_role == "pixel-transport-only"


def _write_real_lsst_cell_coadd(path: Path) -> None:
    images = pytest.importorskip("lsst.images")
    cells = pytest.importorskip("lsst.images.cells")
    serialization = pytest.importorskip("lsst.images.serialization")
    tests_api = pytest.importorskip("lsst.images.tests")
    units = pytest.importorskip("astropy.units")
    assert images.__version__ == "30.0.10"

    bbox = images.Box.from_shape(
        (300, 300),
        start=images.YX(y=1000, x=2000),
    )
    grid = cells.CellGrid(
        bbox=bbox,
        cell_shape=images.YX(y=150, x=150),
    )
    missing = cells.CellIJ(i=0, j=1)
    bounds = cells.CellGridBounds(
        grid=grid,
        bbox=bbox,
        missing=frozenset({missing}),
    )
    yy, xx = np.mgrid[-2:3, -2:3]
    kernel = np.exp(-(xx * xx + yy * yy) / 2.0)
    psf_array = np.stack(
        [
            np.stack([kernel, np.full_like(kernel, np.nan)]),
            np.stack([kernel * 2.0, kernel * 3.0]),
        ]
    )
    psf = cells.CellPointSpreadFunction(psf_array, bounds=bounds)
    tract_frame = images.TractFrame(
        skymap="test_v1",
        tract=42,
        bbox=bbox,
    )
    sky_projection = tests_api.make_random_sky_projection(
        np.random.default_rng(17),
        tract_frame,
        bbox,
    )
    image = images.Image(
        np.arange(300 * 300, dtype=np.float32).reshape(300, 300),
        bbox=bbox,
        unit=units.nJy,
        sky_projection=sky_projection,
    )
    variance = images.Image(
        np.full((300, 300), 4.0, dtype=np.float32),
        bbox=bbox,
        unit=units.nJy**2,
    )
    schema = images.MaskSchema(
        [
            images.MaskPlane("BAD", "bad pixel"),
            images.MaskPlane("DETECTED", "detected source"),
        ]
    )
    mask = images.Mask(schema=schema, bbox=bbox)
    bad = np.zeros((300, 300), dtype=bool)
    bad[0, 0] = True
    mask.set("BAD", bad)
    patch = cells.PatchDefinition(
        id=42,
        index=images.YX(y=2, x=3),
        inner_bbox=bbox,
        cells=grid,
    )
    inputs = cells.CoaddProvenance.make_empty_input_table(1)
    inputs["instrument"][0] = "LSSTCam"
    inputs["visit"][0] = 200001010001
    inputs["detector"][0] = 42
    inputs["physical_filter"][0] = "r_03"
    inputs["day_obs"][0] = 20000101
    inputs["polygon"][0] = images.Polygon.from_box(bbox)
    contributions = cells.CoaddProvenance.make_empty_contribution_table(1)
    contributions["cell_i"][0] = 0
    contributions["cell_j"][0] = 0
    contributions["instrument"][0] = "LSSTCam"
    contributions["visit"][0] = 200001010001
    contributions["detector"][0] = 42
    contributions["overlaps_center"][0] = True
    contributions["overlap_fraction"][0] = 1.0
    contributions["unmasked_fraction"][0] = 0.95
    contributions["weight"][0] = 1.0
    contributions["psf_shape_xx"][0] = 1.0
    contributions["psf_shape_yy"][0] = 1.0
    contributions["psf_shape_xy"][0] = 0.0
    contributions["psf_shape_flag"][0] = False
    provenance = cells.CoaddProvenance(
        inputs=inputs,
        contributions=contributions,
    )
    aperture_corrections = {
        "base_PsfFlux": cells.CellField(
            bounds,
            np.array([[1.1, np.nan], [1.2, 1.3]], dtype=np.float64),
        )
    }
    coadd = cells.CellCoadd(
        image,
        mask=mask,
        variance=variance,
        sky_projection=sky_projection,
        band="r",
        psf=psf,
        aperture_corrections=aperture_corrections,
        patch=patch,
        provenance=provenance,
    )
    serialization.write_archive(coadd, path)


def _write_real_lsst_cutdown(
    path: Path,
    *,
    indexed: bool = False,
) -> None:
    images = pytest.importorskip("lsst.images")
    tests_api = pytest.importorskip("lsst.images.tests")
    units = pytest.importorskip("astropy.units")
    assert images.__version__ == "30.0.10"

    shape = (20, 25)
    bbox = images.Box.from_shape(
        shape,
        start=images.YX(y=1234, x=5678),
    )
    projection = tests_api.make_random_sky_projection(
        np.random.default_rng(23),
        images.GeneralFrame(unit=units.pix),
        bbox,
    )
    image = images.Image(
        np.arange(np.prod(shape), dtype=np.float32).reshape(shape),
        bbox=bbox,
        unit=units.nJy,
        sky_projection=projection,
    )
    schema = images.MaskSchema(
        [
            images.MaskPlane("BAD", "bad pixel"),
            images.MaskPlane("DETECTED", "detected source"),
        ]
    )
    masked_image = images.MaskedImage(image, mask_schema=schema)
    bad = np.zeros(shape, dtype=bool)
    bad[2, 3] = True
    masked_image.mask.set("BAD", bad)
    masked_image.variance.array[...] = 9.0
    if indexed:
        masked_image.write(path)
        return
    full_path = path.with_name("full-masked-image.fits")
    masked_image.write(full_path)
    with fits.open(full_path) as hdul:
        cutdown_hdus = [fits.PrimaryHDU(header=hdul[0].header.copy())]
        for name in ("IMAGE", "MASK", "VARIANCE"):
            source = hdul[name]
            cutdown_hdus.append(
                fits.ImageHDU(
                    data=np.asarray(source.data),
                    header=source.header.copy(),
                    name=name,
                )
            )
    fits.HDUList(cutdown_hdus).writeto(path)


def test_real_lsst_images_cell_coadd_archive_and_missing_cell(
    tmp_path,
) -> None:
    path = tmp_path / "cell-coadd.fits"
    _write_real_lsst_cell_coadd(path)

    geometry = inspect_rubin_image_geometry(
        path,
        source_format="lsst-images-archive",
    )
    view = open_rubin_image(
        path,
        source_format="lsst-images-archive",
        section=(slice(10, 30), slice(10, 30)),
    )

    assert geometry.bounds.to_dict() == {
        "y_start": 1000,
        "y_stop": 1300,
        "x_start": 2000,
        "x_stop": 2300,
        "shape_yx": [300, 300],
    }
    assert view.bounds.to_dict() == {
        "y_start": 1010,
        "y_stop": 1030,
        "x_start": 2010,
        "x_stop": 2030,
        "shape_yx": [20, 20],
    }
    assert view.image.shape == view.variance.shape == view.bad_mask.shape
    assert view.unit == "nJy"
    assert view.variance_unit == "nJy2"
    assert view.wcs_fidelity == "exact-lsst-images"
    assert view.reconstruction_ready is False
    expected_world = geometry.wcs.pixel_to_world_values(10.0, 10.0)
    actual_world = view.wcs.pixel_to_world_values(0.0, 0.0)
    assert np.allclose(actual_world, expected_world)
    kernel = view.sample_psf_kernel_at(x=2015.0, y=1015.0, size=7)
    assert kernel.shape == (7, 7)
    assert np.isfinite(kernel).all()
    assert np.isclose(kernel.sum(), 1.0)
    assert view.sample_aperture_corrections_at(
        x=2015.0,
        y=1015.0,
    )["base_PsfFlux"]["value"] == pytest.approx(1.1)
    diagnostics = view.cell_diagnostics_at(x=2015.0, y=1015.0)
    assert diagnostics["cell_index_ij"] == [0, 0]
    assert diagnostics["missing"] is False
    assert diagnostics["provenance"]["present"] is True
    empty_provenance_view = open_rubin_image(
        path,
        source_format="lsst-images-archive",
        section=(slice(160, 170), slice(10, 20)),
    )
    empty_provenance = empty_provenance_view.cell_diagnostics_at(
        x=2015.0,
        y=1165.0,
    )
    assert empty_provenance["missing"] is False
    assert empty_provenance["provenance"] == {
        "present": True,
        "available_for_cell": True,
        "input_count": 0,
        "contribution_count": 0,
    }
    planes = rubin_warp.rubin_view_to_warp_planes(view)
    assert planes.image.shape == planes.variance.shape == planes.mask.shape
    assert planes.mask.dtype == np.uint8
    assert np.count_nonzero(planes.mask) == 0
    assert planes.wcs is view.wcs
    assert planes.source_origin_xy == (2010, 1010)
    assert planes.local_wcs_source_origin_xy == (0, 0)
    assert planes.cell_key["cell_index_ij"] == [0, 0]
    assert planes.psf is view.psf
    assert planes.provenance is view.provenance
    assert planes.reconstruction_ready is False

    missing_view = open_rubin_image(
        path,
        source_format="lsst-images-archive",
        section=(slice(10, 20), slice(160, 170)),
    )
    missing = missing_view.cell_diagnostics_at(x=2165.0, y=1015.0)
    assert missing["cell_index_ij"] == [0, 1]
    assert missing["missing"] is True
    assert missing["psf_available"] is False
    with pytest.raises(RubinImageFormatError, match="cell may be missing"):
        missing_view.sample_psf_kernel_at(
            x=2165.0,
            y=1015.0,
            size=7,
        )


def test_real_lsst_images_datamodel_cutdown_is_transport_only(
    tmp_path,
) -> None:
    path = tmp_path / "cutdown.fits"
    _write_real_lsst_cutdown(path)

    with fits.open(path) as hdul:
        assert "DATAMODL" in hdul[0].header
        assert {"JSON", "INDEX"}.isdisjoint(
            {str(hdu.name).upper() for hdu in hdul}
        )
    bounds = read_rubin_image_bounds(
        path,
        source_format="lsst-images-cutdown",
    )
    view = open_rubin_image(
        path,
        source_format="lsst-images-cutdown",
        section=(slice(2, 7), slice(3, 9)),
    )

    assert bounds.to_dict() == {
        "y_start": 1234,
        "y_stop": 1254,
        "x_start": 5678,
        "x_stop": 5703,
        "shape_yx": [20, 25],
    }
    assert view.bounds.to_dict() == {
        "y_start": 1236,
        "y_stop": 1241,
        "x_start": 5681,
        "x_stop": 5687,
        "shape_yx": [5, 6],
    }
    assert view.bad_mask[0, 0]
    assert view.product_role == "pixel-transport-only"
    assert view.reconstruction_ready is False
    assert view.psf is None
    assert view.provenance is None
    planes = rubin_warp.rubin_view_to_warp_planes(view)
    assert planes.image.shape == planes.variance.shape == planes.mask.shape
    assert planes.mask[0, 0] == 1
    assert planes.wcs is view.wcs
    assert planes.source_origin_xy == (5681, 1236)
    assert planes.local_wcs_source_origin_xy == (0, 0)
    assert planes.cell_key is None
    assert planes.reconstruction_ready is False


def test_real_indexed_masked_image_cutdown_is_transport_only(
    tmp_path,
) -> None:
    path = tmp_path / "indexed-cutdown.fits"
    _write_real_lsst_cutdown(path, indexed=True)

    with fits.open(path) as hdul:
        data_model = str(hdul[0].header["DATAMODL"])
        names = {str(hdu.name).upper() for hdu in hdul}
        assert "/masked_image-" in data_model
        assert {"IMAGE", "MASK", "VARIANCE", "JSON", "INDEX"} <= names
        assert {"INDXADDR", "INDXSIZE", "JSONADDR", "JSONSIZE"} <= set(
            hdul[0].header
        )

    view = open_rubin_image(
        path,
        source_format="lsst-images-cutdown",
        section=(slice(2, 7), slice(3, 9)),
    )

    assert view.bounds.to_dict() == {
        "y_start": 1236,
        "y_stop": 1241,
        "x_start": 5681,
        "x_stop": 5687,
        "shape_yx": [5, 6],
    }
    assert view.image.shape == view.variance.shape == view.bad_mask.shape
    assert view.unit == "nJy"
    assert view.variance_unit == "nJy2"
    assert view.wcs is not None
    assert list(view.wcs.world_axis_units) == ["rad", "rad"]
    assert set(view.mask_plane_names) == {"BAD", "DETECTED"}
    assert view.product_role == "pixel-transport-only"
    assert view.reconstruction_ready is False
    assert view.psf is None
    assert view.provenance is None
    x = 3.0
    y = 2.0
    sky = view.pixel_to_skycoord(x, y)
    assert sky.ra.unit.to_string() == "deg"
    assert sky.dec.unit.to_string() == "deg"
    sky_x, sky_y = view.skycoord_to_pixel(sky)
    degree_x, degree_y = view.world_degrees_to_pixel(
        sky.icrs.ra.deg,
        sky.icrs.dec.deg,
    )
    low_level_x, low_level_y = view.wcs.world_to_pixel_values(
        sky.icrs.ra.rad,
        sky.icrs.dec.rad,
    )
    assert float(sky_x) == pytest.approx(x)
    assert float(sky_y) == pytest.approx(y)
    assert float(degree_x) == pytest.approx(x)
    assert float(degree_y) == pytest.approx(y)
    assert float(low_level_x) == pytest.approx(x)
    assert float(low_level_y) == pytest.approx(y)


def test_lsst_images_version_is_pinned_and_lazy(monkeypatch) -> None:
    modules = {
        "lsst.images": SimpleNamespace(__version__="30.0.9"),
        "lsst.images.cells": SimpleNamespace(CellCoadd=object),
        "lsst.images.serialization": SimpleNamespace(
            open_archive=lambda *args, **kwargs: None
        ),
    }
    monkeypatch.setattr(
        rubin_image,
        "import_module",
        modules.__getitem__,
    )
    with pytest.raises(
        RubinImageDependencyError,
        match="requires exactly lsst-images==30.0.10",
    ):
        rubin_image._load_lsst_images_api()

    env = dict(os.environ)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import cuphoton.ximage.adapters.rubin.image; "
                "assert 'lsst.images' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
