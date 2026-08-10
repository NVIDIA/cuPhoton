# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from cuphoton.xrep import (
    BBox,
    ReprojectionSpec,
    prepare_reprojection,
    reproject_array,
    reproject_masked_array,
)
from cuphoton.xrep.backends import get_backend

cp = pytest.importorskip("cupy")

cupy_kernels = pytest.importorskip("cuphoton.xrep.backends._cupy_kernels")
cupy_backend = pytest.importorskip("cuphoton.xrep.backends.cupy_backend")
xrep_reproject = pytest.importorskip("cuphoton.xrep.reproject")

sample_lanczos3_cupy_raw = cupy_kernels.sample_lanczos3_cupy_raw
sample_lanczos_cupy = cupy_kernels.sample_lanczos_cupy
reproject_lanczos3_cupy_raw = cupy_kernels.reproject_lanczos3_cupy_raw
propagate_mask_or_cupy = cupy_kernels.propagate_mask_or_cupy
propagate_mask_or_cupy_raw = cupy_kernels.propagate_mask_or_cupy_raw
CupyBackend = cupy_backend.CupyBackend
reproject_array_device = xrep_reproject.reproject_array_device

if not get_backend("cupy").is_available():  # pragma: no cover - no GPU
    pytest.skip("no CUDA device available", allow_module_level=True)


def test_cupy_backend_registered() -> None:
    assert isinstance(get_backend("cupy"), CupyBackend)


@pytest.mark.parametrize("interpolation", ["bilinear", "lanczos3"])
def test_cupy_backend_matches_cpu_for_identity(interpolation: str) -> None:
    source = np.arange(36, dtype=np.float64).reshape(6, 6)
    spec = ReprojectionSpec(
        mapping=lambda coords: np.asarray(coords, dtype=np.float64),
        output_bbox=BBox(min_x=0, min_y=0, width=6, height=6),
        interpolation=interpolation,
        mapping_grid_step=1,
    )

    cpu = reproject_array(source, spec, backend="cpu")
    cupy_result = reproject_array(source, spec, backend="cupy")

    assert np.allclose(cupy_result.image, cpu.image, equal_nan=True)
    assert np.array_equal(cupy_result.mask, cpu.mask)
    assert cupy_result.backend == "cupy"


def test_cupy_lanczos_matches_cpu_for_subpixel_mapping() -> None:
    source = np.arange(64, dtype=np.float64).reshape(8, 8)
    offset = np.array([0.25, 0.375], dtype=np.float64)
    spec = ReprojectionSpec(
        mapping=lambda coords: np.asarray(coords, dtype=np.float64) + offset,
        output_bbox=BBox(min_x=0, min_y=0, width=5, height=5),
        interpolation="lanczos3",
        mapping_grid_step=1,
    )

    cpu = reproject_array(source, spec, backend="cpu")
    cupy_result = reproject_array(source, spec, backend="cupy")

    assert np.allclose(
        cupy_result.image,
        cpu.image,
        atol=1e-10,
        rtol=1e-12,
        equal_nan=True,
    )
    assert np.array_equal(cupy_result.mask, cpu.mask)


def test_cupy_raw_lanczos_matches_elementwise() -> None:
    source = cp.asarray(np.arange(64, dtype=np.float64).reshape(8, 8))
    yy, xx = cp.meshgrid(
        cp.arange(5, dtype=cp.float64),
        cp.arange(5, dtype=cp.float64),
        indexing="ij",
    )
    x = xx + 0.25
    y = yy + 0.375

    elementwise = sample_lanczos_cupy(source, x, y)
    raw = sample_lanczos3_cupy_raw(source, x, y)

    assert np.allclose(
        cp.asnumpy(raw),
        cp.asnumpy(elementwise),
        atol=1e-12,
        rtol=1e-12,
        equal_nan=True,
    )


def test_cupy_fused_raw_lanczos_matches_postprocessed_elementwise() -> None:
    source = cp.asarray(np.arange(64, dtype=np.float64).reshape(8, 8))
    yy, xx = cp.meshgrid(
        cp.arange(5, dtype=cp.float64),
        cp.arange(5, dtype=cp.float64),
        indexing="ij",
    )
    x = xx + 0.25
    y = yy + 0.375
    valid = cp.ones(x.shape, dtype=cp.bool_)
    valid[0, 0] = False
    area = cp.linspace(0.5, 1.5, x.size, dtype=cp.float64).reshape(x.shape)
    fill_value = -999.0

    elementwise = sample_lanczos_cupy(source, x, y) * area
    expected = cp.where(valid, elementwise, fill_value)
    fused = reproject_lanczos3_cupy_raw(
        source,
        x,
        y,
        valid,
        area,
        area_scaling=True,
        fill_value=fill_value,
    )

    assert np.allclose(
        cp.asnumpy(fused),
        cp.asnumpy(expected),
        atol=1e-12,
        rtol=1e-12,
        equal_nan=True,
    )


def test_cupy_raw_mask_matches_vectorized_mask() -> None:
    source_mask = cp.zeros((8, 8), dtype=cp.uint16)
    source_mask[1, 1] = 2
    source_mask[3, 4] = 4
    yy, xx = cp.meshgrid(
        cp.arange(6, dtype=cp.float64),
        cp.arange(6, dtype=cp.float64),
        indexing="ij",
    )
    x = xx + 0.25
    y = yy + 0.375
    x = x.copy()
    y = y.copy()
    x[0, 0] = -1.0

    vectorized = propagate_mask_or_cupy(
        source_mask,
        x,
        y,
        invalid_mask_value=1,
    )
    raw = propagate_mask_or_cupy_raw(
        source_mask,
        x,
        y,
        invalid_mask_value=1,
    )

    assert np.array_equal(cp.asnumpy(raw), cp.asnumpy(vectorized))


def test_cupy_backend_can_use_raw_lanczos(monkeypatch) -> None:
    monkeypatch.setenv("CUPHOTON_XREP_CUPY_LANCZOS_KERNEL", "raw")
    source = np.arange(64, dtype=np.float64).reshape(8, 8)
    offset = np.array([0.25, 0.375], dtype=np.float64)
    spec = ReprojectionSpec(
        mapping=lambda coords: np.asarray(coords, dtype=np.float64) + offset,
        output_bbox=BBox(min_x=0, min_y=0, width=5, height=5),
        interpolation="lanczos3",
        mapping_grid_step=1,
    )

    cpu = reproject_array(source, spec, backend="cpu")
    cupy_result = reproject_array(source, spec, backend="cupy")

    assert cupy_result.metadata["lanczos_kernel"] == "raw"
    assert np.allclose(
        cupy_result.image,
        cpu.image,
        atol=1e-10,
        rtol=1e-12,
        equal_nan=True,
    )
    assert np.array_equal(cupy_result.mask, cpu.mask)


def test_cupy_backend_raw_lanczos_propagates_source_mask(
    monkeypatch,
) -> None:
    monkeypatch.setenv("CUPHOTON_XREP_CUPY_LANCZOS_KERNEL", "raw")
    source = np.arange(64, dtype=np.float64).reshape(8, 8)
    source_mask = np.zeros(source.shape, dtype=np.uint16)
    source_mask[1, 1] = 2
    source_mask[3, 4] = 4
    offset = np.array([0.25, 0.375], dtype=np.float64)
    spec = ReprojectionSpec(
        mapping=lambda coords: np.asarray(coords, dtype=np.float64) + offset,
        output_bbox=BBox(min_x=0, min_y=0, width=5, height=5),
        interpolation="lanczos3",
        mapping_grid_step=1,
    )

    cpu = reproject_array(
        source, spec, source_mask=source_mask, backend="cpu"
    )
    cupy_result = reproject_array(
        source,
        spec,
        source_mask=source_mask,
        backend="cupy",
    )

    assert np.array_equal(cupy_result.mask, cpu.mask)


def test_cupy_reproject_array_device_returns_cupy_arrays(monkeypatch) -> None:
    monkeypatch.setenv("CUPHOTON_XREP_CUPY_LANCZOS_KERNEL", "raw")
    source = np.arange(64, dtype=np.float64).reshape(8, 8)
    offset = np.array([0.25, 0.375], dtype=np.float64)
    spec = ReprojectionSpec(
        mapping=lambda coords: np.asarray(coords, dtype=np.float64) + offset,
        output_bbox=BBox(min_x=0, min_y=0, width=5, height=5),
        interpolation="lanczos3",
        mapping_grid_step=1,
    )

    cpu = reproject_array(source, spec, backend="cpu")
    device = reproject_array_device(source, spec)

    assert isinstance(device.image, cp.ndarray)
    assert isinstance(device.mask, cp.ndarray)
    assert device.metadata["result_location"] == "device"
    assert device.metadata["lanczos_kernel"] == "raw"
    assert np.allclose(
        cp.asnumpy(device.image),
        cpu.image,
        atol=1e-10,
        rtol=1e-12,
        equal_nan=True,
    )
    assert np.array_equal(cp.asnumpy(device.mask), cpu.mask)


def test_cupy_reproject_array_device_reuses_prepared_geometry() -> None:
    source = np.arange(64, dtype=np.float64).reshape(8, 8)
    source_dev = cp.asarray(source)
    offset = np.array([0.25, 0.375], dtype=np.float64)
    spec = ReprojectionSpec(
        mapping=lambda coords: np.asarray(coords, dtype=np.float64) + offset,
        output_bbox=BBox(min_x=0, min_y=0, width=5, height=5),
        interpolation="lanczos3",
        mapping_grid_step=1,
    )
    prepared = prepare_reprojection(spec, source_shape=source.shape, xp=cp)

    cpu = reproject_array(source, spec, backend="cpu")
    device = reproject_array_device(source_dev, spec, prepared=prepared)

    assert isinstance(device.image, cp.ndarray)
    assert np.allclose(
        cp.asnumpy(device.image),
        cpu.image,
        atol=1e-10,
        rtol=1e-12,
        equal_nan=True,
    )


def test_cupy_legacy_and_plane_aware_mask_policies_are_distinct() -> None:
    image = np.ones((9, 9), dtype=np.float64)
    variance = np.ones_like(image)
    mask = np.zeros(image.shape, dtype=np.uint64)
    mask[2, 2] = np.uint64(1) << np.uint64(40)
    spec = ReprojectionSpec(
        mapping=lambda coords: np.broadcast_to(
            np.array([4.25, 4.375]),
            np.asarray(coords).shape,
        ),
        output_bbox=BBox(min_x=0, min_y=0, width=1, height=1),
        interpolation="lanczos3",
        mapping_grid_step=1,
        area_scaling=False,
    )

    legacy = reproject_array(
        image,
        spec,
        source_mask=mask,
        backend="cupy",
    )
    legacy_device = reproject_array_device(
        cp.asarray(image),
        spec,
        source_mask=cp.asarray(mask),
    )
    plane_aware = reproject_masked_array(
        image,
        variance,
        mask,
        spec,
        backend="cupy",
    )

    assert legacy.mask[0, 0] == 0
    assert cp.asnumpy(legacy_device.mask)[0, 0] == 0
    assert plane_aware.mask[0, 0] == np.uint64(1) << np.uint64(40)


@pytest.mark.parametrize("interpolation", ["bilinear", "lanczos3"])
def test_cupy_masked_reproject_matches_cpu(interpolation: str) -> None:
    rng = np.random.default_rng(42)
    image = rng.normal(size=(9, 9))
    variance = rng.uniform(0.5, 3.0, size=(9, 9))
    mask = np.arange(81, dtype=np.uint64).reshape(9, 9)
    mask[0, 0] = np.uint64(1) << np.uint64(63)
    spec = ReprojectionSpec(
        mapping=lambda coords: (
            np.asarray(coords, dtype=np.float64) + np.array([0.25, 0.375])
        ),
        output_bbox=BBox(min_x=0, min_y=0, width=7, height=7),
        interpolation=interpolation,
        mapping_grid_step=1,
    )

    cpu = reproject_masked_array(
        image,
        variance,
        mask,
        spec,
        backend="cpu",
    )
    result = reproject_masked_array(
        image,
        variance,
        mask,
        spec,
        backend="cupy",
    )

    assert np.allclose(
        result.image,
        cpu.image,
        atol=1e-10,
        rtol=1e-12,
        equal_nan=True,
    )
    assert np.allclose(
        result.variance,
        cpu.variance,
        atol=1e-10,
        rtol=1e-12,
        equal_nan=True,
    )
    assert np.array_equal(result.mask, cpu.mask)


@pytest.mark.parametrize(
    ("interpolation", "kernel"),
    [
        ("bilinear", "elementwise"),
        ("lanczos3", "elementwise"),
        ("lanczos3", "raw"),
    ],
)
def test_cupy_ignores_zero_weight_nonfinite_neighbors(
    interpolation: str,
    kernel: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUPHOTON_XREP_CUPY_LANCZOS_KERNEL", kernel)
    image = np.array([[5.0, np.nan], [np.inf, -np.inf]])
    variance = np.array([[2.0, np.inf], [np.inf, np.inf]])
    mask = np.zeros((2, 2), dtype=np.uint16)
    spec = ReprojectionSpec(
        mapping=lambda coords: np.asarray(coords, dtype=np.float64),
        output_bbox=BBox(min_x=0, min_y=0, width=1, height=1),
        interpolation=interpolation,
        mapping_grid_step=1,
    )

    result = reproject_masked_array(
        image,
        variance,
        mask,
        spec,
        backend="cupy",
    )

    assert result.image[0, 0] == pytest.approx(5.0)
    assert result.variance[0, 0] == pytest.approx(2.0)


def test_cupy_uses_variance_fill_independent_from_image_fill() -> None:
    image = np.ones((2, 2), dtype=np.float64)
    variance = np.ones_like(image)
    mask = np.zeros(image.shape, dtype=np.uint16)
    spec = ReprojectionSpec(
        mapping=lambda coords: np.asarray(coords, dtype=np.float64) - 10.0,
        output_bbox=BBox(min_x=0, min_y=0, width=1, height=1),
        interpolation="bilinear",
        mapping_grid_step=1,
        fill_value=-99.0,
    )

    result = reproject_masked_array(
        image,
        variance,
        mask,
        spec,
        backend="cupy",
        variance_fill_value=123.0,
    )

    assert result.image[0, 0] == -99.0
    assert result.variance[0, 0] == 123.0


@pytest.mark.parametrize("interpolation", ["bilinear", "lanczos3"])
def test_cupy_masked_reproject_matches_cpu_with_nonunit_area(
    interpolation: str,
) -> None:
    rng = np.random.default_rng(7)
    image = rng.normal(size=(12, 12))
    variance = rng.uniform(0.5, 3.0, size=image.shape)
    mask = np.arange(image.size, dtype=np.uint64).reshape(image.shape)
    scale = np.array([1.2, 0.8], dtype=np.float64)
    offset = np.array([2.25, 2.375], dtype=np.float64)
    spec = ReprojectionSpec(
        mapping=lambda coords: (
            np.asarray(coords, dtype=np.float64) * scale + offset
        ),
        output_bbox=BBox(min_x=0, min_y=0, width=5, height=5),
        interpolation=interpolation,
        mapping_grid_step=1,
        area_scaling=True,
    )

    prepared = prepare_reprojection(spec, source_shape=image.shape)
    cpu = reproject_masked_array(
        image,
        variance,
        mask,
        spec,
        backend="cpu",
        prepared=prepared,
    )
    result = reproject_masked_array(
        image,
        variance,
        mask,
        spec,
        backend="cupy",
    )

    assert np.allclose(prepared.relative_area, 0.96)
    assert np.allclose(result.image, cpu.image, atol=1e-10, rtol=1e-12)
    assert np.allclose(
        result.variance,
        cpu.variance,
        atol=1e-10,
        rtol=1e-12,
    )
    assert np.array_equal(result.mask, cpu.mask)
