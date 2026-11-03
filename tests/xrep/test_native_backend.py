# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from cuphoton.xrep import (
    BBox,
    ReprojectionSpec,
    prepare_reprojection,
    reproject_array_device,
    reproject_masked_array,
)
from cuphoton.xrep.backends import get_backend

cp = pytest.importorskip("cupy")
if not get_backend("native").is_available():
    pytest.skip(
        "native CUDA imaging extension unavailable", allow_module_level=True
    )


def _disable_runtime_compilation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError(
            "native reprojection attempted runtime compilation"
        )

    for name in (
        "_compile_module_with_cache",
        "compile_using_nvrtc",
        "compile_using_nvcc",
    ):
        monkeypatch.setattr(cp.cuda.compiler, name, forbidden)
    for name in (
        "RawKernel",
        "RawModule",
        "ElementwiseKernel",
        "ReductionKernel",
    ):
        monkeypatch.setattr(cp, name, forbidden)


def test_native_reprojection_without_compilation_in_fresh_process(tmp_path):
    source = textwrap.dedent("""
        import numpy as np
        import cupy as cp
        from cupy._core import core
        from cupy.cuda import compiler
        from cuphoton.xrep import (
            BBox, ReprojectionSpec, prepare_reprojection,
            reproject_array_device, reproject_masked_array,
        )

        def forbidden(*args, **kwargs):
            raise AssertionError("CuPy runtime compilation attempted")

        core.compile_with_cache = forbidden
        compiler._compile_module_with_cache = forbidden
        compiler.compile_using_nvrtc = forbidden
        compiler.compile_using_nvcc = forbidden
        compiler.nvrtc.compileProgram = forbidden
        image = np.arange(13 * 15, dtype=np.float64).reshape(13, 15)
        variance = np.ones_like(image)
        mask = np.zeros_like(image, dtype=np.uint32)
        mask[4, 4] = 8
        for interpolation in ("bilinear", "lanczos3"):
            spec = ReprojectionSpec(
                mapping=lambda xy: xy + np.array([0.2, -0.3]),
                output_bbox=BBox(0, 0, 15, 13), mapping_grid_step=4,
                interpolation=interpolation,
            )
            prepared = prepare_reprojection(
                spec, source_shape=image.shape, backend="native"
            )
            result = reproject_array_device(
                cp.asarray(image), spec, source_variance=cp.asarray(variance),
                source_mask=cp.asarray(mask), prepared=prepared,
                backend="native",
            )
            assert result.backend == "native"
            assert np.isfinite(cp.asnumpy(result.image)[3:-3, 3:-3]).all()
            host = reproject_masked_array(
                image, variance, mask, spec, backend="native"
            )
            assert host.image.shape == image.shape
    """)
    environment = os.environ.copy()
    environment["CUPY_CACHE_DIR"] = str(tmp_path / "cupy-empty")
    completed = subprocess.run(
        [sys.executable, "-c", source],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert not list((tmp_path / "cupy-empty").glob("**/*.cubin"))


def test_native_mapping_preserves_cupy_rounding_without_compilation(
    monkeypatch,
):
    def mapping(points):
        x, y = points.T
        return np.column_stack(
            (x * 1.17 + y * 0.013 + x * y * 0.002, y * 0.91 - x * 0.07)
        )

    spec = ReprojectionSpec(
        mapping=mapping,
        output_bbox=BBox(0, 0, 19, 17),
        mapping_grid_step=6,
        source_origin=(1, -1),
    )
    reference = prepare_reprojection(spec, source_shape=(19, 23), xp=cp)
    expected = {
        name: cp.asnumpy(getattr(reference, name))
        for name in ("x_local", "y_local", "relative_area", "valid")
    }
    _disable_runtime_compilation(monkeypatch)
    actual = prepare_reprojection(
        spec, source_shape=(19, 23), backend="native"
    )
    for name, value in expected.items():
        np.testing.assert_array_equal(
            cp.asnumpy(getattr(actual, name)), value
        )


@pytest.mark.parametrize("interpolation", ["bilinear", "lanczos3"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_native_device_planes_match_cupy_without_compilation(
    interpolation, dtype, monkeypatch
):
    rng = np.random.default_rng(42)
    source = rng.normal(size=(13, 30)).astype(dtype)
    source[4, 4] = np.nan
    variance = rng.uniform(0.1, 2.0, size=source.shape)
    variance[6, 10] = np.inf
    mask = np.zeros(source.shape, dtype=np.uint64)
    mask[2, 4] = 1 << 63
    mask[5, 12] = 8
    source_gpu = cp.asarray(source)[:, ::2]
    variance_gpu = cp.asarray(variance)[:, ::2]
    mask_gpu = cp.asarray(mask)[:, ::2]
    spec = ReprojectionSpec(
        mapping=lambda points: (
            points * np.array([0.93, 1.07]) + np.array([-0.3, 0.2])
        ),
        output_bbox=BBox(0, 0, 15, 13),
        interpolation=interpolation,
        mapping_grid_step=4,
        fill_value=-19.0,
        invalid_mask_value=2,
    )
    reference = reproject_array_device(
        source_gpu,
        spec,
        source_mask=mask_gpu,
        source_variance=variance_gpu,
        variance_fill_value=37.0,
        backend="cupy",
    )
    expected = {
        name: cp.asnumpy(getattr(reference, name))
        for name in ("image", "variance", "mask")
    }
    _disable_runtime_compilation(monkeypatch)
    with cp.cuda.Stream(non_blocking=True):
        actual = reproject_array_device(
            source_gpu,
            spec,
            source_mask=mask_gpu,
            source_variance=variance_gpu,
            variance_fill_value=37.0,
            backend="native",
        )
        arrays = {
            name: cp.asnumpy(getattr(actual, name)) for name in expected
        }
    assert actual.backend == "native"
    for name in ("image", "variance"):
        np.testing.assert_allclose(
            arrays[name], expected[name], rtol=1e-12, atol=1e-12
        )
    np.testing.assert_array_equal(arrays["mask"], expected["mask"])


@pytest.mark.parametrize("two_a_footprint,lanczos_a", [(True, 3), (False, 2)])
def test_native_masked_api_preserves_full_kernel_mask_support(
    two_a_footprint, lanczos_a, monkeypatch
):
    rng = np.random.default_rng(8)
    source = rng.normal(size=(15, 17))
    variance = rng.uniform(size=source.shape)
    mask = np.zeros(source.shape, dtype=np.int16)
    mask[3, 3] = 2
    mask[7, 7] = -32768
    spec = ReprojectionSpec(
        mapping=lambda points: points + np.array([0.25, 0.375]),
        output_bbox=BBox(0, 0, 17, 15),
        mapping_grid_step=4,
        lanczos_a=lanczos_a,
        two_a_footprint=two_a_footprint,
    )
    reference = reproject_masked_array(
        source, variance, mask, spec, backend="cupy"
    )
    _disable_runtime_compilation(monkeypatch)
    actual = reproject_masked_array(
        source, variance, mask, spec, backend="native"
    )
    np.testing.assert_allclose(
        actual.image, reference.image, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        actual.variance, reference.variance, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_array_equal(actual.mask, reference.mask)


def test_native_mapping_diagnostics_without_compilation(monkeypatch):
    from cuphoton.xrep.backends.native_backend import (
        mapping_edge_score,
        sample_mapping,
    )
    from cuphoton.xrep.mapping import PreparedReprojection

    x = np.array([[0.0, 3.5, np.nan], [-1.0, 2.0, 5.0]])
    y = np.array([[-0.0, 1.25, 2.0], [np.inf, 3.0, -0.25]])
    prepared = PreparedReprojection(
        cp.asarray(x),
        cp.asarray(y),
        cp.asarray(np.ones_like(x)),
        cp.asarray(np.ones_like(x, dtype=bool)),
    )
    expected = np.minimum(np.minimum(np.minimum(x, y), 6 - x), 4 - y)
    ys, xs = np.array([1, 0, 1]), np.array([2, 1, 0])
    _disable_runtime_compilation(monkeypatch)
    actual = cp.asnumpy(mapping_edge_score(prepared, (5, 7)))
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))
    np.testing.assert_array_equal(
        sample_mapping(prepared, ys, xs),
        np.column_stack((x[ys, xs], y[ys, xs])),
    )
    with pytest.raises(ValueError, match="inside"):
        sample_mapping(prepared, np.array([2]), np.array([0]))
