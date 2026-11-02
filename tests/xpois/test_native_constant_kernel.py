# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Native image subtraction parity and absence of CuPy-generated kernels."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import numpy as np
import pytest
from scipy.signal import fftconvolve

from cuphoton.xpois.ois import (
    GaussianBasisComponent,
    build_gaussian_polynomial_basis,
    solve_constant_kernel,
    solve_constant_kernel_device,
)


@pytest.fixture
def native_cupy() -> Any:
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("native xPois requires an actual CUDA device")
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    pytest.importorskip("cuphoton.core._imaging_ext")
    return cp


@contextmanager
def _forbid_cupy_compilation(monkeypatch: pytest.MonkeyPatch):
    from cupy._core import core
    from cupy.cuda import compiler

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("native xPois attempted CuPy runtime kernel compilation")

    with monkeypatch.context() as patch:
        patch.setattr(core, "compile_with_cache", forbidden)
        patch.setattr(compiler, "_compile_module_with_cache", forbidden)
        patch.setattr(compiler.nvrtc, "compileProgram", forbidden)
        yield


@pytest.mark.parametrize(
    "background_degree,flux_conserve", [(0, False), (1, True)]
)
def test_native_constant_kernel_matches_cupy_with_chunk_tail(
    native_cupy: Any,
    monkeypatch: pytest.MonkeyPatch,
    background_degree: int,
    flux_conserve: bool,
) -> None:
    cp = native_cupy
    rng = np.random.default_rng(42)
    reference = rng.normal(0.0, 2.0, size=(277, 283))
    components = [
        GaussianBasisComponent(1.5, 2),
        GaussianBasisComponent(3.0, 1),
    ]
    basis, _ = build_gaussian_polynomial_basis(
        (9, 9), components, flux_conserve=flux_conserve
    )
    kernel = np.tensordot(rng.normal(size=basis.shape[0]), basis, axes=(0, 0))
    target = fftconvolve(reference, kernel, mode="same") + 0.3
    variance = rng.uniform(1.0, 2.0, size=reference.shape)
    mask = np.ones(reference.shape, dtype=np.uint8)
    mask[::9, ::11] = 0
    reference[73, 59] = np.nan
    target[67, 68] = np.nan
    variance[81, 83] = np.inf
    # Non-contiguous FP32 device inputs also exercise native conversion.
    device_reference = cp.asarray(reference.astype(np.float32).T).T
    device_target = cp.asarray(target)
    device_variance = cp.asarray(variance)
    device_mask = cp.asarray(mask)
    options: dict[str, Any] = dict(
        kernel_shape=(9, 9),
        variance=device_variance,
        fit_mask=device_mask,
        background_degree=background_degree,
        flux_conserve=flux_conserve,
    )
    with _forbid_cupy_compilation(monkeypatch):
        actual = solve_constant_kernel_device(
            device_reference,
            device_target,
            components,
            backend="native",
            **options,
        )
    expected = solve_constant_kernel_device(
        device_reference, device_target, components, **options
    )
    assert actual.backend == "native"
    assert actual.fit_pixel_count == expected.fit_pixel_count > 65536
    assert actual.dof == expected.dof
    assert actual.basis_terms == expected.basis_terms
    np.testing.assert_array_equal(
        cp.asnumpy(actual.fit_mask), cp.asnumpy(expected.fit_mask)
    )
    for field in (
        "kernel",
        "background",
        "matched",
        "residual",
        "kernel_coefficients",
        "background_coefficients",
        "basis_kernels",
    ):
        left = cp.asnumpy(getattr(actual, field))
        right = cp.asnumpy(getattr(expected, field))
        np.testing.assert_array_equal(np.isnan(left), np.isnan(right))
        np.testing.assert_allclose(left, right, rtol=5e-12, atol=5e-12)
    assert actual.chi2 == pytest.approx(expected.chi2, rel=5e-12, abs=5e-12)
    np.testing.assert_array_equal(
        cp.asnumpy(device_reference), reference.astype(np.float32)
    )
    np.testing.assert_array_equal(cp.asnumpy(device_target), target)
    np.testing.assert_array_equal(cp.asnumpy(device_variance), variance)


@pytest.mark.parametrize("kernel_shape", [(7, 9), (9, 5)])
def test_native_convolution_matches_cupy_for_identical_kernel(
    native_cupy: Any, monkeypatch: pytest.MonkeyPatch, kernel_shape
) -> None:
    from cupyx.scipy.signal import fftconvolve as cupy_fftconvolve

    cp = native_cupy
    reference = np.random.default_rng(83).normal(size=(37, 43))
    components = [GaussianBasisComponent(1.3, 1)]
    basis, _ = build_gaussian_polynomial_basis(kernel_shape, components)
    # An asymmetric kernel exercises both products of the imaginary FFT
    # component; reassociating its fused multiply-add changes output bits.
    kernel = np.tensordot([1.1, -0.3, 0.17], basis, axes=(0, 0))
    target = fftconvolve(reference, kernel, mode="same") + 0.07
    device_reference, device_target = (
        cp.asarray(reference),
        cp.asarray(target),
    )
    with _forbid_cupy_compilation(monkeypatch):
        actual = solve_constant_kernel_device(
            device_reference,
            device_target,
            components,
            kernel_shape=kernel_shape,
            backend="native",
        )

    # Reuse the fitted coefficients' arrays to isolate convolution and output
    # arithmetic from any differences in the normal-equation solve.
    matched = (
        cupy_fftconvolve(device_reference, actual.kernel, mode="same")
        + actual.background
    )
    expected_matched = cp.asnumpy(matched)
    expected_residual = cp.asnumpy(device_target - matched)
    margin_y, margin_x = (length // 2 for length in kernel_shape)
    boundary = np.ones(reference.shape, dtype=bool)
    boundary[margin_y:-margin_y, margin_x:-margin_x] = False
    expected_matched[boundary] = np.nan
    expected_residual[boundary] = np.nan
    assert np.isfinite(cp.asnumpy(actual.matched)[~boundary]).all()
    assert np.isnan(cp.asnumpy(actual.matched)[boundary]).all()
    assert actual.backend == "native"
    np.testing.assert_array_equal(
        cp.asnumpy(actual.matched), expected_matched
    )
    np.testing.assert_array_equal(
        cp.asnumpy(actual.residual), expected_residual
    )


@pytest.mark.parametrize(
    "case,message",
    [
        ("variance", "variance must be strictly positive"),
        ("zero_variance", "variance must be strictly positive"),
        ("nonbinary", "fit_mask array must contain only 0/1 values"),
        ("nonfinite", "fit_mask array must not contain NaN or inf values"),
        ("empty", "fit_mask must select at least one interior pixel"),
    ],
)
def test_native_constant_kernel_preserves_validation(
    native_cupy: Any,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    message: str,
) -> None:
    cp = native_cupy
    reference = cp.asarray(np.ones((25, 27)))
    target = cp.asarray(np.ones((25, 27)))
    variance = np.ones((25, 27))
    mask = np.ones((25, 27))
    if case == "variance":
        variance[5, 5] = -1.0
    elif case == "zero_variance":
        variance[5, 5] = 0.0
    elif case == "nonbinary":
        mask[5, 5] = 0.5
    elif case == "nonfinite":
        mask[5, 5] = np.nan
    else:
        mask[:] = 0.0
    variance_device, mask_device = cp.asarray(variance), cp.asarray(mask)
    with (
        _forbid_cupy_compilation(monkeypatch),
        pytest.raises(ValueError, match=message),
    ):
        solve_constant_kernel_device(
            reference,
            target,
            [GaussianBasisComponent(1.5, 0)],
            kernel_shape=(5, 5),
            variance=variance_device,
            fit_mask=mask_device,
            backend="native",
        )


def test_host_native_constant_kernel_matches_cupy(native_cupy):
    reference = np.random.default_rng(7).normal(size=(41, 43))
    target = 1.1 * reference + 0.2
    components = [GaussianBasisComponent(1.3, 0)]
    expected = solve_constant_kernel(
        reference, target, components, kernel_shape=(5, 5), backend="cupy"
    )
    actual = solve_constant_kernel(
        reference, target, components, kernel_shape=(5, 5), backend="native"
    )
    assert actual.backend == "native"
    assert isinstance(actual.kernel, np.ndarray)
    np.testing.assert_allclose(
        actual.kernel, expected.kernel, rtol=5e-12, atol=5e-12
    )
    np.testing.assert_allclose(
        actual.residual, expected.residual, rtol=5e-12, atol=5e-12
    )


@pytest.mark.parametrize("kernel_shape", [(1, 1), (1, 3), (3, 1)])
def test_native_constant_kernel_rejects_length_one_axes(
    native_cupy: Any,
    monkeypatch: pytest.MonkeyPatch,
    kernel_shape: tuple[int, int],
) -> None:
    cp = native_cupy
    image = cp.asarray(np.random.default_rng(41).normal(size=(25, 27)))
    with (
        _forbid_cupy_compilation(monkeypatch),
        pytest.raises(
            ValueError, match="both kernel dimensions greater than one"
        ),
    ):
        solve_constant_kernel_device(
            image,
            image,
            [GaussianBasisComponent(1.5, 0)],
            kernel_shape=kernel_shape,
            backend="native",
        )
