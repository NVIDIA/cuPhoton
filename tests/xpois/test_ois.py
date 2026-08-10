# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import fftconvolve

from cuphoton.xpois.ois import (
    GaussianBasisComponent,
    build_compact_source_stamp_mask,
    build_gaussian_polynomial_basis,
    make_stamp_mask,
    resolve_backend,
    solve_constant_kernel,
    solve_separable_kernel,
    triangular_degree_pairs,
)


def test_auto_backend_prefers_cupy_then_numba_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    available = {"cupy": False, "numba-cuda": True, "cpu": True}
    seen: list[str] = []

    def is_available(backend: str) -> bool:
        seen.append(backend)
        return available[backend]

    monkeypatch.setattr(
        "cuphoton.xpois.ois._backend_available",
        is_available,
    )

    assert resolve_backend("auto") == "numba-cuda"
    assert seen == ["cupy", "numba-cuda"]

    available["cupy"] = True
    seen.clear()
    assert resolve_backend("auto") == "cupy"
    assert seen == ["cupy"]


def test_auto_backend_falls_back_to_cpu_without_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "cuphoton.xpois.ois._backend_available",
        lambda backend: backend == "cpu",
    )

    assert resolve_backend("auto") == "cpu"


def test_auto_backend_never_selects_cutile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []

    def is_available(backend: str) -> bool:
        seen.append(backend)
        return backend == "cpu"

    monkeypatch.setattr(
        "cuphoton.xpois.ois._backend_available",
        is_available,
    )

    assert resolve_backend("auto") == "cpu"
    assert "cutile" not in seen


def _reference_image(shape: tuple[int, int]) -> np.ndarray:
    y_coords, x_coords = np.meshgrid(
        np.arange(shape[0], dtype=np.float64),
        np.arange(shape[1], dtype=np.float64),
        indexing="ij",
    )
    stars = [
        (18.5, 17.0, 10.0, 120.0),
        (42.0, 39.5, 14.0, 180.0),
        (28.0, 49.0, 8.0, 90.0),
    ]
    image = np.zeros(shape, dtype=np.float64)
    for cy, cx, sigma, amp in stars:
        image += amp * np.exp(
            -(((x_coords - cx) ** 2) + ((y_coords - cy) ** 2))
            / (2.0 * sigma**2)
        )
    return image


def _compact_reference_image(shape: tuple[int, int]) -> np.ndarray:
    y_coords, x_coords = np.meshgrid(
        np.arange(shape[0], dtype=np.float64),
        np.arange(shape[1], dtype=np.float64),
        indexing="ij",
    )
    stars = [
        (18.0, 17.0, 2.0, 120.0),
        (42.0, 39.0, 2.5, 180.0),
        (28.0, 49.0, 1.8, 90.0),
    ]
    image = np.zeros(shape, dtype=np.float64)
    for cy, cx, sigma, amp in stars:
        image += amp * np.exp(
            -(((x_coords - cx) ** 2) + ((y_coords - cy) ** 2))
            / (2.0 * sigma**2)
        )
    return image


def _constant_kernel_parity_fixture() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[GaussianBasisComponent],
]:
    reference = _reference_image((80, 80))
    components = [
        GaussianBasisComponent(sigma=1.4, degree=1),
        GaussianBasisComponent(sigma=2.5, degree=0),
    ]
    basis, _ = build_gaussian_polynomial_basis((9, 9), components)
    true_kernel_coeffs = np.array([0.8, 0.03, -0.02, 0.1])
    true_kernel = np.tensordot(true_kernel_coeffs, basis, axes=(0, 0))
    y_gradient = np.linspace(-1.0, 1.0, 80)[:, None]
    x_gradient = np.linspace(-1.0, 1.0, 80)[None, :]
    background = 0.07 + 0.02 * x_gradient - 0.01 * y_gradient
    target = fftconvolve(reference, true_kernel, mode="same") + background
    variance = np.full_like(target, 0.5)
    return reference, target, variance, components


def _assert_constant_kernel_matches_cpu(cpu, result, *, backend: str) -> None:
    finite = np.isfinite(cpu.matched) & np.isfinite(result.matched)
    assert result.backend == backend
    assert result.fit_pixel_count == cpu.fit_pixel_count
    assert result.dof == cpu.dof
    assert np.array_equal(result.fit_mask, cpu.fit_mask)
    assert np.allclose(result.kernel, cpu.kernel, rtol=1e-9, atol=1e-8)
    assert np.allclose(
        result.background,
        cpu.background,
        rtol=1e-9,
        atol=1e-8,
    )
    assert np.allclose(
        result.matched[finite],
        cpu.matched[finite],
        rtol=1e-9,
        atol=1e-8,
    )
    assert np.allclose(
        result.residual[finite],
        cpu.residual[finite],
        rtol=1e-9,
        atol=1e-8,
    )
    assert np.isclose(result.chi2, cpu.chi2, rtol=1e-9, atol=1e-8)


def _solve_parity_fixture_backend(backend: str):
    reference, target, variance, components = (
        _constant_kernel_parity_fixture()
    )
    return solve_constant_kernel(
        reference,
        target,
        components,
        kernel_shape=(9, 9),
        variance=variance,
        background_degree=1,
        backend=backend,
    )


def test_triangular_degree_pairs_order() -> None:
    assert triangular_degree_pairs(2) == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (2, 0),
    ]


def test_flux_conserving_basis_has_zero_sum_residual_terms() -> None:
    basis, terms = build_gaussian_polynomial_basis(
        (9, 9),
        [GaussianBasisComponent(sigma=1.5, degree=2)],
        flux_conserve=True,
    )

    assert np.isclose(basis[0].sum(), 1.0)
    assert any(term.zero_sum for term in terms[1:])
    for kernel in basis[1:]:
        assert abs(float(kernel.sum())) < 1e-10


def test_make_stamp_mask_marks_requested_rectangles() -> None:
    mask = make_stamp_mask((8, 8), [(1, 3, 2, 5), (5, 7, 1, 4)])

    assert mask.sum() == (2 * 3) + (2 * 3)
    assert mask[1, 2]
    assert mask[6, 3]
    assert not mask[0, 0]


def test_make_stamp_mask_rejects_out_of_bounds_rectangles() -> None:
    with np.testing.assert_raises(ValueError):
        make_stamp_mask((8, 8), [(-1, 3, 2, 5)])


def test_build_compact_source_stamp_mask_selects_bright_compact_peaks() -> (
    None
):
    reference = _compact_reference_image((64, 64))
    variance = np.ones_like(reference, dtype=np.float64)

    result = build_compact_source_stamp_mask(
        reference,
        variance=variance,
        stamp_size=11,
        max_stamps=2,
        peak_percentile=98.0,
    )

    assert result.mask.shape == reference.shape
    assert len(result.centers) == 2
    assert result.mask.sum() == 2 * 11 * 11
    expected = [(42, 40), (18, 17)]
    for target_y, target_x in expected:
        assert any(
            abs(center_y - target_y) <= 2 and abs(center_x - target_x) <= 2
            for center_y, center_x in result.centers
        )


def test_build_compact_source_stamp_mask_respects_valid_mask() -> None:
    reference = _compact_reference_image((64, 64))
    variance = np.ones_like(reference, dtype=np.float64)
    valid_mask = np.ones_like(reference, dtype=bool)
    valid_mask[35:49, 32:47] = False

    result = build_compact_source_stamp_mask(
        reference,
        variance=variance,
        valid_mask=valid_mask,
        stamp_size=11,
        max_stamps=2,
        peak_percentile=98.0,
    )

    assert len(result.centers) == 2
    assert all(not (35 <= y < 49 and 32 <= x < 47) for y, x in result.centers)
    assert any(
        abs(y - 18) <= 2 and abs(x - 17) <= 2 for y, x in result.centers
    )


def test_solve_constant_kernel_recovers_synthetic_match() -> None:
    reference = _reference_image((64, 64))
    components = [
        GaussianBasisComponent(sigma=1.4, degree=0),
        GaussianBasisComponent(sigma=2.5, degree=2),
    ]
    basis, _ = build_gaussian_polynomial_basis(
        (11, 11),
        components,
        flux_conserve=True,
    )
    true_kernel_coeffs = np.array([0.92, 0.04, -0.015, 0.0, 0.008, 0.0, 0.0])
    true_kernel = np.tensordot(true_kernel_coeffs, basis, axes=(0, 0))
    background = 0.05 + 0.01 * np.linspace(-1.0, 1.0, 64)[None, :]
    target = fftconvolve(reference, true_kernel, mode="same") + background
    variance = np.full_like(target, 0.25)

    result = solve_constant_kernel(
        reference,
        target,
        components,
        kernel_shape=(11, 11),
        variance=variance,
        background_degree=1,
        flux_conserve=True,
    )

    assert result.fit_pixel_count == (64 - 10) * (64 - 10)
    assert result.dof > 0
    valid = np.zeros_like(target, dtype=bool)
    valid[5:-5, 5:-5] = True
    assert np.allclose(result.matched[valid], target[valid], atol=1e-5)
    assert np.allclose(result.residual[valid], 0.0, atol=1e-5)
    assert np.allclose(result.kernel, true_kernel, atol=5e-4)
    assert result.chi2 < 1e-6
    assert result.backend == "cpu"


def test_solve_constant_kernel_cupy_matches_cpu() -> None:
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception as exc:
        pytest.skip(f"CuPy CUDA runtime is not usable: {exc}")

    cpu = _solve_parity_fixture_backend("cpu")
    gpu = _solve_parity_fixture_backend("cupy")
    _assert_constant_kernel_matches_cpu(cpu, gpu, backend="cupy")


def test_solve_constant_kernel_numba_cuda_matches_cpu() -> None:
    cuda = pytest.importorskip("numba.cuda")
    if not cuda.is_available():
        pytest.skip("Numba CUDA runtime is not usable")

    cpu = _solve_parity_fixture_backend("cpu")
    gpu = _solve_parity_fixture_backend("numba-cuda")
    _assert_constant_kernel_matches_cpu(cpu, gpu, backend="numba-cuda")


def test_solve_constant_kernel_cutile_matches_cpu() -> None:
    cp = pytest.importorskip("cupy")
    try:
        import cuda.tile  # noqa: F401
    except Exception as exc:
        pytest.skip(f"cuda.tile is not usable: {exc}")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception as exc:
        pytest.skip(f"CuPy CUDA runtime is not usable: {exc}")

    cpu = _solve_parity_fixture_backend("cpu")
    gpu = _solve_parity_fixture_backend("cutile")
    _assert_constant_kernel_matches_cpu(cpu, gpu, backend="cutile")


def test_solve_constant_kernel_rejects_unknown_backend() -> None:
    reference = _reference_image((32, 32))

    with pytest.raises(ValueError, match="backend"):
        solve_constant_kernel(
            reference,
            reference.copy(),
            [GaussianBasisComponent(sigma=1.5, degree=0)],
            kernel_shape=(9, 9),
            backend="bogus",
        )


def test_solve_constant_kernel_supports_sparse_fit_mask() -> None:
    reference = _reference_image((64, 64))
    components = [GaussianBasisComponent(sigma=1.8, degree=0)]
    basis, _ = build_gaussian_polynomial_basis(
        (9, 9),
        components,
        flux_conserve=False,
    )
    target = fftconvolve(reference, basis[0], mode="same")
    variance = np.full_like(target, 1.0)
    mask = make_stamp_mask((64, 64), [(8, 28, 8, 28), (30, 56, 30, 58)])

    result = solve_constant_kernel(
        reference,
        target,
        components,
        kernel_shape=(9, 9),
        variance=variance,
        fit_mask=mask,
        background_degree=0,
        flux_conserve=False,
    )

    assert result.fit_pixel_count == int(mask.sum())
    assert np.allclose(result.matched[mask], target[mask], atol=1e-5)


def test_solve_separable_kernel_recovers_synthetic_match() -> None:
    reference = _reference_image((64, 64))
    components = [
        GaussianBasisComponent(sigma=1.4, degree=1),
        GaussianBasisComponent(sigma=2.2, degree=2),
    ]
    x_coords = np.arange(11, dtype=np.float64) - 5
    y_coords = np.arange(11, dtype=np.float64) - 5
    horizontal = np.exp(-(x_coords**2) / (2.0 * 1.4**2)) * (
        1.0 + 0.03 * x_coords
    )
    vertical = np.exp(-(y_coords**2) / (2.0 * 2.2**2)) * (
        1.0 - 0.02 * y_coords + 0.005 * (y_coords**2)
    )
    true_kernel = np.outer(vertical, horizontal)
    true_kernel = true_kernel / true_kernel.sum()
    background = 0.05 + 0.01 * np.linspace(-1.0, 1.0, 64)[:, None]
    target = fftconvolve(reference, true_kernel, mode="same") + background
    variance = np.full_like(target, 0.25)

    result = solve_separable_kernel(
        reference,
        target,
        components,
        kernel_shape=(11, 11),
        variance=variance,
        background_degree=1,
        flux_conserve=True,
        max_iterations=10,
        tolerance=1e-8,
    )

    valid = np.zeros_like(target, dtype=bool)
    valid[5:-5, 5:-5] = True
    assert result.fit_pixel_count == int(valid.sum())
    assert result.dof > 0
    assert np.allclose(result.matched[valid], target[valid], atol=1e-4)
    assert np.allclose(result.residual[valid], 0.0, atol=1e-4)
    assert np.allclose(result.kernel, true_kernel, atol=1e-2)
    assert result.chi2 < 1e-3


def test_solve_constant_kernel_clips_edge_touching_fit_mask() -> None:
    reference = _reference_image((32, 32))
    components = [GaussianBasisComponent(sigma=1.5, degree=0)]
    basis, _ = build_gaussian_polynomial_basis((9, 9), components)
    target = fftconvolve(reference, basis[0], mode="same")
    variance = np.full_like(target, 1.0)
    mask = np.ones_like(target, dtype=bool)

    result = solve_constant_kernel(
        reference,
        target,
        components,
        kernel_shape=(9, 9),
        variance=variance,
        fit_mask=mask,
        background_degree=0,
        flux_conserve=False,
    )

    assert result.fit_pixel_count == (32 - 8) * (32 - 8)


def test_solve_constant_kernel_rejects_underdetermined_mask() -> None:
    reference = _reference_image((32, 32))
    target = reference.copy()
    components = [GaussianBasisComponent(sigma=1.5, degree=6)]
    mask = make_stamp_mask((32, 32), [(10, 14, 10, 14)])

    with np.testing.assert_raises(ValueError):
        solve_constant_kernel(
            reference,
            target,
            components,
            kernel_shape=(15, 15),
            fit_mask=mask,
            background_degree=0,
            flux_conserve=False,
        )


def test_solve_constant_kernel_masks_non_finite_pixels() -> None:
    reference = _reference_image((64, 64))
    components = [GaussianBasisComponent(sigma=1.8, degree=0)]
    basis, _ = build_gaussian_polynomial_basis((9, 9), components)
    target = fftconvolve(reference, basis[0], mode="same")
    variance = np.full_like(target, 1.0)
    reference[24, 24] = np.nan
    target[30, 30] = np.nan
    variance[34, 34] = np.inf

    result = solve_constant_kernel(
        reference,
        target,
        components,
        kernel_shape=(9, 9),
        variance=variance,
        background_degree=0,
        flux_conserve=False,
    )

    assert np.isfinite(result.kernel).all()
    assert np.isfinite(result.background).all()
    assert result.fit_pixel_count < (64 - 8) * (64 - 8)


def test_solve_constant_kernel_rejects_all_non_finite_pixels() -> None:
    reference = np.full((32, 32), np.nan, dtype=np.float64)
    target = np.full((32, 32), np.nan, dtype=np.float64)
    components = [GaussianBasisComponent(sigma=1.5, degree=0)]

    with np.testing.assert_raises(ValueError):
        solve_constant_kernel(
            reference,
            target,
            components,
            kernel_shape=(9, 9),
            background_degree=0,
            flux_conserve=False,
        )


def test_solve_constant_kernel_rejects_malformed_numeric_fit_mask() -> None:
    reference = _reference_image((32, 32))
    target = reference.copy()
    components = [GaussianBasisComponent(sigma=1.5, degree=0)]
    mask = np.zeros((32, 32), dtype=np.float64)
    mask[10:20, 10:20] = 2.0

    with np.testing.assert_raises(ValueError):
        solve_constant_kernel(
            reference,
            target,
            components,
            kernel_shape=(9, 9),
            fit_mask=mask,
            background_degree=0,
            flux_conserve=False,
        )
