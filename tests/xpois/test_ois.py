# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import pickle
from dataclasses import FrozenInstanceError, replace
from typing import Any

import numpy as np
import pytest
from scipy.signal import fftconvolve

from cuphoton.xpois.ois import (
    DeviceConstantKernelFitResult,
    GaussianBasisComponent,
    _accumulate_normal_equations,
    _accumulate_normal_equations_cutile,
    _default_fit_mask,
    background_design,
    build_compact_source_stamp_mask,
    build_gaussian_polynomial_basis,
    make_stamp_mask,
    resolve_backend,
    solve_constant_kernel,
    solve_constant_kernel_device,
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


def _require_cupy_device(*, minimum_count: int = 1):
    cp = pytest.importorskip("cupy")
    try:
        count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:
        pytest.skip(f"CuPy CUDA runtime is not usable: {exc}")
    if count < minimum_count:
        pytest.skip(
            f"test requires {minimum_count} visible CUDA device(s); "
            f"got {count}"
        )
    return cp


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
    _require_cupy_device()

    cpu = _solve_parity_fixture_backend("cpu")
    gpu = _solve_parity_fixture_backend("cupy")
    _assert_constant_kernel_matches_cpu(cpu, gpu, backend="cupy")


@pytest.mark.parametrize("input_location", ["host", "device"])
def test_solve_constant_kernel_device_matches_cpu_and_stays_on_device(
    monkeypatch: pytest.MonkeyPatch,
    input_location: str,
) -> None:
    cp = _require_cupy_device()

    reference, target, variance, components = (
        _constant_kernel_parity_fixture()
    )
    cpu = _solve_parity_fixture_backend("cpu")
    asarray = cp.asarray if input_location == "device" else np.asarray
    active_device = cp.cuda.Device()
    reference_device = asarray(reference)
    target_device = asarray(target)
    variance_device = asarray(variance)
    fit_mask = asarray(np.ones_like(target, dtype=bool))
    reference_before = reference_device.copy()
    target_before = target_device.copy()
    variance_before = variance_device.copy()

    with monkeypatch.context() as context:
        context.setattr(
            cp,
            "asnumpy",
            lambda *_args, **_kwargs: pytest.fail(
                "device solve materialized an array on the host"
            ),
        )
        result = solve_constant_kernel_device(
            reference_device,
            target_device,
            components,
            kernel_shape=(9, 9),
            variance=variance_device,
            fit_mask=fit_mask,
            background_degree=1,
            flux_conserve=np.bool_(False),
        )

    assert isinstance(result, DeviceConstantKernelFitResult)
    assert result.schema == "cuphoton.xpois.device-fit-result/v1"
    assert result.solver == "constant"
    assert result.backend == "cupy"
    assert result.result_location == "device"
    assert result.flux_conserve is False
    assert result.device_id == int(active_device.id)
    for value in (
        result.kernel,
        result.matched,
        result.residual,
        result.fit_mask,
        result.background,
        result.kernel_coefficients,
        result.background_coefficients,
        result.basis_kernels,
    ):
        assert isinstance(value, cp.ndarray)
        assert value.device == active_device
    assert result.kernel.dtype == cp.float64
    assert result.matched.dtype == cp.float64
    assert result.residual.dtype == cp.float64
    assert result.fit_mask.dtype == cp.bool_
    assert result.background.dtype == cp.float64
    assert result.kernel_coefficients.dtype == cp.float64
    assert result.background_coefficients.dtype == cp.float64
    assert result.basis_kernels.dtype == cp.float64
    assert result.matched.shape == reference_device.shape
    assert result.residual.shape == reference_device.shape
    assert result.fit_mask.shape == reference_device.shape
    assert result.background.shape == reference_device.shape
    assert result.basis_kernels.shape[1:] == result.kernel.shape
    assert result.kernel_coefficients.size == result.basis_kernels.shape[0]
    assert len(result.basis_terms) == result.basis_kernels.shape[0]
    assert np.array_equal(
        cp.asnumpy(reference_device), cp.asnumpy(reference_before)
    )
    assert np.array_equal(
        cp.asnumpy(target_device), cp.asnumpy(target_before)
    )
    assert np.array_equal(
        cp.asnumpy(variance_device), cp.asnumpy(variance_before)
    )

    finite = np.isfinite(cpu.matched)
    assert result.fit_pixel_count == cpu.fit_pixel_count
    assert result.dof == cpu.dof
    assert np.array_equal(cp.asnumpy(result.fit_mask), cpu.fit_mask)
    assert np.allclose(
        cp.asnumpy(result.kernel),
        cpu.kernel,
        rtol=1e-9,
        atol=1e-8,
    )
    assert np.allclose(
        cp.asnumpy(result.background),
        cpu.background,
        rtol=1e-9,
        atol=1e-8,
    )
    assert np.allclose(
        cp.asnumpy(result.matched)[finite],
        cpu.matched[finite],
        rtol=1e-9,
        atol=1e-8,
    )
    assert np.allclose(
        cp.asnumpy(result.residual)[finite],
        cpu.residual[finite],
        rtol=1e-9,
        atol=1e-8,
    )
    assert np.allclose(
        cp.asnumpy(result.kernel_coefficients),
        cpu.kernel_coefficients,
        rtol=1e-9,
        atol=1e-8,
    )
    assert np.allclose(
        cp.asnumpy(result.background_coefficients),
        cpu.background_coefficients,
        rtol=1e-9,
        atol=1e-8,
    )
    assert np.allclose(
        cp.asnumpy(result.basis_kernels),
        cpu.basis_kernels,
        rtol=0.0,
        atol=0.0,
    )
    assert np.isclose(result.chi2, cpu.chi2, rtol=1e-9, atol=1e-8)

    with pytest.raises(FrozenInstanceError):
        result.device_id = result.device_id + 1  # type: ignore[misc]
    with pytest.raises(ValueError, match="CUDA device"):
        replace(result, device_id=result.device_id + 1)
    with pytest.raises(TypeError, match="float64"):
        replace(result, kernel=result.kernel.astype(cp.float32))
    with pytest.raises(ValueError, match="matched image shape"):
        replace(result, background=result.background[:-1])
    with pytest.raises(ValueError):
        replace(result, solver="spatial-als")  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="cannot be pickled"):
        pickle.dumps(result)

    residual_sum = float(cp.nansum(result.residual).item())
    del reference_device, target_device, variance_device
    gc.collect()
    assert float(cp.nansum(result.residual).item()) == residual_sum


@pytest.mark.parametrize(
    "argument_name",
    ["reference", "target", "variance", "fit_mask"],
)
def test_solve_constant_kernel_device_rejects_wrong_device_inputs(
    argument_name: str,
) -> None:
    cp = _require_cupy_device(minimum_count=2)
    active_device_id = int(cp.cuda.runtime.getDevice())
    other_device_id = next(
        device_id
        for device_id in range(int(cp.cuda.runtime.getDeviceCount()))
        if device_id != active_device_id
    )
    try:
        with cp.cuda.Device(other_device_id):
            wrong_device_values = {
                "reference": cp.ones((16, 16), dtype=cp.float64),
                "target": cp.ones((16, 16), dtype=cp.float64),
                "variance": cp.ones((16, 16), dtype=cp.float64),
                "fit_mask": cp.ones((16, 16), dtype=cp.bool_),
            }
    except Exception as exc:
        pytest.skip(f"second CUDA device is not usable: {exc}")

    arguments = {
        "reference": cp.ones((16, 16), dtype=cp.float64),
        "target": cp.ones((16, 16), dtype=cp.float64),
        "variance": cp.ones((16, 16), dtype=cp.float64),
        "fit_mask": cp.ones((16, 16), dtype=cp.bool_),
    }
    arguments[argument_name] = wrong_device_values[argument_name]

    with pytest.raises(ValueError, match=argument_name):
        solve_constant_kernel_device(
            arguments["reference"],
            arguments["target"],
            [GaussianBasisComponent(sigma=1.5, degree=0)],
            kernel_shape=(9, 9),
            variance=arguments["variance"],
            fit_mask=arguments["fit_mask"],
        )


def test_solve_constant_kernel_device_reports_no_visible_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeRuntime:
        @staticmethod
        def getDeviceCount() -> int:
            return 0

    class FakeCuda:
        runtime = FakeRuntime()

    class FakeCupy:
        cuda = FakeCuda()

    monkeypatch.setattr(
        "cuphoton.xpois.ois._load_cupy",
        lambda: (FakeCupy(), None),
    )

    with pytest.raises(
        RuntimeError,
        match="requires at least one visible CUDA device",
    ):
        solve_constant_kernel_device(
            np.ones((16, 16)),
            np.ones((16, 16)),
            [GaussianBasisComponent(sigma=1.5, degree=0)],
            kernel_shape=(9, 9),
        )


def test_solve_constant_kernel_device_masks_non_finite_pixels_exactly() -> (
    None
):
    cp = _require_cupy_device()

    reference = _reference_image((64, 64))
    components = [GaussianBasisComponent(sigma=1.8, degree=0)]
    basis, _ = build_gaussian_polynomial_basis((9, 9), components)
    target = fftconvolve(reference, basis[0], mode="same")
    variance = np.full_like(target, 1.0)
    fit_mask = np.ones_like(target, dtype=np.uint8)
    reference[24, 24] = np.nan
    target[30, 30] = np.nan
    variance[34, 34] = np.inf

    cpu = solve_constant_kernel(
        reference,
        target,
        components,
        kernel_shape=(9, 9),
        variance=variance,
        fit_mask=fit_mask,
        background_degree=0,
        backend="cpu",
    )
    device = solve_constant_kernel_device(
        cp.asarray(reference),
        cp.asarray(target),
        components,
        kernel_shape=(9, 9),
        variance=cp.asarray(variance),
        fit_mask=cp.asarray(fit_mask),
        background_degree=0,
    )

    assert np.array_equal(cp.asnumpy(device.fit_mask), cpu.fit_mask)
    assert device.fit_pixel_count == cpu.fit_pixel_count


def test_solve_constant_kernel_numba_cuda_matches_cpu() -> None:
    cuda = pytest.importorskip("numba.cuda")
    if not cuda.is_available():
        pytest.skip("Numba CUDA runtime is not usable")

    cpu = _solve_parity_fixture_backend("cpu")
    gpu = _solve_parity_fixture_backend("numba-cuda")
    _assert_constant_kernel_matches_cpu(cpu, gpu, backend="numba-cuda")


def test_solve_constant_kernel_cutile_matches_cpu() -> None:
    _require_cupy_device()
    try:
        import cuda.tile  # noqa: F401
    except Exception as exc:
        pytest.skip(f"cuda.tile is not usable: {exc}")

    cpu = _solve_parity_fixture_backend("cpu")
    gpu = _solve_parity_fixture_backend("cutile")
    _assert_constant_kernel_matches_cpu(cpu, gpu, backend="cutile")


_SPARSE_ROWS_COMPONENTS = (
    GaussianBasisComponent(sigma=1.5, degree=2),
    GaussianBasisComponent(sigma=3.0, degree=1),
    GaussianBasisComponent(sigma=6.0, degree=0),
)


def _require_cutile() -> Any:
    cp = pytest.importorskip("cupy")
    try:
        import cuda.tile as ct
    except Exception as exc:
        pytest.skip(f"cuda.tile is not usable: {exc}")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA devices visible")
    except Exception as exc:
        pytest.skip(f"CuPy CUDA runtime is not usable: {exc}")
    return ct


def _sparse_rows_fixture(
    *,
    component_count: int,
    background_degree: int,
    constant_basis: bool = False,
) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(1219)
    shape = (97, 91)
    kernel_shape = (15, 15)
    reference = rng.normal(size=shape)
    target = rng.normal(size=shape)
    variance = rng.uniform(0.25, 2.0, size=shape)
    mask = _default_fit_mask(shape, kernel_shape)
    mask &= rng.random(size=shape) < 0.61
    basis, _ = build_gaussian_polynomial_basis(
        kernel_shape,
        (GaussianBasisComponent(sigma=1.5, degree=0),)
        if constant_basis
        else _SPARSE_ROWS_COMPONENTS[:component_count],
    )
    background = background_design(shape, degree=background_degree)
    return reference, target, variance, mask, basis, background


@pytest.mark.parametrize(
    "large_tiles",
    [False, True],
    ids=["rows32", "rows128"],
)
@pytest.mark.parametrize(
    ("component_count", "background_degree", "column_count"),
    [
        pytest.param(1, 0, 2, id="width4"),
        pytest.param(3, 0, 11, id="width16"),
        pytest.param(2, 2, 15, id="width16-full"),
        pytest.param(3, 2, 16, id="width32"),
    ],
)
def test_cutile_mma_normal_equations_match_cpu_for_sparse_rows(
    monkeypatch: pytest.MonkeyPatch,
    component_count: int,
    background_degree: int,
    column_count: int,
    large_tiles: bool,
) -> None:
    _require_cutile()
    if large_tiles:
        # Force the 128-row specialization; the fixture row count is not a
        # multiple of either tile size, so the last tile is always partial.
        monkeypatch.setattr(
            "cuphoton.xpois.ois._CUTILE_MMA_LARGE_ROW_THRESHOLD",
            0,
        )
    reference, target, variance, mask, basis, background = (
        _sparse_rows_fixture(
            component_count=component_count,
            background_degree=background_degree,
            constant_basis=column_count == 2,
        )
    )
    assert basis.shape[0] + background.shape[0] == column_count

    expected_gram, expected_rhs, expected_rows = _accumulate_normal_equations(
        reference,
        target,
        variance,
        mask,
        basis,
        background,
    )
    gram, rhs, rows = _accumulate_normal_equations_cutile(
        reference,
        target,
        variance,
        mask,
        basis,
        background,
    )

    assert rows == expected_rows
    assert np.allclose(gram, expected_gram, rtol=2e-12, atol=1e-8)
    assert np.allclose(rhs, expected_rhs, rtol=2e-12, atol=1e-9)


def test_cutile_mma_kernel_signature_ignores_row_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ct = _require_cutile()
    reference, target, variance, mask, basis, background = (
        _sparse_rows_fixture(component_count=3, background_degree=0)
    )
    launches: list[tuple[Any, tuple[Any, ...]]] = []
    real_launch = ct.launch

    def recording_launch(stream, grid, kernel, kernel_args):
        launches.append((kernel, kernel_args))
        return real_launch(stream, grid, kernel, kernel_args)

    monkeypatch.setattr(ct, "launch", recording_launch)
    # Drop one fit pixel so only the row count changes. Neither count is a
    # multiple of 16, so cuda.tile's array-length divisibility
    # specialization does not apply to the index arrays.
    ys, xs = np.nonzero(mask)
    smaller_mask = mask.copy()
    smaller_mask[ys[0], xs[0]] = False
    assert int(mask.sum()) % 16 != 0
    assert int(smaller_mask.sum()) % 16 != 0
    for fit_mask in (mask, smaller_mask):
        _accumulate_normal_equations_cutile(
            reference,
            target,
            variance,
            fit_mask,
            basis,
            background,
        )

    convention = ct.compilation.CallingConvention.cutile_python_v1()
    signatures = [
        ct.compilation.KernelSignature.from_kernel_args(
            kernel,
            kernel_args,
            convention,
        )
        for kernel, kernel_args in launches
    ]
    assert len(signatures) == 2
    assert signatures[0].parameters == signatures[1].parameters


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


@pytest.mark.parametrize(
    ("component", "message"),
    [
        (GaussianBasisComponent(sigma=np.nan, degree=0), "sigma must be"),
        (GaussianBasisComponent(sigma=1.2, degree=1.5), "degree must be"),
    ],
)
def test_solve_separable_kernel_rejects_invalid_line_basis_components(
    component: GaussianBasisComponent,
    message: str,
) -> None:
    image = np.ones((9, 9))

    with pytest.raises(ValueError, match=message):
        solve_separable_kernel(image, image, [component], kernel_shape=(3, 3))
