# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Numerical and buffer checks for installed structured primitives."""

from __future__ import annotations

import cmath
import math
import os
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from cuphoton.xray import _structured_native as native

GPU_REQUIRED = pytest.mark.skipif(
    os.environ.get("CUPHOTON_XRAY_TEST_GPU") != "1",
    reason="Set CUPHOTON_XRAY_TEST_GPU=1 for CUDA primitive checks",
)


@pytest.fixture(autouse=True)
def extension():
    return pytest.importorskip("cuphoton.xray._structured_ext")


def assert_roots(actual, expected, tolerance=2e-11):
    remaining = list(expected)
    assert len(actual) == len(remaining)
    for value in actual:
        nearest = min(
            range(len(remaining)), key=lambda i: abs(value - remaining[i])
        )
        assert abs(value - remaining.pop(nearest)) <= tolerance


@pytest.mark.parametrize("degree", [1, 2, 5, 16, 32])
def test_known_roots_and_scaled_coefficients(degree):
    polynomial = np.array([1.0] + [0.0] * (degree - 1) + [-1.0])
    rows = np.array([scale * polynomial for scale in (1.0, 0.125, 1024.0)])
    roots, stats, _ = native.solve_roots(rows, workers=2)
    expected = [cmath.exp(2j * math.pi * k / degree) for k in range(degree)]
    for values, status in zip(roots, stats, strict=True):
        assert status[:3].tolist() == [0, degree, 0]
        assert_roots(values, expected)


def test_bad_rows_are_isolated_and_zero_roots_preserved():
    rows = np.array(
        [
            [1, -1, 0, 0],
            [0, 1, 2, 3],
            [1, math.nan, 2, 3],
            [1e308, 1, 0, 1e-308],
            [1, 0, 0, -1],
        ]
    )
    roots, stats, errors = native.solve_roots(rows, workers=2)
    assert stats[:, 0].tolist() == [0, 2, 2, 4, 0]
    assert_roots(roots[0], [0, 0, 1])
    assert_roots(
        roots[4], [cmath.exp(2j * math.pi * k / 3) for k in range(3)]
    )
    for row in (1, 2, 3):
        assert stats[row, 1:3].tolist() == [0, 3]
        assert (
            np.isnan(roots[row].real).all()
            and np.isnan(roots[row].imag).all()
        )
        assert bytes(errors[row]).split(b"\0", 1)[0]


def test_sweep_cap_preserves_failure():
    roots, stats, _ = native.solve_roots(
        np.array([[1, 0.1, 0.2, 0.3, 0.4, 1]]), max_sweeps=0
    )
    assert stats[0, :4].tolist() == [1, 0, 5, 0]
    assert np.isnan(roots).all()


@pytest.mark.parametrize(
    "kwargs", [{"workers": 0}, {"workers": 257}, {"max_sweeps": -2}]
)
def test_invalid_dispatch(kwargs):
    with pytest.raises(ValueError, match="limits"):
        native.solve_roots(np.array([[1.0, 0, 1]]), **kwargs)


def test_independent_concurrent_calls():
    rows = np.array([[1.0] + [0.0] * 15 + [-1.0]])
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(native.solve_roots, [rows, rows]))
    for a, b in zip(results[0], results[1], strict=True):
        np.testing.assert_array_equal(a, b)


def test_native_binding_rejects_alias_and_readonly_outputs(extension):
    polynomial = np.array([[1.0, 0.0, -1.0]])
    storage = np.empty(8)
    storage[:3] = polynomial[0]
    stats = np.empty((1, 8), np.int32)
    errors = np.empty((1, 256), np.uint8)
    with pytest.raises(ValueError, match="overlap"):
        extension.solve_roots(
            storage[:3].reshape(1, 3),
            storage[2:6].reshape(1, 2, 2),
            stats,
            errors,
            1,
            -1,
        )
    output = np.empty((1, 2, 2))
    output.flags.writeable = False
    with pytest.raises((ValueError, BufferError)):
        extension.solve_roots(polynomial, output, stats, errors, 1, -1)
    with pytest.raises(ValueError, match="dtype"):
        extension.solve_roots(
            polynomial.astype(np.float32), output.copy(), stats, errors, 1, -1
        )


def refine(seed, separation=0.01):
    polynomial = np.array([1.0] + [0.0] * 596 + [-1.0])
    z = complex(seed)
    return native.refine_selected(
        polynomial, np.array([[z.real, z.imag]]), np.array([separation])
    )


def test_double_double_refinement_preserves_convergence_words():
    output, history, status = refine(1 + 2**-20)
    assert status[0, 0] == 0
    assert output.tolist() == [[1.0, 0.0]]
    steps = status[0, 1]
    assert 2 <= steps <= 8
    final = history[0, steps - 1]
    assert final[4] <= 2**-80 and final[5] <= 2**-80 and final[7] >= 2


@pytest.mark.parametrize(
    "seed,separation,status",
    [
        (math.nan, 0.01, 1),
        (2.0**451, 0.01, 2),
        (1 + 2**-30, 0.01, 2),
        (0.0, 0.01, 3),
        (1 + 2**-30, 1e-12, 4),
        (1.1, 100.0, 5),
        (1.0, 0.0, 6),
    ],
)
def test_double_double_rejects_unqualified_roots(seed, separation, status):
    _, _, actual = refine(seed, separation)
    assert actual[0, 0] == status


def test_exact_real_and_complex_refinement():
    polynomial = np.array([1.0] + [0.0] * 595 + [-1.0, 0.0])
    seeds = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    output, _, status = native.refine_selected(
        polynomial, seeds, np.full(3, 0.01)
    )
    np.testing.assert_array_equal(output, seeds)
    np.testing.assert_array_equal(status, [[0, 2], [0, 2], [0, 2]])


def test_refinement_rejects_unsupported_degree():
    with pytest.raises(ValueError, match="degree 597"):
        native.refine_selected(np.ones(597), np.ones((1, 2)), np.ones(1))


@pytest.mark.parametrize("runtime_version", [12090, 14000])
def test_gpu_rejects_unsupported_runtime_before_device_setup(runtime_version):
    from cuphoton.xray._structured_gpu import StructuredRootsGPU

    cp = SimpleNamespace(
        cuda=SimpleNamespace(
            runtime=SimpleNamespace(runtimeGetVersion=lambda: runtime_version)
        )
    )
    with pytest.raises(RuntimeError, match="requires CUDA 13"):
        StructuredRootsGPU(cp)


@pytest.mark.xray_gpu
@GPU_REQUIRED
def test_gpu_packaged_kernel_cpu_parity_status_and_lifecycle():
    import cupy as cp

    from cuphoton.xray._structured_gpu import StructuredRootsGPU

    rows = np.array([[1.0] + [0.0] * 15 + [-1.0]] * 3)
    rows[1, 0] = 0
    rows[2, 1] = math.nan
    cpu_roots, cpu_stats, _ = native.solve_roots(rows)
    solver = StructuredRootsGPU(cp)
    try:
        roots, stats, _ = solver.solve(rows)
        np.testing.assert_array_equal(stats[:, 0], cpu_stats[:, 0])
        assert_roots(roots[0], cpu_roots[0])
        with ThreadPoolExecutor(max_workers=1) as pool:
            with pytest.raises(RuntimeError, match="owner thread"):
                pool.submit(solver.solve, rows).result()
        _, capped, _ = solver.solve(
            np.array([[1, 0.1, 0.2, 0.3, 0.4, 1]]), max_sweeps=0
        )
        assert capped[0, :4].tolist() == [1, 0, 5, 0]
    finally:
        solver.close()
    with pytest.raises(RuntimeError, match="closed"):
        solver.solve(rows)


def degree597_rows():
    constants = (0.5, 2.0, -1.25)
    rows = np.zeros((len(constants), 598))
    rows[:, 0] = 1
    rows[:, -1] = -np.asarray(constants)
    expected = np.array(
        [
            [
                cmath.rect(
                    abs(value) ** (1 / 597),
                    (cmath.phase(value) + 2 * math.pi * k) / 597,
                )
                for k in range(597)
            ]
            for value in constants
        ]
    )
    return rows, expected


def check_degree597_refinement(rows, roots, stats, expected):
    np.testing.assert_array_equal(stats[:, :3], [[0, 597, 0]] * len(rows))
    assert np.isfinite(roots).all()
    refined = []
    for polynomial, values, reference in zip(
        rows, roots, expected, strict=True
    ):
        assert_roots(values, reference)
        # Include a near-real root and both signs of a complex mode.
        targets = reference[[0, 37, 560]]
        indices = [int(np.argmin(np.abs(values - z))) for z in targets]
        seeds = np.column_stack((values[indices].real, values[indices].imag))
        separation = np.array(
            [
                np.min(np.abs(np.delete(values, index) - values[index]))
                for index in indices
            ]
        )
        output, history, status = native.refine_selected(
            polynomial, seeds, separation
        )
        np.testing.assert_array_equal(status[:, 0], 0)
        assert np.all((status[:, 1] >= 2) & (status[:, 1] <= 8))
        final = history[np.arange(len(status)), status[:, 1] - 1]
        assert np.all(final[:, 4:6] <= 2**-80)
        assert np.all(final[:, 7] >= 2)
        values = output[:, 0] + 1j * output[:, 1]
        np.testing.assert_allclose(values, targets, rtol=0, atol=2e-15)
        refined.append(output)
    return np.array(refined)


def test_degree597_roots_and_double_double_refinement():
    rows, expected = degree597_rows()
    roots, stats, _ = native.solve_roots(rows)
    check_degree597_refinement(rows, roots, stats, expected)


@pytest.mark.xray_gpu
@GPU_REQUIRED
def test_gpu_degree597_roots_status_and_refined_value_parity():
    import cupy as cp

    from cuphoton.xray._structured_gpu import StructuredRootsGPU

    rows, expected = degree597_rows()
    cpu_roots, cpu_stats, _ = native.solve_roots(rows)
    cpu_refined = check_degree597_refinement(
        rows, cpu_roots, cpu_stats, expected
    )
    solver = StructuredRootsGPU(cp)
    try:
        roots, stats, _ = solver.solve(rows)
    finally:
        solver.close()
    np.testing.assert_array_equal(stats[:, :3], cpu_stats[:, :3])
    for actual, reference in zip(roots, cpu_roots, strict=True):
        assert_roots(actual, reference)
    gpu_refined = check_degree597_refinement(rows, roots, stats, expected)
    np.testing.assert_array_equal(gpu_refined, cpu_refined)


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("gpu", marks=[pytest.mark.xray_gpu, GPU_REQUIRED])],
)
def test_native_noisy_trace_complete_fit(device):
    from cuphoton.xray import linear_prediction as lp
    from cuphoton.xray import structured

    time, rows = lp.synthetic_trace_batch(samples=798, traces=1)
    trace = rows[0] + 1e-3 * np.random.default_rng(0).standard_normal(
        time.size
    )
    prepared = structured.prepare_fit(time, trace)
    rows = structured.prediction_polynomials([prepared])
    roots, stats, _ = native.solve_roots(rows)
    assert stats[0, :3].tolist() == [0, 597, 0]
    cpu = structured.finish_fit(prepared, roots[0], fit_diagnostics=True)
    assert cpu.backend == "structured-cpu"
    assert cpu.roots_stats is not None
    assert cpu.roots_stats.array_module == "numpy"
    assert cpu.decay.size > 0
    assert cpu.reconstruction.shape == (798,)
    assert cpu.time_components.shape == (798, cpu.decay.size)
    assert cpu.spectrum_components.shape == (1000, cpu.decay.size)
    assert cpu.diagnostics is not None and cpu.diagnostics.p2_rank > 1
    fields = (
        "time_components",
        "reconstruction",
        "frequency",
        "spectrum_components",
        "spectrum_total",
        "angular_frequency",
        "decay",
        "amplitude",
        "phase",
        "singular_values",
    )
    for field in fields:
        assert np.isfinite(getattr(cpu, field)).all(), field
    assert math.isfinite(cpu.chi2)
    if device == "cpu":
        return

    import cupy as cp

    from cuphoton.xray._structured_gpu import StructuredRootsGPU

    solver = StructuredRootsGPU(cp)
    try:
        gpu_roots, gpu_stats, _ = solver.solve(rows)
    finally:
        solver.close()
    np.testing.assert_array_equal(gpu_stats[:, :3], stats[:, :3])
    gpu = structured.finish_fit(
        prepared, gpu_roots[0], root_device="gpu", fit_diagnostics=True
    )
    assert gpu.backend == "structured-gpu"
    assert gpu.roots_stats is not None
    assert gpu.roots_stats.array_module == "cupy"
    assert gpu.selected_model_order == cpu.selected_model_order
    assert gpu.decaying_root_count == cpu.decaying_root_count
    for field in fields:
        np.testing.assert_allclose(
            getattr(gpu, field), getattr(cpu, field), rtol=1e-9, atol=1e-12
        )
    np.testing.assert_allclose(gpu.chi2, cpu.chi2, rtol=1e-9, atol=1e-12)
    assert gpu.diagnostics is not None
    assert gpu.diagnostics.p2_rank == cpu.diagnostics.p2_rank
    np.testing.assert_allclose(
        gpu.diagnostics.p2_singular_values,
        cpu.diagnostics.p2_singular_values,
        rtol=1e-9,
        atol=1e-12,
    )
