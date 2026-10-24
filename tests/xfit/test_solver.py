# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import fields
from math import fsum, sqrt

import numpy as np
import pytest
from scipy.optimize import least_squares

from cuphoton.xfit import (
    BatchedLeastSquaresProblem,
    GaussianDipoleModel,
    LMConfig,
    LMStatus,
    batched_levenberg_marquardt,
    fit_dipoles,
)


@pytest.mark.parametrize("use_finite_difference", [False, True])
def test_batched_solver_matches_scipy_for_nonlinear_fits(
    use_finite_difference: bool,
) -> None:
    sample = np.linspace(0.0, 1.0, 12)
    truth = np.asarray([[2.0, -0.7], [0.8, 1.1], [3.2, 0.25]])
    observations = truth[:, :1] * np.exp(truth[:, 1:] * sample)

    def residual(x, *, indices, out=None):
        value = x[:, :1] * np.exp(x[:, 1:] * sample) - observations[indices]
        if out is not None:
            out[:] = value
            return out
        return value

    def jacobian(x, *, indices, out=None):
        del indices
        exponential = np.exp(x[:, 1:] * sample)
        value = np.stack(
            (exponential, x[:, :1] * sample * exponential), axis=1
        )
        if out is not None:
            out[:] = value
            return out
        return value

    initial = np.asarray([[1.6, -0.3], [1.0, 0.7], [2.7, 0.0]])
    result = batched_levenberg_marquardt(
        BatchedLeastSquaresProblem(
            residual,
            None if use_finite_difference else jacobian,
        ),
        initial,
        config=LMConfig(use_finite_difference=use_finite_difference),
    )
    scipy_solutions = np.stack(
        [
            least_squares(
                lambda value, row=row: (
                    value[0] * np.exp(value[1] * sample) - observations[row]
                ),
                initial[row],
            ).x
            for row in range(initial.shape[0])
        ]
    )

    assert result.converged.all()
    assert np.allclose(
        result.parameters, scipy_solutions, rtol=2e-6, atol=2e-7
    )
    assert np.all(result.rank == 2)
    assert np.isfinite(result.covariance).all()


def test_active_batch_is_compacted_after_independent_termination() -> None:
    target = np.asarray([[0.0], [2.0], [-3.0]])
    seen_indices: list[tuple[int, ...]] = []

    def residual(x, *, indices, out=None):
        value = x - target[indices]
        if out is not None:
            out[:] = value
            return out
        return value

    def jacobian(x, *, indices, out=None):
        seen_indices.append(tuple(int(value) for value in indices))
        value = np.ones((x.shape[0], 1, 1))
        if out is not None:
            out[:] = value
            return out
        return value

    result = batched_levenberg_marquardt(
        BatchedLeastSquaresProblem(residual, jacobian),
        np.zeros((3, 1)),
        config=LMConfig(max_evaluations=3),
    )

    assert seen_indices[0] == (0, 1, 2)
    assert seen_indices[1] == (1, 2)
    assert result.status[0] == LMStatus.CONVERGED_G_TOL
    assert np.all(result.status[1:] == LMStatus.MAX_EVALUATIONS)
    assert result.evaluations.tolist() == [1, 3, 3]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("use_normal_equations", [False, True])
def test_mixed_batch_matches_independent_rows(
    dtype, use_normal_equations: bool
) -> None:
    # Interleave successful fits, initially invalid rows, an already solved
    # row, an invalid derivative, a wrong-signed derivative, and a residual
    # that becomes invalid at every trial. Each callback uses original row
    # identities even after the active batch shrinks.
    target = np.asarray([3.0, 1.0, 4.0, 2.0, -1.0, -2.0, 1.0], dtype=dtype)
    initial = np.asarray([0.0, 0.0, 4.0, 0.0, 0.0, 5.0, 0.0], dtype=dtype)
    slopes = np.arange(1, 8, dtype=dtype)

    def solve(rows):
        evaluated = np.zeros(rows.size, dtype=np.int32)

        def residual(x, *, indices):
            assert np.all(np.diff(indices) > 0)
            original = rows[indices]
            evaluated[indices] += 1
            value = (x - target[original, None]) * slopes[original, None]
            value[original == 1] = np.nan
            value[(original == 6) & (x[:, 0] != 0.0)] = np.nan
            return value

        def jacobian(x, *, indices):
            del x
            assert np.all(np.diff(indices) > 0)
            original = rows[indices]
            value = slopes[original, None, None].copy()
            value[original == 3] = np.nan
            value[original == 4] *= -1.0
            return value

        def normal_equations(x, residuals, *, indices):
            jac = jacobian(x, indices=indices)
            return (
                np.einsum("knm,km->kn", jac, residuals),
                np.einsum("knm,kpm->knp", jac, jac),
            )

        result = batched_levenberg_marquardt(
            BatchedLeastSquaresProblem(
                residual,
                jacobian,
                normal_equations=(
                    normal_equations if use_normal_equations else None
                ),
            ),
            initial[rows, None],
            config=LMConfig(max_evaluations=6),
        )
        np.testing.assert_array_equal(result.evaluations, evaluated)
        return result

    batched = solve(np.arange(initial.size))
    independent = [solve(np.asarray([row])) for row in range(initial.size)]
    for field in fields(batched):
        expected = np.concatenate(
            [getattr(result, field.name) for result in independent]
        )
        np.testing.assert_array_equal(getattr(batched, field.name), expected)

    assert batched.converged.tolist() == [
        True,
        False,
        True,
        False,
        False,
        True,
        False,
    ]
    assert batched.status[[1, 2, 3, 4, 6]].tolist() == [
        LMStatus.INVALID_RESIDUAL,
        LMStatus.CONVERGED_G_TOL,
        LMStatus.INVALID_RESIDUAL,
        LMStatus.MAX_EVALUATIONS,
        LMStatus.MAX_EVALUATIONS,
    ]
    assert batched.evaluations[[1, 2, 3, 4, 6]].tolist() == [1, 1, 1, 6, 6]


def test_nonfinite_step_does_not_displace_other_active_rows(
    monkeypatch,
) -> None:
    original_solve = np.linalg.solve

    def solve_with_one_failed_row(system, rhs):
        step = original_solve(system, rhs)
        step[rhs[:, 0, 0] == -2.0] = np.nan
        return step

    monkeypatch.setattr(np.linalg, "solve", solve_with_one_failed_row)

    def residual(x, *, indices):
        del indices
        return x.copy()

    def jacobian(x, *, indices):
        del indices
        return np.ones((x.shape[0], 1, 1))

    result = batched_levenberg_marquardt(
        BatchedLeastSquaresProblem(residual, jacobian),
        np.asarray([[1.0], [2.0], [3.0]]),
        config=LMConfig(f_tol=1.0, max_evaluations=2),
    )

    assert result.status.tolist() == [
        LMStatus.CONVERGED_F_TOL,
        LMStatus.SINGULAR,
        LMStatus.CONVERGED_F_TOL,
    ]
    assert result.evaluations.tolist() == [2, 1, 2]
    np.testing.assert_array_equal(
        result.parameters[:, 0],
        np.asarray([1.0, 2.0, 3.0]) - np.asarray([1.0, 0.0, 3.0]) / 1.001,
    )
    assert result.rank.tolist() == [1, 1, 1]
    np.testing.assert_array_equal(result.residuals, result.parameters)
    np.testing.assert_array_equal(result.gradient, result.parameters)


def test_finite_difference_jacobian_converges_and_counts_evaluations() -> (
    None
):
    target = np.asarray([[1.5, -0.5], [-2.0, 3.0]])
    residual_rows_evaluated = 0

    def residual(x, *, indices, out=None):
        nonlocal residual_rows_evaluated
        residual_rows_evaluated += x.shape[0]
        value = x * x - target[indices] * target[indices]
        if out is not None:
            out[:] = value
            return out
        return value

    result = batched_levenberg_marquardt(
        BatchedLeastSquaresProblem(residual),
        np.asarray([[1.0, -1.0], [-1.0, 2.0]]),
        config=LMConfig(use_finite_difference=True),
    )

    assert result.converged.all()
    assert np.allclose(result.parameters, target, rtol=3e-5, atol=3e-5)
    assert np.all(result.evaluations > 1)
    assert residual_rows_evaluated == int(result.evaluations.sum())


def test_callbacks_do_not_need_to_accept_an_out_argument() -> None:
    target = np.asarray([[1.0], [-2.0]])

    def residual(x, *, indices):
        return x - target[indices]

    def jacobian(x, *, indices):
        del indices
        return np.ones((x.shape[0], 1, 1))

    result = batched_levenberg_marquardt(
        BatchedLeastSquaresProblem(residual, jacobian),
        np.zeros((2, 1)),
    )

    assert result.converged.all()
    assert np.allclose(result.parameters, target)


def test_specialized_normal_equations_preserve_final_diagnostics() -> None:
    sample = np.asarray([1.0, 2.0, 3.0])
    target = np.asarray([[2.0, -1.0], [-3.0, 4.0]])
    observations = target[:, :1] + target[:, 1:] * sample
    normal_equation_calls = 0
    jacobian_calls = 0

    def residual(x, *, indices):
        return x[:, :1] + x[:, 1:] * sample - observations[indices]

    def jacobian(x, *, indices):
        nonlocal jacobian_calls
        del indices
        jacobian_calls += 1
        return np.broadcast_to(
            np.stack((np.ones_like(sample), sample))[None, :, :],
            (x.shape[0], 2, sample.size),
        )

    def normal_equations(x, residuals, *, indices):
        nonlocal normal_equation_calls
        jac = jacobian(x, indices=indices)
        normal_equation_calls += 1
        return (
            np.einsum("knm,km->kn", jac, residuals),
            np.einsum("knm,kpm->knp", jac, jac),
        )

    result = batched_levenberg_marquardt(
        BatchedLeastSquaresProblem(
            residual,
            jacobian,
            normal_equations=normal_equations,
        ),
        np.zeros_like(target),
    )

    assert normal_equation_calls > 0
    assert jacobian_calls == normal_equation_calls + 1
    assert result.converged.all()
    assert np.allclose(result.parameters, target)
    assert result.rank.tolist() == [2, 2]
    assert np.isfinite(result.covariance).all()


@pytest.mark.parametrize(
    ("max_evaluations", "expected_status"),
    (
        pytest.param(4, LMStatus.MAX_EVALUATIONS, id="max-evaluations"),
        pytest.param(None, LMStatus.NO_PROGRESS, id="no-progress"),
    ),
)
def test_specialized_normal_equations_report_non_converged_diagnostics(
    max_evaluations: int | None, expected_status: LMStatus
) -> None:
    # A wrong-signed Jacobian rejects every step, so the last analytic
    # Jacobian stays current for rows that stop without converging.
    def residual(x, *, indices):
        del indices
        return x.copy()

    def jacobian(x, *, indices):
        del indices
        return np.broadcast_to(-np.eye(2)[None, :, :], (x.shape[0], 2, 2))

    def normal_equations(x, residuals, *, indices):
        jac = jacobian(x, indices=indices)
        return (
            np.einsum("knm,km->kn", jac, residuals),
            np.einsum("knm,kpm->knp", jac, jac),
        )

    config = LMConfig(max_evaluations=max_evaluations)
    plain = batched_levenberg_marquardt(
        BatchedLeastSquaresProblem(residual, jacobian),
        np.ones((2, 2)),
        config=config,
    )
    specialized = batched_levenberg_marquardt(
        BatchedLeastSquaresProblem(
            residual,
            jacobian,
            normal_equations=normal_equations,
        ),
        np.ones((2, 2)),
        config=config,
    )

    assert plain.status.tolist() == [expected_status] * 2
    assert np.array_equal(specialized.status, plain.status)
    assert np.array_equal(specialized.evaluations, plain.evaluations)
    assert np.array_equal(specialized.parameters, plain.parameters)
    assert plain.rank.tolist() == [2, 2]
    assert np.array_equal(specialized.rank, plain.rank)
    for field in ("jacobian", "gradient", "gn_hessian", "covariance"):
        assert np.isfinite(getattr(plain, field)).all()
        assert np.allclose(getattr(specialized, field), getattr(plain, field))


@pytest.mark.parametrize(
    ("dtype", "large_scale"), [(np.float32, 1.0e25), (np.float64, 1.0e200)]
)
def test_specialized_normal_overflow_preserves_finite_jacobian_results(
    dtype, large_scale
) -> None:
    scales = np.asarray([1.0, large_scale, np.nan], dtype=dtype)
    jacobian_calls: list[tuple[int, ...]] = []

    def residual(x, *, indices):
        del indices
        return np.zeros_like(x)

    def jacobian(x, *, indices):
        del x
        jacobian_calls.append(tuple(int(value) for value in indices))
        return scales[indices, None, None]

    def normal_equations(x, residuals, *, indices):
        del x
        jac = scales[indices, None, None]
        gradient = np.einsum("knm,km->kn", jac, residuals)
        hessian = np.einsum("knm,kpm->knp", jac, jac)
        # Callback arrays may be borrowed or readonly.
        gradient.flags.writeable = False
        hessian.flags.writeable = False
        return gradient, hessian

    initial = np.zeros((3, 1), dtype=dtype)
    with np.errstate(over="ignore", invalid="ignore"):
        plain = batched_levenberg_marquardt(
            BatchedLeastSquaresProblem(residual, jacobian), initial
        )
        jacobian_calls.clear()
        specialized = batched_levenberg_marquardt(
            BatchedLeastSquaresProblem(
                residual, jacobian, normal_equations=normal_equations
            ),
            initial,
        )

    assert jacobian_calls[0] == (1, 2)
    assert plain.status.tolist() == [
        LMStatus.CONVERGED_G_TOL,
        LMStatus.CONVERGED_G_TOL,
        LMStatus.INVALID_RESIDUAL,
    ]
    assert plain.evaluations.tolist() == [1, 1, 1]
    assert plain.rank.tolist() == [1, 1, -1]
    for name, expected in vars(plain).items():
        np.testing.assert_array_equal(getattr(specialized, name), expected)


def test_specialized_overflow_jacobian_shape_is_validated() -> None:
    def residual(x, *, indices):
        del indices
        return x.copy()

    def jacobian(x, *, indices):
        del indices
        return np.ones((x.shape[0], 2, 1))

    def normal_equations(x, residuals, *, indices):
        del indices
        return residuals.copy(), np.full((x.shape[0], 1, 1), np.inf)

    with pytest.raises(ValueError, match="jacobian must return shape"):
        batched_levenberg_marquardt(
            BatchedLeastSquaresProblem(
                residual, jacobian, normal_equations=normal_equations
            ),
            np.zeros((2, 1)),
        )


def test_specialized_normal_equations_require_an_analytic_jacobian() -> None:
    def residual(x, *, indices):
        del indices
        return x

    def normal_equations(x, residuals, *, indices):
        del residuals, indices
        return x, x[:, :, None] * x[:, None, :]

    with pytest.raises(TypeError, match="requires an analytic Jacobian"):
        BatchedLeastSquaresProblem(
            residual,
            normal_equations=normal_equations,
        )


@pytest.mark.parametrize(
    ("gradient_shape", "hessian_shape", "message"),
    (
        pytest.param((2,), (1, 1), "gradient must have shape", id="gradient"),
        pytest.param((1,), (1, 2), "Hessian must have shape", id="hessian"),
    ),
)
def test_specialized_normal_equation_shapes_are_validated(
    gradient_shape: tuple[int, ...],
    hessian_shape: tuple[int, ...],
    message: str,
) -> None:
    def residual(x, *, indices):
        del indices
        return x

    def jacobian(x, *, indices):
        del indices
        return np.ones((x.shape[0], 1, 1))

    def normal_equations(x, residuals, *, indices):
        del residuals, indices
        return (
            np.zeros((x.shape[0], *gradient_shape)),
            np.zeros((x.shape[0], *hessian_shape)),
        )

    with pytest.raises(ValueError, match=message):
        batched_levenberg_marquardt(
            BatchedLeastSquaresProblem(
                residual,
                jacobian,
                normal_equations=normal_equations,
            ),
            np.ones((2, 1)),
        )


@pytest.mark.parametrize(
    "fail_batch", [False, True], ids=["batch", "row-retry"]
)
def test_specialized_final_jacobian_shape_is_validated(
    fail_batch: bool,
) -> None:
    seen_indices: list[tuple[int, ...]] = []

    def residual(x, *, indices):
        del indices
        return x.copy()

    def jacobian(x, *, indices):
        seen_indices.append(tuple(int(value) for value in indices))
        if fail_batch and x.shape[0] > 1:
            raise RuntimeError("batched callback failed")
        return np.ones((x.shape[0], 2, 1))

    def normal_equations(x, residuals, *, indices):
        del indices
        return residuals.copy(), np.ones((x.shape[0], 1, 1))

    with pytest.raises(ValueError, match="jacobian must return shape"):
        batched_levenberg_marquardt(
            BatchedLeastSquaresProblem(
                residual,
                jacobian,
                normal_equations=normal_equations,
            ),
            np.zeros((2, 1)),
        )

    assert seen_indices == ([(0, 1), (0,)] if fail_batch else [(0, 1)])


def test_finite_difference_configuration_bypasses_normal_equations() -> None:
    target = np.asarray([[2.0]])

    def residual(x, *, indices):
        return x - target[indices]

    def jacobian(x, *, indices):
        del indices
        return np.ones((x.shape[0], 1, 1))

    def normal_equations(x, residuals, *, indices):
        del x, residuals, indices
        raise AssertionError("normal equations must not be called")

    result = batched_levenberg_marquardt(
        BatchedLeastSquaresProblem(
            residual,
            jacobian,
            normal_equations=normal_equations,
        ),
        np.zeros_like(target),
        config=LMConfig(use_finite_difference=True),
    )

    assert result.converged.all()
    assert np.allclose(result.parameters, target)


def test_invalid_residual_and_max_evaluations_have_stable_statuses() -> None:
    def residual(x, *, indices, out=None):
        value = x.copy()
        value[indices == 0] = np.nan
        if out is not None:
            out[:] = value
            return out
        return value

    result = batched_levenberg_marquardt(
        BatchedLeastSquaresProblem(residual),
        np.ones((2, 1)),
        config=LMConfig(max_evaluations=1),
    )

    assert result.status.tolist() == [
        LMStatus.INVALID_RESIDUAL,
        LMStatus.MAX_EVALUATIONS,
    ]


def test_rank_deficient_problem_returns_nan_covariance() -> None:
    target = np.asarray([3.0, -2.0])

    def residual(x, *, indices, out=None):
        value = (x[:, 0] + x[:, 1] - target[indices])[:, None]
        if out is not None:
            out[:] = value
            return out
        return value

    def jacobian(x, *, indices, out=None):
        del indices
        value = np.ones((x.shape[0], 2, 1))
        if out is not None:
            out[:] = value
            return out
        return value

    result = batched_levenberg_marquardt(
        BatchedLeastSquaresProblem(F=residual, J=jacobian),
        np.zeros((2, 2)),
    )

    assert result.converged.all()
    assert result.rank.tolist() == [1, 1]
    assert np.isnan(result.covariance).all()


def test_covariance_uses_jacobian_singular_values_for_rank() -> None:
    jacobian_matrix = np.diag([1.0, 1.0e-8])

    def residual(x, *, indices):
        del indices
        return x @ jacobian_matrix

    def jacobian(x, *, indices):
        del indices
        return np.broadcast_to(
            jacobian_matrix[None, :, :],
            (x.shape[0], 2, 2),
        )

    result = batched_levenberg_marquardt(
        BatchedLeastSquaresProblem(residual, jacobian),
        np.zeros((1, 2)),
    )

    assert result.rank.tolist() == [2]
    assert np.allclose(
        result.covariance[0],
        np.diag([1.0, 1.0e16]),
        rtol=1.0e-14,
    )


def test_final_diagnostics_factorize_batch_once(monkeypatch) -> None:
    target = np.asarray(
        [[1.0, -2.0], [-3.0, 4.0], [5.0, -6.0]], dtype=np.float64
    )
    jacobian_matrix = np.asarray(
        [[1.0, 0.0, 1.0], [0.0, 2.0, 1.0]], dtype=np.float64
    )

    def residual(x, *, indices):
        return (x - target[indices]) @ jacobian_matrix

    def jacobian(x, *, indices):
        del indices
        return np.broadcast_to(
            jacobian_matrix[None, :, :],
            (x.shape[0], *jacobian_matrix.shape),
        )

    original_svd = np.linalg.svd
    seen_shapes: list[tuple[int, ...]] = []

    def recording_svd(value, *args, **kwargs):
        seen_shapes.append(value.shape)
        return original_svd(value, *args, **kwargs)

    monkeypatch.setattr(np.linalg, "svd", recording_svd)
    result = batched_levenberg_marquardt(
        BatchedLeastSquaresProblem(residual, jacobian),
        target.copy(),
    )

    assert seen_shapes == [(3, 2, 3)]
    assert result.converged.all()
    assert result.rank.tolist() == [2, 2, 2]
    assert np.isfinite(result.covariance).all()


def test_failed_final_analytic_jacobian_isolates_the_failing_row() -> None:
    seen_indices: list[tuple[int, ...]] = []

    def residual(x, *, indices):
        del indices
        return x

    def jacobian(x, *, indices):
        batch = tuple(int(value) for value in indices)
        seen_indices.append(batch)
        # Row 1 fails in every post-convergence batch that contains it.
        if len(seen_indices) > 1 and 1 in batch:
            raise RuntimeError("row 1 failed")
        return np.broadcast_to(np.eye(2)[None, :, :], (x.shape[0], 2, 2))

    result = batched_levenberg_marquardt(
        BatchedLeastSquaresProblem(residual, jacobian),
        np.ones((3, 2)),
        config=LMConfig(f_tol=1.0),
    )

    assert seen_indices == [(0, 1, 2), (0, 1, 2), (0,), (1,), (2,)]
    assert result.converged.all()
    assert result.evaluations.tolist() == [2, 2, 2]
    assert result.rank.tolist() == [2, -1, 2]
    assert np.isfinite(result.covariance[[0, 2]]).all()
    assert np.isnan(result.covariance[1]).all()
    assert np.isnan(result.gradient[1]).all()


@pytest.mark.parametrize("failing_row", [False, True])
def test_final_diagnostics_fall_back_to_rows_when_batched_svd_fails(
    monkeypatch,
    failing_row,
) -> None:
    target = np.asarray(
        [[1.0, -2.0], [-3.0, 4.0], [5.0, -6.0]], dtype=np.float64
    )
    jacobian_matrix = np.asarray(
        [[1.0, 0.0, 1.0], [0.0, 2.0, 1.0]], dtype=np.float64
    )

    def residual(x, *, indices):
        return (x - target[indices]) @ jacobian_matrix

    def jacobian(x, *, indices):
        del indices
        return np.broadcast_to(
            jacobian_matrix[None, :, :],
            (x.shape[0], *jacobian_matrix.shape),
        )

    problem = BatchedLeastSquaresProblem(residual, jacobian)
    expected = batched_levenberg_marquardt(problem, target.copy())

    original_svd = np.linalg.svd
    seen_shapes: list[tuple[int, ...]] = []

    def row_only_svd(value, *args, **kwargs):
        seen_shapes.append(value.shape)
        if value.ndim == 3:
            raise RuntimeError("batched svd is unavailable")
        if failing_row and len(seen_shapes) == 3:
            raise RuntimeError("row 1 factorization failed")
        return original_svd(value, *args, **kwargs)

    monkeypatch.setattr(np.linalg, "svd", row_only_svd)
    result = batched_levenberg_marquardt(problem, target.copy())

    assert seen_shapes == [(3, 2, 3), *[(2, 3)] * 3]
    assert result.rank.tolist() == ([2, -1, 2] if failing_row else [2, 2, 2])
    assert np.array_equal(result.status, expected.status)
    for field in ("gradient", "gn_hessian"):
        assert np.allclose(
            getattr(result, field),
            getattr(expected, field),
            rtol=1.0e-12,
            atol=1.0e-12,
        )
    if failing_row:
        expected.covariance[1] = np.nan
    np.testing.assert_allclose(
        result.covariance,
        expected.covariance,
        rtol=1.0e-12,
        atol=1.0e-12,
        equal_nan=True,
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cupy_tall_jacobian_diagnostics_match_numpy(dtype) -> None:
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device is unavailable")
    except Exception:
        pytest.skip("CUDA runtime is unavailable")

    matrices = np.asarray(
        [
            [[1, 0, 1, 0, 1, 0], [0, 2, 0, 2, 0, 2]],
            [[1, 1, 2, 2, 3, 3], [2, 2, 4, 4, 6, 6]],
            [[3, 0, 0, 1, 1, 0], [0, 2, 1, 0, 1, 1]],
        ],
        dtype=dtype,
    )
    target = np.asarray([[1, -2], [-3, 4], [5, -6]], dtype=dtype)

    def solve(ap, initial):
        backend_matrices = ap.asarray(matrices)
        backend_target = ap.asarray(target)

        def residual(x, *, indices):
            delta = x - backend_target[indices]
            return ap.einsum("kn,knm->km", delta, backend_matrices[indices])

        def jacobian(x, *, indices):
            del x
            return backend_matrices[indices]

        return batched_levenberg_marquardt(
            BatchedLeastSquaresProblem(residual, jacobian), initial
        )

    cpu = solve(np, target.copy())
    gpu = solve(cp, cp.asarray(target))

    assert np.array_equal(cp.asnumpy(gpu.rank), cpu.rank)
    assert np.allclose(
        cp.asnumpy(gpu.covariance)[[0, 2]],
        cpu.covariance[[0, 2]],
        rtol=2.0e-5 if dtype is np.float32 else 2.0e-12,
        atol=2.0e-6 if dtype is np.float32 else 2.0e-13,
    )
    assert np.isnan(cp.asnumpy(gpu.covariance)[1]).all()


def test_solver_preserves_valid_directions_across_jacobian_scales() -> None:
    jacobian_matrix = np.diag([1.0e12, 1.0])
    truth = np.asarray([[0.0, 1.0]])
    observations = truth @ jacobian_matrix

    def residual(x, *, indices):
        return x @ jacobian_matrix - observations[indices]

    def jacobian(x, *, indices):
        del indices
        return np.broadcast_to(
            jacobian_matrix[None, :, :],
            (x.shape[0], 2, 2),
        )

    result = batched_levenberg_marquardt(
        BatchedLeastSquaresProblem(residual, jacobian),
        np.zeros((1, 2)),
    )

    assert result.converged.all()
    assert np.allclose(result.parameters, truth)
    assert result.rank.tolist() == [2]
    np.testing.assert_allclose(result.covariance[0], np.diag([1e-24, 1.0]))


def test_final_diagnostics_honor_the_residual_evaluation_limit() -> None:
    residual_rows_evaluated = 0

    def residual(x, *, indices):
        nonlocal residual_rows_evaluated
        del indices
        residual_rows_evaluated += x.shape[0]
        return x

    result = batched_levenberg_marquardt(
        BatchedLeastSquaresProblem(residual),
        np.ones((1, 2)),
        config=LMConfig(max_evaluations=1),
    )

    assert result.status.tolist() == [LMStatus.MAX_EVALUATIONS]
    assert result.evaluations.tolist() == [1]
    assert residual_rows_evaluated == 1
    assert result.rank.tolist() == [-1]
    assert np.isnan(result.covariance).all()


@pytest.mark.parametrize(
    ("failure_call", "expected_evaluations", "expected_rank"),
    (
        # Nothing was charged before the failure, so both rows still fit
        # one per-row retry within the six-evaluation budget.
        pytest.param(5, 6, 2, id="first-perturbation"),
        # One completed perturbation leaves five evaluations; a retry would
        # need two more, so the rows stay without diagnostics.
        pytest.param(6, 5, -1, id="last-perturbation"),
    ),
)
def test_failed_batched_final_finite_difference_is_counted_exactly(
    failure_call: int, expected_evaluations: int, expected_rank: int
) -> None:
    residual_calls = 0
    residual_rows_evaluated = np.zeros(2, dtype=np.int64)

    def residual(x, *, indices):
        nonlocal residual_calls
        residual_calls += 1
        if residual_calls == failure_call:
            raise RuntimeError("final finite-difference batch failed")
        residual_rows_evaluated[indices] += 1
        return x

    result = batched_levenberg_marquardt(
        BatchedLeastSquaresProblem(residual),
        np.ones((2, 2)),
        config=LMConfig(f_tol=1.0, max_evaluations=6),
    )

    assert result.converged.all()
    assert result.evaluations.tolist() == [expected_evaluations] * 2
    assert residual_rows_evaluated.tolist() == [expected_evaluations] * 2
    assert result.rank.tolist() == [expected_rank] * 2
    if expected_rank == 2:
        assert np.isfinite(result.covariance).all()
    else:
        assert np.isnan(result.covariance).all()


def test_failed_final_finite_difference_isolates_the_failing_row() -> None:
    seen_indices: list[tuple[int, ...]] = []
    residual_rows_evaluated = np.zeros(3, dtype=np.int64)

    def residual(x, *, indices):
        batch = tuple(int(value) for value in indices)
        seen_indices.append(batch)
        # Row 1 fails in every post-convergence batch that contains it.
        if len(seen_indices) > 4 and 1 in batch:
            raise RuntimeError("row 1 failed")
        residual_rows_evaluated[indices] += 1
        return x

    result = batched_levenberg_marquardt(
        BatchedLeastSquaresProblem(residual),
        np.ones((3, 2)),
        config=LMConfig(f_tol=1.0, max_evaluations=6),
    )

    assert seen_indices == [
        (0, 1, 2),
        (0, 1, 2),
        (0, 1, 2),
        (0, 1, 2),
        (0, 1, 2),
        (0,),
        (0,),
        (1,),
        (2,),
        (2,),
    ]
    assert result.converged.all()
    assert result.evaluations.tolist() == [6, 4, 6]
    assert residual_rows_evaluated.tolist() == [6, 4, 6]
    assert result.rank.tolist() == [2, -1, 2]
    assert np.isfinite(result.covariance[[0, 2]]).all()
    assert np.isnan(result.covariance[1]).all()


def test_solver_validates_configuration() -> None:
    def residual(x, *, indices, out=None):
        del indices, out
        return x

    with pytest.raises(ValueError, match="max_evaluations"):
        batched_levenberg_marquardt(
            BatchedLeastSquaresProblem(residual),
            np.ones((1, 1)),
            config=LMConfig(max_evaluations=0),
        )


def test_f_tol_continues_after_poorly_predicted_nonlinear_progress() -> None:
    coefficient = 1.001 - 1.0e-8

    def residual(x, *, indices):
        del indices
        return 1.0 - x + coefficient * x * x

    def jacobian(x, *, indices):
        del indices
        return (-1.0 + 2.0 * coefficient * x)[:, :, None]

    # The first damped step is almost one: actual cost improvement is about
    # 1e-8, although the linear model predicts about 0.5. Small actual
    # progress alone would stop far from this positive parabola's minimum.
    problem = BatchedLeastSquaresProblem(residual, jacobian)
    initial = np.zeros((1, 1))
    limited = batched_levenberg_marquardt(
        problem, initial, config=LMConfig(max_evaluations=2)
    )
    assert limited.status.tolist() == [LMStatus.MAX_EVALUATIONS]
    assert limited.evaluations.tolist() == [2]
    np.testing.assert_allclose(limited.parameters, 1.0 / 1.001)

    result = batched_levenberg_marquardt(problem, initial)

    assert result.converged.all()
    assert result.evaluations[0] > 2
    np.testing.assert_allclose(
        result.parameters, 1.0 / (2.0 * coefficient), atol=5.0e-5, rtol=0
    )
    np.testing.assert_allclose(
        result.residuals, 1.0 - 1.0 / (4.0 * coefficient), atol=1.0e-8
    )


def test_f_tol_still_accepts_small_well_predicted_progress() -> None:
    def residual(x, *, indices):
        del indices
        return np.concatenate((x, np.ones_like(x)), axis=1)

    def jacobian(x, *, indices):
        del indices
        return np.broadcast_to(np.asarray([[[1.0, 0.0]]]), (len(x), 1, 2))

    result = batched_levenberg_marquardt(
        BatchedLeastSquaresProblem(residual, jacobian),
        np.asarray([[1.0e-5]]),
    )

    assert result.status.tolist() == [LMStatus.CONVERGED_F_TOL]
    assert result.evaluations.tolist() == [2]
    np.testing.assert_allclose(result.parameters, 1.0e-5 / 1001.0)


def test_ill_scaled_covariance_and_rank_are_independent_of_units() -> None:
    orthogonal = (
        np.asarray(
            [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]],
            dtype=np.float64,
        )
        / 2
    )
    singular_values = np.asarray([1, 0.5, 0.25, 2.0**-8])
    scales = np.asarray([2.0**-24, 1, 32, 2.0**-8])
    factor = scales[:, None] * orthogonal * singular_values[None, :]
    matrix = np.tile(factor, (1, 3))
    matrices = np.stack((matrix, matrix.copy()))
    # Changing one parameter's units leaves its independent direction intact.
    matrices[1, 0] *= 2.0**-36
    expected = (
        ((orthogonal / singular_values**2) @ orthogonal.T)
        / scales[:, None]
        / scales[None, :]
        / 3
    )

    def residual(x, *, indices):
        return np.einsum("kn,knm->km", x, matrices[indices])

    def jacobian(x, *, indices):
        del x
        return matrices[indices]

    result = batched_levenberg_marquardt(
        BatchedLeastSquaresProblem(residual, jacobian), np.zeros((2, 4))
    )

    assert result.rank.tolist() == [4, 4]
    np.testing.assert_allclose(
        result.covariance[0], expected, rtol=2.0e-8, atol=2.0e-8
    )
    converted = result.covariance[1].copy()
    converted[0, :] *= 2.0**-36
    converted[:, 0] *= 2.0**-36
    np.testing.assert_allclose(converted, expected, rtol=2.0e-8, atol=2.0e-8)


def test_covariance_factor_preserves_a_well_measured_parameter_combination():
    # Independent unit-noise observations measure the mean directly and the
    # difference very weakly. Var((x+y)/2) is exactly one, although rounding
    # the ~2**60 covariance entries loses that information completely.
    scale = 2.0**30
    matrix = np.asarray([[0.5, 0.5 / scale], [0.5, -0.5 / scale]])

    def residual(x, *, indices):
        del indices
        return x @ matrix

    def jacobian(x, *, indices):
        del indices
        return np.broadcast_to(matrix, (len(x), 2, 2))

    result = batched_levenberg_marquardt(
        BatchedLeastSquaresProblem(residual, jacobian), np.zeros((1, 2))
    )

    assert result.converged.all()
    assert result.rank.tolist() == [2]
    assert result.covariance_factor is not None
    gradient = np.asarray([0.5, 0.5])
    projected = gradient @ result.covariance_factor[0]
    np.testing.assert_allclose(projected @ projected, 1.0, rtol=1.0e-12)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_gaussian_uncertainty_is_independent_of_flux_units(dtype) -> None:
    model = GaussianDipoleModel((25, 25), dtype=dtype)
    units = np.asarray([1.0e-8, 1.0, 1.0e8], dtype=dtype)
    truth = np.tile(
        np.asarray([1, 1.7, 1.1, 0.35, -1.8, 0.7, 2.2, -0.4], dtype=dtype),
        (3, 1),
    )
    truth[:, 0] *= units
    images = model.evaluate(truth)
    variance = np.broadcast_to(units[:, None, None] ** 2, images.shape)
    result = fit_dipoles(
        images,
        model=model,
        initial=truth,
        variance=variance,
        backend="numpy",
    )

    assert result.converged.all()
    assert result.uncertainty_valid.all()
    assert result.covariance_factor is not None
    np.testing.assert_allclose(
        result.covariance_factor @ result.covariance_factor.swapaxes(-2, -1),
        result.covariance,
        rtol=2.0e-5 if dtype is np.float32 else 2.0e-12,
        atol=0,
    )
    normalized_covariance = result.covariance.copy()
    normalized_covariance[:, 0, :] /= units[:, None]
    normalized_covariance[:, :, 0] /= units[:, None]
    tolerance = 2.0e-5 if dtype is np.float32 else 2.0e-12
    for row in (0, 2):
        np.testing.assert_allclose(
            normalized_covariance[row],
            normalized_covariance[1],
            rtol=tolerance,
            atol=tolerance,
        )


def test_gaussian_equilibration_preserves_true_rank_deficiency() -> None:
    model = GaussianDipoleModel((25, 25))
    truth = np.asarray(
        [
            # A circular Gaussian has no identifiable orientation.
            [1e20, 1.7, 1.7, 0, -1.8, 0.7, 2.2, -0.4],
            # Coincident lobes cancel regardless of amplitude and shape.
            [1e-20, 1.7, 1.1, 0.35, -1.8, 0.7, -1.8, 0.7],
        ]
    )
    images = model.evaluate(truth)
    variance = np.broadcast_to(truth[:, 0, None, None] ** 2, images.shape)
    result = fit_dipoles(
        images,
        model=model,
        initial=truth,
        variance=variance,
        backend="numpy",
    )

    assert result.converged.all()
    assert not result.uncertainty_valid.any()
    assert result.uncertainty_reason == ("rank-deficient Jacobian",) * 2
    assert np.isnan(result.covariance).all()
    assert result.covariance_factor is not None
    assert np.isnan(result.covariance_factor).all()


@pytest.mark.parametrize(
    ("dtype", "exponent", "atol"),
    [(np.float32, 15, 2.0e-6), (np.float64, 20, 1.0e-9)],
)
@pytest.mark.parametrize("specialized", [False, True])
def test_solver_convergence_is_independent_of_parameter_units(
    dtype, exponent, atol, specialized
) -> None:
    units = np.asarray([10.0**-exponent, 1, 10.0**exponent, 1], dtype=dtype)

    def residual(x, *, indices):
        value = x * units[indices, None] - 1
        value[indices == 3] = np.nan
        return value

    def jacobian(x, *, indices):
        del x
        return units[indices, None, None]

    def normal_equations(x, residuals, *, indices):
        jac = jacobian(x, indices=indices)
        return (
            np.einsum("knm,km->kn", jac, residuals),
            np.einsum("knm,kpm->knp", jac, jac),
        )

    result = batched_levenberg_marquardt(
        BatchedLeastSquaresProblem(
            residual,
            jacobian,
            normal_equations=normal_equations if specialized else None,
        ),
        np.zeros((4, 1), dtype=dtype),
    )

    # Changing units must not turn the initial point into G_TOL, or the
    # first damped step (residual about 1e-3) into X_TOL.
    assert result.converged[:3].all()
    np.testing.assert_allclose(result.residuals[:3], 0, atol=atol, rtol=0)
    np.testing.assert_allclose(
        result.parameters[:3, 0] * units[:3], 1, atol=atol, rtol=0
    )
    assert result.status[3] == LMStatus.INVALID_RESIDUAL
    assert result.evaluations[3] == 1


def test_solver_equilibrates_coupled_parameter_units() -> None:
    # The three physical columns are well conditioned, but their units
    # produce a normal system spanning sixty orders of magnitude.
    physical_jacobian = np.asarray(
        [[1, 2, -1, 0], [2, -1, 0, 1], [-1, 1, 2, 1]], dtype=np.float64
    )
    units = np.asarray([1.0e-15, 1.0, 1.0e15])
    target = np.asarray([[1.0, -2.0, 0.5]])

    def residual(x, *, indices):
        return (x * units - target[indices]) @ physical_jacobian

    def jacobian(x, *, indices):
        del indices
        return np.broadcast_to(
            (units[:, None] * physical_jacobian)[None], (len(x), 3, 4)
        )

    result = batched_levenberg_marquardt(
        BatchedLeastSquaresProblem(residual, jacobian), np.zeros((1, 3))
    )

    assert result.converged.all()
    np.testing.assert_allclose(result.parameters * units, target, atol=1e-8)
    np.testing.assert_allclose(result.residuals, 0, atol=1e-8)
    assert result.rank.tolist() == [3]
    np.testing.assert_allclose(
        result.covariance[0] * units[:, None] * units[None, :],
        np.linalg.inv(physical_jacobian @ physical_jacobian.T),
        rtol=1e-12,
        atol=1e-12,
    )


def test_zero_jacobian_column_does_not_set_the_step_tolerance() -> None:
    def residual(x, *, indices):
        del indices
        return x[:, :1] - 1

    def jacobian(x, *, indices):
        del indices
        return np.broadcast_to(np.asarray([[[1.0], [0.0]]]), (len(x), 2, 1))

    result = batched_levenberg_marquardt(
        BatchedLeastSquaresProblem(residual, jacobian),
        np.asarray([[0.0, 1.0e200]]),
    )

    assert result.converged.all()
    np.testing.assert_allclose(result.residuals, 0, atol=1e-9)
    assert result.parameters[0, 1] == 1.0e200
    assert result.rank.tolist() == [1]
    assert np.isnan(result.covariance).all()


def test_large_parameter_does_not_hide_motion_in_other_coordinates() -> None:
    target = np.asarray([[1.0e10, 1.0]])

    def residual(x, *, indices):
        return x - target[indices]

    def jacobian(x, *, indices):
        del indices
        return np.broadcast_to(np.eye(2)[None], (len(x), 2, 2))

    result = batched_levenberg_marquardt(
        BatchedLeastSquaresProblem(residual, jacobian),
        np.asarray([[1.0e10, 0.0]]),
    )

    assert result.converged.all()
    np.testing.assert_allclose(result.residuals, 0, atol=1e-9)


@pytest.mark.parametrize("count", [1, 2])
def test_cupy_gradient_convergence_retains_small_residual_contributions(
    count,
):
    ap = pytest.importorskip("cupy")
    try:
        if ap.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device is unavailable")
    except Exception:
        pytest.skip("CUDA runtime is unavailable")
    # The 164 small components collectively affect the stopping boundary.
    # Exact summation supplies an independent reference for this linear fit.
    row = np.full(165, 2.0**-27)
    row[0] = 1.0
    residual_norm = sqrt(fsum(float(value * value) for value in row))
    template = ap.asarray(row)

    def residual(x, *, indices):
        del indices
        value = ap.broadcast_to(template, (x.shape[0], row.size)).copy()
        value[:, 0] += x[:, 0]
        return value

    def jacobian(x, *, indices):
        del indices
        value = ap.zeros((x.shape[0], 1, row.size))
        value[:, 0, 0] = 1
        return value

    result = batched_levenberg_marquardt(
        BatchedLeastSquaresProblem(residual, jacobian),
        ap.zeros((count, 1)),
        config=LMConfig(g_tol=1.0 / residual_norm, max_evaluations=2),
    )
    assert (result.status == int(LMStatus.CONVERGED_G_TOL)).all()
    assert (result.evaluations == 1).all()
    assert (result.parameters == 0).all()
