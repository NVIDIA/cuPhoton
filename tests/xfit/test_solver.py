# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import least_squares

from cuphoton.xfit import (
    BatchedLeastSquaresProblem,
    LMConfig,
    LMStatus,
    batched_levenberg_marquardt,
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


def test_final_diagnostics_fall_back_to_rows_when_batched_svd_fails(
    monkeypatch,
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
        return original_svd(value, *args, **kwargs)

    monkeypatch.setattr(np.linalg, "svd", row_only_svd)
    result = batched_levenberg_marquardt(problem, target.copy())

    assert seen_shapes == [(3, 2, 3), (2, 3), (2, 3), (2, 3)]
    assert result.rank.tolist() == [2, 2, 2]
    assert np.array_equal(result.status, expected.status)
    for field in ("gradient", "gn_hessian", "covariance"):
        assert np.allclose(
            getattr(result, field),
            getattr(expected, field),
            rtol=1.0e-12,
            atol=1.0e-12,
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
