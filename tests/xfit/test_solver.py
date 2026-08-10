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
