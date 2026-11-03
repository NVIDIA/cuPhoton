# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The v0.1.3 Levenberg--Marquardt iteration policy.

Preserve the numerical operation order from release b713503c7b81. Shared
contracts and final-diagnostic plumbing live in solver.py; the finalizer
selects the release's unscaled rank and dense covariance arithmetic.
"""

from __future__ import annotations

import numpy as np

from ._types import BackendArray
from .solver import (
    _MAX_REJECTED_STEPS,
    BatchedLeastSquaresProblem,
    LMConfig,
    LMResult,
    LMStatus,
    _bool,
    _call_jacobian,
    _call_normal_equations,
    _call_residual,
    _finalize_lm,
    _finite_difference_jacobian,
    _prepare_solver,
)


def solve(
    problem: BatchedLeastSquaresProblem,
    x0: BackendArray,
    *,
    config: LMConfig | None = None,
) -> LMResult:
    """Solve independent nonlinear least-squares problems in one batch.

    Arrays stay in the array package that owns ``x0``. A problem Jacobian is
    interpreted as parameter-major, with shape ``(K, N, M)``. When it is
    absent, or finite differences are requested, the solver allocates the
    perturbation workspace in the same backend as ``x0``.
    """

    ap, dtype, settings, x, indices, residuals, (k, n, m) = _prepare_solver(
        problem, x0, config
    )

    status = ap.full(k, int(LMStatus.ACTIVE), dtype=np.int8)
    evaluations = ap.ones(k, dtype=np.int32)
    eps = float(np.finfo(dtype).eps)
    damping = ap.full(k, settings.initial_damping, dtype=dtype)
    rejected_steps = ap.zeros(k, dtype=np.int16)

    def finite_rows(value: BackendArray) -> BackendArray:
        return ap.isfinite(value).reshape(value.shape[0], -1).all(axis=1)

    initially_valid = finite_rows(x) & finite_rows(residuals)
    status[~initially_valid] = int(LMStatus.INVALID_RESIDUAL)
    identity = ap.eye(n, dtype=dtype)[None, :, :]
    jacobians = ap.full((k, n, m), ap.nan, dtype=dtype)
    jacobian_current = ap.zeros(k, dtype=bool)
    # Rows whose latest normal equations describe the current parameters.
    # They mirror ``jacobian_current`` so the final diagnostics of the
    # specialized path match the plain analytic path row by row.
    diagnostics_current = ap.zeros(k, dtype=bool)
    use_normal_equations = (
        problem.normal_equations is not None
        and problem.jacobian is not None
        and not settings.use_finite_difference
    )

    while _bool((status == int(LMStatus.ACTIVE)).any()):
        active_indices = indices[status == int(LMStatus.ACTIVE)]
        enough_for_jacobian = evaluations[active_indices] < (
            settings.max_evaluations
        )
        if problem.jacobian is None or settings.use_finite_difference:
            enough_for_jacobian &= (
                evaluations[active_indices] + n <= settings.max_evaluations
            )
        exhausted = active_indices[~enough_for_jacobian]
        status[exhausted] = int(LMStatus.MAX_EVALUATIONS)
        active_indices = active_indices[enough_for_jacobian]
        if active_indices.shape[0] == 0:
            continue

        x_active = x[active_indices]
        residual_active = residuals[active_indices]
        if use_normal_equations:
            gradient, hessian = _call_normal_equations(
                problem,
                x_active,
                residual_active,
                active_indices,
            )
            expected_gradient_shape = (active_indices.shape[0], n)
            expected_hessian_shape = (active_indices.shape[0], n, n)
            if gradient.shape != expected_gradient_shape:
                raise ValueError(
                    "normal-equation gradient must have shape "
                    "(batch, parameters)"
                )
            if hessian.shape != expected_hessian_shape:
                raise ValueError(
                    "normal-equation Hessian must have shape "
                    "(batch, parameters, parameters)"
                )
            valid_normal_equations = finite_rows(gradient) & finite_rows(
                hessian
            )
            fallback_positions = ap.flatnonzero(~valid_normal_equations)
            if fallback_positions.shape[0] > 0:
                # Finite analytic derivatives can overflow their products.
                # Recover the ordinary path's finite-Jacobian predicate and
                # let its gradient/solve decisions handle those products.
                fallback_indices = active_indices[fallback_positions]
                fallback_jacobian = _call_jacobian(
                    problem,
                    x_active[fallback_positions],
                    fallback_indices,
                )
                if fallback_jacobian.shape != (
                    fallback_positions.shape[0],
                    n,
                    m,
                ):
                    raise ValueError(
                        "jacobian must return shape "
                        "(batch, parameters, observations)"
                    )
                valid_fallback = finite_rows(fallback_jacobian)
                status[fallback_indices[~valid_fallback]] = int(
                    LMStatus.INVALID_RESIDUAL
                )
                recovered_positions = fallback_positions[valid_fallback]
                if recovered_positions.shape[0] > 0:
                    recovered_indices = fallback_indices[valid_fallback]
                    fallback_jacobian = fallback_jacobian[valid_fallback]
                    gradient = gradient.copy()
                    hessian = hessian.copy()
                    gradient[recovered_positions] = ap.einsum(
                        "knm,km->kn",
                        fallback_jacobian,
                        residual_active[recovered_positions],
                    )
                    hessian[recovered_positions] = ap.einsum(
                        "knm,kpm->knp", fallback_jacobian, fallback_jacobian
                    )
                    valid_normal_equations[recovered_positions] = True
                    jacobians[recovered_indices] = fallback_jacobian
                    jacobian_current[recovered_indices] = True
            active_indices = active_indices[valid_normal_equations]
            if active_indices.shape[0] == 0:
                continue
            x_active = x_active[valid_normal_equations]
            residual_active = residual_active[valid_normal_equations]
            gradient = gradient[valid_normal_equations]
            hessian = hessian[valid_normal_equations]
            diagnostics_current[active_indices] = True
        else:
            if (
                problem.jacobian is not None
                and not settings.use_finite_difference
            ):
                jacobian = _call_jacobian(problem, x_active, active_indices)
            else:
                jacobian = _finite_difference_jacobian(
                    problem,
                    x_active,
                    residual_active,
                    active_indices,
                    evaluations,
                    ap,
                    settings.finite_difference_step,
                )
            expected_shape = (active_indices.shape[0], n, m)
            if jacobian.shape != expected_shape:
                raise ValueError(
                    "jacobian must return shape "
                    "(batch, parameters, observations)"
                )

            valid_jacobian = finite_rows(jacobian)
            status[active_indices[~valid_jacobian]] = int(
                LMStatus.INVALID_RESIDUAL
            )
            active_indices = active_indices[valid_jacobian]
            if active_indices.shape[0] == 0:
                continue
            x_active = x_active[valid_jacobian]
            residual_active = residual_active[valid_jacobian]
            jacobian = jacobian[valid_jacobian]
            jacobians[active_indices] = jacobian
            jacobian_current[active_indices] = True
            gradient = ap.einsum("knm,km->kn", jacobian, residual_active)
            hessian = ap.einsum("knm,kpm->knp", jacobian, jacobian)
        diagonal = ap.maximum(ap.diagonal(hessian, axis1=1, axis2=2), eps)
        scaled_gradient = ap.max(ap.abs(gradient) / ap.sqrt(diagonal), axis=1)
        residual_norm = ap.linalg.norm(residual_active, axis=1)
        gradient_converged = scaled_gradient <= (
            settings.g_tol * ap.maximum(1.0, residual_norm)
        )
        status[active_indices[gradient_converged]] = int(
            LMStatus.CONVERGED_G_TOL
        )
        active_indices = active_indices[~gradient_converged]
        if active_indices.shape[0] == 0:
            continue
        x_active = x_active[~gradient_converged]
        residual_active = residual_active[~gradient_converged]
        gradient = gradient[~gradient_converged]
        hessian = hessian[~gradient_converged]
        diagonal = diagonal[~gradient_converged]

        system = hessian + (
            damping[active_indices, None, None]
            * identity
            * diagonal[:, None, :]
        )
        try:
            step = ap.linalg.solve(system, -gradient[..., None])[..., 0]
        except Exception:
            status[active_indices] = int(LMStatus.SINGULAR)
            continue
        valid_step = finite_rows(step)
        status[active_indices[~valid_step]] = int(LMStatus.SINGULAR)
        active_indices = active_indices[valid_step]
        if active_indices.shape[0] == 0:
            continue
        x_active = x_active[valid_step]
        residual_active = residual_active[valid_step]
        step = step[valid_step]

        can_evaluate = evaluations[active_indices] < settings.max_evaluations
        status[active_indices[~can_evaluate]] = int(LMStatus.MAX_EVALUATIONS)
        active_indices = active_indices[can_evaluate]
        if active_indices.shape[0] == 0:
            continue
        x_active = x_active[can_evaluate]
        residual_active = residual_active[can_evaluate]
        step = step[can_evaluate]

        trial_x = x_active + step
        trial_residual = _call_residual(problem, trial_x, active_indices)
        evaluations[active_indices] += 1
        trial_valid = finite_rows(trial_residual)
        cost = 0.5 * ap.einsum("km,km->k", residual_active, residual_active)
        trial_cost = 0.5 * ap.einsum(
            "km,km->k", trial_residual, trial_residual
        )
        reduction = cost - trial_cost
        accepted = trial_valid & (reduction > 0)

        accepted_indices = active_indices[accepted]
        if accepted_indices.shape[0] > 0:
            accepted_step = step[accepted]
            x[accepted_indices] = trial_x[accepted]
            residuals[accepted_indices] = trial_residual[accepted]
            jacobians[accepted_indices] = ap.nan
            jacobian_current[accepted_indices] = False
            diagnostics_current[accepted_indices] = False
            damping[accepted_indices] = ap.maximum(
                eps,
                damping[accepted_indices] * settings.damping_decrease,
            )
            rejected_steps[accepted_indices] = 0
            f_converged = reduction[accepted] <= (
                settings.f_tol * ap.maximum(1.0, cost[accepted])
            )
            x_converged = ap.linalg.norm(accepted_step, axis=1) <= (
                settings.x_tol * (settings.x_tol + 1.0)
            )
            status[accepted_indices[f_converged]] = int(
                LMStatus.CONVERGED_F_TOL
            )
            status[accepted_indices[~f_converged & x_converged]] = int(
                LMStatus.CONVERGED_X_TOL
            )

        rejected = ~accepted
        rejected_indices = active_indices[rejected]
        if rejected_indices.shape[0] > 0:
            damping[rejected_indices] *= settings.damping_increase
            rejected_steps[rejected_indices] += 1
            hopeless = (damping[rejected_indices] > 1.0 / eps) | (
                rejected_steps[rejected_indices] >= _MAX_REJECTED_STEPS
            )
            rejected_trial_valid = trial_valid[rejected]
            status[rejected_indices[hopeless & ~rejected_trial_valid]] = int(
                LMStatus.INVALID_RESIDUAL
            )
            status[rejected_indices[hopeless & rejected_trial_valid]] = int(
                LMStatus.NO_PROGRESS
            )

    return _finalize_lm(
        problem,
        x,
        residuals,
        status,
        evaluations,
        jacobians,
        jacobian_current,
        diagnostics_current,
        settings,
        ap,
    )
