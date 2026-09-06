# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Backend-native batched Levenberg--Marquardt least-squares solver."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from types import ModuleType
from typing import Any, Protocol

import numpy as np

from ._types import ArrayLike, BackendArray
from .backend import as_numpy


class LMStatus(IntEnum):
    """Stable termination codes returned for every batch member."""

    ACTIVE = 0
    CONVERGED_F_TOL = 1
    CONVERGED_X_TOL = 2
    CONVERGED_G_TOL = 3
    MAX_EVALUATIONS = 4
    INVALID_RESIDUAL = 5
    SINGULAR = 6
    NO_PROGRESS = 7


_CONVERGED_STATUSES = {
    LMStatus.CONVERGED_F_TOL,
    LMStatus.CONVERGED_X_TOL,
    LMStatus.CONVERGED_G_TOL,
}

_DEFAULT_MAX_EVALUATIONS_FACTOR = 200
_MAX_REJECTED_STEPS = 32


class ResidualFunction(Protocol):
    """Callable contract for one backend-native batch of residuals."""

    def __call__(
        self, parameters: BackendArray, *, indices: BackendArray
    ) -> BackendArray: ...


class JacobianFunction(Protocol):
    """Callable contract for one backend-native batch of Jacobians."""

    def __call__(
        self, parameters: BackendArray, *, indices: BackendArray
    ) -> BackendArray: ...


class NormalEquationsFunction(Protocol):
    """Backend-specialized gradient and positive-semidefinite Hessian.

    The caller validates shapes; symmetry and positive semidefiniteness are
    the callback author's responsibility.
    """

    def __call__(
        self,
        parameters: BackendArray,
        residuals: BackendArray,
        *,
        indices: BackendArray,
    ) -> tuple[BackendArray, BackendArray]: ...


@dataclass(frozen=True, init=False)
class BatchedLeastSquaresProblem:
    """Residual and Jacobian functions for independent batched problems.

    The residual function returns ``(batch, observations)``. The optional
    Jacobian returns ``(batch, parameters, observations)``. An optional
    normal-equations callback may return the gradient and Gauss--Newton
    Hessian directly as ``(batch, parameters)`` and
    ``(batch, parameters, parameters)``. It requires an analytic Jacobian for
    final diagnostics. When :attr:`LMConfig.use_finite_difference` is set the
    solver ignores ``normal_equations`` and differences the residual instead.
    Callbacks receive the current batch rows plus ``indices`` identifying
    their rows in the original problem.
    """

    residual: ResidualFunction
    jacobian: JacobianFunction | None
    normal_equations: NormalEquationsFunction | None

    def __init__(
        self,
        residual: ResidualFunction | None = None,
        jacobian: JacobianFunction | None = None,
        *,
        F: ResidualFunction | None = None,
        J: JacobianFunction | None = None,
        normal_equations: NormalEquationsFunction | None = None,
    ) -> None:
        residual_fn = residual if residual is not None else F
        jacobian_fn = jacobian if jacobian is not None else J
        if residual_fn is None:
            raise TypeError("a residual function is required")
        if residual is not None and F is not None:
            raise TypeError("pass residual or F, not both")
        if jacobian is not None and J is not None:
            raise TypeError("pass jacobian or J, not both")
        if normal_equations is not None and jacobian_fn is None:
            raise TypeError("normal_equations requires an analytic Jacobian")
        object.__setattr__(self, "residual", residual_fn)
        object.__setattr__(self, "jacobian", jacobian_fn)
        object.__setattr__(self, "normal_equations", normal_equations)

    @property
    def F(self) -> ResidualFunction:
        """Donor-compatible alias for :attr:`residual`."""

        return self.residual

    @property
    def J(self) -> JacobianFunction | None:
        """Donor-compatible alias for :attr:`jacobian`."""

        return self.jacobian


@dataclass(frozen=True)
class LMConfig:
    """Numerical controls for :func:`batched_levenberg_marquardt`."""

    f_tol: float | None = None
    x_tol: float | None = None
    g_tol: float | None = None
    max_evaluations: int | None = None
    initial_damping: float = 1.0e-3
    damping_increase: float = 10.0
    damping_decrease: float = 0.3
    finite_difference_step: float | None = None
    use_finite_difference: bool = False

    def resolved_max_evaluations(self, parameter_count: int) -> int:
        """Return the configured or default residual-evaluation budget."""

        if parameter_count < 1:
            raise ValueError("parameter_count must be positive")
        value = (
            _DEFAULT_MAX_EVALUATIONS_FACTOR * (parameter_count + 1)
            if self.max_evaluations is None
            else self.max_evaluations
        )
        if value < 1:
            raise ValueError("max_evaluations must be positive")
        return int(value)


@dataclass(frozen=True)
class LMResult:
    """Backend-native result arrays from a batched least-squares solve.

    ``evaluations`` counts residual evaluations per row, including finite
    differences. ``rank`` is ``-1`` when no final Jacobian was available
    within the configured evaluation budget.
    """

    parameters: BackendArray
    status: BackendArray
    converged: BackendArray
    evaluations: BackendArray
    residuals: BackendArray
    residual_norm: BackendArray
    jacobian: BackendArray
    gradient: BackendArray
    gn_hessian: BackendArray
    covariance: BackendArray
    rank: BackendArray

    @property
    def solution(self) -> BackendArray:
        """Alias for :attr:`parameters`."""

        return self.parameters

    @property
    def residual(self) -> BackendArray:
        """Alias for :attr:`residuals`."""

        return self.residuals

    @property
    def info(self) -> BackendArray:
        """Alias for :attr:`status`."""

        return self.status

    @property
    def pcov(self) -> BackendArray:
        """Alias for :attr:`covariance`."""

        return self.covariance


@dataclass(frozen=True)
class _Settings:
    f_tol: float
    x_tol: float
    g_tol: float
    max_evaluations: int
    initial_damping: float
    damping_increase: float
    damping_decrease: float
    finite_difference_step: float
    use_finite_difference: bool


def _settings(config: LMConfig, dtype: np.dtype[Any], n: int) -> _Settings:
    eps = float(np.finfo(dtype).eps)
    root_eps = eps**0.5
    values = {
        "f_tol": root_eps if config.f_tol is None else config.f_tol,
        "x_tol": root_eps if config.x_tol is None else config.x_tol,
        "g_tol": root_eps if config.g_tol is None else config.g_tol,
    }
    for name, value in values.items():
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    max_evaluations = config.resolved_max_evaluations(n)
    if not np.isfinite(config.initial_damping) or config.initial_damping <= 0:
        raise ValueError("initial_damping must be finite and positive")
    if (
        not np.isfinite(config.damping_increase)
        or config.damping_increase <= 1
    ):
        raise ValueError(
            "damping_increase must be finite and greater than one"
        )
    if (
        not np.isfinite(config.damping_decrease)
        or config.damping_decrease <= 0
        or config.damping_decrease >= 1
    ):
        raise ValueError("damping_decrease must be between zero and one")
    fd_step = (
        root_eps
        if config.finite_difference_step is None
        else config.finite_difference_step
    )
    if not np.isfinite(fd_step) or fd_step <= 0:
        raise ValueError("finite_difference_step must be finite and positive")
    return _Settings(
        f_tol=float(values["f_tol"]),
        x_tol=float(values["x_tol"]),
        g_tol=float(values["g_tol"]),
        max_evaluations=int(max_evaluations),
        initial_damping=float(config.initial_damping),
        damping_increase=float(config.damping_increase),
        damping_decrease=float(config.damping_decrease),
        finite_difference_step=float(fd_step),
        use_finite_difference=config.use_finite_difference,
    )


def _bool(value: ArrayLike) -> bool:
    return bool(as_numpy(value).item())


def _int(value: ArrayLike) -> int:
    return int(as_numpy(value).item())


def _finite(value: BackendArray, ap: ModuleType) -> bool:
    return _bool(ap.isfinite(value).all())


def _call_residual(
    problem: BatchedLeastSquaresProblem,
    x: BackendArray,
    indices: BackendArray,
) -> BackendArray:
    return problem.residual(x, indices=indices)


def _call_jacobian(
    problem: BatchedLeastSquaresProblem,
    x: BackendArray,
    indices: BackendArray,
) -> BackendArray:
    if problem.jacobian is None:
        raise TypeError("the problem does not provide a Jacobian")
    return problem.jacobian(x, indices=indices)


def _call_normal_equations(
    problem: BatchedLeastSquaresProblem,
    x: BackendArray,
    residual: BackendArray,
    indices: BackendArray,
) -> tuple[BackendArray, BackendArray]:
    if problem.normal_equations is None:
        raise TypeError("the problem does not provide normal equations")
    return problem.normal_equations(x, residual, indices=indices)


def _finite_difference_jacobian(
    problem: BatchedLeastSquaresProblem,
    x: BackendArray,
    residual: BackendArray,
    indices: BackendArray,
    evaluations: BackendArray,
    ap: ModuleType,
    step_scale: float,
) -> BackendArray:
    batch, n = x.shape
    m = residual.shape[1]
    jacobian = ap.empty((batch, n, m), dtype=x.dtype)
    trial = x.copy()
    completed = 0
    try:
        for parameter in range(n):
            step = step_scale * ap.maximum(1.0, ap.abs(x[:, parameter]))
            trial[:] = x
            trial[:, parameter] += step
            perturbed = _call_residual(problem, trial, indices)
            completed += 1
            jacobian[:, parameter, :] = (perturbed - residual) / step[:, None]
    finally:
        if completed:
            evaluations[indices] += completed
    return jacobian


def _final_jacobian(
    problem: BatchedLeastSquaresProblem,
    x: BackendArray,
    residual: BackendArray,
    indices: BackendArray,
    evaluations: BackendArray,
    ap: ModuleType,
    settings: _Settings,
) -> BackendArray:
    if problem.jacobian is not None and not settings.use_finite_difference:
        return _call_jacobian(problem, x, indices)
    result = _finite_difference_jacobian(
        problem,
        x,
        residual,
        indices,
        evaluations,
        ap,
        settings.finite_difference_step,
    )
    return result


def _factorize_jacobians(
    jacobians: BackendArray, ap: ModuleType
) -> tuple[BackendArray, BackendArray]:
    """Return left singular vectors and values for covariance diagnostics."""

    batch, parameters, observations = jacobians.shape
    module_name = type(jacobians).__module__.split(".", maxsplit=1)[0]
    if module_name == "cupy" and batch > 1 and observations > 2 * parameters:
        # For a tall J.T = Q R, the small R.T has the same singular values and
        # left singular vectors as J. Avoid cuSOLVER's expensive batched SVD
        # setup for the wide parameter-major Jacobians used by image fits.
        reduced = ap.linalg.qr(
            jacobians.transpose(0, 2, 1), mode="r"
        ).transpose(0, 2, 1)
        left_vectors, singular_values, _ = ap.linalg.svd(
            reduced, full_matrices=False
        )
        return left_vectors, singular_values
    left_vectors, singular_values, _ = ap.linalg.svd(
        jacobians, full_matrices=False
    )
    return left_vectors, singular_values


def batched_levenberg_marquardt(
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

    if not hasattr(x0, "ndim") or x0.ndim != 2:
        raise ValueError("x0 must have shape (batch, parameters)")
    dtype = np.dtype(x0.dtype)
    if dtype.kind != "f":
        raise TypeError("x0 must have a floating-point dtype")
    module_name = type(x0).__module__.split(".", maxsplit=1)[0]
    if module_name == "cupy":
        import cupy as ap
    else:
        ap = np
        x0 = np.asarray(x0)

    k, n = x0.shape
    if k < 1 or n < 1:
        raise ValueError("x0 batch and parameter dimensions must be nonzero")
    settings = _settings(LMConfig() if config is None else config, dtype, n)
    x = x0.copy()
    indices = ap.arange(k, dtype=np.int64)
    residuals = _call_residual(problem, x, indices)
    if residuals.ndim != 2 or residuals.shape[0] != k:
        raise ValueError("residual must return shape (batch, observations)")
    if np.dtype(residuals.dtype).kind != "f":
        raise TypeError("residual arrays must have a floating-point dtype")
    m = residuals.shape[1]
    if m < 1:
        raise ValueError("the residual must contain at least one observation")

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
            status[active_indices[~valid_normal_equations]] = int(
                LMStatus.INVALID_RESIDUAL
            )
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

    gradients = ap.full((k, n), ap.nan, dtype=dtype)
    hessians = ap.full((k, n, n), ap.nan, dtype=dtype)
    covariance = ap.full((k, n, n), ap.nan, dtype=dtype)
    rank = ap.full(k, -1, dtype=np.int32)
    status_host = as_numpy(status)
    jacobian_current_host = as_numpy(jacobian_current)
    diagnostics_current_host = as_numpy(diagnostics_current)
    evaluations_host = as_numpy(evaluations)
    finite_difference = (
        problem.jacobian is None or settings.use_finite_difference
    )
    diagnostic_rows: list[int] = []
    refresh_rows: list[int] = []
    for row in range(k):
        if int(status_host[row]) == int(LMStatus.INVALID_RESIDUAL):
            continue
        if bool(jacobian_current_host[row]):
            diagnostic_rows.append(row)
            continue
        if bool(diagnostics_current_host[row]):
            # The specialized path never stored this row's Jacobian, but the
            # analytic callback reproduces it without residual evaluations.
            # As with jacobian_current, current diagnostics remain available
            # for non-converged rows; uncertainty validity is checked later.
            refresh_rows.append(row)
            continue
        if int(status_host[row]) not in {
            int(value) for value in _CONVERGED_STATUSES
        }:
            continue
        if (
            finite_difference
            and int(evaluations_host[row]) + n > settings.max_evaluations
        ):
            continue
        refresh_rows.append(row)

    if refresh_rows:
        refresh_indices = ap.asarray(refresh_rows, dtype=np.int64)
        try:
            refreshed = _final_jacobian(
                problem,
                x[refresh_indices],
                residuals[refresh_indices],
                refresh_indices,
                evaluations,
                ap,
                settings,
            )
            if refreshed.shape != (len(refresh_rows), n, m):
                raise ValueError("final Jacobian has an unexpected shape")
            valid_refresh = as_numpy(finite_rows(refreshed))
            valid_rows = [
                row
                for row, valid in zip(
                    refresh_rows, valid_refresh, strict=True
                )
                if bool(valid)
            ]
            if valid_rows:
                valid_indices = ap.asarray(valid_rows, dtype=np.int64)
                jacobians[valid_indices] = refreshed[
                    ap.asarray(valid_refresh, dtype=bool)
                ]
                diagnostic_rows.extend(valid_rows)
        except Exception:
            # Retry row by row so one failing callback does not discard the
            # diagnostics of every row in the batch. The finite-difference
            # helper already charged the perturbations it completed, so a
            # retry must still fit the remaining residual budget.
            evaluations_host = as_numpy(evaluations)
            for row in refresh_rows:
                if (
                    finite_difference
                    and int(evaluations_host[row]) + n
                    > settings.max_evaluations
                ):
                    continue
                try:
                    jacobian = _final_jacobian(
                        problem,
                        x[row : row + 1],
                        residuals[row : row + 1],
                        indices[row : row + 1],
                        evaluations,
                        ap,
                        settings,
                    )
                    if jacobian.shape != (1, n, m) or not _finite(
                        jacobian, ap
                    ):
                        continue
                    jacobians[row] = jacobian[0]
                    diagnostic_rows.append(row)
                except Exception:
                    continue

    if diagnostic_rows:
        diagnostic_indices = ap.asarray(diagnostic_rows, dtype=np.int64)
        diagnostic_jacobians = jacobians[diagnostic_indices]
        diagnostic_residuals = residuals[diagnostic_indices]
        try:
            gradients[diagnostic_indices] = ap.einsum(
                "knm,km->kn", diagnostic_jacobians, diagnostic_residuals
            )
            hessians[diagnostic_indices] = ap.einsum(
                "knm,kpm->knp", diagnostic_jacobians, diagnostic_jacobians
            )
            left_vectors, singular_values = _factorize_jacobians(
                diagnostic_jacobians, ap
            )
            tolerance = max(n, m) * eps * ap.max(singular_values, axis=1)
            rank_values = ap.sum(singular_values > tolerance[:, None], axis=1)
            rank[diagnostic_indices] = rank_values
            full_rank = rank_values == n
            full_rank_indices = diagnostic_indices[full_rank]
            if full_rank_indices.shape[0] > 0:
                full_rank_vectors = left_vectors[full_rank]
                full_rank_singular_values = singular_values[full_rank]
                inverse_squared = 1.0 / (
                    full_rank_singular_values * full_rank_singular_values
                )
                covariance[full_rank_indices] = (
                    full_rank_vectors * inverse_squared[:, None, :]
                ) @ full_rank_vectors.transpose(0, 2, 1)
        except Exception:
            # Preserve row-level failure isolation if a backend cannot perform
            # one of the batched linear-algebra operations.
            for row in diagnostic_rows:
                try:
                    jacobian = jacobians[row]
                    gradients[row] = jacobian @ residuals[row]
                    hessians[row] = jacobian @ jacobian.T
                    left_vectors, singular_values, _ = ap.linalg.svd(
                        jacobian, full_matrices=False
                    )
                    tolerance = max(n, m) * eps * ap.max(singular_values)
                    rank[row] = ap.sum(singular_values > tolerance)
                    if _int(rank[row]) == n:
                        inverse_squared = 1.0 / (
                            singular_values * singular_values
                        )
                        covariance[row] = (
                            left_vectors * inverse_squared[None, :]
                        ) @ left_vectors.T
                except Exception:
                    continue

    residual_norm = ap.linalg.norm(residuals, axis=1)
    converged = ap.zeros(k, dtype=bool)
    for converged_status in _CONVERGED_STATUSES:
        converged |= status == int(converged_status)
    return LMResult(
        parameters=x,
        status=status,
        converged=converged,
        evaluations=evaluations,
        residuals=residuals,
        residual_norm=residual_norm,
        jacobian=jacobians,
        gradient=gradients,
        gn_hessian=hessians,
        covariance=covariance,
        rank=rank,
    )


__all__ = [
    "BatchedLeastSquaresProblem",
    "JacobianFunction",
    "LMConfig",
    "LMResult",
    "LMStatus",
    "NormalEquationsFunction",
    "ResidualFunction",
    "batched_levenberg_marquardt",
]
