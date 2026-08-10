# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""High-level portable xFit API."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Literal, cast

import numpy as np

from ._types import (
    ArrayLike,
    BackendArray,
    BackendRequest,
    FitMode,
    FloatDType,
    ModelName,
    ResolvedBackend,
)
from .backend import as_numpy, resolve_backend
from .models import GaussianDipoleModel, StampDipoleModel
from .solver import (
    BatchedLeastSquaresProblem,
    LMConfig,
    LMStatus,
    batched_levenberg_marquardt,
)


@dataclass(frozen=True)
class DipoleFitResult:
    """Portable fit outputs with residual and uncertainty diagnostics."""

    parameters: np.ndarray
    parameter_names: tuple[str, ...]
    status: np.ndarray
    converged: np.ndarray
    evaluations: np.ndarray
    residual_norm: np.ndarray
    chi_square: np.ndarray
    valid_pixel_count: np.ndarray
    valid_pixel_fraction: np.ndarray
    null_chi_square: np.ndarray
    delta_chi_square: np.ndarray
    fractional_null_improvement: np.ndarray
    degrees_of_freedom: np.ndarray
    reduced_chi_square: np.ndarray
    covariance: np.ndarray
    standard_errors: np.ndarray
    uncertainty_valid: np.ndarray
    uncertainty_reason: tuple[str, ...]
    residuals: np.ndarray
    backend: ResolvedBackend
    device: str
    dtype: FloatDType
    model: ModelName
    mode: FitMode


def _floating_dtype(value: BackendArray) -> np.dtype[Any]:
    dtype = np.dtype(value.dtype)
    if dtype == np.dtype(np.float32):
        return dtype
    if dtype == np.dtype(np.float64):
        return dtype
    return np.dtype(np.float64)


def _validate_images(
    images: BackendArray, mode: FitMode
) -> tuple[int, int, int]:
    if mode == "difference":
        if images.ndim != 3:
            raise ValueError(
                "difference images must have shape (batch, height, width)"
            )
    elif mode == "split":
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                "split images must have shape (batch, 3, height, width)"
            )
    else:
        raise ValueError("mode must be 'difference' or 'split'")
    batch, height, width = images.shape[0], images.shape[-2], images.shape[-1]
    if min(batch, height, width) < 1:
        raise ValueError("image dimensions must be nonzero")
    return int(batch), int(height), int(width)


def _broadcast_auxiliary(
    value: ArrayLike,
    image_shape: tuple[int, ...],
    mode: FitMode,
    ap: ModuleType,
    *,
    name: str,
) -> BackendArray:
    array = ap.asarray(value)
    batch, height, width = image_shape[0], image_shape[-2], image_shape[-1]
    if array.shape == (height, width):
        array = array[None, ...]
        if mode == "split":
            array = array[:, None, ...]
    elif mode == "split" and array.shape == (batch, height, width):
        array = array[:, None, ...]
    elif mode == "split" and array.shape == (3, height, width):
        array = array[None, ...]
    try:
        return ap.broadcast_to(array, image_shape)
    except ValueError as exc:
        raise ValueError(
            f"{name} with shape {array.shape} cannot broadcast to images "
            f"with shape {image_shape}"
        ) from exc


def _initial_guess(
    images: BackendArray,
    model: GaussianDipoleModel | StampDipoleModel,
    mode: FitMode,
    ap: ModuleType,
    weights: BackendArray,
) -> BackendArray:
    batch, height, width = images.shape[0], images.shape[-2], images.shape[-1]
    difference = images if mode == "difference" else images[:, 0]
    if mode == "split":
        positive_plane = images[:, 1]
        negative_plane = images[:, 2]
        positive_active = weights[:, 1] != 0
        negative_active = weights[:, 2] != 0
        positive_score = ap.where(
            positive_active,
            positive_plane * weights[:, 1],
            -ap.inf,
        )
        negative_score = ap.where(
            negative_active,
            negative_plane * weights[:, 2],
            -ap.inf,
        )
        positive_index = ap.argmax(positive_score.reshape(batch, -1), axis=1)
        negative_index = ap.argmax(negative_score.reshape(batch, -1), axis=1)
    else:
        active = weights != 0
        score = difference * weights
        positive_index = ap.argmax(
            ap.where(active, score, -ap.inf).reshape(batch, -1), axis=1
        )
        negative_index = ap.argmin(
            ap.where(active, score, ap.inf).reshape(batch, -1), axis=1
        )
    x_positive = positive_index % width - (width - 1) / 2
    y_positive = positive_index // width - (height - 1) / 2
    x_negative = negative_index % width - (width - 1) / 2
    y_negative = negative_index // width - (height - 1) / 2

    if isinstance(model, GaussianDipoleModel):
        if mode == "split":
            positive_peak = ap.max(
                ap.where(positive_active, images[:, 1], -ap.inf),
                axis=(-2, -1),
            )
            negative_peak = ap.max(
                ap.where(negative_active, images[:, 2], -ap.inf),
                axis=(-2, -1),
            )
            amplitude = ap.maximum(
                positive_peak,
                negative_peak,
            )
        else:
            positive_peak = ap.max(
                ap.where(active, difference, -ap.inf), axis=(-2, -1)
            )
            negative_peak = -ap.min(
                ap.where(active, difference, ap.inf), axis=(-2, -1)
            )
            amplitude = ap.maximum(
                positive_peak,
                negative_peak,
            )
        amplitude = ap.where(ap.isfinite(amplitude), amplitude, 0)
        sigma = max(1.0, min(height, width) / 8.0)
        return ap.stack(
            (
                amplitude,
                ap.full(batch, sigma, dtype=model.dtype),
                ap.full(batch, sigma, dtype=model.dtype),
                ap.zeros(batch, dtype=model.dtype),
                x_positive,
                y_positive,
                x_negative,
                y_negative,
            ),
            axis=1,
        ).astype(model.dtype, copy=False)

    if mode == "split":
        positive_flux = ap.sum(
            ap.where(positive_active, ap.maximum(images[:, 1], 0), 0),
            axis=(-2, -1),
        )
        negative_flux = ap.sum(
            ap.where(negative_active, ap.maximum(images[:, 2], 0), 0),
            axis=(-2, -1),
        )
        flux = ap.maximum(positive_flux, negative_flux)
    else:
        flux = ap.sum(
            ap.where(active, ap.maximum(difference, 0), 0),
            axis=(-2, -1),
        )
    return ap.stack(
        (x_positive, y_positive, x_negative, y_negative, flux), axis=1
    ).astype(model.dtype, copy=False)


def _status_names(status: np.ndarray) -> np.ndarray:
    return np.asarray(
        [LMStatus(int(value)).name.lower() for value in status], dtype="U32"
    )


def fit_dipoles(
    images: ArrayLike,
    *,
    model: Literal["gaussian"] | GaussianDipoleModel | StampDipoleModel,
    initial: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    variance: ArrayLike | None = None,
    mode: FitMode = "difference",
    backend: BackendRequest = "auto",
    config: LMConfig | None = None,
) -> DipoleFitResult:
    """Fit batched Gaussian or sampled-stamp dipoles.

    Inputs may be NumPy or CuPy arrays, or array-like objects accepted by the
    resolved backend. ``backend="auto"`` may transfer host inputs to CuPy when
    a usable CUDA device is available. The returned object contains only
    NumPy arrays, including when fitting on a CUDA device. Solver residuals
    are weighted by the optional mask,
    variance, and split-plane weights; ``residuals`` retains the unweighted
    model-minus-data values in the original input shape. Nonfinite image or
    variance values are accepted only at excluded pixels. A nonfinite image
    value uses a zero placeholder internally, so its excluded residual entry
    is not meaningful. ``valid_pixel_count`` follows the broadcast mask,
    while degrees of freedom count only residuals with nonzero effective
    weights. Null-model statistics use the same effective weights as the fit;
    delta chi-square and fractional improvement preserve worse-than-null
    negative values. Fractional improvement is undefined for a nonpositive
    null chi-square. Split mode treats all three planes as independent; when
    difference equals positive minus negative, the reported degrees of
    freedom and uncertainties are not statistically calibrated for that
    dependence.
    """

    resolved = resolve_backend(backend)
    ap = resolved.module
    images_array = ap.asarray(images)
    batch, height, width = _validate_images(images_array, mode)
    dtype = _floating_dtype(images_array)
    images_array = images_array.astype(dtype, copy=False)

    if isinstance(model, str):
        if model == "stamp":
            raise ValueError(
                "stamp model requires a StampDipoleModel instance"
            )
        if model != "gaussian":
            raise ValueError(
                "model must be 'gaussian' or a dipole model instance"
            )
        selected_model: GaussianDipoleModel | StampDipoleModel = (
            GaussianDipoleModel(
                (height, width), backend=resolved.name, dtype=dtype
            )
        )
    elif isinstance(model, (GaussianDipoleModel, StampDipoleModel)):
        if model.image_shape != (height, width):
            raise ValueError(
                f"model image_shape {model.image_shape} does not match "
                f"images {(height, width)}"
            )
        selected_model = model.to_backend(resolved.name, dtype=dtype)
    else:
        raise TypeError("model must be a model name or dipole model instance")

    image_shape = tuple(int(value) for value in images_array.shape)
    if mask is None:
        included = ap.ones(image_shape, dtype=bool)
    else:
        mask_array = _broadcast_auxiliary(
            mask, image_shape, mode, ap, name="mask"
        )
        if mask_array.dtype.kind not in "biuf":
            raise TypeError("mask must be boolean or numeric")
        if not bool(as_numpy(ap.isfinite(mask_array).all()).item()):
            raise ValueError("mask must contain only finite values")
        included = mask_array != 0

    invalid_images = ~ap.isfinite(images_array)
    if bool(as_numpy((included & invalid_images).any()).item()):
        raise ValueError("images must be finite on included pixels")
    images_array = ap.where(invalid_images, 0, images_array)

    weights = included.astype(dtype)
    if variance is not None:
        variance_array = _broadcast_auxiliary(
            variance, image_shape, mode, ap, name="variance"
        ).astype(dtype, copy=False)
        invalid = included & (
            ~ap.isfinite(variance_array) | (variance_array <= 0)
        )
        if bool(as_numpy(invalid.any()).item()):
            raise ValueError(
                "variance must be finite and positive on included pixels"
            )
        safe_variance = ap.where(included, variance_array, 1)
        weights /= ap.sqrt(safe_variance)
    if mode == "split":
        plane_weights = ap.asarray(
            selected_model.split_weights, dtype=dtype
        ).reshape(1, 3, 1, 1)
        weights *= plane_weights

    parameter_count = len(selected_model.parameter_names)
    if initial is None:
        initial_array = _initial_guess(
            images_array,
            selected_model,
            mode,
            ap,
            weights,
        )
    else:
        initial_array = ap.asarray(initial, dtype=dtype)
        if initial_array.shape == (parameter_count,):
            initial_array = ap.broadcast_to(
                initial_array[None, :], (batch, parameter_count)
            ).copy()
        if initial_array.shape != (batch, parameter_count):
            raise ValueError(
                f"initial must have shape ({batch}, {parameter_count})"
            )
    if not bool(as_numpy(ap.isfinite(initial_array).all()).item()):
        raise ValueError("initial parameters must contain only finite values")
    gaussian_model = isinstance(selected_model, GaussianDipoleModel)
    if gaussian_model and bool(
        as_numpy((initial_array[:, 1:3] <= 0).any()).item()
    ):
        raise ValueError(
            "initial sigma_x and sigma_y values must be positive"
        )

    solver_initial = initial_array
    if gaussian_model:
        solver_initial = initial_array.copy()
        solver_initial[:, 1:3] = ap.log(initial_array[:, 1:3])

    def model_parameters(parameters: BackendArray) -> BackendArray:
        if not gaussian_model:
            return parameters
        physical = parameters.copy()
        error_state = (
            np.errstate(over="ignore", invalid="ignore")
            if ap is np
            else nullcontext()
        )
        with error_state:
            physical[:, 1:3] = ap.exp(parameters[:, 1:3])
        return physical

    def evaluate_model(parameters: BackendArray) -> BackendArray:
        if gaussian_model:
            return selected_model._evaluate_positive_unchecked(
                parameters, mode=mode
            )
        return selected_model.evaluate(parameters, mode=mode)

    def residual(
        parameters: BackendArray, *, indices: BackendArray
    ) -> BackendArray:
        prediction = evaluate_model(model_parameters(parameters))
        value = (prediction - images_array[indices]) * weights[indices]
        return value.reshape(parameters.shape[0], -1)

    jacobian_function = None
    if gaussian_model:

        def jacobian(
            parameters: BackendArray, *, indices: BackendArray
        ) -> BackendArray:
            physical = model_parameters(parameters)
            value = selected_model._jacobian_positive_unchecked(
                physical, mode=mode
            )
            chain_shape = (parameters.shape[0], 2) + (1,) * (value.ndim - 2)
            value[:, 1:3] *= physical[:, 1:3].reshape(chain_shape)
            value *= weights[indices, None, ...]
            return value.reshape(parameters.shape[0], parameter_count, -1)

        jacobian_function = jacobian

    problem = BatchedLeastSquaresProblem(residual, jacobian_function)
    low_level = batched_levenberg_marquardt(
        problem,
        solver_initial,
        config=config,
    )
    parameters_backend = model_parameters(low_level.parameters)
    if gaussian_model:
        # An ellipse orientation is periodic modulo pi. Keep portable fit
        # artifacts in one canonical interval even if an optimizer crosses a
        # periodic boundary.
        parameters_backend = parameters_backend.copy()
        half_period = ap.asarray(np.pi / 2, dtype=dtype)
        parameters_backend[:, 3] = (
            (parameters_backend[:, 3] + half_period) % (2 * half_period)
        ) - half_period
    parameters = as_numpy(parameters_backend).copy()
    raw_residuals_backend = evaluate_model(parameters_backend) - images_array
    weighted_residuals = raw_residuals_backend * weights
    chi_square = as_numpy(
        ap.sum(
            weighted_residuals * weighted_residuals,
            axis=tuple(range(1, weighted_residuals.ndim)),
        )
    )
    residual_norm = np.sqrt(chi_square)
    reduction_axes = tuple(range(1, weights.ndim))
    valid_pixel_count = as_numpy(
        ap.sum(included, axis=reduction_axes)
    ).astype(np.int64)
    possible_pixel_count = int(np.prod(image_shape[1:], dtype=np.int64))
    valid_pixel_fraction = (
        valid_pixel_count.astype(np.float64) / possible_pixel_count
    )
    weighted_null = images_array * weights
    null_chi_square = as_numpy(
        ap.sum(weighted_null * weighted_null, axis=reduction_axes)
    )
    finite_fit_quality = np.isfinite(null_chi_square) & np.isfinite(
        chi_square
    )
    delta_chi_square = np.full_like(null_chi_square, np.nan)
    delta_chi_square[finite_fit_quality] = (
        null_chi_square[finite_fit_quality] - chi_square[finite_fit_quality]
    )
    fractional_null_improvement = np.full(batch, np.nan, dtype=np.float64)
    positive_null = finite_fit_quality & (null_chi_square > 0)
    fractional_null_improvement[positive_null] = 1 - np.divide(
        chi_square[positive_null],
        null_chi_square[positive_null],
    )
    effective_observation_count = as_numpy(
        ap.sum(weights != 0, axis=reduction_axes)
    ).astype(np.int64)
    degrees_of_freedom = effective_observation_count - parameter_count
    reduced_chi_square = np.full(batch, np.nan, dtype=np.float64)
    positive_dof = degrees_of_freedom > 0
    reduced_chi_square[positive_dof] = (
        chi_square[positive_dof] / degrees_of_freedom[positive_dof]
    )

    status_codes = as_numpy(low_level.status).astype(np.int8)
    status = _status_names(status_codes)
    converged = as_numpy(low_level.converged).astype(bool)
    rank = as_numpy(low_level.rank).astype(np.int64)
    jacobian_available = np.isfinite(as_numpy(low_level.jacobian)).all(
        axis=(1, 2)
    )
    base_covariance = as_numpy(low_level.covariance)
    if gaussian_model:
        covariance_chain = np.ones_like(parameters)
        covariance_chain[:, 1:3] = parameters[:, 1:3]
        base_covariance = (
            base_covariance
            * covariance_chain[:, :, None]
            * covariance_chain[:, None, :]
        )
    covariance = np.full(
        (batch, parameter_count, parameter_count), np.nan, dtype=dtype
    )
    standard_errors = np.full((batch, parameter_count), np.nan, dtype=dtype)
    uncertainty_valid = np.zeros(batch, dtype=bool)
    reasons: list[str] = []
    for row in range(batch):
        reason = ""
        if not converged[row]:
            reason = f"fit did not converge: {status[row]}"
        elif degrees_of_freedom[row] <= 0:
            reason = "insufficient degrees of freedom"
        elif not jacobian_available[row]:
            reason = "final Jacobian is unavailable"
        elif rank[row] < parameter_count:
            reason = "rank-deficient Jacobian"
        elif not np.isfinite(base_covariance[row]).all():
            reason = "covariance is not finite"
        else:
            covariance[row] = base_covariance[row]
            if variance is None:
                covariance[row] *= reduced_chi_square[row]
            diagonal = np.diag(covariance[row])
            if np.any(diagonal < 0) or not np.isfinite(diagonal).all():
                covariance[row] = np.nan
                reason = "covariance diagonal is invalid"
            else:
                standard_errors[row] = np.sqrt(diagonal)
                uncertainty_valid[row] = True
        reasons.append(reason)

    return DipoleFitResult(
        parameters=parameters,
        parameter_names=selected_model.parameter_names,
        status=status,
        converged=converged,
        evaluations=as_numpy(low_level.evaluations).astype(np.int64),
        residual_norm=residual_norm,
        chi_square=chi_square,
        valid_pixel_count=valid_pixel_count,
        valid_pixel_fraction=valid_pixel_fraction,
        null_chi_square=null_chi_square,
        delta_chi_square=delta_chi_square,
        fractional_null_improvement=fractional_null_improvement,
        degrees_of_freedom=degrees_of_freedom,
        reduced_chi_square=reduced_chi_square,
        covariance=covariance,
        standard_errors=standard_errors,
        uncertainty_valid=uncertainty_valid,
        uncertainty_reason=tuple(reasons),
        residuals=as_numpy(raw_residuals_backend).copy(),
        backend=resolved.name,
        device=resolved.device,
        dtype=cast(FloatDType, dtype.name),
        model=selected_model.name,
        mode=mode,
    )


__all__ = ["DipoleFitResult", "fit_dipoles"]
