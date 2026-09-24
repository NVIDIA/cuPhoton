# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""High-level portable xFit API."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from enum import IntEnum
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
from .backend import Backend, as_numpy, resolve_backend
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


class DipoleFitUncertaintyReason(IntEnum):
    """Stable reason codes for unavailable dipole-fit uncertainties."""

    VALID = 0
    FIT_NOT_CONVERGED = 1
    INSUFFICIENT_DEGREES_OF_FREEDOM = 2
    FINAL_JACOBIAN_UNAVAILABLE = 3
    RANK_DEFICIENT_JACOBIAN = 4
    COVARIANCE_NOT_FINITE = 5
    COVARIANCE_DIAGONAL_INVALID = 6


@dataclass(frozen=True, slots=True)
class DeviceDipoleFitResult:
    """CuPy-resident result of a batched dipole fit.

    All array-valued fields remain on ``device_id`` and are held by strong
    references. This object is an in-process device result, not a Dragon
    payload or persisted artifact, and cannot be pickled. Callers should treat
    its arrays as read-only and consume them on the CuPy stream that was
    current during ``fit_dipoles_device``. The result does not retain that
    stream or expose a completion event for direct cross-stream consumption.
    Initiate a stream-aware DLPack handoff while that producer stream is still
    current. The solver still copies per-fit status, evaluation counts and
    boolean diagnostic masks to the host for control flow. Eliminating those
    bookkeeping copies or scalar synchronizations is not part of this
    result-residency contract.

    ``status_codes`` contains :class:`LMStatus` values and
    ``uncertainty_reason_codes`` contains
    :class:`DipoleFitUncertaintyReason` values. Floating-point compute arrays
    use ``dtype``; portable fractions and reduced chi-square remain float64.
    Fitting returns residual arrays. Callers that no longer need them may
    use ``dataclasses.replace(result, residuals=None)`` and release the
    original result before transforming xScan features.
    """

    parameters: Any
    parameter_names: tuple[str, ...]
    status_codes: Any
    converged: Any
    evaluations: Any
    residual_norm: Any
    chi_square: Any
    valid_pixel_count: Any
    valid_pixel_fraction: Any
    null_chi_square: Any
    delta_chi_square: Any
    fractional_null_improvement: Any
    degrees_of_freedom: Any
    reduced_chi_square: Any
    covariance: Any
    standard_errors: Any
    uncertainty_valid: Any
    uncertainty_reason_codes: Any
    residuals: Any | None
    device_id: int
    dtype: FloatDType
    model: ModelName
    mode: FitMode
    schema: Literal["cuphoton.xfit.device-fit-result/v1"] = field(
        init=False,
        default="cuphoton.xfit.device-fit-result/v1",
    )
    solver: Literal["levenberg-marquardt"] = field(
        init=False,
        default="levenberg-marquardt",
    )
    backend: Literal["cupy"] = field(init=False, default="cupy")
    result_location: Literal["device"] = field(
        init=False,
        default="device",
    )

    def __post_init__(self) -> None:
        if isinstance(self.device_id, bool) or not isinstance(
            self.device_id, (int, np.integer)
        ):
            raise TypeError(
                "device_id must be an integer CUDA device ordinal"
            )
        if self.device_id < 0:
            raise ValueError("device_id must be non-negative")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be 'float32' or 'float64'")
        if self.model not in {"gaussian", "stamp"}:
            raise ValueError("model must be 'gaussian' or 'stamp'")
        if self.mode not in {"difference", "split"}:
            raise ValueError("mode must be 'difference' or 'split'")
        if not self.parameter_names or not all(
            isinstance(name, str) and name for name in self.parameter_names
        ):
            raise ValueError("parameter_names must contain nonempty strings")

        cp = _load_cupy()
        arrays = {
            "parameters": self.parameters,
            "status_codes": self.status_codes,
            "converged": self.converged,
            "evaluations": self.evaluations,
            "residual_norm": self.residual_norm,
            "chi_square": self.chi_square,
            "valid_pixel_count": self.valid_pixel_count,
            "valid_pixel_fraction": self.valid_pixel_fraction,
            "null_chi_square": self.null_chi_square,
            "delta_chi_square": self.delta_chi_square,
            "fractional_null_improvement": (self.fractional_null_improvement),
            "degrees_of_freedom": self.degrees_of_freedom,
            "reduced_chi_square": self.reduced_chi_square,
            "covariance": self.covariance,
            "standard_errors": self.standard_errors,
            "uncertainty_valid": self.uncertainty_valid,
            "uncertainty_reason_codes": self.uncertainty_reason_codes,
            "residuals": self.residuals,
        }
        if self.residuals is None:
            del arrays["residuals"]
        for name, value in arrays.items():
            if not isinstance(value, cp.ndarray):
                raise TypeError(f"{name} must be a CuPy array")
            if int(value.device.id) != int(self.device_id):
                raise ValueError(
                    f"{name} is on CUDA device {int(value.device.id)}, "
                    f"expected {int(self.device_id)}"
                )

        if self.parameters.ndim != 2:
            raise ValueError("parameters must have shape (batch, parameters)")
        batch, parameter_count = (
            int(self.parameters.shape[0]),
            int(self.parameters.shape[1]),
        )
        if batch < 1 or parameter_count < 1:
            raise ValueError("parameter dimensions must be nonzero")
        if len(self.parameter_names) != parameter_count:
            raise ValueError("parameter_names must align with parameters")
        for name in (
            "status_codes",
            "converged",
            "evaluations",
            "residual_norm",
            "chi_square",
            "valid_pixel_count",
            "valid_pixel_fraction",
            "null_chi_square",
            "delta_chi_square",
            "fractional_null_improvement",
            "degrees_of_freedom",
            "reduced_chi_square",
            "uncertainty_valid",
            "uncertainty_reason_codes",
        ):
            if arrays[name].shape != (batch,):
                raise ValueError(f"{name} must have shape ({batch},)")
        expected_square_shape = (batch, parameter_count, parameter_count)
        if self.covariance.shape != expected_square_shape:
            raise ValueError(
                f"covariance must have shape {expected_square_shape}"
            )
        if self.standard_errors.shape != (batch, parameter_count):
            raise ValueError(
                "standard_errors must align with the parameter array"
            )
        if self.residuals is not None:
            if self.residuals.shape[0] != batch:
                raise ValueError(
                    "residuals must align with the parameter batch"
                )
            if self.mode == "difference" and self.residuals.ndim != 3:
                raise ValueError(
                    "difference residuals must have shape "
                    "(batch, height, width)"
                )
            if self.mode == "split" and (
                self.residuals.ndim != 4 or self.residuals.shape[1] != 3
            ):
                raise ValueError(
                    "split residuals must have shape "
                    "(batch, 3, height, width)"
                )

        compute_float_arrays = (
            "parameters",
            "residual_norm",
            "chi_square",
            "null_chi_square",
            "delta_chi_square",
            "covariance",
            "standard_errors",
            "residuals",
        )
        for name in compute_float_arrays:
            if name == "residuals" and self.residuals is None:
                continue
            if np.dtype(arrays[name].dtype) != np.dtype(self.dtype):
                raise TypeError(f"{name} must have {self.dtype} dtype")
        for name in (
            "valid_pixel_fraction",
            "fractional_null_improvement",
            "reduced_chi_square",
        ):
            if np.dtype(arrays[name].dtype) != np.dtype(np.float64):
                raise TypeError(f"{name} must have float64 dtype")
        expected_dtypes = {
            "status_codes": np.dtype(np.int8),
            "converged": np.dtype(bool),
            "evaluations": np.dtype(np.int64),
            "valid_pixel_count": np.dtype(np.int64),
            "degrees_of_freedom": np.dtype(np.int64),
            "uncertainty_valid": np.dtype(bool),
            "uncertainty_reason_codes": np.dtype(np.int8),
        }
        for name, expected_dtype in expected_dtypes.items():
            if np.dtype(arrays[name].dtype) != expected_dtype:
                raise TypeError(
                    f"{name} must have {expected_dtype.name} dtype"
                )

    def __reduce_ex__(self, protocol: int) -> Any:
        raise TypeError(
            "DeviceDipoleFitResult cannot be pickled; keep device results "
            "inside the producing process"
        )


@dataclass(frozen=True, slots=True)
class _BackendDipoleFitResult:
    parameters: BackendArray
    parameter_names: tuple[str, ...]
    status_codes: BackendArray
    converged: BackendArray
    evaluations: BackendArray
    residual_norm: BackendArray
    chi_square: BackendArray
    valid_pixel_count: BackendArray
    valid_pixel_fraction: BackendArray
    null_chi_square: BackendArray
    delta_chi_square: BackendArray
    fractional_null_improvement: BackendArray
    degrees_of_freedom: BackendArray
    reduced_chi_square: BackendArray
    covariance: BackendArray
    standard_errors: BackendArray
    uncertainty_valid: BackendArray
    uncertainty_reason_codes: BackendArray
    residuals: BackendArray
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


def _uncertainty_reasons(
    reason_codes: np.ndarray,
    status: np.ndarray,
) -> tuple[str, ...]:
    messages = {
        DipoleFitUncertaintyReason.VALID: "",
        DipoleFitUncertaintyReason.INSUFFICIENT_DEGREES_OF_FREEDOM: (
            "insufficient degrees of freedom"
        ),
        DipoleFitUncertaintyReason.FINAL_JACOBIAN_UNAVAILABLE: (
            "final Jacobian is unavailable"
        ),
        DipoleFitUncertaintyReason.RANK_DEFICIENT_JACOBIAN: (
            "rank-deficient Jacobian"
        ),
        DipoleFitUncertaintyReason.COVARIANCE_NOT_FINITE: (
            "covariance is not finite"
        ),
        DipoleFitUncertaintyReason.COVARIANCE_DIAGONAL_INVALID: (
            "covariance diagonal is invalid"
        ),
    }
    reasons: list[str] = []
    for code_value, status_name in zip(reason_codes, status, strict=True):
        try:
            code = DipoleFitUncertaintyReason(int(code_value))
        except ValueError as exc:
            raise ValueError(
                f"unknown dipole uncertainty reason code {int(code_value)}"
            ) from exc
        if code is DipoleFitUncertaintyReason.FIT_NOT_CONVERGED:
            reasons.append(f"fit did not converge: {status_name}")
        else:
            reasons.append(messages[code])
    return tuple(reasons)


def _load_cupy() -> Any:
    try:
        import cupy as cp
    except ImportError as exc:
        raise ImportError(
            "fit_dipoles_device requires CuPy; run 'uv sync --extra gpu' "
            "for development or install 'cuphoton[gpu]'"
        ) from exc
    return cp


def _active_cupy_device_id(cp: Any) -> int:
    message = "fit_dipoles_device requires at least one visible CUDA device"
    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:
        raise RuntimeError(message) from exc
    if device_count < 1:
        raise RuntimeError(message)
    try:
        return int(cp.cuda.runtime.getDevice())
    except Exception as exc:
        raise RuntimeError(message) from exc


def _validate_cupy_input_device(
    value: Any,
    *,
    name: str,
    active_device_id: int,
    cp: Any,
) -> None:
    if value is None or not isinstance(value, cp.ndarray):
        return
    device_id = int(value.device.id)
    if device_id != active_device_id:
        raise ValueError(
            f"{name} is on CUDA device {device_id}; "
            f"active CUDA device is {active_device_id}"
        )


def _validate_cupy_model_device(
    model: Literal["gaussian"] | GaussianDipoleModel | StampDipoleModel,
    *,
    active_device_id: int,
    cp: Any,
) -> None:
    if not isinstance(model, (GaussianDipoleModel, StampDipoleModel)):
        return
    if model.backend == "numpy":
        return
    if model.device != f"cuda:{active_device_id}":
        raise ValueError(
            f"model is on {model.device}; active CUDA device is "
            f"{active_device_id}"
        )
    if isinstance(model, StampDipoleModel):
        for attribute in ("stamp_basis", "basis_weights", "_stamp"):
            _validate_cupy_input_device(
                getattr(model, attribute),
                name=f"model.{attribute}",
                active_device_id=active_device_id,
                cp=cp,
            )


def _validate_cupy_stamp_model_dtype(
    model: Literal["gaussian"] | GaussianDipoleModel | StampDipoleModel,
    images: ArrayLike,
    *,
    cp: Any,
) -> None:
    if not isinstance(model, StampDipoleModel) or model.backend != "cupy":
        return
    image_array = (
        images if isinstance(images, cp.ndarray) else np.asarray(images)
    )
    image_dtype = _floating_dtype(image_array)
    if model.dtype != image_dtype:
        raise ValueError(
            f"CuPy StampDipoleModel dtype {model.dtype.name} must match "
            f"image compute dtype {image_dtype.name}"
        )


def _fit_dipoles_backend(
    images: ArrayLike,
    *,
    model: Literal["gaussian"] | GaussianDipoleModel | StampDipoleModel,
    initial: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    variance: ArrayLike | None = None,
    mode: FitMode = "difference",
    resolved: Backend,
    config: LMConfig | None = None,
) -> _BackendDipoleFitResult:
    """Fit dipoles while retaining all result arrays in ``resolved``."""

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
        if (
            model.backend == resolved.name
            and model.device == resolved.device
            and model.dtype == dtype
        ):
            selected_model = model
        else:
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

    normal_equations = None
    if gaussian_model and resolved.name == "cutile":
        from ._cutile import gaussian_normal_equations

        def normal_equations(
            parameters: BackendArray,
            residuals: BackendArray,
            *,
            indices: BackendArray,
        ) -> tuple[BackendArray, BackendArray]:
            active_weights = weights[indices].reshape(residuals.shape)
            return gaussian_normal_equations(
                parameters,
                residuals,
                active_weights,
                image_shape=(height, width),
                mode=mode,
            )

    problem = BatchedLeastSquaresProblem(
        residual,
        jacobian_function,
        normal_equations=normal_equations,
    )
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
    raw_residuals_backend = evaluate_model(parameters_backend) - images_array
    weighted_residuals = raw_residuals_backend * weights
    chi_square = ap.sum(
        weighted_residuals * weighted_residuals,
        axis=tuple(range(1, weighted_residuals.ndim)),
    )
    residual_norm = ap.sqrt(chi_square)
    reduction_axes = tuple(range(1, weights.ndim))
    valid_pixel_count = ap.sum(included, axis=reduction_axes).astype(np.int64)
    possible_pixel_count = int(np.prod(image_shape[1:], dtype=np.int64))
    valid_pixel_fraction = (
        valid_pixel_count.astype(np.float64) / possible_pixel_count
    )
    weighted_null = images_array * weights
    null_chi_square = ap.sum(
        weighted_null * weighted_null, axis=reduction_axes
    )
    finite_fit_quality = ap.isfinite(null_chi_square) & ap.isfinite(
        chi_square
    )
    delta_chi_square = ap.full_like(null_chi_square, ap.nan)
    delta_chi_square[finite_fit_quality] = (
        null_chi_square[finite_fit_quality] - chi_square[finite_fit_quality]
    )
    fractional_null_improvement = ap.full(batch, ap.nan, dtype=np.float64)
    positive_null = finite_fit_quality & (null_chi_square > 0)
    fractional_null_improvement[positive_null] = 1 - ap.divide(
        chi_square[positive_null], null_chi_square[positive_null]
    )
    effective_observation_count = ap.sum(
        weights != 0, axis=reduction_axes
    ).astype(np.int64)
    degrees_of_freedom = effective_observation_count - parameter_count
    reduced_chi_square = ap.full(batch, ap.nan, dtype=np.float64)
    positive_dof = degrees_of_freedom > 0
    reduced_chi_square[positive_dof] = (
        chi_square[positive_dof] / degrees_of_freedom[positive_dof]
    )

    status_codes = low_level.status.astype(np.int8, copy=True)
    converged = low_level.converged.astype(bool, copy=True)
    rank = low_level.rank.astype(np.int64)
    jacobian_available = ap.isfinite(low_level.jacobian).all(axis=(1, 2))
    base_covariance = low_level.covariance
    if gaussian_model:
        covariance_chain = ap.ones_like(parameters_backend)
        covariance_chain[:, 1:3] = parameters_backend[:, 1:3]
        base_covariance = (
            base_covariance
            * covariance_chain[:, :, None]
            * covariance_chain[:, None, :]
        )
    scaled_covariance = base_covariance
    if variance is None:
        scaled_covariance = base_covariance.copy()
        scaled_covariance *= reduced_chi_square[:, None, None]
    base_covariance_finite = ap.isfinite(base_covariance).all(axis=(1, 2))
    diagonal = ap.diagonal(scaled_covariance, axis1=1, axis2=2)
    diagonal_valid = ap.isfinite(diagonal).all(axis=1) & ~(diagonal < 0).any(
        axis=1
    )

    uncertainty_reason_codes = ap.full(
        batch,
        int(DipoleFitUncertaintyReason.VALID),
        dtype=np.int8,
    )
    uncertainty_reason_codes[~diagonal_valid] = int(
        DipoleFitUncertaintyReason.COVARIANCE_DIAGONAL_INVALID
    )
    uncertainty_reason_codes[~base_covariance_finite] = int(
        DipoleFitUncertaintyReason.COVARIANCE_NOT_FINITE
    )
    uncertainty_reason_codes[rank < parameter_count] = int(
        DipoleFitUncertaintyReason.RANK_DEFICIENT_JACOBIAN
    )
    uncertainty_reason_codes[~jacobian_available] = int(
        DipoleFitUncertaintyReason.FINAL_JACOBIAN_UNAVAILABLE
    )
    uncertainty_reason_codes[degrees_of_freedom <= 0] = int(
        DipoleFitUncertaintyReason.INSUFFICIENT_DEGREES_OF_FREEDOM
    )
    uncertainty_reason_codes[~converged] = int(
        DipoleFitUncertaintyReason.FIT_NOT_CONVERGED
    )
    uncertainty_valid = uncertainty_reason_codes == int(
        DipoleFitUncertaintyReason.VALID
    )
    covariance = ap.where(
        uncertainty_valid[:, None, None], scaled_covariance, ap.nan
    )
    safe_diagonal = ap.where(diagonal_valid[:, None], diagonal, 0)
    standard_errors = ap.where(
        uncertainty_valid[:, None], ap.sqrt(safe_diagonal), ap.nan
    )

    return _BackendDipoleFitResult(
        parameters=parameters_backend,
        parameter_names=selected_model.parameter_names,
        status_codes=status_codes,
        converged=converged,
        evaluations=low_level.evaluations.astype(np.int64),
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
        uncertainty_reason_codes=uncertainty_reason_codes,
        residuals=raw_residuals_backend,
        backend=resolved.name,
        device=resolved.device,
        dtype=cast(FloatDType, dtype.name),
        model=selected_model.name,
        mode=mode,
    )


def _materialize_dipole_fit_result(
    result: _BackendDipoleFitResult,
) -> DipoleFitResult:
    status_codes = as_numpy(result.status_codes).astype(np.int8, copy=False)
    status = _status_names(status_codes)
    reason_codes = as_numpy(result.uncertainty_reason_codes).astype(
        np.int8, copy=False
    )
    return DipoleFitResult(
        parameters=as_numpy(result.parameters).copy(),
        parameter_names=result.parameter_names,
        status=status,
        converged=as_numpy(result.converged).astype(bool, copy=True),
        evaluations=as_numpy(result.evaluations).astype(np.int64, copy=True),
        residual_norm=as_numpy(result.residual_norm).copy(),
        chi_square=as_numpy(result.chi_square).copy(),
        valid_pixel_count=as_numpy(result.valid_pixel_count).astype(
            np.int64, copy=True
        ),
        valid_pixel_fraction=as_numpy(result.valid_pixel_fraction).copy(),
        null_chi_square=as_numpy(result.null_chi_square).copy(),
        delta_chi_square=as_numpy(result.delta_chi_square).copy(),
        fractional_null_improvement=as_numpy(
            result.fractional_null_improvement
        ).copy(),
        degrees_of_freedom=as_numpy(result.degrees_of_freedom).astype(
            np.int64, copy=True
        ),
        reduced_chi_square=as_numpy(result.reduced_chi_square).copy(),
        covariance=as_numpy(result.covariance).copy(),
        standard_errors=as_numpy(result.standard_errors).copy(),
        uncertainty_valid=as_numpy(result.uncertainty_valid).astype(
            bool, copy=True
        ),
        uncertainty_reason=_uncertainty_reasons(reason_codes, status),
        residuals=as_numpy(result.residuals).copy(),
        backend=result.backend,
        device=result.device,
        dtype=result.dtype,
        model=result.model,
        mode=result.mode,
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
    a usable CUDA device is available. The returned array-valued fields are
    independently owned NumPy arrays, including when fitting on a CUDA device.
    Solver residuals
    are weighted by the optional mask, variance, and split-plane weights;
    ``residuals`` retains the unweighted model-minus-data values in the
    original input shape. Nonfinite image or variance values are accepted only
    at excluded pixels. A nonfinite image value uses a zero placeholder
    internally, so its excluded residual entry is not meaningful.
    ``valid_pixel_count`` follows the broadcast mask, while degrees of freedom
    count only residuals with nonzero effective weights. Null-model statistics
    use the same effective weights as the fit; delta chi-square and fractional
    improvement preserve worse-than-null negative values. Fractional
    improvement is undefined for a nonpositive null chi-square. Split mode
    treats all three planes as independent; when difference equals positive
    minus negative, the reported degrees of freedom and uncertainties are not
    statistically calibrated for that dependence.
    """

    if backend == "cutile" and isinstance(model, StampDipoleModel):
        # Report the model restriction before the finite-difference one:
        # sampled-stamp fits always difference the residual.
        raise ValueError(
            "backend='cutile' currently supports only the Gaussian model"
        )
    if (
        backend == "cutile"
        and config is not None
        and config.use_finite_difference
    ):
        raise ValueError(
            "backend='cutile' does not support finite-difference fitting"
        )
    resolved = resolve_backend(backend)
    backend_result = _fit_dipoles_backend(
        images,
        model=model,
        initial=initial,
        mask=mask,
        variance=variance,
        mode=mode,
        resolved=resolved,
        config=config,
    )
    return _materialize_dipole_fit_result(backend_result)


def fit_dipoles_device(
    images: ArrayLike,
    *,
    model: Literal["gaussian"] | GaussianDipoleModel | StampDipoleModel,
    initial: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    variance: ArrayLike | None = None,
    mode: FitMode = "difference",
    config: LMConfig | None = None,
) -> DeviceDipoleFitResult:
    """Fit dipoles while keeping every result array on the active GPU.

    This experimental seam is explicitly CuPy-only and has no backend
    selector. NumPy inputs are copied to the active device. Existing CuPy
    inputs must already reside on that device and are checked before any input
    conversion. Input arrays are borrowed and are not mutated.

    The solver may copy per-fit status, evaluation counts and boolean
    diagnostic masks to the host for control flow. Parameters, residuals,
    covariance and other numerical fit arrays remain on the device; this API
    does not materialize the returned array fields on the host.

    Inputs must be ready on CuPy's current stream. Work and result arrays use
    that same stream; consume the results there, or initiate a stream-aware
    DLPack handoff before leaving its context. This function does not return a
    stream or completion event and does not support unordered direct reads
    from another stream. Retain the result until its consumers have finished.
    """

    cp = _load_cupy()
    active_device_id = _active_cupy_device_id(cp)
    for name, value in (
        ("images", images),
        ("initial", initial),
        ("mask", mask),
        ("variance", variance),
    ):
        _validate_cupy_input_device(
            value,
            name=name,
            active_device_id=active_device_id,
            cp=cp,
        )
    _validate_cupy_model_device(
        model,
        active_device_id=active_device_id,
        cp=cp,
    )
    _validate_cupy_stamp_model_dtype(model, images, cp=cp)

    result = _fit_dipoles_backend(
        images,
        model=model,
        initial=initial,
        mask=mask,
        variance=variance,
        mode=mode,
        resolved=resolve_backend("cupy"),
        config=config,
    )
    return DeviceDipoleFitResult(
        parameters=result.parameters,
        parameter_names=result.parameter_names,
        status_codes=result.status_codes,
        converged=result.converged,
        evaluations=result.evaluations,
        residual_norm=result.residual_norm,
        chi_square=result.chi_square,
        valid_pixel_count=result.valid_pixel_count,
        valid_pixel_fraction=result.valid_pixel_fraction,
        null_chi_square=result.null_chi_square,
        delta_chi_square=result.delta_chi_square,
        fractional_null_improvement=result.fractional_null_improvement,
        degrees_of_freedom=result.degrees_of_freedom,
        reduced_chi_square=result.reduced_chi_square,
        covariance=result.covariance,
        standard_errors=result.standard_errors,
        uncertainty_valid=result.uncertainty_valid,
        uncertainty_reason_codes=result.uncertainty_reason_codes,
        residuals=result.residuals,
        device_id=active_device_id,
        dtype=result.dtype,
        model=result.model,
        mode=result.mode,
    )


__all__ = [
    "DeviceDipoleFitResult",
    "DipoleFitResult",
    "DipoleFitUncertaintyReason",
    "fit_dipoles",
    "fit_dipoles_device",
]
