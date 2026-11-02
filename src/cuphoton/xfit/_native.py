# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Optional coarse CUDA C++ Gaussian fitting, with thread-owned workspaces."""

from __future__ import annotations

import importlib
import importlib.util
import re
import threading
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from ._types import ArrayLike, FitMode, FloatDType
from .backend import Backend
from .models import GaussianDipoleModel, StampDipoleModel
from .solver import LMConfig, _settings

if TYPE_CHECKING:
    from .api import _BackendDipoleFitResult

_worker = threading.local()


def load_extension() -> ModuleType:
    """Load the optional extension only for an explicitly selected backend."""

    if importlib.util.find_spec("cuphoton.xfit._native_ext") is None:
        raise RuntimeError(
            "the native xFit extension is unavailable; install cuphoton[io] "
            "from a native wheel, or build cuphoton with "
            "CUPHOTON_XFIT_BUILD_EXT=1 and a CUDA 13 toolkit"
        )
    try:
        from cuda.pathfinder import load_nvidia_dynamic_lib

        for name, major in (("cudart", 13), ("cublas", 13), ("cusolver", 12)):
            # Pathfinder loads cuBLASLt before cuBLAS and retains the handles.
            library = load_nvidia_dynamic_lib(name)
            if not re.fullmatch(
                rf"lib{name}\.so\.{major}(?:\.\d+)*",
                Path(library.abs_path).name,
            ):
                raise RuntimeError(
                    f"expected CUDA 13 lib{name} (SONAME {major}), "
                    f"found {library.abs_path}"
                )
    except (ImportError, OSError, RuntimeError) as exc:
        raise RuntimeError(
            "native xFit requires the CUDA 13 runtime, cuBLAS and "
            "cuSOLVER libraries; "
            "install cuphoton[io]. "
            f"Library loading failed: {exc}"
        ) from exc
    try:
        return importlib.import_module("cuphoton.xfit._native_ext")
    except ImportError as exc:
        raise RuntimeError(
            "the native xFit extension could not be loaded; install "
            "cuphoton[io] from a native wheel, or build cuphoton with "
            "CUPHOTON_XFIT_BUILD_EXT=1 and a CUDA 13 toolkit. "
            f"Extension loading failed: {exc}"
        ) from exc


def release_workspace() -> None:
    """Release this worker's native workspace after its synchronous calls."""

    if hasattr(_worker, "workspace"):
        del _worker.workspace


def last_timings() -> dict[str, Any]:
    """Return native iteration timings from the last call in this worker."""

    return dict(getattr(_worker, "timings", {}))


def fit_gaussian_pipeline(
    images: ArrayLike,
    *,
    model: Literal["gaussian"] | GaussianDipoleModel | StampDipoleModel,
    initial: ArrayLike | None,
    mask: ArrayLike | None,
    variance: ArrayLike | None,
    mode: FitMode,
    resolved: Backend,
    config: LMConfig | None,
) -> _BackendDipoleFitResult:
    """Prepare, solve and finalize using CuPy only as a buffer owner.

    Broadcasts and dimension insertion create views. All numerical conversion,
    validation, reductions, model evaluation and diagnostics execute in the
    ahead-of-time extension or directly called CUDA libraries.
    """
    from .api import (
        _BackendDipoleFitResult,
        _broadcast_auxiliary,
        _floating_dtype,
        _validate_images,
    )

    cp = resolved.module
    extension = load_extension()
    images_array = cp.asarray(images)
    batch, height, width = _validate_images(images_array, mode)
    dtype = _floating_dtype(images_array)
    if isinstance(model, str):
        if model != "gaussian":
            raise ValueError(
                "model must be 'gaussian' or a dipole model instance"
            )
        selected = GaussianDipoleModel(
            (height, width), backend="native", dtype=dtype
        )
    elif isinstance(model, GaussianDipoleModel):
        if model.image_shape != (height, width):
            raise ValueError(
                f"model image_shape {model.image_shape} does not match "
                f"images {(height, width)}"
            )
        selected = model
    else:
        raise TypeError("model must be a model name or dipole model instance")
    settings = _settings(LMConfig() if config is None else config, dtype, 8)
    if settings.use_finite_difference:
        raise ValueError("native xFit requires an analytic Gaussian Jacobian")
    image_shape = tuple(images_array.shape)

    def image_view(array: Any) -> Any:
        return array[:, None, :, :] if mode == "difference" else array

    def auxiliary(value: Any, name: str) -> Any:
        if value is None:
            return None
        array = _broadcast_auxiliary(value, image_shape, mode, cp, name=name)
        if name == "mask" and array.dtype.kind not in "biuf":
            raise TypeError("mask must be boolean or numeric")
        return image_view(array)

    mask_array = auxiliary(mask, "mask")
    variance_array = auxiliary(variance, "variance")
    initial_array = None
    if initial is not None:
        initial_array = cp.asarray(initial)
        if initial_array.shape == (8,):
            initial_array = cp.broadcast_to(
                initial_array[None, :], (batch, 8)
            )
        if initial_array.shape != (batch, 8):
            raise ValueError(f"initial must have shape ({batch}, 8)")
        initial_array = initial_array[:, None, None, :]
    device = int(cp.cuda.Device().id)
    cached = getattr(_worker, "workspace", None)
    if cached is None or cached[0] != device:
        cached = (device, extension.create_workspace(device))
        _worker.workspace = cached
    observations = height * width * (1 if mode == "difference" else 3)
    vector = (batch,)
    pixels = (batch, observations)
    matrix = (batch, 8)
    square = (batch, 8, 8)
    specifications = (
        (matrix, dtype),
        (pixels, dtype),
        (pixels, dtype),
        (pixels, dtype),
        (vector, np.int8),
        (vector, np.int32),
        (vector, bool),
        (pixels, bool),
        (matrix, dtype),
        (pixels, dtype),
        (vector, bool),
        (vector, np.int64),
        (vector, dtype),
        (vector, dtype),
        (vector, np.int64),
        (vector, np.float64),
        (vector, dtype),
        (vector, dtype),
        (vector, np.float64),
        (vector, np.int64),
        (vector, np.float64),
        (square, dtype),
        (square, dtype),
        (matrix, dtype),
        (vector, bool),
        (vector, np.int8),
        (matrix, dtype),
        (square, dtype),
        (vector, np.int32),
    )
    outputs = tuple(
        cp.empty(shape, dtype=kind) for shape, kind in specifications
    )
    producer = cp.cuda.get_current_stream()
    _worker.timings = extension.fit(
        cached[1],
        (image_view(images_array), mask_array, variance_array, initial_array),
        outputs,
        (
            settings.f_tol,
            settings.x_tol,
            settings.g_tol,
            settings.initial_damping,
            settings.damping_increase,
            settings.damping_decrease,
            settings.max_evaluations,
        ),
        (batch, height, width, 1 if mode == "difference" else 3),
        selected.split_weights,
        producer.ptr,
    )
    return _BackendDipoleFitResult(
        parameters=outputs[8],
        parameter_names=selected.parameter_names,
        status_codes=outputs[4],
        converged=outputs[10],
        evaluations=outputs[11],
        residual_norm=outputs[12],
        chi_square=outputs[13],
        valid_pixel_count=outputs[14],
        valid_pixel_fraction=outputs[15],
        null_chi_square=outputs[16],
        delta_chi_square=outputs[17],
        fractional_null_improvement=outputs[18],
        degrees_of_freedom=outputs[19],
        reduced_chi_square=outputs[20],
        covariance=outputs[21],
        covariance_factor=outputs[22],
        standard_errors=outputs[23],
        uncertainty_valid=outputs[24],
        uncertainty_reason_codes=outputs[25],
        residuals=outputs[9].reshape(image_shape),
        backend="native",
        device=resolved.device,
        dtype=cast(FloatDType, dtype.name),
        model="gaussian",
        mode=mode,
    )
