# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Batched nonlinear least-squares fitting for astronomical dipoles."""

from cuphoton import __version__

from ._types import (
    ArrayLike,
    BackendArray,
    BackendRequest,
    ComputeDType,
    FitMode,
    FloatDType,
    ModelName,
    ResolvedBackend,
    StampEvaluation,
)
from .api import DipoleFitResult, fit_dipoles
from .models import GaussianDipoleModel, StampDipoleModel
from .solver import (
    BatchedLeastSquaresProblem,
    JacobianFunction,
    LMConfig,
    LMResult,
    LMStatus,
    ResidualFunction,
    batched_levenberg_marquardt,
)

__all__ = [
    "__version__",
    "ArrayLike",
    "BackendArray",
    "BackendRequest",
    "BatchedLeastSquaresProblem",
    "ComputeDType",
    "DipoleFitResult",
    "FitMode",
    "FloatDType",
    "GaussianDipoleModel",
    "JacobianFunction",
    "LMConfig",
    "LMResult",
    "LMStatus",
    "ModelName",
    "ResidualFunction",
    "ResolvedBackend",
    "StampEvaluation",
    "StampDipoleModel",
    "batched_levenberg_marquardt",
    "fit_dipoles",
]
