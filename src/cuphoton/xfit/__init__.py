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
from .api import (
    DeviceDipoleFitResult,
    DipoleFitResult,
    DipoleFitUncertaintyReason,
    fit_dipoles,
    fit_dipoles_device,
)
from .derived import DerivedDipoleQuantities, derived_dipole_quantities
from .models import GaussianDipoleModel, StampDipoleModel
from .photometry import ForcedPhotometryResult, forced_psf_photometry
from .single import SingleSourceFitResult, fit_single_sources
from .solver import (
    BatchedLeastSquaresProblem,
    JacobianFunction,
    LMConfig,
    LMResult,
    LMStatus,
    NormalEquationsFunction,
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
    "DerivedDipoleQuantities",
    "DeviceDipoleFitResult",
    "DipoleFitResult",
    "DipoleFitUncertaintyReason",
    "FitMode",
    "FloatDType",
    "ForcedPhotometryResult",
    "GaussianDipoleModel",
    "JacobianFunction",
    "LMConfig",
    "LMResult",
    "LMStatus",
    "ModelName",
    "NormalEquationsFunction",
    "ResidualFunction",
    "ResolvedBackend",
    "SingleSourceFitResult",
    "StampEvaluation",
    "StampDipoleModel",
    "batched_levenberg_marquardt",
    "derived_dipole_quantities",
    "fit_dipoles",
    "fit_dipoles_device",
    "forced_psf_photometry",
    "fit_single_sources",
]
