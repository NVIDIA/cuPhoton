# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared static and runtime type contracts for xFit."""

from __future__ import annotations

from typing import Any, Literal, Protocol, TypeAlias

import numpy as np
import numpy.typing as npt

FitMode: TypeAlias = Literal["difference", "split"]
BackendRequest: TypeAlias = Literal["auto", "numpy", "cupy"]
ResolvedBackend: TypeAlias = Literal["numpy", "cupy"]
ModelName: TypeAlias = Literal["gaussian", "stamp"]
StampEvaluation: TypeAlias = Literal[
    "bilinear", "bilinear-vignetted", "finite-volume"
]
ComputeDType: TypeAlias = Literal["input", "float32", "float64"]
FloatDType: TypeAlias = Literal["float32", "float64"]

FIT_MODES: frozenset[FitMode] = frozenset(("difference", "split"))
BACKEND_REQUESTS: frozenset[BackendRequest] = frozenset(
    ("auto", "numpy", "cupy")
)
MODEL_NAMES: frozenset[ModelName] = frozenset(("gaussian", "stamp"))
STAMP_EVALUATIONS: frozenset[StampEvaluation] = frozenset(
    ("bilinear", "bilinear-vignetted", "finite-volume")
)
COMPUTE_DTYPES: frozenset[ComputeDType] = frozenset(
    ("input", "float32", "float64")
)


class BackendArray(Protocol):
    """Minimum structural contract used for NumPy and CuPy arrays."""

    ndim: int
    shape: tuple[int, ...]
    dtype: np.dtype[Any]

    def copy(self) -> BackendArray: ...


ArrayLike: TypeAlias = npt.ArrayLike | BackendArray


__all__ = [
    "ArrayLike",
    "BackendArray",
    "BackendRequest",
    "ComputeDType",
    "FitMode",
    "FloatDType",
    "ModelName",
    "ResolvedBackend",
    "StampEvaluation",
]
