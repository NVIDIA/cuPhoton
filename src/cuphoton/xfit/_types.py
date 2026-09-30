# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared static and runtime type contracts for xFit."""

from __future__ import annotations

from typing import Any, Literal, Protocol, Self, TypeAlias

import numpy as np
import numpy.typing as npt

# Keep aliases eagerly expanded for public get_type_hints/get_args users.
FitMode: TypeAlias = Literal["difference", "split"]
BackendRequest: TypeAlias = Literal[
    "auto", "numpy", "cupy", "cutile", "numba-cuda-mlir"
]
ResolvedBackend: TypeAlias = Literal[
    "numpy", "cupy", "cutile", "numba-cuda-mlir"
]
ModelName: TypeAlias = Literal["gaussian", "stamp"]
StampEvaluation: TypeAlias = Literal[
    "bilinear", "bilinear-vignetted", "finite-volume"
]
ComputeDType: TypeAlias = Literal["input", "float32", "float64"]
FloatDType: TypeAlias = Literal["float32", "float64"]

FIT_MODES: frozenset[FitMode] = frozenset(("difference", "split"))
BACKEND_REQUESTS: frozenset[BackendRequest] = frozenset(
    ("auto", "numpy", "cupy", "cutile", "numba-cuda-mlir")
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

    @property
    def ndim(self) -> int: ...

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> np.dtype[Any]: ...

    def copy(self) -> Self: ...
    def reshape(self, shape: Any, /, *dimensions: int) -> Self: ...
    def transpose(self, *axes: Any) -> Self: ...
    def astype(self, dtype: npt.DTypeLike, *, copy: bool = True) -> Self: ...

    # NumPy and CuPy indexing can yield an array or a dtype-dependent scalar.
    def __getitem__(self, key: Any, /) -> Any: ...
    def __setitem__(self, key: Any, value: Any, /) -> None: ...
    def __add__(self, other: Any, /) -> Self: ...
    def __radd__(self, other: Any, /) -> Self: ...
    def __sub__(self, other: Any, /) -> Self: ...
    def __rsub__(self, other: Any, /) -> Self: ...
    def __mul__(self, other: Any, /) -> Self: ...
    def __rmul__(self, other: Any, /) -> Self: ...
    def __truediv__(self, other: Any, /) -> Self: ...
    def __rtruediv__(self, other: Any, /) -> Self: ...
    def __pow__(self, other: Any, /) -> Self: ...
    def __and__(self, other: Any, /) -> Self: ...
    def __or__(self, other: Any, /) -> Self: ...
    def __lt__(self, other: Any, /) -> Self: ...
    def __le__(self, other: Any, /) -> Self: ...
    def __gt__(self, other: Any, /) -> Self: ...
    def __ge__(self, other: Any, /) -> Self: ...
    def __neg__(self) -> Self: ...
    def __invert__(self) -> Self: ...


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
