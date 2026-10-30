# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared lazy CUDA translation-unit cache."""

from functools import cache
from typing import Any


@cache
def _raw_module(source: str) -> Any:
    """Compile each CUDA translation unit once per process.

    Several public operations select different entry points from the same
    source. A ``RawKernel`` per entry point caused CuPy to load
    duplicate CUDA modules even though their generated code was identical.
    """

    import cupy

    return cupy.RawModule(
        code=source,
        options=("--std=c++11",),
        backend="nvrtc",
    )


def _raw_function(source: str, name: str) -> Any:
    return _raw_module(source).get_function(name)
