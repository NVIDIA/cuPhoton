# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build the optional CPU structured-root and double-double extension."""

from __future__ import annotations

import platform
from pathlib import Path

from setuptools import Extension


def get_extensions():
    """Use ordinary C++17; the CUDA kernel is a separate package resource."""
    if platform.system() != "Linux":
        raise RuntimeError("The native xRay extension supports Linux only")
    source = Path("src/cuphoton/xray/src")
    return [
        Extension(
            "cuphoton.xray._structured_ext",
            sources=[
                str(source / name)
                for name in (
                    "structured_ext.cpp",
                    "structured_roots.cpp",
                    "structured_refine.cpp",
                )
            ],
            depends=[str(source / "structured_api.h")],
            language="c++",
            extra_compile_args=[
                "-O3",
                "-std=c++17",
                "-fvisibility=hidden",
                "-pthread",
                "-fno-fast-math",
                "-ffp-contract=off",
            ],
            extra_link_args=["-pthread"],
        )
    ]
