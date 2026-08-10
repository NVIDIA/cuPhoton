# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

GPU_FIRST_POLICY = """\
XRay GPU-capable paths prefer NVIDIA GPU execution through CuPy. Supported
CPU paths use NumPy for portable runs, diagnostics, and correctness checks.
Optional distributed-array adapters are used only by workflows that select
them explicitly.
"""


def gpu_first_policy():
    return GPU_FIRST_POLICY.strip()
