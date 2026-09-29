# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared static and runtime type contracts for xRay."""

from __future__ import annotations

from typing import Literal, TypeAlias, get_args

FitDiagnosticsLevel: TypeAlias = Literal["none", "summary", "full"]
FIT_DIAGNOSTICS_LEVELS: tuple[FitDiagnosticsLevel, ...] = get_args(
    FitDiagnosticsLevel
)
