# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared infrastructure for cuPhoton Python tools."""

from cuphoton import __version__ as __version__
from cuphoton.core.runtime import runtime_metadata

__all__ = ["__version__", "runtime_metadata"]
