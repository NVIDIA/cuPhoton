# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""GPU-accelerated astronomy and imaging tools from NVIDIA."""

try:
    from ._version import __version__
except ModuleNotFoundError:
    # An unbuilt source checkout has no resolved distribution version yet.
    __version__ = "0.0.0.dev0"

__all__ = ["__version__"]
