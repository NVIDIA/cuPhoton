# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Internal buffered-reader compatibility planner.

Physical addresses, logical clocks, FFT admission and payload validity remain
independent. The engine consumes only the file-adapter entry point.
"""

from .adapter import build_returned_window_replay_plans

__all__ = ["build_returned_window_replay_plans"]
