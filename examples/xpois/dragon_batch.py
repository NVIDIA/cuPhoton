# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run XPOIS image-pair batches with Dragon workers."""

from __future__ import annotations

import sys

from cuphoton.core.cli import run_component

if __name__ == "__main__":
    raise SystemExit(
        run_component("xpois", ["fit-batch-dragon", *sys.argv[1:]])
    )
