# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Write a synthetic-mode review through the shared XRay command.

    uv run python examples/xray_lp_modes_review.py --output review.html
    uv run python examples/xray_lp_modes_review.py --distortion chirp:0.05

Python callers can use
``cuphoton.xray.mode_refinement_viz.write_modes_review``.
Requires the ``viz`` extra.
"""

from __future__ import annotations

import sys

from cuphoton.core.cli import run_component


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    return run_component("xray", ["linear-prediction-modes-review", *args])


if __name__ == "__main__":
    sys.exit(main())
