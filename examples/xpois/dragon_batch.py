# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run xPois image-pair batches with Dragon workers."""

from __future__ import annotations

import sys

from cuphoton.core.cli import run_component


def main(argv: list[str] | None = None) -> int:
    """Run the fixed Dragon executor without accepting executor overrides."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    if any(
        len(name := argument.split("=", 1)[0]) > 2
        and "--executor".startswith(name)
        for argument in arguments
    ):
        raise SystemExit(
            "dragon_batch.py fixes --executor=dragon; remove --executor"
        )
    return run_component(
        "xpois",
        ["fit-batch", "--executor", "dragon", *arguments],
    )


if __name__ == "__main__":
    raise SystemExit(main())
