# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Prepare a synthetic MPI/Dragon pipeline example without using a GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cuphoton.xscan.pipeline_benchmark.fixture import prepare_fixture
from cuphoton.xscan.pipeline_executor import PIPELINE_MANIFEST_SCHEMA


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New shared input directory",
    )
    parser.add_argument("--images", type=int, default=8)
    args = parser.parse_args()
    root = args.output.expanduser().resolve()
    config_path, items_path = prepare_fixture(
        root,
        images=args.images,
        image_size=256,
        candidates=9,
        stamp_size=17,
        seed=2026,
        device="cuda:0",
    )
    manifest = {
        "schema": PIPELINE_MANIFEST_SCHEMA,
        "configuration": json.loads(config_path.read_text(encoding="utf-8")),
        "items": json.loads(items_path.read_text(encoding="utf-8")),
    }
    path = root / "pipeline.json"
    path.write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(path)


if __name__ == "__main__":
    main()
