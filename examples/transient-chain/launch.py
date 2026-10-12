#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Run the GPU-resident difference-imaging chain on a selections file.

Local, one GPU, no launcher (smoke tests and parity runs):
    python launch.py --executor local --selections sel.json --output-dir out
    [--checkpoint-dir CK] [--only R22_S00/2024121000367]

Dragon, one persistent worker per GPU (the script must be started by the
dragon launcher):
    dragon -s -t tcp -o tcp launch.py --executor dragon --selections sel.json
    --output-dir runs --name run1 [--max-workers 4]

MPI, one prebound GPU per rank (Open MPI; CUDA_VISIBLE_DEVICES lists the
allocation):
    mpirun -n 4 --map-by slot --bind-to none -x CUDA_VISIBLE_DEVICES
    cuphoton-openmpi-rank-exec -- \\
        python launch.py --executor mpi --selections sel.json --output-dir
        runs --name run1

The FITS reader (io.reader), the survey adapter (survey.name) and every stage
parameter live in
config.yaml; override any key from the command line with --set, for example
--set io.reader=astropy --set align.star_finder=daofind. See README.md for the
selections
schema, the config reference, and the catalog columns.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from transient_chain import workload  # noqa: E402

HERE = Path(__file__).resolve().parent


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--executor",
        choices=("local", "dragon", "mpi"),
        required=True,
        help="local: this process, one GPU; dragon: one worker per GPU "
        "under the dragon launcher; mpi: one worker per rank under mpirun",
    )
    ap.add_argument(
        "--selections",
        required=True,
        help='JSON file {"selections": [...]}; see README.md',
    )
    ap.add_argument(
        "--config",
        default=str(HERE / "config.yaml"),
        help="chain config YAML (default: %(default)s)",
    )
    ap.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="BLOCK.KEY=VALUE",
        help="override one config value (YAML syntax for the value); "
        "repeatable",
    )
    ap.add_argument(
        "--checkpoint-dir",
        default=None,
        help="directory holding an XScan checkpoint.pt; omit to stop after "
        "xFit (classifier columns are then NaN/false)",
    )
    ap.add_argument(
        "--output-dir",
        required=True,
        help="where catalogs and summaries go "
        "(per item: items/<CCD>-v<EXPOSURE>/)",
    )
    ap.add_argument(
        "--only",
        action="append",
        default=None,
        metavar="CCD[/EXPOSURE]",
        help="process only these detectors or detector-exposures; repeatable",
    )
    ap.add_argument(
        "--name",
        default=None,
        help="run id for dragon and mpi "
        "(default: the executor's timestamped id)",
    )
    dragon = ap.add_argument_group("dragon executor")
    dragon.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="workers to place (default: one per visible GPU)",
    )
    dragon.add_argument(
        "--worker-timeout-sec",
        type=float,
        default=None,
        help="per-worker wall limit in seconds "
        "(default: library default, 3600)",
    )
    dragon.add_argument(
        "--result-timeout-sec",
        type=float,
        default=None,
        help="seconds to wait for a worker receipt "
        "(default: library default, 60)",
    )
    mpi = ap.add_argument_group("mpi executor")
    mpi.add_argument(
        "--rank-setup-timeout-sec",
        type=float,
        default=None,
        help="rank startup limit in seconds (default: library default, 600)",
    )
    bench = ap.add_argument_group("benchmark layout (dragon and mpi)")
    bench.add_argument(
        "--warmup-rounds",
        type=int,
        default=None,
        help="unmeasured rounds before timing (outputs under rounds/)",
    )
    bench.add_argument(
        "--measure-rounds",
        type=int,
        default=None,
        help="measured rounds (at least 1)",
    )
    ap.epilog = (
        "Docs: the library's docs/distributed-execution.md and "
        "docs/dragon.md explain the dragon and mpirun forms; "
        "this example's README.md covers inputs, config, outputs and other "
        "surveys."
    )
    return ap


def _reject_inapplicable(a: argparse.Namespace) -> None:
    given = {"dragon": [], "mpi": [], "bench": []}
    for flag, group in (
        ("max_workers", "dragon"),
        ("worker_timeout_sec", "dragon"),
        ("result_timeout_sec", "dragon"),
        ("rank_setup_timeout_sec", "mpi"),
        ("warmup_rounds", "bench"),
        ("measure_rounds", "bench"),
    ):
        if getattr(a, flag) is not None:
            given[group].append("--" + flag.replace("_", "-"))
    bad = []
    if a.executor != "dragon":
        bad += given["dragon"]
    if a.executor != "mpi":
        bad += given["mpi"]
    if a.executor == "local":
        bad += given["bench"] + (["--name"] if a.name else [])
    if bad:
        raise SystemExit(
            f"error: {', '.join(bad)} do not apply to --executor {a.executor}"
        )


def main() -> int:
    a = _parser().parse_args()
    _reject_inapplicable(a)
    ck = Path(a.checkpoint_dir) if a.checkpoint_dir else None
    common = dict(only=a.only, overrides=a.set)
    if a.executor == "local":
        s = workload.run_local(
            Path(a.selections),
            Path(a.config),
            ck,
            Path(a.output_dir),
            **common,
        )
        fin = s["finalize"]
        print(
            json.dumps(
                {
                    "executor": "local",
                    "items": len(s["records"]),
                    "failed": fin["failed"],
                    "candidates": fin["candidates"],
                    "catalog": fin["catalog"],
                    "wall_sec": s["wall_sec"],
                    "summary": str(Path(a.output_dir) / "summary.json"),
                },
                indent=1,
            )
        )
        return 1 if fin["failed"] else 0
    opts = {
        k: getattr(a, k)
        for k in (
            workload.DRAGON_OPTIONS
            if a.executor == "dragon"
            else workload.MPI_OPTIONS
        )
        if getattr(a, k) is not None
    }
    result = workload.run_distributed(
        a.executor,
        Path(a.selections),
        Path(a.config),
        ck,
        Path(a.output_dir),
        run_id=a.name,
        warmup_rounds=a.warmup_rounds,
        measure_rounds=a.measure_rounds,
        **common,
        **opts,
    )
    if result is None:  # non-root MPI rank
        return 0
    out = result.to_dict()
    summary = result.summary if isinstance(result.summary, dict) else {}
    fin = (
        summary.get("result") or {}
    )  # the finalize_round payload the executor stores under "result"
    out.update(
        {
            "items": len(summary.get("records", [])) or None,
            "failed": fin.get("failed"),
            "candidates": fin.get("candidates"),
            "catalog": fin.get("catalog"),
        }
    )
    print(json.dumps(out, indent=1, default=str))
    return 0 if result.status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
