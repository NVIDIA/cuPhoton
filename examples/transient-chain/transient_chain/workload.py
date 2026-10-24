# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Selections to executor workload, and the three ways to run it (local,
Dragon, MPI).
"""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .config import config_sha256, load_config
from .surveys import SurveyAdapter, make_adapter
from .worker import OPTIONS_SCHEMA, create_worker

MANIFEST_SCHEMA = "transient-chain.manifest/v2"
DRAGON_OPTIONS = ("max_workers", "worker_timeout_sec", "result_timeout_sec")
MPI_OPTIONS = ("rank_setup_timeout_sec",)


def load_selections(
    path: Path,
    adapter: SurveyAdapter,
    only: Sequence[str] | None = None,
    *,
    check_paths: bool = True,
) -> list[dict[str, Any]]:
    """Read ``{"selections": [...]}`` (or a bench ``{"shards": [...]}``),
    validate every entry, filter by ``--only``.

    Every missing key or file is reported with the entry index before any GPU
    work starts.
    """
    path = Path(path)
    try:
        d = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: not valid JSON ({exc})") from exc
    if isinstance(d, list):
        raise ValueError(
            f"{path}: top level must be an object with a 'selections' list, "
            f"not a bare list (see README.md)"
        )
    if not isinstance(d, Mapping):
        raise ValueError(
            f"{path}: top level must be an object with a 'selections' list"
        )
    sels = d.get("selections") or d.get("shards")
    if not sels:
        raise ValueError(
            f"{path}: 'selections' is missing or empty (see README.md)"
        )
    problems: list[str] = []
    for i, sel in enumerate(sels):
        if not isinstance(sel, Mapping):
            problems.append(f"selection {i} is not an object")
            continue
        try:
            adapter.validate_selection(sel, i)
        except ValueError as exc:
            problems.append(str(exc))
            continue
        if check_paths:
            missing = [
                p
                for p in adapter.input_paths(sel)
                if not Path(p).expanduser().exists()
            ]
            if missing:
                problems.append(
                    f"selection {i} ({adapter.identity(sel).item_id}): "
                    f"{len(missing)} input file(s) not found, first: "
                    f"{missing[0]}"
                )
    if problems:
        shown = "\n  ".join(problems[:10])
        more = (
            f"\n  ... and {len(problems) - 10} more"
            if len(problems) > 10
            else ""
        )
        raise ValueError(
            f"{path}: {len(problems)} invalid selection(s):\n  {shown}{more}"
        )
    if only:
        all_sels = sels
        sels = [
            s
            for s in sels
            if any(adapter.identity(s).matches(o) for o in only)
        ]
        if not sels:
            raise ValueError(
                f"--only {list(only)} matched nothing; tokens are CCD or "
                "CCD/EXPOSURE (for example "
                f"{adapter.identity(all_sels[0]).ccd_id})"
            )
    return [dict(s) for s in sels]


def build_items(
    selections: Sequence[Mapping[str, Any]], adapter: SurveyAdapter
):
    from cuphoton.core.bulk import WorkItem

    items = []
    for sel in selections:
        weight = sum(
            Path(p).expanduser().stat().st_size
            for p in adapter.input_paths(sel)
        )
        items.append(
            WorkItem(
                item_id=adapter.identity(sel).item_id,
                payload=adapter.payload(sel),
                weight_bytes=int(weight),
            )
        )
    return items


def options_payload(
    cfg: Mapping[str, Any], checkpoint_dir: Path | None
) -> dict[str, Any]:
    from cuphoton.core.bulk import json_mapping

    cfg = json_mapping(cfg, field="chain config")
    sha = None
    if checkpoint_dir is not None:
        ckpt = Path(checkpoint_dir).expanduser().resolve() / "checkpoint.pt"
        if not ckpt.is_file():
            raise FileNotFoundError(
                f"--checkpoint-dir {checkpoint_dir}: no checkpoint.pt inside"
            )
        sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
        cfg["classify"]["checkpoint_dir"] = str(ckpt.parent)
    else:
        cfg["classify"]["checkpoint_dir"] = None
    return {"schema": OPTIONS_SCHEMA, "config": cfg, "checkpoint_sha256": sha}


def finalize_round(
    round_dir: Path, records: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any]:
    """Coordinator side: concatenate the per-item catalogs of a round into one
    parquet.
    """
    import pandas as pd

    frames, stage = [], {}
    for rec in records:
        if rec.get("status") != "success":
            continue
        cat = Path(round_dir) / rec["run_dir"] / "catalog.parquet"
        if cat.exists():
            frames.append(pd.read_parquet(cat))
        for k, v in (rec.get("timings_sec") or {}).items():
            stage.setdefault(k, []).append(float(v))
    out = Path(round_dir) / "catalog.parquet"
    n = 0
    if frames:
        df = pd.concat(frames, ignore_index=True)
        df.to_parquet(out, index=False)
        n = int(len(df))
    failed = [
        rec["item_id"] for rec in records if rec.get("status") != "success"
    ]
    return {
        "catalog": str(out) if frames else None,
        "candidates": n,
        "items": len(frames),
        "failed": failed,
        "stage_seconds_median": {
            k: float(statistics.median(v)) for k, v in stage.items()
        },
    }


def prepare_workload(
    selections_path: Path,
    config_path: Path,
    checkpoint_dir: Path | None,
    *,
    only=None,
    overrides: Sequence[str] | None = None,
    rank: int = 0,
):
    """Every rank builds the same workload; the executor checks the digests
    agree (``rank`` is informational).
    """
    from cuphoton.core.execution import WorkloadSpec

    cfg = load_config(config_path, overrides)
    adapter = make_adapter(cfg["survey"])
    sels = load_selections(selections_path, adapter, only)
    items = build_items(sels, adapter)
    opts = options_payload(cfg, checkpoint_dir)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "selections": str(Path(selections_path).resolve()),
        "config": str(Path(config_path).resolve()),
        "overrides": list(overrides or ()),
        "config_sha256": config_sha256(cfg),
        "survey": adapter.name,
        "reader": cfg["io"]["reader"],
        "items": [i.item_id for i in items],
    }
    identity = {
        i.item_id: {
            "inputs": adapter.input_paths(s),
            "weight_bytes": i.weight_bytes,
        }
        for i, s in zip(items, sels, strict=True)
    }
    digest = hashlib.sha256(
        json.dumps(
            {"manifest": manifest, "options": opts, "identity": identity},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return WorkloadSpec(
        items=items,
        options_payload=opts,
        manifest_payload=manifest,
        input_identity_payload=identity,
        manifest_sha256=digest,
        backend="cupy",
        worker_factory=create_worker,
        finalize_round=finalize_round,
    )


def run_local(
    selections_path: Path,
    config_path: Path,
    checkpoint_dir: Path | None,
    output_dir: Path,
    *,
    only=None,
    overrides: Sequence[str] | None = None,
) -> dict[str, Any]:
    """One process, one GPU, no launcher: the same worker and the executor's
    own per-item record loop.
    """
    from cuphoton.core.bulk import atomic_write_json
    from cuphoton.core.execution import execute_worker_round

    spec = prepare_workload(
        selections_path,
        config_path,
        checkpoint_dir,
        only=only,
        overrides=overrides,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    worker = create_worker(spec.options_payload)
    setup = time.perf_counter() - t0
    run_id = output_dir.resolve().name or "local"
    result = execute_worker_round(
        worker,
        items=spec.items,
        run_id=run_id,
        run_dir=output_dir,
        manifest_sha256=spec.manifest_sha256,
        worker_id=0,
        backend="cupy",
        provenance={
            "executor": "local",
            "manifest": dict(spec.manifest_payload),
        },
    )
    worker.close()
    if result.get("record_write_errors"):
        raise RuntimeError(
            f"the executor could not write item records: "
            f"{result['record_write_errors']}"
        )
    records = [
        json.loads(
            (output_dir / "records" / f"{item.item_id}.json").read_text()
        )
        for item in spec.items
    ]
    for rec in records:
        print(
            f"{rec['item_id']}: {rec['status']} {rec.get('candidates', '')} "
            f"{rec['worker_seconds']:.1f}s",
            flush=True,
        )
    summary = {
        "executor": "local",
        "setup_sec": setup,
        "worker": result,
        "records": records,
        "finalize": finalize_round(output_dir, records),
        "wall_sec": time.perf_counter() - t0,
    }
    atomic_write_json(output_dir / "summary.json", summary)
    return summary


def run_distributed(
    executor: str,
    selections_path: Path,
    config_path: Path,
    checkpoint_dir: Path | None,
    output_dir: Path,
    *,
    only=None,
    overrides: Sequence[str] | None = None,
    run_id: str | None = None,
    warmup_rounds: int | None = None,
    measure_rounds: int | None = None,
    **executor_options,
):
    """Dragon or MPI through cuPhoton's shared ``run_workload``; returns None
    on non-root MPI ranks.

    ``executor_options`` are passed through unchanged (Dragon: max_workers,
    worker_timeout_sec,
    result_timeout_sec; MPI: rank_setup_timeout_sec). Benchmark rounds are
    optional and validated
    by the library (``measure_rounds`` must be at least 1 when given).
    """
    from cuphoton.core.benchmark import BenchmarkOptions
    from cuphoton.core.executors import run_workload

    allowed = {"dragon": DRAGON_OPTIONS, "mpi": MPI_OPTIONS}.get(executor)
    if allowed is None:
        raise ValueError("executor must be dragon or mpi")
    unknown = sorted(set(executor_options) - set(allowed))
    if unknown:
        raise ValueError(
            f"options {unknown} do not apply to the {executor} executor (it "
            f"takes {list(allowed)})"
        )
    benchmark = None
    if warmup_rounds is not None or measure_rounds is not None:
        benchmark = BenchmarkOptions(
            warmup_rounds=0 if warmup_rounds is None else warmup_rounds,
            measure_rounds=1 if measure_rounds is None else measure_rounds,
        )
    return run_workload(
        executor=executor,
        prepare_workload=lambda rank: prepare_workload(
            selections_path,
            config_path,
            checkpoint_dir,
            only=only,
            overrides=overrides,
            rank=rank,
        ),
        output_root=Path(output_dir),
        run_id=run_id,
        benchmark=benchmark,
        **executor_options,
    )
