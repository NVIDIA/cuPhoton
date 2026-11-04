# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Display and scientific-audit helpers for the Dragon imaging notebook."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _show(value):
    """Render notebook output, with plain text available outside IPython."""
    try:
        from IPython.display import display
    except ImportError:
        print(value)
    else:
        display(value)


def image_grid(
    arrays,
    titles,
    *,
    diverging=False,
    columns=3,
    path=None,
    shared_scale=False,
):
    """Use asinh for intensities and a linear scale for signed residuals."""
    arrays = [
        np.asarray(a.get() if hasattr(a, "get") else a, dtype=np.float64)
        for a in arrays
    ]
    if len(arrays) != len(titles) or not arrays:
        raise ValueError("Provide one title for every image")
    signed = (
        [diverging] * len(arrays)
        if isinstance(diverging, bool)
        else list(diverging)
    )
    if len(signed) != len(arrays):
        raise ValueError("Provide one diverging flag for every image")
    common = np.concatenate([a[np.isfinite(a)] for a in arrays])
    columns = min(columns, len(arrays))
    width, height, gutter = 360, 420, 12
    rows = (len(arrays) + columns - 1) // columns
    canvas = Image.new(
        "RGB", (columns * (width + gutter), rows * (height + gutter)), "white"
    )
    font = ImageFont.load_default(size=14)
    ascii_symbols = str.maketrans(
        {"−": "-", "→": "->", "←": "<-", "↔": "<->", "⇒": "=>"}
    )
    for index, (a, title, is_signed) in enumerate(
        zip(arrays, titles, signed, strict=True)
    ):
        finite = common if shared_scale else a[np.isfinite(a)]
        if finite.size == 0:
            raise ValueError(f"No finite pixels in {title}")
        if is_signed:
            limit = max(float(np.percentile(np.abs(finite), 99.5)), 1e-12)
            t = np.clip(np.nan_to_num(a) / limit, -1, 1)
            rgb = np.stack(
                [1 + np.minimum(t, 0), 1 - np.abs(t), 1 - np.maximum(t, 0)],
                axis=-1,
            )
            scale = f"linear: +/- {limit:.3g}"
        else:
            low, high = np.percentile(finite, (1, 99.7))
            t = np.clip(
                (np.nan_to_num(a) - low) / max(high - low, 1e-12), 0, 1
            )
            t = np.arcsinh(t * 12) / np.arcsinh(12)
            rgb = np.repeat(t[..., None], 3, axis=-1)
            scale = f"asinh: {low:.3g} to {high:.3g}"
        rgb[~np.isfinite(a)] = 0
        panel = Image.fromarray((rgb * 255).astype(np.uint8)).resize(
            (width, width), Image.Resampling.NEAREST
        )
        x = index % columns * (width + gutter)
        y = index // columns * (height + gutter)
        canvas.paste(panel, (x, y + 60))
        draw = ImageDraw.Draw(canvas)
        lines = [""]
        for word in str(title).translate(ascii_symbols).split():
            candidate = " ".join(filter(None, (lines[-1], word)))
            if font.getlength(candidate) > width - 8 and lines[-1]:
                lines.append(word)
            else:
                lines[-1] = candidate
        if len(lines) > 2:
            lines = lines[:2]
            while font.getlength(lines[-1] + "...") > width - 8:
                lines[-1] = lines[-1][:-1]
            lines[-1] += "..."
        draw.multiline_text(
            (x + 4, y + 4), "\n".join(lines), fill="black", font=font
        )
        draw.text((x + 4, y + 40), scale, fill="#555555", font=font)
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(path)
    _show(canvas)
    return canvas


def compare_arrays(expected, actual, *, rtol=1e-5, atol=1e-7, label=""):
    expected = np.asarray(
        expected.get() if hasattr(expected, "get") else expected
    )
    actual = np.asarray(actual.get() if hasattr(actual, "get") else actual)
    if expected.dtype.kind == "b" and actual.dtype.kind == "b":
        np.testing.assert_array_equal(actual, expected)
        error = 0.0
    else:
        np.testing.assert_allclose(
            actual, expected, rtol=rtol, atol=atol, equal_nan=True
        )
        finite = np.isfinite(expected) & np.isfinite(actual)
        error = float(
            np.max(np.abs(actual[finite] - expected[finite]), initial=0)
        )
    receipt = {
        "comparison": label,
        "shape": expected.shape,
        "max_abs_error": error,
        "rtol": rtol,
        "atol": atol,
    }
    return receipt


def summarize_run(path):
    """Show retained worker identities, completions, audits, and timing."""
    path = Path(path)
    if path.is_dir():
        path /= "summary.json"
    summary = load_json(path)
    if summary.get("status") != "success":
        raise RuntimeError(f"Run did not report success: {path}")
    print("status:", summary["status"], "executor:", summary.get("executor"))
    if summary.get("lifecycle_errors"):
        raise RuntimeError(summary["lifecycle_errors"])
    audit = summary.get("process_exit_audit", {})
    if audit and not audit.get("ok"):
        raise RuntimeError(audit)
    print("Worker process exit audit:", audit)
    ready = {
        message["worker_id"]: message["provenance"]
        for message in summary.get("ready_messages", [])
    }
    rounds = summary.get("benchmark", {}).get("rounds", [])
    round_paths = (
        [
            (path.parent / row["summary_path"], row["round_id"])
            for row in rounds
        ]
        if rounds
        else [(path, "single")]
    )
    identities, completions, audits = {}, [], []
    for round_path, declared_round_id in round_paths:
        payload = load_json(round_path)
        if payload.get("status") != "success":
            raise RuntimeError(f"Round failed: {round_path}")
        round_id = payload.get("round_id", declared_round_id)
        item_audit = payload["terminal_record_audit"]
        if not item_audit["ok"] or item_audit["failed_item_ids"]:
            raise RuntimeError(item_audit)
        shard_audit = payload.get("shard_result_audit")
        if shard_audit and not shard_audit["ok"]:
            raise RuntimeError(shard_audit)
        audits.append(
            {
                "round": round_id,
                "expected_items": item_audit["expected_count"],
                "completion_records": item_audit["terminal_record_count"],
                "terminal_audit_ok": item_audit["ok"],
            }
        )
        records = payload.get("records")
        if records is None:
            records = [
                load_json(record)
                for record in sorted(
                    (round_path.parent / "records").glob("*.json")
                )
            ]
        workers = payload.get("worker_results", payload.get("shard_results"))
        if workers is None:
            workers = [
                load_json(worker)
                for worker in sorted(
                    (round_path.parent / "workers").glob("*.json")
                )
            ]
        for worker in workers:
            worker_id, provenance = worker["worker_id"], worker["provenance"]
            identities.setdefault(worker_id, provenance)
            retained = provenance == ready.get(worker_id, provenance)
            if not retained:
                raise RuntimeError(f"Worker {worker_id} changed after READY")
            assigned = worker.get("assigned_item_ids")
            if assigned is None:
                assigned = [
                    record["item_id"]
                    for record in records
                    if record["worker_id"] == worker_id
                ]
            completions.append(
                {
                    "round": round_id,
                    "worker_id": worker_id,
                    "pid": provenance["pid"],
                    "same_as_ready": retained if worker_id in ready else None,
                    "success_count": worker["success_count"],
                    "failed_count": worker["failed_count"],
                    "item_count": len(assigned),
                    "item_examples": ", ".join(assigned[:3]),
                }
            )
    placement = pd.json_normalize(list(identities.values()))
    placement_columns = [
        "worker_id",
        "requested_host",
        "requested_gpu_id",
        "hostname",
        "pid",
        "cuda_visible_devices",
        "gpu.device_index",
        "gpu.uuid",
    ]
    placement = placement[[c for c in placement_columns if c in placement]]
    print("Requested and observed worker placement:")
    _show(placement.head(16))
    print("Worker assignments and completions (first 24 rows):")
    _show(pd.DataFrame(completions).head(24))
    _show(pd.DataFrame(audits))
    print(f"Terminal records from {round_id} (first 12 of {len(records)}):")
    record_columns = [
        "item_id",
        "worker_id",
        "status",
        "worker_seconds",
        "completed_at_utc",
    ]
    record_table = pd.DataFrame(records)
    _show(
        record_table[[c for c in record_columns if c in record_table]].head(
            12
        )
    )
    if rounds:
        timing_columns = [
            "round_id",
            "phase",
            "batch_wall_sec",
            "worker_wall_max_sec",
            "finalization_sec",
        ]
        timing_table = pd.DataFrame(rounds)
        _show(timing_table[[c for c in timing_columns if c in timing_table]])
    timings = summary.get("coordinator_timings_sec", {})
    if timings:
        _show(pd.DataFrame(timings.items(), columns=["phase", "seconds"]))
    return placement
