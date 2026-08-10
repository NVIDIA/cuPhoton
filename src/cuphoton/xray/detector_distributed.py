# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from cuphoton import __version__ as CUPHOTON_VERSION

from .detector_artifacts import (
    DETECTOR_ARTIFACT_MANIFEST_VERSION,
    detector_artifact_complete,
    detector_artifact_input_identity,
    detector_artifact_resume_identity,
    detector_normalization_identity,
    merge_detector_artifact_shards,
)
from .detector_mask import format_y_ranges, parse_y_ranges
from .hdf5 import probe_hdf5_file

_DETECTOR_OPTION_DEFAULTS: dict[str, Any] = {
    "exclude_y": (),
    "drop_leading": 1,
    "chunk_frames": 16,
    "zero_offset": 0.0,
    "zero_offset_index": None,
    "fit_trailing_drop": 1,
    "integrate": 3,
    "components": 30,
    "roots_backend": "eigvals",
    "savgol_window": 5,
    "savgol_polyorder": 3,
    "amp_threshold": 1.6,
    "max_fit_failures": 0,
    "hdf5_reader": "h5py",
    "hdf5_reader_workers": 2,
    "max_tiles": None,
}


@dataclass(frozen=True)
class DetectorArtifactShard:
    index: int
    count: int
    roi_lower: tuple[int, int]
    roi_dim: tuple[int, int]
    output_dir: Path
    log_path: Path

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["roi_lower"] = list(self.roi_lower)
        payload["roi_dim"] = list(self.roi_dim)
        payload["output_dir"] = str(self.output_dir)
        payload["log_path"] = str(self.log_path)
        return payload


@dataclass(frozen=True)
class DetectorArtifactDistributedResult:
    executor: str
    output_dir: Path
    plan_path: Path | None
    shard_count: int
    submitted: bool
    merged: bool
    returncode: int | None
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.payload)
        payload.update(
            {
                "executor": self.executor,
                "output_dir": str(self.output_dir),
                "plan_path": (
                    None if self.plan_path is None else str(self.plan_path)
                ),
                "shard_count": int(self.shard_count),
                "submitted": bool(self.submitted),
                "merged": bool(self.merged),
                "returncode": self.returncode,
            }
        )
        return payload


def build_detector_artifact_distributed_plan(
    *,
    h5dir: Path | str,
    fon: str,
    foff: str,
    output_dir: Path | str,
    roi_lower: tuple[int, int] = (0, 0),
    roi_dim: tuple[int, int] | None = None,
    tile_shape: tuple[int, int] = (16, 16),
    shard_count: int | None = None,
    shard_width: int | None = None,
    gpus: int = 1,
    gpus_per_node: int | None = None,
    nodes: int | None = None,
    run_label: str | None = None,
    detector_options: dict[str, Any] | None = None,
    normalization_cache: Path | str | None = None,
    python_executable: str | None = None,
    work_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Build a serializable plan for an x-sharded detector artifact run."""

    _require_positive("tile width", tile_shape[0])
    _require_positive("tile height", tile_shape[1])
    _require_positive("gpus", gpus)
    if gpus_per_node is not None:
        _require_positive("gpus_per_node", gpus_per_node)
    if nodes is not None:
        _require_positive("nodes", nodes)
    if shard_count is not None and shard_width is not None:
        raise ValueError("provide only one of shard_count or shard_width")
    if shard_count is not None:
        _require_positive("shard_count", shard_count)
    if shard_width is not None:
        _require_positive("shard_width", shard_width)
        if shard_width % int(tile_shape[0]) != 0:
            raise ValueError("shard_width must be a multiple of tile width")

    h5dir_path = Path(h5dir)
    output_path = Path(output_dir)
    input_spec = _read_detector_input_spec(
        h5dir=h5dir_path,
        fon=fon,
        foff=foff,
    )
    detector_height, detector_width = input_spec["detector_shape"]
    roi_x, roi_y, roi_width, roi_height = _resolve_roi(
        roi_lower=roi_lower,
        roi_dim=roi_dim,
        detector_width=detector_width,
        detector_height=detector_height,
    )
    if shard_count is None and shard_width is None:
        shard_count = min(int(gpus), roi_width)
    shard_ranges = _plan_x_shards(
        roi_x=roi_x,
        roi_width=roi_width,
        tile_width=int(tile_shape[0]),
        shard_count=shard_count,
        shard_width=shard_width,
    )

    label = _run_label(run_label, output_path)
    logs_dir = output_path / "_logs" / label
    shards_root = output_path / "_shards" / label
    shard_total = len(shard_ranges)
    shards = tuple(
        DetectorArtifactShard(
            index=index,
            count=shard_total,
            roi_lower=(x0, roi_y),
            roi_dim=(x1 - x0, roi_height),
            output_dir=shards_root / f"shard-{index:04d}-x{x0}-{x1}",
            log_path=logs_dir / f"shard-{index:04d}.log",
        )
        for index, (x0, x1) in enumerate(shard_ranges)
    )
    options = dict(_DETECTOR_OPTION_DEFAULTS)
    options.update(detector_options or {})
    options["tile_shape"] = tuple(tile_shape)
    on_path = _resolve_path(h5dir_path, fon)
    off_path = _resolve_path(h5dir_path, foff)
    input_identity = detector_artifact_input_identity(on_path, off_path)
    normalization_identity = detector_normalization_identity(
        normalization_cache
    )
    shard_payloads = []
    for shard in shards:
        payload = shard.to_dict()
        request_manifest = _request_manifest_for_shard(
            shard=shard,
            global_roi_lower=(roi_x, roi_y),
            global_roi_dim=(roi_width, roi_height),
            input_spec=input_spec,
            input_identity=input_identity,
            normalization_identity=normalization_identity,
            detector_options=options,
        )
        payload["resume_identity"] = detector_artifact_resume_identity(
            request_manifest
        )
        shard_payloads.append(payload)
    commands = [
        _worker_command(
            python_executable=python_executable or sys.executable,
            h5dir=h5dir_path,
            fon=fon,
            foff=foff,
            shard=shard,
            global_roi_lower=(roi_x, roi_y),
            global_roi_dim=(roi_width, roi_height),
            detector_options=options,
            normalization_cache=normalization_cache,
        )
        for shard in shards
    ]
    concurrency = int(gpus)
    if gpus_per_node is not None and nodes is not None:
        concurrency = min(shard_total, int(gpus_per_node) * int(nodes))

    return {
        "kind": "xray-detector-artifact-distributed-plan",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_label": label,
        "work_dir": str(Path.cwd() if work_dir is None else Path(work_dir)),
        "h5dir": str(h5dir_path),
        "fon": fon,
        "foff": foff,
        "output_dir": str(output_path),
        "detector_shape": [int(detector_height), int(detector_width)],
        "global_roi_lower": [int(roi_x), int(roi_y)],
        "global_roi_dim": [int(roi_width), int(roi_height)],
        "tile_shape": [int(tile_shape[0]), int(tile_shape[1])],
        "normalization_cache": (
            None if normalization_cache is None else str(normalization_cache)
        ),
        "manifest_schema_version": DETECTOR_ARTIFACT_MANIFEST_VERSION,
        "package_version": CUPHOTON_VERSION,
        "dtype": "float64",
        "input_identity": input_identity,
        "normalization_identity": normalization_identity,
        "schema": input_spec["schema"],
        "image_dataset": input_spec["image_dataset"],
        "delay_dataset": input_spec["delay_dataset"],
        "normalization_dataset": input_spec["normalization_dataset"],
        "gpus": int(gpus),
        "gpus_per_node": (
            None if gpus_per_node is None else int(gpus_per_node)
        ),
        "nodes": None if nodes is None else int(nodes),
        "concurrency": int(concurrency),
        "shards": shard_payloads,
        "commands": commands,
        "detector_options": _json_safe_options(options),
        "plan_identity": _plan_identity(shard_payloads),
    }


def run_detector_artifact_distributed(
    *,
    plan: dict[str, Any],
    executor: str,
    submit: bool,
    merge: bool,
    resume: bool,
    slurm_options: dict[str, Any] | None = None,
) -> DetectorArtifactDistributedResult:
    """Execute or render a distributed detector artifact plan."""

    if executor not in ("dry-run", "local", "slurm"):
        raise ValueError("executor must be dry-run, local, or slurm")
    if executor == "dry-run":
        payload = dict(plan)
        payload["executor"] = "dry-run"
        payload["slurm_script"] = render_slurm_script(
            plan,
            slurm_options or {},
        )
        payload["local_script"] = render_local_shell_script(
            plan,
            merge=merge,
            resume=resume,
        )
        return DetectorArtifactDistributedResult(
            executor=executor,
            output_dir=Path(plan["output_dir"]),
            plan_path=None,
            shard_count=len(plan["shards"]),
            submitted=False,
            merged=False,
            returncode=0,
            payload=payload,
        )
    if executor == "local":
        payload = _run_local_plan(
            plan,
            merge=merge,
            resume=resume,
            submit=submit,
        )
        return DetectorArtifactDistributedResult(
            executor=executor,
            output_dir=Path(plan["output_dir"]),
            plan_path=(
                None
                if payload.get("plan_path") is None
                else Path(payload["plan_path"])
            ),
            shard_count=len(plan["shards"]),
            submitted=submit,
            merged=bool(payload.get("merged", False)),
            returncode=int(payload.get("returncode", 0)),
            payload=payload,
        )
    if executor == "slurm":
        payload = _write_or_submit_slurm_plan(
            plan,
            options=slurm_options or {},
            merge=merge,
            submit=submit,
            resume=resume,
        )
        return DetectorArtifactDistributedResult(
            executor=executor,
            output_dir=Path(plan["output_dir"]),
            plan_path=Path(payload["plan_path"]),
            shard_count=len(plan["shards"]),
            submitted=submit,
            merged=False,
            returncode=payload.get("returncode"),
            payload=payload,
        )
    raise AssertionError(f"unhandled executor {executor!r}")


def render_slurm_script(
    plan: dict[str, Any],
    options: dict[str, Any],
    *,
    merge: bool = False,
    resume: bool = False,
) -> str:
    output = Path(plan["output_dir"])
    log_dir = output / "_logs" / str(plan["run_label"])
    throttle = max(1, int(plan.get("concurrency", 1)))
    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={options.get('job_name') or plan['run_label']}",
        f"#SBATCH --output={log_dir}/slurm-%A_%a.out",
        f"#SBATCH --array=0-{len(plan['shards']) - 1}%{throttle}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
    ]
    for key, directive in (
        ("partition", "--partition"),
        ("account", "--account"),
        ("time", "--time"),
        ("constraint", "--constraint"),
        ("qos", "--qos"),
        ("gres", "--gres"),
        ("cpus_per_task", "--cpus-per-task"),
    ):
        value = options.get(key)
        if value:
            lines.append(f"#SBATCH {directive}={value}")
    lines.extend(
        [
            "set -euo pipefail",
            f"cd {shlex.quote(str(plan['work_dir']))}",
            _pythonpath_export(Path(plan["work_dir"])),
            f"mkdir -p {shlex.quote(str(log_dir))}",
            "if [[ -v CUDA_VISIBLE_DEVICES ]] && "
            '[[ -z "${CUDA_VISIBLE_DEVICES//[[:space:]]/}" '
            '|| "${CUDA_VISIBLE_DEVICES}" == "-1" ]]; then',
            '  echo "no CUDA devices are visible" >&2; exit 2',
            "fi",
            'case "${SLURM_ARRAY_TASK_ID:?}" in',
        ]
    )
    for index, command in enumerate(plan["commands"]):
        command_text = shlex.join(command)
        if resume:
            out_dir = plan["shards"][index]["output_dir"]
            resume_identity = (
                plan["shards"][index].get("resume_identity") or ""
            )
            command_text = (
                f"if python - <<'PY'\n"
                "from cuphoton.xray.detector_artifacts import "
                "detector_artifact_complete\n"
                "raise SystemExit(\n"
                "    0 if detector_artifact_complete(\n"
                f"        {out_dir!r},\n"
                f"        expected_resume_identity={resume_identity!r},\n"
                "    ) else 1\n"
                ")\n"
                f"PY\nthen echo shard {index} complete; "
                f"else {command_text}; fi"
            )
        lines.append(f"  {index}) {command_text} ;;")
    lines.extend(['  *) echo "invalid shard index" >&2; exit 2 ;;', "esac"])
    if merge:
        lines.append(
            "# Merge is submitted as a dependent job by the launcher."
        )
    return "\n".join(lines) + "\n"


def render_local_shell_script(
    plan: dict[str, Any],
    *,
    merge: bool,
    resume: bool,
) -> str:
    output = Path(plan["output_dir"])
    log_dir = output / "_logs" / str(plan["run_label"])
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(plan['work_dir']))}",
        _pythonpath_export(Path(plan["work_dir"])),
        f"mkdir -p {shlex.quote(str(log_dir))}",
        f"REQUESTED_GPUS={max(1, int(plan.get('gpus', 1)))}",
        "if [[ -v CUDA_VISIBLE_DEVICES ]]; then",
        '  if [[ -z "${CUDA_VISIBLE_DEVICES//[[:space:]]/}" '
        '|| "${CUDA_VISIBLE_DEVICES}" == "-1" ]]; then',
        '    echo "no CUDA devices are visible" >&2; exit 2',
        "  fi",
        "  IFS=',' read -r -a CUDA_DEVICE_TOKENS "
        '<<< "${CUDA_VISIBLE_DEVICES}"',
        "else",
        "  CUDA_DEVICE_TOKENS=($(seq 0 $((REQUESTED_GPUS - 1))))",
        "fi",
        "if [[ ${#CUDA_DEVICE_TOKENS[@]} -lt ${REQUESTED_GPUS} ]]; then",
        '  echo "requested ${REQUESTED_GPUS} GPUs but only '
        '${#CUDA_DEVICE_TOKENS[@]} are visible" >&2',
        "  exit 2",
        "fi",
        "MAX_JOBS=${REQUESTED_GPUS}",
        "running=0",
    ]
    for index, command in enumerate(plan["commands"]):
        shard = plan["shards"][index]
        device = index % max(1, int(plan.get("gpus", 1)))
        prefix = f'CUDA_VISIBLE_DEVICES="${{CUDA_DEVICE_TOKENS[{device}]}}"'
        command_text = f"{prefix} {shlex.join(command)}"
        log_path = shlex.quote(shard["log_path"])
        if resume:
            resume_identity = shard.get("resume_identity") or ""
            command_text = (
                "if python - <<'PY'\n"
                "from cuphoton.xray.detector_artifacts import "
                "detector_artifact_complete\n"
                "raise SystemExit(\n"
                "    0\n"
                "    if detector_artifact_complete("
                f"{shard['output_dir']!r}, "
                "expected_resume_identity="
                f"{resume_identity!r})\n"
                "    else 1\n"
                ")\n"
                "PY\n"
                f"then echo shard {index} complete; "
                f"else {command_text}; fi"
            )
        lines.extend(
            [
                f"({command_text}) > {log_path} 2>&1 &",
                "running=$((running + 1))",
                "if [[ ${running} -ge ${MAX_JOBS} ]]; then",
                "  wait -n",
                "  running=$((running - 1))",
                "fi",
            ]
        )
    lines.append("wait")
    if merge:
        lines.append(shlex.join(_merge_command(plan)))
    return "\n".join(lines) + "\n"


def merge_shards_from_plan(
    *,
    plan: dict[str, Any],
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    result = merge_detector_artifact_shards(
        shard_dirs=tuple(
            Path(shard["output_dir"]) for shard in plan["shards"]
        ),
        output_dir=Path(
            plan["output_dir"] if output_dir is None else output_dir
        ),
    )
    return result.to_dict()


def _run_local_plan(
    plan: dict[str, Any],
    *,
    merge: bool,
    resume: bool,
    submit: bool,
) -> dict[str, Any]:
    plan_path = _write_plan_file(plan)
    if not submit:
        return {
            "kind": "xray-detector-artifact-local-plan",
            "plan_path": str(plan_path),
            "submitted": False,
            "commands": [shlex.join(command) for command in plan["commands"]],
            "returncode": 0,
        }
    log_dir = Path(plan["output_dir"]) / "_logs" / str(plan["run_label"])
    log_dir.mkdir(parents=True, exist_ok=True)
    gpus = max(1, int(plan.get("gpus", 1)))
    device_tokens = _visible_cuda_device_tokens(gpus)
    with ThreadPoolExecutor(max_workers=gpus) as executor:
        futures = [
            executor.submit(
                _run_shard_command,
                command,
                shard,
                device=device_tokens[index % gpus],
                work_dir=Path(plan["work_dir"]),
                resume=resume,
            )
            for index, (command, shard) in enumerate(
                zip(plan["commands"], plan["shards"], strict=True)
            )
        ]
        results = [future.result() for future in futures]
    failures = [result for result in results if result["returncode"] != 0]
    merged = False
    merge_payload = None
    if failures:
        returncode = int(failures[0]["returncode"])
    else:
        returncode = 0
        if merge:
            merge_payload = merge_shards_from_plan(plan=plan)
            merged = True
    return {
        "kind": "xray-detector-artifact-local-result",
        "plan_path": str(plan_path),
        "submitted": True,
        "shards": results,
        "failures": failures,
        "merged": merged,
        "merge": merge_payload,
        "returncode": int(returncode),
    }


def _run_shard_command(
    command: list[str],
    shard: dict[str, Any],
    *,
    device: str,
    work_dir: Path,
    resume: bool,
) -> dict[str, Any]:
    output_dir = Path(shard["output_dir"])
    resume_identity = shard.get("resume_identity")
    if (
        resume
        and resume_identity is not None
        and detector_artifact_complete(
            output_dir,
            expected_resume_identity=str(resume_identity),
        )
    ):
        return {
            "index": int(shard["index"]),
            "output_dir": str(output_dir),
            "log_path": str(shard["log_path"]),
            "returncode": 0,
            "skipped": True,
        }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(shard["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device
    _prepend_pythonpath(env, work_dir)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            command,
            cwd=work_dir,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return {
        "index": int(shard["index"]),
        "output_dir": str(output_dir),
        "log_path": str(log_path),
        "returncode": int(proc.returncode),
        "skipped": False,
        "device": device,
    }


def _write_or_submit_slurm_plan(
    plan: dict[str, Any],
    *,
    options: dict[str, Any],
    merge: bool,
    submit: bool,
    resume: bool,
) -> dict[str, Any]:
    plan_path = _write_plan_file(plan)
    run_dir = plan_path.parent
    script_path = run_dir / f"{plan['run_label']}.slurm.sh"
    script_path.write_text(
        render_slurm_script(plan, options, merge=merge, resume=resume),
        encoding="utf-8",
    )
    script_path.chmod(0o755)
    merge_script_path = None
    if merge:
        merge_script_path = run_dir / f"{plan['run_label']}.merge.sh"
        merge_script_path.write_text(
            _render_merge_script(plan),
            encoding="utf-8",
        )
        merge_script_path.chmod(0o755)
    payload = {
        "kind": "xray-detector-artifact-slurm-plan",
        "plan_path": str(plan_path),
        "script_path": str(script_path),
        "merge_script_path": (
            None if merge_script_path is None else str(merge_script_path)
        ),
        "submitted": False,
        "submit_command": shlex.join(["sbatch", str(script_path)]),
    }
    if not submit:
        payload["returncode"] = 0
        return payload
    submit_result = subprocess.run(
        ["sbatch", str(script_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    payload["submitted"] = submit_result.returncode == 0
    payload["returncode"] = int(submit_result.returncode)
    payload["stdout"] = submit_result.stdout
    payload["stderr"] = submit_result.stderr
    job_id = _parse_sbatch_job_id(submit_result.stdout)
    payload["job_id"] = job_id
    if submit_result.returncode == 0 and merge_script_path is not None:
        merge_command = ["sbatch"]
        if job_id is not None:
            merge_command.append(f"--dependency=afterok:{job_id}")
        merge_command.append(str(merge_script_path))
        merge_result = subprocess.run(
            merge_command,
            capture_output=True,
            text=True,
            check=False,
        )
        payload["merge_submit_command"] = shlex.join(merge_command)
        payload["merge_stdout"] = merge_result.stdout
        payload["merge_stderr"] = merge_result.stderr
        payload["merge_returncode"] = int(merge_result.returncode)
        payload["merge_job_id"] = _parse_sbatch_job_id(merge_result.stdout)
    return payload


def _write_plan_file(plan: dict[str, Any]) -> Path:
    run_dir = (
        Path(plan["output_dir"]) / "_distributed" / str(plan["run_label"])
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    plan_path = run_dir / "plan.json"
    plan_path.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return plan_path


def _render_merge_script(plan: dict[str, Any]) -> str:
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"cd {shlex.quote(str(plan['work_dir']))}",
            _pythonpath_export(Path(plan["work_dir"])),
            shlex.join(_merge_command(plan)),
            "",
        ]
    )


def _merge_command(plan: dict[str, Any]) -> list[str]:
    plan_path = (
        Path(plan["output_dir"])
        / "_distributed"
        / str(plan["run_label"])
        / "plan.json"
    )
    return [
        str(sys.executable),
        "-m",
        "cuphoton",
        "xray",
        "detector-artifact-merge",
        "--shards-manifest",
        str(plan_path),
        "--output-dir",
        str(plan["output_dir"]),
        "--json",
    ]


def _request_manifest_for_shard(
    *,
    shard: DetectorArtifactShard,
    global_roi_lower: tuple[int, int],
    global_roi_dim: tuple[int, int],
    input_spec: dict[str, Any],
    input_identity: dict[str, Any],
    normalization_identity: dict[str, Any],
    detector_options: dict[str, Any],
) -> dict[str, Any]:
    options = detector_options
    return {
        "kind": "xray-detector-artifacts",
        "manifest_schema_version": DETECTOR_ARTIFACT_MANIFEST_VERSION,
        "package_version": CUPHOTON_VERSION,
        "backend": "cupy",
        "dtype": "float64",
        "input_identity": input_identity,
        "normalization_identity": normalization_identity,
        "schema": input_spec["schema"],
        "image_dataset": input_spec["image_dataset"],
        "delay_dataset": input_spec["delay_dataset"],
        "normalization_dataset": input_spec["normalization_dataset"],
        "drop_leading": int(options["drop_leading"]),
        "chunk_frames": int(options["chunk_frames"]),
        "fit_trailing_drop": int(options["fit_trailing_drop"]),
        "zero_offset": float(options["zero_offset"]),
        "requested_zero_offset_index": (
            None
            if options["zero_offset_index"] is None
            else int(options["zero_offset_index"])
        ),
        "detector_shape": [
            int(input_spec["detector_shape"][0]),
            int(input_spec["detector_shape"][1]),
        ],
        "roi_lower": [int(value) for value in shard.roi_lower],
        "roi_dim": [int(value) for value in shard.roi_dim],
        "tile_shape": [
            int(options["tile_shape"][0]),
            int(options["tile_shape"][1]),
        ],
        "exclude_y": format_y_ranges(parse_y_ranges(options["exclude_y"])),
        "integrate_pixels": int(options["integrate"]),
        "components": int(options["components"]),
        "roots_backend": str(options["roots_backend"]),
        "savgol_window": int(options["savgol_window"]),
        "savgol_polyorder": int(options["savgol_polyorder"]),
        "amp_threshold": float(options["amp_threshold"]),
        "max_fit_failures": int(options["max_fit_failures"]),
        "hdf5_reader": str(options["hdf5_reader"]),
        "hdf5_reader_workers": int(options["hdf5_reader_workers"]),
        "max_tiles": (
            None
            if options["max_tiles"] is None
            else int(options["max_tiles"])
        ),
        "shard": {
            "index": int(shard.index),
            "count": int(shard.count),
            "global_roi_lower": [
                int(global_roi_lower[0]),
                int(global_roi_lower[1]),
            ],
            "global_roi_dim": [
                int(global_roi_dim[0]),
                int(global_roi_dim[1]),
            ],
            "x_range": [
                int(shard.roi_lower[0]),
                int(shard.roi_lower[0] + shard.roi_dim[0]),
            ],
        },
    }


def _plan_identity(shards: list[dict[str, Any]]) -> str:
    payload = {
        "kind": "xray-detector-artifact-distributed-plan",
        "resume_identities": [item["resume_identity"] for item in shards],
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _worker_command(
    *,
    python_executable: str,
    h5dir: Path,
    fon: str,
    foff: str,
    shard: DetectorArtifactShard,
    global_roi_lower: tuple[int, int],
    global_roi_dim: tuple[int, int],
    detector_options: dict[str, Any],
    normalization_cache: Path | str | None,
) -> list[str]:
    cmd = [
        python_executable,
        "-m",
        "cuphoton",
        "xray",
        "detector-artifacts",
        "--h5dir",
        str(h5dir),
        "--fon",
        fon,
        "--foff",
        foff,
        "--output-dir",
        str(shard.output_dir),
        "--roi-lower",
        str(shard.roi_lower[0]),
        str(shard.roi_lower[1]),
        "--roi-dim",
        str(shard.roi_dim[0]),
        str(shard.roi_dim[1]),
        "--shard-index",
        str(shard.index),
        "--shard-count",
        str(shard.count),
        "--global-roi-lower",
        str(global_roi_lower[0]),
        str(global_roi_lower[1]),
        "--global-roi-dim",
        str(global_roi_dim[0]),
        str(global_roi_dim[1]),
    ]
    _append_detector_worker_options(cmd, detector_options)
    if normalization_cache is not None:
        cmd.extend(["--normalization-cache", str(normalization_cache)])
    cmd.append("--json")
    return cmd


def _append_detector_worker_options(
    cmd: list[str],
    options: dict[str, Any],
) -> None:
    tuple_options = {
        "tile_shape": "--tile-shape",
    }
    scalar_options = {
        "drop_leading": "--drop-leading",
        "chunk_frames": "--chunk-frames",
        "zero_offset": "--zero-offset",
        "zero_offset_index": "--zero-offset-index",
        "fit_trailing_drop": "--fit-trailing-drop",
        "integrate": "--integrate",
        "components": "--components",
        "roots_backend": "--roots-backend",
        "savgol_window": "--savgol-window",
        "savgol_polyorder": "--savgol-polyorder",
        "amp_threshold": "--amp-threshold",
        "max_fit_failures": "--max-fit-failures",
        "hdf5_reader": "--hdf5-reader",
        "hdf5_reader_workers": "--hdf5-reader-workers",
        "max_tiles": "--max-tiles",
    }
    for key, flag in tuple_options.items():
        value = options.get(key)
        if value is not None:
            cmd.extend([flag, str(value[0]), str(value[1])])
    for key, flag in scalar_options.items():
        value = options.get(key)
        if value is not None:
            cmd.extend([flag, str(value)])
    for item in options.get("exclude_y", ()) or ():
        cmd.extend(["--exclude-y", str(item)])


def _read_detector_input_spec(
    *,
    h5dir: Path,
    fon: str,
    foff: str,
) -> dict[str, Any]:
    import h5py

    on_path = _resolve_path(h5dir, fon)
    off_path = _resolve_path(h5dir, foff)
    on_probe = probe_hdf5_file(on_path)
    off_probe = probe_hdf5_file(off_path)
    if on_probe.load_plan is None or off_probe.load_plan is None:
        raise ValueError("unsupported HDF5 schema or IPM layout")
    if on_probe.load_plan.schema != off_probe.load_plan.schema:
        raise ValueError("on/off HDF5 schemas do not match")
    dataset = on_probe.load_plan.image_dataset
    with h5py.File(on_path, "r") as on_h5, h5py.File(off_path, "r") as off_h5:
        on_shape = tuple(int(value) for value in on_h5[dataset].shape)
        off_shape = tuple(int(value) for value in off_h5[dataset].shape)
    if len(on_shape) != 3 or on_shape != off_shape:
        raise ValueError("on/off image shapes do not match")
    plan = on_probe.load_plan
    return {
        "detector_shape": (int(on_shape[1]), int(on_shape[2])),
        "schema": plan.schema,
        "image_dataset": plan.image_dataset,
        "delay_dataset": plan.delay_dataset,
        "normalization_dataset": plan.ipm_pair.normalization,
    }


def _resolve_roi(
    *,
    roi_lower: tuple[int, int],
    roi_dim: tuple[int, int] | None,
    detector_width: int,
    detector_height: int,
) -> tuple[int, int, int, int]:
    roi_x, roi_y = int(roi_lower[0]), int(roi_lower[1])
    if roi_x < 0 or roi_y < 0:
        raise ValueError("ROI origin must be non-negative")
    if roi_x >= detector_width or roi_y >= detector_height:
        raise ValueError("ROI origin is outside detector")
    if roi_dim is None:
        roi_width = detector_width - roi_x
        roi_height = detector_height - roi_y
    else:
        roi_width, roi_height = int(roi_dim[0]), int(roi_dim[1])
        if roi_width == 0:
            roi_width = detector_width - roi_x
        if roi_height == 0:
            roi_height = detector_height - roi_y
    if roi_width <= 0 or roi_height <= 0:
        raise ValueError("ROI dimensions must be positive")
    if roi_x + roi_width > detector_width:
        raise ValueError("ROI x range exceeds detector width")
    if roi_y + roi_height > detector_height:
        raise ValueError("ROI y range exceeds detector height")
    return roi_x, roi_y, roi_width, roi_height


def _plan_x_shards(
    *,
    roi_x: int,
    roi_width: int,
    tile_width: int,
    shard_count: int | None,
    shard_width: int | None,
) -> tuple[tuple[int, int], ...]:
    if shard_width is not None:
        ranges = []
        for x0 in range(roi_x, roi_x + roi_width, shard_width):
            ranges.append((x0, min(x0 + shard_width, roi_x + roi_width)))
        return tuple(ranges)
    count = int(shard_count or 1)
    if count > roi_width:
        raise ValueError("shard_count cannot exceed ROI width")
    ranges = []
    cursor = roi_x
    remaining = roi_width
    for index in range(count):
        slots = count - index
        if slots == 1:
            width = remaining
        else:
            target = int(np.ceil(remaining / slots))
            width = int(np.ceil(target / tile_width) * tile_width)
            max_width = remaining - (slots - 1)
            width = min(width, max_width)
        ranges.append((cursor, cursor + width))
        cursor += width
        remaining -= width
    return tuple(ranges)


def _json_safe_options(options: dict[str, Any]) -> dict[str, Any]:
    payload = {}
    for key, value in options.items():
        if isinstance(value, Path):
            payload[key] = str(value)
        elif isinstance(value, tuple):
            payload[key] = list(value)
        else:
            payload[key] = value
    return payload


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def _run_label(value: str | None, output_dir: Path) -> str:
    if value:
        raw = value
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        raw = f"{output_dir.name}-{stamp}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip("-") or "xray"


def _parse_sbatch_job_id(stdout: str) -> str | None:
    match = re.search(r"Submitted batch job\s+(\S+)", stdout)
    if match:
        return match.group(1)
    return None


def _prepend_pythonpath(env: dict[str, str], work_dir: Path) -> None:
    src = work_dir / "src"
    if not src.exists():
        return
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(src) if not current else str(src) + os.pathsep + current
    )


def _pythonpath_export(work_dir: Path) -> str:
    src = work_dir / "src"
    if not src.exists():
        return ":"
    return f"export PYTHONPATH={shlex.quote(str(src))}:${{PYTHONPATH:-}}"


def _visible_cuda_device_tokens(
    requested: int,
    environ: dict[str, str] | None = None,
) -> tuple[str, ...]:
    _require_positive("gpus", requested)
    env = os.environ if environ is None else environ
    if "CUDA_VISIBLE_DEVICES" not in env:
        return tuple(str(index) for index in range(requested))

    raw = env["CUDA_VISIBLE_DEVICES"]
    if not raw.strip() or raw.strip() == "-1":
        raise ValueError(
            f"requested {requested} GPUs but CUDA_VISIBLE_DEVICES is empty"
        )
    tokens = tuple(token.strip() for token in raw.split(","))
    if any(not token for token in tokens):
        raise ValueError("CUDA_VISIBLE_DEVICES contains an empty token")
    if len(set(tokens)) != len(tokens):
        raise ValueError("CUDA_VISIBLE_DEVICES contains duplicate tokens")
    if len(tokens) < requested:
        raise ValueError(
            f"requested {requested} GPUs but only {len(tokens)} are visible"
        )
    return tokens[:requested]


def _require_positive(name: str, value: int) -> None:
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive")
