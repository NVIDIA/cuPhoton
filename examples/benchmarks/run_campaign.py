# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run declared local benchmark adapters in fresh, rotated launch blocks.

See the adjacent README.md for the manifest and adapter receipt contract.
The caller owns hardware reservation, input preparation and resource limits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import signal
import statistics
import subprocess
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_TERMINATE_GRACE_SECONDS = 2.0
_NORMAL_EXIT_GRACE_SECONDS = 0.25
_STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)


@contextmanager
def _defer_stop_signals():
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, _STOP_SIGNALS)
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def _write_bytes(path: Path, data: bytes) -> None:
    # Keep the previous complete receipt until the replacement is durable.
    with _defer_stop_signals():
        temporary = path.with_name(
            f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
        )
        stream = temporary.open("xb")
        try:
            with stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temporary.unlink(missing_ok=True)


def _write(path: Path, value: Any) -> None:
    _write_bytes(
        path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    )


def _read(path: Path, raw: bytes | None = None) -> dict[str, Any]:
    value = json.loads(path.read_bytes() if raw is None else raw)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object")
    # Reject non-standard NaN/Infinity values, including in opaque metadata.
    json.dumps(value, allow_nan=False)
    return value


def _positive_integer(value: Any, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _seconds(value: Any, name: str, *, positive: bool = False) -> None:
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or value < 0
        or (positive and value == 0)
    ):
        bound = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {bound}")


def _object(value: Any, name: str) -> None:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{name} must be a nonempty object")


def _known_keys(value: dict, allowed: set[str], name: str) -> None:
    unknown = value.keys() - allowed
    if unknown:
        raise ValueError(
            f"{name}: unknown keys: {', '.join(sorted(unknown))}"
        )


def prepare(manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    """Validate declarations and freeze a plan without running it."""
    raw = manifest_path.read_bytes()
    manifest = _read(manifest_path, raw)
    _known_keys(
        manifest,
        {
            "schema_version",
            "experiment_kind",
            "blocks",
            "timeout_seconds",
            "comparison",
            "workload",
            "output_contract",
            "cache",
            "disclosure",
            "environment",
            "input_hashes",
            "config",
            "treatments",
        },
        "manifest",
    )
    if (
        type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
    ):
        raise ValueError("schema_version must be 1")
    kind = manifest.setdefault("experiment_kind", "pilot")
    if kind not in {"pilot", "headline"}:
        raise ValueError("experiment_kind must be pilot or headline")
    blocks = manifest.setdefault("blocks", 6)
    _positive_integer(blocks, "blocks")
    if kind == "headline" and blocks < 6:
        raise ValueError("headline experiments require at least six blocks")
    _seconds(
        manifest.setdefault("timeout_seconds", 3600),
        "timeout_seconds",
        positive=True,
    )
    for key in (
        "comparison",
        "workload",
        "output_contract",
        "cache",
        "disclosure",
    ):
        _object(manifest.get(key), key)
    for key in ("environment", "input_hashes", "config"):
        if key in manifest and not isinstance(manifest[key], dict):
            raise ValueError(f"{key} must be an object")
    _known_keys(manifest["workload"], {"name", "measured_rounds"}, "workload")
    if "measured_rounds" in manifest["workload"]:
        _positive_integer(
            manifest["workload"]["measured_rounds"], "measured_rounds"
        )
    treatments = manifest.get("treatments")
    if not isinstance(treatments, list) or not treatments:
        raise ValueError("treatments must be a nonempty list")
    names = set()
    for treatment in treatments:
        if not isinstance(treatment, dict):
            raise ValueError("each treatment must be an object")
        _known_keys(
            treatment,
            {
                "name",
                "cwd",
                "argv",
                "validation_argv",
                "source_sha",
                "backend",
                "expected_completed_work",
                "env",
            },
            "treatment",
        )
        name = treatment.get("name")
        if not isinstance(name, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_-]*", name
        ):
            raise ValueError(
                "treatment name must contain only letters, digits, '_' or '-'"
            )
        if name in names:
            raise ValueError(f"duplicate treatment name: {name}")
        names.add(name)
        cwd = treatment.get("cwd")
        if not isinstance(cwd, str) or not cwd:
            raise ValueError(f"{name}: cwd must be a directory")
        path = Path(cwd)
        if not path.is_absolute():
            path = manifest_path.resolve().parent / path
        if not path.is_dir():
            raise ValueError(f"{name}: cwd does not exist: {path}")
        treatment["cwd"] = str(path.resolve())
        env = treatment.setdefault("env", {})
        if not isinstance(env, dict) or any(
            not isinstance(k, str) or not isinstance(v, str)
            for k, v in env.items()
        ):
            raise ValueError(f"{name}: env must map strings to strings")
        for key in ("argv", "validation_argv"):
            argv = treatment.get(key)
            if (
                not isinstance(argv, list)
                or not argv
                or any(not isinstance(arg, str) or not arg for arg in argv)
            ):
                raise ValueError(
                    f"{name}: {key} must be a nonempty string list"
                )
            executable = argv[0]
            if "/" in executable and not Path(executable).is_absolute():
                executable = str(path / executable)
            executable = shutil.which(
                executable, path=env.get("PATH", os.environ.get("PATH"))
            )
            if not executable or "{launch_dir}" in executable:
                raise ValueError(f"{name}: executable not found for {key}")
            # Preserve a virtual environment's interpreter symlink.
            argv[0] = os.path.abspath(executable)
        if not isinstance(
            treatment.get("source_sha"), str
        ) or not re.fullmatch(r"[0-9a-f]{40}", treatment["source_sha"]):
            raise ValueError(
                f"{name}: source_sha must be a full lowercase Git SHA"
            )
        if (
            not isinstance(treatment.get("backend"), str)
            or not treatment["backend"]
        ):
            raise ValueError(f"{name}: backend must be a nonempty string")
        _positive_integer(
            treatment.get("expected_completed_work"),
            "expected_completed_work",
        )
    output_dir = output_dir.absolute()
    if output_dir.exists():
        raise ValueError(f"output directory already exists: {output_dir}")
    schedule = []
    for block in range(blocks):
        base = (
            treatments
            if (block // len(treatments)) % 2 == 0
            else treatments[::-1]
        )
        offset = block % len(treatments)
        for position, treatment in enumerate(base[offset:] + base[:offset]):
            schedule.append(
                {
                    "block": block,
                    "position": position,
                    "treatment": treatment["name"],
                }
            )
    plan = {
        "manifest": manifest,
        "manifest_path": str(manifest_path.resolve()),
        "output_dir": str(output_dir),
        "schedule": schedule,
    }
    digest = hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        **plan,
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "plan_sha256": digest,
    }


def _command(
    argv: list[str],
    treatment: dict[str, Any],
    launch: Path,
    stage: str,
    deadline: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    command = [arg.replace("{launch_dir}", str(launch)) for arg in argv]
    row: dict[str, Any] = {
        "argv": command,
        "returncode": None,
        "timed_out": False,
        "leaked_children": False,
        "terminate_incomplete": False,
        "group_linger_seconds": 0.0,
    }
    with (
        (launch / f"{stage}.stdout.log").open("wb") as stdout,
        (launch / f"{stage}.stderr.log").open("wb") as stderr,
    ):
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            row.update(
                timed_out=True,
                error="launch deadline expired before stage",
                elapsed_seconds=time.perf_counter() - started,
            )
            _write(launch / f"{stage}-result.json", row)
            return row
        process = subprocess.Popen(
            command,
            cwd=treatment["cwd"],
            env={**os.environ, **treatment["env"]},
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        try:
            try:
                row["returncode"] = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                row["timed_out"] = True
            else:
                linger_started = time.perf_counter()
                linger_deadline = min(
                    deadline,
                    linger_started
                    + (
                        _NORMAL_EXIT_GRACE_SECONDS
                        if row["returncode"] == 0
                        else 0
                    ),
                )
                while _group_exists(process.pid):
                    if time.perf_counter() >= linger_deadline:
                        row["leaked_children"] = True
                        break
                    time.sleep(0.01)
                row["group_linger_seconds"] = (
                    time.perf_counter() - linger_started
                )
            if row["timed_out"] or row["leaked_children"]:
                row.update(_terminate(process))
        except BaseException:
            row["interrupted"] = True
            row.update(_terminate(process))
            raise
        finally:
            row["elapsed_seconds"] = time.perf_counter() - started
            _write(launch / f"{stage}-result.json", row)
    return row


def _group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate(process: subprocess.Popen) -> dict[str, Any]:
    # Probe the group, not just its leader; poll also reaps the owned child.
    # A surviving group (including unreaped zombies) is conservatively
    # incomplete cleanup and must stop the campaign.
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        except OSError as exc:
            return {
                "returncode": process.poll(),
                "terminate_incomplete": True,
                "cleanup_error": f"{type(exc).__name__}: {exc}",
            }
        deadline = time.perf_counter() + _TERMINATE_GRACE_SECONDS
        while True:
            returncode = process.poll()
            if returncode is not None and not _group_exists(process.pid):
                return {
                    "returncode": returncode,
                    "terminate_incomplete": False,
                }
            if time.perf_counter() >= deadline:
                break
            time.sleep(0.01)
    return {"returncode": process.poll(), "terminate_incomplete": True}


def _quality(value: dict[str, Any], filename: str) -> None:
    if value.get("status") != "ok":
        raise ValueError(f"{filename}: status is not ok")
    checks = value.get("quality_checks")
    if not isinstance(checks, list) or not checks:
        raise ValueError(
            f"{filename}: independent science checks are missing"
        )
    names = set()
    for check in checks:
        if (
            not isinstance(check, dict)
            or not isinstance(check.get("name"), str)
            or not check["name"]
        ):
            raise ValueError(f"{filename}: every quality check needs a name")
        if check["name"] in names or check.get("passed") is not True:
            raise ValueError(
                f"{filename}: duplicate or failed check {check['name']}"
            )
        names.add(check["name"])


def validate(
    launch: Path,
    treatment: dict[str, Any],
    expected_rounds: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Check adapter receipts, keeping warm rounds within one launch."""
    result = _read(launch / "result.json")
    validation = _read(launch / "validation.json")
    _quality(result, "result.json")
    _quality(validation, "validation.json")
    for actual, expected in (
        ("source_sha", "source_sha"),
        ("actual_backend", "backend"),
    ):
        if result.get(actual) != treatment[expected]:
            raise ValueError(
                f"result.json: {actual} does not match treatment: "
                f"found {result.get(actual)!r}, "
                f"expected {treatment[expected]!r}"
            )
    _positive_integer(result.get("completed_work"), "completed_work")
    if result["completed_work"] != treatment["expected_completed_work"]:
        raise ValueError(
            "result.json: completed_work does not match treatment"
        )
    if "source_sha" in validation:
        if validation["source_sha"] != treatment["source_sha"]:
            raise ValueError(
                "validation.json: source_sha does not match treatment"
            )
    if "completed_work" in validation:
        _positive_integer(
            validation["completed_work"], "validation completed_work"
        )
        if (
            validation["completed_work"]
            != treatment["expected_completed_work"]
        ):
            raise ValueError(
                "validation.json: completed_work does not match treatment"
            )
    for key in ("input_identities", "output_identities"):
        _object(result.get(key), key)
    rounds = result.get("warm_rounds", [])
    if not isinstance(rounds, list):
        raise ValueError("warm_rounds must be a list")
    if expected_rounds is not None and len(rounds) != expected_rounds:
        raise ValueError("warm_rounds count does not match measured_rounds")
    for item in rounds:
        if (
            not isinstance(item, dict)
            or item.get("phase", "measure") != "measure"
        ):
            raise ValueError(
                "warm_rounds must contain measured round objects only"
            )
        _seconds(
            item.get("elapsed_seconds"), "warm elapsed_seconds", positive=True
        )
    if any("completed_work" in item for item in rounds):
        for item in rounds:
            _positive_integer(
                item.get("completed_work"), "round completed_work"
            )
        if (
            sum(item["completed_work"] for item in rounds)
            != result["completed_work"]
        ):
            raise ValueError(
                "warm_rounds completed_work sum does not match result"
            )
    return result, validation


def execute(plan: dict[str, Any]) -> dict[str, Any]:
    """Execute a plan, retaining every planned launch in the summary."""
    root = Path(plan["output_dir"])
    raw_manifest = Path(plan["manifest_path"]).read_bytes()
    if hashlib.sha256(raw_manifest).hexdigest() != plan["manifest_sha256"]:
        raise ValueError("manifest changed after preparation")
    root.mkdir(parents=True, exist_ok=False)
    with _defer_stop_signals():
        _write_bytes(root / "manifest.json", raw_manifest)
        _write(root / "plan.json", plan)
    manifest = plan["manifest"]
    treatments = {item["name"]: item for item in manifest["treatments"]}
    rows = []
    halt_reason = None
    for entry in plan["schedule"]:
        treatment = treatments[entry["treatment"]]
        launch = root / f"block-{entry['block']:03d}" / treatment["name"]
        launch.mkdir(parents=True, exist_ok=False)
        row: dict[str, Any] = {
            **entry,
            "directory": str(launch),
            "status": "failed",
            "errors": [],
            "warm_median_seconds": None,
            "warm_round_count": None,
        }
        _write(
            launch / "launch.json",
            {
                **entry,
                "plan_sha256": plan["plan_sha256"],
                "treatment": treatment,
            },
        )
        started = time.perf_counter()
        deadline = started + manifest["timeout_seconds"]
        try:
            for stage, key in (
                ("workload", "argv"),
                ("validation", "validation_argv"),
            ):
                row[stage] = _command(
                    treatment[key], treatment, launch, stage, deadline
                )
                if row[stage]["terminate_incomplete"]:
                    halt_reason = f"{stage} process-group cleanup incomplete"
                    raise ValueError(halt_reason)
                if (
                    row[stage]["timed_out"]
                    or row[stage]["returncode"] != 0
                    or row[stage]["leaked_children"]
                ):
                    raise ValueError(
                        f"{stage} failed: "
                        f"returncode={row[stage]['returncode']}, "
                        f"timed_out={row[stage]['timed_out']}, "
                        f"leaked_children={row[stage]['leaked_children']}"
                    )
                result_digest = hashlib.sha256(
                    (launch / "result.json").read_bytes()
                ).hexdigest()
                if stage == "workload":
                    row["result_sha256"] = result_digest
                    validation_path = launch / "validation.json"
                    if (
                        validation_path.exists()
                        or validation_path.is_symlink()
                    ):
                        raise ValueError(
                            "workload created validation.json "
                            "before independent validation"
                        )
                else:
                    row["result_sha256_after_validation"] = result_digest
                    row["validation_sha256"] = hashlib.sha256(
                        (launch / "validation.json").read_bytes()
                    ).hexdigest()
                    if result_digest != row["result_sha256"]:
                        raise ValueError("validation modified result.json")
            result, validation = validate(
                launch, treatment, manifest["workload"].get("measured_rounds")
            )
            row["result"] = result
            row["validation_result"] = validation
            rounds = result.get("warm_rounds", [])
            row["warm_round_count"] = len(rounds)
            if rounds:
                row["warm_median_seconds"] = statistics.median(
                    item["elapsed_seconds"] for item in rounds
                )
            row["status"] = "ok"
        except Exception as exc:
            row["errors"].append(f"{type(exc).__name__}: {exc}")
        except BaseException as exc:
            row["status"] = "interrupted"
            row["errors"].append(type(exc).__name__)
            stage_receipt = launch / f"{stage}-result.json"
            if stage_receipt.is_file():
                row[stage] = _read(stage_receipt)
            raise
        finally:
            row["launch_wall_seconds"] = time.perf_counter() - started
            with _defer_stop_signals():
                _write(launch / "launch-result.json", row)
                rows.append(row)
                _write(root / "launches.json", rows)
        if halt_reason is not None:
            break
    summaries = {}
    for name in treatments:
        selected = [row for row in rows if row["treatment"] == name]
        accepted = len(selected) == manifest["blocks"] and all(
            row["status"] == "ok" for row in selected
        )
        summaries[name] = {
            "planned_launches": manifest["blocks"],
            "accepted_launches": sum(
                row["status"] == "ok" for row in selected
            ),
            "failed_launches": sum(row["status"] != "ok" for row in selected),
            "skipped_launches": manifest["blocks"] - len(selected),
            "realized_position_counts": {
                str(position): sum(
                    row["position"] == position for row in selected
                )
                for position in range(len(treatments))
            },
            "launch_wall_median_seconds": statistics.median(
                row["launch_wall_seconds"] for row in selected
            )
            if accepted
            else None,
            "launch_warm_medians_seconds": [
                row["warm_median_seconds"] for row in selected
            ],
            "warm_round_counts": [
                row["warm_round_count"] for row in selected
            ],
            "warm_round_count_consistent": len(
                {
                    row["warm_round_count"]
                    for row in selected
                    if row["warm_round_count"] is not None
                }
            )
            <= 1,
        }
    passed = len(rows) == len(plan["schedule"]) and all(
        row["status"] == "ok" for row in rows
    )
    report = {
        "status": "ok" if passed else "failed",
        "experiment_kind": manifest["experiment_kind"],
        "headline_eligible": passed
        and manifest["experiment_kind"] == "headline",
        "plan_sha256": plan["plan_sha256"],
        "manifest_sha256": plan["manifest_sha256"],
        "halt_reason": halt_reason,
        "independent_observation": "fresh launch; warm rounds remain nested",
        "timing_boundary": (
            "before adapter launch through independent validation "
            "and receipt checks"
        ),
        "treatments": summaries,
        "launches": rows,
    }
    _write(root / "summary.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    stopping = False

    def terminate(signum, _frame):
        nonlocal stopping
        if not stopping:
            stopping = True
            raise SystemExit(128 + signum)
        # Repeated stop requests must not interrupt cleanup or receipt writes.

    previous_handlers = {
        signum: signal.signal(signum, terminate) for signum in _STOP_SIGNALS
    }
    try:
        plan = prepare(args.manifest, args.output_dir)
        if args.dry_run:
            print(json.dumps(plan, indent=2, allow_nan=False))
            return 0
        report = execute(plan)
        print(
            json.dumps(
                {
                    key: report[key]
                    for key in (
                        "status",
                        "headline_eligible",
                        "plan_sha256",
                        "treatments",
                    )
                },
                indent=2,
            )
        )
        return 0 if report["status"] == "ok" else 1
    except Exception as exc:
        parser.exit(2, f"{type(exc).__name__}: {exc}\n")
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
