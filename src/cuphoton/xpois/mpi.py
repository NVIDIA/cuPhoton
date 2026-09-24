# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Optional MPI execution for independent xPois image pairs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import socket
import stat
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from cuphoton.core.bulk import (
    WorkItem,
    atomic_write_json,
    audit_terminal_records,
    classify_physical_gpu_pair,
    error_payload,
    json_mapping,
    new_run_id,
    partition_byte_balanced,
    read_json_mapping,
    timestamp_utc,
    validate_identifier,
)
from cuphoton.core.bulk import (
    collect_gpu_identity as _gpu_identity,
)
from cuphoton.core.bulk import (
    item_ids_sha256 as _item_ids_sha256,
)
from cuphoton.core.bulk import (
    regular_file as _regular_file,
)

from .batch import (
    BatchFitOptions,
    ImagePairManifest,
    load_image_pair_manifest,
    preflight_image_pair_manifest,
    run_image_pair_item,
)

_AGGREGATION_MODES = frozenset({"files", "mpi"})
_GPU_BACKENDS = frozenset({"cupy", "cutile", "numba-cuda"})
_ALLOCATED_DEVICES = "CUPHOTON_ALLOCATED_CUDA_VISIBLE_DEVICES"
_CUDA_MODULES = ("cupy", "numba.cuda", "cuda.tile")
_UNSIGNED = re.compile(r"[0-9]{1,18}")
_POLL_INITIAL_SEC = 0.05
_POLL_MAX_SEC = 1.0


@dataclass(frozen=True)
class MPIBatchResult:
    """Root-rank handle for a terminal xPois MPI batch."""

    run_id: str
    run_dir: Path
    summary_path: Path
    status: str
    summary: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a compact command result."""

        return {
            "executor": "mpi",
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "summary_path": str(self.summary_path),
            "status": self.status,
        }


@dataclass(frozen=True)
class _RankContext:
    rank: int
    local_rank: int
    world_size: int
    host: str
    launcher: str
    launch_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _MPIAPI:
    MPI: Any
    comm: Any
    mpi4py_version: str | None
    library_version: str | None


class _FileOwnershipError(RuntimeError):
    """Raised when a rank does not own the requested file-run state."""


class _FileAttemptExistsError(_FileOwnershipError):
    """Raised when an immutable file-attempt identity is already claimed."""


def run_mpi_image_pair_batch(
    *,
    manifest_path: Path,
    output_root: Path,
    run_id: str | None,
    aggregation_mode: str,
    rank_timeout_sec: float | None,
    attempt_id: str | None,
    options: BatchFitOptions,
    rank_setup_timeout_sec: float = 600.0,
) -> MPIBatchResult | None:
    """Run byte-balanced xPois shards under an external MPI launcher."""

    started_at = timestamp_utc()
    start = time.perf_counter()
    api = None
    if aggregation_mode == "mpi":
        visibility = None
        startup_error = None
        try:
            visibility = _validate_rank_startup(
                aggregation_mode,
                rank_setup_timeout_sec,
                rank_timeout_sec,
                run_id,
                attempt_id,
                options,
                os.environ,
            )
        except Exception as exc:
            startup_error = error_payload(exc)
        if startup_error is None and visibility is None:
            startup_error = error_payload(
                RuntimeError("rank startup did not resolve GPU visibility")
            )
        api = _load_mpi_api()
        _mpi_failure_consensus(
            api.comm,
            "MPI rank-startup validation",
            startup_error,
        )
        if visibility is None:
            raise RuntimeError(
                "MPI startup consensus completed without GPU visibility"
            )
        context = _mpi_context(api.MPI, api.comm, os.environ)
        effective_run_id = _mpi_run_id(api.comm, context, run_id)
    else:
        visibility = _validate_rank_startup(
            aggregation_mode,
            rank_setup_timeout_sec,
            rank_timeout_sec,
            run_id,
            attempt_id,
            options,
            os.environ,
        )
        context = _environment_context(os.environ)
        assert run_id is not None
        effective_run_id = run_id

    manifest, manifest_error = _manifest_preflight(
        manifest_path, options, validate_inputs=context.rank == 0
    )
    run_dir = None
    run_dir_error = None
    try:
        run_dir = (
            output_root.expanduser().resolve(strict=False) / effective_run_id
        )
    except Exception as exc:
        if api is None:
            raise
        run_dir_error = error_payload(exc)
    shards = None
    if manifest_error is None:
        assert manifest is not None
        try:
            items = manifest.work_items()
            if context.world_size > len(items):
                raise ValueError(
                    "MPI world size cannot exceed the number of manifest "
                    "items"
                )
            shards = partition_byte_balanced(items, context.world_size)
        except Exception as exc:
            manifest_error = error_payload(exc)
    if manifest_error is None and (manifest is None or shards is None):
        manifest_error = error_payload(
            RuntimeError("manifest preflight did not produce work shards")
        )
    if run_dir_error is None and run_dir is None:
        run_dir_error = error_payload(
            RuntimeError("run-directory resolution produced no path")
        )
    if api is not None:
        _mpi_failure_consensus(
            api.comm,
            "MPI run-directory resolution",
            run_dir_error,
        )
        _mpi_manifest_consensus(
            api.comm,
            context,
            manifest,
            manifest_error,
            run_id,
            run_dir,
            options,
            api.mpi4py_version,
            api.library_version,
            rank_setup_timeout_sec,
        )
        if manifest is None or shards is None or run_dir is None:
            raise RuntimeError(
                "MPI manifest consensus completed without launch state"
            )
        _mpi_prepare_run(
            api.comm,
            context,
            run_dir,
            effective_run_id,
            started_at,
            manifest,
            options,
            api,
            rank_setup_timeout_sec,
            shards=shards,
            start=start,
        )
        setup_error = None
    else:
        assert attempt_id is not None
        assert rank_timeout_sec is not None
        assert run_dir is not None
        try:
            setup_error = _file_prepare_run(
                context,
                run_dir,
                effective_run_id,
                attempt_id,
                rank_setup_timeout_sec,
                started_at,
                manifest,
                options,
                manifest_error,
                rank_timeout_sec=rank_timeout_sec,
            )
        except Exception as exc:
            if (
                isinstance(exc, _FileOwnershipError)
                or context.rank == 0
                or manifest is None
                or shards is None
                or not _active_file_attempt_for_rank_failure(
                    run_dir,
                    effective_run_id,
                    attempt_id,
                    manifest.sha256,
                    context.launch_id,
                )
            ):
                raise
            setup_error = error_payload(exc)
        assert manifest is not None
        assert shards is not None

    rank_run_dir = run_dir
    if api is None:
        assert attempt_id is not None
        rank_run_dir = _file_rank_staging_dir(
            run_dir, effective_run_id, attempt_id, context.rank
        )
    rank_result = _execute_rank(
        run_id=effective_run_id,
        run_dir=rank_run_dir,
        manifest_sha256=manifest.sha256,
        context=context,
        items=shards[context.rank],
        options=options,
        visibility=visibility,
        setup_error=setup_error,
        item_runner=run_image_pair_item,
        gpu_identity_loader=_gpu_identity,
        attempt_id=attempt_id if api is None else None,
    )
    if api is not None:
        return _mpi_aggregate(
            api.comm,
            context,
            run_dir,
            effective_run_id,
            manifest,
            options,
            shards,
            rank_result,
            started_at,
            start,
            api,
            rank_setup_timeout_sec=rank_setup_timeout_sec,
        )
    assert attempt_id is not None
    try:
        _complete_file_rank(
            rank_run_dir,
            effective_run_id,
            attempt_id,
            manifest.sha256,
            context.rank,
            rank_result,
        )
    except Exception as exc:
        if context.rank == 0:
            error = error_payload(exc)
            _aggregate_error(run_dir, effective_run_id, error)
            try:
                _set_attempt_terminal(
                    run_dir,
                    effective_run_id,
                    attempt_id,
                    manifest.sha256,
                    "failed",
                    error,
                )
            except Exception as terminal_exc:
                raise RuntimeError(
                    "cannot publish or terminalize MPI rank-zero completion"
                ) from terminal_exc
            raise RuntimeError(
                "cannot publish MPI rank-zero completion"
            ) from exc
        raise
    return _file_aggregate(
        context,
        run_dir,
        effective_run_id,
        attempt_id,
        manifest,
        options,
        shards,
        rank_timeout_sec,
        started_at,
        start,
        rank_setup_timeout_sec=rank_setup_timeout_sec,
    )


def _validate_rank_startup(
    aggregation_mode: str,
    rank_setup_timeout_sec: float,
    rank_timeout_sec: float | None,
    run_id: str | None,
    attempt_id: str | None,
    options: BatchFitOptions,
    environ: Mapping[str, str],
) -> str:
    _validate_arguments(
        aggregation_mode,
        rank_setup_timeout_sec,
        rank_timeout_sec,
        run_id,
        attempt_id,
        options,
    )
    visibility = _prebound_visibility(environ)
    premature = sorted(name for name in _CUDA_MODULES if name in sys.modules)
    if premature:
        raise RuntimeError(
            "CUDA modules were imported before MPI rank binding: "
            + ", ".join(premature)
        )
    return visibility


def _mpi_failure_consensus(
    comm: Any,
    phase: str,
    error: Mapping[str, Any] | None,
    *,
    root_failure_handler: Callable[[Mapping[str, str]], None] | None = None,
) -> None:
    rank = int(comm.Get_rank())
    size = int(comm.Get_size())
    gathered = comm.gather(
        {"rank": rank, "error": dict(error) if error else None},
        root=0,
    )
    decision = None
    if rank == 0:
        failures: list[str] = []
        if not isinstance(gathered, Sequence) or len(gathered) != size:
            failures.append("invalid consensus result")
        else:
            for expected_rank, item in enumerate(gathered):
                if not isinstance(item, Mapping):
                    failures.append(
                        f"rank {expected_rank}: invalid consensus result"
                    )
                    continue
                if _integer(item.get("rank")) != expected_rank:
                    failures.append(
                        f"rank {expected_rank}: invalid rank identity"
                    )
                item_error = item.get("error")
                if item_error is None:
                    continue
                if not _valid_error(item_error):
                    failures.append(
                        f"rank {expected_rank}: invalid error payload"
                    )
                    continue
                failures.append(
                    f"rank {expected_rank}: {item_error['type']}: "
                    f"{item_error['message']}"
                )
        message = (
            f"{phase} failed: " + "; ".join(failures) if failures else None
        )
        if message is not None and root_failure_handler is not None:
            try:
                root_failure_handler(
                    {"type": "RuntimeError", "message": message}
                )
            except Exception as exc:
                persistence_error = error_payload(exc)
                message += (
                    "; failed to persist terminal setup evidence: "
                    f"{persistence_error['type']}: "
                    f"{persistence_error['message']}"
                )
        decision = {"error": message}
    decision = comm.bcast(decision, root=0)
    if not isinstance(decision, Mapping):
        raise RuntimeError(f"{phase} decision was not a mapping")
    if decision.get("error"):
        raise RuntimeError(str(decision["error"]))


def _validate_arguments(
    aggregation_mode: str,
    rank_setup_timeout_sec: float,
    rank_timeout_sec: float | None,
    run_id: str | None,
    attempt_id: str | None,
    options: BatchFitOptions,
) -> None:
    if aggregation_mode not in _AGGREGATION_MODES:
        raise ValueError("aggregation_mode must be one of: files, mpi")
    if options.backend not in _GPU_BACKENDS:
        raise ValueError("MPI xPois ranks require an explicit GPU backend")
    if (
        isinstance(rank_setup_timeout_sec, bool)
        or not isinstance(rank_setup_timeout_sec, (int, float))
        or not math.isfinite(rank_setup_timeout_sec)
        or rank_setup_timeout_sec <= 0
    ):
        raise ValueError("MPI requires a positive rank_setup_timeout_sec")
    if run_id is not None:
        validate_identifier(run_id, field="run_id")
    if aggregation_mode == "mpi":
        if rank_timeout_sec is not None or attempt_id is not None:
            raise ValueError(
                "rank_timeout_sec and attempt_id are only valid for files"
            )
        return
    if run_id is None or attempt_id is None:
        raise ValueError(
            "file aggregation requires explicit run_id and attempt_id"
        )
    if (
        isinstance(rank_timeout_sec, bool)
        or not isinstance(rank_timeout_sec, (int, float))
        or not math.isfinite(rank_timeout_sec)
        or rank_timeout_sec <= 0
    ):
        raise ValueError(
            "file aggregation requires a positive rank_timeout_sec"
        )
    if (
        not attempt_id
        or len(attempt_id) > 256
        or any(ord(character) < 32 for character in attempt_id)
    ):
        raise ValueError("attempt_id is invalid")


def _manifest_preflight(
    path: Path, options: BatchFitOptions, *, validate_inputs: bool = True
) -> tuple[ImagePairManifest | None, dict[str, str] | None]:
    try:
        manifest = load_image_pair_manifest(path)
        if validate_inputs:
            preflight_image_pair_manifest(manifest, options)
        return manifest, None
    except Exception as exc:
        return None, error_payload(exc)


def _mpi_manifest_consensus(
    comm: Any,
    context: _RankContext,
    manifest: ImagePairManifest | None,
    error: Mapping[str, str] | None,
    run_id: str | None,
    resolved_run_dir: Path | None,
    options: BatchFitOptions,
    mpi4py_version: str | None,
    library_version: str | None,
    rank_setup_timeout_sec: float = 600.0,
) -> None:
    local = {
        "rank": context.rank,
        "manifest_sha256": manifest.sha256 if manifest else None,
        "run_id": run_id,
        "resolved_run_dir": (
            str(resolved_run_dir) if resolved_run_dir is not None else None
        ),
        "options": options.to_payload(),
        "mpi4py_version": mpi4py_version,
        "mpi_library_version": _normalize_mpi_library_version(
            library_version
        ),
        "rank_setup_timeout_sec": rank_setup_timeout_sec,
        "error": dict(error) if error else None,
    }
    gathered = comm.gather(local, root=0)
    decision = None
    if context.rank == 0:
        message = None
        records: list[Mapping[str, Any]] = []
        invalid: list[str] = []
        if (
            not isinstance(gathered, Sequence)
            or len(gathered) != context.world_size
        ):
            invalid.append("invalid consensus result")
        else:
            for expected_rank, item in enumerate(gathered):
                if not isinstance(item, Mapping):
                    invalid.append(
                        f"rank {expected_rank}: invalid consensus result"
                    )
                    continue
                if _integer(item.get("rank")) != expected_rank:
                    invalid.append(
                        f"rank {expected_rank}: invalid rank identity"
                    )
                item_error = item.get("error")
                if item_error is not None and not _valid_error(item_error):
                    invalid.append(
                        f"rank {expected_rank}: invalid error payload"
                    )
                if not isinstance(item.get("options"), Mapping):
                    invalid.append(f"rank {expected_rank}: invalid options")
                for field in (
                    "manifest_sha256",
                    "run_id",
                    "resolved_run_dir",
                    "mpi4py_version",
                    "mpi_library_version",
                ):
                    if item.get(field) is not None and not isinstance(
                        item.get(field), str
                    ):
                        invalid.append(
                            f"rank {expected_rank}: invalid {field}"
                        )
                setup_timeout = item.get("rank_setup_timeout_sec")
                if (
                    isinstance(setup_timeout, bool)
                    or not isinstance(setup_timeout, (int, float))
                    or not math.isfinite(setup_timeout)
                    or setup_timeout <= 0
                ):
                    invalid.append(
                        f"rank {expected_rank}: invalid setup timeout"
                    )
                records.append(item)
        if invalid:
            message = "invalid MPI manifest consensus: " + "; ".join(invalid)
        else:
            failures = [
                f"rank {item['rank']}: {item['error']['type']}: "
                f"{item['error']['message']}"
                for item in records
                if item.get("error") is not None
            ]
            identities = {
                (
                    item.get("manifest_sha256"),
                    item.get("run_id"),
                    item.get("resolved_run_dir"),
                )
                for item in records
            }
            if failures:
                message = "manifest preflight failed: " + "; ".join(failures)
            elif len(identities) != 1:
                message = (
                    "manifest, requested run ID, or resolved run directory "
                    "differs across MPI ranks"
                )
            elif any(
                item["options"] != records[0]["options"] for item in records
            ):
                message = "xPois options differ across MPI ranks"
            elif any(
                item.get("mpi4py_version") != records[0].get("mpi4py_version")
                for item in records
            ):
                message = "mpi4py versions differ across MPI ranks"
            elif any(
                item.get("mpi_library_version")
                != records[0].get("mpi_library_version")
                for item in records
            ):
                message = "MPI library versions differ across MPI ranks"
            elif any(
                item.get("rank_setup_timeout_sec")
                != records[0].get("rank_setup_timeout_sec")
                for item in records
            ):
                message = "MPI setup timeouts differ across MPI ranks"
        decision = {
            "error": message,
            "manifest_sha256": manifest.sha256 if manifest else None,
        }
    decision = comm.bcast(decision, root=0)
    if not isinstance(decision, Mapping):
        raise RuntimeError("MPI manifest decision was not a mapping")
    if decision.get("error"):
        raise RuntimeError(str(decision["error"]))
    if manifest is None or decision.get("manifest_sha256") != manifest.sha256:
        raise RuntimeError("MPI root preflight manifest digest differs")


def _mpi_run_id(
    comm: Any, context: _RankContext, requested: str | None
) -> str:
    candidate = None
    if context.rank == 0:
        candidate = requested or new_run_id("mpi-xpois")
    return validate_identifier(comm.bcast(candidate, root=0), field="run_id")


def _mpi_prepare_run(
    comm: Any,
    context: _RankContext,
    run_dir: Path,
    run_id: str,
    started_at: str,
    manifest: ImagePairManifest,
    options: BatchFitOptions,
    api: _MPIAPI,
    rank_setup_timeout_sec: float = 600.0,
    *,
    shards: Sequence[Sequence[WorkItem]],
    start: float,
) -> None:
    failure = None
    shared_nonce = None
    run_dir_claimed = False
    if context.rank == 0:
        try:
            shared_nonce = secrets.token_hex(32)
            run_dir.mkdir(parents=True, exist_ok=False)
            run_dir_claimed = True
            _create_run(
                run_dir,
                run_id,
                started_at,
                manifest,
                options,
                "mpi",
                context.world_size,
                None,
                None,
                api.mpi4py_version,
                api.library_version,
                shared_nonce,
                rank_setup_timeout_sec=rank_setup_timeout_sec,
                directory_claimed=True,
            )
        except Exception as exc:
            failure = error_payload(exc)
            if run_dir_claimed:
                try:
                    _finalize_collective_setup_failure(
                        run_dir,
                        run_id,
                        manifest,
                        options,
                        shards,
                        started_at,
                        start,
                        api,
                        rank_setup_timeout_sec,
                        failure,
                    )
                except Exception as persistence_exc:
                    persistence_error = error_payload(persistence_exc)
                    failure["message"] += (
                        "; failed to persist terminal setup evidence: "
                        f"{persistence_error['type']}: "
                        f"{persistence_error['message']}"
                    )
    failure = comm.bcast(failure, root=0)
    if failure:
        raise RuntimeError(
            f"cannot prepare MPI run: {failure['type']}: {failure['message']}"
        )
    shared_nonce = comm.bcast(shared_nonce, root=0)
    shared_error = None
    try:
        if not isinstance(shared_nonce, str) or not shared_nonce:
            raise RuntimeError("MPI shared-run nonce is invalid")
        ready = _wait_identity_mapping(
            run_dir / ".ready.json",
            rank_setup_timeout_sec,
            {
                "run_id": run_id,
                "attempt_id": None,
                "shared_nonce": shared_nonce,
            },
        )
        mismatched = [
            field
            for field, expected in (
                ("manifest_sha256", manifest.sha256),
                ("resolved_run_dir", str(run_dir)),
                ("status", "ready"),
            )
            if ready.get(field) != expected
        ]
        if mismatched:
            raise RuntimeError(
                "MPI shared run directory differs in: "
                + ", ".join(mismatched)
            )
    except Exception as exc:
        shared_error = error_payload(exc)
    _mpi_failure_consensus(
        comm,
        "MPI shared run-directory validation",
        shared_error,
        root_failure_handler=lambda error: _finalize_collective_setup_failure(
            run_dir,
            run_id,
            manifest,
            options,
            shards,
            started_at,
            start,
            api,
            rank_setup_timeout_sec,
            error,
        ),
    )


def _finalize_collective_setup_failure(
    run_dir: Path,
    run_id: str,
    manifest: ImagePairManifest,
    options: BatchFitOptions,
    shards: Sequence[Sequence[WorkItem]],
    started_at: str,
    start: float,
    api: _MPIAPI,
    rank_setup_timeout_sec: float,
    error: Mapping[str, str],
) -> None:
    _finalize(
        run_dir,
        run_id,
        manifest,
        options,
        shards,
        (),
        ({"phase": "setup", **dict(error)},),
        "mpi",
        started_at,
        start,
        api.mpi4py_version,
        api.library_version,
        rank_setup_timeout_sec=rank_setup_timeout_sec,
    )


def _file_prepare_run(
    context: _RankContext,
    run_dir: Path,
    run_id: str,
    attempt_id: str,
    rank_setup_timeout_sec: float,
    started_at: str,
    manifest: ImagePairManifest | None,
    options: BatchFitOptions,
    preflight_error: Mapping[str, Any] | None = None,
    *,
    rank_timeout_sec: float | None = None,
) -> dict[str, str] | None:
    if rank_timeout_sec is None:
        rank_timeout_sec = rank_setup_timeout_sec
    if not context.launch_id:
        raise _FileOwnershipError(
            "file aggregation requires a launch identity"
        )
    marker = _attempt_path(run_dir, run_id, attempt_id)
    _ensure_real_file_attempt_root(marker)
    if context.rank == 0:
        staging_dir = _attempt_staging_dir(run_dir, run_id, attempt_id)
        try:
            _claim_file_attempt(marker)
        except _FileAttemptExistsError as claim_error:
            repaired = _repair_committed_file_attempt(
                run_dir,
                run_id,
                attempt_id,
                manifest,
                options,
                context.world_size,
                rank_setup_timeout_sec,
                rank_timeout_sec,
                preflight_error,
            )
            if repaired is None:
                raise
            raise RuntimeError(
                "cannot prepare MPI file run: repaired the terminal marker "
                f"for the existing immutable summary at {repaired}; inspect "
                "that summary instead of rerunning the attempt"
            ) from claim_error
        failure = None
        try:
            staging_dir.mkdir(parents=True, exist_ok=False)
            (staging_dir / "preflight").mkdir()
            atomic_write_json(
                marker,
                {
                    "run_id": run_id,
                    "attempt_id": attempt_id,
                    "launch_id": context.launch_id,
                    "manifest_sha256": (
                        manifest.sha256 if manifest is not None else None
                    ),
                    "status": "starting",
                    "terminal": False,
                },
            )
            rank_preflight_error = _prepare_file_rank_for_preflight(
                context,
                run_dir,
                run_id,
                attempt_id,
                preflight_error,
                rank_setup_timeout_sec,
            )
            _publish_file_preflight(
                context,
                run_dir,
                run_id,
                attempt_id,
                manifest,
                options,
                rank_preflight_error,
                rank_setup_timeout_sec,
                rank_timeout_sec,
            )
            if rank_preflight_error is not None:
                failure = {
                    "rank": context.rank,
                    **dict(rank_preflight_error),
                }
            else:
                assert manifest is not None
                failure = _file_preflight_consensus(
                    run_dir,
                    run_id,
                    attempt_id,
                    rank_setup_timeout_sec,
                    manifest,
                    options,
                    context.world_size,
                    rank_setup_timeout_sec,
                    rank_timeout_sec,
                    context.launch_id,
                )
            if failure is None:
                assert manifest is not None
                _create_run(
                    run_dir,
                    run_id,
                    started_at,
                    manifest,
                    options,
                    "files",
                    context.world_size,
                    attempt_id,
                    rank_timeout_sec,
                    None,
                    None,
                    None,
                    rank_setup_timeout_sec=rank_setup_timeout_sec,
                )
                atomic_write_json(
                    marker,
                    {
                        "run_id": run_id,
                        "attempt_id": attempt_id,
                        "launch_id": context.launch_id,
                        "manifest_sha256": manifest.sha256,
                        "status": "ready",
                        "terminal": False,
                        "error": None,
                    },
                )
        except Exception as exc:
            failure = error_payload(exc)
        if failure:
            _set_attempt_terminal(
                run_dir,
                run_id,
                attempt_id,
                manifest.sha256 if manifest is not None else None,
                "failed",
                failure,
            )
            raise RuntimeError(
                "cannot prepare MPI file run: "
                f"{failure['type']}: {failure['message']}"
            )
        return None

    attempt = _wait_mapping(
        marker,
        rank_setup_timeout_sec,
        {"starting", "ready", "failed"},
        accept_terminal=True,
    )
    if (
        attempt.get("status") == "ready"
        and attempt.get("terminal") is False
        and os.path.lexists(run_dir / "summary.json")
    ):
        raise RuntimeError(
            "MPI file-attempt has a committed summary awaiting rank-zero "
            "terminal-marker repair"
        )
    _validate_file_attempt(attempt, run_id, attempt_id, context.launch_id)
    if attempt.get("status") == "ready":
        raise _FileOwnershipError("MPI file rank is already claimed")
    rank_preflight_error = (
        dict(preflight_error) if preflight_error is not None else None
    )
    if attempt.get("status") == "starting":
        rank_preflight_error = _prepare_file_rank_for_preflight(
            context,
            run_dir,
            run_id,
            attempt_id,
            preflight_error,
            rank_setup_timeout_sec,
        )
        _publish_file_preflight(
            context,
            run_dir,
            run_id,
            attempt_id,
            manifest,
            options,
            rank_preflight_error,
            rank_setup_timeout_sec,
            rank_timeout_sec,
        )
        attempt = _wait_mapping(
            marker,
            rank_setup_timeout_sec,
            {"ready", "failed"},
            accept_terminal=True,
        )
        _validate_file_attempt(attempt, run_id, attempt_id, context.launch_id)
    if rank_preflight_error is not None:
        raise RuntimeError(
            "rank preflight failed: "
            f"{rank_preflight_error['type']}: "
            f"{rank_preflight_error['message']}"
        )
    assert manifest is not None
    return _file_ready_error(
        context,
        run_dir,
        run_id,
        attempt_id,
        rank_setup_timeout_sec,
        manifest,
        options,
        rank_timeout_sec=rank_timeout_sec,
    )


def _prepare_file_rank_for_preflight(
    context: _RankContext,
    run_dir: Path,
    run_id: str,
    attempt_id: str,
    error: Mapping[str, Any] | None,
    rank_setup_timeout_sec: float,
) -> dict[str, Any] | None:
    rank_dir = _file_rank_staging_dir(
        run_dir, run_id, attempt_id, context.rank
    )
    try:
        _prepare_file_rank_staging(
            rank_dir,
            parent_visibility_timeout_sec=rank_setup_timeout_sec,
        )
    except FileExistsError as exc:
        raise _FileOwnershipError("MPI file rank is already claimed") from exc
    except Exception as exc:
        return error_payload(exc)
    return dict(error) if error is not None else None


def _publish_file_preflight(
    context: _RankContext,
    run_dir: Path,
    run_id: str,
    attempt_id: str,
    manifest: ImagePairManifest | None,
    options: BatchFitOptions,
    error: Mapping[str, Any] | None,
    rank_setup_timeout_sec: float,
    rank_timeout_sec: float,
) -> None:
    atomic_write_json(
        _file_preflight_path(run_dir, run_id, attempt_id, context.rank),
        {
            "schema": "cuphoton.xpois.mpi-file-preflight/v2",
            "run_id": run_id,
            "attempt_id": attempt_id,
            "launch_id": context.launch_id,
            "resolved_run_dir": str(run_dir),
            "attempt_marker": str(_attempt_path(run_dir, run_id, attempt_id)),
            "rank": context.rank,
            "world_size": context.world_size,
            "manifest_sha256": (
                manifest.sha256 if manifest is not None else None
            ),
            "options": options.to_payload(),
            "rank_setup_timeout_sec": rank_setup_timeout_sec,
            "rank_timeout_sec": rank_timeout_sec,
            "status": "failed" if error is not None else "success",
            "error": dict(error) if error is not None else None,
        },
        overwrite=False,
    )


def _file_preflight_consensus(
    run_dir: Path,
    run_id: str,
    attempt_id: str,
    timeout: float,
    manifest: ImagePairManifest,
    options: BatchFitOptions,
    world_size: int,
    rank_setup_timeout_sec: float,
    rank_timeout_sec: float,
    launch_id: str,
) -> dict[str, Any] | None:
    pending = set(range(world_size))
    deadline = time.monotonic() + timeout
    delay = _POLL_INITIAL_SEC
    while pending and time.monotonic() < deadline:
        for rank in tuple(pending):
            path = _file_preflight_path(run_dir, run_id, attempt_id, rank)
            if not os.path.lexists(path):
                continue
            try:
                record = _read_regular_mapping(path)
            except Exception as exc:
                return {"rank": rank, **error_payload(exc)}
            failure = _file_preflight_record_error(
                record,
                rank,
                run_id,
                attempt_id,
                run_dir,
                manifest,
                options,
                world_size,
                launch_id=launch_id,
                rank_setup_timeout_sec=rank_setup_timeout_sec,
                rank_timeout_sec=rank_timeout_sec,
            )
            if failure is not None:
                return failure
            pending.remove(rank)
        if pending:
            delay = _shared_filesystem_wait(deadline, delay)
    if pending:
        return {
            "rank": min(pending),
            "type": "TimeoutError",
            "message": "rank preflight did not appear before timeout",
        }
    return None


def _file_preflight_record_error(
    record: Mapping[str, Any],
    rank: int,
    run_id: str,
    attempt_id: str,
    run_dir: Path,
    manifest: ImagePairManifest,
    options: BatchFitOptions,
    world_size: int,
    *,
    launch_id: str,
    rank_setup_timeout_sec: float | None = None,
    rank_timeout_sec: float | None = None,
) -> dict[str, Any] | None:
    mismatched = [
        field
        for field, expected in (
            ("schema", "cuphoton.xpois.mpi-file-preflight/v2"),
            ("run_id", run_id),
            ("attempt_id", attempt_id),
            ("launch_id", launch_id),
            ("resolved_run_dir", str(run_dir)),
            (
                "attempt_marker",
                str(_attempt_path(run_dir, run_id, attempt_id)),
            ),
            ("rank", rank),
            ("world_size", world_size),
        )
        if record.get(field) != expected
    ]
    if record.get("status") not in {"success", "failed"}:
        mismatched.append("status")
    if mismatched:
        return {
            "rank": rank,
            "type": "RunSetupMismatchError",
            "message": "file preflight differs in: "
            + ", ".join(sorted(set(mismatched))),
        }
    if record.get("status") == "failed":
        error = record.get("error")
        if (
            not isinstance(error, Mapping)
            or not isinstance(error.get("type"), str)
            or not isinstance(error.get("message"), str)
        ):
            return {
                "rank": rank,
                "type": "RunSetupMismatchError",
                "message": "file preflight differs in: error",
            }
        return {"rank": rank, **dict(error)}
    expected_setup = [
        field
        for field, expected in (
            ("manifest_sha256", manifest.sha256),
            ("options", options.to_payload()),
            ("error", None),
        )
        if record.get(field) != expected
    ]
    for field, expected in (
        ("rank_setup_timeout_sec", rank_setup_timeout_sec),
        ("rank_timeout_sec", rank_timeout_sec),
    ):
        if expected is not None and record.get(field) != expected:
            expected_setup.append(field)
    mismatched = expected_setup
    if mismatched:
        return {
            "rank": rank,
            "type": "RunSetupMismatchError",
            "message": "file preflight differs in: " + ", ".join(mismatched),
        }
    return None


def _validate_file_attempt(
    attempt: Mapping[str, Any], run_id: str, attempt_id: str, launch_id: str
) -> None:
    if (
        attempt.get("run_id") != run_id
        or attempt.get("attempt_id") != attempt_id
    ):
        raise RuntimeError("MPI file-attempt identity mismatch")
    if attempt.get("terminal") is True:
        if attempt.get("status") == "failed":
            failure = attempt.get("error")
            if isinstance(failure, Mapping):
                raise RuntimeError(
                    "MPI file-run setup failed: "
                    f"{failure.get('type')}: {failure.get('message')}"
                )
            if failure is None:
                raise RuntimeError(
                    "MPI file-attempt is already terminal; inspect "
                    "summary.json"
                )
            raise RuntimeError("MPI file-run setup failed on rank zero")
        raise RuntimeError("MPI file-attempt is already terminal")
    if attempt.get("launch_id") != launch_id:
        raise _FileOwnershipError(
            "MPI file-attempt belongs to another launch"
        )


def _file_ready_error(
    context: _RankContext,
    run_dir: Path,
    run_id: str,
    attempt_id: str,
    rank_setup_timeout_sec: float,
    manifest: ImagePairManifest,
    options: BatchFitOptions,
    *,
    rank_timeout_sec: float | None = None,
) -> dict[str, str] | None:
    if rank_timeout_sec is None:
        rank_timeout_sec = rank_setup_timeout_sec
    ready = _wait_identity_mapping(
        run_dir / ".ready.json",
        rank_setup_timeout_sec,
        {"run_id": run_id, "attempt_id": attempt_id},
    )
    mismatched = [
        field
        for field, expected in (("manifest_sha256", manifest.sha256),)
        if ready.get(field) != expected
    ]
    if not mismatched:
        run = _wait_identity_mapping(
            run_dir / "run.json",
            rank_setup_timeout_sec,
            {"run_id": run_id, "attempt_id": attempt_id},
        )
        mismatched = [
            field
            for field, expected in (
                ("manifest_sha256", manifest.sha256),
                ("aggregation_mode", "files"),
                ("world_size", context.world_size),
                ("rank_setup_timeout_sec", rank_setup_timeout_sec),
                ("rank_timeout_sec", rank_timeout_sec),
                ("options", options.to_payload()),
            )
            if run.get(field) != expected
        ]
        if not mismatched:
            return None
    return {
        "type": "RunSetupMismatchError",
        "message": "file setup differs in: " + ", ".join(mismatched),
    }


def _create_run(
    run_dir: Path,
    run_id: str,
    started_at: str,
    manifest: ImagePairManifest,
    options: BatchFitOptions,
    mode: str,
    world_size: int,
    attempt_id: str | None,
    rank_timeout_sec: float | None,
    mpi4py_version: str | None,
    library_version: str | None,
    shared_nonce: str | None,
    rank_setup_timeout_sec: float | None = None,
    *,
    directory_claimed: bool = False,
) -> None:
    if not directory_claimed:
        run_dir.mkdir(parents=True, exist_ok=False)
    elif not _real_directory(run_dir):
        raise ValueError("claimed MPI run path is not a real directory")
    for name in ("items", "ranks", "records"):
        (run_dir / name).mkdir()
    atomic_write_json(run_dir / "manifest.json", manifest.canonical_payload())
    atomic_write_json(
        run_dir / "input-identity.json", manifest.input_identity_payload()
    )
    run = {
        "schema": "cuphoton.xpois.mpi-run/v1",
        "record_type": "immutable-launch",
        "executor": "mpi",
        "run_id": run_id,
        "started_at_utc": started_at,
        "manifest_sha256": manifest.sha256,
        "aggregation_mode": mode,
        "world_size": world_size,
        "attempt_id": attempt_id,
        "mpi4py_version": mpi4py_version,
        "mpi_library_version": library_version,
        "options": options.to_payload(),
    }
    if rank_setup_timeout_sec is not None:
        run["rank_setup_timeout_sec"] = rank_setup_timeout_sec
    if mode == "files":
        run["rank_timeout_sec"] = rank_timeout_sec
    atomic_write_json(run_dir / "run.json", run)
    ready = {
        "run_id": run_id,
        "attempt_id": attempt_id,
        "manifest_sha256": manifest.sha256,
        "resolved_run_dir": str(run_dir),
        "status": "ready",
    }
    if shared_nonce is not None:
        ready["shared_nonce"] = shared_nonce
    atomic_write_json(run_dir / ".ready.json", ready)


def _execute_rank(
    *,
    run_id: str,
    run_dir: Path,
    manifest_sha256: str,
    context: _RankContext,
    items: Sequence[WorkItem],
    options: BatchFitOptions,
    visibility: str,
    setup_error: Mapping[str, str] | None,
    item_runner: Callable[
        [WorkItem, Path, BatchFitOptions], Mapping[str, Any]
    ],
    gpu_identity_loader: Callable[[str], Mapping[str, Any]],
    attempt_id: str | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    rank_error = dict(setup_error) if setup_error else None
    gpu = None
    if rank_error is None:
        try:
            gpu = json_mapping(
                gpu_identity_loader(options.backend),
                field="MPI GPU identity",
            )
        except Exception as exc:
            rank_error = error_payload(exc)
    successes = 0
    failures = 0
    write_errors: list[dict[str, str]] = []
    for item in items:
        item_start = time.perf_counter()
        item_started_at = timestamp_utc()
        identity = {
            "schema": "cuphoton.xpois.mpi-item/v1",
            "run_id": run_id,
            "manifest_sha256": manifest_sha256,
            "item_id": item.item_id,
            "rank": context.rank,
            "weight_bytes": item.weight_bytes,
            "requested_backend": options.backend,
            "started_at_utc": item_started_at,
        }
        if attempt_id is not None:
            identity["attempt_id"] = attempt_id
        record: dict[str, Any] = dict(identity)
        try:
            if rank_error is not None:
                raise RuntimeError(
                    "rank setup failed: "
                    f"{rank_error['type']}: {rank_error['message']}"
                )
            item_dir = run_dir / "items" / item.item_id
            metadata = json_mapping(
                item_runner(item, item_dir, options),
                field=f"metadata for item {item.item_id!r}",
            )
            _validate_success_item_output(item_dir, metadata)
            for field in ("run_dir", "summary_path"):
                if field in metadata:
                    metadata[field] = str(
                        Path(metadata[field]).relative_to(run_dir)
                    )
            record.update(metadata)
            record["status"] = "success"
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = error_payload(exc)
            notes = getattr(exc, "__notes__", ())
            if notes:
                record["error"]["notes"] = "\n".join(map(str, notes))
        record.update(identity)
        record.update(
            {
                "worker_seconds": time.perf_counter() - item_start,
                "completed_at_utc": timestamp_utc(),
            }
        )
        try:
            atomic_write_json(
                run_dir / "records" / f"{item.item_id}.json", record
            )
        except Exception as exc:
            write_errors.append(
                {"item_id": item.item_id, **error_payload(exc)}
            )
            failures += 1
        else:
            successes += record["status"] == "success"
            failures += record["status"] != "success"

    result = {
        "schema": "cuphoton.xpois.mpi-rank/v1",
        "run_id": run_id,
        "manifest_sha256": manifest_sha256,
        "rank": context.rank,
        "context": context.to_dict(),
        "assigned_item_ids": [item.item_id for item in items],
        "item_ids_sha256": _item_ids_sha256(items),
        "weight_bytes": sum(item.weight_bytes for item in items),
        "success_count": successes,
        "failed_count": failures,
        "status": (
            "success"
            if rank_error is None and not failures and not write_errors
            else "failed"
        ),
        "rank_wall_sec": time.perf_counter() - start,
        "error": rank_error,
        "record_write_errors": write_errors,
        "provenance": {
            "rank": context.rank,
            "local_rank": context.local_rank,
            "world_size": context.world_size,
            "hostname": context.host,
            "launcher": context.launcher,
            "pid": os.getpid(),
            "cuda_visible_devices": visibility,
            "allocated_cuda_visible_devices": os.environ.get(
                _ALLOCATED_DEVICES
            ),
            "gpu": gpu,
        },
    }
    if attempt_id is not None:
        result["attempt_id"] = attempt_id
    try:
        atomic_write_json(
            run_dir / "ranks" / f"rank-{context.rank:04d}.json", result
        )
    except Exception as exc:
        result["status"] = "failed"
        result["artifact_error"] = error_payload(exc)
    return result


def _mpi_aggregate(
    comm: Any,
    context: _RankContext,
    run_dir: Path,
    run_id: str,
    manifest: ImagePairManifest,
    options: BatchFitOptions,
    shards: Sequence[Sequence[WorkItem]],
    rank_result: Mapping[str, Any],
    started_at: str,
    start: float,
    api: _MPIAPI,
    *,
    rank_setup_timeout_sec: float = 600.0,
) -> MPIBatchResult | None:
    gathered = comm.gather(dict(rank_result), root=0)
    result = None
    decision = None
    if context.rank == 0:
        try:
            aggregation_errors = _wait_collective_artifacts(
                run_dir,
                shards,
                rank_setup_timeout_sec,
                rank_results=gathered,
            )
            result = _finalize(
                run_dir,
                run_id,
                manifest,
                options,
                shards,
                gathered,
                aggregation_errors,
                "mpi",
                started_at,
                start,
                api.mpi4py_version,
                api.library_version,
                rank_setup_timeout_sec=rank_setup_timeout_sec,
            )
            decision = {"status": result.status, "error": None}
        except Exception as exc:
            decision = {"status": "failed", "error": error_payload(exc)}
            _aggregate_error(run_dir, run_id, decision["error"])
    decision = comm.bcast(decision, root=0)
    if not isinstance(decision, Mapping):
        raise RuntimeError("MPI aggregate decision was not a mapping")
    if decision.get("error"):
        raise RuntimeError("cannot persist MPI aggregate")
    if context.rank != 0:
        if decision.get("status") != "success":
            raise RuntimeError(f"MPI batch {run_id!r} failed")
        return None
    assert result is not None
    return result


def _file_aggregate(
    context: _RankContext,
    run_dir: Path,
    run_id: str,
    attempt_id: str,
    manifest: ImagePairManifest,
    options: BatchFitOptions,
    shards: Sequence[Sequence[WorkItem]],
    timeout: float,
    started_at: str,
    start: float,
    *,
    rank_setup_timeout_sec: float | None = None,
) -> MPIBatchResult | None:
    if context.rank != 0:
        return None
    ranks, errors = _publish_completed_file_ranks(
        run_dir,
        run_id,
        attempt_id,
        manifest.sha256,
        shards,
        timeout,
        artifact_visibility_timeout=rank_setup_timeout_sec,
    )
    try:
        result = _finalize(
            run_dir,
            run_id,
            manifest,
            options,
            shards,
            ranks,
            errors,
            "files",
            started_at,
            start,
            None,
            None,
            attempt_id,
            timeout,
            rank_setup_timeout_sec,
        )
    except Exception as exc:
        error = error_payload(exc)
        _aggregate_error(run_dir, run_id, error)
        _set_attempt_terminal(
            run_dir,
            run_id,
            attempt_id,
            manifest.sha256,
            "failed",
            error,
        )
        raise RuntimeError("cannot persist MPI file aggregate") from exc
    try:
        _set_attempt_terminal(
            run_dir,
            run_id,
            attempt_id,
            manifest.sha256,
            result.status,
            None,
        )
    except Exception as exc:
        raise RuntimeError(
            "MPI file summary is committed but its terminal marker is not; "
            "rerun the exact --name and --attempt-id to repair it"
        ) from exc
    return result


def _finalize(
    run_dir: Path,
    run_id: str,
    manifest: ImagePairManifest,
    options: BatchFitOptions,
    shards: Sequence[Sequence[WorkItem]],
    rank_results: Sequence[Mapping[str, Any]],
    aggregation_errors: Sequence[Mapping[str, Any]],
    mode: str,
    started_at: str,
    start: float,
    mpi4py_version: str | None,
    library_version: str | None,
    attempt_id: str | None = None,
    rank_timeout_sec: float | None = None,
    rank_setup_timeout_sec: float | None = None,
) -> MPIBatchResult:
    records, record_errors = _read_mappings(run_dir / "records")
    expected_items = [item.item_id for shard in shards for item in shard]
    item_audit = audit_terminal_records(expected_items, records)
    record_errors.extend(
        _terminal_record_errors(
            run_id,
            manifest.sha256,
            options.backend,
            shards,
            records,
            attempt_id,
            run_dir,
            solver=options.solver,
        )
    )
    rank_audit = _audit_ranks(
        run_id,
        manifest.sha256,
        options.backend,
        shards,
        rank_results,
        records,
        attempt_id,
    )
    status = (
        "success"
        if item_audit["ok"]
        and not item_audit["failed_item_ids"]
        and rank_audit["ok"]
        and not rank_audit["setup_failed_ranks"]
        and not record_errors
        and not aggregation_errors
        else "failed"
    )
    summary = {
        "schema": "cuphoton.xpois.mpi-summary/v1",
        "executor": "mpi",
        "status": status,
        "run_id": run_id,
        "started_at_utc": started_at,
        "completed_at_utc": timestamp_utc(),
        "coordinator_wall_sec": time.perf_counter() - start,
        "coordinator_wall_definition": (
            "Root entry through terminal audits; summary commit and launcher "
            "exit are excluded."
        ),
        "manifest_sha256": manifest.sha256,
        "aggregation_mode": mode,
        "world_size": len(shards),
        "mpi4py_version": mpi4py_version,
        "mpi_library_version": library_version,
        "options": options.to_payload(),
        "shards": [
            {
                "rank": rank,
                "item_ids": [item.item_id for item in shard],
                "weight_bytes": sum(item.weight_bytes for item in shard),
                "item_ids_sha256": _item_ids_sha256(shard),
            }
            for rank, shard in enumerate(shards)
        ],
        "rank_results": sorted(
            (dict(item) for item in rank_results),
            key=_rank_result_sort_key,
        ),
        "rank_result_audit": rank_audit,
        "terminal_record_audit": item_audit,
        "terminal_record_errors": record_errors,
        "aggregation_errors": [dict(item) for item in aggregation_errors],
        "launcher_exit_note": (
            "Collective completion is not launcher process-exit proof."
        ),
    }
    if rank_setup_timeout_sec is not None:
        summary["rank_setup_timeout_sec"] = rank_setup_timeout_sec
    if mode == "files":
        summary["attempt_id"] = attempt_id
        summary["rank_timeout_sec"] = rank_timeout_sec
    summary_path = run_dir / "summary.json"
    atomic_write_json(summary_path, summary)
    return MPIBatchResult(run_id, run_dir, summary_path, status, summary)


def _audit_ranks(
    run_id: str,
    manifest_sha256: str,
    backend: str,
    shards: Sequence[Sequence[WorkItem]],
    results: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    attempt_id: str | None = None,
) -> dict[str, Any]:
    expected = set(range(len(shards)))
    ranks = [_integer(item.get("rank")) for item in results]
    valid_ranks = [rank for rank in ranks if rank is not None]
    duplicates = sorted(
        rank for rank, count in Counter(valid_ranks).items() if count > 1
    )
    missing = sorted(expected - set(valid_ranks))
    unexpected = sorted(set(valid_ranks) - expected)
    item_ranks = {
        str(record.get("item_id")): _integer(record.get("rank"))
        for record in records
    }
    records_by_rank: dict[int, list[Mapping[str, Any]]] = {}
    for record in records:
        record_rank = _integer(record.get("rank"))
        if record_rank is not None:
            records_by_rank.setdefault(record_rank, []).append(record)
    mismatched: list[dict[str, Any]] = []
    failed_candidates: set[int] = set()
    setup_failed_candidates: set[int] = set()
    write_failed_candidates: list[dict[str, Any]] = []
    physical: list[tuple[int, str, frozenset[tuple[str, str]]]] = []
    for result, rank in zip(results, ranks):
        if rank not in expected:
            continue
        shard = shards[rank]
        assigned = [item.item_id for item in shard]
        assigned_set = set(assigned)
        write_errors = result.get("record_write_errors")
        write_error_ids: set[str] = set()
        write_errors_valid = isinstance(write_errors, list)
        if write_errors_valid:
            for error in write_errors:
                if (
                    not isinstance(error, Mapping)
                    or not isinstance(error.get("item_id"), str)
                    or error["item_id"] not in assigned_set
                    or error["item_id"] in write_error_ids
                    or not isinstance(error.get("type"), str)
                    or not error["type"]
                    or not isinstance(error.get("message"), str)
                ):
                    write_errors_valid = False
                    break
                write_error_ids.add(error["item_id"])
        if not write_errors_valid:
            write_error_ids.clear()
        assigned_records = [
            record
            for record in records_by_rank.get(rank, ())
            if isinstance(record.get("item_id"), str)
            and record.get("item_id") in assigned_set
        ]
        success_count = sum(
            record.get("status") == "success"
            for record in assigned_records
            if record["item_id"] not in write_error_ids
        )
        # A directory fsync failure can leave a visible record. It remains
        # a failed write, rather than a successfully persisted outcome.
        failed_count = len(write_error_ids) + sum(
            record.get("status") == "failed"
            for record in assigned_records
            if record["item_id"] not in write_error_ids
        )
        success_count_matches = (
            _integer(result.get("success_count")) == success_count
        )
        failed_count_matches = (
            _integer(result.get("failed_count")) == failed_count
        )
        terminal_assignment = all(
            item_ranks.get(item_id) == rank
            or (item_id in write_error_ids and item_id not in item_ranks)
            for item_id in assigned
        )
        rank_error = result.get("error")
        expected_setup_error = (
            {
                "type": "RuntimeError",
                "message": (
                    "rank setup failed: "
                    f"{rank_error['type']}: {rank_error['message']}"
                ),
            }
            if _valid_error(rank_error)
            else None
        )
        setup_records_match = (
            expected_setup_error is not None
            and sum(
                record["item_id"] not in write_error_ids
                for record in assigned_records
            )
            == len(assigned) - len(write_error_ids)
            and all(
                record.get("status") == "failed"
                and record.get("error") == expected_setup_error
                for record in assigned_records
                if record["item_id"] not in write_error_ids
            )
        )
        setup_failure = (
            result.get("status") == "failed"
            and expected_setup_error is not None
            and write_errors_valid
            and result.get("artifact_error") is None
            and success_count == 0
            and failed_count == len(shard)
            and success_count_matches
            and failed_count_matches
            and terminal_assignment
            and setup_records_match
        )
        workload_failure = (
            result.get("status") == "failed"
            and result.get("error") is None
            and result.get("record_write_errors") == []
            and result.get("artifact_error") is None
            and failed_count > 0
            and success_count_matches
            and failed_count_matches
            and terminal_assignment
        )
        fields: set[str] = set()
        checks = {
            "schema": result.get("schema") == "cuphoton.xpois.mpi-rank/v1",
            "run_id": result.get("run_id") == run_id,
            "manifest_sha256": result.get("manifest_sha256")
            == manifest_sha256,
            "assigned_item_ids": result.get("assigned_item_ids") == assigned,
            "item_ids_sha256": result.get("item_ids_sha256")
            == _item_ids_sha256(shard),
            "weight_bytes": _integer(result.get("weight_bytes"))
            == sum(item.weight_bytes for item in shard),
            "success_count": success_count_matches,
            "failed_count": failed_count_matches,
            "rank_wall_sec": _finite_nonnegative(result.get("rank_wall_sec")),
            "status": result.get("status")
            == ("failed" if failed_count or setup_failure else "success"),
            "error": setup_failure or rank_error is None,
            "record_write_errors": write_errors_valid,
            "artifact_error": result.get("artifact_error") is None,
            "terminal_assignment": terminal_assignment,
        }
        if attempt_id is not None:
            checks["attempt_id"] = result.get("attempt_id") == attempt_id
        fields.update(name for name, ok in checks.items() if not ok)
        context = result.get("context")
        context_identity = _context_identity(context, rank, len(shards))
        if context_identity is None:
            fields.add("context")
        provenance = result.get("provenance")
        ids = (
            None
            if context_identity is None
            else _provenance_ids(
                provenance,
                rank,
                len(shards),
                backend,
                *context_identity,
                require_missing_gpu=setup_failure,
            )
        )
        if ids is None:
            fields.add("provenance")
        elif ids:
            assert isinstance(provenance, Mapping)
            physical.append((rank, str(provenance["hostname"]), ids))
        if fields:
            mismatched.append({"rank": rank, "fields": sorted(fields)})
        elif write_error_ids:
            write_failed_candidates.append(
                {"rank": rank, "record_write_errors": write_errors}
            )
        elif setup_failure:
            setup_failed_candidates.add(rank)
        elif workload_failure:
            failed_candidates.add(rank)
    duplicate_gpu_ranks: set[int] = set()
    incomparable_gpu_ranks: set[int] = set()
    for index, (rank, host, ids) in enumerate(physical):
        for other_rank, other_host, other_ids in physical[index + 1 :]:
            if rank == other_rank:
                continue
            comparison = classify_physical_gpu_pair(
                dict(ids),
                dict(other_ids),
                same_host=_same_host(host, other_host),
            )
            if comparison == "duplicate":
                duplicate_gpu_ranks.update((rank, other_rank))
            elif comparison == "incomparable":
                incomparable_gpu_ranks.update((rank, other_rank))
    invalid_gpu_ranks = duplicate_gpu_ranks | incomparable_gpu_ranks
    for rank in sorted(invalid_gpu_ranks):
        entry = next(
            (item for item in mismatched if item["rank"] == rank), None
        )
        if entry is None:
            mismatched.append({"rank": rank, "fields": ["provenance"]})
        else:
            entry["fields"] = sorted(set(entry["fields"]) | {"provenance"})
    mismatched_rank_ids = {item["rank"] for item in mismatched}
    failed_ranks = sorted(failed_candidates - mismatched_rank_ids)
    setup_failed_ranks = sorted(setup_failed_candidates - mismatched_rank_ids)
    write_failed_ranks = [
        item
        for item in write_failed_candidates
        if item["rank"] not in mismatched_rank_ids
    ]
    return {
        "ok": not missing
        and not duplicates
        and not unexpected
        and all(rank is not None for rank in ranks)
        and not mismatched
        and not write_failed_ranks,
        "expected_count": len(shards),
        "observed_count": len(results),
        "missing_ranks": missing,
        "duplicate_ranks": duplicates,
        "unexpected_ranks": unexpected,
        "duplicate_physical_gpu_ranks": sorted(duplicate_gpu_ranks),
        "incomparable_physical_gpu_ranks": sorted(incomparable_gpu_ranks),
        "failed_ranks": failed_ranks,
        "setup_failed_ranks": setup_failed_ranks,
        "write_failed_ranks": sorted(
            write_failed_ranks, key=lambda item: item["rank"]
        ),
        "mismatched_ranks": sorted(mismatched, key=lambda item: item["rank"]),
    }


def _terminal_record_errors(
    run_id: str,
    manifest_sha256: str,
    backend: str,
    shards: Sequence[Sequence[WorkItem]],
    records: Sequence[Mapping[str, Any]],
    attempt_id: str | None = None,
    run_dir: Path | None = None,
    *,
    solver: str,
) -> list[dict[str, str]]:
    expected = {
        item.item_id: (rank, item.weight_bytes)
        for rank, shard in enumerate(shards)
        for item in shard
    }
    errors: list[dict[str, str]] = []
    for index, record in enumerate(records):
        item_id_value = record.get("item_id")
        item_id = item_id_value if isinstance(item_id_value, str) else ""
        assignment = expected.get(item_id)
        invalid: list[str] = []
        if record.get("schema") != "cuphoton.xpois.mpi-item/v1":
            invalid.append("schema")
        if record.get("run_id") != run_id:
            invalid.append("run_id")
        if record.get("manifest_sha256") != manifest_sha256:
            invalid.append("manifest_sha256")
        if assignment is None:
            invalid.append("item_id")
        else:
            expected_rank, expected_weight_bytes = assignment
            if _integer(record.get("rank")) != expected_rank:
                invalid.append("rank")
            if _integer(record.get("weight_bytes")) != expected_weight_bytes:
                invalid.append("weight_bytes")
        if record.get("requested_backend") != backend:
            invalid.append("requested_backend")
        if attempt_id is not None and record.get("attempt_id") != attempt_id:
            invalid.append("attempt_id")
        started_at = _parse_timestamp(record.get("started_at_utc"))
        completed_at = _parse_timestamp(record.get("completed_at_utc"))
        if started_at is None:
            invalid.append("started_at_utc")
        if completed_at is None or (
            started_at is not None and completed_at < started_at
        ):
            invalid.append("completed_at_utc")
        if not _finite_nonnegative(record.get("worker_seconds")):
            invalid.append("worker_seconds")
        status = record.get("status")
        if status == "success":
            if record.get("solver") != solver:
                invalid.append("solver")
            expected_item_dir = f"items/{item_id}"
            if record.get("run_dir") != expected_item_dir:
                invalid.append("run_dir")
            expected_summary = f"{expected_item_dir}/summary.json"
            if record.get("summary_path") != expected_summary:
                invalid.append("summary_path")
            if record.get("backend") != backend:
                invalid.append("backend")
            if (
                not isinstance(record.get("device"), str)
                or not record["device"]
            ):
                invalid.append("device")
            if not isinstance(record.get("runtime"), Mapping):
                invalid.append("runtime")
            for field in ("timings_sec", "wall_sec"):
                if not _valid_timings(record.get(field)):
                    invalid.append(field)
            if record.get("error") is not None:
                invalid.append("error")
            if run_dir is not None:
                item_dir = run_dir / "items" / item_id
                if not _real_directory(item_dir):
                    invalid.append("run_dir")
                if not _regular_file(item_dir / "summary.json"):
                    invalid.append("summary_path")
        elif status == "failed":
            if not _valid_error(record.get("error")):
                invalid.append("error")
        else:
            invalid.append("status")
        if invalid:
            errors.append(
                {
                    "item_id": item_id,
                    "type": "InvalidTerminalRecord",
                    "message": (
                        f"record {index} has invalid field(s): "
                        + ", ".join(sorted(set(invalid)))
                    ),
                }
            )
    return errors


def _context_identity(
    value: Any, rank: int, world_size: int
) -> tuple[int, str, str] | None:
    if not isinstance(value, Mapping):
        return None
    local_rank = _integer(value.get("local_rank"))
    host = value.get("host")
    launcher = value.get("launcher")
    if (
        _integer(value.get("rank")) != rank
        or local_rank is None
        or local_rank < 0
        or local_rank >= world_size
        or _integer(value.get("world_size")) != world_size
        or not isinstance(host, str)
        or not host
        or not isinstance(launcher, str)
        or not launcher
    ):
        return None
    return local_rank, host, launcher


def _provenance_ids(
    value: Any,
    rank: int,
    world_size: int,
    backend: str,
    local_rank: int,
    host: str,
    launcher: str,
    *,
    require_missing_gpu: bool = False,
) -> frozenset[tuple[str, str]] | None:
    if not isinstance(value, Mapping):
        return None
    if (
        _integer(value.get("rank")) != rank
        or _integer(value.get("local_rank")) != local_rank
        or _integer(value.get("world_size")) != world_size
        or value.get("hostname") != host
        or value.get("launcher") != launcher
        or not isinstance(value.get("cuda_visible_devices"), str)
    ):
        return None
    try:
        visible_devices = _visible_tokens(value["cuda_visible_devices"])
    except RuntimeError:
        return None
    if len(visible_devices) != 1:
        return None
    gpu = value.get("gpu")
    if require_missing_gpu:
        return frozenset() if gpu is None else None
    expected_backend = "cupy" if backend in {"cupy", "cutile"} else backend
    if (
        not isinstance(gpu, Mapping)
        or gpu.get("backend") != expected_backend
        or _integer(gpu.get("device_index")) != 0
        or gpu.get("identity_error") is not None
    ):
        return None
    ids = frozenset(
        (field, identifier.strip().casefold())
        for field in ("uuid", "pci_bus_id")
        if isinstance((identifier := gpu.get(field)), str)
        and identifier.strip()
    )
    return ids or None


def _validate_success_item_output(
    item_dir: Path, metadata: Mapping[str, Any]
) -> None:
    if not _real_directory(item_dir):
        raise ValueError("item runner did not create a real output directory")
    try:
        recorded_dir = Path(metadata["run_dir"])
        summary_path = Path(metadata["summary_path"])
    except (KeyError, TypeError) as exc:
        raise ValueError("item runner did not report output paths") from exc
    if recorded_dir != item_dir:
        raise ValueError("item runner reported a different output directory")
    if summary_path != item_dir / "summary.json":
        raise ValueError("item runner reported a different summary path")
    if not _regular_file(summary_path):
        raise ValueError("item runner did not create a regular summary file")


def _real_directory(path: Path) -> bool:
    try:
        return stat.S_ISDIR(path.lstat().st_mode)
    except OSError:
        return False


def _valid_timings(value: Any) -> bool:
    return isinstance(value, Mapping) and all(
        isinstance(phase, str)
        and bool(phase)
        and _finite_nonnegative(elapsed)
        for phase, elapsed in value.items()
    )


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed


def _valid_error(value: Any) -> bool:
    return isinstance(value, Mapping) and all(
        isinstance(value.get(field), str) and bool(value[field])
        for field in ("type", "message")
    )


def _same_json_value(left: Any, right: Any) -> bool:
    try:
        return json.dumps(
            left,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ) == json.dumps(
            right,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError):
        return False


def _read_mappings(
    directory: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for path in sorted(directory.glob("*.json")):
        try:
            records.append(_read_regular_mapping(path))
        except Exception as exc:
            errors.append({"path": path.name, **error_payload(exc)})
    return records, errors


def _wait_collective_artifacts(
    run_dir: Path,
    shards: Sequence[Sequence[WorkItem]],
    timeout: float,
    *,
    rank_results: Sequence[Any] | None = None,
) -> list[dict[str, Any]]:
    expected = {
        f"ranks/rank-{rank:04d}.json": (
            run_dir / "ranks" / f"rank-{rank:04d}.json"
        )
        for rank in range(len(shards))
    }
    expected.update(
        {
            f"records/{item.item_id}.json": (
                run_dir / "records" / f"{item.item_id}.json"
            )
            for shard in shards
            for item in shard
        }
    )
    declared_errors: list[dict[str, Any]] = []
    if rank_results is not None:
        failed_paths, declared_errors = _declared_collective_write_failures(
            shards, rank_results
        )
        for path in failed_paths:
            expected.pop(path, None)
    deadline = time.monotonic() + timeout
    delay = _POLL_INITIAL_SEC
    missing: list[str] = []
    record_errors: list[dict[str, Any]] = []
    missing_summaries: list[str] = []
    rank_errors: list[dict[str, Any]] = []
    while True:
        missing = []
        invalid_artifacts: list[dict[str, Any]] = []
        for label, path in expected.items():
            try:
                mode = path.lstat().st_mode
            except OSError:
                missing.append(label)
                continue
            if not stat.S_ISREG(mode):
                invalid_artifacts.append(
                    {
                        "path": label,
                        "type": "InvalidArtifact",
                        "message": (
                            "collective artifact is not a regular file"
                        ),
                    }
                )
        if invalid_artifacts:
            return [*declared_errors, *invalid_artifacts]
        record_errors = []
        missing_summaries = []
        rank_errors = []
        if not missing:
            if rank_results is not None:
                rank_errors = _collective_rank_artifact_errors(
                    run_dir, expected, rank_results
                )
                permanent_rank_errors = [
                    {
                        key: value
                        for key, value in error.items()
                        if key != "retryable"
                    }
                    for error in rank_errors
                    if not error.get("retryable", False)
                ]
                if permanent_rank_errors:
                    return [*declared_errors, *permanent_rank_errors]
                rank_errors = [
                    {
                        key: value
                        for key, value in error.items()
                        if key != "retryable"
                    }
                    for error in rank_errors
                ]
            if not rank_errors:
                for shard in shards:
                    for item in shard:
                        label = f"records/{item.item_id}.json"
                        if label not in expected:
                            continue
                        path = run_dir / "records" / f"{item.item_id}.json"
                        try:
                            record = _read_regular_mapping(path)
                        except Exception as exc:
                            record_errors.append(
                                {
                                    "path": str(path.relative_to(run_dir)),
                                    **error_payload(exc),
                                }
                            )
                            continue
                        if record.get("status") == "success":
                            label = f"items/{item.item_id}/summary.json"
                            try:
                                mode = (run_dir / label).lstat().st_mode
                            except OSError:
                                missing_summaries.append(label)
                                continue
                            if not stat.S_ISREG(mode):
                                invalid_artifacts.append(
                                    {
                                        "path": label,
                                        "type": "InvalidArtifact",
                                        "message": (
                                            "collective artifact is not a "
                                            "regular file"
                                        ),
                                    }
                                )
                if invalid_artifacts:
                    return [*declared_errors, *invalid_artifacts]
            if (
                not rank_errors
                and not record_errors
                and not missing_summaries
            ):
                return declared_errors
        if time.monotonic() >= deadline:
            break
        delay = _shared_filesystem_wait(deadline, delay)
    errors: list[dict[str, Any]] = [*declared_errors]
    errors.extend(rank_errors)
    errors.extend(
        {
            "type": "TimeoutError",
            "message": f"collective artifact did not become visible: {path}",
        }
        for path in (*missing, *missing_summaries)
    )
    errors.extend(record_errors)
    return errors


def _collective_rank_artifact_errors(
    run_dir: Path,
    expected: Mapping[str, Path],
    rank_results: Sequence[Any],
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for rank, reported in enumerate(rank_results):
        label = f"ranks/rank-{rank:04d}.json"
        if label not in expected:
            continue
        path = run_dir / label
        try:
            mode = path.lstat().st_mode
            if not stat.S_ISREG(mode):
                raise ValueError(
                    f"staged JSON is not a regular file: {path.name}"
                )
            durable = _read_strict_mapping(path)
        except Exception as exc:
            error = {"rank": rank, "path": label, **error_payload(exc)}
            if isinstance(exc, OSError):
                error["retryable"] = True
            errors.append(error)
            continue
        matches_reported = False
        if isinstance(reported, Mapping):
            try:
                normalized_reported = json_mapping(
                    reported, field=f"gathered rank result {rank}"
                )
                matches_reported = _same_json_value(
                    durable, normalized_reported
                )
            except (TypeError, ValueError):
                pass
        if not matches_reported:
            errors.append(
                {
                    "rank": rank,
                    "path": label,
                    "type": "InvalidRankArtifact",
                    "message": (
                        "durable rank result differs from gathered result"
                    ),
                }
            )
    return errors


def _declared_collective_write_failures(
    shards: Sequence[Sequence[WorkItem]], rank_results: Sequence[Any]
) -> tuple[set[str], list[dict[str, Any]]]:
    failed_paths: set[str] = set()
    errors: list[dict[str, Any]] = []
    for rank, shard in enumerate(shards):
        assigned = {item.item_id for item in shard}
        result = rank_results[rank] if rank < len(rank_results) else None
        if not isinstance(result, Mapping):
            labels = {
                f"ranks/rank-{rank:04d}.json",
                *(f"records/{item_id}.json" for item_id in assigned),
            }
            failed_paths.update(labels)
            errors.append(
                {
                    "rank": rank,
                    "type": "InvalidRankResult",
                    "message": "rank did not report its artifact writes",
                }
            )
            continue
        artifact_error = result.get("artifact_error")
        if artifact_error is not None:
            label = f"ranks/rank-{rank:04d}.json"
            failed_paths.add(label)
            payload = (
                {
                    "type": artifact_error["type"],
                    "message": artifact_error["message"],
                }
                if _valid_error(artifact_error)
                else {
                    "type": "InvalidArtifactError",
                    "message": "rank reported an invalid artifact error",
                }
            )
            errors.append({"rank": rank, "path": label, **payload})
        failed_items, record_errors, _ = _declared_record_write_failures(
            rank, shard, result
        )
        failed_paths.update(
            f"records/{item_id}.json" for item_id in failed_items
        )
        errors.extend(record_errors)
    return failed_paths, errors


def _declared_record_write_failures(
    rank: int,
    shard: Sequence[WorkItem],
    result: Mapping[str, Any],
) -> tuple[set[str], list[dict[str, Any]], bool]:
    assigned = {item.item_id for item in shard}
    write_errors = result.get("record_write_errors")
    if write_errors == []:
        return set(), [], True
    if not isinstance(write_errors, list) or any(
        not isinstance(error, Mapping) for error in write_errors
    ):
        return (
            assigned,
            [
                {
                    "rank": rank,
                    "type": "InvalidRecordWriteErrors",
                    "message": "rank reported invalid record-write errors",
                }
            ],
            False,
        )
    item_ids = [error.get("item_id") for error in write_errors]
    if (
        any(
            not isinstance(item_id, str) or item_id not in assigned
            for item_id in item_ids
        )
        or len(set(item_ids)) != len(item_ids)
        or any(not _valid_error(error) for error in write_errors)
    ):
        return (
            assigned,
            [
                {
                    "rank": rank,
                    "type": "InvalidRecordWriteError",
                    "message": "rank reported an invalid record-write error",
                }
            ],
            False,
        )
    errors = [
        {
            "rank": rank,
            "path": f"records/{error['item_id']}.json",
            "type": error["type"],
            "message": error["message"],
        }
        for error in write_errors
    ]
    return set(item_ids), errors, True


def _source_identified_record_write_errors(
    completion_errors: Sequence[Mapping[str, Any]],
    artifact_errors: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    positions: dict[tuple[Any, ...], int] = {}
    for source, reported in (
        ("completion_receipt", completion_errors),
        ("rank_artifact", artifact_errors),
    ):
        for error in reported:
            key = (
                error["rank"],
                error["path"],
                error["type"],
                error["message"],
            )
            position = positions.get(key)
            if position is None:
                positions[key] = len(merged)
                merged.append({**error, "evidence_sources": [source]})
            else:
                merged[position]["evidence_sources"].append(source)
    return merged


def _prepare_file_rank_staging(
    rank_dir: Path,
    *,
    parent_visibility_timeout_sec: float = 0.0,
) -> None:
    deadline = time.monotonic() + parent_visibility_timeout_sec
    delay = _POLL_INITIAL_SEC
    while True:
        try:
            rank_dir.mkdir(parents=False, exist_ok=False)
            break
        except FileNotFoundError:
            if time.monotonic() >= deadline:
                raise
            delay = _shared_filesystem_wait(deadline, delay)
    for name in ("items", "ranks", "records"):
        (rank_dir / name).mkdir()


def _complete_file_rank(
    rank_dir: Path,
    run_id: str,
    attempt_id: str,
    manifest_sha256: str,
    rank: int,
    result: Mapping[str, Any],
) -> None:
    atomic_write_json(
        rank_dir / ".complete.json",
        {
            "run_id": run_id,
            "attempt_id": attempt_id,
            "manifest_sha256": manifest_sha256,
            "rank": rank,
            "status": result.get("status"),
            "artifact_error": result.get("artifact_error"),
            "record_write_errors": result.get("record_write_errors"),
            "completed_at_utc": timestamp_utc(),
        },
    )


def _read_regular_mapping(path: Path) -> dict[str, Any]:
    if not _regular_file(path):
        raise ValueError(f"staged JSON is not a regular file: {path.name}")
    return _read_strict_mapping(path)


def _read_strict_mapping(path: Path) -> dict[str, Any]:
    return json_mapping(
        read_json_mapping(path), field=f"JSON object in {path}"
    )


def _staged_rank_publications(
    rank_dir: Path,
    run_dir: Path,
    rank: int,
    shard: Sequence[WorkItem],
    declared_failed_items: set[str] | None = None,
) -> tuple[dict[str, Any], list[tuple[Path, Path]], list[dict[str, Any]]]:
    if not _real_directory(rank_dir):
        raise ValueError("staged rank path is not a real directory")
    for name in ("items", "ranks", "records"):
        if not _real_directory(rank_dir / name):
            raise ValueError(f"staged {name} path is not a real directory")
    rank_path = rank_dir / "ranks" / f"rank-{rank:04d}.json"
    rank_result = _read_regular_mapping(rank_path)
    if rank_result.get("status") not in {"success", "failed"}:
        raise ValueError("staged rank status is invalid")
    rank_failed_items, declared_errors, valid = (
        _declared_record_write_failures(rank, shard, rank_result)
    )
    if not valid:
        raise ValueError(
            f"{declared_errors[0]['type']}: {declared_errors[0]['message']}"
        )
    failed_items = set(declared_failed_items or ())
    failed_items.update(rank_failed_items)
    publications: list[tuple[Path, Path]] = []
    for item in shard:
        if item.item_id in failed_items:
            continue
        record_path = rank_dir / "records" / f"{item.item_id}.json"
        record = _read_regular_mapping(record_path)
        if record.get("status") not in {"success", "failed"}:
            raise ValueError(
                f"staged item {item.item_id!r} status is invalid"
            )
        item_dir = rank_dir / "items" / item.item_id
        if os.path.lexists(item_dir):
            tree_error = _real_tree_error(item_dir)
            if tree_error is not None:
                raise ValueError(
                    f"staged item {item.item_id!r} is invalid: {tree_error}"
                )
            publications.append((item_dir, run_dir / "items" / item.item_id))
        elif record.get("status") == "success":
            raise ValueError(
                f"successful staged item {item.item_id!r} has no output"
            )
        if record.get("status") == "success" and not _regular_file(
            item_dir / "summary.json"
        ):
            raise ValueError(
                f"successful staged item {item.item_id!r} has no summary"
            )
        publications.append(
            (record_path, run_dir / "records" / f"{item.item_id}.json")
        )
    publications.append(
        (rank_path, run_dir / "ranks" / f"rank-{rank:04d}.json")
    )
    return rank_result, publications, declared_errors


def _wait_staged_rank_publications(
    rank_dir: Path,
    run_dir: Path,
    rank: int,
    shard: Sequence[WorkItem],
    deadline: float,
    declared_failed_items: set[str] | None = None,
) -> tuple[dict[str, Any], list[tuple[Path, Path]], list[dict[str, Any]]]:
    delay = _POLL_INITIAL_SEC
    missing = _missing_staged_rank_paths(
        rank_dir, rank, shard, declared_failed_items
    )
    while missing:
        if time.monotonic() >= deadline:
            raise FileNotFoundError(
                "completed rank artifacts did not become visible: "
                + ", ".join(missing)
            )
        delay = _shared_filesystem_wait(deadline, delay)
        missing = _missing_staged_rank_paths(
            rank_dir, rank, shard, declared_failed_items
        )
    return _staged_rank_publications(
        rank_dir,
        run_dir,
        rank,
        shard,
        declared_failed_items,
    )


def _missing_staged_rank_paths(
    rank_dir: Path,
    rank: int,
    shard: Sequence[WorkItem],
    declared_failed_items: set[str] | None = None,
) -> list[str]:
    directories = [
        (rank_dir, "rank"),
        *((rank_dir / name, name) for name in ("items", "ranks", "records")),
    ]
    for path, label in directories:
        if os.path.lexists(path) and not _real_directory(path):
            raise ValueError(f"staged {label} path is not a real directory")
    rank_path = rank_dir / "ranks" / f"rank-{rank:04d}.json"
    record_paths = {
        item.item_id: rank_dir / "records" / f"{item.item_id}.json"
        for item in shard
    }
    mappings: dict[Path, dict[str, Any]] = {}
    if os.path.lexists(rank_path):
        if not _regular_file(rank_path):
            raise ValueError(
                f"staged JSON is not a regular file: {rank_path.name}"
            )
        mappings[rank_path] = _read_regular_mapping(rank_path)
    rank_result = mappings.get(rank_path)
    failed_items = set(declared_failed_items or ())
    if rank_result is not None and rank_result.get("status") not in {
        "success",
        "failed",
    }:
        raise ValueError("staged rank status is invalid")
    if rank_result is not None:
        rank_failed_items, declared_errors, valid = (
            _declared_record_write_failures(rank, shard, rank_result)
        )
        if not valid:
            raise ValueError(
                f"{declared_errors[0]['type']}: "
                f"{declared_errors[0]['message']}"
            )
        failed_items.update(rank_failed_items)
    visible_record_paths = {
        item_id: path
        for item_id, path in record_paths.items()
        if item_id not in failed_items
    }
    for item_id, record_path in visible_record_paths.items():
        if os.path.lexists(record_path) and not _regular_file(record_path):
            raise ValueError(
                f"staged JSON is not a regular file: {record_path.name}"
            )
        if os.path.lexists(record_path):
            mappings[record_path] = _read_regular_mapping(record_path)
        record = mappings.get(record_path)
        if record is not None and record.get("status") not in {
            "success",
            "failed",
        }:
            raise ValueError(f"staged item {item_id!r} status is invalid")
    expected = [
        *(path for path, _ in directories),
        rank_path,
        *visible_record_paths.values(),
    ]
    missing = [
        str(path.relative_to(rank_dir))
        for path in expected
        if not os.path.lexists(path)
    ]
    for item in shard:
        if item.item_id in failed_items:
            continue
        record = mappings.get(record_paths[item.item_id])
        if record is None or record.get("status") != "success":
            continue
        item_dir = rank_dir / "items" / item.item_id
        if not os.path.lexists(item_dir):
            missing.append(str(item_dir.relative_to(rank_dir)))
            continue
        if not _real_directory(item_dir):
            raise ValueError(
                f"staged item {item.item_id!r} path is not a real directory"
            )
        summary_path = item_dir / "summary.json"
        if not os.path.lexists(summary_path):
            missing.append(str(summary_path.relative_to(rank_dir)))
        elif not _regular_file(summary_path):
            raise ValueError(
                f"staged JSON is not a regular file: {summary_path.name}"
            )
    return list(dict.fromkeys(missing))


def _real_tree_error(path: Path) -> str | None:
    if not _real_directory(path):
        return "root is not a real directory"
    pending = [path]
    while pending:
        directory = pending.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError as exc:
            return f"cannot scan directory: {exc}"
        for entry in entries:
            if entry.is_symlink():
                return f"symbolic link is not allowed: {entry.name}"
            if entry.is_dir(follow_symlinks=False):
                pending.append(Path(entry.path))
            elif not entry.is_file(follow_symlinks=False):
                return f"non-regular artifact is not allowed: {entry.name}"
    return None


def _file_rank_finished_or_invalid(
    rank_dir: Path, completion_path: Path
) -> bool:
    if os.path.lexists(completion_path):
        return True
    return os.path.lexists(rank_dir) and not _real_directory(rank_dir)


def _publish_completed_file_ranks(
    run_dir: Path,
    run_id: str,
    attempt_id: str,
    manifest_sha256: str,
    shards: Sequence[Sequence[WorkItem]],
    timeout: float,
    *,
    artifact_visibility_timeout: float | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rank_dirs = {
        rank: _file_rank_staging_dir(run_dir, run_id, attempt_id, rank)
        for rank in range(len(shards))
    }
    expected = {
        rank: rank_dir / ".complete.json"
        for rank, rank_dir in rank_dirs.items()
    }
    deadline = time.monotonic() + timeout
    delay = _POLL_INITIAL_SEC
    while time.monotonic() < deadline and not all(
        _file_rank_finished_or_invalid(rank_dirs[rank], path)
        for rank, path in expected.items()
    ):
        delay = _shared_filesystem_wait(deadline, delay)
    completed = {
        rank: path
        for rank, path in expected.items()
        if _file_rank_finished_or_invalid(rank_dirs[rank], path)
    }
    visibility_deadline = time.monotonic() + (
        timeout
        if artifact_visibility_timeout is None
        else artifact_visibility_timeout
    )
    errors: list[dict[str, Any]] = [
        {
            "rank": rank,
            "type": "TimeoutError",
            "message": "rank completion did not appear before timeout",
        }
        for rank in range(len(shards))
        if rank not in completed
    ]
    for rank, completion_path in completed.items():
        expected_identity = {
            "run_id": run_id,
            "attempt_id": attempt_id,
            "manifest_sha256": manifest_sha256,
            "rank": rank,
        }
        try:
            completion = _read_regular_mapping(completion_path)
        except Exception as exc:
            errors.append({"rank": rank, **error_payload(exc)})
            continue
        invalid = [
            field
            for field, value in expected_identity.items()
            if not _same_json_value(completion.get(field), value)
        ]
        completion_status = completion.get("status")
        valid_status = isinstance(completion_status, str) and (
            completion_status in {"success", "failed"}
        )
        if not valid_status:
            invalid.append("status")
        artifact_error = completion.get("artifact_error")
        if "artifact_error" not in completion or (
            artifact_error is not None and not _valid_error(artifact_error)
        ):
            invalid.append("artifact_error")
        (
            completion_failed_items,
            completion_record_errors,
            valid_record_errors,
        ) = _declared_record_write_failures(rank, shards[rank], completion)
        if not valid_record_errors:
            invalid.append("record_write_errors")
        if (
            artifact_error is not None or completion_record_errors
        ) and completion_status != "failed":
            invalid.append("status")
        if invalid:
            if _valid_error(artifact_error):
                errors.append(
                    {
                        "rank": rank,
                        "path": f"ranks/rank-{rank:04d}.json",
                        "type": artifact_error["type"],
                        "message": artifact_error["message"],
                    }
                )
            if valid_record_errors:
                errors.extend(completion_record_errors)
            errors.append(
                {
                    "rank": rank,
                    "type": "InvalidRankCompletion",
                    "message": "invalid field(s): "
                    + ", ".join(sorted(set(invalid))),
                }
            )
            continue
        if artifact_error is not None:
            errors.append(
                {
                    "rank": rank,
                    "path": f"ranks/rank-{rank:04d}.json",
                    "type": artifact_error["type"],
                    "message": artifact_error["message"],
                }
            )
            errors.extend(completion_record_errors)
            continue
        try:
            rank_result, publications, declared_errors = (
                _wait_staged_rank_publications(
                    rank_dirs[rank],
                    run_dir,
                    rank,
                    shards[rank],
                    visibility_deadline,
                    completion_failed_items,
                )
            )
        except Exception as exc:
            errors.extend(completion_record_errors)
            errors.append({"rank": rank, **error_payload(exc)})
            continue
        invalid = []
        if rank_result.get("status") != completion_status:
            invalid.append("status")
        if not _same_json_value(
            rank_result.get("artifact_error"), artifact_error
        ):
            invalid.append("artifact_error")
        if not _same_json_value(
            rank_result.get("record_write_errors"),
            completion.get("record_write_errors"),
        ):
            invalid.append("record_write_errors")
        if (
            rank_result.get("status") == "success"
            and rank_result.get("error") is not None
        ):
            invalid.append("error")
        if invalid:
            if "record_write_errors" in invalid:
                errors.extend(
                    _source_identified_record_write_errors(
                        completion_record_errors, declared_errors
                    )
                )
            else:
                errors.extend(declared_errors)
            errors.append(
                {
                    "rank": rank,
                    "type": "InvalidRankCompletion",
                    "message": "invalid field(s): "
                    + ", ".join(sorted(set(invalid))),
                }
            )
            continue
        errors.extend(declared_errors)
        promoted: list[str] = []
        for source, destination in publications:
            try:
                if os.path.lexists(destination):
                    raise FileExistsError(destination)
                source.replace(destination)
            except Exception as exc:
                failure = {"rank": rank, **error_payload(exc)}
                if promoted:
                    failure = {
                        "rank": rank,
                        "type": "PartialRankPromotion",
                        "message": (
                            "rank promotion stopped after publishing "
                            + ", ".join(promoted)
                        ),
                        "promoted_paths": promoted,
                        "failed_path": str(destination.relative_to(run_dir)),
                        "cause": error_payload(exc),
                    }
                errors.append(failure)
                break
            promoted.append(str(destination.relative_to(run_dir)))
    records, read_errors = _read_mappings(run_dir / "ranks")
    errors.extend(read_errors)
    return records, errors


def _mpi_context(
    mpi: Any, comm: Any, environ: Mapping[str, str]
) -> _RankContext:
    rank = _runtime_integer(comm.Get_rank(), "MPI rank")
    size = _runtime_integer(comm.Get_size(), "MPI world size")
    _validate_topology(rank, size)
    context = None
    failure = None
    try:
        try:
            shared = comm.Split_type(mpi.COMM_TYPE_SHARED, key=rank)
            try:
                local_rank = _runtime_integer(
                    shared.Get_rank(), "MPI local rank"
                )
            finally:
                shared.Free()
        except Exception as exc:
            raise RuntimeError("cannot determine MPI local rank") from exc
        launcher = _launcher_context(environ)
        if launcher is not None and launcher[1:] != (rank, size):
            raise RuntimeError("launcher topology disagrees with mpi4py")
        launcher_local_rank = _local_rank(
            environ, launcher[0] if launcher is not None else None
        )
        if (
            launcher_local_rank is not None
            and launcher_local_rank != local_rank
        ):
            raise RuntimeError(
                "launcher local rank disagrees with MPI shared communicator"
            )
        context = _RankContext(
            rank,
            local_rank,
            size,
            socket.gethostname(),
            launcher[0] if launcher is not None else "mpi",
        )
    except Exception as exc:
        failure = error_payload(exc)
    if failure is None and context is None:
        failure = error_payload(
            RuntimeError("rank context validation produced no context")
        )
    _mpi_failure_consensus(comm, "MPI rank-context validation", failure)
    if context is None:
        raise RuntimeError("MPI rank-context consensus returned no context")
    return context


def _environment_context(environ: Mapping[str, str]) -> _RankContext:
    launcher = _launcher_context(environ)
    if launcher is None:
        raise RuntimeError(
            "file aggregation requires launcher rank variables"
        )
    name, rank, size = launcher
    _validate_topology(rank, size)
    local_rank = _local_rank(environ, name)
    if local_rank is None:
        raise RuntimeError("launcher does not provide a local rank")
    if local_rank >= size:
        raise RuntimeError(
            "launcher local rank must be smaller than world size"
        )
    return _RankContext(
        rank,
        local_rank,
        size,
        socket.gethostname(),
        name,
        _file_launch_id(environ, name),
    )


def _file_launch_id(environ: Mapping[str, str], launcher: str) -> str:
    """Bind file-mode ranks to one launcher invocation, not its allocation."""

    explicit = environ.get("CUPHOTON_MPI_LAUNCH_ID")
    namespace = environ.get("PMIX_NAMESPACE")
    if explicit is not None:
        parts = ("explicit", explicit)
    elif namespace:
        # An mpirun inside a Slurm allocation has its own PMIx namespace.
        parts = ("pmix", namespace)
    elif launcher == "slurm" or (
        launcher == "pmi"
        and environ.get("PMI_RANK") == environ.get("SLURM_PROCID")
        and environ.get("PMI_SIZE") == environ.get("SLURM_NTASKS")
    ):
        job = environ.get("SLURM_JOB_ID", "")
        step = environ.get("SLURM_STEP_ID", "")
        parts = (
            ("slurm", job, step)
            if job.isdecimal() and step.isdecimal()
            else ()
        )
    else:
        parts = ()
    if not parts or any(
        not part or len(part) > 1024 or any(ord(c) < 32 for c in part)
        for part in parts
    ):
        raise RuntimeError(
            "file aggregation needs a launcher-specific identity; "
            "export a fresh "
            "CUPHOTON_MPI_LAUNCH_ID before each mpirun/srun invocation "
            "(see docs/components/xpois.md)"
        )
    return hashlib.sha256(json.dumps(parts).encode()).hexdigest()


def _launcher_context(
    environ: Mapping[str, str],
) -> tuple[str, int, int] | None:
    candidates = (
        ("openmpi", "OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE"),
        ("pmi", "PMI_RANK", "PMI_SIZE"),
        ("slurm", "SLURM_PROCID", "SLURM_NTASKS"),
    )
    for name, rank_key, size_key in candidates:
        present = rank_key in environ, size_key in environ
        if all(present):
            return (
                name,
                _environment_integer(environ, rank_key),
                _environment_integer(environ, size_key),
            )
        if any(present):
            raise RuntimeError(
                f"{name} launcher rank and size must appear together"
            )
    return None


def _local_rank(
    environ: Mapping[str, str], launcher: str | None = None
) -> int | None:
    keys_by_launcher = {
        "openmpi": ("OMPI_COMM_WORLD_LOCAL_RANK",),
        "pmi": ("PMI_LOCAL_RANK", "MPI_LOCALRANKID", "PMIX_LOCAL_RANK"),
        "slurm": (
            "SLURM_LOCALID",
            "PMIX_LOCAL_RANK",
            "MPI_LOCALRANKID",
            "PMI_LOCAL_RANK",
        ),
    }
    keys = keys_by_launcher.get(
        launcher,
        (
            "OMPI_COMM_WORLD_LOCAL_RANK",
            "PMIX_LOCAL_RANK",
            "MPI_LOCALRANKID",
            "PMI_LOCAL_RANK",
            "SLURM_LOCALID",
        ),
    )
    values = [
        (key, _environment_integer(environ, key))
        for key in keys
        if key in environ
    ]
    if launcher == "pmi" and not values and "SLURM_LOCALID" in environ:
        slurm_keys = ("SLURM_PROCID", "SLURM_NTASKS", "SLURM_LOCALID")
        topology_keys = ("PMI_RANK", "PMI_SIZE", *slurm_keys)
        if all(key in environ for key in topology_keys):
            pmi_topology = (
                _environment_integer(environ, "PMI_RANK"),
                _environment_integer(environ, "PMI_SIZE"),
            )
            slurm_topology = (
                _environment_integer(environ, "SLURM_PROCID"),
                _environment_integer(environ, "SLURM_NTASKS"),
            )
            if pmi_topology != slurm_topology:
                raise RuntimeError(
                    "PMI and Slurm launcher rank topology disagree"
                )
            values = [
                (
                    "SLURM_LOCALID",
                    _environment_integer(environ, "SLURM_LOCALID"),
                )
            ]
    if not values:
        return None
    if any(value != values[0][1] for _, value in values[1:]):
        family = launcher or "MPI"
        raise RuntimeError(f"{family} launcher local-rank variables disagree")
    return values[0][1]


def _prebound_visibility(environ: Mapping[str, str]) -> str:
    raw_visibility = environ.get("CUDA_VISIBLE_DEVICES")
    if raw_visibility is None:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES is unset; bind one device per MPI rank "
            "with cuphoton-openmpi-rank-exec or scheduler GPU binding"
        )
    tokens = _visible_tokens(raw_visibility)
    if len(tokens) != 1:
        raise RuntimeError(
            "MPI rank must see one CUDA device before mpi4py import; use "
            "cuphoton-openmpi-rank-exec or scheduler GPU binding"
        )
    allocated_raw = environ.get(_ALLOCATED_DEVICES)
    if allocated_raw is None:
        return tokens[0]
    allocated = _visible_tokens(allocated_raw)
    local_rank = _local_rank(environ, "openmpi")
    if local_rank is None or local_rank >= len(allocated):
        raise RuntimeError(
            f"{_ALLOCATED_DEVICES} requires a valid Open MPI local rank; "
            "unset it when launching without cuphoton-openmpi-rank-exec"
        )
    if allocated[local_rank] != tokens[0]:
        raise RuntimeError("pre-bound CUDA device does not match allocation")
    return tokens[0]


def _visible_tokens(raw: Any) -> tuple[str, ...]:
    if not isinstance(raw, str) or not raw.strip() or raw.strip() == "-1":
        raise RuntimeError("CUDA_VISIBLE_DEVICES exposes no devices")
    tokens = tuple(token.strip() for token in raw.split(","))
    if any(not token or token.startswith("-") for token in tokens) or len(
        tokens
    ) != len(set(tokens)):
        raise RuntimeError("CUDA_VISIBLE_DEVICES is invalid")
    return tokens


def _environment_integer(environ: Mapping[str, str], key: str) -> int:
    raw = environ[key]
    if not isinstance(raw, str) or not _UNSIGNED.fullmatch(raw):
        raise RuntimeError(f"launcher variable {key} must be unsigned")
    return int(raw)


def _runtime_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"{field} must be non-negative")
    return value


def _validate_topology(rank: int, size: int) -> None:
    if size <= 0 or rank >= size:
        raise RuntimeError("invalid launcher rank topology")


def _integer(value: Any) -> int | None:
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool)
        else None
    )


def _rank_result_sort_key(result: Mapping[str, Any]) -> tuple[bool, int]:
    rank = _integer(result.get("rank"))
    return (rank is None, rank if rank is not None else 0)


def _finite_nonnegative(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def _same_host(left: str, right: str) -> bool:
    left = left.rstrip(".").casefold()
    right = right.rstrip(".").casefold()
    return (
        left == right
        or ("." not in left and right.startswith(left + "."))
        or ("." not in right and left.startswith(right + "."))
    )


def _attempt_path(run_dir: Path, run_id: str, attempt_id: str) -> Path:
    digest = hashlib.sha256(attempt_id.encode()).hexdigest()[:16]
    return run_dir.parent / ".mpi-attempts" / f"{run_id}-{digest}.json"


def _active_file_attempt_for_rank_failure(
    run_dir: Path,
    run_id: str,
    attempt_id: str,
    manifest_sha256: str,
    launch_id: str | None,
) -> bool:
    if (
        not _real_directory(run_dir)
        or not _real_directory(
            _attempt_staging_dir(run_dir, run_id, attempt_id)
        )
        or os.path.lexists(run_dir / "summary.json")
    ):
        return False
    try:
        marker = _read_regular_mapping(
            _attempt_path(run_dir, run_id, attempt_id)
        )
    except Exception:
        return False
    return (
        launch_id is not None
        and marker.get("launch_id") == launch_id
        and marker.get("run_id") == run_id
        and marker.get("attempt_id") == attempt_id
        and marker.get("manifest_sha256") == manifest_sha256
        and marker.get("status") == "ready"
        and marker.get("terminal") is False
        and marker.get("error") is None
    )


def _repair_committed_file_attempt(
    run_dir: Path,
    run_id: str,
    attempt_id: str,
    manifest: ImagePairManifest | None,
    options: BatchFitOptions,
    world_size: int,
    rank_setup_timeout_sec: float,
    rank_timeout_sec: float,
    preflight_error: Mapping[str, Any] | None,
) -> Path | None:
    if manifest is None or preflight_error is not None:
        return None
    marker_path = _attempt_path(run_dir, run_id, attempt_id)
    try:
        attempt = _read_regular_mapping(marker_path)
    except Exception:
        return None
    marker_mismatches = [
        field
        for field, expected in (
            ("run_id", run_id),
            ("attempt_id", attempt_id),
            ("manifest_sha256", manifest.sha256),
        )
        if attempt.get(field) != expected
    ]
    if marker_mismatches:
        raise RuntimeError(
            "cannot recover MPI file-attempt; marker differs in: "
            + ", ".join(marker_mismatches)
        )
    if (
        attempt.get("status") != "ready"
        or attempt.get("terminal") is not False
    ):
        return None

    summary_path = run_dir / "summary.json"
    if not os.path.lexists(summary_path):
        return None
    try:
        ready = _read_regular_mapping(run_dir / ".ready.json")
        run = _read_regular_mapping(run_dir / "run.json")
        summary = _read_regular_mapping(summary_path)
    except Exception as exc:
        raise RuntimeError(
            "cannot recover MPI file-attempt from non-regular or invalid "
            "committed evidence"
        ) from exc

    mismatches = [
        f"ready.{field}"
        for field, expected in (
            ("run_id", run_id),
            ("attempt_id", attempt_id),
            ("manifest_sha256", manifest.sha256),
            ("resolved_run_dir", str(run_dir)),
            ("status", "ready"),
        )
        if ready.get(field) != expected
    ]
    expected_common = (
        ("executor", "mpi"),
        ("run_id", run_id),
        ("manifest_sha256", manifest.sha256),
        ("aggregation_mode", "files"),
        ("world_size", world_size),
        ("attempt_id", attempt_id),
        ("rank_setup_timeout_sec", rank_setup_timeout_sec),
        ("rank_timeout_sec", rank_timeout_sec),
        ("options", options.to_payload()),
    )
    mismatches.extend(
        f"run.{field}"
        for field, expected in (
            ("schema", "cuphoton.xpois.mpi-run/v1"),
            ("record_type", "immutable-launch"),
            *expected_common,
        )
        if run.get(field) != expected
    )
    mismatches.extend(
        f"summary.{field}"
        for field, expected in (
            ("schema", "cuphoton.xpois.mpi-summary/v1"),
            *expected_common,
        )
        if summary.get(field) != expected
    )
    expected_shards = [
        {
            "rank": rank,
            "item_ids": [item.item_id for item in shard],
            "weight_bytes": sum(item.weight_bytes for item in shard),
            "item_ids_sha256": _item_ids_sha256(shard),
        }
        for rank, shard in enumerate(
            partition_byte_balanced(manifest.work_items(), world_size)
        )
    ]
    if summary.get("shards") != expected_shards:
        mismatches.append("summary.shards")
    item_audit = summary.get("terminal_record_audit")
    rank_audit = summary.get("rank_result_audit")
    record_errors = summary.get("terminal_record_errors")
    aggregation_errors = summary.get("aggregation_errors")
    if not isinstance(item_audit, Mapping):
        mismatches.append("summary.terminal_record_audit")
    if not isinstance(rank_audit, Mapping):
        mismatches.append("summary.rank_result_audit")
    if not isinstance(record_errors, list):
        mismatches.append("summary.terminal_record_errors")
    if not isinstance(aggregation_errors, list):
        mismatches.append("summary.aggregation_errors")
    if not mismatches:
        success = (
            item_audit.get("ok") is True
            and not item_audit.get("failed_item_ids")
            and rank_audit.get("ok") is True
            and not rank_audit.get("setup_failed_ranks")
            and not record_errors
            and not aggregation_errors
        )
        expected_status = "success" if success else "failed"
        if summary.get("status") != expected_status:
            mismatches.append("summary.status")
    if mismatches:
        raise RuntimeError(
            "cannot recover MPI file-attempt; committed evidence differs in: "
            + ", ".join(mismatches)
        )
    _set_attempt_terminal(
        run_dir,
        run_id,
        attempt_id,
        manifest.sha256,
        str(summary["status"]),
        None,
    )
    return summary_path


def _ensure_real_file_attempt_root(marker: Path) -> Path:
    output_root = marker.parent.parent
    output_root.mkdir(parents=True, exist_ok=True)
    if not _real_directory(output_root):
        raise RuntimeError("MPI output root is not a real directory")
    attempt_root = marker.parent
    try:
        attempt_root.mkdir()
    except FileExistsError:
        pass
    if not _real_directory(attempt_root):
        raise RuntimeError("MPI file-attempt path is not a real directory")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(attempt_root, directory_flags)
    os.close(descriptor)
    return attempt_root


def _claim_file_attempt(marker: Path) -> None:
    attempt_root = _ensure_real_file_attempt_root(marker)
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    directory_fd = os.open(attempt_root, directory_flags)
    try:
        try:
            descriptor = os.open(
                marker.name,
                os.O_CREAT
                | os.O_EXCL
                | os.O_WRONLY
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=directory_fd,
            )
        except FileExistsError as exc:
            raise _FileAttemptExistsError(
                "cannot prepare MPI file run: attempt_id already exists"
            ) from exc
        os.close(descriptor)
    finally:
        os.close(directory_fd)


def _attempt_staging_dir(run_dir: Path, run_id: str, attempt_id: str) -> Path:
    return _attempt_path(run_dir, run_id, attempt_id).with_suffix("")


def _file_preflight_path(
    run_dir: Path, run_id: str, attempt_id: str, rank: int
) -> Path:
    return (
        _attempt_staging_dir(run_dir, run_id, attempt_id)
        / "preflight"
        / f"rank-{rank:04d}.json"
    )


def _file_rank_staging_dir(
    run_dir: Path, run_id: str, attempt_id: str, rank: int
) -> Path:
    return (
        _attempt_staging_dir(run_dir, run_id, attempt_id) / f"rank-{rank:04d}"
    )


def _set_attempt_terminal(
    run_dir: Path,
    run_id: str,
    attempt_id: str,
    manifest_sha256: str | None,
    status: str,
    error: Mapping[str, Any] | None,
) -> None:
    atomic_write_json(
        _attempt_path(run_dir, run_id, attempt_id),
        {
            "run_id": run_id,
            "attempt_id": attempt_id,
            "manifest_sha256": manifest_sha256,
            "status": status,
            "terminal": True,
            "completed_at_utc": timestamp_utc(),
            "error": dict(error) if error else None,
        },
    )


def _wait_mapping(
    path: Path,
    timeout: float,
    statuses: set[str],
    *,
    accept_terminal: bool = False,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    delay = _POLL_INITIAL_SEC
    while time.monotonic() < deadline:
        if path.exists():
            try:
                payload = _read_regular_mapping(path)
                if payload.get("status") in statuses or (
                    accept_terminal and payload.get("terminal") is True
                ):
                    return payload
            except (OSError, ValueError):
                pass
        delay = _shared_filesystem_wait(deadline, delay)
    raise TimeoutError(f"timed out waiting for {path}")


def _wait_identity_mapping(
    path: Path,
    timeout: float,
    expected_identity: Mapping[str, Any],
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    delay = _POLL_INITIAL_SEC
    last_mismatched: list[str] = []
    while time.monotonic() < deadline:
        try:
            payload = _read_regular_mapping(path)
        except (OSError, ValueError):
            pass
        else:
            last_mismatched = [
                field
                for field, expected in expected_identity.items()
                if payload.get(field) != expected
            ]
            if not last_mismatched:
                return payload
        delay = _shared_filesystem_wait(deadline, delay)
    detail = (
        "; identity differs in: " + ", ".join(last_mismatched)
        if last_mismatched
        else ""
    )
    raise TimeoutError(f"timed out waiting for {path}{detail}")


def _shared_filesystem_wait(deadline: float, delay: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining > 0:
        time.sleep(min(delay, remaining))
    return min(delay * 2, _POLL_MAX_SEC)


def _aggregate_error(
    run_dir: Path, run_id: str, error: Mapping[str, Any]
) -> None:
    try:
        atomic_write_json(
            run_dir / "aggregate-error.json",
            {"run_id": run_id, "error": dict(error)},
        )
    except Exception:
        pass


def _load_mpi_api() -> _MPIAPI:
    try:
        from mpi4py import MPI
    except (ImportError, OSError, RuntimeError) as exc:
        raise RuntimeError(
            "MPI collective aggregation requires mpi4py built against the "
            "allocation's MPI runtime"
        ) from exc
    try:
        library_version = _normalize_mpi_library_version(
            MPI.Get_library_version()
        )
    except Exception:  # pragma: no cover - implementation-specific
        library_version = None
    return _MPIAPI(
        MPI,
        MPI.COMM_WORLD,
        _distribution_version("mpi4py"),
        library_version,
    )


def _normalize_mpi_library_version(value: Any) -> str | None:
    if value is None:
        return None
    return str(value).rstrip("\x00 \t\r\n")


def _distribution_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


__all__ = ["MPIBatchResult", "run_mpi_image_pair_batch"]
