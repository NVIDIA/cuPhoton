# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Invariant-aware command-line interface for xBeam."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TypeVar

from cuphoton.core.cli import (
    BoolInvariant,
    CommandError,
    ExistingPathInvariant,
    InvariantAwareCommand,
    NonNegativeIntegerInvariant,
    PathValueInvariant,
    PositiveIntegerInvariant,
    SetInvariant,
    StringInvariant,
    VariablePositionalInvariant,
)
from cuphoton.xbeam.adapters.radio.difx_job import load_interop_job
from cuphoton.xbeam.adapters.radio.engine import (
    CorrelationRun,
    prepare_correlation,
)
from cuphoton.xbeam.adapters.radio.models import (
    AutocorrelationMode,
    Job,
    OutputFormat,
    VisibilityRecord,
    VisibilitySink,
    job_to_dict,
    load_job,
)
from cuphoton.xbeam.adapters.radio.output import XBeamV1Writer
from cuphoton.xbeam.adapters.radio.swin import SwinWriter
from cuphoton.xbeam.adapters.vdif.ingest import VdifPlanningCache
from cuphoton.xbeam.adapters.vdif.vdif_index import (
    extract_vdif_headers,
    inspect_vdif,
)
from cuphoton.xbeam.adapters.vdif.vdif_sidecar import (
    VdifBinaryIndexSummary,
    validate_vdif_binary_index,
)
from cuphoton.xbeam.adapters.vdif.vdif_sidecar_builder import (
    build_vdif_binary_index,
)
from cuphoton.xbeam.runtime.policy import (
    resolve_execution_policy,
)

from .errors import XBeamError
from .integrity import execution_contract
from .profiling import nvtx_enabled, nvtx_range
from .sharding import (
    TimeShard,
    load_ordered_shard_outputs,
    merge_native_shards,
    merge_swin_shards,
    plan_time_shards,
    select_time_shard,
)
from .windowing import CorrelationWindow

T = TypeVar("T")

_DEFAULT_MAX_READ_BYTES = 1 << 30
_INGEST_MANIFEST = "ingest-manifest.json"
_LOCAL_WORKER_TERMINATE_TIMEOUT_S = 30.0
_KNOWN_COMMAND_EXCEPTIONS = (
    FileExistsError,
    FileNotFoundError,
    ImportError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
    XBeamError,
)


def _job_fingerprint(job: Job) -> str:
    document = job_to_dict(job)
    # Resolved source paths can differ across worker mount aliases. Their
    # stable positions remain here; manifests compare source evidence.
    for datastream in document["datastreams"]:
        source_files = datastream["source_files"]
        datastream["source_files"] = list(range(len(source_files)))
    encoded = json.dumps(
        document,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _local_source_snapshot(
    job: Job,
) -> tuple[tuple[int, int, int, int], ...]:
    """Return assembly-job source evidence available on this node."""

    snapshot = []
    for datastream in job.datastreams:
        for source_index, value in enumerate(datastream.source_files):
            try:
                source_stat = Path(value).expanduser().stat()
            except OSError:
                continue
            snapshot.append(
                (
                    datastream.identifier,
                    source_index,
                    source_stat.st_size,
                    source_stat.st_mtime_ns,
                )
            )
    return tuple(sorted(snapshot))


def _local_source_snapshot_document(job: Job) -> list[dict[str, object]]:
    """Return complete local source identities for an output manifest."""

    snapshot = []
    for datastream in job.datastreams:
        for source_index, value in enumerate(datastream.source_files):
            path = Path(value).expanduser().resolve(strict=True)
            source_stat = path.stat()
            snapshot.append(
                {
                    "datastream_id": datastream.identifier,
                    "source_index": source_index,
                    "path": str(path),
                    "size_bytes": source_stat.st_size,
                    "mtime_ns": source_stat.st_mtime_ns,
                    "device": source_stat.st_dev,
                    "inode": source_stat.st_ino,
                }
            )
    return snapshot


def _write_records(
    records: Iterable[VisibilityRecord], sink: VisibilitySink
) -> None:
    iterator = iter(records)
    if not nvtx_enabled():
        for record in iterator:
            sink.write(record)
        return
    while True:
        with nvtx_range("xbeam:record-generation"):
            try:
                record = next(iterator)
            except StopIteration:
                return
        with nvtx_range("xbeam:output-write"):
            sink.write(record)


class BackendInvariant(SetInvariant):
    """Supported correlation backend."""

    _set = {"cupy", "numpy"}


class VdifIoInvariant(SetInvariant):
    """Supported file-to-device VDIF data paths."""

    _set = {"cufile", "posix"}


class AutocorrelationModeInvariant(SetInvariant):
    """Supported parallel-autocorrelation channel reductions."""

    _set = {item.value for item in AutocorrelationMode}


class SubintegrationTilingInvariant(SetInvariant):
    """Supported GPU subintegration-tiling policies."""

    _set = {"auto", "off"}


class SidecarBackendInvariant(SetInvariant):
    """Supported exact-frame sidecar construction paths."""

    _set = {"cufile", "posix"}


class XBeamCommand(InvariantAwareCommand):
    """Shared xBeam command behavior."""

    def _call(self, func: Callable[..., T], *args, **kwargs) -> T:
        try:
            return func(*args, **kwargs)
        except _KNOWN_COMMAND_EXCEPTIONS as exc:
            raise CommandError(str(exc)) from exc

    def _emit_json(self, payload: object) -> None:
        self._out(json.dumps(payload, indent=2, sort_keys=True))


class _UploadWorkersArg(StringInvariant):
    _arg: str | tuple[str, ...] = "--upload-workers"
    _help = "Maximum input read/upload workers: auto or 1 through 16."
    _mandatory = False
    _default: str | None = None


class CorrelateCommand(XBeamCommand):
    """Correlate a normalized JSON or supported DiFX job."""

    job: str
    output_path: Path
    backend = "numpy"
    profile: str | None = None
    batch_blocks: int | None = None
    scan: str | None = None
    first_integration: int | None = None
    integration_count: int | None = None
    ingest_window_integrations: int | None = None
    max_read_bytes = _DEFAULT_MAX_READ_BYTES
    assume_ordered = False
    index_dir: Path | None = None
    vdif_io = "posix"
    shard_count: int | None = None
    shard_index: int | None = None
    devices: str | None = None
    autocorrelation_mode = AutocorrelationMode.AVERAGE.value
    subintegration_tiling = "auto"
    max_tile_subintegrations: int | None = None
    upload_workers = "auto"

    class JobArg(ExistingPathInvariant):
        _help = "Normalized JSON job or supported .input job."
        _metavar = "JOB"

    class OutputPathArg(PathValueInvariant):
        _arg = "--output"
        _help = "New output directory."
        _mandatory = True
        _metavar = "OUTPUT"

    class BackendArg(BackendInvariant):
        _arg = "--backend"
        _help = "Correlation backend. [default: %default]"
        _mandatory = False
        _default = "numpy"

    class ProfileArg(StringInvariant):
        _arg = "--profile"
        _help = "Compatibility profile."
        _mandatory = False
        _default = None

    class BatchBlocksArg(PositiveIntegerInvariant):
        _arg = "--batch-blocks"
        _help = "Maximum FFT blocks per processing batch."
        _mandatory = False
        _default = None

    class ScanArg(StringInvariant):
        _arg = "--scan"
        _help = "Scan identifier for a bounded correlation window."
        _mandatory = False
        _default = None

    class FirstIntegrationArg(NonNegativeIntegerInvariant):
        _arg = "--first-integration"
        _help = "Zero-based first integration within --scan."
        _mandatory = False
        _default = None

    class IntegrationCountArg(PositiveIntegerInvariant):
        _arg = "--integration-count"
        _help = "Number of complete integrations to correlate."
        _mandatory = False
        _default = None

    class IngestWindowIntegrationsArg(PositiveIntegerInvariant):
        _arg = "--ingest-window-integrations"
        _help = (
            "Maximum integrations resident in one bounded ingest window; "
            "all windows run in this process."
        )
        _mandatory = False
        _default = None

    class MaxReadBytesArg(PositiveIntegerInvariant):
        _arg = "--max-read-bytes"
        _help = "Maximum planned input and materialized bytes per process."
        _mandatory = False
        _default = _DEFAULT_MAX_READ_BYTES

    class AssumeOrderedArg(BoolInvariant):
        _arg = "--assume-ordered"
        _help = "Use validated direct seeks for independently ordered VDIF."

    class IndexDirArg(PathValueInvariant):
        _arg = "--index-dir"
        _help = (
            "Directory containing sidecars from build-vdif-sidecar. "
            "[default: beside each recording]"
        )
        _mandatory = False
        _default = None

    class VdifIoArg(VdifIoInvariant):
        _arg = "--vdif-io"
        _help = (
            "VDIF GPU ingest path: pinned POSIX or GPUDirect Storage. "
            "[default: %default]"
        )
        _mandatory = False
        _default = "posix"

    class ShardCountArg(PositiveIntegerInvariant):
        _arg = "--shard-count"
        _help = "Total deterministic time shards for this distributed job."
        _mandatory = False
        _default = None

    class ShardIndexArg(NonNegativeIntegerInvariant):
        _arg = "--shard-index"
        _help = "Zero-based time shard executed by this worker."
        _mandatory = False
        _default = None

    class DevicesArg(StringInvariant):
        _arg = "--devices"
        _help = (
            "Comma-separated logical CUDA devices for local process sharding."
        )
        _mandatory = False
        _default = None

    class AutocorrelationModeArg(AutocorrelationModeInvariant):
        _arg = "--autocorrelation-mode"
        _help = (
            "Parallel-auto channel reduction: average adjacent channels, or "
            "truncation to the output channel count. [default: %default]"
        )
        _mandatory = False
        _default = AutocorrelationMode.AVERAGE.value

    class SubintegrationTilingArg(SubintegrationTilingInvariant):
        _arg = "--subintegration-tiling"
        _help = (
            "Tile oversized GPU integrations by subintegration. "
            "[default: %default]"
        )
        _mandatory = False
        _default = "auto"

    class MaxTileSubintegrationsArg(PositiveIntegerInvariant):
        _arg = "--max-tile-subintegrations"
        _help = "Optional maximum subintegrations per internal GPU tile."
        _mandatory = False
        _default = None

    class UploadWorkersArg(_UploadWorkersArg):
        _arg = ("--upload-workers", "--vdif-upload-workers")
        _default = "auto"

    def run(self) -> None:
        self._call(self._run_correlation)

    def _run_correlation(self) -> None:
        resolve_execution_policy(
            self.vdif_io, upload_workers=self.upload_workers
        )
        job_path = Path(self.job).expanduser()
        output = self.output_path.expanduser()
        is_text = job_path.suffix.lower() == ".input"
        job = load_interop_job(job_path) if is_text else load_job(job_path)
        profile = self.profile or (
            "interop-real-usb-v1" if is_text else "analytic-v1"
        )
        window_values = (
            self.scan,
            self.first_integration,
            self.integration_count,
        )
        if any(value is not None for value in window_values) and not all(
            value is not None for value in window_values
        ):
            raise ValueError(
                "--scan, --first-integration, and --integration-count must "
                "be provided together"
            )
        window = (
            CorrelationWindow(
                self.scan, self.first_integration, self.integration_count
            )
            if self.scan is not None
            and self.first_integration is not None
            and self.integration_count is not None
            else None
        )
        shard_values = (self.shard_count, self.shard_index)
        if any(value is not None for value in shard_values) and not all(
            value is not None for value in shard_values
        ):
            raise ValueError(
                "--shard-count and --shard-index must be provided together"
            )
        if (
            any(value is not None for value in shard_values) or self.devices
        ) and (window is None):
            raise ValueError(
                "time sharding requires --scan, --first-integration, and "
                "--integration-count"
            )
        shard: TimeShard | None = None
        if self.shard_count is not None:
            assert self.shard_index is not None
            assert window is not None
            shard = select_time_shard(
                window, self.shard_count, self.shard_index
            )
            window = shard.window
        if self.devices is not None:
            if self.backend != "cupy":
                raise ValueError("--devices requires --backend cupy")
            assert window is not None
            _run_local_shards(
                command=self,
                job_path=job_path.resolve(),
                job=job,
                profile=profile,
                parent_window=window,
                distributed_shard=shard,
                devices=_resolve_local_devices(
                    _parse_devices(self.devices),
                    os.environ.get("CUDA_VISIBLE_DEVICES"),
                ),
                output=output,
            )
            return
        if self.ingest_window_integrations is not None and window is None:
            raise ValueError(
                "--ingest-window-integrations requires --scan, "
                "--first-integration, and --integration-count"
            )
        windows = _split_window(window, self.ingest_window_integrations)
        if output.exists():
            raise FileExistsError(
                f"output directory already exists: {output}"
            )
        planning_cache = VdifPlanningCache()
        prepare_started = time.perf_counter()
        prepared = prepare_correlation(
            job,
            backend=self.backend,
            profile=profile,
            batch_blocks=self.batch_blocks,
            window=windows[0],
            max_read_bytes=self.max_read_bytes,
            assume_ordered=self.assume_ordered,
            index_dir=(
                str(self.index_dir) if self.index_dir is not None else None
            ),
            vdif_io=self.vdif_io,
            autocorrelation_mode=self.autocorrelation_mode,
            subintegration_tiling=self.subintegration_tiling,
            max_tile_subintegrations=self.max_tile_subintegrations,
            upload_workers=self.upload_workers,
            planning_cache=planning_cache,
            recovery_plan_window=(window if len(windows) > 1 else None),
        )
        prepared_seconds = time.perf_counter() - prepare_started
        output.mkdir(parents=True)
        manifest = {
            "schema_version": 1,
            "status": "running",
            "job": str(job_path),
            "job_fingerprint": _job_fingerprint(job),
            "backend": self.backend,
            "profile": profile,
            "vdif_io": self.vdif_io,
            "autocorrelation_mode": self.autocorrelation_mode,
            "upload_workers": self.upload_workers,
            "window": (
                {
                    "scan_id": window.scan_id,
                    "first_integration": window.first_integration,
                    "integration_count": window.integration_count,
                }
                if window is not None
                else None
            ),
            "ingest_window_integrations": self.ingest_window_integrations,
            "estimated_read_bytes": prepared.estimated_read_bytes,
            "ingest": (
                prepared.ingest_manifest if len(windows) == 1 else None
            ),
            "ingest_windows": [],
            "shard": shard.to_dict() if shard is not None else None,
        }
        _append_ingest_window(manifest, windows[0], prepared)
        _write_manifest(output, manifest)
        recovery_plans = prepared.recovery_plans

        def prepare_window(
            selected_window: CorrelationWindow | None,
        ) -> tuple[CorrelationRun, float]:
            prepare_started = time.perf_counter()
            result = prepare_correlation(
                job,
                backend=self.backend,
                profile=profile,
                batch_blocks=self.batch_blocks,
                window=selected_window,
                max_read_bytes=self.max_read_bytes,
                assume_ordered=self.assume_ordered,
                index_dir=(
                    str(self.index_dir)
                    if self.index_dir is not None
                    else None
                ),
                vdif_io=self.vdif_io,
                autocorrelation_mode=self.autocorrelation_mode,
                subintegration_tiling=self.subintegration_tiling,
                max_tile_subintegrations=self.max_tile_subintegrations,
                upload_workers=self.upload_workers,
                planning_cache=planning_cache,
                # The first preparation compiles recovery across the parent
                # shard. Every bounded ingest window can therefore reuse the
                # same immutable plan without replaying the physical timeline.
                recovery_plans=recovery_plans,
                # The first window has already initialized the job-specific
                # CUDA pipeline. Prefetched windows perform host planning
                # only and inherit the warm process state when consumed.
                start_gpu_warmup=False,
            )
            return result, time.perf_counter() - prepare_started

        def write_all(sink: VisibilitySink) -> None:
            nonlocal prepared, prepared_seconds
            future: Future[tuple[CorrelationRun, float]] | None = None
            with ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="cuphoton-xbeam-window-plan",
            ) as planner:
                for index, selected_window in enumerate(windows):
                    if index:
                        assert future is not None
                        prepared, prepared_seconds = future.result()
                        _append_ingest_window(
                            manifest, selected_window, prepared
                        )
                        _write_manifest(output, manifest)
                    future = (
                        planner.submit(prepare_window, windows[index + 1])
                        if index + 1 < len(windows)
                        else None
                    )
                    if len(windows) > 1:
                        assert selected_window is not None
                        last_integration = (
                            selected_window.first_integration
                            + selected_window.integration_count
                            - 1
                        )
                        print(
                            f"correlating ingest window {index + 1}/"
                            f"{len(windows)}: integrations "
                            f"{selected_window.first_integration}.."
                            f"{last_integration}",
                            file=self.error,
                            flush=True,
                        )
                    window_started = time.perf_counter()
                    _write_records(prepared.records(), sink)
                    if len(windows) > 1:
                        assert selected_window is not None
                        correlated_seconds = (
                            time.perf_counter() - window_started
                        )
                        observation_seconds = float(
                            job.integration_duration_s
                            * selected_window.integration_count
                        )
                        realtime_factor = (
                            observation_seconds / correlated_seconds
                        )
                        print(
                            f"completed ingest window {index + 1}/"
                            f"{len(windows)}: planning="
                            f"{prepared_seconds:.3f}s "
                            f"correlation={correlated_seconds:.3f}s "
                            f"observation={observation_seconds:.6f}s "
                            f"realtime={realtime_factor:.3f}x",
                            file=self.error,
                            flush=True,
                        )

        sink: SwinWriter | XBeamV1Writer
        try:
            if job.output.format is OutputFormat.SWIN:
                antenna_indices: dict[str | int, int] = {
                    station.name: index
                    for index, station in enumerate(job.stations)
                }
                with SwinWriter(
                    output / _swin_filename(job), antenna_indices
                ) as sink:
                    write_all(sink)
            else:
                with XBeamV1Writer(output) as sink:
                    write_all(sink)
        except BaseException as exc:
            manifest["status"] = "incomplete"
            manifest["failure"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            _write_manifest(output, manifest)
            raise
        manifest["output_integrity"] = sink.integrity
        manifest["execution_contract"] = execution_contract(
            manifest["job_fingerprint"],
            profile,
            self.backend,
            self.autocorrelation_mode,
        )
        manifest["status"] = "complete"
        _write_manifest(output, manifest)


def _parse_devices(value: str) -> tuple[str, ...]:
    devices = tuple(item.strip() for item in value.split(","))
    if not devices or any(not item for item in devices):
        raise ValueError(
            "--devices must contain comma-separated CUDA devices"
        )
    if len(set(devices)) != len(devices):
        raise ValueError("--devices must not contain duplicates")
    return devices


def _resolve_local_devices(
    devices: tuple[str, ...], inherited_visibility: str | None
) -> tuple[str, ...]:
    """Map logical selections through an inherited CUDA visibility mask."""

    if inherited_visibility is None:
        return devices
    visible = tuple(item.strip() for item in inherited_visibility.split(","))
    if not visible or any(not item or item == "-1" for item in visible):
        raise ValueError("CUDA_VISIBLE_DEVICES exposes no usable devices")
    resolved = []
    for device in devices:
        if device.isdecimal():
            index = int(device)
            if index >= len(visible):
                raise ValueError(
                    f"logical CUDA device {device} is outside inherited "
                    "CUDA_VISIBLE_DEVICES"
                )
            resolved.append(visible[index])
        elif device in visible:
            resolved.append(device)
        else:
            raise ValueError(
                f"CUDA device {device!r} is outside inherited "
                "CUDA_VISIBLE_DEVICES"
            )
    if len(set(resolved)) != len(resolved):
        raise ValueError("--devices resolves to duplicate CUDA devices")
    return tuple(resolved)


def _local_worker_argv(
    command: CorrelateCommand,
    *,
    job_path: Path,
    output: Path,
    profile: str,
    parent_window: CorrelationWindow,
    shard_count: int,
    shard_index: int,
) -> list[str]:
    arguments = [
        sys.executable,
        "-m",
        "cuphoton",
        "xbeam",
        "correlate",
        str(job_path),
        f"--output={output}",
        "--backend=cupy",
        f"--profile={profile}",
        f"--scan={parent_window.scan_id}",
        f"--first-integration={parent_window.first_integration}",
        f"--integration-count={parent_window.integration_count}",
        f"--shard-count={shard_count}",
        f"--shard-index={shard_index}",
        f"--max-read-bytes={command.max_read_bytes}",
        f"--vdif-io={command.vdif_io}",
        "--autocorrelation-mode="
        + getattr(
            command,
            "autocorrelation_mode",
            AutocorrelationMode.AVERAGE.value,
        ),
        "--subintegration-tiling="
        + getattr(command, "subintegration_tiling", "auto"),
        "--upload-workers=" + str(getattr(command, "upload_workers", "auto")),
    ]
    if command.batch_blocks is not None:
        arguments.append(f"--batch-blocks={command.batch_blocks}")
    if command.ingest_window_integrations is not None:
        arguments.append(
            "--ingest-window-integrations="
            f"{command.ingest_window_integrations}"
        )
    if getattr(command, "max_tile_subintegrations", None) is not None:
        arguments.append(
            f"--max-tile-subintegrations={command.max_tile_subintegrations}"
        )
    if command.assume_ordered:
        arguments.append("--assume-ordered")
    if command.index_dir is not None:
        arguments.append(f"--index-dir={command.index_dir}")
    return arguments


def _merge_shard_set(
    job: Job,
    shard_directories: tuple[Path, ...] | list[Path],
    output: Path,
) -> tuple[tuple[Path, ...], int]:
    ordered, _shards = load_ordered_shard_outputs(
        shard_directories,
        expected_job_fingerprint=_job_fingerprint(job),
        expected_source_snapshot=_local_source_snapshot(job),
    )
    if job.output.format is OutputFormat.SWIN:
        metric = merge_swin_shards(
            ordered, output, _swin_filename(job), job=job
        )
    else:
        metric = merge_native_shards(ordered, output)
    return ordered, metric


def _assembled_metric_unit(job: Job) -> str:
    return "bytes" if job.output.format is OutputFormat.SWIN else "records"


def _prepare_local_shard_indexes(command: CorrelateCommand, job: Job) -> None:
    """Validate shared sidecars before local workers start concurrently."""

    if command.assume_ordered:
        return
    prepared: set[tuple[Path, int]] = set()
    for datastream in job.datastreams:
        for value in datastream.source_files:
            path = Path(value).expanduser().resolve(strict=True)
            key = (path, datastream.frame_size_bytes)
            if key in prepared:
                continue
            prepared.add(key)
            validate_vdif_binary_index(
                path,
                frame_size_bytes=datastream.frame_size_bytes,
                index_dir=command.index_dir,
            )


def _terminate_local_workers(
    processes: dict[int, subprocess.Popen[bytes]],
    *,
    timeout_s: float = _LOCAL_WORKER_TERMINATE_TIMEOUT_S,
) -> dict[int, tuple[int | None, str]]:
    """Terminate workers under shared TERM and KILL deadlines."""

    if timeout_s < 0:
        raise ValueError(
            "local worker termination timeout must be nonnegative"
        )
    outcomes: dict[int, tuple[int | None, str]] = {}
    pending: list[tuple[int, subprocess.Popen[bytes]]] = []
    for index, process in processes.items():
        while True:
            try:
                returncode = process.poll()
                if returncode is None:
                    process.terminate()
                    pending.append((index, process))
                else:
                    status = "complete" if returncode == 0 else "failed"
                    outcomes[index] = (returncode, status)
                break
            except KeyboardInterrupt:
                # Preserve the outer interrupt, but do not let another Ctrl-C
                # strand a child while cleanup is in progress.
                continue

    terminate_deadline = time.monotonic() + timeout_s
    kill_pending: list[tuple[int, subprocess.Popen[bytes]]] = []
    for index, process in pending:
        while True:
            try:
                returncode = process.wait(
                    timeout=max(0.0, terminate_deadline - time.monotonic())
                )
                outcomes[index] = (returncode, "terminated")
                break
            except KeyboardInterrupt:
                continue
            except subprocess.TimeoutExpired:
                kill_pending.append((index, process))
                break

    for _index, process in kill_pending:
        while True:
            try:
                process.kill()
                break
            except KeyboardInterrupt:
                continue
    kill_deadline = time.monotonic() + timeout_s
    for index, process in kill_pending:
        while True:
            try:
                returncode = process.wait(
                    timeout=max(0.0, kill_deadline - time.monotonic())
                )
                break
            except KeyboardInterrupt:
                continue
            except subprocess.TimeoutExpired:
                returncode = process.poll()
                break
        status = "killed" if returncode is not None else "unreaped"
        outcomes[index] = (returncode, status)
    return outcomes


def _run_local_shards(
    *,
    command: CorrelateCommand,
    job_path: Path,
    job: Job,
    profile: str,
    parent_window: CorrelationWindow,
    distributed_shard: TimeShard | None = None,
    devices: tuple[str, ...],
    output: Path,
) -> None:
    shards = plan_time_shards(parent_window, len(devices))
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    policy = resolve_execution_policy(
        command.vdif_io,
        upload_workers=getattr(command, "upload_workers", None),
    )
    _prepare_local_shard_indexes(command, job)
    parts = output / "shards"
    parts.mkdir(parents=True)
    shard_outputs = tuple(
        parts / f"shard-{shard.index:05d}" for shard in shards
    )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "running",
        "mode": "local-multi-gpu",
        "axis": "time",
        "job": str(job_path),
        "job_fingerprint": _job_fingerprint(job),
        "backend": "cupy",
        "profile": profile,
        "autocorrelation_mode": getattr(
            command,
            "autocorrelation_mode",
            AutocorrelationMode.AVERAGE.value,
        ),
        "upload_workers": getattr(command, "upload_workers", "auto"),
        "shard": (
            distributed_shard.to_dict()
            if distributed_shard is not None
            else None
        ),
        "source_snapshot": _local_source_snapshot_document(job),
        "parent_window": {
            "scan_id": parent_window.scan_id,
            "first_integration": parent_window.first_integration,
            "integration_count": parent_window.integration_count,
        },
        "workers": [
            {
                "shard": shard.to_dict(),
                "device": device,
                "output": str(shard_output),
                "status": "pending",
            }
            for shard, device, shard_output in zip(
                shards, devices, shard_outputs, strict=True
            )
        ],
    }
    manifest["execution_policy"] = policy.to_manifest()
    _write_manifest(output, manifest)
    workers = manifest["workers"]
    assert isinstance(workers, list)
    processes: dict[int, subprocess.Popen[bytes]] = {}
    read_token_fds: tuple[int, int] | None = None
    read_limit = policy.global_gds_reads
    if read_limit is not None:
        read_fd, return_fd = os.pipe()
        os.write(return_fd, b"\0" * read_limit)
        read_token_fds = (read_fd, return_fd)
        manifest["global_gds_reads"] = read_limit
        _write_manifest(output, manifest)
    try:
        for shard, device, shard_output in zip(
            shards, devices, shard_outputs, strict=True
        ):
            arguments = _local_worker_argv(
                command,
                job_path=job_path,
                output=shard_output,
                profile=profile,
                parent_window=parent_window,
                shard_count=len(shards),
                shard_index=shard.index,
            )
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = device
            if read_token_fds is not None:
                environment["_CUPHOTON_XBEAM_CUFILE_READ_TOKEN_FD"] = str(
                    read_token_fds[0]
                )
                environment["_CUPHOTON_XBEAM_CUFILE_RETURN_TOKEN_FD"] = str(
                    read_token_fds[1]
                )
            timing_path = environment.get("CUPHOTON_XBEAM_HOST_TIMING_FILE")
            if timing_path:
                environment["CUPHOTON_XBEAM_HOST_TIMING_FILE"] = (
                    f"{timing_path}.shard-{shard.index:05d}"
                )
            print(
                f"launching time shard {shard.index + 1}/{len(shards)} "
                f"on CUDA device {device}",
                file=command.error,
                flush=True,
            )
            if read_token_fds is None:
                process = subprocess.Popen(arguments, env=environment)
            else:
                process = subprocess.Popen(
                    arguments, env=environment, pass_fds=read_token_fds
                )
            processes[shard.index] = process
            worker = workers[shard.index]
            assert isinstance(worker, dict)
            worker["status"] = "running"
            worker["pid"] = process.pid
        _write_manifest(output, manifest)

        active = set(processes)
        failure: tuple[int, int] | None = None
        while active and failure is None:
            for index in tuple(active):
                returncode = processes[index].poll()
                if returncode is None:
                    continue
                active.remove(index)
                worker = workers[index]
                assert isinstance(worker, dict)
                worker["returncode"] = returncode
                worker["status"] = "complete" if returncode == 0 else "failed"
                if returncode != 0:
                    failure = (index, returncode)
            if active and failure is None:
                time.sleep(0.05)
        if failure is not None:
            index, returncode = failure
            raise RuntimeError(
                f"local shard {index} failed with exit status {returncode}"
            )
        _write_manifest(output, manifest)
        ordered, metric = _merge_shard_set(job, list(shard_outputs), output)
        manifest["assembled_shards"] = [str(path) for path in ordered]
        manifest["assembled_metric"] = metric
        manifest["assembled_metric_unit"] = _assembled_metric_unit(job)
        manifest["status"] = "complete"
        _write_manifest(output, manifest)
    except BaseException as error:
        outcomes = _terminate_local_workers(processes)
        for index, (returncode, status) in outcomes.items():
            worker = workers[index]
            assert isinstance(worker, dict)
            worker["returncode"] = returncode
            worker["status"] = status
        manifest["status"] = "incomplete"
        manifest["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        _write_manifest(output, manifest)
        raise
    finally:
        if read_token_fds is not None:
            for descriptor in read_token_fds:
                os.close(descriptor)


class CorrelateSeriesCommand(XBeamCommand):
    """Correlate generic uniformly sampled HDF5 time series."""

    config: str
    output_path: Path
    backend = "numpy"
    upload_workers: str | None = None

    class ConfigArg(ExistingPathInvariant):
        _help = "Series JSON configuration."
        _metavar = "CONFIG"

    class OutputPathArg(PathValueInvariant):
        _arg = "--output"
        _help = "New HDF5 result file (never overwritten)."
        _mandatory = True
        _metavar = "OUTPUT"

    class BackendArg(BackendInvariant):
        _arg = "--backend"
        _help = "Array backend. [default: %default]"
        _mandatory = False
        _default = "numpy"

    class UploadWorkersArg(_UploadWorkersArg):
        pass

    def run(self):
        from .adapters.series_cli import run_series_config

        self._call(
            run_series_config,
            self.config,
            self.output_path,
            self.backend,
            upload_workers=self.upload_workers,
        )


class MergeShardsCommand(XBeamCommand):
    """Assemble a complete set of independently executed time shards."""

    job: str
    shards: tuple[Path, ...] | None = None
    output_path: Path

    class JobArg(ExistingPathInvariant):
        _help = "Normalized JSON job or supported .input job."
        _metavar = "JOB"

    class ShardsArg(VariablePositionalInvariant):
        _help = "Complete shard output directories."
        _required = True
        _nargs = "+"
        _item_type = Path
        _metavar = "SHARD"

    class OutputPathArg(PathValueInvariant):
        _arg = "--output"
        _help = "New assembled output directory."
        _mandatory = True
        _metavar = "OUTPUT"

    def run(self) -> None:
        self._call(self._run_merge)

    def _run_merge(self) -> None:
        job_path = Path(self.job).expanduser()
        output = self.output_path.expanduser()
        job = (
            load_interop_job(job_path)
            if job_path.suffix.lower() == ".input"
            else load_job(job_path)
        )
        if output.exists():
            raise FileExistsError(
                f"output directory already exists: {output}"
            )
        shard_directories = tuple(Path(path) for path in self.shards or ())
        ordered, shard_plan = load_ordered_shard_outputs(
            shard_directories,
            expected_job_fingerprint=_job_fingerprint(job),
            expected_source_snapshot=_local_source_snapshot(job),
        )
        worker_manifest = json.loads(
            (ordered[0] / _INGEST_MANIFEST).read_text(encoding="utf-8")
        )
        output.mkdir(parents=True)
        manifest: dict[str, object] = {
            "schema_version": 1,
            "status": "running",
            "mode": "distributed-assembly",
            "axis": "time",
            "job": str(job_path),
            "job_fingerprint": _job_fingerprint(job),
            "backend": worker_manifest["backend"],
            "profile": worker_manifest["profile"],
            "autocorrelation_mode": worker_manifest["autocorrelation_mode"],
            "execution_contract": worker_manifest["execution_contract"],
            "parent_window": shard_plan[0].to_dict()["parent_window"],
            "shards": [str(path) for path in ordered],
        }
        _write_manifest(output, manifest)
        try:
            if job.output.format is OutputFormat.SWIN:
                metric = merge_swin_shards(
                    ordered, output, _swin_filename(job), job=job
                )
            else:
                metric = merge_native_shards(ordered, output)
        except BaseException as error:
            manifest["status"] = "incomplete"
            manifest["failure"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
            _write_manifest(output, manifest)
            raise
        manifest["assembled_metric"] = metric
        manifest["assembled_metric_unit"] = _assembled_metric_unit(job)
        manifest["status"] = "complete"
        _write_manifest(output, manifest)


def _split_window(
    window: CorrelationWindow | None,
    integration_limit: int | None,
) -> tuple[CorrelationWindow | None, ...]:
    if window is None or integration_limit is None:
        return (window,)
    result = []
    first = window.first_integration
    remaining = window.integration_count
    while remaining:
        count = min(integration_limit, remaining)
        result.append(CorrelationWindow(window.scan_id, first, count))
        first += count
        remaining -= count
    return tuple(result)


def _append_ingest_window(
    manifest: dict[str, object],
    window: CorrelationWindow | None,
    prepared: CorrelationRun,
) -> None:
    snapshot = _source_snapshot(prepared.ingest_manifest)
    expected_snapshot = manifest.get("source_snapshot")
    if expected_snapshot is None:
        manifest["source_snapshot"] = snapshot
    elif snapshot != expected_snapshot:
        raise OSError("VDIF source snapshot changed across ingest windows")
    windows = manifest["ingest_windows"]
    assert isinstance(windows, list)
    estimated = prepared.estimated_read_bytes
    windows.append(
        {
            "window": (
                {
                    "scan_id": window.scan_id,
                    "first_integration": window.first_integration,
                    "integration_count": window.integration_count,
                }
                if window is not None
                else None
            ),
            "estimated_read_bytes": estimated,
            "ingest": prepared.ingest_manifest,
        }
    )
    estimates = [
        item["estimated_read_bytes"]
        for item in windows
        if item["estimated_read_bytes"] is not None
    ]
    manifest["estimated_read_bytes"] = (
        sum(estimates) if len(estimates) == len(windows) else None
    )


def _source_snapshot(ingest_manifest: object) -> list[dict[str, object]]:
    """Return the stable source identities recorded by ingest planning."""

    if not isinstance(ingest_manifest, dict):
        raise ValueError("ingest manifest must be an object")
    sources = ingest_manifest.get("sources")
    if not isinstance(sources, list):
        raise ValueError("ingest manifest sources must be a list")
    snapshot = []
    source_indices: dict[int, int] = {}
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("ingest manifest source must be an object")
        try:
            identity = {
                "datastream_id": source["datastream_id"],
                "path": source["path"],
                "size_bytes": source["size_bytes"],
                "mtime_ns": source["mtime_ns"],
                "device": source["device"],
                "inode": source["inode"],
            }
        except KeyError as error:
            raise ValueError(
                f"ingest manifest source has no {error.args[0]}"
            ) from error
        if not isinstance(identity["datastream_id"], int) or not isinstance(
            identity["path"], str
        ):
            raise ValueError("ingest manifest source identity is invalid")
        datastream_id = identity["datastream_id"]
        source_index = source_indices.get(datastream_id, 0)
        source_indices[datastream_id] = source_index + 1
        identity["source_index"] = source_index
        for field in ("size_bytes", "mtime_ns", "device", "inode"):
            if not isinstance(identity[field], int):
                raise ValueError(
                    f"ingest manifest source {field} must be an integer"
                )
        snapshot.append(identity)
    return sorted(
        snapshot,
        key=lambda item: (
            item["datastream_id"],
            item["source_index"],
        ),
    )


class InspectVdifCommand(XBeamCommand):
    """Inspect or extract bounded VDIF headers."""

    files: tuple[Path, ...] | None = None
    frame_size: int | None = None
    probe_count = 33
    extract_headers: Path | None = None
    header_first_frame = 0
    header_frame_count: int | None = None
    json = False

    class FilesArg(VariablePositionalInvariant):
        _help = "One or more VDIF recording files."
        _required = True
        _nargs = "+"
        _item_type = Path
        _metavar = "RECORDING"

    class FrameSizeArg(PositiveIntegerInvariant):
        _arg = "--frame-size"
        _help = "Explicit VDIF frame size in bytes."
        _mandatory = False
        _default = None

    class ProbeCountArg(PositiveIntegerInvariant):
        _arg = "--probe-count"
        _help = "Number of bounded header probes. [default: %default]"
        _mandatory = False
        _default = 33

    class ExtractHeadersArg(PathValueInvariant):
        _arg = "--extract-headers"
        _help = "Write consecutive raw 32-byte headers to OUTPUT."
        _mandatory = False
        _default = None
        _metavar = "OUTPUT"

    class HeaderFirstFrameArg(NonNegativeIntegerInvariant):
        _arg = "--header-first-frame"
        _help = "First frame to extract. [default: %default]"
        _mandatory = False
        _default = 0

    class HeaderFrameCountArg(PositiveIntegerInvariant):
        _arg = "--header-frame-count"
        _help = "Number of headers to extract; default is through EOF."
        _mandatory = False
        _default = None

    class JsonArg(BoolInvariant):
        _arg = "--json"
        _help = "Emit machine-readable JSON."

    def run(self) -> None:
        self._call(self._run_inspection)

    def _run_inspection(self) -> None:
        if self.extract_headers is None and (
            self.header_first_frame != 0
            or self.header_frame_count is not None
        ):
            raise ValueError(
                "header extraction bounds require --extract-headers"
            )
        if self.extract_headers is not None and len(self.files or ()) != 1:
            raise ValueError(
                "--extract-headers requires exactly one recording file"
            )
        reports: list[dict[str, object]] = []
        for path in self.files or ():
            inspection = inspect_vdif(
                path,
                frame_size_bytes=self.frame_size,
                sample_count=self.probe_count,
            )
            extraction = (
                extract_vdif_headers(
                    path,
                    self.extract_headers,
                    frame_size_bytes=inspection.frame_size_bytes,
                    first_frame=self.header_first_frame,
                    frame_count=self.header_frame_count,
                )
                if self.extract_headers is not None
                else None
            )
            reports.append(
                {
                    "inspection": inspection.to_dict(),
                    "header_extraction": (
                        None if extraction is None else extraction.to_dict()
                    ),
                }
            )
        document = {"schema_version": 1, "sources": reports}
        if self.json:
            self._emit_json(document)
            return
        for report in reports:
            inspection_document = report["inspection"]
            assert isinstance(inspection_document, dict)
            self._out(
                f"{inspection_document['path']}: "
                f"{inspection_document['frame_count']} frames, "
                f"{inspection_document['frame_size_bytes']} bytes/frame, "
                f"sampled threads {inspection_document['thread_ids']}, "
                f"sampled_monotonic="
                f"{inspection_document['sampled_monotonic']}"
            )
            extraction_document = report["header_extraction"]
            if isinstance(extraction_document, dict):
                self._out(
                    f"  headers: {extraction_document['output_path']} "
                    f"({extraction_document['frame_count']} records, "
                    f"{extraction_document['output_bytes']} bytes)"
                )


class BuildVdifSidecarCommand(XBeamCommand):
    """Build compact, sorted exact-frame indexes for VDIF recordings."""

    files: tuple[Path, ...] | None = None
    frame_size: int | None = None
    index_dir: Path | None = None
    backend = "cufile"
    devices: str | None = None
    force = False
    json = False

    class FilesArg(VariablePositionalInvariant):
        _help = "One or more VDIF recording files."
        _required = True
        _nargs = "+"
        _item_type = Path
        _metavar = "RECORDING"

    class FrameSizeArg(PositiveIntegerInvariant):
        _arg = "--frame-size"
        _help = "Explicit constant VDIF frame size in bytes."
        _mandatory = False
        _default = None

    class IndexDirArg(PathValueInvariant):
        _arg = "--index-dir"
        _help = "Directory receiving portable .vdix sidecars."
        _mandatory = False
        _default = None

    class BackendArg(SidecarBackendInvariant):
        _arg = "--backend"
        _help = "Sidecar builder data path. [default: %default]"
        _mandatory = False
        _default = "cufile"

    class DevicesArg(StringInvariant):
        _arg = "--devices"
        _help = (
            "Comma-separated CUDA device ordinals used round-robin by "
            "concurrent builders."
        )
        _mandatory = False
        _default = None

    class ForceArg(BoolInvariant):
        _arg = "--force"
        _help = "Replace an existing sidecar."

    class JsonArg(BoolInvariant):
        _arg = "--json"
        _help = "Emit machine-readable JSON."

    def run(self) -> None:
        self._call(self._run_build)

    def _run_build(self) -> None:
        paths = tuple(self.files or ())
        devices: tuple[int, ...]
        if self.devices is None:
            devices = (0,)
        else:
            values = _parse_devices(self.devices)
            if any(not value.isdecimal() for value in values):
                raise ValueError(
                    "sidecar --devices requires CUDA device ordinals"
                )
            devices = tuple(int(value) for value in values)
        worker_count = min(len(devices), len(paths))

        def build(index: int) -> tuple[int, VdifBinaryIndexSummary, float]:
            started = time.perf_counter()
            summary = build_vdif_binary_index(
                paths[index],
                frame_size_bytes=self.frame_size,
                index_dir=self.index_dir,
                backend=self.backend,
                chunk_bytes=256 * 1024 * 1024,
                device=devices[index % len(devices)],
                force=self.force,
            )
            return index, summary, time.perf_counter() - started

        started = time.perf_counter()
        completed: list[tuple[VdifBinaryIndexSummary, float] | None] = [
            None
        ] * len(paths)
        if worker_count == 1:
            for index in range(len(paths)):
                position, summary, seconds = build(index)
                completed[position] = (summary, seconds)
        else:
            with ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="vdif-sidecar",
            ) as executor:
                futures = {
                    executor.submit(build, index): index
                    for index in range(len(paths))
                }
                for future in as_completed(futures):
                    position, summary, seconds = future.result()
                    completed[position] = (summary, seconds)
        elapsed = time.perf_counter() - started
        results = [item for item in completed if item is not None]
        summaries = [item[0] for item in results]
        source_bytes = sum(item.source_size_bytes for item in summaries)
        document = {
            "schema_version": 1,
            "backend": self.backend,
            "workers": worker_count,
            "devices": list(devices),
            "elapsed_seconds": elapsed,
            "source_bytes": source_bytes,
            "throughput_gbytes_per_second": (
                source_bytes / elapsed / 1e9 if elapsed else 0.0
            ),
            "sources": [
                {**summary.to_dict(), "elapsed_seconds": seconds}
                for summary, seconds in results
            ],
        }
        if self.json:
            self._emit_json(document)
            return
        for summary in summaries:
            ordering = "ordered" if summary.source_ordered else "unordered"
            self._out(
                f"{summary.source_path}: {summary.entry_count} frames -> "
                f"{summary.index_path} ({ordering})"
            )
        self._out(
            f"processed {source_bytes:,} bytes in {elapsed:.3f} seconds "
            f"({document['throughput_gbytes_per_second']:.3f} GB/s; "
            f"workers={worker_count}, devices={','.join(map(str, devices))})"
        )


def _swin_filename(job: Job) -> str:
    start = job.start_time
    seconds = int(start.seconds_of_day)
    return f"DIFX_{start.mjd}_{seconds:06d}.s0000.b0000"


def _write_manifest(output: Path, manifest: dict[str, object]) -> None:
    if manifest.get("status") == "complete":
        manifest["execution_contract"] = execution_contract(
            manifest["job_fingerprint"],
            manifest["profile"],
            manifest["backend"],
            manifest["autocorrelation_mode"],
        )
        assembled = output / ".assembly-integrity.json"
        if assembled.exists():
            manifest["output_integrity"] = json.loads(assembled.read_text())
    (output / _INGEST_MANIFEST).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "BuildVdifSidecarCommand",
    "CorrelateCommand",
    "CorrelateSeriesCommand",
    "InspectVdifCommand",
    "MergeShardsCommand",
]
