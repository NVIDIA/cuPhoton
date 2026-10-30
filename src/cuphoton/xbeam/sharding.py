# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Deterministic time sharding and output assembly."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from decimal import Decimal
from functools import wraps
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from types import SimpleNamespace
from typing import Any

import numpy as np

from cuphoton.xbeam.adapters.radio.output import (
    FORMAT_NAME,
    METADATA_FILENAME,
    SPECTRA_FILENAME,
)
from cuphoton.xbeam.adapters.radio.swin import (
    FORMAT_VERSION,
    HEADER,
    SYNC_WORD,
    read_record,
)

from .integrity import execution_contract
from .windowing import CorrelationWindow

SHARD_MANIFEST_FILENAME = "ingest-manifest.json"


@dataclass(frozen=True, slots=True)
class TimeShard:
    """One contiguous, non-empty range of owned output integrations."""

    index: int
    count: int
    parent: CorrelationWindow
    window: CorrelationWindow

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "axis": "time",
            "index": self.index,
            "count": self.count,
            "parent_window": _window_dict(self.parent),
            "window": _window_dict(self.window),
        }


def _window_dict(window: CorrelationWindow) -> dict[str, Any]:
    return {
        "scan_id": window.scan_id,
        "first_integration": window.first_integration,
        "integration_count": window.integration_count,
    }


def _window_from_dict(value: object, *, field: str) -> CorrelationWindow:
    if not isinstance(value, dict):
        raise ValueError(f"shard manifest {field} must be an object")
    try:
        return CorrelationWindow(
            str(value["scan_id"]),
            int(value["first_integration"]),
            int(value["integration_count"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"shard manifest contains invalid {field}"
        ) from error


def plan_time_shards(
    window: CorrelationWindow, shard_count: int
) -> tuple[TimeShard, ...]:
    """Split complete integrations into balanced contiguous time shards."""

    if not isinstance(shard_count, int) or isinstance(shard_count, bool):
        raise TypeError("shard_count must be an integer")
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if shard_count > window.integration_count:
        raise ValueError(
            "shard_count cannot exceed the selected integration count"
        )
    quotient, remainder = divmod(window.integration_count, shard_count)
    shards = []
    first = window.first_integration
    for index in range(shard_count):
        integration_count = quotient + int(index < remainder)
        selected = CorrelationWindow(window.scan_id, first, integration_count)
        shards.append(TimeShard(index, shard_count, window, selected))
        first += integration_count
    return tuple(shards)


def select_time_shard(
    window: CorrelationWindow, shard_count: int, shard_index: int
) -> TimeShard:
    """Return one validated member of a deterministic shard plan."""

    if not isinstance(shard_index, int) or isinstance(shard_index, bool):
        raise TypeError("shard_index must be an integer")
    shards = plan_time_shards(window, shard_count)
    if shard_index < 0 or shard_index >= len(shards):
        raise ValueError("shard_index must be in [0, shard_count)")
    return shards[shard_index]


def load_ordered_shard_outputs(
    shard_directories: Sequence[str | os.PathLike[str]],
    *,
    expected_job_fingerprint: str | None = None,
    expected_source_snapshot: Sequence[tuple[int, int, int, int]]
    | None = None,
) -> tuple[tuple[Path, ...], tuple[TimeShard, ...]]:
    """Validate a complete shard set and return it in time order."""

    if not shard_directories:
        raise ValueError("at least one shard output is required")
    by_index: dict[int, tuple[Path, TimeShard]] = {}
    expected_count: int | None = None
    expected_parent: CorrelationWindow | None = None
    observed_fingerprint: str | None = None
    observed_profile: str | None = None
    observed_backend: str | None = None
    observed_contract = None
    observed_source_snapshot: tuple[tuple[int, int, int, int], ...] | None = (
        None
    )
    for value in shard_directories:
        directory = Path(value).expanduser().resolve(strict=True)
        manifest_path = directory / SHARD_MANIFEST_FILENAME
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise ValueError(
                f"shard output has no {SHARD_MANIFEST_FILENAME}: {directory}"
            ) from error
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid shard manifest: {manifest_path}"
            ) from error
        if not isinstance(manifest, dict):
            raise ValueError(
                f"shard manifest must be an object: {manifest_path}"
            )
        if manifest.get("status") != "complete":
            raise ValueError(f"shard output is not complete: {directory}")
        fingerprint = manifest.get("job_fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError(f"shard has no job fingerprint: {directory}")
        if observed_fingerprint is None:
            observed_fingerprint = fingerprint
        elif fingerprint != observed_fingerprint:
            raise ValueError(
                "shard outputs were produced from different jobs"
            )
        if (
            expected_job_fingerprint is not None
            and fingerprint != expected_job_fingerprint
        ):
            raise ValueError("shard output does not match the assembly job")
        profile = manifest.get("profile")
        backend = manifest.get("backend")
        mode = manifest.get("autocorrelation_mode")
        contract = manifest.get("execution_contract")
        if mode not in {
            "average",
            "truncation",
        } or contract != execution_contract(
            fingerprint, profile, backend, mode
        ):
            raise ValueError(
                "shard lacks a current numerical execution contract; "
                "regenerate shards"
            )
        if observed_contract is not None and contract != observed_contract:
            raise ValueError(
                "shards used different numerical execution contracts"
            )
        observed_contract = contract
        if not isinstance(profile, str) or not profile:
            raise ValueError(
                f"shard has no compatibility profile: {directory}"
            )
        if not isinstance(backend, str) or not backend:
            raise ValueError(f"shard has no backend identity: {directory}")
        if observed_profile is None:
            observed_profile = profile
            observed_backend = backend
        elif profile != observed_profile or backend != observed_backend:
            raise ValueError(
                "shard outputs used different profiles or backends"
            )
        source_snapshot = _portable_source_snapshot(manifest, directory)
        if observed_source_snapshot is None:
            observed_source_snapshot = source_snapshot
        elif source_snapshot != observed_source_snapshot:
            raise ValueError(
                "shard outputs used different recording source snapshots"
            )
        metadata = manifest.get("shard")
        if not isinstance(metadata, dict) or metadata.get("axis") != "time":
            raise ValueError(f"output is not a time shard: {directory}")
        try:
            index = int(metadata["index"])
            count = int(metadata["count"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"invalid shard identity: {directory}"
            ) from error
        if count <= 0:
            raise ValueError(f"invalid shard count: {directory}")
        if count != len(shard_directories):
            raise ValueError(
                "shard set is incomplete; declared "
                f"{count} shards but received {len(shard_directories)}"
            )
        if index < 0 or index >= count:
            raise ValueError(f"invalid shard index: {directory}")
        parent = _window_from_dict(
            metadata.get("parent_window"), field="parent_window"
        )
        selected = _window_from_dict(metadata.get("window"), field="window")
        shard = TimeShard(index, count, parent, selected)
        if index in by_index:
            raise ValueError(f"duplicate shard index {index}")
        if expected_count is None:
            expected_count = count
            expected_parent = parent
        elif count != expected_count or parent != expected_parent:
            raise ValueError("shard outputs do not share one parent plan")
        by_index[index] = (directory, shard)

    assert expected_count is not None
    assert expected_parent is not None
    assert observed_source_snapshot is not None
    if expected_source_snapshot is not None and not set(
        expected_source_snapshot
    ).issubset(observed_source_snapshot):
        raise ValueError(
            "shard recording sources do not match the assembly job"
        )
    if len(by_index) != expected_count or set(by_index) != set(
        range(expected_count)
    ):
        missing = sorted(set(range(expected_count)) - set(by_index))
        raise ValueError(
            f"shard set is incomplete; missing indices {missing}"
        )
    expected = plan_time_shards(expected_parent, expected_count)
    ordered = tuple(by_index[index] for index in range(expected_count))
    for (_directory, actual), planned in zip(ordered, expected, strict=True):
        if actual != planned:
            raise ValueError("shard window does not match its declared plan")
    return (
        tuple(directory for directory, _shard in ordered),
        tuple(shard for _directory, shard in ordered),
    )


def _portable_source_snapshot(
    manifest: dict[str, object], directory: Path
) -> tuple[tuple[int, int, int, int], ...]:
    """Return source identity fields that are stable across worker nodes."""

    sources = manifest.get("source_snapshot")
    if not isinstance(sources, list) or not sources:
        raise ValueError(f"shard has no source snapshot: {directory}")
    snapshot: list[tuple[int, int, int, int]] = []
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError(
                f"shard has invalid source snapshot: {directory}"
            )
        datastream_id = source.get("datastream_id")
        source_index = source.get("source_index")
        path = source.get("path")
        size_bytes = source.get("size_bytes")
        mtime_ns = source.get("mtime_ns")
        if (
            not isinstance(datastream_id, int)
            or not isinstance(source_index, int)
            or source_index < 0
            or not isinstance(path, str)
            or not path
            or not isinstance(size_bytes, int)
            or not isinstance(mtime_ns, int)
        ):
            raise ValueError(
                f"shard has invalid source snapshot: {directory}"
            )
        snapshot.append((datastream_id, source_index, size_bytes, mtime_ns))
    return tuple(sorted(snapshot))


def _integrity(shard):
    document = json.loads((shard / SHARD_MANIFEST_FILENAME).read_text())
    integrity = document.get("output_integrity")
    if (
        not isinstance(integrity, dict)
        or integrity.get("version") != 1
        or not isinstance(integrity.get("record_count"), int)
        or integrity["record_count"] < 0
        or not isinstance(integrity.get("files"), dict)
    ):
        raise ValueError(
            "shard lacks output integrity metadata; regenerate shards"
        )
    return integrity


def _verify_digest(expected, digest, size, path):
    if expected != {"size_bytes": size, "sha256": digest.hexdigest()}:
        raise ValueError(f"output integrity mismatch: {path}")


def _copy_verified(source, target, expected):
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as stream:
        while block := stream.read(8 << 20):
            target.write(block)
            digest.update(block)
            size += len(block)
    _verify_digest(expected, digest, size, source)
    return size


def _swin_window(job, shard):
    manifest = json.loads((shard / SHARD_MANIFEST_FILENAME).read_text())
    window = manifest["shard"]["window"]
    scan = next(
        (s for s in job.scans if s.identifier == window["scan_id"]), None
    )
    if scan is None:
        raise ValueError(
            f"unknown scan {window['scan_id']!r} in SWIN shard {shard}"
        )
    origin = max(Decimal(0), scan.start_offset_s)
    return origin, window["first_integration"], window["integration_count"]


def _check_swin_window(record, job, bounds):
    origin, first, count = bounds
    elapsed = (
        Decimal(record.mjd - job.start_time.mjd) * 86400
        + Decimal(str(record.seconds))
        - job.start_time.seconds_of_day
    )
    index = (elapsed - origin) / job.integration_duration_s - Decimal("0.5")
    nearest = int(index.to_integral_value())
    if (
        abs(index - nearest) > Decimal("0.000001")
        or not first <= nearest < first + count
    ):
        raise ValueError("SWIN record outside declared shard window")


def _assembly(function):
    """Own temporaries and serialize no-overwrite publication."""

    @wraps(function)
    def run(shards, output, *args, **kwargs):
        lock = output / ".assembly.lock"
        # A failed exclusive open must not unlink another writer's lock.
        descriptor = os.open(
            lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
        )
        promoted = []
        try:
            with TemporaryDirectory(
                prefix=".assembly-", dir=output
            ) as directory:
                temporary = Path(directory)
                result = function(shards, temporary, *args, **kwargs)
                files = {}
                for path in temporary.iterdir():
                    with path.open("rb") as stream:
                        digest = hashlib.file_digest(stream, "sha256")
                    files[path.name] = {
                        "size_bytes": path.stat().st_size,
                        "sha256": digest.hexdigest(),
                    }
                inputs = [_integrity(shard) for shard in shards]
                integrity = {
                    "version": 1,
                    "record_count": sum(
                        item["record_count"] for item in inputs
                    ),
                    "files": files,
                    "channels_by_frequency": {
                        key: count
                        for item in inputs
                        for key, count in item.get(
                            "channels_by_frequency", {}
                        ).items()
                    },
                }
                (temporary / ".assembly-integrity.json").write_text(
                    json.dumps(integrity)
                )
                for source in sorted(temporary.iterdir()):
                    with source.open("rb") as stream:
                        os.fsync(stream.fileno())
                    destination = output / source.name
                    try:
                        os.link(source, destination)
                    except FileExistsError as error:
                        raise FileExistsError(
                            f"shard assembly already exists: {destination}"
                        ) from error
                    promoted.append(destination)
                directory_fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
                return result
        except BaseException:
            for destination in reversed(promoted):
                destination.unlink(missing_ok=True)
            raise
        finally:
            os.close(descriptor)
            lock.unlink()

    return run


@_assembly
def merge_native_shards(shards: Sequence[Path], output: Path) -> int:
    """Assemble native shard streams without materializing spectra."""

    if not shards:
        raise ValueError("at least one shard output is required")
    metadata_destination = output / METADATA_FILENAME
    spectra_destination = output / SPECTRA_FILENAME
    metadata_partial = output / f"{METADATA_FILENAME}.partial"
    spectra_partial = output / f"{SPECTRA_FILENAME}.partial"
    if any(
        path.exists()
        for path in (
            metadata_destination,
            spectra_destination,
            metadata_partial,
            spectra_partial,
        )
    ):
        raise FileExistsError("native shard assembly already exists")
    global_offset = 0
    record_count = 0
    promoted: list[Path] = []
    try:
        with metadata_partial.open(
            "x", encoding="utf-8", newline="\n"
        ) as metadata_out:
            with spectra_partial.open("xb") as spectra_out:
                for shard in shards:
                    integrity = _integrity(shard)
                    first_record = record_count
                    metadata_hash = hashlib.sha256()
                    metadata_size = 0
                    local_offset = 0
                    metadata_path = shard / METADATA_FILENAME
                    spectra_path = shard / SPECTRA_FILENAME
                    with metadata_path.open("rb") as metadata_in:
                        for line_number, line in enumerate(metadata_in, 1):
                            metadata_hash.update(line)
                            metadata_size += len(line)
                            try:
                                item = json.loads(line)
                            except json.JSONDecodeError as error:
                                raise ValueError(
                                    f"invalid metadata in {metadata_path}:"
                                    f"{line_number}"
                                ) from error
                            if not isinstance(item, dict):
                                raise ValueError(
                                    "invalid metadata object in "
                                    f"{metadata_path}:{line_number}"
                                )
                            offset = item.get("spectrum_offset")
                            channels = item.get("channel_count")
                            if item.get("format") != FORMAT_NAME:
                                raise ValueError(
                                    "unsupported native output in "
                                    f"{metadata_path}"
                                )
                            if offset != local_offset:
                                raise ValueError(
                                    "non-contiguous spectra in "
                                    f"{metadata_path}"
                                )
                            if not isinstance(channels, int) or channels < 0:
                                raise ValueError(
                                    "invalid channel count in "
                                    f"{metadata_path}"
                                )
                            item["spectrum_offset"] = (
                                global_offset + local_offset
                            )
                            metadata_out.write(
                                json.dumps(
                                    item,
                                    separators=(",", ":"),
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            local_offset += (
                                channels * np.dtype("<c8").itemsize
                            )
                            record_count += 1
                    _verify_digest(
                        integrity["files"].get(METADATA_FILENAME),
                        metadata_hash,
                        metadata_size,
                        metadata_path,
                    )
                    if (
                        record_count - first_record
                        != integrity["record_count"]
                    ):
                        raise ValueError("native shard record count mismatch")
                    if spectra_path.stat().st_size != local_offset:
                        raise ValueError(
                            "spectra size disagrees with metadata: "
                            f"{spectra_path}"
                        )
                    _copy_verified(
                        spectra_path,
                        spectra_out,
                        integrity["files"].get(SPECTRA_FILENAME),
                    )
                    global_offset += local_offset
        os.replace(spectra_partial, spectra_destination)
        promoted.append(spectra_destination)
        os.replace(metadata_partial, metadata_destination)
        promoted.append(metadata_destination)
    except BaseException:
        metadata_partial.unlink(missing_ok=True)
        spectra_partial.unlink(missing_ok=True)
        for destination in reversed(promoted):
            destination.unlink(missing_ok=True)
        raise
    return record_count


def _fixed_swin_copy(
    shard, filename, target_fd, offset, integrity, channels, job, cancelled
):
    """Validate bounded blocks and copy a shard to its exclusive range."""
    source = shard / filename
    channel_count = next(iter(channels.values()))
    record_size = HEADER.size + channel_count * 8
    # Offsets follow the little-endian HEADER struct without alignment.
    dtype = np.dtype(
        {
            "names": [
                "sync",
                "version",
                "baseline",
                "mjd",
                "seconds",
                "frequency",
                "pol",
            ],
            "formats": ["<u4", "<u4", "<i4", "<u4", "<f8", "<i4", ("u1", 2)],
            "offsets": [0, 4, 8, 12, 16, 32, 36],
            "itemsize": record_size,
        }
    )
    bounds = _swin_window(job, shard) if job is not None else None
    digest = hashlib.sha256()
    size = count = 0
    block_size = max(1, (8 << 20) // record_size) * record_size
    if cancelled.is_set():
        return 0
    with source.open("rb") as stream:
        while not cancelled.is_set():
            block = stream.read(block_size)
            if not block:
                break
            if len(block) % record_size:
                raise ValueError(
                    f"{source}: truncated SWIN record at byte "
                    f"{size + len(block) // record_size * record_size}"
                )
            headers = np.frombuffer(block, dtype=dtype)
            baseline = headers["baseline"]
            checks = (
                ("sync word", headers["sync"] != SYNC_WORD),
                ("version", headers["version"] != FORMAT_VERSION),
                (
                    "baseline",
                    (baseline // 256 < 1)
                    | (baseline // 256 > 255)
                    | (baseline % 256 < 1),
                ),
                ("polarization", np.any(headers["pol"] > 127, axis=1)),
            )
            for field, invalid in checks:
                if np.any(invalid):
                    position = int(np.flatnonzero(invalid)[0])
                    raise ValueError(
                        f"{source}: invalid SWIN {field} at byte "
                        f"{size + position * record_size}"
                    )
            for frequency in np.unique(headers["frequency"]):
                if str(int(frequency)) not in channels:
                    position = int(
                        np.flatnonzero(headers["frequency"] == frequency)[0]
                    )
                    raise ValueError(
                        f"{source}: missing SWIN channel geometry for "
                        f"frequency {frequency} at byte "
                        f"{size + position * record_size}"
                    )
            if job is not None:
                # Apply the existing Decimal rule to each distinct timestamp.
                times = np.empty(
                    len(headers), dtype=[("mjd", "<u4"), ("seconds", "<f8")]
                )
                times["mjd"] = headers["mjd"]
                times["seconds"] = headers["seconds"]
                times = np.unique(times)
                for timestamp in times:
                    _check_swin_window(
                        SimpleNamespace(
                            mjd=int(timestamp["mjd"]),
                            seconds=float(timestamp["seconds"]),
                        ),
                        job,
                        bounds,
                    )
            view = memoryview(block)
            written = 0
            while written < len(view):
                if cancelled.is_set():
                    return 0
                n = os.pwrite(
                    target_fd, view[written:], offset + size + written
                )
                if n <= 0:
                    raise OSError("zero-byte SWIN merge write")
                written += n
            digest.update(block)
            size += len(block)
            count += len(headers)
    if cancelled.is_set():
        return 0
    _verify_digest(integrity["files"].get(filename), digest, size, source)
    if count != integrity["record_count"]:
        raise ValueError("SWIN shard record count mismatch")
    return size


def _fixed_swin_plan(shards, filename, job):
    plan = []
    offset = 0
    for shard in shards:
        integrity = _integrity(shard)
        channels = integrity.get("channels_by_frequency", {})
        if job is not None:
            job_channels = {
                str(f.identifier): f.channel_count // f.channels_to_average
                for f in job.frequencies
            }
            # Parent bands may differ in size but never appear in the output.
            # Use declared output bands only, checked against the loaded job.
            if any(job_channels.get(key) != n for key, n in channels.items()):
                return None
        if not channels or any(
            type(n) is not int or n <= 0 for n in channels.values()
        ):
            return None
        if len(set(channels.values())) != 1:
            return None
        size = (shard / filename).stat().st_size
        expected = integrity["files"].get(filename)
        if (
            not isinstance(expected, dict)
            or expected.get("size_bytes") != size
        ):
            raise ValueError(f"output integrity mismatch: {shard / filename}")
        plan.append((shard, offset, integrity, channels))
        offset += size
    return plan


@_assembly
def merge_swin_shards(
    shards: Sequence[Path],
    output: Path,
    filename: str,
    *,
    job=None,
    workers=4,
) -> int:
    """Concatenate independently framed SWIN streams in shard order."""

    if not shards:
        raise ValueError("at least one shard output is required")
    if type(workers) is not int or workers <= 0:
        raise ValueError("merge workers must be a positive integer")
    if job is None:
        declared_geometry: dict[str, int] = {}
        for shard in shards:
            for key, count in (
                _integrity(shard).get("channels_by_frequency", {}).items()
            ):
                if (
                    key in declared_geometry
                    and declared_geometry[key] != count
                ):
                    raise ValueError(
                        "SWIN channel geometry disagrees between shards"
                    )
                declared_geometry[key] = count
    destination = output / filename
    partial = output / f"{filename}.partial"
    if destination.exists() or partial.exists():
        raise FileExistsError(
            f"SWIN shard assembly already exists: {destination}"
        )
    byte_count = 0
    try:
        with partial.open("xb") as target:
            plan = _fixed_swin_plan(shards, filename, job)
            if plan is not None:
                cancelled = Event()
                pool = ThreadPoolExecutor(max_workers=min(workers, len(plan)))
                try:
                    futures = [
                        pool.submit(
                            _fixed_swin_copy,
                            shard,
                            filename,
                            target.fileno(),
                            offset,
                            integrity,
                            channels,
                            job,
                            cancelled,
                        )
                        for shard, offset, integrity, channels in plan
                    ]
                    for future in as_completed(futures):
                        byte_count += future.result()
                except BaseException:
                    cancelled.set()
                    raise
                finally:
                    # Stop queued work and join active writers before cleanup.
                    pool.shutdown(wait=True, cancel_futures=True)
                target.flush()
                os.replace(partial, destination)
                return byte_count
            for shard in shards:
                source = shard / filename
                bounds = _swin_window(job, shard) if job is not None else None
                integrity = _integrity(shard)
                declared_channels = integrity.get("channels_by_frequency", {})
                channels = declared_channels
                if job is not None:
                    channels = {
                        str(f.identifier): f.channel_count
                        // f.channels_to_average
                        for f in job.frequencies
                    }
                    if any(
                        channels.get(key) != value
                        for key, value in declared_channels.items()
                    ):
                        raise ValueError(
                            f"SWIN channel geometry disagrees with job: "
                            f"{source}"
                        )
                digest = hashlib.sha256()
                size = count = 0
                with source.open("rb") as stream:
                    while header := stream.read(HEADER.size):
                        if len(header) != HEADER.size:
                            raise ValueError("truncated SWIN header")
                        frequency = str(HEADER.unpack(header)[7])
                        if frequency not in declared_channels:
                            raise ValueError(
                                f"{source}: missing SWIN channel geometry "
                                "for "
                                f"frequency {frequency} at byte {size}"
                            )
                        channel_count = channels.get(frequency)
                        if (
                            not isinstance(channel_count, int)
                            or channel_count <= 0
                        ):
                            raise ValueError("missing SWIN channel geometry")
                        payload = stream.read(channel_count * 8)
                        raw = header + payload
                        from io import BytesIO

                        record = read_record(BytesIO(raw), channel_count)
                        if job is not None:
                            _check_swin_window(record, job, bounds)
                        target.write(raw)
                        digest.update(raw)
                        size += len(raw)
                        count += 1
                _verify_digest(
                    integrity["files"].get(filename), digest, size, source
                )
                if count != integrity["record_count"]:
                    raise ValueError("SWIN shard record count mismatch")
                byte_count += size
        os.replace(partial, destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return byte_count


__all__ = [
    "SHARD_MANIFEST_FILENAME",
    "TimeShard",
    "load_ordered_shard_outputs",
    "merge_native_shards",
    "merge_swin_shards",
    "plan_time_shards",
    "select_time_shard",
]
