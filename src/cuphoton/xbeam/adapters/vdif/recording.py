# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Filesystem adapters for the neutral recording-source contract."""

from __future__ import annotations

import math
import os
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import numpy as np

from cuphoton.xbeam.adapters.radio.models import (
    Datastream,
    Frequency,
    Job,
    MJDTime,
    PackedVoltageLayout,
    RecordingSource,
    SamplingKind,
    VoltageBatch,
)
from cuphoton.xbeam.adapters.vdif.vdif import (
    HEADER_BYTES,
    DuplicatePolicy,
    DuplicateVdifFrameError,
    TruncatedVdifFrameError,
    VdifError,
    VdifHeader,
    absolute_frame_slot,
    seconds_since_vdif_origin,
    vdif_seconds_to_mjd_seconds,
)
from cuphoton.xbeam.adapters.vdif.vdif_index import (
    VdifThreadMetadata,
    VdifTimestampKey,
    timestamp_ranges_for_slots,
)
from cuphoton.xbeam.adapters.vdif.vdif_sidecar import (
    binary_index_reference_epochs,
    read_vdif_binary_frames,
    read_vdif_binary_physical_frames,
    validate_vdif_binary_index,
)
from cuphoton.xbeam.errors import ConfigurationError
from cuphoton.xbeam.profiles import CompatibilityProfile
from cuphoton.xbeam.profiling import nvtx_range

_VDIF_HEADER_SCAN_CHUNK_BYTES = 64 * 1024 * 1024


def _sample_rate(datastream: Datastream, frequency: Frequency) -> float:
    if datastream.sample_rate_hz is not None:
        return datastream.sample_rate_hz
    factor = 2.0 if datastream.sampling is SamplingKind.REAL else 1.0
    return frequency.bandwidth_hz * factor


def _mjd_time(absolute_seconds: Decimal) -> MJDTime:
    mjd = int(absolute_seconds // Decimal(86400))
    return MJDTime(mjd, absolute_seconds - Decimal(mjd * 86400))


def _timeline_slot_limit(
    execution_duration_s: Decimal,
    frames_per_second: int,
    observed_frame_count: int,
) -> int:
    """Bound sparse timeline expansion before allocating gap-filled arrays."""

    declared_frames = math.ceil(
        execution_duration_s * Decimal(frames_per_second)
    )
    return max(declared_frames, observed_frame_count * 4, 1)


def _text_station_id(name: str) -> int | None:
    try:
        encoded = name.encode("ascii")
    except UnicodeEncodeError:
        return None
    if len(encoded) != 2:
        return None
    return (encoded[0] << 8) | encoded[1]


@dataclass(frozen=True, slots=True)
class _BulkVdifFrames:
    words: np.ndarray
    source_indices: np.ndarray
    offsets: np.ndarray
    paths: tuple[Path, ...]
    file_frame_starts: tuple[int, ...]
    file_frame_counts: tuple[int, ...]
    file_sizes: tuple[int, ...]
    file_mtimes_ns: tuple[int, ...]
    file_devices: tuple[int, ...]
    file_inodes: tuple[int, ...]

    @property
    def count(self) -> int:
        return int(self.words.shape[0])


@dataclass(frozen=True, slots=True)
class _BinarySidecarFrames:
    """Compact sidecar entries needed to build static payload mappings."""

    timestamps: np.ndarray
    thread_ids: np.ndarray
    source_indices: np.ndarray
    physical_frames: np.ndarray
    invalid: np.ndarray
    paths: tuple[Path, ...]
    file_frame_starts: tuple[int, ...]
    file_frame_counts: tuple[int, ...]
    file_sizes: tuple[int, ...]
    file_mtimes_ns: tuple[int, ...]
    file_devices: tuple[int, ...]
    file_inodes: tuple[int, ...]
    thread_metadata: tuple[VdifThreadMetadata, ...]

    @property
    def count(self) -> int:
        return int(self.timestamps.size)


@dataclass(frozen=True, slots=True)
class VdifSourceRange:
    """A bounded contiguous range of raw frames within one VDIF file."""

    path: Path
    first_frame: int
    frame_count: int
    expected_size_bytes: int | None = None
    expected_mtime_ns: int | None = None
    expected_device: int | None = None
    expected_inode: int | None = None

    @property
    def stop_frame(self) -> int:
        return self.first_frame + self.frame_count


@dataclass(frozen=True, slots=True)
class VdifThreadWindow:
    """Frame-grid coverage required for one thread in a bounded run."""

    template: VdifHeader
    frames_per_second: int
    start_slot: int
    stop_slot: int


@dataclass(frozen=True, slots=True)
class VdifPayloadFilePlan:
    """One validated raw VDIF file and its raw-frame destination mapping."""

    path: Path
    frame_count: int
    destination_slots: np.ndarray | None
    first_frame: int = 0
    expected_size_bytes: int | None = None
    expected_mtime_ns: int | None = None
    expected_device: int | None = None
    expected_inode: int | None = None
    # Optional sparse physical-frame coordinates parallel to
    # ``destination_slots``.  Coordinates are relative to ``first_frame``.
    # The ordinary dense form leaves this unset and has one destination per
    # raw frame.  Sparse mappings may select frames out of order or select
    # one raw frame more than once.
    source_frame_indices: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class VdifStreamingPlan:
    """Header-derived mapping metadata for one ordered VDIF thread."""

    thread_id: int
    frames_per_second: int
    start_slot: int
    template: VdifHeader
    duplicate_policy: DuplicatePolicy
    frame_validity: np.ndarray


@dataclass(frozen=True, slots=True)
class VdifPayloadPlan:
    """Lazy payload plan shared by bands on one VDIF thread."""

    frame_size_bytes: int
    payload_bytes: int
    slot_count: int
    files: tuple[VdifPayloadFilePlan, ...]
    streaming: VdifStreamingPlan | None = None

    @property
    def byte_count(self) -> int:
        return self.slot_count * self.payload_bytes

    def materialize(self) -> np.ndarray:
        if self.streaming is not None:
            raise RuntimeError(
                "streaming VDIF plans can only be materialized by the GPU "
                "uploader"
            )
        return materialize_vdif_payload_plans([self])[0]

    def __array__(
        self, dtype: Any = None, copy: bool | None = None
    ) -> np.ndarray:
        result = self.materialize()
        if dtype is not None:
            result = result.astype(dtype, copy=False)
        if copy:
            result = result.copy()
        return result


def _canonical_payload_file_mapping(
    frame_count: int,
    source_frame_indices: np.ndarray,
    destination_slots: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Use the dense ingest form unless a physical frame is reused."""

    dense = np.full(frame_count, -1, dtype=np.int32)
    dense[source_frame_indices] = destination_slots
    if np.count_nonzero(dense >= 0) == destination_slots.size:
        return dense, None
    return (
        np.asarray(destination_slots, dtype=np.int32),
        np.asarray(source_frame_indices, dtype=np.int64),
    )


def _sidecar_query_bounds(
    windows: dict[int, VdifThreadWindow],
    epochs: tuple[int, ...],
) -> tuple[tuple[VdifTimestampKey, VdifTimestampKey], ...]:
    """Merge raw query envelopes after translating each recorded epoch."""
    ranges = sorted(
        bounds
        for window in windows.values()
        for bounds in timestamp_ranges_for_slots(
            window.start_slot,
            window.stop_slot,
            window.frames_per_second,
            epochs,
        )
    )
    merged: list[tuple[VdifTimestampKey, VdifTimestampKey]] = []
    for start, stop in ranges:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(stop, merged[-1][1]))
        else:
            merged.append((start, stop))
    return tuple(merged)


def _binary_sidecar_frames(
    datastream: Datastream,
    requested: tuple[VdifSourceRange, ...],
    windows: dict[int, VdifThreadWindow],
    index_dir: Path | None,
    physical_presentation: (
        tuple[tuple[int, int, int, int], ...] | None
    ) = None,
) -> _BinarySidecarFrames:
    """Load compact entries without reconstructing synthetic VDIF headers."""

    timestamp_parts: list[np.ndarray] = []
    thread_parts: list[np.ndarray] = []
    source_parts: list[np.ndarray] = []
    physical_parts: list[np.ndarray] = []
    invalid_parts: list[np.ndarray] = []
    paths: list[Path] = []
    starts: list[int] = []
    counts: list[int] = []
    sizes: list[int] = []
    mtimes_ns: list[int] = []
    devices: list[int] = []
    inodes: list[int] = []
    metadata_by_thread: dict[int, VdifThreadMetadata] = {}
    source_entries = {}

    for source_index, item in enumerate(requested):
        summary = validate_vdif_binary_index(
            item.path,
            frame_size_bytes=datastream.frame_size_bytes,
            index_dir=index_dir,
        )
        stat = item.path.stat()
        _validate_source_identity(
            item.path,
            (
                stat.st_size
                if item.expected_size_bytes is None
                else item.expected_size_bytes
            ),
            (
                stat.st_mtime_ns
                if item.expected_mtime_ns is None
                else item.expected_mtime_ns
            ),
            (
                stat.st_dev
                if item.expected_device is None
                else item.expected_device
            ),
            (
                stat.st_ino
                if item.expected_inode is None
                else item.expected_inode
            ),
        )
        key = (
            item.path,
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )
        if key not in source_entries:
            if physical_presentation is None:
                parts = [
                    read_vdif_binary_frames(
                        item.path,
                        start_key,
                        stop_key,
                        frame_size_bytes=datastream.frame_size_bytes,
                        index_dir=index_dir,
                    )
                    for start_key, stop_key in _sidecar_query_bounds(
                        windows, binary_index_reference_epochs(summary)
                    )
                ]
                arrays = (
                    tuple(
                        np.concatenate(values)
                        for values in zip(*parts, strict=True)
                    )
                    if parts
                    else (
                        np.empty(0, dtype=np.uint64),
                        np.empty(0, dtype=np.int16),
                        np.empty(0, dtype=np.uint64),
                        np.empty(0, dtype=np.bool_),
                    )
                )
            else:
                ranges = []
                source_indices = [
                    source_index
                    for source_index, source_path in enumerate(
                        datastream.source_files
                    )
                    if Path(source_path).expanduser().resolve() == item.path
                ]
                if len(source_indices) != 1:
                    raise VdifError(
                        "physical frame presentation requires each source "
                        "path exactly once in the datastream"
                    )
                # Presentation indices address datastream.source_files;
                # frames.source_indices address bounded requested ranges.
                presentation_source_index = source_indices[0]
                for requested_item in requested:
                    if requested_item.path != item.path:
                        continue
                    for (
                        interval_source_index,
                        start,
                        stop,
                        _destination,
                    ) in physical_presentation:
                        if interval_source_index != presentation_source_index:
                            continue
                        clipped_start = max(start, requested_item.first_frame)
                        clipped_stop = min(stop, requested_item.stop_frame)
                        if clipped_start < clipped_stop:
                            ranges.append((clipped_start, clipped_stop))
                if not ranges:
                    raise VdifError(
                        "physical frame presentation does not cover its "
                        "bounded source range"
                    )
                arrays = read_vdif_binary_physical_frames(
                    item.path,
                    ranges,
                    frame_size_bytes=datastream.frame_size_bytes,
                    index_dir=index_dir,
                )
            # Older CPU sidecars sort equal-time entries by thread/validity.
            # Restore physical order before slicing ranges and resolving
            # FIRST/LAST duplicates, including for ordered source files.
            if np.any(arrays[2][1:] < arrays[2][:-1]):
                order = np.argsort(arrays[2], kind="stable")
                arrays = (
                    arrays[0][order],
                    arrays[1][order],
                    arrays[2][order],
                    arrays[3][order],
                )
            source_entries[key] = arrays
        arrays = source_entries[key]
        first = int(np.searchsorted(arrays[2], item.first_frame, side="left"))
        stop = int(np.searchsorted(arrays[2], item.stop_frame, side="left"))
        timestamps, thread_ids, physical, invalid = (
            array[first:stop] for array in arrays
        )
        for item_metadata in summary.thread_metadata:
            existing = metadata_by_thread.get(item_metadata.thread_id)
            if existing is not None and existing != item_metadata:
                raise VdifError(
                    f"inconsistent VDIF sidecar metadata for thread "
                    f"{item_metadata.thread_id}"
                )
            metadata_by_thread[item_metadata.thread_id] = item_metadata
        timestamp_parts.append(timestamps)
        thread_parts.append(thread_ids)
        source_parts.append(
            np.full(physical.size, source_index, dtype=np.int32)
        )
        physical_parts.append(physical)
        invalid_parts.append(invalid)
        paths.append(item.path)
        starts.append(item.first_frame)
        counts.append(item.frame_count)
        sizes.append(stat.st_size)
        mtimes_ns.append(stat.st_mtime_ns)
        devices.append(stat.st_dev)
        inodes.append(stat.st_ino)

    return _BinarySidecarFrames(
        (
            np.concatenate(timestamp_parts)
            if timestamp_parts
            else np.empty(0, dtype=np.uint64)
        ),
        (
            np.concatenate(thread_parts)
            if thread_parts
            else np.empty(0, dtype=np.int16)
        ),
        (
            np.concatenate(source_parts)
            if source_parts
            else np.empty(0, dtype=np.int32)
        ),
        (
            np.concatenate(physical_parts)
            if physical_parts
            else np.empty(0, dtype=np.uint64)
        ),
        (
            np.concatenate(invalid_parts)
            if invalid_parts
            else np.empty(0, dtype=np.bool_)
        ),
        tuple(paths),
        tuple(starts),
        tuple(counts),
        tuple(sizes),
        tuple(mtimes_ns),
        tuple(devices),
        tuple(inodes),
        tuple(metadata_by_thread[key] for key in sorted(metadata_by_thread)),
    )


def materialize_vdif_payload_plans(
    plans: list[VdifPayloadPlan],
    *,
    target_chunk_bytes: int = 64 * 1024 * 1024,
) -> list[np.ndarray]:
    """Read shared raw extents once and scatter payloads to thread plans."""

    if not plans:
        return []
    if target_chunk_bytes <= 0:
        raise ValueError("target_chunk_bytes must be positive")
    if any(plan.streaming is not None for plan in plans):
        raise RuntimeError(
            "streaming VDIF plans can only be materialized by the GPU "
            "uploader"
        )
    first = plans[0]
    reference_files = tuple(
        (
            item.path,
            item.first_frame,
            item.frame_count,
            item.expected_size_bytes,
            item.expected_mtime_ns,
            item.expected_device,
            item.expected_inode,
        )
        for item in first.files
    )
    for plan in plans:
        files = tuple(
            (
                item.path,
                item.first_frame,
                item.frame_count,
                item.expected_size_bytes,
                item.expected_mtime_ns,
                item.expected_device,
                item.expected_inode,
            )
            for item in plan.files
        )
        if (
            plan.frame_size_bytes != first.frame_size_bytes
            or plan.payload_bytes != first.payload_bytes
            or files != reference_files
        ):
            raise ValueError(
                "bulk VDIF plans must describe the same raw extents"
            )
        if plan.slot_count <= 0:
            raise ValueError("VDIF payload plan slot count must be positive")
        used_destinations = np.zeros(plan.slot_count, dtype=np.bool_)
        for item in plan.files:
            mapping = np.asarray(item.destination_slots)
            source_frames = item.source_frame_indices
            expected_shape = (
                (item.frame_count,)
                if source_frames is None
                else np.asarray(source_frames).shape
            )
            if (
                mapping.dtype.kind != "i"
                or mapping.ndim != 1
                or mapping.shape != expected_shape
            ):
                raise TypeError(
                    "VDIF destination slots must be a one-dimensional "
                    "signed integer mapping"
                )
            if source_frames is not None:
                source_frames = np.asarray(source_frames)
                if source_frames.dtype.kind != "i" or source_frames.ndim != 1:
                    raise TypeError(
                        "VDIF source frame indices must be one-dimensional "
                        "signed integers"
                    )
                if np.any(
                    (source_frames < 0) | (source_frames >= item.frame_count)
                ):
                    raise ValueError(
                        "VDIF source frame index is out of range"
                    )
            if np.any((mapping < -1) | (mapping >= plan.slot_count)):
                raise ValueError("VDIF destination slot is out of range")
            destinations = mapping[mapping >= 0]
            unique_destinations = np.unique(destinations)
            if unique_destinations.size != destinations.size or np.any(
                used_destinations[unique_destinations]
            ):
                raise ValueError("duplicate VDIF destination slot")
            used_destinations[unique_destinations] = True

    outputs = [np.zeros(plan.byte_count, dtype=np.uint8) for plan in plans]
    output_rows = [
        output.reshape(plan.slot_count, plan.payload_bytes)
        for output, plan in zip(outputs, plans, strict=True)
    ]
    chunk_frames = max(1, target_chunk_bytes // first.frame_size_bytes)
    for file_index, (
        path,
        first_frame,
        frame_count,
        expected_size_bytes,
        expected_mtime_ns,
        expected_device,
        expected_inode,
    ) in enumerate(reference_files):
        selected_parts = []
        for plan in plans:
            item = plan.files[file_index]
            mapping = np.asarray(item.destination_slots)
            if item.source_frame_indices is None:
                selected = np.flatnonzero(mapping >= 0)
            else:
                source_frames = np.asarray(item.source_frame_indices)
                selected = source_frames[mapping >= 0]
            selected_parts.append(selected)
        selected_parts = [part for part in selected_parts if part.size]
        if not selected_parts:
            continue
        read_start = min(int(part.min()) for part in selected_parts)
        read_stop = max(int(part.max()) + 1 for part in selected_parts)
        if read_stop > frame_count:
            raise ValueError(
                "VDIF destination mapping exceeds its file frame range"
            )
        with path.open("rb", buffering=0) as stream:
            _validate_open_source_identity(
                stream.fileno(),
                path,
                expected_size_bytes,
                expected_mtime_ns,
                expected_device,
                expected_inode,
            )
            stream.seek((first_frame + read_start) * first.frame_size_bytes)
            cursor = read_start
            while cursor < read_stop:
                count = min(chunk_frames, read_stop - cursor)
                raw = bytearray(count * first.frame_size_bytes)
                received = 0
                view = memoryview(raw)
                while received < len(raw):
                    count_read = stream.readinto(view[received:])
                    if not count_read:
                        raise OSError(
                            "VDIF file changed during host ingest: expected "
                            f"{len(raw)} bytes, got {received}"
                        )
                    received += count_read
                rows = np.frombuffer(raw, dtype=np.uint8).reshape(
                    count, first.frame_size_bytes
                )
                for plan, rows_out in zip(plans, output_rows, strict=True):
                    item = plan.files[file_index]
                    mapping = np.asarray(item.destination_slots)
                    if item.source_frame_indices is None:
                        destinations = mapping[cursor : cursor + count]
                        selected = np.flatnonzero(destinations >= 0)
                        if selected.size:
                            rows_out[destinations[selected]] = rows[
                                selected, HEADER_BYTES:
                            ]
                        continue
                    source_frames = np.asarray(item.source_frame_indices)
                    selected = np.flatnonzero(
                        (mapping >= 0)
                        & (source_frames >= cursor)
                        & (source_frames < cursor + count)
                    )
                    if selected.size:
                        rows_out[mapping[selected]] = rows[
                            source_frames[selected] - cursor, HEADER_BYTES:
                        ]
                cursor += count
            _validate_open_source_identity(
                stream.fileno(),
                path,
                expected_size_bytes,
                expected_mtime_ns,
                expected_device,
                expected_inode,
            )
    return outputs


def _frame_size_error(
    datastream: Datastream, source_index: int, observed: int
) -> ConfigurationError:
    return ConfigurationError(
        f"datastream[{datastream.identifier}]",
        "frame_size_bytes",
        f"declares {datastream.frame_size_bytes} bytes but "
        f"source_files[{source_index}] contains a {observed}-byte VDIF frame",
    )


def _file_frame_count(
    path: Path, datastream: Datastream, source_index: int
) -> int:
    frame_size = datastream.frame_size_bytes
    if frame_size <= HEADER_BYTES or frame_size % 8:
        raise ConfigurationError(
            f"datastream[{datastream.identifier}]",
            "frame_size_bytes",
            "must exceed the standard header and be a multiple of eight",
        )
    size = path.stat().st_size
    if size:
        with path.open("rb") as stream:
            first_header = VdifHeader.from_bytes(stream.read(HEADER_BYTES))
        if first_header.frame_length_bytes != frame_size:
            raise _frame_size_error(
                datastream, source_index, first_header.frame_length_bytes
            )
    count, remainder = divmod(size, frame_size)
    if not remainder:
        return count
    offset = count * frame_size
    with path.open("rb") as stream:
        stream.seek(offset)
        tail = stream.read()
    if remainder < HEADER_BYTES:
        raise TruncatedVdifFrameError(
            f"truncated VDIF header at byte {offset}: got {remainder} bytes"
        )
    header = VdifHeader.from_bytes(tail[:HEADER_BYTES])
    if remainder >= header.frame_length_bytes:
        raise _frame_size_error(
            datastream, source_index, header.frame_length_bytes
        )
    available = remainder - HEADER_BYTES
    expected = header.frame_length_bytes - HEADER_BYTES
    raise TruncatedVdifFrameError(
        f"truncated VDIF frame at byte {offset}: expected {expected} payload "
        f"bytes, got {available}"
    )


def _validate_source_identity(
    path: Path,
    expected_size_bytes: int | None,
    expected_mtime_ns: int | None,
    expected_device: int | None = None,
    expected_inode: int | None = None,
) -> tuple[int, int, int, int]:
    stat = path.stat()
    _validate_stat_identity(
        stat,
        path,
        expected_size_bytes,
        expected_mtime_ns,
        expected_device,
        expected_inode,
    )
    return stat.st_size, stat.st_mtime_ns, stat.st_dev, stat.st_ino


def _validate_stat_identity(
    stat: os.stat_result,
    path: Path,
    expected_size_bytes: int | None,
    expected_mtime_ns: int | None,
    expected_device: int | None,
    expected_inode: int | None,
) -> None:
    if (
        (
            expected_size_bytes is not None
            and stat.st_size != expected_size_bytes
        )
        or (
            expected_mtime_ns is not None
            and stat.st_mtime_ns != expected_mtime_ns
        )
        or (expected_device is not None and stat.st_dev != expected_device)
        or (expected_inode is not None and stat.st_ino != expected_inode)
    ):
        raise OSError(f"VDIF source changed after ingest planning: {path}")


def _validate_open_source_identity(
    fd: int,
    path: Path,
    expected_size_bytes: int | None,
    expected_mtime_ns: int | None,
    expected_device: int | None,
    expected_inode: int | None,
) -> None:
    _validate_stat_identity(
        os.fstat(fd),
        path,
        expected_size_bytes,
        expected_mtime_ns,
        expected_device,
        expected_inode,
    )


def _source_ranges(
    datastream: Datastream,
    requested: tuple[VdifSourceRange, ...] | None,
) -> tuple[VdifSourceRange, ...]:
    if requested is None:
        result = []
        for source_index, source_file in enumerate(datastream.source_files):
            path = Path(source_file)
            size, mtime_ns, device, inode = _validate_source_identity(
                path, None, None
            )
            result.append(
                VdifSourceRange(
                    path,
                    0,
                    _file_frame_count(path, datastream, source_index),
                    size,
                    mtime_ns,
                    device,
                    inode,
                )
            )
        return tuple(result)

    allowed = {
        Path(item).expanduser().resolve(strict=False)
        for item in datastream.source_files
    }
    result = []
    for source_index, item in enumerate(requested):
        canonical_path = item.path.expanduser().resolve(strict=False)
        if canonical_path not in allowed:
            raise ConfigurationError(
                f"datastream[{datastream.identifier}]",
                "source_files",
                f"bounded range references undeclared path {item.path}",
            )
        if item.first_frame < 0 or item.frame_count < 0:
            raise ConfigurationError(
                f"datastream[{datastream.identifier}]",
                f"source_range[{source_index}]",
                "frame bounds must be nonnegative",
            )
        size, mtime_ns, device, inode = _validate_source_identity(
            canonical_path,
            item.expected_size_bytes,
            item.expected_mtime_ns,
            item.expected_device,
            item.expected_inode,
        )
        total = _file_frame_count(canonical_path, datastream, source_index)
        if item.stop_frame > total:
            raise ConfigurationError(
                f"datastream[{datastream.identifier}]",
                f"source_range[{source_index}]",
                f"frame range ends at {item.stop_frame}, beyond {total}",
            )
        result.append(
            VdifSourceRange(
                canonical_path,
                item.first_frame,
                item.frame_count,
                size,
                mtime_ns,
                device,
                inode,
            )
        )
    return tuple(result)


def _bulk_vdif_frames(
    datastream: Datastream,
    requested: tuple[VdifSourceRange, ...] | None = None,
) -> _BulkVdifFrames:
    frame_size = datastream.frame_size_bytes
    payload_size = frame_size - HEADER_BYTES
    ranges = _source_ranges(datastream, requested)
    paths = [item.path for item in ranges]
    starts = [item.first_frame for item in ranges]
    counts = [item.frame_count for item in ranges]
    sizes = [cast(int, item.expected_size_bytes) for item in ranges]
    mtimes_ns = [cast(int, item.expected_mtime_ns) for item in ranges]
    devices = [cast(int, item.expected_device) for item in ranges]
    inodes = [cast(int, item.expected_inode) for item in ranges]
    total = sum(counts)
    words = np.empty((total, 8), dtype="<u4")
    source_indices = np.empty(total, dtype=np.int32)
    offsets = np.empty(total, dtype=np.int64)

    cursor = 0
    for source_index, (path, count) in enumerate(
        zip(paths, counts, strict=True)
    ):
        if count == 0:
            continue
        stop = cursor + count
        with path.open("rb", buffering=0) as stream:
            _validate_open_source_identity(
                stream.fileno(),
                path,
                sizes[source_index],
                mtimes_ns[source_index],
                devices[source_index],
                inodes[source_index],
            )
            # Read the selected extent sequentially through one reusable
            # buffer.  A whole-file mmap faults once per source page even
            # though only the 32-byte header from each frame is retained.
            chunk_frames = max(1, _VDIF_HEADER_SCAN_CHUNK_BYTES // frame_size)
            read_buffer = bytearray(min(count, chunk_frames) * frame_size)
            for chunk_first in range(0, count, chunk_frames):
                chunk_count = min(chunk_frames, count - chunk_first)
                chunk_bytes = chunk_count * frame_size
                file_offset = (
                    starts[source_index] + chunk_first
                ) * frame_size
                raw = memoryview(read_buffer)[:chunk_bytes]
                received = 0
                with nvtx_range("xbeam:vdif-header-sequential-read"):
                    while received < chunk_bytes:
                        read_count = os.preadv(
                            stream.fileno(),
                            [raw[received:]],
                            file_offset + received,
                        )
                        if read_count == 0:
                            raise TruncatedVdifFrameError(
                                f"truncated VDIF extent at byte "
                                f"{file_offset}: expected {chunk_bytes} "
                                f"bytes, got {received}"
                            )
                        received += read_count
                rows = np.frombuffer(raw, dtype=np.uint8).reshape(
                    chunk_count, frame_size
                )
                with nvtx_range("xbeam:vdif-header-strided-copy"):
                    headers = np.ascontiguousarray(rows[:, :HEADER_BYTES])
                chunk_cursor = cursor + chunk_first
                words[chunk_cursor : chunk_cursor + chunk_count] = (
                    headers.view("<u4").reshape(chunk_count, 8)
                )
                del headers, rows, raw
            _validate_open_source_identity(
                stream.fileno(),
                path,
                sizes[source_index],
                mtimes_ns[source_index],
                devices[source_index],
                inodes[source_index],
            )
        source_indices[cursor:stop] = source_index
        offsets[cursor:stop] = (
            np.arange(
                starts[source_index],
                starts[source_index] + count,
                dtype=np.int64,
            )
            * frame_size
        )
        cursor = stop

    if total == 0:
        return _BulkVdifFrames(
            words,
            source_indices,
            offsets,
            tuple(paths),
            tuple(starts),
            tuple(counts),
            tuple(int(item) for item in sizes),
            tuple(int(item) for item in mtimes_ns),
            tuple(int(item) for item in devices),
            tuple(int(item) for item in inodes),
        )

    word0, word1, word2, word3 = (words[:, index] for index in range(4))
    legacy = ((word0 >> np.uint32(30)) & np.uint32(1)).astype(bool)
    reserved = (word1 >> np.uint32(30)) != 0
    versions = (word2 >> np.uint32(29)) & np.uint32(0x7)
    frame_lengths = (word2 & np.uint32(0x00FF_FFFF)).astype(np.int64) * 8
    bits = ((word3 >> np.uint32(26)) & np.uint32(0x1F)).astype(np.int64) + 1
    log2_channels = ((word2 >> np.uint32(24)) & np.uint32(0x1F)).astype(
        np.int64
    )
    complex_data = (word3 >> np.uint32(31)).astype(bool)
    safe_log2 = np.minimum(log2_channels, 30)
    channels = np.left_shift(np.int64(1), safe_log2)
    complete_bits = channels * bits * np.where(complex_data, 2, 1)
    sample_invalid = (log2_channels > 30) | (
        (payload_size * 8) % complete_bits != 0
    )
    parser_invalid = (
        legacy
        | reserved
        | (versions != 0)
        | (frame_lengths < HEADER_BYTES)
        | (bits != 2)
        | sample_invalid
    )
    if parser_invalid.any():
        first = int(np.flatnonzero(parser_invalid)[0])
        VdifHeader.from_bytes(words[first].tobytes())
        raise AssertionError(
            "vector VDIF validation disagrees with scalar parser"
        )

    length_mismatch = frame_lengths != frame_size
    if length_mismatch.any():
        first = int(np.flatnonzero(length_mismatch)[0])
        source_index = int(source_indices[first])
        raise _frame_size_error(
            datastream, source_index, int(frame_lengths[first])
        )
    expected_complex = datastream.sampling is not SamplingKind.REAL
    complex_mismatch = complex_data != expected_complex
    if complex_mismatch.any():
        first = int(np.flatnonzero(complex_mismatch)[0])
        source_index = int(source_indices[first])
        expected = "complex" if expected_complex else "real"
        observed = "complex" if complex_data[first] else "real"
        raise ConfigurationError(
            f"datastream[{datastream.identifier}]",
            "sampling",
            f"declares {expected} samples but source_files[{source_index}] "
            f"contains {observed} VDIF samples",
        )
    return _BulkVdifFrames(
        words,
        source_indices,
        offsets,
        tuple(paths),
        tuple(starts),
        tuple(counts),
        tuple(int(item) for item in sizes),
        tuple(int(item) for item in mtimes_ns),
        tuple(int(item) for item in devices),
        tuple(int(item) for item in inodes),
    )


def _bulk_frame_cache_key(
    datastream: Datastream,
    requested: tuple[VdifSourceRange, ...] | None,
) -> tuple[Any, ...]:
    if requested is None:
        sources: tuple[Any, ...] = tuple(
            Path(item).expanduser().resolve(strict=False)
            for item in datastream.source_files
        )
    else:
        sources = tuple(
            (
                item.path.expanduser().resolve(strict=False),
                item.first_frame,
                item.frame_count,
                item.expected_size_bytes,
                item.expected_mtime_ns,
                item.expected_device,
                item.expected_inode,
            )
            for item in requested
        )
    return (
        datastream.frame_size_bytes,
        datastream.sampling,
        tuple(datastream.source_files),
        sources,
    )


def _validate_cached_frames(frames: _BulkVdifFrames) -> None:
    for values in zip(
        frames.paths,
        frames.file_sizes,
        frames.file_mtimes_ns,
        frames.file_devices,
        frames.file_inodes,
        strict=True,
    ):
        _validate_source_identity(*values)


def _validate_cached_sidecar_frames(frames: _BinarySidecarFrames) -> None:
    for values in zip(
        frames.paths,
        frames.file_sizes,
        frames.file_mtimes_ns,
        frames.file_devices,
        frames.file_inodes,
        strict=True,
    ):
        _validate_source_identity(*values)


def _header(frames: _BulkVdifFrames, index: int) -> VdifHeader:
    return VdifHeader.from_bytes(frames.words[index].tobytes())


def _absolute_slots(
    frames: _BulkVdifFrames, indices: np.ndarray, frames_per_second: int
) -> np.ndarray:
    words = frames.words[indices]
    word0 = words[:, 0]
    word1 = words[:, 1]
    seconds = (word0 & np.uint32(0x3FFF_FFFF)).astype(np.int64)
    epochs = ((word1 >> np.uint32(24)) & np.uint32(0x3F)).astype(np.int64)
    frame_numbers = (word1 & np.uint32(0x00FF_FFFF)).astype(np.int64)
    outside = frame_numbers >= frames_per_second
    if outside.any():
        first = int(np.flatnonzero(outside)[0])
        raise VdifError(
            f"frame number {int(frame_numbers[first])} is outside "
            f"frames_per_second={frames_per_second}"
        )
    epoch_origins = np.empty(indices.size, dtype=np.int64)
    for epoch in np.unique(epochs):
        position = int(np.flatnonzero(epochs == epoch)[0])
        template = _header(frames, int(indices[position]))
        origin = (
            seconds_since_vdif_origin(template) - template.seconds_from_epoch
        )
        epoch_origins[epochs == epoch] = origin
    return (epoch_origins + seconds) * frames_per_second + frame_numbers


def _selected_frame_indices(
    frames: _BulkVdifFrames,
    slots: np.ndarray,
    indices: np.ndarray,
    invalid: np.ndarray,
    policy: DuplicatePolicy,
) -> tuple[np.ndarray, np.ndarray]:
    order_key = np.arange(indices.size, dtype=np.int64)
    if policy is DuplicatePolicy.FIRST_VALID:
        order = np.lexsort((order_key, invalid.astype(np.int8), slots))
        take_last = False
    elif policy is DuplicatePolicy.LAST_VALID:
        order = np.lexsort((order_key, (~invalid).astype(np.int8), slots))
        take_last = True
    else:
        order = np.lexsort((order_key, slots))
        take_last = policy is DuplicatePolicy.LAST
    ordered_slots = slots[order]
    starts = np.r_[True, ordered_slots[1:] != ordered_slots[:-1]]
    group_starts = np.flatnonzero(starts)
    group_ends = np.r_[group_starts[1:], order.size]
    if policy is DuplicatePolicy.REJECT and np.any(
        group_ends - group_starts > 1
    ):
        duplicate_group = int(
            np.flatnonzero(group_ends - group_starts > 1)[0]
        )
        duplicate_position = group_starts[duplicate_group] + 1
        duplicate_index = int(indices[order[duplicate_position]])
        header = _header(frames, duplicate_index)
        raise DuplicateVdifFrameError(
            f"duplicate VDIF frame for thread {header.thread_id}, "
            f"epoch {header.reference_epoch}, second "
            f"{header.seconds_from_epoch}, frame {header.frame_number}"
        )
    chosen_positions = group_ends - 1 if take_last else group_starts
    return indices[order[chosen_positions]], ordered_slots[chosen_positions]


def _sidecar_slots(
    timestamps: np.ndarray,
    template: VdifHeader,
    frames_per_second: int,
) -> np.ndarray:
    """Map packed sidecar timestamps directly onto the planned frame grid."""

    epochs = timestamps >> np.uint64(54)
    seconds = ((timestamps >> np.uint64(24)) & np.uint64(0x3FFF_FFFF)).astype(
        np.int64
    )
    frame_numbers = (timestamps & np.uint64(0x00FF_FFFF)).astype(np.int64)
    outside = frame_numbers >= frames_per_second
    if outside.any():
        first = int(np.flatnonzero(outside)[0])
        raise VdifError(
            f"frame number {int(frame_numbers[first])} is outside "
            f"frames_per_second={frames_per_second}"
        )
    slots = np.empty(timestamps.size, dtype=np.int64)
    for epoch_value in np.unique(epochs):
        epoch_template = replace(
            template,
            reference_epoch=int(epoch_value),
            seconds_from_epoch=0,
            frame_number=0,
        )
        epoch_slot = absolute_frame_slot(epoch_template, frames_per_second)
        selected = epochs == epoch_value
        slots[selected] = (
            epoch_slot
            + seconds[selected] * frames_per_second
            + frame_numbers[selected]
        )
    return slots


def _presented_sidecar_slots(
    slots: np.ndarray,
    intervals: tuple[tuple[int, int, int], ...],
) -> np.ndarray:
    """Map physical VDIF timestamp slots onto calibrated reader slots."""

    result = np.full(slots.shape, -1, dtype=np.int64)
    if not intervals or not slots.size:
        return result
    starts = np.fromiter(
        (item[0] for item in intervals), dtype=np.int64, count=len(intervals)
    )
    stops = np.fromiter(
        (item[1] for item in intervals), dtype=np.int64, count=len(intervals)
    )
    destinations = np.fromiter(
        (item[2] for item in intervals), dtype=np.int64, count=len(intervals)
    )
    positions = np.searchsorted(starts, slots, side="right") - 1
    candidates = positions >= 0
    selected = np.flatnonzero(candidates)
    if selected.size:
        interval_positions = positions[selected]
        selected = selected[slots[selected] < stops[interval_positions]]
        interval_positions = positions[selected]
        result[selected] = (
            destinations[interval_positions]
            + slots[selected]
            - starts[interval_positions]
        )
    return result


def _presented_physical_sidecar_slots(
    source_indices: np.ndarray,
    physical_frames: np.ndarray,
    intervals: tuple[tuple[int, int, int, int], ...],
) -> np.ndarray:
    """Map physical file-order frames onto calibrated reader slots."""

    result = np.full(physical_frames.shape, -1, dtype=np.int64)
    rows, slots = _presented_physical_sidecar_mapping(
        source_indices, physical_frames, intervals
    )
    if rows.size:
        # Presentation intervals may overlap.  Preserve the original helper's
        # last-interval-wins behavior without relying on repeated advanced
        # assignment indices, whose write order is not part of NumPy's API.
        reverse_rows = rows[::-1]
        selected_rows, reverse_positions = np.unique(
            reverse_rows, return_index=True
        )
        result[selected_rows] = slots[::-1][reverse_positions]
    return result


def _presented_physical_sidecar_mapping(
    source_indices: np.ndarray,
    physical_frames: np.ndarray,
    intervals: tuple[tuple[int, int, int, int], ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Return sidecar rows and destinations for physical presentation.

    Destination-ordered intervals make the returned slots ordered while the
    corresponding physical rows may be out of order or repeated.
    """

    if source_indices.shape != physical_frames.shape:
        raise ValueError(
            "physical frame coordinates must have matching shapes"
        )
    if not intervals or physical_frames.size == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )

    # Sort the sidecar rows once.  The former implementation evaluated an
    # N-row boolean mask for every presentation interval, which becomes
    # quadratic for a long recovery plan.  Grouping by source and physical
    # frame lets every interval use two binary searches while retaining stable
    # physical order.  Overlapping intervals still intentionally repeat rows.
    row_numbers = np.arange(physical_frames.size, dtype=np.int64)
    order = np.lexsort((row_numbers, physical_frames, source_indices))
    ordered_sources = source_indices[order]
    ordered_frames = physical_frames[order]
    sources, source_starts, source_counts = np.unique(
        ordered_sources, return_index=True, return_counts=True
    )
    source_bounds = {
        int(source): (int(start), int(start + count))
        for source, start, count in zip(
            sources, source_starts, source_counts, strict=True
        )
    }

    row_parts: list[np.ndarray] = []
    slot_parts: list[np.ndarray] = []
    for source_index, start, stop, destination in intervals:
        bounds = source_bounds.get(source_index)
        if bounds is None or stop <= start:
            continue
        source_start, source_stop = bounds
        source_frames = ordered_frames[source_start:source_stop]
        lower = source_start + int(
            np.searchsorted(source_frames, start, side="left")
        )
        upper = source_start + int(
            np.searchsorted(source_frames, stop, side="left")
        )
        rows = order[lower:upper]
        if rows.size:
            row_parts.append(rows)
            slot_parts.append(destination + physical_frames[rows] - start)
    if not row_parts:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )
    return np.concatenate(row_parts), np.concatenate(slot_parts)


def _apply_physical_sidecar_overrides(
    slots: np.ndarray,
    source_indices: np.ndarray,
    physical_frames: np.ndarray,
    intervals: tuple[tuple[int, int, int, int], ...],
) -> np.ndarray:
    """Layer sparse physical moves and drops onto timestamp-derived slots."""

    result = slots.copy()
    moves = tuple(item for item in intervals if item[3] >= 0)
    if moves:
        rows, destinations = _presented_physical_sidecar_mapping(
            source_indices, physical_frames, moves
        )
        if rows.size:
            # Match presentation mapping: the last overlapping move wins.
            selected_rows, reverse_positions = np.unique(
                rows[::-1], return_index=True
            )
            result[selected_rows] = destinations[::-1][reverse_positions]
    drops = tuple((*item[:3], 0) for item in intervals if item[3] == -1)
    if drops:
        rows, _destinations = _presented_physical_sidecar_mapping(
            source_indices, physical_frames, drops
        )
        result[rows] = -1
    return result


def _selected_sidecar_indices(
    timestamps: np.ndarray,
    slots: np.ndarray,
    indices: np.ndarray,
    invalid: np.ndarray,
    policy: DuplicatePolicy,
    thread_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve duplicate sidecar entries using physical source order."""

    order_key = np.arange(indices.size, dtype=np.int64)
    if policy is DuplicatePolicy.FIRST_VALID:
        order = np.lexsort((order_key, invalid.astype(np.int8), slots))
        take_last = False
    elif policy is DuplicatePolicy.LAST_VALID:
        order = np.lexsort((order_key, (~invalid).astype(np.int8), slots))
        take_last = True
    else:
        order = np.lexsort((order_key, slots))
        take_last = policy is DuplicatePolicy.LAST
    ordered_slots = slots[order]
    starts = np.r_[True, ordered_slots[1:] != ordered_slots[:-1]]
    group_starts = np.flatnonzero(starts)
    group_ends = np.r_[group_starts[1:], order.size]
    duplicate_groups = group_ends - group_starts > 1
    if policy is DuplicatePolicy.REJECT and np.any(duplicate_groups):
        duplicate_group = int(np.flatnonzero(duplicate_groups)[0])
        duplicate_position = group_starts[duplicate_group] + 1
        duplicate_index = int(indices[order[duplicate_position]])
        key = VdifTimestampKey.from_packed(int(timestamps[duplicate_index]))
        raise DuplicateVdifFrameError(
            f"duplicate VDIF frame for thread {thread_id}, epoch "
            f"{key.reference_epoch}, second {key.seconds_from_epoch}, "
            f"frame {key.frame_number}"
        )
    chosen_positions = group_ends - 1 if take_last else group_starts
    return indices[order[chosen_positions]], ordered_slots[chosen_positions]


class VdifFileRecordingSource(RecordingSource):
    """Read file-backed VDIF datastreams into neutral voltage batches.

    Timeline normalization and duplicate selection live here rather than in
    the signal engine.  Payload bytes stay packed so CuPy can decode them in
    the fused front-end kernel.
    """

    def __init__(
        self,
        job: Job,
        profile: CompatibilityProfile,
        *,
        source_ranges: dict[int, tuple[VdifSourceRange, ...]] | None = None,
        thread_windows: dict[int, dict[int, VdifThreadWindow]] | None = None,
        source_thread_windows: dict[int, dict[int, VdifThreadWindow]]
        | None = None,
        sample_ranges: dict[tuple[int, int], tuple[int, int]] | None = None,
        max_materialized_bytes: int | None = None,
        defer_ordered_headers: bool = False,
        binary_sidecars: bool = False,
        binary_index_dir: str | Path | None = None,
        frame_presentation_plans: Mapping[
            int, tuple[tuple[int, int, int], ...]
        ]
        | None = None,
        physical_frame_presentation_plans: Mapping[
            int, tuple[tuple[int, int, int, int], ...]
        ]
        | None = None,
        physical_frame_override_plans: Mapping[
            int, tuple[tuple[int, int, int, int], ...]
        ]
        | None = None,
    ) -> None:
        self._frequencies = {
            item.identifier: item for item in job.frequencies
        }
        self._profile = profile
        try:
            self._duplicate_policy = DuplicatePolicy(
                profile.duplicate_policy.replace("_", "-")
            )
        except (AttributeError, ValueError) as exc:
            raise VdifError(
                f"unknown VDIF duplicate policy {profile.duplicate_policy!r}"
            ) from exc
        self._job_start = (
            Decimal(job.start_time.mjd) * Decimal(86400)
            + job.start_time.seconds_of_day
        )
        self._execution_duration_s = job.execution_duration_s
        self._source_ranges = source_ranges or {}
        self._thread_windows = thread_windows or {}
        self._source_thread_windows = (
            source_thread_windows or self._thread_windows
        )
        self._sample_ranges = sample_ranges
        self._max_materialized_bytes = max_materialized_bytes
        self._defer_ordered_headers = defer_ordered_headers
        self._binary_sidecars = binary_sidecars
        self._binary_index_dir = (
            None if binary_index_dir is None else Path(binary_index_dir)
        )
        self._frame_presentation_plans = dict(frame_presentation_plans or {})
        self._physical_frame_presentation_plans = dict(
            physical_frame_presentation_plans or {}
        )
        self._physical_frame_override_plans = dict(
            physical_frame_override_plans or {}
        )
        self._frame_cache: dict[
            tuple[Any, ...], _BulkVdifFrames | _BinarySidecarFrames
        ] = {}
        self._station_codes = (
            {
                station.identifier: (code, station.name)
                for station in job.stations
                if (code := _text_station_id(station.name)) is not None
            }
            if profile.validate_vdif_station_code
            else {}
        )

    def batch_cache_key(self, datastream: Datastream) -> tuple[Any, ...]:
        """Return the immutable header-cache key for one datastream."""

        requested = self._source_ranges.get(datastream.identifier)
        return _bulk_frame_cache_key(datastream, requested)

    def _streaming_batches(
        self, datastream: Datastream
    ) -> Iterable[VoltageBatch]:
        """Build bounded output geometry without a separate header pass."""

        requested = self._source_ranges.get(datastream.identifier)
        thread_windows = self._thread_windows.get(datastream.identifier, {})
        if requested is None or not thread_windows:
            raise RuntimeError(
                "streaming VDIF ingest requires bounded thread windows"
            )
        ranges = _source_ranges(datastream, requested)
        bands_by_thread = defaultdict(list)
        for band in datastream.recorded_bands:
            bands_by_thread[band.thread_id].append(band)

        policy = self._duplicate_policy
        datastream_station_id: int | None = None
        materialized_bytes = 0
        for thread_id, bands in bands_by_thread.items():
            planned_window = thread_windows.get(thread_id)
            if planned_window is None:
                raise RuntimeError(
                    "streaming VDIF ingest requires every recorded thread "
                    "to have a bounded window"
                )
            template = planned_window.template
            physical_thread_id = template.thread_id
            rates = [
                _sample_rate(datastream, self._frequencies[band.frequency_id])
                for band in bands
            ]
            sample_rate = rates[0]
            if any(
                not math.isclose(
                    rate, sample_rate, rel_tol=1e-12, abs_tol=1e-6
                )
                for rate in rates[1:]
            ):
                raise ConfigurationError(
                    f"datastream[{datastream.identifier}]",
                    "recorded_bands",
                    f"bands sharing VDIF thread {thread_id} imply "
                    "different sample rates",
                )
            frames_per_second_float = sample_rate / template.samples_per_frame
            frames_per_second = round(frames_per_second_float)
            if not math.isclose(
                frames_per_second_float,
                frames_per_second,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ConfigurationError(
                    f"datastream[{datastream.identifier}]",
                    "sample_rate_hz",
                    "does not imply an integral number of VDIF frames per "
                    "second",
                )
            if planned_window.frames_per_second != frames_per_second:
                raise ConfigurationError(
                    f"datastream[{datastream.identifier}]",
                    "sample_rate_hz",
                    f"VDIF thread {thread_id} frame rate changed after "
                    "bounded ingest planning",
                )
            if datastream_station_id is None:
                datastream_station_id = template.station_id
            elif template.station_id != datastream_station_id:
                raise VdifError(
                    "referenced VDIF threads contain different station "
                    "identifiers"
                )
            expected_station = self._station_codes.get(datastream.station_id)
            if (
                expected_station is not None
                and template.station_id != expected_station[0]
            ):
                observed = (
                    template.station_code or f"0x{template.station_id:04x}"
                )
                raise ConfigurationError(
                    f"datastream[{datastream.identifier}]",
                    "source_files",
                    f"VDIF station {observed!r} does not match job station "
                    f"{expected_station[1]!r}",
                )

            start_slot = planned_window.start_slot
            slot_count = planned_window.stop_slot - start_slot
            if not 0 < slot_count <= np.iinfo(np.int32).max:
                raise ConfigurationError(
                    f"datastream[{datastream.identifier}]",
                    "source_files",
                    f"VDIF thread {thread_id} has an invalid planned window",
                )
            materialized_bytes += slot_count * template.payload_bytes
            if (
                self._max_materialized_bytes is not None
                and materialized_bytes > self._max_materialized_bytes
            ):
                raise ConfigurationError(
                    "correlate",
                    "max_read_bytes",
                    "planned gap-filled VDIF payload exceeds the bounded "
                    "materialization limit",
                )
            channel_count = template.channel_count
            if len(bands) > channel_count:
                raise ConfigurationError(
                    f"datastream[{datastream.identifier}]",
                    "recorded_bands",
                    f"thread {thread_id} contains {channel_count} VDIF "
                    f"channels but maps {len(bands)} logical bands",
                )
            validity = np.zeros(slot_count, dtype=np.bool_)
            file_plans = tuple(
                VdifPayloadFilePlan(
                    item.path,
                    item.frame_count,
                    None,
                    item.first_frame,
                    item.expected_size_bytes,
                    item.expected_mtime_ns,
                    item.expected_device,
                    item.expected_inode,
                )
                for item in ranges
            )
            streaming = VdifStreamingPlan(
                physical_thread_id,
                frames_per_second,
                start_slot,
                template,
                policy,
                validity,
            )
            payload = VdifPayloadPlan(
                datastream.frame_size_bytes,
                template.payload_bytes,
                slot_count,
                file_plans,
                streaming,
            )
            first_time = vdif_seconds_to_mjd_seconds(
                Decimal(start_slot) / Decimal(frames_per_second)
            )
            start_time = _mjd_time(first_time)
            for channel_index, band in enumerate(bands):
                yield VoltageBatch(
                    samples=payload,
                    start_time=start_time,
                    sample_rate_hz=sample_rate,
                    band_id=band.identifier,
                    polarization=band.polarization,
                    validity=validity,
                    packed_layout=PackedVoltageLayout(
                        frame_payload_bytes=template.payload_bytes,
                        samples_per_frame=template.samples_per_frame,
                        channel_count=channel_count,
                        channel_index=channel_index,
                        complex_data=template.complex_data,
                    ),
                )

    def _binary_sidecar_batches(
        self,
        datastream: Datastream,
        frames: _BinarySidecarFrames,
        thread_windows: dict[int, VdifThreadWindow],
    ) -> Iterable[VoltageBatch]:
        """Build payload plans directly from compact sidecar entries."""

        if not thread_windows:
            raise RuntimeError(
                "binary sidecar ingest requires bounded thread windows"
            )
        policy = self._duplicate_policy
        bands_by_thread = defaultdict(list)
        for band in datastream.recorded_bands:
            bands_by_thread[band.thread_id].append(band)
        metadata_by_thread = {
            item.thread_id: item for item in frames.thread_metadata
        }
        datastream_station_id: int | None = None
        materialized_bytes = 0
        result: list[VoltageBatch] = []

        with nvtx_range("xbeam:vdif-sidecar-plan"):
            for thread_id, bands in bands_by_thread.items():
                planned_window = thread_windows.get(thread_id)
                if planned_window is None:
                    raise RuntimeError(
                        "binary sidecar ingest requires every recorded "
                        "thread to have a bounded window"
                    )
                template = planned_window.template
                physical_thread_id = template.thread_id
                rates = [
                    _sample_rate(
                        datastream, self._frequencies[band.frequency_id]
                    )
                    for band in bands
                ]
                sample_rate = rates[0]
                if any(
                    not math.isclose(
                        rate, sample_rate, rel_tol=1e-12, abs_tol=1e-6
                    )
                    for rate in rates[1:]
                ):
                    raise ConfigurationError(
                        f"datastream[{datastream.identifier}]",
                        "recorded_bands",
                        f"bands sharing VDIF thread {thread_id} imply "
                        "different sample rates",
                    )
                frames_per_second_float = (
                    sample_rate / template.samples_per_frame
                )
                frames_per_second = round(frames_per_second_float)
                if not math.isclose(
                    frames_per_second_float,
                    frames_per_second,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ConfigurationError(
                        f"datastream[{datastream.identifier}]",
                        "sample_rate_hz",
                        "does not imply an integral number of VDIF frames "
                        "per second",
                    )
                if planned_window.frames_per_second != frames_per_second:
                    raise ConfigurationError(
                        f"datastream[{datastream.identifier}]",
                        "sample_rate_hz",
                        f"VDIF thread {thread_id} frame rate changed after "
                        "bounded ingest planning",
                    )

                thread_indices = np.flatnonzero(
                    frames.thread_ids == physical_thread_id
                )
                presentation = self._frame_presentation_plans.get(
                    datastream.identifier
                )
                physical_presentation = (
                    self._physical_frame_presentation_plans.get(
                        datastream.identifier
                    )
                )
                physical_overrides = self._physical_frame_override_plans.get(
                    datastream.identifier
                )
                if (
                    presentation is not None
                    and physical_presentation is not None
                ):
                    raise RuntimeError(
                        "one datastream cannot use timestamp and physical "
                        "frame presentations together"
                    )
                if thread_indices.size:
                    metadata = metadata_by_thread.get(physical_thread_id)
                    if metadata is None:
                        raise VdifError(
                            f"binary sidecar omits metadata for VDIF thread "
                            f"{physical_thread_id}"
                        )
                    observed_format = (
                        metadata.frame_size_bytes,
                        metadata.log2_channels,
                        metadata.complex_data,
                        metadata.bits_per_sample,
                        metadata.station_id,
                    )
                    planned_format = (
                        template.frame_length_bytes,
                        template.log2_channels,
                        template.complex_data,
                        template.bits_per_sample,
                        template.station_id,
                    )
                    if observed_format != planned_format:
                        raise VdifError(
                            f"VDIF thread {thread_id} format changed after "
                            "bounded ingest planning"
                        )
                if datastream_station_id is None:
                    datastream_station_id = template.station_id
                elif template.station_id != datastream_station_id:
                    raise VdifError(
                        "referenced VDIF threads contain different station "
                        "identifiers"
                    )
                expected_station = self._station_codes.get(
                    datastream.station_id
                )
                if (
                    expected_station is not None
                    and template.station_id != expected_station[0]
                ):
                    observed = (
                        template.station_code
                        or f"0x{template.station_id:04x}"
                    )
                    raise ConfigurationError(
                        f"datastream[{datastream.identifier}]",
                        "source_files",
                        f"VDIF station {observed!r} does not match job "
                        f"station {expected_station[1]!r}",
                    )

                if thread_indices.size:
                    slots = _sidecar_slots(
                        frames.timestamps[thread_indices],
                        template,
                        frames_per_second,
                    )
                    if physical_presentation is not None:
                        source_paths = tuple(
                            Path(item).expanduser().resolve()
                            for item in datastream.source_files
                        )
                        presentation_sources = np.fromiter(
                            (
                                source_paths.index(path)
                                for path in frames.paths
                            ),
                            dtype=np.int32,
                            count=len(frames.paths),
                        )
                        presented_rows, slots = (
                            _presented_physical_sidecar_mapping(
                                presentation_sources[
                                    frames.source_indices[thread_indices]
                                ],
                                frames.physical_frames[thread_indices],
                                physical_presentation,
                            )
                        )
                        thread_indices = thread_indices[presented_rows]
                    elif presentation is not None:
                        slots = _presented_sidecar_slots(slots, presentation)
                    if physical_overrides is not None:
                        source_paths = tuple(
                            Path(item).expanduser().resolve()
                            for item in datastream.source_files
                        )
                        presentation_sources = np.fromiter(
                            (
                                source_paths.index(path)
                                for path in frames.paths
                            ),
                            dtype=np.int32,
                            count=len(frames.paths),
                        )
                        slots = _apply_physical_sidecar_overrides(
                            slots,
                            presentation_sources[
                                frames.source_indices[thread_indices]
                            ],
                            frames.physical_frames[thread_indices],
                            physical_overrides,
                        )
                    inside = (slots >= planned_window.start_slot) & (
                        slots < planned_window.stop_slot
                    )
                    thread_indices = thread_indices[inside]
                    slots = slots[inside]
                else:
                    slots = np.empty(0, dtype=np.int64)
                if thread_indices.size:
                    selected, selected_slots = _selected_sidecar_indices(
                        frames.timestamps,
                        slots,
                        thread_indices,
                        frames.invalid[thread_indices],
                        policy,
                        physical_thread_id,
                    )
                else:
                    selected = np.empty(0, dtype=np.int64)
                    selected_slots = np.empty(0, dtype=np.int64)

                start_slot = planned_window.start_slot
                slot_count = planned_window.stop_slot - start_slot
                if not 0 < slot_count <= np.iinfo(np.int32).max:
                    raise ConfigurationError(
                        f"datastream[{datastream.identifier}]",
                        "source_files",
                        f"VDIF thread {thread_id} has an invalid planned "
                        "window",
                    )
                channel_count = template.channel_count
                if len(bands) > channel_count:
                    raise ConfigurationError(
                        f"datastream[{datastream.identifier}]",
                        "recorded_bands",
                        f"thread {thread_id} contains {channel_count} VDIF "
                        f"channels but maps {len(bands)} logical bands",
                    )
                positions = selected_slots - start_slot
                validity = np.zeros(slot_count, dtype=np.bool_)
                validity[positions] = ~frames.invalid[selected]
                materialized_bytes += slot_count * template.payload_bytes
                if (
                    self._max_materialized_bytes is not None
                    and materialized_bytes > self._max_materialized_bytes
                ):
                    raise ConfigurationError(
                        "correlate",
                        "max_read_bytes",
                        "planned gap-filled VDIF payload exceeds the "
                        "bounded materialization limit",
                    )

                file_plans = []
                for source_index, values in enumerate(
                    zip(
                        frames.paths,
                        frames.file_frame_starts,
                        frames.file_frame_counts,
                        frames.file_sizes,
                        frames.file_mtimes_ns,
                        frames.file_devices,
                        frames.file_inodes,
                        strict=True,
                    )
                ):
                    (
                        path,
                        first_frame,
                        frame_count,
                        expected_size_bytes,
                        expected_mtime_ns,
                        expected_device,
                        expected_inode,
                    ) = values
                    in_file = frames.source_indices[selected] == source_index
                    selected_in_file = selected[in_file]
                    local_indices = (
                        (
                            frames.physical_frames[selected_in_file]
                            - first_frame
                        )
                        if selected_in_file.size
                        else np.empty(0, dtype=np.int64)
                    )
                    if physical_presentation is None:
                        destination_slots = np.full(
                            frame_count, -1, dtype=np.int32
                        )
                        if selected_in_file.size:
                            destination_slots[local_indices] = positions[
                                in_file
                            ]
                        source_frame_indices = None
                    else:
                        (
                            destination_slots,
                            source_frame_indices,
                        ) = _canonical_payload_file_mapping(
                            frame_count,
                            local_indices,
                            positions[in_file],
                        )
                    file_plans.append(
                        VdifPayloadFilePlan(
                            path,
                            frame_count,
                            destination_slots,
                            first_frame,
                            expected_size_bytes,
                            expected_mtime_ns,
                            expected_device,
                            expected_inode,
                            source_frame_indices,
                        )
                    )
                payload = VdifPayloadPlan(
                    datastream.frame_size_bytes,
                    template.payload_bytes,
                    slot_count,
                    tuple(file_plans),
                )
                first_time = vdif_seconds_to_mjd_seconds(
                    Decimal(start_slot) / Decimal(frames_per_second)
                )
                start_time = _mjd_time(first_time)
                for channel_index, band in enumerate(bands):
                    result.append(
                        VoltageBatch(
                            samples=payload,
                            start_time=start_time,
                            sample_rate_hz=sample_rate,
                            band_id=band.identifier,
                            polarization=band.polarization,
                            validity=validity,
                            packed_layout=PackedVoltageLayout(
                                frame_payload_bytes=template.payload_bytes,
                                samples_per_frame=template.samples_per_frame,
                                channel_count=channel_count,
                                channel_index=channel_index,
                                complex_data=template.complex_data,
                            ),
                        )
                    )
        yield from result

    def batches(self, datastream: Datastream) -> Iterable[VoltageBatch]:
        try:
            yield from self._batches(datastream)
        except (ConfigurationError, VdifError) as error:
            aliases = [
                f"logical thread {logical} maps to physical thread "
                f"{window.template.thread_id}"
                for logical, window in self._thread_windows.get(
                    datastream.identifier, {}
                ).items()
                if logical != window.template.thread_id
            ]
            if not aliases:
                raise
            context = "; ".join(aliases)
            if isinstance(error, ConfigurationError):
                raise type(error)(
                    error.entity,
                    error.field,
                    f"{error.reason} ({context})",
                ) from error
            raise type(error)(f"{error} ({context})") from error

    def _batches(self, datastream: Datastream) -> Iterable[VoltageBatch]:
        if self._defer_ordered_headers:
            yield from self._streaming_batches(datastream)
            return
        requested = self._source_ranges.get(datastream.identifier)
        cache_key = _bulk_frame_cache_key(datastream, requested)
        thread_windows = self._thread_windows.get(datastream.identifier, {})
        if self._binary_sidecars:
            if requested is None:
                raise RuntimeError(
                    "binary sidecar ingest requires bounded source ranges"
                )
            cached = self._frame_cache.get(cache_key)
            if cached is None:
                with nvtx_range("xbeam:vdif-sidecar-map"):
                    sidecar_frames = _binary_sidecar_frames(
                        datastream,
                        requested,
                        self._source_thread_windows.get(
                            datastream.identifier, thread_windows
                        ),
                        self._binary_index_dir,
                        self._physical_frame_presentation_plans.get(
                            datastream.identifier
                        ),
                    )
                self._frame_cache[cache_key] = sidecar_frames
            else:
                if not isinstance(cached, _BinarySidecarFrames):
                    raise RuntimeError("incompatible VDIF frame cache entry")
                _validate_cached_sidecar_frames(cached)
                sidecar_frames = cached
            yield from self._binary_sidecar_batches(
                datastream, sidecar_frames, thread_windows
            )
            return
        with nvtx_range("xbeam:vdif-header-scan"):
            frames = self._frame_cache.get(cache_key)
            if frames is None:
                frames = _bulk_vdif_frames(datastream, requested)
                self._frame_cache[cache_key] = frames
            else:
                if not isinstance(frames, _BulkVdifFrames):
                    raise RuntimeError("incompatible VDIF frame cache entry")
                _validate_cached_frames(frames)
        if frames.count == 0 and not thread_windows:
            raise ConfigurationError(
                f"datastream[{datastream.identifier}]",
                "source_files",
                "contained no VDIF frames",
            )
        with nvtx_range("xbeam:vdif-header-classification"):
            word0 = frames.words[:, 0]
            word2 = frames.words[:, 2]
            word3 = frames.words[:, 3]
            thread_ids = ((word3 >> np.uint32(16)) & np.uint32(0x3FF)).astype(
                np.int64
            )
            invalid = (word0 >> np.uint32(31)).astype(bool)

        policy = self._duplicate_policy

        bands_by_thread = defaultdict(list)
        for band in datastream.recorded_bands:
            bands_by_thread[band.thread_id].append(band)

        datastream_station_id: int | None = None
        materialized_bytes = 0
        for thread_id, bands in bands_by_thread.items():
            planned_window = thread_windows.get(thread_id)
            physical_thread_id = (
                thread_id
                if planned_window is None
                else planned_window.template.thread_id
            )
            with nvtx_range("xbeam:vdif-thread-selection"):
                thread_indices = np.flatnonzero(
                    thread_ids == physical_thread_id
                )
            if thread_indices.size == 0 and planned_window is None:
                raise ConfigurationError(
                    f"datastream[{datastream.identifier}]",
                    "recorded_bands",
                    f"VDIF thread {thread_id} is unavailable",
                )

            rates = [
                _sample_rate(datastream, self._frequencies[band.frequency_id])
                for band in bands
            ]
            sample_rate = rates[0]
            if any(
                not math.isclose(
                    rate, sample_rate, rel_tol=1e-12, abs_tol=1e-6
                )
                for rate in rates[1:]
            ):
                raise ConfigurationError(
                    f"datastream[{datastream.identifier}]",
                    "recorded_bands",
                    f"bands sharing VDIF thread {thread_id} imply "
                    "different sample rates",
                )

            if thread_indices.size:
                template = _header(frames, int(thread_indices[0]))
            else:
                assert planned_window is not None
                template = planned_window.template
            if planned_window is not None:
                planned_template = planned_window.template
                planned_format = (
                    planned_template.frame_length_bytes,
                    planned_template.log2_channels,
                    planned_template.complex_data,
                    planned_template.bits_per_sample,
                    planned_template.station_id,
                )
                observed_format = (
                    template.frame_length_bytes,
                    template.log2_channels,
                    template.complex_data,
                    template.bits_per_sample,
                    template.station_id,
                )
                if observed_format != planned_format:
                    raise VdifError(
                        f"VDIF thread {thread_id} format changed after "
                        "bounded ingest planning"
                    )
                template = planned_template
            if datastream_station_id is None:
                datastream_station_id = template.station_id
            elif template.station_id != datastream_station_id:
                raise VdifError(
                    "referenced VDIF threads contain different station "
                    "identifiers"
                )
            if thread_indices.size:
                log2_channels = (
                    word2[thread_indices] >> np.uint32(24)
                ) & np.uint32(0x1F)
                station_ids = word3[thread_indices] & np.uint32(0xFFFF)
                format_mismatch = (
                    log2_channels != template.log2_channels
                ) | (station_ids != template.station_id)
                if format_mismatch.any():
                    first = int(
                        thread_indices[np.flatnonzero(format_mismatch)[0]]
                    )
                    raise VdifError(
                        f"inconsistent VDIF frame format in thread "
                        f"{thread_id} at byte {int(frames.offsets[first])}"
                    )
            expected_station = self._station_codes.get(datastream.station_id)
            if (
                expected_station is not None
                and template.station_id != expected_station[0]
            ):
                observed = (
                    template.station_code or f"0x{template.station_id:04x}"
                )
                raise ConfigurationError(
                    f"datastream[{datastream.identifier}]",
                    "source_files",
                    f"VDIF station {observed!r} does not match job station "
                    f"{expected_station[1]!r}",
                )

            samples_per_frame = template.samples_per_frame
            frames_per_second_float = sample_rate / samples_per_frame
            frames_per_second = round(frames_per_second_float)
            if not math.isclose(
                frames_per_second_float,
                frames_per_second,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ConfigurationError(
                    f"datastream[{datastream.identifier}]",
                    "sample_rate_hz",
                    "does not imply an integral number of VDIF frames "
                    "per second",
                )

            if (
                planned_window is not None
                and planned_window.frames_per_second != frames_per_second
            ):
                raise ConfigurationError(
                    f"datastream[{datastream.identifier}]",
                    "sample_rate_hz",
                    f"VDIF thread {thread_id} frame rate changed after "
                    "bounded ingest planning",
                )
            with nvtx_range("xbeam:vdif-slot-construction"):
                slots = _absolute_slots(
                    frames, thread_indices, frames_per_second
                )
            if planned_window is not None:
                start_slot = planned_window.start_slot
                stop_slot = planned_window.stop_slot
                inside = (slots >= start_slot) & (slots < stop_slot)
                thread_indices = thread_indices[inside]
                slots = slots[inside]
            if thread_indices.size:
                with nvtx_range("xbeam:vdif-duplicate-selection"):
                    selected, selected_slots = _selected_frame_indices(
                        frames,
                        slots,
                        thread_indices,
                        invalid[thread_indices],
                        policy,
                    )
            else:
                selected = np.empty(0, dtype=np.int64)
                selected_slots = np.empty(0, dtype=np.int64)
            if planned_window is None:
                start_slot = int(selected_slots[0])
                stop_slot = int(selected_slots[-1]) + 1
            slot_count = stop_slot - start_slot
            if slot_count <= 0:
                raise ConfigurationError(
                    f"datastream[{datastream.identifier}]",
                    "source_files",
                    f"VDIF thread {thread_id} has an empty planned window",
                )
            if planned_window is None:
                slot_limit = _timeline_slot_limit(
                    self._execution_duration_s,
                    frames_per_second,
                    int(thread_indices.size),
                )
                if slot_count > slot_limit:
                    raise ConfigurationError(
                        f"datastream[{datastream.identifier}]",
                        "source_files",
                        "VDIF timestamp span exceeds the declared job "
                        "interval and observed frame count",
                    )
            if slot_count > np.iinfo(np.int32).max:
                raise ConfigurationError(
                    f"datastream[{datastream.identifier}]",
                    "source_files",
                    "VDIF timeline exceeds the supported GPU frame "
                    "index range",
                )
            channel_count = template.channel_count
            if len(bands) > channel_count:
                raise ConfigurationError(
                    f"datastream[{datastream.identifier}]",
                    "recorded_bands",
                    f"thread {thread_id} contains {channel_count} VDIF "
                    "channels "
                    f"but maps {len(bands)} logical bands",
                )

            positions = selected_slots - start_slot
            first_time = vdif_seconds_to_mjd_seconds(
                Decimal(start_slot) / Decimal(frames_per_second)
            )
            if planned_window is None and self._sample_ranges is not None:
                requested_samples = [
                    self._sample_ranges[
                        (datastream.identifier, band.identifier)
                    ]
                    for band in bands
                    if (datastream.identifier, band.identifier)
                    in self._sample_ranges
                ]
                if requested_samples:
                    requested_start = min(
                        item[0] for item in requested_samples
                    )
                    requested_stop = max(
                        item[1] for item in requested_samples
                    )
                    timeline_start = int(
                        (
                            (first_time - self._job_start)
                            * Decimal(str(sample_rate))
                        ).to_integral_value()
                    )
                    relative_start = requested_start - timeline_start
                    relative_stop = requested_stop - timeline_start
                    frame_start = min(
                        slot_count,
                        max(0, relative_start // samples_per_frame),
                    )
                    frame_stop = min(
                        slot_count,
                        -(-relative_stop // samples_per_frame),
                    )
                    frame_stop = max(frame_start, frame_stop)
                    keep = (positions >= frame_start) & (
                        positions < frame_stop
                    )
                    selected = selected[keep]
                    positions = positions[keep] - frame_start
                    start_slot += frame_start
                    slot_count = frame_stop - frame_start
                    first_time += Decimal(frame_start) / Decimal(
                        frames_per_second
                    )
                    if slot_count <= 0:
                        raise ConfigurationError(
                            f"datastream[{datastream.identifier}]",
                            "source_files",
                            f"VDIF thread {thread_id} does not overlap the "
                            "required payload interval",
                        )
            with nvtx_range("xbeam:vdif-validity-map"):
                validity = np.zeros(slot_count, dtype=np.bool_)
                validity[positions] = ~invalid[selected]
            materialized_bytes += slot_count * template.payload_bytes
            if (
                self._max_materialized_bytes is not None
                and materialized_bytes > self._max_materialized_bytes
            ):
                raise ConfigurationError(
                    "correlate",
                    "max_read_bytes",
                    "planned gap-filled VDIF payload exceeds the bounded "
                    "materialization limit",
                )
            with nvtx_range("xbeam:vdif-destination-map"):
                file_plans = []
                for source_index, (
                    path,
                    first_frame,
                    frame_count,
                    expected_size_bytes,
                    expected_mtime_ns,
                    expected_device,
                    expected_inode,
                ) in enumerate(
                    zip(
                        frames.paths,
                        frames.file_frame_starts,
                        frames.file_frame_counts,
                        frames.file_sizes,
                        frames.file_mtimes_ns,
                        frames.file_devices,
                        frames.file_inodes,
                        strict=True,
                    )
                ):
                    destination_slots = np.full(
                        frame_count, -1, dtype=np.int32
                    )
                    in_file = frames.source_indices[selected] == source_index
                    selected_in_file = selected[in_file]
                    if selected_in_file.size:
                        local_indices = (
                            frames.offsets[selected_in_file]
                            // datastream.frame_size_bytes
                            - first_frame
                        )
                        destination_slots[local_indices] = positions[in_file]
                    file_plans.append(
                        VdifPayloadFilePlan(
                            path,
                            frame_count,
                            destination_slots,
                            first_frame,
                            expected_size_bytes,
                            expected_mtime_ns,
                            expected_device,
                            expected_inode,
                        )
                    )
            payload = VdifPayloadPlan(
                datastream.frame_size_bytes,
                template.payload_bytes,
                slot_count,
                tuple(file_plans),
            )

            start_time = _mjd_time(first_time)
            for channel_index, band in enumerate(bands):
                yield VoltageBatch(
                    samples=payload,
                    start_time=start_time,
                    sample_rate_hz=sample_rate,
                    band_id=band.identifier,
                    polarization=band.polarization,
                    validity=validity,
                    packed_layout=PackedVoltageLayout(
                        frame_payload_bytes=template.payload_bytes,
                        samples_per_frame=template.samples_per_frame,
                        channel_count=channel_count,
                        channel_index=channel_index,
                        complex_data=template.complex_data,
                    ),
                )


__all__ = [
    "VdifFileRecordingSource",
    "VdifPayloadFilePlan",
    "VdifPayloadPlan",
    "VdifSourceRange",
    "VdifStreamingPlan",
    "VdifThreadWindow",
    "materialize_vdif_payload_plans",
]
