# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Bounded VDIF inspection and coarse sidecar indexing.

Only standard VDIF headers are read by this module.  Payload bytes are never
decoded or retained.  The sidecar is deliberately coarse: it maps timestamp
ranges to bounded raw-frame spans, leaving exact thread and timestamp
selection to the caller while it streams the returned candidates.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from bisect import bisect_left
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from cuphoton.xbeam.adapters.vdif.vdif import (
    HEADER_BYTES,
    TruncatedVdifFrameError,
    VdifError,
    VdifHeader,
    _epoch_origin_seconds,
)

DEFAULT_INDEX_CHUNK_BYTES = 64 * 1024 * 1024
DEFAULT_INSPECTION_SAMPLE_COUNT = 17
_INSPECTION_SAMPLE_LAYOUTS = frozenset(("distributed", "edge-blocks"))
_INSPECTION_CACHE_ENTRIES = 32


class VdifIndexError(VdifError):
    """A binary VDIF sidecar is missing, stale, or invalid."""


class VdifOrderedSeekError(VdifError):
    """Direct ordered seek was unsafe or was not explicitly authorized."""


class VdifSourceChangedError(VdifError):
    """A source changed while it was being inspected or indexed."""


@dataclass(frozen=True, order=True, slots=True)
class VdifTimestampKey:
    """A lexicographically ordered raw VDIF timestamp.

    This is intentionally the raw six-bit reference epoch rather than a UTC
    timestamp.  Files crossing the epoch-field wrap require an index instead
    of ordered direct seek.
    """

    reference_epoch: int
    seconds_from_epoch: int
    frame_number: int

    def __post_init__(self) -> None:
        if not 0 <= self.reference_epoch < (1 << 6):
            raise ValueError("reference_epoch must be in [0, 63]")
        if not 0 <= self.seconds_from_epoch < (1 << 30):
            raise ValueError("seconds_from_epoch must be in [0, 2**30)")
        if not 0 <= self.frame_number < (1 << 24):
            raise ValueError("frame_number must be in [0, 2**24)")

    @classmethod
    def from_header(cls, header: VdifHeader) -> VdifTimestampKey:
        return cls(
            header.reference_epoch,
            header.seconds_from_epoch,
            header.frame_number,
        )

    @classmethod
    def from_packed(cls, value: int) -> VdifTimestampKey:
        if not 0 <= value < (1 << 60):
            raise ValueError("packed VDIF timestamp must be in [0, 2**60)")
        return cls(
            reference_epoch=(value >> 54) & 0x3F,
            seconds_from_epoch=(value >> 24) & 0x3FFF_FFFF,
            frame_number=value & 0x00FF_FFFF,
        )

    @property
    def packed(self) -> int:
        """Return a sortable 60-bit integer timestamp key."""

        return (
            (self.reference_epoch << 54)
            | (self.seconds_from_epoch << 24)
            | self.frame_number
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "reference_epoch": self.reference_epoch,
            "seconds_from_epoch": self.seconds_from_epoch,
            "frame_number": self.frame_number,
        }


def timestamp_ranges_for_slots(
    start_slot: int,
    stop_slot: int,
    frames_per_second: int,
    reference_epochs: Sequence[int],
) -> tuple[tuple[VdifTimestampKey, VdifTimestampKey], ...]:
    """Express one absolute interval in each recorded reference epoch.

    VDIF permits an old epoch to remain fixed beyond its half-year. The raw
    index keys therefore need a separate query for each source epoch.
    """
    result = []
    for epoch in reference_epochs:
        origin = _epoch_origin_seconds(epoch, 0) * frames_per_second
        first = max(0, start_slot - origin)
        stop = stop_slot - origin
        if first >= stop:
            continue
        first_second, first_frame = divmod(first, frames_per_second)
        stop_second, stop_frame = divmod(stop, frames_per_second)
        result.append(
            (
                VdifTimestampKey(epoch, first_second, first_frame),
                VdifTimestampKey(epoch, stop_second, stop_frame),
            )
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class VdifFrameRange:
    """A half-open raw-frame range in one constant-frame-size source."""

    start_frame: int
    stop_frame: int
    frame_size_bytes: int

    def __post_init__(self) -> None:
        if self.start_frame < 0 or self.stop_frame < self.start_frame:
            raise ValueError("frame range requires 0 <= start <= stop")
        _validate_frame_size(self.frame_size_bytes)

    @property
    def frame_count(self) -> int:
        return self.stop_frame - self.start_frame

    @property
    def start_offset(self) -> int:
        return self.start_frame * self.frame_size_bytes

    @property
    def stop_offset(self) -> int:
        return self.stop_frame * self.frame_size_bytes

    @property
    def byte_count(self) -> int:
        return self.stop_offset - self.start_offset

    def to_dict(self) -> dict[str, int]:
        return {
            "start_frame": self.start_frame,
            "stop_frame": self.stop_frame,
            "start_offset": self.start_offset,
            "stop_offset": self.stop_offset,
            "frame_count": self.frame_count,
            "byte_count": self.byte_count,
        }


@dataclass(frozen=True, slots=True)
class VdifThreadMetadata:
    """Immutable format metadata for one VDIF thread."""

    thread_id: int
    version: int
    frame_size_bytes: int
    samples_per_frame: int
    log2_channels: int
    channel_count: int
    complex_data: bool
    bits_per_sample: int
    station_id: int
    station_code: str | None
    extended_data_version: int

    @classmethod
    def from_header(cls, header: VdifHeader) -> VdifThreadMetadata:
        return cls(
            thread_id=header.thread_id,
            version=header.version,
            frame_size_bytes=header.frame_length_bytes,
            samples_per_frame=header.samples_per_frame,
            log2_channels=header.log2_channels,
            channel_count=header.channel_count,
            complex_data=header.complex_data,
            bits_per_sample=header.bits_per_sample,
            station_id=header.station_id,
            station_code=header.station_code,
            extended_data_version=header.extended_data_version,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "version": self.version,
            "frame_size_bytes": self.frame_size_bytes,
            "samples_per_frame": self.samples_per_frame,
            "log2_channels": self.log2_channels,
            "channel_count": self.channel_count,
            "complex_data": self.complex_data,
            "bits_per_sample": self.bits_per_sample,
            "station_id": self.station_id,
            "station_code": self.station_code,
            "extended_data_version": self.extended_data_version,
        }


@dataclass(frozen=True, slots=True)
class VdifInspection:
    """Bounded, sampled facts about one VDIF source file."""

    path: Path
    size_bytes: int
    mtime_ns: int
    device: int
    inode: int
    frame_size_bytes: int
    frame_count: int
    trailing_bytes: int
    sampled_frame_indices: tuple[int, ...]
    sampled_threads: tuple[VdifThreadMetadata, ...]
    inconsistent_thread_ids: tuple[int, ...]
    header_versions: tuple[int, ...]
    bits_per_sample: tuple[int, ...]
    channel_counts: tuple[int, ...]
    sampling_kinds: tuple[str, ...]
    station_ids: tuple[int, ...]
    station_codes: tuple[str, ...]
    thread_ids: tuple[int, ...]
    extended_data_versions: tuple[int, ...]
    reference_epochs: tuple[int, ...]
    min_timestamp: VdifTimestampKey | None
    max_timestamp: VdifTimestampKey | None
    sampled_monotonic: bool
    invalid_sample_count: int
    fingerprint_sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Return a stable structure accepted directly by ``json.dumps``."""

        return {
            "path": str(self.path),
            "size_bytes": self.size_bytes,
            "mtime_ns": self.mtime_ns,
            "device": self.device,
            "inode": self.inode,
            "frame_size_bytes": self.frame_size_bytes,
            "frame_count": self.frame_count,
            "trailing_bytes": self.trailing_bytes,
            "sampled_frame_indices": list(self.sampled_frame_indices),
            "sampled_header_count": len(self.sampled_frame_indices),
            "sampled_threads": [
                item.to_dict() for item in self.sampled_threads
            ],
            "inconsistent_thread_ids": list(self.inconsistent_thread_ids),
            "header_versions": list(self.header_versions),
            "bits_per_sample": list(self.bits_per_sample),
            "channel_counts": list(self.channel_counts),
            "sampling_kinds": list(self.sampling_kinds),
            "station_ids": list(self.station_ids),
            "station_codes": list(self.station_codes),
            "thread_ids": list(self.thread_ids),
            "extended_data_versions": list(self.extended_data_versions),
            "reference_epochs": list(self.reference_epochs),
            "min_timestamp": (
                None
                if self.min_timestamp is None
                else self.min_timestamp.to_dict()
            ),
            "max_timestamp": (
                None
                if self.max_timestamp is None
                else self.max_timestamp.to_dict()
            ),
            "sampled_monotonic": self.sampled_monotonic,
            "invalid_sample_count": self.invalid_sample_count,
            "fingerprint_sha256": self.fingerprint_sha256,
        }


@dataclass(slots=True)
class VdifOrderedSearchCache:
    """Reusable monotonic-search anchors for one unchanged VDIF source.

    Each cached lower-bound query safely brackets later timestamp searches.
    The cache never weakens ordered-ingest validation: it only narrows the
    binary-search interval used to find the exact byte range.
    """

    inspection: VdifInspection
    target_keys: list[int] = field(default_factory=list)
    frame_indices: list[int] = field(default_factory=list)

    def bounds(self, target: VdifTimestampKey) -> tuple[int, int, int]:
        """Return insertion position and a safe frame-index bracket."""

        position = bisect_left(self.target_keys, target.packed)
        low = 0 if position == 0 else self.frame_indices[position - 1]
        high = (
            self.inspection.frame_count
            if position == len(self.target_keys)
            else self.frame_indices[position]
        )
        return position, low, high

    def remember(
        self, position: int, target: VdifTimestampKey, frame_index: int
    ) -> None:
        if (
            position < len(self.target_keys)
            and self.target_keys[position] == target.packed
        ):
            if self.frame_indices[position] != frame_index:
                raise VdifOrderedSeekError(
                    "cached ordered VDIF boundary changed"
                )
            return
        self.target_keys.insert(position, target.packed)
        self.frame_indices.insert(position, frame_index)


@dataclass(frozen=True, slots=True)
class VdifHeaderExtraction:
    """Completed raw-header extraction from a stable VDIF source."""

    source_path: Path
    output_path: Path
    frame_size_bytes: int
    first_frame: int
    frame_count: int
    output_bytes: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": str(self.source_path),
            "output_path": str(self.output_path),
            "frame_size_bytes": self.frame_size_bytes,
            "header_size_bytes": HEADER_BYTES,
            "first_frame": self.first_frame,
            "frame_count": self.frame_count,
            "output_bytes": self.output_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class _SourceStat:
    size_bytes: int
    mtime_ns: int
    device: int
    inode: int

    @classmethod
    def from_path(cls, path: Path) -> _SourceStat:
        value = path.stat()
        return cls(
            value.st_size,
            value.st_mtime_ns,
            value.st_dev,
            value.st_ino,
        )


def _validate_frame_size(frame_size_bytes: int) -> None:
    if frame_size_bytes < HEADER_BYTES or frame_size_bytes % 8:
        raise VdifError(
            "VDIF frame size must be at least 32 bytes and a multiple of 8"
        )


def _read_header(
    fd: int, frame_index: int, frame_size_bytes: int
) -> tuple[VdifHeader, bytes]:
    offset = frame_index * frame_size_bytes
    data = os.pread(fd, HEADER_BYTES, offset)
    if len(data) != HEADER_BYTES:
        raise TruncatedVdifFrameError(
            f"truncated VDIF header at byte {offset}: got {len(data)} bytes"
        )
    try:
        header = VdifHeader.from_bytes(data, require_two_bit=False)
    except VdifError as exc:
        raise type(exc)(f"VDIF header at byte {offset}: {exc}") from exc
    if header.frame_length_bytes != frame_size_bytes:
        raise VdifError(
            f"VDIF header at byte {offset} declares "
            f"{header.frame_length_bytes} bytes; expected {frame_size_bytes}"
        )
    return header, data


def _sample_indices(frame_count: int, sample_count: int) -> tuple[int, ...]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if frame_count <= 0:
        return ()
    if frame_count <= sample_count:
        return tuple(range(frame_count))
    if sample_count == 1:
        return (0,)
    return tuple(
        sorted(
            {
                index * (frame_count - 1) // (sample_count - 1)
                for index in range(sample_count)
            }
        )
    )


def _edge_block_sample_indices(
    frame_count: int, sample_count: int
) -> tuple[int, ...]:
    """Favor coalesced edge reads while retaining distributed evidence."""

    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if frame_count <= 0:
        return ()
    if frame_count <= sample_count:
        return tuple(range(frame_count))
    if sample_count <= DEFAULT_INSPECTION_SAMPLE_COUNT:
        return _sample_indices(frame_count, sample_count)

    distributed_count = min(DEFAULT_INSPECTION_SAMPLE_COUNT, sample_count)
    edge_count = sample_count - distributed_count
    prefix_count = (edge_count + 1) // 2
    suffix_count = edge_count // 2
    indices = set(range(prefix_count))
    indices.update(range(frame_count - suffix_count, frame_count))
    indices.update(
        (index + 1) * (frame_count - 1) // (distributed_count + 1)
        for index in range(distributed_count)
    )

    # Close rare collisions between a distributed point and an edge block
    # without ever exceeding the caller's bounded probe count.
    candidate = prefix_count
    while len(indices) < sample_count:
        if candidate < frame_count - suffix_count:
            indices.add(candidate)
        candidate += 1
    return tuple(sorted(indices))


def _sample_runs(indices: tuple[int, ...]) -> Iterator[tuple[int, int]]:
    """Yield half-open consecutive runs from sorted frame indices."""

    if not indices:
        return
    start = previous = indices[0]
    for index in indices[1:]:
        if index != previous + 1:
            yield start, previous + 1
            start = index
        previous = index
    yield start, previous + 1


def _read_sampled_headers(
    fd: int,
    indices: tuple[int, ...],
    frame_size_bytes: int,
    *,
    coalesce: bool,
) -> tuple[list[VdifHeader], list[bytes]]:
    """Read selected headers, coalescing adjacent frames when requested."""

    headers: list[VdifHeader] = []
    raw_headers: list[bytes] = []
    runs = (
        _sample_runs(indices)
        if coalesce
        else ((item, item + 1) for item in indices)
    )
    for start, stop in runs:
        run_frames = stop - start
        if run_frames == 1:
            header, raw = _read_header(fd, start, frame_size_bytes)
            headers.append(header)
            raw_headers.append(raw)
            continue

        offset = start * frame_size_bytes
        byte_count = run_frames * frame_size_bytes
        buffer = bytearray(byte_count)
        view = memoryview(buffer)
        received = 0
        while received < byte_count:
            count = os.preadv(fd, [view[received:]], offset + received)
            if count == 0:
                raise TruncatedVdifFrameError(
                    f"truncated VDIF extent at byte {offset}: expected "
                    f"{byte_count} bytes, got {received}"
                )
            received += count
        for local_index in range(run_frames):
            frame_index = start + local_index
            header_offset = local_index * frame_size_bytes
            raw = bytes(view[header_offset : header_offset + HEADER_BYTES])
            try:
                header = VdifHeader.from_bytes(raw, require_two_bit=False)
            except VdifError as exc:
                raise type(exc)(
                    f"VDIF header at byte "
                    f"{frame_index * frame_size_bytes}: {exc}"
                ) from exc
            if header.frame_length_bytes != frame_size_bytes:
                raise VdifError(
                    f"VDIF header at byte "
                    f"{frame_index * frame_size_bytes} declares "
                    f"{header.frame_length_bytes} bytes; expected "
                    f"{frame_size_bytes}"
                )
            headers.append(header)
            raw_headers.append(raw)
        view.release()
    return headers, raw_headers


def _assert_stat_unchanged(path: Path, expected: _SourceStat) -> None:
    if _SourceStat.from_path(path) != expected:
        raise VdifSourceChangedError(
            f"VDIF source changed while it was being read: {path}"
        )


def extract_vdif_headers(
    source: str | os.PathLike[str],
    output: str | os.PathLike[str],
    *,
    frame_size_bytes: int | None = None,
    first_frame: int = 0,
    frame_count: int | None = None,
    output_buffer_bytes: int = 8 * 1024 * 1024,
) -> VdifHeaderExtraction:
    """Extract exact standard headers while seeking over frame payloads.

    The output contains one contiguous 32-byte VDIF header per selected input
    frame and no container header. It is linked into place only after the
    complete source range has been read and the source identity revalidated.
    """

    if first_frame < 0:
        raise ValueError("first_frame must be non-negative")
    if frame_count is not None and frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if output_buffer_bytes < HEADER_BYTES:
        raise ValueError("output_buffer_bytes must hold at least one header")

    inspection = inspect_vdif(
        source,
        frame_size_bytes=frame_size_bytes,
        sample_count=1,
    )
    if inspection.trailing_bytes:
        raise TruncatedVdifFrameError(
            f"VDIF source has {inspection.trailing_bytes} trailing bytes "
            f"after {inspection.frame_count} complete frames"
        )
    available = inspection.frame_count - first_frame
    if available < 0:
        raise ValueError(
            f"first_frame={first_frame} exceeds the {inspection.frame_count} "
            "frames in the source"
        )
    selected_count = available if frame_count is None else frame_count
    if selected_count > available:
        raise ValueError(
            f"requested frames [{first_frame}, "
            f"{first_frame + selected_count}) exceed the "
            f"{inspection.frame_count} frames in the source"
        )

    destination = Path(output).expanduser().resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"header output already exists: {destination}")
    if not destination.parent.is_dir():
        raise FileNotFoundError(
            f"header output directory does not exist: {destination.parent}"
        )

    before = _SourceStat.from_path(inspection.path)
    source_fd = os.open(inspection.path, os.O_RDONLY)
    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    buffered = bytearray()
    try:
        if selected_count and hasattr(os, "posix_fadvise"):
            try:
                os.posix_fadvise(
                    source_fd,
                    first_frame * inspection.frame_size_bytes,
                    selected_count * inspection.frame_size_bytes,
                    os.POSIX_FADV_RANDOM,
                )
            except OSError:
                # Advice is an optimization only; sparse extraction remains
                # correct on filesystems that reject or ignore it.
                pass
        with os.fdopen(
            temporary_fd, "wb", buffering=output_buffer_bytes
        ) as sink:
            os.lseek(
                source_fd,
                first_frame * inspection.frame_size_bytes,
                os.SEEK_SET,
            )
            payload_skip = inspection.frame_size_bytes - HEADER_BYTES
            for index in range(selected_count):
                raw = os.read(source_fd, HEADER_BYTES)
                if len(raw) != HEADER_BYTES:
                    frame_index = first_frame + index
                    raise TruncatedVdifFrameError(
                        f"truncated VDIF header at byte "
                        f"{frame_index * inspection.frame_size_bytes}: "
                        f"got {len(raw)} bytes"
                    )
                buffered.extend(raw)
                digest.update(raw)
                if len(buffered) >= output_buffer_bytes:
                    sink.write(buffered)
                    buffered.clear()
                if index + 1 < selected_count:
                    os.lseek(source_fd, payload_skip, os.SEEK_CUR)
            if buffered:
                sink.write(buffered)
                buffered.clear()
            sink.flush()
            os.fsync(sink.fileno())
        _assert_stat_unchanged(inspection.path, before)
        # A hard link supplies atomic no-overwrite publication. Both paths are
        # in the same directory/filesystem by construction.
        os.link(temporary, destination)
        temporary.unlink()
    except BaseException:
        try:
            os.close(temporary_fd)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise
    finally:
        os.close(source_fd)

    return VdifHeaderExtraction(
        inspection.path,
        destination,
        inspection.frame_size_bytes,
        first_frame,
        selected_count,
        selected_count * HEADER_BYTES,
        digest.hexdigest(),
    )


def _inspect_vdif_uncached(
    source: str | os.PathLike[str],
    *,
    frame_size_bytes: int | None = None,
    sample_count: int = DEFAULT_INSPECTION_SAMPLE_COUNT,
    sample_layout: str = "distributed",
) -> VdifInspection:
    """Read a fixed number of distributed headers from a VDIF source."""

    path = Path(source).expanduser().resolve(strict=True)
    before = _SourceStat.from_path(path)
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if sample_layout not in _INSPECTION_SAMPLE_LAYOUTS:
        raise ValueError(
            f"sample_layout must be one of "
            f"{sorted(_INSPECTION_SAMPLE_LAYOUTS)}"
        )

    fd = os.open(path, os.O_RDONLY)
    try:
        if frame_size_bytes is None:
            if before.size_bytes == 0:
                raise VdifError(
                    "cannot infer a VDIF frame size from an empty source"
                )
            first_bytes = os.pread(fd, HEADER_BYTES, 0)
            if len(first_bytes) != HEADER_BYTES:
                raise TruncatedVdifFrameError(
                    "standard VDIF header needs 32 bytes; "
                    f"got {len(first_bytes)}"
                )
            first_header = VdifHeader.from_bytes(
                first_bytes, require_two_bit=False
            )
            frame_size_bytes = first_header.frame_length_bytes
        _validate_frame_size(frame_size_bytes)

        if before.size_bytes >= HEADER_BYTES:
            _read_header(fd, 0, frame_size_bytes)

        frame_count, trailing_bytes = divmod(
            before.size_bytes, frame_size_bytes
        )
        indices = (
            _sample_indices(frame_count, sample_count)
            if sample_layout == "distributed"
            else _edge_block_sample_indices(frame_count, sample_count)
        )
        headers, raw_headers = _read_sampled_headers(
            fd,
            indices,
            frame_size_bytes,
            coalesce=sample_layout == "edge-blocks",
        )
    finally:
        os.close(fd)
    _assert_stat_unchanged(path, before)

    digest = hashlib.sha256()
    digest.update(
        b"cuphoton.xbeam.adapters.vdif.vdif-header-fingerprint-v1\0"
    )
    digest.update(before.size_bytes.to_bytes(8, "little", signed=False))
    digest.update(frame_size_bytes.to_bytes(8, "little", signed=False))
    for frame_index, raw in zip(indices, raw_headers, strict=True):
        digest.update(frame_index.to_bytes(8, "little", signed=False))
        digest.update(raw)

    formats: dict[int, VdifThreadMetadata] = {}
    inconsistent: set[int] = set()
    for header in headers:
        metadata = VdifThreadMetadata.from_header(header)
        previous = formats.setdefault(header.thread_id, metadata)
        if previous != metadata:
            inconsistent.add(header.thread_id)

    keys = [VdifTimestampKey.from_header(header) for header in headers]
    monotonic = all(
        left <= right for left, right in zip(keys, keys[1:], strict=False)
    )
    return VdifInspection(
        path=path,
        size_bytes=before.size_bytes,
        mtime_ns=before.mtime_ns,
        device=before.device,
        inode=before.inode,
        frame_size_bytes=frame_size_bytes,
        frame_count=frame_count,
        trailing_bytes=trailing_bytes,
        sampled_frame_indices=indices,
        sampled_threads=tuple(formats[key] for key in sorted(formats)),
        inconsistent_thread_ids=tuple(sorted(inconsistent)),
        header_versions=tuple(sorted({item.version for item in headers})),
        bits_per_sample=tuple(
            sorted({item.bits_per_sample for item in headers})
        ),
        channel_counts=tuple(
            sorted({item.channel_count for item in headers})
        ),
        sampling_kinds=tuple(
            sorted(
                {
                    "complex" if item.complex_data else "real"
                    for item in headers
                }
            )
        ),
        station_ids=tuple(sorted({item.station_id for item in headers})),
        station_codes=tuple(
            sorted(
                {
                    item.station_code
                    for item in headers
                    if item.station_code is not None
                }
            )
        ),
        thread_ids=tuple(sorted({item.thread_id for item in headers})),
        extended_data_versions=tuple(
            sorted({item.extended_data_version for item in headers})
        ),
        reference_epochs=tuple(
            sorted({item.reference_epoch for item in headers})
        ),
        min_timestamp=min(keys, default=None),
        max_timestamp=max(keys, default=None),
        sampled_monotonic=monotonic,
        invalid_sample_count=sum(item.invalid for item in headers),
        fingerprint_sha256=digest.hexdigest(),
    )


@lru_cache(maxsize=_INSPECTION_CACHE_ENTRIES)
def _cached_inspection(
    path: Path,
    frame_size_bytes: int | None,
    sample_count: int,
    sample_layout: str,
    source_stat: _SourceStat,
) -> VdifInspection:
    # ``source_stat`` participates in the cache key. The uncached inspector
    # independently verifies the same identity before and after reading.
    return _inspect_vdif_uncached(
        path,
        frame_size_bytes=frame_size_bytes,
        sample_count=sample_count,
        sample_layout=sample_layout,
    )


def inspect_vdif(
    source: str | os.PathLike[str],
    *,
    frame_size_bytes: int | None = None,
    sample_count: int = DEFAULT_INSPECTION_SAMPLE_COUNT,
    sample_layout: str = "distributed",
) -> VdifInspection:
    """Read bounded headers, reusing unchanged source inspections.

    ``distributed`` performs sparse probes across the complete source.
    ``edge-blocks`` coalesces most probes into bounded prefix/suffix reads and
    retains a small distributed sample; it is intended for sources whose
    global ordering is independently asserted and validated during ingest.
    """

    path = Path(source).expanduser().resolve(strict=True)
    before = _SourceStat.from_path(path)
    result = _cached_inspection(
        path, frame_size_bytes, sample_count, sample_layout, before
    )
    _assert_stat_unchanged(path, before)
    return result


def _lower_bound(
    fd: int,
    frame_count: int,
    frame_size_bytes: int,
    target: VdifTimestampKey,
    *,
    low: int = 0,
    high: int | None = None,
) -> int:
    if high is None:
        high = frame_count
    if not 0 <= low <= high <= frame_count:
        raise ValueError("invalid ordered VDIF binary-search bounds")
    while low < high:
        middle = (low + high) // 2
        header, _ = _read_header(fd, middle, frame_size_bytes)
        if VdifTimestampKey.from_header(header) < target:
            low = middle + 1
        else:
            high = middle
    return low


def _cached_lower_bound(
    fd: int,
    frame_count: int,
    frame_size_bytes: int,
    target: VdifTimestampKey,
    cache: VdifOrderedSearchCache | None,
) -> int:
    if cache is None:
        return _lower_bound(fd, frame_count, frame_size_bytes, target)
    position, low, high = cache.bounds(target)
    if (
        position < len(cache.target_keys)
        and cache.target_keys[position] == target.packed
    ):
        return cache.frame_indices[position]
    result = _lower_bound(
        fd,
        frame_count,
        frame_size_bytes,
        target,
        low=low,
        high=high,
    )
    cache.remember(position, target, result)
    return result


def _header_keys(words: np.ndarray) -> np.ndarray:
    word0 = words[:, 0]
    word1 = words[:, 1]
    keys = (
        (word1 >> np.uint32(24)).astype(np.uint64) & np.uint64(0x3F)
    ) << np.uint64(54)
    keys |= (word0.astype(np.uint64) & np.uint64(0x3FFF_FFFF)) << np.uint64(
        24
    )
    keys |= word1.astype(np.uint64) & np.uint64(0x00FF_FFFF)
    return keys


def _parse_format_header(
    headers: np.ndarray,
    local_index: int,
    chunk_start: int,
    frame_size_bytes: int,
) -> VdifThreadMetadata:
    frame_index = chunk_start + local_index
    try:
        header = VdifHeader.from_bytes(
            headers[local_index].tobytes(), require_two_bit=False
        )
    except VdifError as exc:
        raise type(exc)(
            f"VDIF header at byte {frame_index * frame_size_bytes}: {exc}"
        ) from exc
    if header.frame_length_bytes != frame_size_bytes:
        raise VdifError(
            f"VDIF header at byte {frame_index * frame_size_bytes} "
            f"declares {header.frame_length_bytes} bytes; expected "
            f"{frame_size_bytes}"
        )
    return VdifThreadMetadata.from_header(header)


def _validate_format_chunk(
    headers: np.ndarray,
    words: np.ndarray,
    chunk_start: int,
    frame_size_bytes: int,
    formats: dict[int, VdifThreadMetadata],
    signatures: dict[int, tuple[int, int]],
) -> None:
    thread_ids = ((words[:, 3] >> np.uint32(16)) & np.uint32(0x3FF)).astype(
        np.int64
    )
    signature_a = (
        ((words[:, 0] >> np.uint32(30)) & np.uint32(1)).astype(np.uint64)
    ) << np.uint64(34)
    signature_a |= (words[:, 1] >> np.uint32(30)).astype(
        np.uint64
    ) << np.uint64(32)
    signature_a |= words[:, 2].astype(np.uint64)
    signature_b = words[:, 3].astype(np.uint64)
    signature_b |= (words[:, 4] >> np.uint32(24)).astype(
        np.uint64
    ) << np.uint64(32)
    for thread_id in np.unique(thread_ids):
        positions = np.flatnonzero(thread_ids == thread_id)
        first = int(positions[0])
        signature = (int(signature_a[first]), int(signature_b[first]))
        expected = signatures.get(int(thread_id))
        if expected is None:
            metadata = _parse_format_header(
                headers, first, chunk_start, frame_size_bytes
            )
            formats[int(thread_id)] = metadata
            signatures[int(thread_id)] = signature
            expected = signature
        mismatches = positions[
            (signature_a[positions] != expected[0])
            | (signature_b[positions] != expected[1])
        ]
        if mismatches.size:
            mismatch = int(mismatches[0])
            metadata = _parse_format_header(
                headers, mismatch, chunk_start, frame_size_bytes
            )
            raise VdifOrderedSeekError(
                f"VDIF format changes in thread {metadata.thread_id} at "
                f"byte {(chunk_start + mismatch) * frame_size_bytes}"
            )


def _validate_ordered_range(
    fd: int,
    frame_count: int,
    frame_size_bytes: int,
    validation_start: int,
    validation_stop: int,
    lower: int,
    upper: int,
    start_key: VdifTimestampKey,
    stop_key: VdifTimestampKey,
    selected_threads: frozenset[int] | None,
) -> tuple[int | None, int | None]:
    """Validate an ordered envelope using one bounded read buffer."""

    if validation_start >= validation_stop:
        return None, None
    first_match: int | None = None
    last_match: int | None = None
    previous_key: int | None = None
    formats: dict[int, VdifThreadMetadata] = {}
    signatures: dict[int, tuple[int, int]] = {}
    # Keep this validation working set consistent with the independently
    # configurable sidecar-index scanner.  This bounds host scratch memory;
    # it is unrelated to GPU ingest transfer granularity.
    chunk_frames = max(1, DEFAULT_INDEX_CHUNK_BYTES // frame_size_bytes)
    read_buffer = bytearray(chunk_frames * frame_size_bytes)
    for chunk_start in range(validation_start, validation_stop, chunk_frames):
        chunk_stop = min(validation_stop, chunk_start + chunk_frames)
        chunk_bytes = (chunk_stop - chunk_start) * frame_size_bytes
        offset = chunk_start * frame_size_bytes
        raw = memoryview(read_buffer)[:chunk_bytes]
        received = 0
        while received < chunk_bytes:
            count = os.preadv(fd, [raw[received:]], offset + received)
            if count == 0:
                raise TruncatedVdifFrameError(
                    f"truncated VDIF extent at byte {offset}: expected "
                    f"{chunk_bytes} bytes, got {received}"
                )
            received += count
        rows = np.frombuffer(raw, dtype=np.uint8).reshape(
            chunk_stop - chunk_start, frame_size_bytes
        )
        headers = np.ascontiguousarray(rows[:, :HEADER_BYTES])
        words = headers.view("<u4").reshape(-1, 8)
        keys = _header_keys(words)
        if previous_key is not None and int(keys[0]) < previous_key:
            raise VdifOrderedSeekError(
                "VDIF ordering decreases at byte "
                f"{chunk_start * frame_size_bytes}"
            )
        decreases = np.flatnonzero(keys[1:] < keys[:-1])
        if decreases.size:
            frame_index = chunk_start + int(decreases[0]) + 1
            raise VdifOrderedSeekError(
                "VDIF ordering decreases at byte "
                f"{frame_index * frame_size_bytes}"
            )
        previous_key = int(keys[-1])

        _validate_format_chunk(
            headers,
            words,
            chunk_start,
            frame_size_bytes,
            formats,
            signatures,
        )

        indices = np.arange(chunk_start, chunk_stop, dtype=np.int64)
        start_packed = np.uint64(start_key.packed)
        stop_packed = np.uint64(stop_key.packed)
        if np.any((indices < lower) & (keys >= start_packed)):
            raise VdifOrderedSeekError(
                "binary-seek lower boundary disagrees with local "
                "VDIF timestamps"
            )
        inside = (indices >= lower) & (indices < upper)
        outside_keys = (keys < start_packed) | (keys >= stop_packed)
        if np.any(inside & outside_keys):
            raise VdifOrderedSeekError(
                "binary-seek bounds disagree with local VDIF timestamps"
            )
        if np.any((indices >= upper) & (keys < stop_packed)):
            raise VdifOrderedSeekError(
                "binary-seek upper boundary disagrees with local "
                "VDIF timestamps"
            )
        matches = inside
        if selected_threads is not None:
            thread_ids = (words[:, 3] >> np.uint32(16)) & np.uint32(0x3FF)
            wanted = np.fromiter(selected_threads, dtype=np.uint32)
            matches &= np.isin(thread_ids, wanted)
        positions = np.flatnonzero(matches)
        if positions.size:
            if first_match is None:
                first_match = chunk_start + int(positions[0])
            last_match = chunk_start + int(positions[-1]) + 1
        del rows, headers, words, keys, indices, raw
    if first_match is None or last_match is None:
        return None, None
    return first_match, last_match


def locate_ordered_vdif_range(
    source: str | os.PathLike[str],
    start_key: VdifTimestampKey,
    stop_key: VdifTimestampKey,
    *,
    frame_size_bytes: int | None = None,
    thread_ids: Sequence[int] | None = None,
    assume_ordered: bool = False,
    max_range_bytes: int | None = None,
    validate_all_headers: bool = True,
    search_cache: VdifOrderedSearchCache | None = None,
) -> tuple[VdifFrameRange, ...]:
    """Binary-seek an authorized ordered source and validate its local range.

    The returned extent can contain interleaved unrequested threads. Every
    header in it is validated before return by default using bounded scratch
    space, while the retained range metadata remains constant-size. Callers
    that unconditionally validate the retained headers during their following
    ingest pass may set ``validate_all_headers=False`` to avoid reading the
    same strided headers twice. In that mode the returned extent can retain
    unrequested thread frames at its edges; downstream thread filtering makes
    this conservative over-read safe.
    """

    if not assume_ordered:
        raise VdifOrderedSeekError(
            "direct VDIF seek requires explicit assume_ordered=True"
        )
    if start_key >= stop_key:
        raise ValueError("timestamp query requires start_key < stop_key")
    selected_threads = None
    if thread_ids is not None:
        selected_threads = frozenset(int(item) for item in thread_ids)
        if any(not 0 <= item < (1 << 10) for item in selected_threads):
            raise ValueError("VDIF thread IDs must be in [0, 1023]")
        if not selected_threads:
            return ()

    inspection = (
        inspect_vdif(source, frame_size_bytes=frame_size_bytes)
        if search_cache is None
        else search_cache.inspection
    )
    requested_path = Path(source).expanduser().resolve(strict=True)
    if requested_path != inspection.path:
        raise VdifOrderedSeekError(
            "ordered VDIF search cache belongs to a different source"
        )
    if (
        frame_size_bytes is not None
        and frame_size_bytes != inspection.frame_size_bytes
    ):
        raise VdifOrderedSeekError(
            "ordered VDIF search cache uses a different frame size"
        )
    if inspection.trailing_bytes:
        raise TruncatedVdifFrameError(
            f"VDIF source has {inspection.trailing_bytes} trailing bytes"
        )
    if not inspection.sampled_monotonic:
        raise VdifOrderedSeekError(
            "sampled VDIF headers are not monotonically ordered"
        )
    before = _SourceStat.from_path(inspection.path)
    if (
        before.size_bytes,
        before.mtime_ns,
        before.device,
        before.inode,
    ) != (
        inspection.size_bytes,
        inspection.mtime_ns,
        inspection.device,
        inspection.inode,
    ):
        raise VdifSourceChangedError(
            f"VDIF source changed since it was inspected: {inspection.path}"
        )
    fd = os.open(inspection.path, os.O_RDONLY)
    try:
        lower = _cached_lower_bound(
            fd,
            inspection.frame_count,
            inspection.frame_size_bytes,
            start_key,
            search_cache,
        )
        upper = _cached_lower_bound(
            fd,
            inspection.frame_count,
            inspection.frame_size_bytes,
            stop_key,
            search_cache,
        )
        if (
            max_range_bytes is not None
            and (upper - lower) * inspection.frame_size_bytes
            > max_range_bytes
        ):
            raise VdifOrderedSeekError(
                "ordered VDIF range exceeds the max_read_bytes bounded "
                "ingest limit"
            )
        if validate_all_headers:
            validation_start = max(0, lower - 1)
            validation_stop = min(inspection.frame_count, upper + 1)
            first_match, last_match = _validate_ordered_range(
                fd,
                inspection.frame_count,
                inspection.frame_size_bytes,
                validation_start,
                validation_stop,
                lower,
                upper,
                start_key,
                stop_key,
                selected_threads,
            )
        else:
            # ``_lower_bound`` has already checked the boundary keys through
            # direct header reads. Preserve the complete timestamp envelope;
            # the recording source will validate and filter every retained
            # header before exposing payload bytes.
            first_match, last_match = (
                (None, None) if lower == upper else (lower, upper)
            )
    finally:
        os.close(fd)
    _assert_stat_unchanged(inspection.path, before)
    if first_match is None or last_match is None:
        return ()
    return (
        VdifFrameRange(
            first_match,
            last_match,
            inspection.frame_size_bytes,
        ),
    )


__all__ = [
    "DEFAULT_INDEX_CHUNK_BYTES",
    "DEFAULT_INSPECTION_SAMPLE_COUNT",
    "VdifFrameRange",
    "VdifHeaderExtraction",
    "VdifIndexError",
    "VdifInspection",
    "VdifOrderedSeekError",
    "VdifOrderedSearchCache",
    "VdifSourceChangedError",
    "VdifThreadMetadata",
    "VdifTimestampKey",
    "extract_vdif_headers",
    "inspect_vdif",
    "locate_ordered_vdif_range",
]
