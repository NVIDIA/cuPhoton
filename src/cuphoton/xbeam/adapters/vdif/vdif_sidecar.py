# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Portable memory-mapped VDIF frame indexes.

Schema v2 entries contain two little-endian uint64 words: timestamp and a
location word (10-bit thread, merged invalid bit, 53-bit physical ordinal).
Schema v3 appends a uint64 VdifFrameState bitmask without changing either
original word. Schema v4 keeps those timestamp-sorted entries unchanged and
appends a compact physical-order timestamp-run table. Header rejection takes
precedence over format rejection; the explicit invalid flag is retained
independently. Rebuilding an older sidecar requires the builder's force
option.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntFlag
from functools import lru_cache
from glob import escape
from pathlib import Path
from typing import Any

import numpy as np

from cuphoton.xbeam.adapters.vdif.vdif_index import (
    VdifFrameRange,
    VdifIndexError,
    VdifThreadMetadata,
    VdifTimestampKey,
)
from cuphoton.xbeam.profiling import nvtx_range

SIDECAR_MAGIC = b"XCVDIX1\n"
SIDECAR_SCHEMA_VERSION = 4
SIDECAR_HEADER_BYTES = 64 * 1024
SIDECAR_ENTRY_BYTES = 24
SIDECAR_PHYSICAL_TIMESTAMP_RUN_BYTES = 24
_ENTRY_BYTES_BY_VERSION = {2: 16, 3: 24, 4: 24}
_HEADER_PREFIX = struct.Struct("<8sII")
_FRAME_INDEX_MASK = (1 << 53) - 1
_INVALID_FRAME_MASK = 1 << 53
_DIRECT_IO_ALIGNMENT = 4096
_BITMAP_MIN_ENTRIES = 4096
_BITMAP_MAX_TEMP_BYTES = 128 * 1024 * 1024
_PHYSICAL_QUERY_BLOCK_ENTRIES = 1 << 16


def _align_up(value: int, alignment: int = _DIRECT_IO_ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def binary_index_size(
    entry_count: int,
    *,
    schema_version: int = SIDECAR_SCHEMA_VERSION,
    physical_timestamp_run_count: int = 0,
) -> int:
    """Return the padded on-disk size of a binary index."""

    if entry_count < 0:
        raise ValueError("entry_count must be nonnegative")
    if physical_timestamp_run_count < 0:
        raise ValueError("physical_timestamp_run_count must be nonnegative")
    if schema_version not in _ENTRY_BYTES_BY_VERSION:
        raise VdifIndexError("unsupported binary VDIF sidecar version")
    if schema_version < 4 and physical_timestamp_run_count:
        raise VdifIndexError(
            "physical timestamp runs require binary sidecar schema v4"
        )
    entries_stop = (
        SIDECAR_HEADER_BYTES
        + entry_count * _ENTRY_BYTES_BY_VERSION[schema_version]
    )
    run_offset = (
        _align_up(entries_stop) if schema_version >= 4 else entries_stop
    )
    return _align_up(
        run_offset
        + physical_timestamp_run_count * SIDECAR_PHYSICAL_TIMESTAMP_RUN_BYTES
    )


def physical_timestamp_run_offset(
    entry_count: int, *, schema_version: int = SIDECAR_SCHEMA_VERSION
) -> int:
    """Return the byte offset of the appended physical timestamp runs."""

    if entry_count < 0:
        raise ValueError("entry_count must be nonnegative")
    if schema_version < 4 or schema_version not in _ENTRY_BYTES_BY_VERSION:
        raise VdifIndexError(
            "physical timestamp runs require binary sidecar schema v4"
        )
    return _align_up(
        SIDECAR_HEADER_BYTES
        + entry_count * _ENTRY_BYTES_BY_VERSION[schema_version]
    )


def encode_binary_index_header(document: dict[str, Any]) -> bytes:
    """Encode metadata into the fixed-size, mmap-friendly header."""

    encoded = json.dumps(
        document,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    limit = SIDECAR_HEADER_BYTES - _HEADER_PREFIX.size
    if not 0 < len(encoded) <= limit:
        raise VdifIndexError("binary VDIF sidecar metadata is too large")
    prefix = _HEADER_PREFIX.pack(
        SIDECAR_MAGIC, SIDECAR_SCHEMA_VERSION, len(encoded)
    )
    padding = SIDECAR_HEADER_BYTES - len(prefix) - len(encoded)
    return prefix + encoded + bytes(padding)


class VdifFrameState(IntFlag):
    """Frame facts; rejection can coexist with an explicit invalid flag.

    UNKNOWN_INVALID is synthesized only for v2, which merged all causes.
    Header rejection means the stored timestamp/thread fields are raw values,
    not an assurance that they describe a trustworthy time interval.
    """

    VALID = 0
    EXPLICIT_INVALID = 1
    HEADER_REJECTED = 2
    FORMAT_REJECTED = 4
    UNKNOWN_INVALID = 8
    RECORDED_FILLER = 16


@dataclass(frozen=True, slots=True)
class VdifBinaryIndexSummary:
    """Validated metadata for one binary sidecar."""

    source_path: Path
    index_path: Path
    frame_size_bytes: int
    frame_count: int
    entry_count: int
    source_size_bytes: int
    source_ordered: bool
    sample_fingerprint_sha256: str
    thread_metadata: tuple[VdifThreadMetadata, ...]
    rejected_format_frame_count: int = 0
    rejected_header_frame_count: int = 0
    recorded_filler_classifier: str | None = None
    physical_timestamp_run_count: int = 0
    index_identity: tuple[int, ...] = ()
    schema_version: int = SIDECAR_SCHEMA_VERSION

    @property
    def has_frame_states(self) -> bool:
        return self.schema_version >= 3

    @property
    def has_physical_timestamp_runs(self) -> bool:
        return self.schema_version >= 4

    @property
    def entry_bytes(self) -> int:
        return _ENTRY_BYTES_BY_VERSION[self.schema_version]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": str(self.source_path),
            "index_path": str(self.index_path),
            "schema_version": self.schema_version,
            "has_frame_states": self.has_frame_states,
            "entry_bytes": self.entry_bytes,
            "format": "vdix-binary",
            "frame_size_bytes": self.frame_size_bytes,
            "frame_count": self.frame_count,
            "entry_count": self.entry_count,
            "source_size_bytes": self.source_size_bytes,
            "source_ordered": self.source_ordered,
            "sample_fingerprint_sha256": self.sample_fingerprint_sha256,
            "rejected_format_frame_count": (self.rejected_format_frame_count),
            "rejected_header_frame_count": (self.rejected_header_frame_count),
            "recorded_filler_classifier": self.recorded_filler_classifier,
            "physical_timestamp_run_count": (
                self.physical_timestamp_run_count
            ),
            "has_physical_timestamp_runs": (self.has_physical_timestamp_runs),
            "thread_metadata": [
                item.to_dict() for item in self.thread_metadata
            ],
        }


def resolve_vdif_binary_index_path(
    source: str | os.PathLike[str],
    index_dir: str | os.PathLike[str] | None = None,
    *,
    frame_size_bytes: int | None = None,
) -> Path:
    """Return the portable binary-sidecar location for one source."""

    source_path = Path(source).expanduser().resolve(strict=True)
    if index_dir is None:
        return Path(f"{source_path}.vdix")
    directory = Path(index_dir).expanduser().resolve(strict=False)
    primary = directory / f"{source_path.name}.vdix"

    def matches(candidate: Path) -> bool:
        try:
            _validate_binary_index_path(
                source_path, candidate, frame_size_bytes=frame_size_bytes
            )
        except VdifIndexError:
            return False
        return True

    owned_fallback = None
    ambiguous_primary = None
    if primary.exists():
        if matches(primary):
            return primary
        # A stale or unreadable basename is not evidence of a collision.
        # Retain its diagnostics unless an existing index proves ownership.
        owner = _sidecar_owner(source_path, primary)
        if owner is True:
            owned_fallback = primary
        elif owner is None:
            ambiguous_primary = primary

    # Preserve existing path-hashed names, including their stale/corrupt
    # diagnostics. New names below do not depend on the mount or parent path.
    old_digest = hashlib.sha256(os.fsencode(str(source_path))).hexdigest()[
        :16
    ]
    old_path = directory / f"{source_path.name}.{old_digest}.vdix"
    collision_path = _collision_index_path(source_path, directory)
    for candidate in (old_path, collision_path):
        if candidate.exists():
            if matches(candidate):
                return candidate
            if owned_fallback is None:
                owned_fallback = candidate

    # A relocated path-hashed index or a stale collision index has no direct
    # name match. Search only this basename's sidecars on that fallback path.
    for candidate in sorted(
        directory.glob(f"{escape(source_path.name)}.*.vdix")
    ):
        if matches(candidate):
            return candidate
        if (
            owned_fallback is None
            and _sidecar_owner(source_path, candidate) is True
        ):
            owned_fallback = candidate
    if owned_fallback is not None:
        return owned_fallback
    if ambiguous_primary is not None:
        return ambiguous_primary
    return collision_path if primary.exists() else primary


def _collision_index_path(source: Path, directory: Path) -> Path:
    return directory / (
        f"{source.name}."
        f"{_source_name_digest(source, _identity(source.stat()))}.vdix"
    )


def _sidecar_owner(source: Path, index: Path) -> bool | None:
    """Return an ownership hint without weakening provenance validation."""

    try:
        locator = _read_document(index).get("source_locator")
    except (VdifIndexError, ValueError, UnicodeError):
        return None
    if (
        not isinstance(locator, dict)
        or not isinstance(locator.get("path"), str)
        or not Path(locator["path"]).is_absolute()
        or any(
            not isinstance(locator.get(key), int) or locator[key] < 0
            for key in ("device", "inode")
        )
    ):
        return None
    stat = source.stat()
    return locator["path"] == str(source) or (
        locator["device"],
        locator["inode"],
    ) == (stat.st_dev, stat.st_ino)


@lru_cache(maxsize=128)
def _source_name_digest(source: Path, identity: tuple[int, ...]) -> str:
    """Name collisions by source bytes and provenance, independent of path."""

    digest = hashlib.sha256(str(identity[2:]).encode("ascii"))
    with source.open("rb", buffering=0) as stream:
        digest.update(stream.read(4096))
        stream.seek(max(0, identity[2] - 4096))
        digest.update(stream.read(4096))
    if _identity(source.stat()) != identity:
        raise VdifIndexError("VDIF source changed during sidecar resolution")
    return digest.hexdigest()[:16]


def _sample_fingerprint(path: Path, frame_size_bytes: int) -> str:
    sample_bytes = min(frame_size_bytes, 4096)
    fd = os.open(path, os.O_RDONLY)
    try:
        first = os.pread(fd, sample_bytes, 0)
        last_offset = path.stat().st_size - frame_size_bytes
        last = os.pread(fd, sample_bytes, last_offset)
    finally:
        os.close(fd)
    if len(first) != sample_bytes or len(last) != sample_bytes:
        raise VdifIndexError(f"cannot fingerprint VDIF source {path}")
    return hashlib.sha256(first + last).hexdigest()


def _edge_fingerprint(path: Path, size: int) -> str:
    """Fingerprint mutable content even on coarse-timestamp filesystems."""

    sample_bytes = min(size, 4096)
    fd = os.open(path, os.O_RDONLY)
    try:
        first = os.pread(fd, sample_bytes, 0)
        last = os.pread(fd, sample_bytes, max(0, size - sample_bytes))
    finally:
        os.close(fd)
    if len(first) != sample_bytes or len(last) != sample_bytes:
        raise VdifIndexError(f"cannot fingerprint {path}")
    return hashlib.sha256(first + last).hexdigest()


def _header_fingerprint(path: Path, size: int) -> str:
    header_bytes = min(size, SIDECAR_HEADER_BYTES)
    fd = os.open(path, os.O_RDONLY)
    try:
        header = os.pread(fd, header_bytes, 0)
    finally:
        os.close(fd)
    if len(header) != header_bytes:
        raise VdifIndexError(f"cannot fingerprint VDIF sidecar {path}")
    return hashlib.sha256(header).hexdigest()


def _identity(stat: os.stat_result) -> tuple[int, ...]:
    return (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )


def _thread_metadata(
    document: dict[str, Any],
) -> tuple[VdifThreadMetadata, ...]:
    result = []
    for item in document.get("threads", []):
        result.append(
            VdifThreadMetadata(
                thread_id=int(item["thread_id"]),
                version=int(item["version"]),
                frame_size_bytes=int(item["frame_size_bytes"]),
                samples_per_frame=int(item["samples_per_frame"]),
                log2_channels=int(item["log2_channels"]),
                channel_count=int(item["channel_count"]),
                complex_data=bool(item["complex_data"]),
                bits_per_sample=int(item["bits_per_sample"]),
                station_id=int(item["station_id"]),
                station_code=(
                    None
                    if item.get("station_code") is None
                    else str(item["station_code"])
                ),
                extended_data_version=int(item["extended_data_version"]),
            )
        )
    if not result:
        raise VdifIndexError("binary VDIF sidecar has no thread metadata")
    return tuple(sorted(result, key=lambda value: value.thread_id))


def _read_document(
    index_path: Path, header_fingerprint: str | None = None
) -> dict[str, Any]:
    try:
        identity = _identity(index_path.stat())
    except OSError as exc:
        raise VdifIndexError(
            f"cannot read VDIF sidecar {index_path}"
        ) from exc
    # Cache encoded metadata, not the mutable parsed mapping returned to
    # callers. Resolution and validation share the same physical header read.
    fingerprint = header_fingerprint or _header_fingerprint(
        index_path, identity[2]
    )
    document = json.loads(
        _read_document_cached(index_path, identity, fingerprint)
    )
    if not isinstance(document, dict):
        raise VdifIndexError("invalid binary VDIF sidecar metadata")
    return document


@lru_cache(maxsize=128)
def _read_document_cached(
    index_path: Path, identity: tuple[int, ...], header_fingerprint: str
) -> str:
    try:
        with index_path.open("rb", buffering=0) as stream:
            prefix = stream.read(_HEADER_PREFIX.size)
            if len(prefix) != _HEADER_PREFIX.size:
                raise VdifIndexError("truncated binary VDIF sidecar header")
            magic, version, document_bytes = _HEADER_PREFIX.unpack(prefix)
            if (
                magic != SIDECAR_MAGIC
                or version not in _ENTRY_BYTES_BY_VERSION
            ):
                raise VdifIndexError("unsupported binary VDIF sidecar")
            if not 0 < document_bytes <= SIDECAR_HEADER_BYTES - len(prefix):
                raise VdifIndexError(
                    "invalid binary VDIF sidecar metadata size"
                )
            encoded = stream.read(document_bytes)
    except OSError as exc:
        message = f"cannot read VDIF sidecar {index_path}"
        raise VdifIndexError(message) from exc
    decoded = encoded.decode("utf-8")
    document = json.loads(decoded)
    if not isinstance(document, dict):
        raise VdifIndexError("invalid binary VDIF sidecar metadata")
    if document.get("schema_version") != version:
        raise VdifIndexError("binary VDIF sidecar version fields disagree")
    if _identity(index_path.stat()) != identity:
        raise VdifIndexError("VDIF sidecar changed while reading metadata")
    return decoded


@lru_cache(maxsize=128)
def _validate_cached(
    source_path: Path,
    index_path: Path,
    source_size: int,
    source_mtime_ns: int,
    index_size: int,
    index_mtime_ns: int,
    requested_frame_size: int | None,
    source_identity: tuple[int, ...],
    index_identity: tuple[int, ...],
    source_edge_fingerprint: str,
    index_header_fingerprint: str,
) -> VdifBinaryIndexSummary:
    del source_mtime_ns, index_mtime_ns
    document = _read_document(index_path, index_header_fingerprint)
    provenance = document.get("source_provenance")
    if provenance != {
        "version": 1,
        "size_bytes": source_size,
        "mtime_ns": source_identity[3],
        "ctime_ns": source_identity[4],
    }:
        raise VdifIndexError(
            "VDIF sidecar provenance is missing or stale; "
            "rebuild with --force"
        )
    frame_size = int(document["frame_size_bytes"])
    frame_count = int(document["frame_count"])
    entry_count = int(document["entry_count"])
    schema_version = int(document["schema_version"])
    physical_run_count = int(document.get("physical_timestamp_run_count", 0))
    if (
        requested_frame_size is not None
        and frame_size != requested_frame_size
    ):
        raise VdifIndexError(
            f"VDIF sidecar uses {frame_size}-byte frames; expected "
            f"{requested_frame_size}"
        )
    if frame_size <= 32 or source_size != frame_size * frame_count:
        raise VdifIndexError("binary VDIF sidecar source geometry is stale")
    if int(document["source_size_bytes"]) != source_size:
        raise VdifIndexError("binary VDIF sidecar source size is stale")
    expected_size = binary_index_size(
        entry_count,
        schema_version=schema_version,
        physical_timestamp_run_count=physical_run_count,
    )
    if index_size != expected_size or entry_count != frame_count:
        raise VdifIndexError("binary VDIF sidecar entry extent is invalid")
    if schema_version >= 4:
        if document.get("physical_timestamp_run_layout") != (
            "physical-start-u64,physical-stop-u64,packed-timestamp-start-u64"
        ):
            raise VdifIndexError(
                "binary VDIF sidecar physical timestamp layout is invalid"
            )
        expected_offset = physical_timestamp_run_offset(
            entry_count, schema_version=schema_version
        )
        if int(document.get("physical_timestamp_run_offset_bytes", -1)) != (
            expected_offset
        ):
            raise VdifIndexError(
                "binary VDIF sidecar physical timestamp offset is invalid"
            )
        runs = np.memmap(
            index_path,
            dtype="<u8",
            mode="r",
            offset=expected_offset,
            shape=(physical_run_count, 3),
        )
        if physical_run_count:
            starts = runs[:, 0]
            stops = runs[:, 1]
            packed = runs[:, 2]
            if (
                np.any(starts >= stops)
                or np.any(stops > frame_count)
                or np.any(starts[1:] < stops[:-1])
                or np.any(packed >= np.uint64(1 << 60))
                or np.any(
                    (packed & np.uint64(0x00FF_FFFF)) + (stops - starts)
                    > np.uint64(1 << 24)
                )
            ):
                raise VdifIndexError(
                    "binary VDIF sidecar physical timestamp runs are invalid"
                )
    elif physical_run_count:
        raise VdifIndexError(
            "legacy binary VDIF sidecar declares physical timestamp runs"
        )
    fingerprint = str(document["sample_fingerprint_sha256"])
    if _sample_fingerprint(source_path, frame_size) != fingerprint:
        raise VdifIndexError(
            "binary VDIF sidecar source fingerprint is stale"
        )
    threads = _thread_metadata(document)
    rejected_format_frames = int(
        document.get("rejected_format_frame_count", 0)
    )
    rejected_header_frames = int(
        document.get("rejected_header_frame_count", 0)
    )
    classifier = document.get("recorded_filler_classifier")
    if classifier is not None:
        classifier = str(classifier)
    if not 0 <= rejected_format_frames <= frame_count:
        raise VdifIndexError(
            "binary VDIF sidecar rejected-frame count is invalid"
        )
    if not 0 <= rejected_header_frames <= frame_count:
        raise VdifIndexError(
            "binary VDIF sidecar rejected-header count is invalid"
        )
    if rejected_format_frames or rejected_header_frames:
        raise VdifIndexError(
            "VDIF sidecar contains rejected malformed frames; "
            "rebuild with --force under strict header validation"
        )
    if any(item.frame_size_bytes != frame_size for item in threads):
        raise VdifIndexError("binary VDIF sidecar thread geometry is invalid")
    if (
        _identity(source_path.stat()) != source_identity
        or _identity(index_path.stat()) != index_identity
    ):
        raise VdifIndexError(
            "VDIF source or sidecar changed during validation"
        )
    return VdifBinaryIndexSummary(
        source_path=source_path,
        index_path=index_path,
        frame_size_bytes=frame_size,
        frame_count=frame_count,
        entry_count=entry_count,
        source_size_bytes=source_size,
        source_ordered=bool(document["source_ordered"]),
        sample_fingerprint_sha256=fingerprint,
        thread_metadata=threads,
        rejected_format_frame_count=rejected_format_frames,
        rejected_header_frame_count=rejected_header_frames,
        recorded_filler_classifier=classifier,
        physical_timestamp_run_count=physical_run_count,
        index_identity=index_identity,
        schema_version=schema_version,
    )


def validate_vdif_binary_index(
    source: str | os.PathLike[str],
    *,
    frame_size_bytes: int | None = None,
    index_dir: str | os.PathLike[str] | None = None,
) -> VdifBinaryIndexSummary:
    source_path = Path(source).expanduser().resolve(strict=True)
    index_path = resolve_vdif_binary_index_path(
        source_path, index_dir, frame_size_bytes=frame_size_bytes
    )
    return _validate_binary_index_path(
        source_path, index_path, frame_size_bytes=frame_size_bytes
    )


def _validate_binary_index_path(
    source_path: Path,
    index_path: Path,
    *,
    frame_size_bytes: int | None = None,
) -> VdifBinaryIndexSummary:
    """Validate an explicitly owned temporary or published sidecar."""

    try:
        source_stat = source_path.stat()
        index_stat = index_path.stat()
    except OSError as exc:
        raise VdifIndexError(
            f"binary VDIF sidecar does not exist: {index_path}; "
            "run the 'build-vdif-sidecar' command or use "
            "--assume-ordered only for independently verified ordered input"
        ) from exc
    retries = 2
    while True:
        try:
            return _validate_cached(
                source_path,
                index_path,
                source_stat.st_size,
                source_stat.st_mtime_ns,
                index_stat.st_size,
                index_stat.st_mtime_ns,
                frame_size_bytes,
                _identity(source_stat),
                _identity(index_stat),
                _edge_fingerprint(source_path, source_stat.st_size),
                _header_fingerprint(index_path, index_stat.st_size),
            )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            if retries:
                try:
                    source_unchanged = _identity(source_path.stat()) == (
                        _identity(source_stat)
                    )
                    current_index = index_path.stat()
                except OSError:
                    pass
                else:
                    if source_unchanged and _identity(current_index) != (
                        _identity(index_stat)
                    ):
                        # A builder may unlink its publication hardlink,
                        # changing ctime without changing content. Validate
                        # anew under the current identity; never ignore ctime
                        # or retry changes to the original source snapshot.
                        index_stat = current_index
                        retries -= 1
                        continue
            if isinstance(exc, VdifIndexError):
                raise
            raise VdifIndexError(
                f"invalid binary VDIF sidecar metadata: {index_path}"
            ) from exc


def _bitmap_frame_bounds(frames: np.ndarray) -> np.ndarray | None:
    """Return exact dense physical runs, or decline to the sorting path.

    Admission bounds extra NumPy workspace, excluding the input and final
    Python range records shared with the sorting path. Physical indices are
    masked to 53 bits by the caller, so offsets fit signed 64-bit indices.
    """

    count = frames.size
    if count < _BITMAP_MIN_ENTRIES:
        return None
    first, last = int(frames.min()), int(frames.max())
    span = last - first + 1
    # Allow two index temporaries plus the bitmap, transition mask and a
    # worst-case edge array. Use Python integers for admission arithmetic.
    estimated_bytes = 16 * count + 10 * span + 32
    if span > 2 * count or estimated_bytes >= _BITMAP_MAX_TEMP_BYTES:
        return None
    try:
        with nvtx_range("xbeam:vdif-sidecar-bitmap-ranges"):
            present = np.zeros(span + 2, dtype=np.bool_)
            present[frames - np.uint64(first) + np.uint64(1)] = True
            edges = np.flatnonzero(present[1:] != present[:-1])
            edges += first
            return edges.reshape(-1, 2)
    except MemoryError:
        # Returning releases this frame's temporaries before sorting starts.
        return None


def binary_index_reference_epochs(
    summary: VdifBinaryIndexSummary,
) -> tuple[int, ...]:
    """Find recorded epochs with bounded binary searches of sorted keys."""
    keys = _summary_entries(summary)[:, 0]
    epochs = []
    position = 0
    while position < len(keys):
        epoch = int(keys[position]) >> 54
        epochs.append(epoch)
        position = int(np.searchsorted(keys, np.uint64((epoch + 1) << 54)))
    return tuple(epochs)


def query_vdif_binary_index(
    source: str | os.PathLike[str],
    start_key: VdifTimestampKey,
    stop_key: VdifTimestampKey,
    *,
    thread_ids: Sequence[int] | None = None,
    frame_size_bytes: int | None = None,
    index_dir: str | os.PathLike[str] | None = None,
) -> tuple[VdifFrameRange, ...]:
    """Resolve an exact timestamp envelope without reading VDIF headers."""

    if start_key >= stop_key:
        raise ValueError("timestamp query requires start_key < stop_key")
    summary = validate_vdif_binary_index(
        source,
        frame_size_bytes=frame_size_bytes,
        index_dir=index_dir,
    )
    available_threads = {item.thread_id for item in summary.thread_metadata}
    if thread_ids is not None and not available_threads.intersection(
        map(int, thread_ids)
    ):
        return ()
    with nvtx_range("xbeam:vdif-binary-sidecar-query"):
        entries = _summary_entries(summary)
        keys = entries[:, 0]
        lower = int(
            np.searchsorted(keys, np.uint64(start_key.packed), side="left")
        )
        upper = int(
            np.searchsorted(keys, np.uint64(stop_key.packed), side="left")
        )
        if lower == upper:
            return ()
        locations = np.asarray(entries[lower:upper, 1], dtype=np.uint64)
        frames = locations & np.uint64(_FRAME_INDEX_MASK)
        if thread_ids is not None:
            wanted = np.asarray(tuple(map(int, thread_ids)), dtype=np.uint64)
            threads = locations >> np.uint64(54)
            selected = np.isin(threads, wanted)
            if not np.any(selected):
                return ()
            # Preserve the complete physical timestamp envelope. Extra thread
            # frames are filtered by the GPU mapper and make interleaved VDIF
            # one sequential read instead of many one-frame requests.
        if summary.source_ordered:
            # A globally ordered source has exactly the same timestamp-group
            # boundaries as its index. Entries within an equal-time group may
            # be permuted by older builders, so use insertion positions rather
            # than the first/last stored physical frame.
            return (
                VdifFrameRange(
                    lower,
                    upper,
                    summary.frame_size_bytes,
                ),
            )
        bounds = _bitmap_frame_bounds(frames)
        if bounds is not None:
            represented = int(np.sum(bounds[:, 1] - bounds[:, 0]))
            if represented != frames.size:
                raise VdifIndexError(
                    "binary VDIF sidecar contains duplicate physical "
                    "frame ordinals"
                )
            return tuple(
                VdifFrameRange(
                    int(start), int(stop), summary.frame_size_bytes
                )
                for start, stop in bounds
            )
        # Every physical source frame has exactly one sidecar entry.  Sorting
        # therefore provides the same ordered set as ``np.unique`` without
        # first building a multi-million-element hash table.  Keep the
        # invariant explicit so a corrupt index cannot silently collapse a
        # duplicated ordinal.
        ordered_frames = np.sort(frames)
        if np.any(ordered_frames[1:] == ordered_frames[:-1]):
            raise VdifIndexError(
                "binary VDIF sidecar contains duplicate physical frame "
                "ordinals"
            )
        boundaries = np.flatnonzero(np.diff(ordered_frames) != 1) + 1
        starts = np.concatenate((np.asarray([0], dtype=np.int64), boundaries))
        stops = np.concatenate(
            (boundaries, np.asarray([ordered_frames.size], dtype=np.int64))
        )
    return tuple(
        VdifFrameRange(
            int(ordered_frames[start]),
            int(ordered_frames[stop - 1]) + 1,
            summary.frame_size_bytes,
        )
        for start, stop in zip(starts, stops, strict=True)
    )


def read_vdif_binary_frames(
    source: str | os.PathLike[str],
    start_key: VdifTimestampKey,
    stop_key: VdifTimestampKey,
    *,
    frame_size_bytes: int | None = None,
    index_dir: str | os.PathLike[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return indexed timestamps, threads, physical frames, and validity.

    The returned arrays are small owned views of only the requested timestamp
    interval.  Correlation uses them to construct exact destination maps
    without rereading or copying VDIF headers from the payload files.
    """

    timestamps, threads, physical, states = read_vdif_binary_frame_states(
        source,
        start_key,
        stop_key,
        frame_size_bytes=frame_size_bytes,
        index_dir=index_dir,
    )
    return timestamps, threads, physical, states != VdifFrameState.VALID


def read_vdif_binary_physical_frame_states(
    source: str | os.PathLike[str],
    frame_ranges: Sequence[tuple[int, int]],
    *,
    frame_size_bytes: int | None = None,
    index_dir: str | os.PathLike[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return sidecar entries selected by bounded physical frame ranges.

    Ranges are half-open physical ordinals in the source file.  Selection is
    independent of the frames' native timestamps, which is required when a
    calibrated presentation maps an earlier or later physical frame into the
    requested destination window.  Overlapping ranges never duplicate an
    entry, and results are returned in physical source order.
    """

    summary = validate_vdif_binary_index(
        source,
        frame_size_bytes=frame_size_bytes,
        index_dir=index_dir,
    )
    source_frames, remainder = divmod(
        summary.source_size_bytes, summary.frame_size_bytes
    )
    if remainder:
        raise VdifIndexError("binary VDIF sidecar source has a partial frame")
    normalized: list[tuple[int, int]] = []
    for start, stop in frame_ranges:
        start = int(start)
        stop = int(stop)
        if start < 0 or start >= stop:
            raise ValueError(
                "physical frame ranges require 0 <= start < stop"
            )
        if stop > source_frames:
            raise VdifIndexError(
                "physical frame range exceeds the indexed source"
            )
        normalized.append((start, stop))
    if not normalized:
        empty_u64 = np.empty(0, dtype=np.uint64)
        return (
            empty_u64,
            np.empty(0, dtype=np.int16),
            empty_u64.copy(),
            np.empty(0, dtype=np.uint8),
        )

    merged: list[tuple[int, int]] = []
    for start, stop in sorted(normalized):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(stop, merged[-1][1]))
        else:
            merged.append((start, stop))
    starts = np.asarray([item[0] for item in merged], dtype=np.uint64)
    stops = np.asarray([item[1] for item in merged], dtype=np.uint64)

    # Timestamp-sorted sidecars are not necessarily in physical file order.
    # Prune their memory-mapped location column by cached physical extrema,
    # then apply a vectorized interval lookup inside each candidate block.
    # Temporary storage stays bounded even for arbitrarily reordered sources.
    entries = _summary_entries(summary)
    selected_parts: list[np.ndarray] = []
    with nvtx_range("xbeam:vdif-binary-sidecar-physical-query"):
        bounds = _physical_query_block_bounds(
            summary.index_path,
            summary.entry_count,
            summary.index_identity,
            summary.schema_version,
        )
        # A block may contain arbitrary timestamp/physical permutations.
        # Its extrema only prune blocks that cannot intersect any request;
        # the exact interval filter below still decides which entries count.
        candidate_blocks = np.zeros(len(bounds), dtype=np.bool_)
        for start, stop in merged:
            candidate_blocks |= (bounds[:, 0] < stop) & (
                bounds[:, 1] >= start
            )
        for block in np.flatnonzero(candidate_blocks):
            offset = int(block) * _PHYSICAL_QUERY_BLOCK_ENTRIES
            chunk = entries[offset : offset + _PHYSICAL_QUERY_BLOCK_ENTRIES]
            physical = chunk[:, 1] & np.uint64(_FRAME_INDEX_MASK)
            intervals = np.searchsorted(starts, physical, side="right") - 1
            candidates = intervals >= 0
            positions = np.flatnonzero(candidates)
            if positions.size:
                interval_positions = intervals[positions]
                positions = positions[
                    physical[positions] < stops[interval_positions]
                ]
            if positions.size:
                selected_parts.append(
                    np.asarray(chunk[positions], dtype=np.uint64)
                )
    selected = (
        np.concatenate(selected_parts)
        if selected_parts
        else np.empty((0, entries.shape[1]), dtype=np.uint64)
    )
    if selected.shape[0] > 1:
        physical = selected[:, 1] & np.uint64(_FRAME_INDEX_MASK)
        order = np.argsort(physical, kind="stable")
        physical = physical[order]
        if np.any(physical[1:] == physical[:-1]):
            raise VdifIndexError(
                "binary VDIF sidecar contains duplicate physical frame "
                "ordinals"
            )
        selected = selected[order]
    expected_count = sum(stop - start for start, stop in merged)
    if selected.shape[0] != expected_count:
        raise VdifIndexError(
            "binary VDIF sidecar does not contain each requested physical "
            "frame exactly once"
        )
    return _decode_binary_frame_entries(summary, selected)


@lru_cache(maxsize=64)
def _physical_query_block_bounds(
    index_path: Path,
    entry_count: int,
    identity: tuple[int, ...],
    schema_version: int,
) -> np.ndarray:
    """Cache tiny physical extrema per index block, keyed by file identity.

    This reads the location column once per sidecar instead of once per
    ingest window. No per-frame lookup array or new on-disk schema is needed.
    """

    entries = _mapped_entries(
        index_path, entry_count, identity, schema_version
    )
    block_count = (
        entry_count + _PHYSICAL_QUERY_BLOCK_ENTRIES - 1
    ) // _PHYSICAL_QUERY_BLOCK_ENTRIES
    bounds = np.empty((block_count, 2), dtype=np.uint64)
    for block in range(block_count):
        offset = block * _PHYSICAL_QUERY_BLOCK_ENTRIES
        physical = entries[
            offset : offset + _PHYSICAL_QUERY_BLOCK_ENTRIES, 1
        ] & np.uint64(_FRAME_INDEX_MASK)
        bounds[block] = (physical.min(), physical.max())
    bounds.setflags(write=False)
    return bounds


def read_vdif_binary_physical_frames(
    source: str | os.PathLike[str],
    frame_ranges: Sequence[tuple[int, int]],
    *,
    frame_size_bytes: int | None = None,
    index_dir: str | os.PathLike[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return physical-range sidecar entries and merged invalid flags."""

    timestamps, threads, physical, states = (
        read_vdif_binary_physical_frame_states(
            source,
            frame_ranges,
            frame_size_bytes=frame_size_bytes,
            index_dir=index_dir,
        )
    )
    return timestamps, threads, physical, states != VdifFrameState.VALID


def read_vdif_binary_frame_states(
    source: str | os.PathLike[str],
    start_key: VdifTimestampKey,
    stop_key: VdifTimestampKey,
    *,
    frame_size_bytes: int | None = None,
    index_dir: str | os.PathLike[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return timestamps, threads, physical ordinals, and uint8 state flags.

    v2 invalid entries have UNKNOWN_INVALID, never an inferred cause. Queries
    use stored timestamps even for rejected headers. Callers reconstructing
    trustworthy timelines must treat those entries as ambiguous.
    """
    if start_key >= stop_key:
        raise ValueError("timestamp query requires start_key < stop_key")
    summary = validate_vdif_binary_index(
        source,
        frame_size_bytes=frame_size_bytes,
        index_dir=index_dir,
    )
    entries = _summary_entries(summary)
    keys = entries[:, 0]
    lower = int(
        np.searchsorted(keys, np.uint64(start_key.packed), side="left")
    )
    upper = int(
        np.searchsorted(keys, np.uint64(stop_key.packed), side="left")
    )
    selected = np.asarray(entries[lower:upper], dtype=np.uint64)
    return _decode_binary_frame_entries(summary, selected)


def _decode_binary_frame_entries(
    summary: VdifBinaryIndexSummary,
    selected: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode selected binary entries while preserving stored frame state."""

    timestamps = selected[:, 0].copy()
    locations = selected[:, 1]
    invalid = (locations & np.uint64(_INVALID_FRAME_MASK)) != 0
    if summary.has_frame_states:
        raw_states = selected[:, 2]
        allowed = np.uint64(
            VdifFrameState.EXPLICIT_INVALID
            | VdifFrameState.HEADER_REJECTED
            | VdifFrameState.FORMAT_REJECTED
            | VdifFrameState.RECORDED_FILLER
        )
        rejection = raw_states & np.uint64(7)
        if np.any(raw_states & ~allowed) or np.any(
            (rejection != 0) != invalid
        ):
            raise VdifIndexError("invalid binary VDIF frame state")
        states = raw_states.astype(np.uint8, copy=True)
    else:
        states = invalid.astype(np.uint8) * int(
            VdifFrameState.UNKNOWN_INVALID
        )
    return (
        timestamps,
        (locations >> np.uint64(54)).astype(np.int16, copy=True),
        (locations & np.uint64(_FRAME_INDEX_MASK)).copy(),
        states,
    )


def query_vdif_recorded_filler_runs(
    summary: VdifBinaryIndexSummary,
) -> tuple[tuple[int, int], ...]:
    """Return exact-marker physical frame runs from a classified sidecar."""

    if summary.recorded_filler_classifier is None:
        raise VdifIndexError("binary VDIF filler classifier is missing")
    if not summary.has_frame_states:
        raise VdifIndexError("binary VDIF filler states are unavailable")
    entries = _summary_entries(summary)
    selected = (
        entries[:, 2] & np.uint64(VdifFrameState.RECORDED_FILLER)
    ) != 0
    physical = np.sort(
        np.asarray(
            entries[selected, 1] & np.uint64(_FRAME_INDEX_MASK),
            dtype=np.uint64,
        )
    )
    if physical.size == 0:
        return ()
    boundaries = np.flatnonzero(np.diff(physical) != 1) + 1
    starts = np.concatenate((np.asarray([0]), boundaries))
    stops = np.concatenate((boundaries, np.asarray([physical.size])))
    return tuple(
        (int(physical[start]), int(physical[stop - 1]) + 1)
        for start, stop in zip(starts, stops, strict=True)
    )


def _summary_entries(summary: VdifBinaryIndexSummary) -> np.memmap:
    # Check even on an LRU hit: a validated summary cannot authorize a
    # different inode published between validation and the query.
    if _identity(summary.index_path.stat()) != summary.index_identity:
        raise VdifIndexError("VDIF sidecar changed before mapping")
    return _mapped_entries(
        summary.index_path,
        summary.entry_count,
        summary.index_identity,
        summary.schema_version,
    )


def _summary_physical_timestamp_runs(
    summary: VdifBinaryIndexSummary,
) -> np.memmap:
    """Map physical-order runs from one validated schema-v4 sidecar."""

    if not summary.has_physical_timestamp_runs:
        raise VdifIndexError(
            "binary VDIF sidecar has no physical timestamp run table"
        )
    if _identity(summary.index_path.stat()) != summary.index_identity:
        raise VdifIndexError("VDIF sidecar changed before mapping")
    return _mapped_physical_timestamp_runs(
        summary.index_path,
        summary.entry_count,
        summary.physical_timestamp_run_count,
        summary.index_identity,
        summary.schema_version,
    )


@lru_cache(maxsize=64)
def _mapped_entries(
    index_path: Path,
    entry_count: int,
    identity: tuple[int, ...],
    schema_version: int,
) -> np.memmap:
    with index_path.open("rb") as stream:
        if _identity(os.fstat(stream.fileno())) != identity:
            raise VdifIndexError("VDIF sidecar changed before mapping")
        return np.memmap(
            stream,
            dtype="<u8",
            mode="r",
            offset=SIDECAR_HEADER_BYTES,
            shape=(entry_count, _ENTRY_BYTES_BY_VERSION[schema_version] // 8),
        )


@lru_cache(maxsize=64)
def _mapped_physical_timestamp_runs(
    index_path: Path,
    entry_count: int,
    run_count: int,
    identity: tuple[int, ...],
    schema_version: int,
) -> np.memmap:
    with index_path.open("rb") as stream:
        if _identity(os.fstat(stream.fileno())) != identity:
            raise VdifIndexError("VDIF sidecar changed before mapping")
        return np.memmap(
            stream,
            dtype="<u8",
            mode="r",
            offset=physical_timestamp_run_offset(
                entry_count, schema_version=schema_version
            ),
            shape=(run_count, 3),
        )


def job_has_binary_sidecars(
    job: Any, index_dir: str | os.PathLike[str] | None
) -> bool:
    """Return whether every source referenced by a job has a valid sidecar."""

    if not job.datastreams:
        return False
    try:
        for datastream in job.datastreams:
            for source in datastream.source_files:
                validate_vdif_binary_index(
                    source,
                    frame_size_bytes=datastream.frame_size_bytes,
                    index_dir=index_dir,
                )
    except (OSError, VdifIndexError):
        return False
    return True


__all__ = [
    "SIDECAR_ENTRY_BYTES",
    "SIDECAR_HEADER_BYTES",
    "SIDECAR_MAGIC",
    "SIDECAR_PHYSICAL_TIMESTAMP_RUN_BYTES",
    "SIDECAR_SCHEMA_VERSION",
    "VdifBinaryIndexSummary",
    "VdifFrameState",
    "binary_index_size",
    "encode_binary_index_header",
    "job_has_binary_sidecars",
    "physical_timestamp_run_offset",
    "query_vdif_binary_index",
    "query_vdif_recorded_filler_runs",
    "read_vdif_binary_frames",
    "read_vdif_binary_frame_states",
    "read_vdif_binary_physical_frames",
    "read_vdif_binary_physical_frame_states",
    "resolve_vdif_binary_index_path",
    "validate_vdif_binary_index",
]
