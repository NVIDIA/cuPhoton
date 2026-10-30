# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""GPU-first construction of compact binary VDIF frame sidecars."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

from cuphoton.xbeam.adapters.vdif.vdif import (
    HEADER_BYTES,
    VdifError,
    VdifHeader,
)
from cuphoton.xbeam.adapters.vdif.vdif_index import (
    VdifIndexError,
    VdifThreadMetadata,
)
from cuphoton.xbeam.adapters.vdif.vdif_sidecar import (
    SIDECAR_ENTRY_BYTES,
    SIDECAR_HEADER_BYTES,
    SIDECAR_SCHEMA_VERSION,
    VdifBinaryIndexSummary,
    VdifFrameState,
    _collision_index_path,
    _sample_fingerprint,
    _sidecar_owner,
    _validate_binary_index_path,
    binary_index_size,
    encode_binary_index_header,
    physical_timestamp_run_offset,
    resolve_vdif_binary_index_path,
    validate_vdif_binary_index,
)
from cuphoton.xbeam.cufile_io import CuFileSession
from cuphoton.xbeam.profiling import nvtx_range

_ALIGNMENT = 4096
_FRAME_INDEX_LIMIT = 1 << 53
_UNSET_U64 = np.iinfo(np.uint64).max
_UNSET_U32 = np.iinfo(np.uint32).max
_RECORDED_FILLER_CLASSIFIER = "exact-abaddeed-repeating-11223344-v2"

_EXTRACT_SOURCE = r"""
extern "C" __global__ void extract_vdif_sidecar_entries(
    const unsigned char* raw,
    const unsigned long long frame_base,
    const unsigned long long frame_count,
    const unsigned int frame_size,
    unsigned long long* entries,
    unsigned long long* formats,
    unsigned int* extended_versions,
    unsigned long long* error)
{
    const unsigned long long local =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= frame_count) return;
    const unsigned long long physical = frame_base + local;
    const unsigned int* words =
        (const unsigned int*)(raw + local * frame_size);
    const unsigned int word0 = words[0];
    const unsigned int word1 = words[1];
    const unsigned int word2 = words[2];
    const unsigned int word3 = words[3];
    const unsigned int thread_id = (word3 >> 16) & 0x3ffU;
    bool recorded_filler =
        word0 == 0x80000000U && word1 == 0U &&
        word2 == frame_size / 8U &&
        (word3 & 0xfc00ffffU) == 0U &&
        words[4] == 0xabaddeedU && words[5] == 0U &&
        words[6] == 0U && words[7] == 0U;
    if (recorded_filler) {
        const unsigned int* payload = words + 8;
        const unsigned int payload_words = (frame_size - 32U) / 4U;
        // Recorder boundary frames can retain two trailing status words.
        // All preceding payload words remain the filler signature.
        const unsigned int signature_words =
            payload_words > 2U ? payload_words - 2U : payload_words;
        for (unsigned int index = 0; index < signature_words; ++index) {
            if (payload[index] != 0x11223344U) {
                recorded_filler = false;
                break;
            }
        }
    }
    if (recorded_filler) {
        entries[physical * 3] = 0ULL;
        entries[physical * 3 + 1] =
            ((unsigned long long)thread_id << 54) |
            (1ULL << 53) | physical;
        entries[physical * 3 + 2] = 17ULL;
        return;
    }
    unsigned int code = 0;
    if ((word0 & 0x40000000U) != 0U || (word1 >> 30) != 0U) code = 1;
    else if (((word2 >> 29) & 7U) != 0U) code = 2;
    else if ((word2 & 0x00ffffffU) * 8U != frame_size) code = 3;
    const unsigned long long payload_bits =
        ((unsigned long long)frame_size - 32ULL) * 8ULL;
    const unsigned long long sample_bits =
        (1ULL << ((word2 >> 24) & 0x1fU)) *
        ((word3 >> 31) ? 2ULL : 1ULL) *
        (((word3 >> 26) & 0x1fU) + 1ULL);
    if (code == 0U && payload_bits % sample_bits != 0ULL) code = 6;
    if (code != 0U) {
        atomicMin(error, (physical << 8) | (unsigned long long)code);
        return;
    }
    const unsigned long long timestamp =
        ((unsigned long long)((word1 >> 24) & 0x3fU) << 54) |
        ((unsigned long long)(word0 & 0x3fffffffU) << 24) |
        (unsigned long long)(word1 & 0x00ffffffU);
    entries[physical * 3] = timestamp;
    entries[physical * 3 + 1] =
        ((unsigned long long)thread_id << 54) |
        ((unsigned long long)(word0 >> 31) << 53) | physical;
    entries[physical * 3 + 2] = (unsigned long long)(word0 >> 31);

    const unsigned long long format =
        ((unsigned long long)word3 << 32) | (unsigned long long)word2;
    const unsigned long long previous = atomicCAS(
        &formats[thread_id], 0xffffffffffffffffULL, format);
    if (previous != 0xffffffffffffffffULL && previous != format)
        atomicMin(error, (physical << 8) | 4ULL);
    const unsigned int extended = (words[4] >> 24) & 0xffU;
    const unsigned int previous_extended = atomicCAS(
        &extended_versions[thread_id], 0xffffffffU, extended);
    if (previous_extended != 0xffffffffU && previous_extended != extended)
        atomicMin(error, (physical << 8) | 5ULL);
}
"""

_ERRORS = {
    1: "legacy header or reserved timestamp bits",
    2: "unsupported header version",
    3: "frame length differs from the selected frame size",
    4: "thread format changes within the source",
    5: "thread extended-data version changes within the source",
    6: "invalid sample geometry",
}


def _align_up(value: int) -> int:
    return (value + _ALIGNMENT - 1) // _ALIGNMENT * _ALIGNMENT


def _aligned_device_bytes(xp: Any, size: int) -> tuple[Any, Any]:
    owner = xp.empty(size + _ALIGNMENT - 1, dtype=xp.uint8)
    offset = (-int(owner.data.ptr)) % _ALIGNMENT
    return owner, owner[offset : offset + size]


def _metadata_from_formats(
    formats: np.ndarray, extended_versions: np.ndarray
) -> tuple[VdifThreadMetadata, ...]:
    result = []
    for thread_id in np.flatnonzero(formats != _UNSET_U64):
        packed = int(formats[thread_id])
        word2 = packed & 0xFFFF_FFFF
        word3 = packed >> 32
        frame_size = (word2 & 0x00FF_FFFF) * 8
        log2_channels = (word2 >> 24) & 0x1F
        channel_count = 1 << log2_channels
        complex_data = bool(word3 >> 31)
        bits_per_sample = ((word3 >> 26) & 0x1F) + 1
        payload_bits = (frame_size - HEADER_BYTES) * 8
        sample_bits = (
            channel_count * (2 if complex_data else 1) * bits_per_sample
        )
        if sample_bits <= 0 or payload_bits % sample_bits:
            raise VdifIndexError(
                f"VDIF thread {thread_id} has invalid sample geometry"
            )
        station_id = word3 & 0xFFFF
        first, second = station_id >> 8, station_id & 0xFF
        try:
            station_code = (
                bytes((first, second)).decode("ascii")
                if first >= ord("0")
                else None
            )
        except UnicodeDecodeError:
            station_code = None
        result.append(
            VdifThreadMetadata(
                thread_id=int(thread_id),
                version=(word2 >> 29) & 0x7,
                frame_size_bytes=frame_size,
                samples_per_frame=payload_bits // sample_bits,
                log2_channels=log2_channels,
                channel_count=channel_count,
                complex_data=complex_data,
                bits_per_sample=bits_per_sample,
                station_id=station_id,
                station_code=station_code,
                extended_data_version=int(extended_versions[thread_id]),
            )
        )
    if not result:
        raise VdifIndexError("VDIF source contains no frames")
    return tuple(result)


def _physical_timestamp_runs(entries: Any, xp: Any) -> Any:
    """Compress trustworthy physical-order timestamps into within-second runs.

    Runs split at filler and VDIF-second boundaries, so construction requires
    no sample-rate assumption. Explicit-invalid frames retain their timestamp
    and remain part of the physical stream, matching correlation semantics.
    """

    states = entries[:, 2]
    filler_state = int(
        VdifFrameState.EXPLICIT_INVALID | VdifFrameState.RECORDED_FILLER
    )
    allowed = (
        (states == int(VdifFrameState.VALID))
        | (states == int(VdifFrameState.EXPLICIT_INVALID))
        | (states == filler_state)
    )
    if not bool(xp.all(allowed)):
        raise VdifIndexError(
            "cannot publish physical timestamp runs for ambiguous frame "
            "states"
        )
    usable = states != filler_state
    continuation = xp.zeros(entries.shape[0], dtype=xp.bool_)
    if entries.shape[0] > 1:
        timestamps = entries[:, 0]
        continuation[1:] = (
            usable[1:]
            & usable[:-1]
            & ((timestamps[1:] >> 24) == (timestamps[:-1] >> 24))
            & (timestamps[1:] == timestamps[:-1] + 1)
        )
    starts = xp.flatnonzero(usable & ~continuation).astype(xp.uint64)
    following = xp.zeros(entries.shape[0], dtype=xp.bool_)
    if entries.shape[0] > 1:
        following[:-1] = continuation[1:]
    stops = (xp.flatnonzero(usable & ~following) + 1).astype(xp.uint64)
    return xp.stack((starts, stops, entries[starts, 0]), axis=1)


def _document(
    source: Path,
    frame_size: int,
    frame_count: int,
    source_ordered: bool,
    threads: tuple[VdifThreadMetadata, ...],
    *,
    backend: str,
    rejected_format_frame_count: int,
    rejected_header_frame_count: int,
    physical_timestamp_run_count: int,
) -> dict[str, Any]:
    stat = source.stat()
    return {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "format": "vdix-binary",
        "entry_layout": (
            "timestamp-u64,thread-and-physical-frame-u64,frame-state-u64"
        ),
        "frame_size_bytes": frame_size,
        "frame_count": frame_count,
        "entry_count": frame_count,
        "source_size_bytes": source.stat().st_size,
        "source_provenance": {
            "version": 1,
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns,
        },
        "source_locator": {
            "path": str(source),
            "device": stat.st_dev,
            "inode": stat.st_ino,
        },
        "source_ordered": source_ordered,
        "sample_fingerprint_sha256": _sample_fingerprint(source, frame_size),
        "build_backend": backend,
        "rejected_format_frame_count": rejected_format_frame_count,
        "rejected_header_frame_count": rejected_header_frame_count,
        "recorded_filler_classifier": _RECORDED_FILLER_CLASSIFIER,
        "physical_timestamp_run_layout": (
            "physical-start-u64,physical-stop-u64,packed-timestamp-start-u64"
        ),
        "physical_timestamp_run_count": physical_timestamp_run_count,
        "physical_timestamp_run_offset_bytes": physical_timestamp_run_offset(
            frame_count
        ),
        "threads": [item.to_dict() for item in threads],
    }


def _frame_geometry(
    source: Path, frame_size_bytes: int | None
) -> tuple[int, int]:
    size = source.stat().st_size
    if frame_size_bytes is None:
        fd = os.open(source, os.O_RDONLY)
        try:
            header_bytes = os.pread(fd, HEADER_BYTES, 0)
        finally:
            os.close(fd)
        try:
            frame_size_bytes = VdifHeader.from_bytes(
                header_bytes, require_two_bit=False
            ).frame_length_bytes
        except VdifError as error:
            raise VdifIndexError(f"invalid VDIF frame 0: {error}") from error
    if (
        frame_size_bytes <= HEADER_BYTES
        or frame_size_bytes % 8
        or size % frame_size_bytes
    ):
        raise VdifIndexError(
            f"{source} is not an integer number of "
            f"{frame_size_bytes}-byte frames"
        )
    frame_count = size // frame_size_bytes
    if frame_count >= _FRAME_INDEX_LIMIT:
        raise VdifIndexError(
            "VDIF source exceeds the binary sidecar frame limit"
        )
    return frame_size_bytes, frame_count


def _write_gpu_index(
    source: Path,
    output: Path,
    frame_size: int,
    frame_count: int,
    *,
    chunk_bytes: int,
    device: int,
) -> None:
    import cupy as cp
    import cupyx

    with cp.cuda.Device(device):
        entry_bytes = frame_count * SIDECAR_ENTRY_BYTES
        padded_entry_bytes = (
            physical_timestamp_run_offset(frame_count) - SIDECAR_HEADER_BYTES
        )
        entry_owner, entry_storage = _aligned_device_bytes(
            cp, padded_entry_bytes
        )
        entries = (
            entry_storage[:entry_bytes]
            .view(cp.uint64)
            .reshape(frame_count, 3)
        )
        if padded_entry_bytes > entry_bytes:
            entry_storage[entry_bytes:].fill(0)
        formats = cp.full(1024, _UNSET_U64, dtype=cp.uint64)
        extended = cp.full(1024, _UNSET_U32, dtype=cp.uint32)
        error = cp.full(1, _UNSET_U64, dtype=cp.uint64)
        chunk_frames = min(frame_count, max(1, chunk_bytes // frame_size))
        raw_capacity = _align_up(
            chunk_frames * frame_size + 2 * (_ALIGNMENT - 1)
        )
        buffer_count = 2
        raw_pairs = tuple(
            _aligned_device_bytes(cp, raw_capacity)
            for _ in range(buffer_count)
        )
        raw_owners = tuple(item[0] for item in raw_pairs)
        raw_buffers = tuple(item[1] for item in raw_pairs)
        compute_stream = cp.cuda.Stream(non_blocking=True)
        initialized = cp.cuda.Event()
        initialized.record(cp.cuda.get_current_stream())
        compute_stream.wait_event(initialized)
        slot_events: list[Any | None] = [None] * buffer_count
        kernel = cp.RawKernel(_EXTRACT_SOURCE, "extract_vdif_sidecar_entries")
        # Slices do not own their CUDA allocations; retain both parents for
        # the complete read/extract/sort/write pipeline.
        allocation_owners = (entry_owner, raw_owners)
        with CuFileSession() as session:
            input_file = session.open(source)
            for raw in raw_buffers:
                session.register_buffer(int(raw.data.ptr), int(raw.nbytes))
            cursor = 0
            chunk_index = 0
            while cursor < frame_count:
                count = min(chunk_frames, frame_count - cursor)
                slot = chunk_index % len(raw_buffers)
                previous = slot_events[slot]
                if previous is not None:
                    previous.synchronize()
                raw = raw_buffers[slot]
                file_offset = cursor * frame_size
                aligned_offset = file_offset // _ALIGNMENT * _ALIGNMENT
                prefix = file_offset - aligned_offset
                required = prefix + count * frame_size
                request = _align_up(required)
                with nvtx_range("xbeam:vdif-sidecar-cufile-read"):
                    received = session.read(
                        input_file,
                        int(raw.data.ptr),
                        request,
                        aligned_offset,
                    )
                if received < required:
                    raise OSError(
                        f"short cuFile read at frame {cursor}: "
                        f"{received} < {required}"
                    )
                with compute_stream:
                    with nvtx_range("xbeam:vdif-sidecar-header-extract"):
                        kernel(
                            ((count + 255) // 256,),
                            (256,),
                            (
                                raw[prefix:].data.ptr,
                                np.uint64(cursor),
                                np.uint64(count),
                                np.uint32(frame_size),
                                entries.data.ptr,
                                formats.data.ptr,
                                extended.data.ptr,
                                error.data.ptr,
                            ),
                        )
                    completed = cp.cuda.Event()
                    completed.record(compute_stream)
                    slot_events[slot] = completed
                cursor += count
                chunk_index += 1
            ordered_device = None
            error_host = cupyx.empty_pinned(1, dtype=np.uint64)
            ordered_host = cupyx.empty_pinned(1, dtype=np.bool_)
            formats_host = cupyx.empty_pinned(1024, dtype=np.uint64)
            extended_host = cupyx.empty_pinned(1024, dtype=np.uint32)
            with compute_stream:
                ordered_device = cp.all(
                    entries[1:, 0] >= entries[:-1, 0]
                ).reshape(1)
                error.get(
                    out=error_host,
                    stream=compute_stream,
                    blocking=False,
                )
                ordered_device.get(
                    out=ordered_host,
                    stream=compute_stream,
                    blocking=False,
                )
                formats.get(
                    out=formats_host,
                    stream=compute_stream,
                    blocking=False,
                )
                extended.get(
                    out=extended_host,
                    stream=compute_stream,
                    blocking=False,
                )
            with nvtx_range("xbeam:vdif-sidecar-finalize-wait"):
                compute_stream.synchronize()
        failure = int(error_host[0])
        if failure != _UNSET_U64:
            raise VdifIndexError(
                f"invalid VDIF frame {failure >> 8}: "
                f"{_ERRORS.get(failure & 0xFF, 'unknown header error')}"
            )
        with compute_stream:
            physical_runs_device = _physical_timestamp_runs(entries, cp)
        compute_stream.synchronize()
        physical_runs = physical_runs_device.get().astype("<u8", copy=False)
        source_ordered = bool(ordered_host[0])
        if not source_ordered:
            with compute_stream:
                with nvtx_range("xbeam:vdif-sidecar-sort"):
                    # Timestamp is the only mmap search key. Equal-time
                    # entries may remain in physical order because word 1
                    # retains the thread identifier and physical frame.
                    order = cp.argsort(entries[:, 0])
                    entries[:] = entries[order]
            with nvtx_range("xbeam:vdif-sidecar-sort-wait"):
                compute_stream.synchronize()
        threads = _metadata_from_formats(formats_host, extended_host)
        # Only fully validated entries are published; an invalid-bit frame
        # must still have a supported header and a consistent thread format.
        # The final compact table is written directly from device memory.
        document = _document(
            source,
            frame_size,
            frame_count,
            source_ordered,
            threads,
            backend="gpu-cufile",
            rejected_format_frame_count=0,
            rejected_header_frame_count=0,
            physical_timestamp_run_count=len(physical_runs),
        )
        encoded_header = encode_binary_index_header(document)
        with CuFileSession() as session:
            output_file = session.create(
                output,
                binary_index_size(
                    frame_count,
                    physical_timestamp_run_count=len(physical_runs),
                ),
                header=encoded_header,
            )
            session.register_buffer(
                int(entry_storage.data.ptr), int(entry_storage.nbytes)
            )
            with nvtx_range("xbeam:vdif-sidecar-cufile-write"):
                written = session.write(
                    output_file,
                    int(entry_storage.data.ptr),
                    padded_entry_bytes,
                    SIDECAR_HEADER_BYTES,
                )
            if written != padded_entry_bytes:
                raise OSError(
                    f"short cuFile sidecar write: {written} < "
                    f"{padded_entry_bytes}"
                )
            os.fsync(output_file.fd)
        if physical_runs.size:
            fd = os.open(output, os.O_WRONLY)
            try:
                payload = physical_runs.tobytes(order="C")
                offset = physical_timestamp_run_offset(frame_count)
                if os.pwrite(fd, payload, offset) != len(payload):
                    raise OSError("short physical timestamp run write")
                os.fsync(fd)
            finally:
                os.close(fd)
        del allocation_owners


def _write_cpu_index(
    source: Path,
    output: Path,
    frame_size: int,
    frame_count: int,
    *,
    chunk_bytes: int,
) -> None:
    entries = np.empty((frame_count, 3), dtype="<u8")
    formats: dict[int, VdifThreadMetadata] = {}
    fd = os.open(source, os.O_RDONLY)
    try:
        chunk_frames = max(1, chunk_bytes // frame_size)
        cursor = 0
        while cursor < frame_count:
            count = min(chunk_frames, frame_count - cursor)
            raw = os.pread(fd, count * frame_size, cursor * frame_size)
            if len(raw) != count * frame_size:
                raise OSError(f"short VDIF read at frame {cursor}")
            rows = np.frombuffer(raw, dtype=np.uint8).reshape(
                count, frame_size
            )
            words = (
                rows[:, :HEADER_BYTES].copy().view("<u4").reshape(count, 8)
            )
            physical = cursor + np.arange(count, dtype=np.uint64)
            entries[cursor : cursor + count, 0] = (
                (((words[:, 1].astype(np.uint64) >> 24) & 0x3F) << 54)
                | ((words[:, 0].astype(np.uint64) & 0x3FFF_FFFF) << 24)
                | (words[:, 1].astype(np.uint64) & 0x00FF_FFFF)
            )
            thread_ids = (words[:, 3] >> 16) & 0x3FF
            entries[cursor : cursor + count, 1] = (
                (thread_ids.astype(np.uint64) << 54)
                | (((words[:, 0].astype(np.uint64) >> 31) & 1) << 53)
                | physical
            )
            entries[cursor : cursor + count, 2] = words[:, 0] >> 31
            for local_index, row in enumerate(words):
                physical_frame = cursor + local_index
                payload = rows[local_index, HEADER_BYTES:]
                if (
                    int(row[0]) == 0x8000_0000
                    and int(row[1]) == 0
                    and int(row[2]) == frame_size // 8
                    and int(row[3]) & 0xFC00_FFFF == 0
                    and tuple(map(int, row[4:])) == (0xABAD_DEED, 0, 0, 0)
                    and payload.size % 4 == 0
                    and np.all(
                        payload.copy().view("<u4")[
                            : max(0, payload.size // 4 - 2)
                            if payload.size // 4 > 2
                            else payload.size // 4
                        ]
                        == 0x1122_3344
                    )
                ):
                    entries[cursor + local_index, 1] = np.uint64(
                        (int(thread_ids[local_index]) << 54)
                        | (1 << 53)
                        | physical_frame
                    )
                    entries[cursor + local_index, 2] = np.uint64(
                        VdifFrameState.EXPLICIT_INVALID
                        | VdifFrameState.RECORDED_FILLER
                    )
                    continue
                try:
                    header = VdifHeader.from_bytes(
                        row.tobytes(), require_two_bit=False
                    )
                except VdifError as error:
                    raise VdifIndexError(
                        f"invalid VDIF frame {physical_frame}: {error}"
                    ) from error
                if header.frame_length_bytes != frame_size:
                    raise VdifIndexError(
                        f"invalid VDIF frame {physical_frame}: "
                        "frame length differs from the selected frame size"
                    )
                metadata = VdifThreadMetadata.from_header(header)
                previous = formats.setdefault(metadata.thread_id, metadata)
                if previous != metadata:
                    raise VdifIndexError(
                        f"invalid VDIF frame {physical_frame}: "
                        f"VDIF thread {metadata.thread_id} changes format"
                    )
            cursor += count
    finally:
        os.close(fd)
    physical_runs = _physical_timestamp_runs(entries, np)
    source_ordered = bool(np.all(entries[1:, 0] >= entries[:-1, 0]))
    # Keep equal-time entries in physical order, including interleaved
    # threads and invalid duplicates, to avoid a later physical reorder.
    if not source_ordered:
        entries[:] = entries[np.argsort(entries[:, 0], kind="stable")]
    document = _document(
        source,
        frame_size,
        frame_count,
        source_ordered,
        tuple(sorted(formats.values(), key=lambda value: value.thread_id)),
        backend="cpu-posix",
        rejected_format_frame_count=0,
        rejected_header_frame_count=0,
        physical_timestamp_run_count=len(physical_runs),
    )
    encoded_header = encode_binary_index_header(document)
    size = binary_index_size(
        frame_count,
        physical_timestamp_run_count=len(physical_runs),
    )
    with output.open("xb", buffering=0) as stream:
        stream.write(encoded_header)
        stream.write(entries.tobytes(order="C"))
        run_offset = physical_timestamp_run_offset(frame_count)
        stream.write(bytes(run_offset - stream.tell()))
        stream.write(
            physical_runs.astype("<u8", copy=False).tobytes(order="C")
        )
        stream.write(bytes(size - stream.tell()))
        os.fsync(stream.fileno())


def build_vdif_binary_index(
    source: str | os.PathLike[str],
    *,
    frame_size_bytes: int | None = None,
    index_dir: str | os.PathLike[str] | None = None,
    backend: str = "cufile",
    chunk_bytes: int = 256 * 1024 * 1024,
    device: int = 0,
    force: bool = False,
) -> VdifBinaryIndexSummary:
    """Build an exact-frame sidecar, using GPU Direct Storage by default."""

    if backend not in {"cufile", "posix"}:
        raise ValueError("sidecar backend must be 'cufile' or 'posix'")
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    selected = Path(source).expanduser().resolve(strict=True)
    before = selected.stat()
    output = resolve_vdif_binary_index_path(
        selected, index_dir, frame_size_bytes=frame_size_bytes
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if not force:
            return validate_vdif_binary_index(
                selected,
                frame_size_bytes=frame_size_bytes,
                index_dir=index_dir,
            )
    frame_size, frame_count = _frame_geometry(selected, frame_size_bytes)
    with TemporaryDirectory(
        prefix=f".{output.name}.", dir=output.parent
    ) as temporary:
        partial = Path(temporary) / "index.partial"
        if backend == "cufile":
            _write_gpu_index(
                selected,
                partial,
                frame_size,
                frame_count,
                chunk_bytes=chunk_bytes,
                device=device,
            )
        else:
            _write_cpu_index(
                selected,
                partial,
                frame_size,
                frame_count,
                chunk_bytes=chunk_bytes,
            )
        after = selected.stat()
        if (
            after.st_size,
            after.st_mtime_ns,
            after.st_dev,
            after.st_ino,
            after.st_ctime_ns,
        ) != (
            before.st_size,
            before.st_mtime_ns,
            before.st_dev,
            before.st_ino,
            before.st_ctime_ns,
        ):
            raise OSError(
                f"VDIF source changed while building its sidecar: {selected}"
            )
        _validate_binary_index_path(
            selected, partial, frame_size_bytes=frame_size
        )
        with partial.open("rb") as stream:
            os.fsync(stream.fileno())
        # First publication is create-if-absent even for force rebuilds.
        # Another source may have claimed the shared basename during I/O.
        try:
            os.link(partial, output)
        except FileExistsError:
            retry = resolve_vdif_binary_index_path(
                selected,
                index_dir,
                frame_size_bytes=frame_size,
            )
            if force:
                if (
                    index_dir is not None
                    and retry.name == f"{selected.name}.vdix"
                    and retry.exists()
                    and _sidecar_owner(selected, retry) is None
                ):
                    try:
                        _validate_binary_index_path(
                            selected, retry, frame_size_bytes=frame_size
                        )
                    except VdifIndexError:
                        # A stale legacy basename may belong to another
                        # source. Preserve it when ownership is unknown.
                        retry = _collision_index_path(selected, retry.parent)
                os.replace(partial, retry)
            elif retry.exists():
                _validate_binary_index_path(
                    selected,
                    retry,
                    frame_size_bytes=frame_size,
                )
            else:
                os.link(partial, retry)
            output = retry
        directory_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    summary = validate_vdif_binary_index(
        selected,
        frame_size_bytes=frame_size,
        index_dir=index_dir,
    )
    return summary


__all__ = ["build_vdif_binary_index"]
