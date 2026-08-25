# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Multi-file prefetch pipeline for `batch_to_device_stream`.

Architecture: a producer builds pinned-host file batches while the main
consumer stages the previous batch H→D and runs one batched nvCOMP inflate
before scattering each HDU into its output. When the optional native batcher
is available, a C++ worker pool reads and queues whole file batches so batch
N+1 can be built while batch N is decoding. Otherwise the original Python
per-file prefetcher is used as a fallback.

The streaming path handles uncompressed image HDUs and `CompImageHDU` with
`ZCMPTYPE in {GZIP_1, GZIP_2}`.
"""

from __future__ import annotations

import queue
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .gds import available_cpu_cores, configure_kvikio_parallelism

# Sentinel value pushed on the queue when the prefetcher has finished so the
# consumer knows to stop waiting.
_SENTINEL = object()


@contextmanager
def _nvtx_range(message: str):
    """Best-effort NVTX range without making nvtx a hard dependency."""
    try:
        import nvtx
    except Exception:
        yield
        return

    rng = nvtx.push_range(message, color="blue")
    try:
        yield
    finally:
        nvtx.pop_range(rng)


@dataclass
class _HduPlan:
    """Per-HDU planning data + a slice of the pinned host heap buffer."""

    hdu_index: int
    kind: str  # "comp" | "image"
    # For "comp": the full plan dict returned by
    # GpuCompImageReader.prepare_plan, plus `heap_rel_offsets` giving tile
    # offsets inside the file's host buffer.
    plan: dict | None
    heap_rel_offsets: np.ndarray | None  # shape (n_tiles,), int64
    heap_span_start: (
        int  # absolute file offset of the first byte the consumer needs
    )
    heap_span_len: int  # total bytes to stage to device
    # For "image" (uncompressed): shape/dtype and the host slice to stage.
    image_shape: tuple | None = None
    image_dtype_char: str | None = None
    image_itemsize: int = 0
    image_bzero: float = 0.0
    image_bscale: float = 1.0
    # For both kinds: where in the per-file pinned host buffer this HDU's
    # bytes live.
    file_host_offset: int = 0
    # Native ImageHDU path can read directly into the final output array.
    direct_output: bool = False


@dataclass
class _ReadSpan:
    """One absolute file read into a planned host-buffer offset."""

    file_offset: int
    nbytes: int
    host_offset: int = 0


@dataclass
class _PlannedFile:
    """Python FITS metadata plus native-reader spans for one file."""

    path: str
    file_index: int
    hdus: list[_HduPlan]
    spans: list[_ReadSpan]
    total_bytes: int


@dataclass
class PrefetchedFile:
    """One file fully staged into a pinned host buffer, ready to consume."""

    path: str
    file_index: int
    host_buf: object  # pinned uint8 host buffer or native device batch buffer
    hdus: list[_HduPlan]
    buffer_base_offset: int = 0
    hdu_header_sizes: dict[int, np.ndarray] | None = None


@dataclass
class _OutputSpec:
    shape: tuple
    dtype_char: str


def _probe_output_specs(
    first_path: str | Path,
    hdu_indices: Sequence[int],
    section=None,
) -> list[_OutputSpec]:
    """Read the first file's HDU metadata and derive stacked output specs."""
    specs: list[_OutputSpec] = []
    planned = _plan_file(first_path, 0, hdu_indices, section=section)
    for hdu in planned.hdus:
        if hdu.kind == "comp":
            specs.append(
                _OutputSpec(
                    shape=tuple(int(v) for v in hdu.plan["out_shape"]),
                    dtype_char=str(hdu.plan["dtype_char"]),
                )
            )
        elif hdu.kind == "image":
            if section is not None:
                raise NotImplementedError(
                    "section= is supported on compressed image HDUs only"
                )
            dtype_char = str(hdu.image_dtype_char)
            dtype = np.dtype(dtype_char)
            scaled = hdu.image_bzero != 0.0 or hdu.image_bscale != 1.0
            if scaled and dtype.kind in ("i", "u"):
                dtype_char = "f4" if hdu.image_itemsize <= 4 else "f8"
            specs.append(
                _OutputSpec(
                    shape=tuple(int(v) for v in hdu.image_shape),
                    dtype_char=dtype_char,
                )
            )
        else:
            raise AssertionError(f"unknown planned HDU kind {hdu.kind!r}")
    return specs


def _prepare_output_arrays(out, specs: Sequence[_OutputSpec], n_files: int):
    """Validate caller-provided output arrays or allocate new CuPy arrays."""
    import cupy as cp

    if out is None:
        return [
            cp.empty(
                (n_files,) + spec.shape,
                dtype=cp.dtype(spec.dtype_char),
            )
            for spec in specs
        ]

    if len(specs) == 1 and not isinstance(out, (tuple, list)):
        outs = [out]
    else:
        outs = list(out)
    if len(outs) != len(specs):
        raise ValueError(
            f"out must contain {len(specs)} array(s), got {len(outs)}"
        )

    for slot, (arr, spec) in enumerate(zip(outs, specs)):
        expected_shape = (n_files,) + spec.shape
        expected_dtype = cp.dtype(spec.dtype_char)
        if arr.shape != expected_shape or arr.dtype != expected_dtype:
            raise ValueError(
                f"out[{slot}] shape/dtype mismatch: expected "
                f"{expected_shape} {expected_dtype}, got "
                f"{arr.shape} {arr.dtype}"
            )
        if not arr.flags.c_contiguous:
            raise ValueError(f"out[{slot}] must be C-contiguous")
    return outs


def _compact_read_spans(
    abs_offsets: np.ndarray,
    lengths: np.ndarray,
) -> tuple[list[_ReadSpan], np.ndarray, int]:
    """Compact selected file ranges into a dense per-HDU buffer layout."""
    abs_offsets = np.asarray(abs_offsets, dtype=np.int64)
    lengths = np.asarray(lengths, dtype=np.int64)
    if abs_offsets.size != lengths.size:
        raise ValueError("abs_offsets and lengths must have the same size")
    if abs_offsets.size == 0:
        return [], np.zeros(0, dtype=np.int64), 0

    order = np.argsort(abs_offsets, kind="stable")
    rel_offsets = np.empty(abs_offsets.size, dtype=np.int64)
    spans: list[_ReadSpan] = []

    cursor = 0
    span_start = int(abs_offsets[order[0]])
    span_end = span_start + int(lengths[order[0]])
    span_host_offset = cursor
    rel_offsets[order[0]] = span_host_offset

    for tile_pos in order[1:]:
        off = int(abs_offsets[tile_pos])
        length = int(lengths[tile_pos])
        end = off + length
        if off > span_end:
            span_len = span_end - span_start
            spans.append(_ReadSpan(span_start, span_len, span_host_offset))
            cursor += span_len
            span_start = off
            span_end = end
            span_host_offset = cursor
        rel_offsets[tile_pos] = span_host_offset + (off - span_start)
        if end > span_end:
            span_end = end

    span_len = span_end - span_start
    spans.append(_ReadSpan(span_start, span_len, span_host_offset))
    cursor += span_len
    return spans, rel_offsets, cursor


@dataclass
class _CompBatchEntry:
    """Compressed HDU work item to include in one nvCOMP decode batch."""

    plan_item: _HduPlan
    file_host_buf: object
    out: object
    file_buffer_offset: int = 0
    header_sizes: np.ndarray | None = None


@dataclass
class _GpuBatchHandle:
    """Lifetime guard for one submitted GPU batch."""

    event: object | None
    keepalive: list
    stream: object
    device_id: int | None = None
    actively_awaited: bool = False


_UNRESOLVED_GPU_BATCHES: list[_GpuBatchHandle] = []
_UNRESOLVED_GPU_BATCHES_LOCK = threading.RLock()
# Deliberately process-wide: pinned owners can outlive any one CUDA stream or
# device. Clearing this state without confirmed completion would recreate the
# use-after-free it prevents, so an unrecoverable CUDA state requires process
# restart rather than an unsafe reset hook.


@contextmanager
def _gpu_batch_device(handle: _GpuBatchHandle):
    """Restore the CUDA device that owns one batch's completion state."""
    if handle.device_id is None:
        yield
        return

    import cupy as cp

    with cp.cuda.Device(handle.device_id):
        yield


def _quarantine_gpu_batches(
    handles: Sequence[_GpuBatchHandle],
    *,
    actively_awaited: bool = False,
) -> None:
    """Durably retain batches before or after uncertain completion."""
    with _UNRESOLVED_GPU_BATCHES_LOCK:
        known = {id(handle) for handle in _UNRESOLVED_GPU_BATCHES}
        new_handles = [
            handle for handle in handles if id(handle) not in known
        ]
        for handle in handles:
            if actively_awaited or id(handle) not in known:
                handle.actively_awaited = actively_awaited
        _UNRESOLVED_GPU_BATCHES.extend(new_handles)


def _mark_gpu_batches_unresolved(
    handles: Sequence[_GpuBatchHandle],
) -> None:
    """Make retained batches block later submissions until completion."""
    with _UNRESOLVED_GPU_BATCHES_LOCK:
        for handle in handles:
            handle.actively_awaited = False


def _release_quarantined_gpu_batches(
    handles: Sequence[_GpuBatchHandle],
) -> None:
    """Release selected safety-quarantined batches by identity."""
    released = {id(handle) for handle in handles}
    with _UNRESOLVED_GPU_BATCHES_LOCK:
        for handle in handles:
            handle.actively_awaited = False
        _UNRESOLVED_GPU_BATCHES[:] = [
            handle
            for handle in _UNRESOLVED_GPU_BATCHES
            if id(handle) not in released
        ]


def _gpu_batch_completion_is_confirmed(handle: _GpuBatchHandle) -> bool:
    """Return whether a quarantined batch is now known to be complete."""
    try:
        with _gpu_batch_device(handle):
            if handle.event is not None:
                return bool(handle.event.done)
            return bool(handle.stream.done)
    except Exception:
        return False


def _ensure_gpu_submissions_safe() -> None:
    """Fail closed while an earlier batch may still use retained owners."""
    with _UNRESOLVED_GPU_BATCHES_LOCK:
        retained = []
        for handle in _UNRESOLVED_GPU_BATCHES:
            if _gpu_batch_completion_is_confirmed(handle):
                handle.actively_awaited = False
            else:
                retained.append(handle)
        _UNRESOLVED_GPU_BATCHES[:] = retained
        unresolved = sum(
            not handle.actively_awaited for handle in _UNRESOLVED_GPU_BATCHES
        )
    if unresolved:
        raise RuntimeError(
            "previous XDR GPU work could not be synchronized; refusing "
            "another submission while its source, scratch, and output "
            "buffers remain quarantined; "
            "restart the process if the CUDA context cannot recover"
        )


@contextmanager
def _gpu_submission_guard():
    """Atomically gate and publish one bounded GPU submission group."""
    with _UNRESOLVED_GPU_BATCHES_LOCK:
        _ensure_gpu_submissions_safe()
        yield


def _synchronize_gpu_batch(
    handle: _GpuBatchHandle,
    *,
    synchronize_stream: bool = False,
) -> None:
    """Synchronize one batch without exposing its owners to interruption."""
    completion_confirmed = False
    try:
        _quarantine_gpu_batches([handle], actively_awaited=True)
        target = handle.stream if synchronize_stream else handle.event
        if target is None:
            raise RuntimeError("GPU batch has no completion event")
        with _gpu_batch_device(handle):
            target.synchronize()
        completion_confirmed = True
    finally:
        if completion_confirmed:
            _release_quarantined_gpu_batches([handle])
        else:
            _quarantine_gpu_batches([handle])
            _mark_gpu_batches_unresolved([handle])


def _plan_file(
    path: str | Path,
    file_index: int,
    hdu_indices: Sequence[int],
    *,
    section=None,
) -> _PlannedFile:
    """Open a FITS file and compute byte spans for the stream path."""
    path_s = str(path)
    from .planning import plan_native_files

    native_items = plan_native_files(
        [path_s],
        [int(file_index)],
        hdu_indices,
        1,
        section,
    )
    if len(native_items) != 1:
        raise RuntimeError(
            "native planner returned "
            f"{len(native_items)} files for one requested file"
        )
    return _planned_file_from_native_tuple(native_items[0])


def _parse_hdu_header_sizes(
    planned: _PlannedFile,
    host_buf: np.ndarray,
) -> dict[int, np.ndarray]:
    """Parse gzip headers from a zero-based, file-local staged buffer.

    Native pooled buffers can have nonzero per-file base offsets and use the
    device header probe instead.
    """
    from .nvcomp_batch import (
        _MAX_GZIP_HEADER_BYTES,
        _parse_gzip_header_len,
        _TruncatedGzipHeader,
        _validate_tile_ranges,
    )

    host_bytes = memoryview(host_buf)
    slots: dict[int, np.ndarray] = {}
    for slot, hdu in enumerate(planned.hdus):
        if hdu.kind != "comp":
            continue

        hdu_base = int(hdu.file_host_offset)
        hdu_nbytes = int(hdu.heap_span_len)
        hdu_end = hdu_base + hdu_nbytes
        if hdu_base < 0 or hdu_nbytes < 0 or hdu_end > int(host_buf.size):
            raise ValueError("compressed HDU exceeds staged host buffer")

        rel_offsets, lengths = _validate_tile_ranges(
            hdu.heap_rel_offsets,
            hdu.plan["sel_lengths"],
            hdu_nbytes,
            minimum_length=20,
        )
        tile_starts = hdu_base + rel_offsets

        sizes = np.empty(rel_offsets.size, dtype=np.int64)
        minimal_headers = (
            (host_buf[tile_starts] == 0x1F)
            & (host_buf[tile_starts + 1] == 0x8B)
            & (host_buf[tile_starts + 2] == 0x08)
            & (host_buf[tile_starts + 3] == 0)
        )
        sizes[minimal_headers] = 10
        for index in np.flatnonzero(~minimal_headers):
            # Preserve the two-byte minimum DEFLATE stream and trailer.
            # A memoryview avoids copying payloads with long optional fields.
            tile_header_limit = int(lengths[index]) - 10
            probe_length = min(tile_header_limit, _MAX_GZIP_HEADER_BYTES)
            header_start = int(tile_starts[index])
            header_bound = header_start + probe_length
            try:
                sizes[index] = _parse_gzip_header_len(
                    host_bytes[header_start:header_bound]
                )
            except _TruncatedGzipHeader:
                if tile_header_limit > probe_length:
                    raise ValueError(
                        "gzip header exceeds "
                        f"{_MAX_GZIP_HEADER_BYTES}-byte safety limit"
                    ) from None
                raise
        slots[slot] = sizes
    return slots


class _FilePrefetcher(threading.Thread):
    """Read each file's required bytes into a pinned host buffer.

    Daemon thread — exits with the main process if something hangs.
    """

    def __init__(
        self,
        paths: Sequence[Path],
        hdu_indices: Sequence[int],
        out_queue: queue.Queue,
        *,
        section=None,
    ):
        super().__init__(daemon=True, name="xdr-prefetcher")
        self.paths = [str(p) for p in paths]
        self.hdu_indices = tuple(hdu_indices)
        self.queue = out_queue
        self.section = section
        self.error: BaseException | None = None
        self._stop_event = threading.Event()

    def request_stop(self) -> None:
        """Ask the prefetcher to exit at the next opportunity. Called by the
        consumer when it raises so we don't deadlock on `queue.put`.
        """
        self._stop_event.set()

    # ---- internal planning ------------------------------------------------

    def _plan_file(self, path: str, file_index: int) -> PrefetchedFile:
        """Open the FITS file, plan each requested HDU, and issue I/O into
        a pinned host buffer. Runs entirely on this thread."""
        planned = _plan_file(
            path,
            file_index,
            self.hdu_indices,
            section=self.section,
        )

        import cupy as cp

        mem = cp.cuda.alloc_pinned_memory(planned.total_bytes)
        host_buf = np.frombuffer(
            mem, dtype=np.uint8, count=planned.total_bytes
        )

        # Issue all planned spans before waiting. KvikIO partitions large
        # spans internally and can execute independent spans in the same
        # thread pool.
        import kvikio

        configure_kvikio_parallelism()
        with kvikio.CuFile(planned.path, "r") as f:
            futures = []
            for span in planned.spans:
                view = host_buf[
                    span.host_offset : span.host_offset + span.nbytes
                ]
                futures.append(
                    f.pread(
                        view,
                        size=span.nbytes,
                        file_offset=span.file_offset,
                    )
                )
            # Block until all reads for this file complete. This is where
            # kvikio releases the GIL and the consumer thread makes progress.
            for fut in futures:
                fut.get()

        hdu_header_sizes = _parse_hdu_header_sizes(planned, host_buf)

        return PrefetchedFile(
            path=planned.path,
            file_index=planned.file_index,
            host_buf=host_buf,
            hdus=planned.hdus,
            hdu_header_sizes=hdu_header_sizes,
        )

    def run(self) -> None:
        try:
            for i, path in enumerate(self.paths):
                if self._stop_event.is_set():
                    break
                item = self._plan_file(path, i)
                # Use timed puts so an abandoned consumer doesn't hang us.
                while not self._stop_event.is_set():
                    try:
                        self.queue.put(item, timeout=0.25)
                        break
                    except queue.Full:
                        continue
        except BaseException as e:
            self.error = e
        finally:
            # Retry until the consumer makes room or asks us to stop.
            while not self._stop_event.is_set():
                try:
                    self.queue.put(_SENTINEL, timeout=0.25)
                    break
                except queue.Full:
                    continue


def _consume_comp_batch(
    entries: list[_CompBatchEntry], stream, keepalive=None
):
    """Decode many compressed HDUs with one batched nvCOMP call.

    Each entry can come from a different FITS file and/or HDU. The compressed
    byte spans are concatenated into one device buffer, all tile metadata is
    concatenated, nvCOMP inflates the entire group in one call, and then each
    HDU's decompressed tile range is postprocessed/scattered into its own
    output array.
    """
    if not entries:
        return

    import cupy as cp

    from .reader import GpuCompImageReader

    device_backed = all(
        isinstance(e.file_host_buf, cp.ndarray) for e in entries
    )
    shared_device_buf = entries[0].file_host_buf if device_backed else None
    if device_backed and any(
        e.file_host_buf is not shared_device_buf for e in entries
    ):
        device_backed = False

    if device_backed:
        d_concat = shared_device_buf
    else:
        total_comp_bytes = sum(
            int(e.plan_item.heap_span_len) for e in entries
        )
        d_concat = cp.empty(total_comp_bytes, dtype=cp.uint8)
    if keepalive is not None:
        keepalive.append(d_concat)

    rel_offsets_parts = []
    lengths_parts = []
    out_bytes_parts = []
    postprocess_ranges = []
    have_all_header_sizes = all(
        entry.header_sizes is not None for entry in entries
    )
    header_sizes_parts = [] if have_all_header_sizes else None

    comp_cursor = 0
    tile_cursor = 0
    decoded_cursor = 0

    with stream or cp.cuda.Stream.null:
        for entry in entries:
            plan_item = entry.plan_item
            plan = plan_item.plan
            buf_start = int(entry.file_buffer_offset) + int(
                plan_item.file_host_offset
            )
            if device_backed:
                comp_base = buf_start
            else:
                host_slice = entry.file_host_buf[
                    buf_start : buf_start + plan_item.heap_span_len
                ]
                target = d_concat[
                    comp_cursor : comp_cursor + plan_item.heap_span_len
                ]
                if isinstance(host_slice, cp.ndarray):
                    target[...] = host_slice
                else:
                    target.set(host_slice)
                comp_base = comp_cursor

            tile_count = int(plan["sel_lengths"].size)
            decoded_nbytes = int(np.asarray(plan["out_bytes"]).sum())

            rel_offsets_parts.append(plan_item.heap_rel_offsets + comp_base)
            lengths_parts.append(plan["sel_lengths"])
            out_bytes_parts.append(plan["out_bytes"])
            if header_sizes_parts is not None:
                header_sizes_parts.append(entry.header_sizes)
            postprocess_ranges.append(
                (
                    entry,
                    tile_cursor,
                    tile_cursor + tile_count,
                    decoded_cursor,
                    decoded_cursor + decoded_nbytes,
                )
            )

            if not device_backed:
                comp_cursor += int(plan_item.heap_span_len)
            tile_cursor += tile_count
            decoded_cursor += decoded_nbytes

        rel_offsets = np.concatenate(rel_offsets_parts)
        lengths = np.concatenate(lengths_parts)
        out_bytes = np.concatenate(out_bytes_parts)
        header_sizes = (
            np.concatenate(header_sizes_parts)
            if header_sizes_parts is not None
            else None
        )

        d_pixels, tile_byte_offsets_np = (
            GpuCompImageReader.inflate_device_heap(
                d_concat,
                rel_offsets,
                lengths=lengths,
                out_bytes=out_bytes,
                stream=stream,
                keepalive=keepalive,
                header_sizes=header_sizes,
            )
        )

        for (
            entry,
            tile_start,
            tile_end,
            decoded_start,
            decoded_end,
        ) in postprocess_ranges:
            local_tile_offsets = (
                tile_byte_offsets_np[tile_start:tile_end] - decoded_start
            )
            GpuCompImageReader.postprocess_decoded_tiles(
                d_pixels[decoded_start:decoded_end],
                local_tile_offsets,
                entry.plan_item.plan,
                out=entry.out,
                stream=stream,
                keepalive=keepalive,
            )


def _consume_image(
    plan_item: _HduPlan,
    file_host_buf,
    out,
    stream,
    keepalive=None,
    file_buffer_offset: int = 0,
):
    """Stage uncompressed pixel bytes to device, then byteswap + scale."""
    import cupy as cp

    from .kernels import apply_bzero_bscale, byteswap_inplace

    native_dtype = cp.dtype(plan_item.image_dtype_char)
    requested_out = out
    if out is None or out.dtype != native_dtype:
        out = cp.empty(plan_item.image_shape, dtype=native_dtype)
        if keepalive is not None:
            keepalive.append(out)
    with stream or cp.cuda.Stream.null:
        if not plan_item.direct_output:
            start = int(file_buffer_offset) + int(plan_item.file_host_offset)
            host_slice = file_host_buf[
                start : start + plan_item.heap_span_len
            ]
            out_bytes = out.view(cp.uint8).ravel()
            if isinstance(host_slice, cp.ndarray):
                out_bytes[...] = host_slice
            else:
                out_bytes.set(host_slice)
        if plan_item.image_itemsize > 1:
            byteswap_inplace(out, plan_item.image_itemsize)
        if plan_item.image_bzero != 0.0 or plan_item.image_bscale != 1.0:
            if out.dtype.kind in ("i", "u"):
                out = out.astype(
                    cp.float32
                    if plan_item.image_itemsize <= 4
                    else cp.float64
                )
                if keepalive is not None:
                    keepalive.append(out)
            apply_bzero_bscale(
                out, plan_item.image_bzero, plan_item.image_bscale
            )
        if requested_out is not None and out is not requested_out:
            requested_out[...] = out
            out = requested_out
    return out


def _consume_prefetched_group(
    items: list[PrefetchedFile], outs: list, stream, keepalive=None
):
    """Consume prefetched files, batching compressed HDUs together."""
    comp_entries: list[_CompBatchEntry] = []

    for item in items:
        for slot, plan_item in enumerate(item.hdus):
            target = outs[slot][item.file_index]
            if plan_item.kind == "comp":
                comp_entries.append(
                    _CompBatchEntry(
                        plan_item=plan_item,
                        file_host_buf=item.host_buf,
                        out=target,
                        file_buffer_offset=item.buffer_base_offset,
                        header_sizes=(item.hdu_header_sizes or {}).get(slot),
                    )
                )
            elif plan_item.kind == "image":
                result = _consume_image(
                    plan_item,
                    item.host_buf,
                    target,
                    stream,
                    keepalive=keepalive,
                    file_buffer_offset=item.buffer_base_offset,
                )
                if result is not target:
                    target[...] = result
            else:
                raise AssertionError(f"unknown kind {plan_item.kind!r}")

    _consume_comp_batch(comp_entries, stream, keepalive=keepalive)


def _native_file_plans(
    planned_files: Sequence[_PlannedFile],
) -> list[tuple]:
    """Convert staged Python file plans to compact C++ tuples."""
    return [
        (
            item.path,
            item.file_index,
            item.total_bytes,
            [
                (span.file_offset, span.nbytes, span.host_offset)
                for span in item.spans
            ],
        )
        for item in planned_files
    ]


def _native_file_plans_with_direct_outputs(
    planned_files: Sequence[_PlannedFile],
    outs: Sequence,
) -> tuple[list[_PlannedFile], list[tuple]]:
    """Build native file plans with direct ImageHDU output reads.

    Compressed HDU spans are compacted into the native scratch buffer.
    ImageHDU spans are submitted as direct GDS reads into the caller-visible
    output slice.
    """
    converted: list[_PlannedFile] = []
    native_plans: list[tuple] = []

    for item in planned_files:
        staged_hdus: list[_HduPlan] = []
        staged_spans: list[_ReadSpan] = []
        direct_spans: list[tuple[int, int, int]] = []
        staged_cursor = 0

        for slot, hdu in enumerate(item.hdus):
            if hdu.kind == "image":
                target = outs[slot][item.file_index]
                if target.dtype == np.dtype(hdu.image_dtype_char):
                    if not target.flags.c_contiguous:
                        raise ValueError(
                            "native direct ImageHDU output slice must be "
                            "C-contiguous"
                        )
                    direct_spans.append(
                        (
                            int(hdu.heap_span_start),
                            int(hdu.heap_span_len),
                            int(target.data.ptr),
                        )
                    )
                    staged_hdus.append(
                        replace(hdu, file_host_offset=0, direct_output=True)
                    )
                    continue

            hdu_start = int(hdu.file_host_offset)
            hdu_end = hdu_start + int(hdu.heap_span_len)
            new_base = staged_cursor
            for span in item.spans:
                span_start = int(span.host_offset)
                if hdu_start <= span_start < hdu_end:
                    staged_spans.append(
                        _ReadSpan(
                            int(span.file_offset),
                            int(span.nbytes),
                            new_base + (span_start - hdu_start),
                        )
                    )
            staged_hdus.append(
                replace(
                    hdu,
                    file_host_offset=new_base,
                    direct_output=False,
                )
            )
            staged_cursor += int(hdu.heap_span_len)

        staged_file = replace(
            item,
            hdus=staged_hdus,
            spans=staged_spans,
            total_bytes=staged_cursor,
        )
        converted.append(staged_file)
        native_plans.append(
            (
                staged_file.path,
                staged_file.file_index,
                staged_file.total_bytes,
                [
                    (span.file_offset, span.nbytes, span.host_offset)
                    for span in staged_file.spans
                ],
                direct_spans,
            )
        )

    return converted, native_plans


def _planned_file_from_native_tuple(item) -> _PlannedFile:
    """Convert native planner tuples to local dataclasses."""
    path, file_index, native_hdus, native_spans, total_bytes = item
    spans = [
        _ReadSpan(int(file_offset), int(nbytes), int(host_offset))
        for file_offset, nbytes, host_offset in native_spans
    ]

    hdus: list[_HduPlan] = []
    for hdu_item in native_hdus:
        (
            hdu_index,
            kind,
            plan,
            heap_rel_offsets,
            heap_span_start,
            heap_span_len,
            image_shape,
            image_dtype_char,
            image_itemsize,
            image_bzero,
            image_bscale,
            file_host_offset,
        ) = hdu_item

        if plan is not None:
            plan = dict(plan)
            for key, dtype in (
                ("tile_idx", np.int64),
                ("sel_abs_offsets", np.int64),
                ("sel_lengths", np.int64),
                ("origins_r", np.int32),
                ("origins_c", np.int32),
                ("heights", np.int32),
                ("widths", np.int32),
                ("out_bytes", np.int64),
                ("src_off_r", np.int32),
                ("src_off_c", np.int32),
                ("tile_full_w", np.int32),
            ):
                plan[key] = np.asarray(plan[key], dtype=dtype)
            if plan.get("quantized", False):
                plan["sel_zscale"] = np.asarray(
                    plan["sel_zscale"], dtype=np.float64
                )
                plan["sel_zzero"] = np.asarray(
                    plan["sel_zzero"], dtype=np.float64
                )
            heap_rel = np.asarray(heap_rel_offsets, dtype=np.int64)
            image_shape_tuple = None
            image_dtype = None
        else:
            heap_rel = None
            image_shape_tuple = tuple(int(v) for v in image_shape)
            image_dtype = str(image_dtype_char)

        hdus.append(
            _HduPlan(
                hdu_index=int(hdu_index),
                kind=str(kind),
                plan=plan,
                heap_rel_offsets=heap_rel,
                heap_span_start=int(heap_span_start),
                heap_span_len=int(heap_span_len),
                image_shape=image_shape_tuple,
                image_dtype_char=image_dtype,
                image_itemsize=int(image_itemsize),
                image_bzero=float(image_bzero),
                image_bscale=float(image_bscale),
                file_host_offset=int(file_host_offset),
            )
        )

    return _PlannedFile(
        path=str(path),
        file_index=int(file_index),
        hdus=hdus,
        spans=spans,
        total_bytes=int(total_bytes),
    )


def _native_batch_builder_class(native_batcher):
    """Resolve the optional native batcher class according to API policy."""
    if native_batcher is False:
        return None
    if native_batcher is not True and native_batcher != "auto":
        raise ValueError("native_batcher must be 'auto', True, or False")

    from .mock_storage import storage_cache

    if storage_cache.active():
        if native_batcher is True:
            raise RuntimeError(
                "native_batcher=True is not supported while mock_storage "
                "is active; "
                "use native_batcher=False for mock-storage benchmarks."
            )
        return None

    from .nvcomp_batch import get_native_batch_builder

    return get_native_batch_builder(required=(native_batcher is True))


def _native_batch_components(native_batcher):
    NativeBatchBuilder = _native_batch_builder_class(native_batcher)
    if NativeBatchBuilder is None:
        return None, None

    from .planning import plan_native_files

    return (
        NativeBatchBuilder,
        plan_native_files,
    )


def _submit_prefetched_group(
    items: list[PrefetchedFile],
    outs: list,
    *,
    stream,
    owner,
    in_flight: deque[_GpuBatchHandle],
) -> None:
    """Queue GPU work and register its lifetime handle."""
    import cupy as cp

    use_stream = stream or cp.cuda.Stream.null
    device_id = int(cp.cuda.Device().id)
    # The output list must survive an exceptional return too: outstanding
    # scatter/image kernels can still be writing caller-unreturned arrays.
    keepalive = [owner, outs]
    guard = _GpuBatchHandle(
        event=None,
        keepalive=keepalive,
        stream=use_stream,
        device_id=device_id,
    )
    submission_error = None
    with _gpu_submission_guard():
        try:
            with use_stream:
                _consume_prefetched_group(
                    items, outs, stream, keepalive=keepalive
                )
                candidate_event = cp.cuda.Event()
                candidate_event.record(use_stream)
                guard.event = candidate_event
            in_flight.append(guard)
        except (KeyboardInterrupt, SystemExit):
            _quarantine_gpu_batches([guard])
            raise
        except BaseException as error:
            submission_error = error
            # Publish before releasing the gate so concurrent submissions
            # cannot pass while this completion state is unknown.
            _quarantine_gpu_batches([guard], actively_awaited=True)

    if submission_error is None:
        return

    try:
        _synchronize_gpu_batch(guard, synchronize_stream=True)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        if guard.event is None:
            # A stream-wide completion query can remain false while unrelated
            # work keeps a long-lived stream busy. Record a batch boundary
            # when the context still accepts work, retaining the stream query
            # as the fail-closed fallback if event recording also fails.
            try:
                with _gpu_batch_device(guard):
                    recovery_event = cp.cuda.Event()
                    recovery_event.record(use_stream)
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException:
                pass
            else:
                guard.event = recovery_event
    raise submission_error


def _release_completed_gpu_batches(
    in_flight: deque[_GpuBatchHandle],
) -> None:
    """Drop completed batch lifetime handles without synchronizing."""
    while in_flight:
        if not _gpu_batch_completion_is_confirmed(in_flight[0]):
            break
        in_flight.popleft()


def _wait_for_oldest_gpu_batch(
    in_flight: deque[_GpuBatchHandle],
) -> None:
    """Bound queued GPU work by waiting for one batch."""
    if not in_flight:
        return
    _synchronize_gpu_batch(in_flight[0])
    in_flight.popleft()


def _wait_for_all_gpu_batches(
    in_flight: deque[_GpuBatchHandle],
) -> None:
    """Drain events and quarantine owners with uncertain completion."""
    handles = tuple(in_flight)
    # Transfer every owner to durable storage before the first event wait.
    # An asynchronous exception between event waits must not leave later
    # handles owned only by a local deque that the caller can unwind past.
    try:
        _quarantine_gpu_batches(handles, actively_awaited=True)
    except BaseException:
        # Lock acquisition can itself be interrupted. Defer the original
        # exception until ownership is durable, even if cleanup is interrupted
        # again; do not enter GPU waits while propagating cancellation.
        while True:
            try:
                _quarantine_gpu_batches(handles)
                _mark_gpu_batches_unresolved(handles)
            except BaseException:
                continue
            break
        raise
    first_error: BaseException | None = None
    control_error: BaseException | None = None
    unresolved: list[_GpuBatchHandle] = []
    drain_finished = False
    try:
        for handle in handles:
            try:
                if handle.event is None:
                    raise RuntimeError("GPU batch has no completion event")
                with _gpu_batch_device(handle):
                    handle.event.synchronize()
            except BaseException as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    control_error = exc
                elif first_error is None:
                    first_error = exc
                unresolved.append(handle)
                _mark_gpu_batches_unresolved([handle])
                if control_error is not None:
                    break
            else:
                _release_quarantined_gpu_batches([handle])

        if control_error is not None:
            _mark_gpu_batches_unresolved(handles)
        elif unresolved:
            # An event wait can be interrupted before completion, or report a
            # sticky asynchronous CUDA error. Owners are already quarantined,
            # so make one best-effort synchronization attempt per stream.
            by_stream: dict[
                tuple[int | None, int],
                tuple[object, list[_GpuBatchHandle]],
            ] = {}
            for handle in unresolved:
                stream_key = (handle.device_id, id(handle.stream))
                if stream_key not in by_stream:
                    by_stream[stream_key] = (handle.stream, [])
                by_stream[stream_key][1].append(handle)
            for use_stream, stream_handles in by_stream.values():
                try:
                    with _gpu_batch_device(stream_handles[0]):
                        use_stream.synchronize()
                except BaseException as exc:
                    # Preserve the first error and quarantined owners.
                    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                        control_error = exc
                        break
                else:
                    _release_quarantined_gpu_batches(stream_handles)

        in_flight.clear()
        drain_finished = True
    finally:
        if not drain_finished:
            # The drain itself was interrupted outside an event wait. Durable
            # owners remain, but they must now fail closed as abandoned work.
            _mark_gpu_batches_unresolved(handles)
    raised_error = control_error or first_error
    if raised_error is not None:
        raise raised_error


class _NativeBatchPlanner(threading.Thread):
    """Plans FITS byte spans and submits batches to native builders."""

    def __init__(
        self,
        paths: Sequence[Path],
        hdu_indices: Sequence[int],
        builder,
        native_plan_files,
        *,
        decode_batch_files: int,
        native_plan_threads: int,
        section=None,
        outs=None,
    ):
        super().__init__(daemon=True, name="xdr-native-planner")
        self.paths = list(paths)
        self.hdu_indices = tuple(hdu_indices)
        self.builder = builder
        self.native_plan_files = native_plan_files
        self.decode_batch_files = int(decode_batch_files)
        self.native_plan_threads = int(native_plan_threads)
        self.section = section
        self.outs = outs
        self.direct_image_output = outs is not None and section is None
        self.planned_by_index: dict[int, _PlannedFile] = {}
        self.error: BaseException | None = None
        self._stop_event = threading.Event()

    def request_stop(self) -> None:
        self._stop_event.set()

    def _plan_one_file(self, file_index: int) -> _PlannedFile:
        with _nvtx_range("fits.stream.plan_native_file"):
            return _plan_file(
                self.paths[file_index],
                file_index,
                self.hdu_indices,
                section=self.section,
            )

    def _plan_group(
        self,
        start: int,
        end: int,
        executor: ThreadPoolExecutor,
    ) -> list[_PlannedFile]:
        file_indices = list(range(start, end))
        futures = {
            file_index: executor.submit(self._plan_one_file, file_index)
            for file_index in file_indices
        }
        group = []
        for file_index in file_indices:
            if self._stop_event.is_set():
                break
            planned = futures[file_index].result()
            self.planned_by_index[file_index] = planned
            group.append(planned)
        return group

    def _plan_group_native(self, start: int, end: int) -> list[_PlannedFile]:
        file_indices = list(range(start, end))
        native_items = self.native_plan_files(
            [str(self.paths[file_index]) for file_index in file_indices],
            file_indices,
            self.hdu_indices,
            self.native_plan_threads,
            self.section,
        )
        group = []
        for native_item in native_items:
            if self._stop_event.is_set():
                break
            planned = _planned_file_from_native_tuple(native_item)
            self.planned_by_index[planned.file_index] = planned
            group.append(planned)
        return group

    def run(self) -> None:
        batch_id = 0
        cursor = 0
        try:
            if self.native_plan_files is not None:
                plan_window_files = max(
                    self.decode_batch_files, self.native_plan_threads
                )
                while (
                    cursor < len(self.paths) and not self._stop_event.is_set()
                ):
                    with _nvtx_range("fits.stream.plan_native_batch"):
                        end = min(
                            cursor + plan_window_files,
                            len(self.paths),
                        )
                        planned_window = self._plan_group_native(cursor, end)
                        if not planned_window:
                            break
                    window_cursor = 0
                    while (
                        window_cursor < len(planned_window)
                        and not self._stop_event.is_set()
                    ):
                        group = planned_window[
                            window_cursor : window_cursor
                            + self.decode_batch_files
                        ]
                        if self.direct_image_output:
                            group, native_plans = (
                                _native_file_plans_with_direct_outputs(
                                    group, self.outs
                                )
                            )
                            for planned in group:
                                self.planned_by_index[planned.file_index] = (
                                    planned
                                )
                        else:
                            native_plans = _native_file_plans(group)
                        self.builder.submit_batch(batch_id, native_plans)
                        batch_id += 1
                        window_cursor += self.decode_batch_files
                    cursor = end
            else:
                max_workers = min(
                    self.native_plan_threads, max(1, len(self.paths))
                )
                with ThreadPoolExecutor(
                    max_workers=max_workers,
                    thread_name_prefix="xdr-native-plan",
                ) as executor:
                    while (
                        cursor < len(self.paths)
                        and not self._stop_event.is_set()
                    ):
                        with _nvtx_range("fits.stream.plan_native_batch"):
                            end = min(
                                cursor + self.decode_batch_files,
                                len(self.paths),
                            )
                            group = self._plan_group(cursor, end, executor)
                            if not group:
                                break
                            if self.direct_image_output:
                                group, native_plans = (
                                    _native_file_plans_with_direct_outputs(
                                        group, self.outs
                                    )
                                )
                                for planned in group:
                                    self.planned_by_index[
                                        planned.file_index
                                    ] = planned
                            else:
                                native_plans = _native_file_plans(group)
                            self.builder.submit_batch(batch_id, native_plans)
                        batch_id += 1
                        cursor = end
            if not self._stop_event.is_set():
                self.builder.close_input()
        except BaseException as exc:
            self.error = exc
            try:
                self.builder.request_stop()
            except Exception:
                pass


def _consume_native_batches(
    paths: Sequence[Path],
    hdu_indices: Sequence[int],
    outs: list,
    *,
    decode_batch_files: int,
    batch_queue_depth: int,
    native_read_threads: int,
    native_plan_threads: int,
    section,
    stream,
    NativeBatchBuilder,
    native_plan_files,
):
    """Consume device batches built by the C++ KvikIO worker pool."""
    import cupy as cp

    device_id = int(cp.cuda.Device().id)
    builder = NativeBatchBuilder(
        int(decode_batch_files),
        int(batch_queue_depth),
        int(native_read_threads),
        device_id,
    )
    planner = _NativeBatchPlanner(
        paths,
        hdu_indices,
        builder,
        native_plan_files,
        decode_batch_files=decode_batch_files,
        native_plan_threads=native_plan_threads,
        section=section,
        outs=outs,
    )
    in_flight: deque[_GpuBatchHandle] = deque()
    max_in_flight = max(1, int(batch_queue_depth))
    planner.start()
    try:
        try:
            while True:
                _release_completed_gpu_batches(in_flight)
                if len(in_flight) >= max_in_flight:
                    with _nvtx_range("fits.stream.release_completed_batches"):
                        _wait_for_oldest_gpu_batch(in_flight)
                with _nvtx_range("fits.stream.wait_native_batch"):
                    native_batch = builder.next_batch()
                if native_batch is None:
                    break

                batch_owner, device_ptr, device_nbytes, native_files = (
                    native_batch
                )
                device_nbytes = int(device_nbytes)
                if device_nbytes > 0:
                    mem = cp.cuda.UnownedMemory(
                        int(device_ptr), device_nbytes, batch_owner
                    )
                    device_buf = cp.ndarray(
                        (device_nbytes,),
                        dtype=cp.uint8,
                        memptr=cp.cuda.MemoryPointer(mem, 0),
                    )
                else:
                    device_buf = cp.empty(0, dtype=cp.uint8)

                items: list[PrefetchedFile] = []
                for file_index, device_offset in native_files:
                    planned = planner.planned_by_index[int(file_index)]
                    # Native reads are already device-resident, so host-side
                    # gzip header parsing is unavailable and the device probe
                    # remains intentional for this path.
                    items.append(
                        PrefetchedFile(
                            path=planned.path,
                            file_index=planned.file_index,
                            host_buf=device_buf,
                            hdus=planned.hdus,
                            buffer_base_offset=int(device_offset),
                        )
                    )

                with _nvtx_range("fits.stream.gpu_submit_batch"):
                    _submit_prefetched_group(
                        items,
                        outs,
                        stream=stream,
                        owner=native_batch,
                        in_flight=in_flight,
                    )
            planner.join()
            if planner.error is not None:
                raise planner.error
        except (KeyboardInterrupt, SystemExit):
            handles = tuple(in_flight)
            _quarantine_gpu_batches(handles)
            _mark_gpu_batches_unresolved(handles)
            raise
        except BaseException:
            try:
                with _nvtx_range("fits.stream.final_gpu_sync"):
                    _wait_for_all_gpu_batches(in_flight)
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException:
                # Preserve the original planner/submission/cancellation error.
                # The drain quarantines every owner it cannot synchronize.
                pass
            raise
        else:
            with _nvtx_range("fits.stream.final_gpu_sync"):
                _wait_for_all_gpu_batches(in_flight)
    finally:
        planner.request_stop()
        builder.request_stop()
        planner.join()


def _consume_python_batches(
    paths: Sequence[Path],
    hdu_indices: Sequence[int],
    outs: list,
    *,
    prefetch_depth: int,
    decode_batch_files: int,
    batch_queue_depth: int,
    section,
    stream,
) -> None:
    """Consume pinned-host batches with bounded event-owned lifetimes."""
    n_files = len(paths)
    q: queue.Queue = queue.Queue(maxsize=prefetch_depth)
    prefetcher = _FilePrefetcher(paths, hdu_indices, q, section=section)
    # Host-side header sizing removes the old blocking device probe. Retain
    # each pinned source group until its stream event confirms completion.
    in_flight: deque[_GpuBatchHandle] = deque()
    max_in_flight = max(1, int(batch_queue_depth))

    def submit(group: list[PrefetchedFile]) -> None:
        _release_completed_gpu_batches(in_flight)
        if len(in_flight) >= max_in_flight:
            _wait_for_oldest_gpu_batch(in_flight)
        _submit_prefetched_group(
            group,
            outs,
            stream=stream,
            owner=group,
            in_flight=in_flight,
        )

    prefetcher.start()
    try:
        try:
            received = 0
            group: list[PrefetchedFile] = []
            while received < n_files:
                try:
                    item = q.get(timeout=0.25)
                except queue.Empty:
                    if not prefetcher.is_alive():
                        break
                    continue
                if item is _SENTINEL:
                    break
                assert isinstance(item, PrefetchedFile)
                group.append(item)
                received += 1
                del item
                if len(group) >= decode_batch_files:
                    submit(group)
                    group = []

            if group and prefetcher.error is None:
                submit(group)
            if prefetcher.error is not None:
                raise prefetcher.error
        except (KeyboardInterrupt, SystemExit):
            handles = tuple(in_flight)
            _quarantine_gpu_batches(handles)
            _mark_gpu_batches_unresolved(handles)
            raise
        except BaseException:
            try:
                _wait_for_all_gpu_batches(in_flight)
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException:
                # Preserve the original prefetch/submission/cancellation
                # error; unresolved owners are already quarantined by drain.
                pass
            raise
        else:
            _wait_for_all_gpu_batches(in_flight)
    finally:
        # Cancellation cannot leave the producer blocked on a full queue.
        prefetcher.request_stop()
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass
        prefetcher.join()


def batch_to_device_stream(
    paths: Sequence[str | Path],
    hdu_indices: Iterable[int] = (1,),
    *,
    out=None,
    prefetch_depth: int = 2,
    decode_batch_files: int = 1,
    batch_queue_depth: int = 2,
    native_read_threads: int | None = None,
    native_plan_threads: int | None = None,
    native_batcher: str | bool = "auto",
    section=None,
    stream=None,
):
    """Pipelined version of `batch_to_device` — prefetches files while the
    GPU decodes the previous one.

    Parameters
    ----------
    paths
        FITS file paths to load.
    hdu_indices
        HDU indices to load from each file (must be the same per file).
    prefetch_depth
        Maximum number of files held in the pinned-host queue. 1 is
        effectively sequential; 2 means "read N+1 while decoding N"; higher
        values give the prefetcher room to run ahead when I/O is bursty.
    decode_batch_files
        Maximum number of prefetched files whose compressed tiles are grouped
        into one `gpu_gzip_decompress_batch` call. 1 batches all selected
        compressed HDUs from one file; higher values batch across files.
    batch_queue_depth
        Native-builder backpressure depth and strict limit for consumer
        event-owned batches. Native read workers can separately retain up to
        ``batch_queue_depth + native_read_threads`` outstanding batches. The
        Python fallback uses ``batch_queue_depth`` for event-owned GPU batches
        and their pinned staging buffers; each event also retains that batch's
        compressed and decompressed device scratch until completion. A
        conservative peak bound is ``prefetch_depth + 1 +
        (batch_queue_depth + 1) * decode_batch_files`` live file-staging
        buffers, including the producer's blocked item and the consumer's
        current group. Use shallow depths for multi-GB files.
    native_read_threads
        Number of native worker threads used for KvikIO reads into the native
        device batch buffer when the native batcher is enabled. None uses all
        CPU cores available to this process.
    native_plan_threads
        Number of native CFITSIO planner threads used to parse FITS headers
        and build native read plans when the rebuilt extension is available.
        None uses all CPU cores available to this process.
    native_batcher
        "auto" uses the C++ KvikIO batch builder when available and falls back
        to the Python prefetcher otherwise. True requires the native builder.
        False always uses the Python prefetcher.
    section
        Optional 2D ROI applied uniformly to CompImageHDUs.
    stream
        Optional CuPy stream for the consumer side. The function waits for
        all submitted work on this stream before returning. If completion
        cannot be confirmed, that work's source, scratch, and output buffers
        remain in a process-wide quarantine; later XDR submissions fail
        closed until completion becomes observable, and an unrecoverable
        CUDA context requires process restart.
    out
        Optional preallocated output array, or one array per HDU index. Each
        array must have shape ``(len(paths), ...)``, the expected dtype, and
        be C-contiguous.

    Returns
    -------
    One cupy.ndarray per HDU index, stacked along a new leading axis of
    length ``len(paths)``.
    """

    hdu_indices = tuple(int(i) for i in hdu_indices)
    paths = [Path(p) for p in paths]
    n_files = len(paths)
    if n_files == 0:
        raise ValueError("paths is empty")
    if prefetch_depth < 1:
        raise ValueError("prefetch_depth must be >= 1")
    if decode_batch_files < 1:
        raise ValueError("decode_batch_files must be >= 1")
    if batch_queue_depth < 1:
        raise ValueError("batch_queue_depth must be >= 1")
    if native_read_threads is None:
        native_read_threads = available_cpu_cores()
    if native_read_threads < 1:
        raise ValueError("native_read_threads must be >= 1")
    if native_plan_threads is None:
        native_plan_threads = available_cpu_cores()
    if native_plan_threads < 1:
        raise ValueError("native_plan_threads must be >= 1")
    if (
        native_batcher is not True
        and native_batcher is not False
        and native_batcher != "auto"
    ):
        raise ValueError("native_batcher must be 'auto', True, or False")
    _ensure_gpu_submissions_safe()
    configure_kvikio_parallelism()
    NativeBatchBuilder, native_plan_files = _native_batch_components(
        native_batcher
    )

    specs = _probe_output_specs(paths[0], hdu_indices, section=section)
    outs = _prepare_output_arrays(out, specs, n_files)

    if NativeBatchBuilder is not None:
        _consume_native_batches(
            paths,
            hdu_indices,
            outs,
            decode_batch_files=decode_batch_files,
            batch_queue_depth=batch_queue_depth,
            native_read_threads=native_read_threads,
            native_plan_threads=native_plan_threads,
            section=section,
            stream=stream,
            NativeBatchBuilder=NativeBatchBuilder,
            native_plan_files=native_plan_files,
        )
        return tuple(outs)

    _consume_python_batches(
        paths,
        hdu_indices,
        outs,
        prefetch_depth=prefetch_depth,
        decode_batch_files=decode_batch_files,
        batch_queue_depth=batch_queue_depth,
        section=section,
        stream=stream,
    )

    return tuple(outs)
