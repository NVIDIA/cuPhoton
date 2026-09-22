# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import gzip
import queue
import struct
import sys
import threading
import weakref
import zlib
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from cuphoton.xdr import kernels, nvcomp_batch, prefetch
from cuphoton.xdr.reader import GpuCompImageReader


class _CudaDevice:
    def __init__(self, device_id=None):
        self.id = 0 if device_id is None else int(device_id)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def _tracking_cuda_device(active_device):
    class Device:
        def __init__(self, device_id=None):
            self.id = (
                active_device["id"] if device_id is None else int(device_id)
            )

        def __enter__(self):
            self.previous = active_device["id"]
            active_device["id"] = self.id
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            active_device["id"] = self.previous
            return False

    return Device


class _ScriptedQueue:
    def __init__(self, items):
        self.items = list(items)

    def get(self, timeout):
        if not self.items:
            raise queue.Empty
        return self.items.pop(0)

    def get_nowait(self):
        if not self.items:
            raise queue.Empty
        return self.items.pop(0)


class _TrackingPrefetcher:
    error = None

    def __init__(self, trace=None):
        self.trace = trace
        self.stopped = False
        self.joined = False

    def start(self):
        pass

    def is_alive(self):
        return True

    def request_stop(self):
        self.stopped = True
        if self.trace is not None:
            self.trace.append("stop")

    def join(self):
        self.joined = True
        if self.trace is not None:
            self.trace.append("join")


def _empty_prefetched_files(count):
    return [
        prefetch.PrefetchedFile(
            path=f"{index}.fits",
            file_index=index,
            host_buf=object(),
            hdus=[],
        )
        for index in range(count)
    ]


def _gzip_member(
    payload=b"x",
    *,
    extra=None,
    filename=None,
    comment=None,
    hcrc=False,
):
    flags = 0
    if extra is not None:
        flags |= 0x04
    if filename is not None:
        flags |= 0x08
    if comment is not None:
        flags |= 0x10
    if hcrc:
        flags |= 0x02

    header = bytearray((0x1F, 0x8B, 0x08, flags))
    header.extend(struct.pack("<I", 0))
    header.extend((0, 255))
    if extra is not None:
        header.extend(struct.pack("<H", len(extra)))
        header.extend(extra)
    if filename is not None:
        header.extend(filename)
        header.append(0)
    if comment is not None:
        header.extend(comment)
        header.append(0)
    if hcrc:
        header.extend(struct.pack("<H", zlib.crc32(header) & 0xFFFF))

    compressor = zlib.compressobj(wbits=-zlib.MAX_WBITS)
    deflate = compressor.compress(payload) + compressor.flush()
    trailer = struct.pack(
        "<II",
        zlib.crc32(payload) & 0xFFFFFFFF,
        len(payload) & 0xFFFFFFFF,
    )
    return bytes(header) + deflate + trailer, len(header)


def _planned_compressed_file(
    *,
    rel_offsets=(0,),
    lengths=(20,),
    file_host_offset=0,
    heap_span_len=20,
):
    compressed = prefetch._HduPlan(
        hdu_index=1,
        kind="comp",
        plan={"sel_lengths": np.asarray(lengths, dtype=np.int64)},
        heap_rel_offsets=np.asarray(rel_offsets, dtype=np.int64),
        heap_span_start=0,
        heap_span_len=heap_span_len,
        file_host_offset=file_host_offset,
    )
    return prefetch._PlannedFile(
        path="image.fits",
        file_index=0,
        hdus=[compressed],
        spans=[],
        total_bytes=file_host_offset + heap_span_len,
    )


def _run_python_batches(
    monkeypatch,
    items,
    submit,
    *,
    batch_queue_depth,
    trace=None,
):
    out_queue = _ScriptedQueue(items)
    producer = _TrackingPrefetcher(trace)
    monkeypatch.setattr(
        prefetch,
        "queue",
        SimpleNamespace(
            Queue=lambda maxsize: out_queue,
            Empty=queue.Empty,
            Full=queue.Full,
        ),
    )
    monkeypatch.setattr(
        prefetch, "_FilePrefetcher", lambda *args, **kwargs: producer
    )
    monkeypatch.setattr(prefetch, "_submit_prefetched_group", submit)
    prefetch._consume_python_batches(
        [item.path for item in items],
        (),
        [],
        prefetch_depth=max(1, len(items)),
        decode_batch_files=1,
        batch_queue_depth=batch_queue_depth,
        section=None,
        stream=None,
    )
    return producer


@pytest.mark.parametrize(
    ("dtype_char", "itemsize", "expected"),
    [("i2", 2, "f4"), ("i8", 8, "f8")],
)
def test_probe_output_specs_promotes_scaled_integer_images(
    monkeypatch, dtype_char, itemsize, expected
):
    image = prefetch._HduPlan(
        hdu_index=0,
        kind="image",
        plan=None,
        heap_rel_offsets=None,
        heap_span_start=0,
        heap_span_len=itemsize,
        image_shape=(1,),
        image_dtype_char=dtype_char,
        image_itemsize=itemsize,
        image_bzero=32768.0,
    )
    planned = SimpleNamespace(hdus=[image])
    monkeypatch.setattr(
        prefetch, "_plan_file", lambda *args, **kwargs: planned
    )

    [spec] = prefetch._probe_output_specs("image.fits", (0,))

    assert np.dtype(spec.dtype_char) == np.dtype(expected)


def test_consume_prefetched_group_stores_promoted_image(monkeypatch):
    image = prefetch._HduPlan(
        hdu_index=0,
        kind="image",
        plan=None,
        heap_rel_offsets=None,
        heap_span_start=0,
        heap_span_len=2,
        image_shape=(1,),
        image_dtype_char="i2",
        image_itemsize=2,
        image_bzero=32768.0,
    )
    item = prefetch.PrefetchedFile(
        path="image.fits",
        file_index=0,
        host_buf=np.zeros(2, dtype=np.uint8),
        hdus=[image],
    )
    outs = [np.zeros((1, 1), dtype=np.float32)]
    scaled = np.array([32768.0], dtype=np.float32)
    monkeypatch.setattr(
        prefetch, "_consume_image", lambda *args, **kwargs: scaled
    )
    monkeypatch.setattr(
        prefetch, "_consume_comp_batch", lambda *args, **kwargs: None
    )

    prefetch._consume_prefetched_group([item], outs, stream=None)

    np.testing.assert_array_equal(outs[0][0], scaled)


def test_parse_hdu_header_sizes_uses_staged_tile_bytes():
    member, header_size = _gzip_member(payload=b"")
    host_buf = np.zeros(2 + len(member), dtype=np.uint8)
    host_buf[2:] = np.frombuffer(member, dtype=np.uint8)
    planned = _planned_compressed_file(file_host_offset=2)

    sizes = prefetch._parse_hdu_header_sizes(planned, host_buf)

    np.testing.assert_array_equal(sizes[0], np.array([header_size]))


def test_parse_hdu_header_sizes_accepts_empty_tile_table():
    planned = _planned_compressed_file(
        rel_offsets=[],
        lengths=[],
        heap_span_len=0,
    )

    sizes = prefetch._parse_hdu_header_sizes(
        planned,
        np.empty(0, dtype=np.uint8),
    )

    assert sizes[0].dtype == np.dtype(np.int64)
    assert sizes[0].shape == (0,)


def test_parse_hdu_header_sizes_accepts_long_optional_header():
    member, header_size = _gzip_member(
        payload=b"variable header",
        extra=b"e" * 17,
        filename=b"long-filename" * 2,
        comment=b"long-comment" * 2,
        hcrc=True,
    )
    assert header_size > 16
    assert gzip.decompress(member) == b"variable header"
    host_buf = np.zeros(3 + len(member), dtype=np.uint8)
    host_buf[3:] = np.frombuffer(member, dtype=np.uint8)
    planned = _planned_compressed_file(
        lengths=[len(member)],
        file_host_offset=3,
        heap_span_len=len(member),
    )

    sizes = prefetch._parse_hdu_header_sizes(planned, host_buf)

    np.testing.assert_array_equal(sizes[0], np.array([header_size]))


def test_host_and_device_header_probes_match_optional_fields(monkeypatch):
    options = [
        {},
        {"extra": b"extra-field"},
        {"filename": b"tile-name"},
        {"comment": b"tile-comment"},
        {"hcrc": True},
        {
            "extra": b"combined-extra",
            "filename": b"combined-name",
            "comment": b"combined-comment",
            "hcrc": True,
        },
    ]
    members = [
        _gzip_member(payload=f"tile-{index}".encode(), **fields)
        for index, fields in enumerate(options)
    ]
    tile_bytes = [member for member, _ in members]
    lengths = np.array([len(member) for member in tile_bytes], dtype=np.int64)
    rel_offsets = np.concatenate(([0], np.cumsum(lengths)[:-1]))
    prefix = b"file-prefix"
    host_buf = np.frombuffer(prefix + b"".join(tile_bytes), dtype=np.uint8)
    planned = _planned_compressed_file(
        rel_offsets=rel_offsets,
        lengths=lengths,
        file_host_offset=len(prefix),
        heap_span_len=int(lengths.sum()),
    )

    class DeviceBuffer:
        size = host_buf.size

        def __getitem__(self, key):
            return host_buf[key]

    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(asarray=np.asarray, asnumpy=np.asarray),
    )

    host_sizes = prefetch._parse_hdu_header_sizes(planned, host_buf)[0]
    device_sizes = nvcomp_batch._compute_header_sizes_from_device(
        DeviceBuffer(),
        rel_offsets + len(prefix),
        lengths,
    )

    expected = np.array([header_size for _, header_size in members])
    np.testing.assert_array_equal(host_sizes, expected)
    np.testing.assert_array_equal(device_sizes, expected)


@pytest.mark.parametrize(
    ("name_length", "expected_size"),
    [(21, 32), (22, None)],
)
def test_parse_hdu_header_sizes_enforces_header_safety_limit(
    monkeypatch, name_length, expected_size
):
    member, header_size = _gzip_member(
        payload=b"bounded host header",
        filename=b"n" * name_length,
    )
    planned = _planned_compressed_file(
        lengths=[len(member)],
        heap_span_len=len(member),
    )
    monkeypatch.setattr(nvcomp_batch, "_MAX_GZIP_HEADER_BYTES", 32)

    if expected_size is None:
        with pytest.raises(ValueError, match="exceeds 32-byte safety limit"):
            prefetch._parse_hdu_header_sizes(
                planned,
                np.frombuffer(member, dtype=np.uint8),
            )
    else:
        sizes = prefetch._parse_hdu_header_sizes(
            planned,
            np.frombuffer(member, dtype=np.uint8),
        )
        assert header_size == expected_size
        np.testing.assert_array_equal(sizes[0], [expected_size])


def test_parse_hdu_header_sizes_does_not_scan_into_trailer():
    malformed = bytearray((0x1F, 0x8B, 0x08, 0x08))
    malformed.extend(struct.pack("<I", 0))
    malformed.extend((0, 255))
    malformed.extend(b"unterminated-name")
    malformed.extend(b"\x00" * 8)
    planned = _planned_compressed_file(
        lengths=[len(malformed)],
        heap_span_len=len(malformed),
    )

    with pytest.raises(ValueError, match="truncated gzip header"):
        prefetch._parse_hdu_header_sizes(
            planned,
            np.frombuffer(malformed, dtype=np.uint8),
        )


@pytest.mark.parametrize(
    ("rel_offsets", "lengths", "match"),
    [
        ([[0]], [20], "must be 1D"),
        ([0, 1], [20], "same size"),
        ([-1], [20], "must be nonnegative"),
        ([0], [-1], "must be nonnegative"),
        ([0], [19], "too short to contain a header"),
    ],
)
def test_parse_hdu_header_sizes_rejects_invalid_tile_ranges(
    rel_offsets, lengths, match
):
    planned = _planned_compressed_file(
        rel_offsets=rel_offsets,
        lengths=lengths,
    )

    with pytest.raises(ValueError, match=match):
        prefetch._parse_hdu_header_sizes(
            planned,
            np.zeros(20, dtype=np.uint8),
        )


def test_parse_hdu_header_sizes_rejects_tile_in_neighboring_region():
    planned = _planned_compressed_file(
        rel_offsets=[20],
        lengths=[20],
        heap_span_len=20,
    )
    host_buf = np.zeros(40, dtype=np.uint8)
    host_buf[20:24] = (0x1F, 0x8B, 0x08, 0x00)

    with pytest.raises(ValueError, match="exceeds containing buffer"):
        prefetch._parse_hdu_header_sizes(planned, host_buf)


def test_parse_hdu_header_sizes_rejects_hdu_past_host_buffer():
    planned = _planned_compressed_file(file_host_offset=1)

    with pytest.raises(ValueError, match="exceeds staged host buffer"):
        prefetch._parse_hdu_header_sizes(
            planned,
            np.zeros(20, dtype=np.uint8),
        )


def test_parse_hdu_header_sizes_preserves_hdu_slots_and_tile_order():
    first_members = [
        _gzip_member(payload=b"a"),
        _gzip_member(payload=b"b", filename=b"first-name"),
    ]
    second_members = [
        _gzip_member(payload=b"c", extra=b"extra"),
        _gzip_member(payload=b"d", comment=b"second-comment"),
    ]
    first_blob = b"".join(member for member, _ in first_members)
    second_blob = b"".join(member for member, _ in second_members)
    host_buf = np.frombuffer(first_blob + second_blob, dtype=np.uint8)

    image = prefetch._HduPlan(
        hdu_index=1,
        kind="image",
        plan=None,
        heap_rel_offsets=None,
        heap_span_start=0,
        heap_span_len=0,
    )

    def compressed_hdu(hdu_index, members, file_host_offset):
        lengths = [len(member) for member, _ in members]
        return prefetch._HduPlan(
            hdu_index=hdu_index,
            kind="comp",
            plan={"sel_lengths": np.asarray(lengths, dtype=np.int64)},
            heap_rel_offsets=np.asarray([0, lengths[0]], dtype=np.int64),
            heap_span_start=0,
            heap_span_len=sum(lengths),
            file_host_offset=file_host_offset,
        )

    planned = prefetch._PlannedFile(
        path="mixed.fits",
        file_index=0,
        hdus=[
            image,
            compressed_hdu(2, first_members, 0),
            compressed_hdu(3, second_members, len(first_blob)),
        ],
        spans=[],
        total_bytes=host_buf.size,
    )

    sizes = prefetch._parse_hdu_header_sizes(planned, host_buf)

    assert list(sizes) == [1, 2]
    np.testing.assert_array_equal(
        sizes[1], [header_size for _, header_size in first_members]
    )
    np.testing.assert_array_equal(
        sizes[2], [header_size for _, header_size in second_members]
    )


def test_consume_group_carries_precomputed_header_sizes(monkeypatch):
    header_sizes = np.array([10], dtype=np.int64)
    compressed = prefetch._HduPlan(
        hdu_index=1,
        kind="comp",
        plan={},
        heap_rel_offsets=np.array([0], dtype=np.int64),
        heap_span_start=0,
        heap_span_len=18,
    )
    item = prefetch.PrefetchedFile(
        path="image.fits",
        file_index=0,
        host_buf=np.zeros(18, dtype=np.uint8),
        hdus=[compressed],
        hdu_header_sizes={0: header_sizes},
    )
    captured = []

    def consume(entries, stream, keepalive=None):
        captured.extend(entries)

    monkeypatch.setattr(prefetch, "_consume_comp_batch", consume)

    prefetch._consume_prefetched_group(
        [item],
        [np.empty((1, 1), dtype=np.uint8)],
        stream=None,
    )

    assert captured[0].header_sizes is header_sizes


def test_consume_comp_batch_maps_real_gzip_payload_span(monkeypatch):
    class NullStream:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    class DeviceArray(np.ndarray):
        def set(self, source):
            self[...] = source

    def empty(size, dtype):
        return np.empty(size, dtype=dtype).view(DeviceArray)

    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(
            empty=empty,
            ndarray=DeviceArray,
            uint8=np.uint8,
            cuda=SimpleNamespace(Stream=SimpleNamespace(null=NullStream())),
        ),
    )
    captured = {}

    def inflate(d_concat, rel_offsets, **kwargs):
        captured["d_concat"] = np.asarray(d_concat).copy()
        captured["rel_offsets"] = np.asarray(rel_offsets).copy()
        captured["lengths"] = np.asarray(kwargs["lengths"]).copy()
        captured["header_sizes"] = np.asarray(kwargs["header_sizes"]).copy()
        return empty(7, np.uint8), np.array([0], dtype=np.int64)

    monkeypatch.setattr(
        GpuCompImageReader,
        "inflate_device_heap",
        staticmethod(inflate),
    )
    monkeypatch.setattr(
        GpuCompImageReader,
        "postprocess_decoded_tiles",
        staticmethod(lambda *args, **kwargs: None),
    )
    payload = b"payload"
    member, header_size = _gzip_member(
        payload=payload,
        filename=b"variable-header-name",
    )
    prefix = b"file-prefix"
    host_buf = np.frombuffer(prefix + member, dtype=np.uint8)
    planned = _planned_compressed_file(
        lengths=[len(member)],
        file_host_offset=len(prefix),
        heap_span_len=len(member),
    )
    compressed = planned.hdus[0]
    compressed.plan["out_bytes"] = np.array([len(payload)], dtype=np.int64)
    header_sizes = prefetch._parse_hdu_header_sizes(planned, host_buf)[0]
    entry = prefetch._CompBatchEntry(
        plan_item=compressed,
        file_host_buf=host_buf,
        out=object(),
        header_sizes=header_sizes,
    )

    prefetch._consume_comp_batch([entry], stream=None)

    start = int(captured["rel_offsets"][0] + captured["header_sizes"][0])
    stop = int(captured["rel_offsets"][0] + captured["lengths"][0] - 8)
    assert header_sizes.tolist() == [header_size]
    assert bytes(captured["d_concat"][start:stop]) == member[header_size:-8]


def test_consume_comp_batch_mixed_headers_fall_back_to_device_probe(
    monkeypatch,
):
    class NullStream:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    class DeviceBuffer:
        pass

    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(
            ndarray=DeviceBuffer,
            cuda=SimpleNamespace(Stream=SimpleNamespace(null=NullStream())),
        ),
    )
    captured = {}

    def inflate(d_concat, rel_offsets, **kwargs):
        captured.update(kwargs)
        return np.empty(2, dtype=np.uint8), np.array([0, 1])

    monkeypatch.setattr(
        GpuCompImageReader,
        "inflate_device_heap",
        staticmethod(inflate),
    )
    monkeypatch.setattr(
        GpuCompImageReader,
        "postprocess_decoded_tiles",
        staticmethod(lambda *args, **kwargs: None),
    )
    shared_buffer = DeviceBuffer()
    entries = []
    for header_sizes in (np.array([10], dtype=np.int64), None):
        compressed = prefetch._HduPlan(
            hdu_index=1,
            kind="comp",
            plan={
                "sel_lengths": np.array([18], dtype=np.int64),
                "out_bytes": np.array([1], dtype=np.int64),
            },
            heap_rel_offsets=np.array([0], dtype=np.int64),
            heap_span_start=0,
            heap_span_len=18,
        )
        entries.append(
            prefetch._CompBatchEntry(
                plan_item=compressed,
                file_host_buf=shared_buffer,
                out=object(),
                header_sizes=header_sizes,
            )
        )

    prefetch._consume_comp_batch(entries, stream=None)

    assert captured["header_sizes"] is None


def test_consume_image_stages_raw_bytes_before_scaling(monkeypatch):
    class NullStream:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    fake_cupy = SimpleNamespace(
        dtype=np.dtype,
        empty=np.empty,
        ndarray=np.ndarray,
        uint8=np.uint8,
        float32=np.float32,
        float64=np.float64,
        cuda=SimpleNamespace(Stream=SimpleNamespace(null=NullStream())),
    )
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    monkeypatch.setattr(
        kernels,
        "byteswap_inplace",
        lambda array, itemsize: array.byteswap(inplace=True),
    )

    def scale(array, bzero, bscale):
        array *= bscale
        array += bzero

    monkeypatch.setattr(kernels, "apply_bzero_bscale", scale)
    image = prefetch._HduPlan(
        hdu_index=0,
        kind="image",
        plan=None,
        heap_rel_offsets=None,
        heap_span_start=0,
        heap_span_len=2,
        image_shape=(1,),
        image_dtype_char="i2",
        image_itemsize=2,
        image_bzero=32768.0,
    )
    raw = np.array([0], dtype=">i2").view(np.uint8)
    target = np.empty(1, dtype=np.float32)

    result = prefetch._consume_image(image, raw, target, stream=None)

    assert result is target
    np.testing.assert_array_equal(target, np.array([32768.0], dtype="f4"))


def test_native_plan_stages_image_when_output_dtype_is_promoted():
    image = prefetch._HduPlan(
        hdu_index=0,
        kind="image",
        plan=None,
        heap_rel_offsets=None,
        heap_span_start=2880,
        heap_span_len=2,
        image_shape=(1,),
        image_dtype_char="i2",
        image_itemsize=2,
        image_bzero=32768.0,
    )
    planned = prefetch._PlannedFile(
        path="image.fits",
        file_index=0,
        hdus=[image],
        spans=[prefetch._ReadSpan(2880, 2, 0)],
        total_bytes=2,
    )
    outs = [np.empty((1, 1), dtype=np.float32)]

    [converted], [native_plan] = (
        prefetch._native_file_plans_with_direct_outputs([planned], outs)
    )

    assert converted.hdus[0].direct_output is False
    assert converted.total_bytes == 2
    assert native_plan[-1] == []


def test_file_prefetcher_retries_sentinel_until_queue_accepts_it():
    class FullTwiceQueue:
        def __init__(self):
            self.calls = []

        def put(self, item, timeout):
            self.calls.append((item, timeout))
            if len(self.calls) < 3:
                raise queue.Full

    out_queue = FullTwiceQueue()
    producer = prefetch._FilePrefetcher([], (), out_queue)

    producer.run()

    assert [item for item, _ in out_queue.calls] == [
        prefetch._SENTINEL,
        prefetch._SENTINEL,
        prefetch._SENTINEL,
    ]


def test_gpu_batch_owner_lives_until_ordered_event_completion(monkeypatch):
    trace = []

    class Stream:
        def __enter__(self):
            trace.append("enter")
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            trace.append("exit")
            return False

        def synchronize(self):
            trace.append("stream-synchronize")

    class Event:
        def record(self, use_stream):
            assert use_stream is stream
            trace.append("event-record")

        def synchronize(self):
            trace.append("event-synchronize")

    class Owner:
        pass

    stream = Stream()
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(
            cuda=SimpleNamespace(
                Device=_CudaDevice,
                Event=Event,
                Stream=SimpleNamespace(null=None),
            )
        ),
    )

    def consume(items, outs, use_stream, keepalive):
        assert use_stream is stream
        trace.append("consume")
        keepalive.append("device-owner")

    monkeypatch.setattr(prefetch, "_consume_prefetched_group", consume)
    owner = Owner()
    owner_ref = weakref.ref(owner)
    in_flight = deque()
    prefetch._submit_prefetched_group(
        [], [], stream=stream, owner=owner, in_flight=in_flight
    )
    assert in_flight[0].device_id == 0

    del owner
    gc.collect()
    assert owner_ref() is not None

    prefetch._wait_for_oldest_gpu_batch(in_flight)
    gc.collect()

    assert owner_ref() is None
    assert trace == [
        "enter",
        "consume",
        "event-record",
        "exit",
        "event-synchronize",
    ]


def test_oldest_gpu_wait_quarantines_before_synchronizing(monkeypatch):
    interrupt = KeyboardInterrupt()
    quarantine = []

    class Event:
        def synchronize(self):
            assert quarantine == [handle]
            assert handle.actively_awaited is True
            raise interrupt

    handle = prefetch._GpuBatchHandle(Event(), ["owner"], object())
    in_flight = deque([handle])
    monkeypatch.setattr(prefetch, "_UNRESOLVED_GPU_BATCHES", quarantine)

    with pytest.raises(KeyboardInterrupt) as exc_info:
        prefetch._wait_for_oldest_gpu_batch(in_flight)

    assert exc_info.value is interrupt
    assert list(in_flight) == [handle]
    assert quarantine == [handle]
    assert handle.actively_awaited is False


def test_release_completed_gpu_batches_uses_event_done_and_stops_on_error():
    trace = []

    class Event:
        def __init__(self, index, *, raises=False):
            self.index = index
            self.raises = raises

        @property
        def done(self):
            trace.append(f"done-{self.index}")
            if self.raises:
                raise RuntimeError("query failed")
            return True

        def synchronize(self):
            pytest.fail("nonblocking retirement must not synchronize")

    retained = prefetch._GpuBatchHandle(
        Event(1, raises=True), [object()], object()
    )
    in_flight = deque(
        [
            prefetch._GpuBatchHandle(Event(0), [object()], object()),
            retained,
        ]
    )

    prefetch._release_completed_gpu_batches(in_flight)

    assert list(in_flight) == [retained]
    assert trace == ["done-0", "done-1"]


def test_gpu_batch_cancellation_synchronizes_before_owner_release(
    monkeypatch,
):
    class Cancelled(BaseException):
        pass

    class Stream:
        def __init__(self):
            self.synchronize_calls = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def synchronize(self):
            self.synchronize_calls += 1

    stream = Stream()
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(
            cuda=SimpleNamespace(
                Device=_CudaDevice,
                Event=lambda: pytest.fail("event must not be recorded"),
                Stream=SimpleNamespace(null=None),
            )
        ),
    )
    monkeypatch.setattr(
        prefetch,
        "_consume_prefetched_group",
        lambda *args, **kwargs: (_ for _ in ()).throw(Cancelled()),
    )

    with pytest.raises(Cancelled):
        prefetch._submit_prefetched_group(
            [], [], stream=stream, owner=object(), in_flight=deque()
        )

    assert stream.synchronize_calls == 1


def test_gpu_batch_registration_cancellation_synchronizes(monkeypatch):
    class Cancelled(BaseException):
        pass

    class Stream:
        def __init__(self):
            self.synchronize_calls = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def synchronize(self):
            self.synchronize_calls += 1

    class Event:
        def record(self, stream):
            pass

    class CancellingRegistry:
        def append(self, handle):
            raise Cancelled

    stream = Stream()
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(
            cuda=SimpleNamespace(
                Device=_CudaDevice,
                Event=Event,
                Stream=SimpleNamespace(null=None),
            )
        ),
    )
    monkeypatch.setattr(
        prefetch, "_consume_prefetched_group", lambda *args, **kwargs: None
    )

    with pytest.raises(Cancelled):
        prefetch._submit_prefetched_group(
            [],
            [],
            stream=stream,
            owner=object(),
            in_flight=CancellingRegistry(),
        )

    assert stream.synchronize_calls == 1


def test_gpu_batch_submit_error_quarantines_failed_stream_sync(monkeypatch):
    class Cancelled(BaseException):
        pass

    class Owner:
        pass

    primary_error = Cancelled("cancelled during submission")
    quarantine = []
    consume_calls = 0

    class Stream:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        @property
        def done(self):
            return False

        def synchronize(self):
            assert len(quarantine) == 1
            assert quarantine[0].actively_awaited is True
            prefetch._ensure_gpu_submissions_safe()
            raise RuntimeError("stream completion is unknown")

    def consume(*args, **kwargs):
        nonlocal consume_calls
        consume_calls += 1
        raise primary_error

    monkeypatch.setattr(prefetch, "_UNRESOLVED_GPU_BATCHES", quarantine)
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(
            cuda=SimpleNamespace(
                Device=_CudaDevice,
                Stream=SimpleNamespace(null=None),
            )
        ),
    )
    monkeypatch.setattr(prefetch, "_consume_prefetched_group", consume)
    owner = Owner()
    outs = [Owner()]

    with pytest.raises(Cancelled) as exc_info:
        prefetch._submit_prefetched_group(
            [],
            outs,
            stream=Stream(),
            owner=owner,
            in_flight=deque(),
        )

    assert exc_info.value is primary_error
    assert consume_calls == 1
    assert len(quarantine) == 1
    assert quarantine[0].keepalive[:2] == [owner, outs]
    assert quarantine[0].device_id == 0
    assert quarantine[0].actively_awaited is False

    with pytest.raises(RuntimeError, match="buffers remain quarantined"):
        prefetch._submit_prefetched_group(
            [],
            [],
            stream=Stream(),
            owner=object(),
            in_flight=deque(),
        )

    assert consume_calls == 1
    assert len(quarantine) == 1


def test_gpu_batch_submit_error_records_recovery_event(monkeypatch):
    primary_error = RuntimeError("submission failed")
    quarantine = []
    trace = []

    class Stream:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        @property
        def done(self):
            return False

        def synchronize(self):
            trace.append("stream-sync")
            raise RuntimeError("stream completion is unknown")

    class Event:
        @property
        def done(self):
            return True

        def record(self, stream):
            trace.append("event-record")

    monkeypatch.setattr(prefetch, "_UNRESOLVED_GPU_BATCHES", quarantine)
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(
            cuda=SimpleNamespace(
                Device=_CudaDevice,
                Event=Event,
                Stream=SimpleNamespace(null=None),
            )
        ),
    )
    monkeypatch.setattr(
        prefetch,
        "_consume_prefetched_group",
        lambda *args, **kwargs: (_ for _ in ()).throw(primary_error),
    )

    with pytest.raises(RuntimeError) as exc_info:
        prefetch._submit_prefetched_group(
            [], [], stream=Stream(), owner=object(), in_flight=deque()
        )

    assert exc_info.value is primary_error
    assert trace == ["stream-sync", "event-record"]
    assert len(quarantine) == 1
    assert isinstance(quarantine[0].event, Event)

    prefetch._ensure_gpu_submissions_safe()
    assert quarantine == []


@pytest.mark.parametrize("interrupt_type", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("stage", ["consume", "record", "register"])
def test_gpu_submit_control_signal_skips_cleanup_waits(
    monkeypatch, interrupt_type, stage
):
    interrupt = interrupt_type("submission interrupted")
    quarantine = []
    trace = []
    should_interrupt = True
    scratch = object()

    class Stream:
        done = False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def synchronize(self):
            trace.append("stream-wait")

    class Event:
        done = False

        def record(self, stream):
            trace.append("record")
            if stage == "record" and should_interrupt:
                raise interrupt

        def synchronize(self):
            trace.append("event-wait")

    class Registry(deque):
        def append(self, handle):
            if stage == "register" and should_interrupt:
                raise interrupt
            super().append(handle)

    def consume(*args, keepalive, **kwargs):
        keepalive.append(scratch)
        trace.append("consume")
        if stage == "consume" and should_interrupt:
            raise interrupt

    stream = Stream()
    monkeypatch.setattr(prefetch, "_UNRESOLVED_GPU_BATCHES", quarantine)
    monkeypatch.setattr(prefetch, "_consume_prefetched_group", consume)
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(
            cuda=SimpleNamespace(
                Device=_CudaDevice,
                Event=Event,
                Stream=SimpleNamespace(null=stream),
            )
        ),
    )
    owner = object()
    outs = [object()]
    in_flight = Registry()

    with pytest.raises(interrupt_type) as exc_info:
        prefetch._submit_prefetched_group(
            [], outs, stream=stream, owner=owner, in_flight=in_flight
        )

    assert exc_info.value is interrupt
    assert "stream-wait" not in trace
    assert "event-wait" not in trace
    assert not in_flight
    assert len(quarantine) == 1
    handle = quarantine[0]
    assert handle.keepalive == [owner, outs, scratch]
    assert handle.actively_awaited is False
    assert (handle.event is not None) == (stage == "register")

    should_interrupt = False
    trace_before_retry = trace.copy()
    with pytest.raises(RuntimeError, match="buffers remain quarantined"):
        prefetch._submit_prefetched_group(
            [], outs, stream=stream, owner=owner, in_flight=in_flight
        )
    assert trace == trace_before_retry

    completion = stream if handle.event is None else handle.event
    completion.done = True
    prefetch._submit_prefetched_group(
        [], outs, stream=stream, owner=owner, in_flight=in_flight
    )
    assert quarantine == []
    assert len(in_flight) == 1


@pytest.mark.parametrize("interrupt_type", [KeyboardInterrupt, SystemExit])
def test_gpu_submit_cleanup_control_signal_wins(monkeypatch, interrupt_type):
    primary_error = RuntimeError("submission failed")
    interrupt = interrupt_type("cleanup interrupted")
    quarantine = []

    class Stream:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def synchronize(self):
            raise interrupt

    monkeypatch.setattr(prefetch, "_UNRESOLVED_GPU_BATCHES", quarantine)
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(
            cuda=SimpleNamespace(
                Device=_CudaDevice,
                Stream=SimpleNamespace(null=None),
            )
        ),
    )
    monkeypatch.setattr(
        prefetch,
        "_consume_prefetched_group",
        lambda *args, **kwargs: (_ for _ in ()).throw(primary_error),
    )

    with pytest.raises(interrupt_type) as exc_info:
        prefetch._submit_prefetched_group(
            [], [], stream=Stream(), owner=object(), in_flight=deque()
        )

    assert exc_info.value is interrupt
    assert len(quarantine) == 1
    assert quarantine[0].actively_awaited is False


def test_gpu_submission_publishes_before_concurrent_quarantine(monkeypatch):
    consume_entered = threading.Event()
    release_consume = threading.Event()
    quarantine_lock_attempted = threading.Event()
    trace = []
    quarantine = []
    submitted = []
    results = {}

    class ObservedRLock:
        def __init__(self):
            self._lock = threading.RLock()

        def __enter__(self):
            if threading.current_thread().name == "quarantine-thread":
                quarantine_lock_attempted.set()
            self._lock.acquire()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self._lock.release()
            return False

    class Stream:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    class Event:
        @property
        def done(self):
            return False

        def record(self, stream):
            trace.append("event-recorded")

    class Registry:
        def append(self, handle):
            trace.append("handle-published")
            submitted.append(handle)

    def consume(*args, **kwargs):
        trace.append("consume-entered")
        consume_entered.set()
        if not release_consume.wait(2):
            raise TimeoutError("test did not release GPU submission")

    monkeypatch.setattr(
        prefetch, "_UNRESOLVED_GPU_BATCHES_LOCK", ObservedRLock()
    )
    monkeypatch.setattr(prefetch, "_UNRESOLVED_GPU_BATCHES", quarantine)
    monkeypatch.setattr(prefetch, "_consume_prefetched_group", consume)
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(
            cuda=SimpleNamespace(
                Device=_CudaDevice,
                Event=Event,
                Stream=SimpleNamespace(null=None),
            )
        ),
    )
    unresolved = prefetch._GpuBatchHandle(Event(), ["old"], Stream())

    def submit():
        try:
            prefetch._submit_prefetched_group(
                [], [], stream=Stream(), owner=object(), in_flight=Registry()
            )
        except BaseException as error:
            results["submit"] = error

    def quarantine_old():
        prefetch._quarantine_gpu_batches([unresolved])
        trace.append("quarantine-published")

    submit_thread = threading.Thread(target=submit, daemon=True)
    quarantine_thread = threading.Thread(
        target=quarantine_old,
        name="quarantine-thread",
        daemon=True,
    )
    submit_thread.start()
    assert consume_entered.wait(2)
    quarantine_thread.start()
    assert quarantine_lock_attempted.wait(2)
    release_consume.set()
    submit_thread.join(2)
    quarantine_thread.join(2)

    assert not submit_thread.is_alive()
    assert not quarantine_thread.is_alive()
    assert "submit" not in results
    assert trace == [
        "consume-entered",
        "event-recorded",
        "handle-published",
        "quarantine-published",
    ]
    assert len(submitted) == 1


def test_gpu_batch_record_error_does_not_trust_unrecorded_event(monkeypatch):
    primary_error = RuntimeError("event record failed")
    quarantine = []

    class Stream:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        @property
        def done(self):
            return False

        def synchronize(self):
            raise RuntimeError("stream completion is unknown")

    class Event:
        @property
        def done(self):
            return True

        def record(self, stream):
            raise primary_error

    monkeypatch.setattr(prefetch, "_UNRESOLVED_GPU_BATCHES", quarantine)
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(
            cuda=SimpleNamespace(
                Device=_CudaDevice,
                Event=Event,
                Stream=SimpleNamespace(null=None),
            )
        ),
    )
    monkeypatch.setattr(
        prefetch, "_consume_prefetched_group", lambda *args, **kwargs: None
    )

    with pytest.raises(RuntimeError) as exc_info:
        prefetch._submit_prefetched_group(
            [],
            [],
            stream=Stream(),
            owner=object(),
            in_flight=deque(),
        )

    assert exc_info.value is primary_error
    assert len(quarantine) == 1
    assert quarantine[0].event is None
    with pytest.raises(RuntimeError, match="buffers remain quarantined"):
        prefetch._ensure_gpu_submissions_safe()


def test_wait_for_all_gpu_batches_retains_handles_after_cancellation(
    monkeypatch,
):
    class Cancelled(BaseException):
        pass

    primary_error = Cancelled("cancelled during event wait")
    trace = []
    in_flight = deque()
    quarantine = []
    monkeypatch.setattr(prefetch, "_UNRESOLVED_GPU_BATCHES", quarantine)

    class Event:
        def __init__(self, index):
            self.index = index

        def synchronize(self):
            trace.append(f"event-{self.index}")
            if self.index == 0:
                raise primary_error
            raise RuntimeError("secondary asynchronous failure")

        @property
        def done(self):
            return False

    class Stream:
        def synchronize(self):
            assert len(in_flight) == 2
            assert all(handle.keepalive for handle in in_flight)
            trace.append("stream-sync")
            raise RuntimeError("sticky stream failure")

    stream = Stream()
    in_flight.extend(
        [
            prefetch._GpuBatchHandle(Event(0), [object()], stream),
            prefetch._GpuBatchHandle(Event(1), [object()], stream),
        ]
    )

    with pytest.raises(Cancelled) as exc_info:
        prefetch._wait_for_all_gpu_batches(in_flight)

    assert exc_info.value is primary_error
    assert trace == ["event-0", "event-1", "stream-sync"]
    assert not in_flight
    assert len(quarantine) == 2
    assert all(handle.keepalive for handle in quarantine)

    with pytest.raises(RuntimeError, match="buffers remain quarantined"):
        prefetch._ensure_gpu_submissions_safe()

    assert len(quarantine) == 2


@pytest.mark.parametrize("interrupt_type", [KeyboardInterrupt, SystemExit])
def test_wait_for_all_honors_control_signal_without_later_waits(
    monkeypatch, interrupt_type
):
    interrupt = interrupt_type("cancelled during event wait")
    quarantine = []
    trace = []
    monkeypatch.setattr(prefetch, "_UNRESOLVED_GPU_BATCHES", quarantine)

    class Event:
        def __init__(self, index):
            self.index = index

        def synchronize(self):
            trace.append(f"event-{self.index}")
            if self.index == 0:
                raise interrupt
            pytest.fail("control signal must stop later event waits")

    class Stream:
        def synchronize(self):
            pytest.fail("control signal must skip stream fallback")

    stream = Stream()
    handles = [
        prefetch._GpuBatchHandle(Event(0), ["first"], stream),
        prefetch._GpuBatchHandle(Event(1), ["second"], stream),
    ]
    in_flight = deque(handles)

    with pytest.raises(interrupt_type) as exc_info:
        prefetch._wait_for_all_gpu_batches(in_flight)

    assert exc_info.value is interrupt
    assert trace == ["event-0"]
    assert not in_flight
    assert quarantine == handles
    assert all(handle.actively_awaited is False for handle in handles)


@pytest.mark.parametrize("interrupt_type", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize(
    "cleanup_error_type", [None, KeyboardInterrupt, SystemExit, RuntimeError]
)
def test_wait_for_all_retains_owners_after_initial_quarantine_interrupt(
    monkeypatch, interrupt_type, cleanup_error_type
):
    interrupt = interrupt_type("initial quarantine interrupted")
    errors = [interrupt]
    if cleanup_error_type is not None:
        errors.extend(
            cleanup_error_type("cleanup interrupted") for _ in range(2)
        )
    quarantine = []
    owner_refs = []

    class InterruptingLock:
        attempts = 0

        def __enter__(self):
            self.attempts += 1
            if errors:
                raise errors.pop(0)
            return self

        def __exit__(self, *_):
            return False

    class Owner:
        pass

    class Event:
        done = False

        def synchronize(self):
            pytest.fail("control signal must skip event waits")

    class Stream:
        def synchronize(self):
            pytest.fail("control signal must skip stream waits")

    lock = InterruptingLock()
    monkeypatch.setattr(prefetch, "_UNRESOLVED_GPU_BATCHES", quarantine)
    monkeypatch.setattr(prefetch, "_UNRESOLVED_GPU_BATCHES_LOCK", lock)

    def final_drain():
        owners = [Owner(), Owner()]
        owner_refs.extend(weakref.ref(owner) for owner in owners)
        pending = deque(
            prefetch._GpuBatchHandle(Event(), [owner], Stream())
            for owner in owners
        )
        prefetch._wait_for_all_gpu_batches(pending)

    with pytest.raises(interrupt_type) as exc_info:
        final_drain()

    assert exc_info.value is interrupt
    # Discard traceback ownership too: only process-wide retention may keep
    # the pending sources alive after the caller's local deque is unwound.
    interrupt.__traceback__ = None
    del exc_info
    gc.collect()

    assert all(owner_ref() is not None for owner_ref in owner_refs)
    assert not errors
    assert len(quarantine) == 2
    assert all(handle.actively_awaited is False for handle in quarantine)
    with pytest.raises(RuntimeError, match="buffers remain quarantined"):
        prefetch._ensure_gpu_submissions_safe()


def test_wait_for_all_quarantines_later_handles_before_interrupt(monkeypatch):
    interrupt = KeyboardInterrupt()
    quarantine = []
    trace = []

    class Event:
        def __init__(self, index):
            self.index = index

        def synchronize(self):
            trace.append(f"event-{self.index}")
            if self.index == 1:
                pytest.fail("interrupt must stop before the second event")

    handles = [
        prefetch._GpuBatchHandle(Event(0), ["complete"], object()),
        prefetch._GpuBatchHandle(Event(1), ["still-active"], object()),
    ]
    in_flight = deque(handles)
    monkeypatch.setattr(prefetch, "_UNRESOLVED_GPU_BATCHES", quarantine)
    release = prefetch._release_quarantined_gpu_batches

    def interrupt_after_first_release(completed):
        release(completed)
        raise interrupt

    monkeypatch.setattr(
        prefetch,
        "_release_quarantined_gpu_batches",
        interrupt_after_first_release,
    )

    with pytest.raises(KeyboardInterrupt) as exc_info:
        prefetch._wait_for_all_gpu_batches(in_flight)

    assert exc_info.value is interrupt
    assert trace == ["event-0"]
    assert quarantine == [handles[1]]
    assert handles[1].keepalive == ["still-active"]
    assert handles[1].actively_awaited is False

    with pytest.raises(RuntimeError, match="buffers remain quarantined"):
        prefetch._ensure_gpu_submissions_safe()


def test_active_gpu_drain_does_not_block_concurrent_submission(monkeypatch):
    quarantine = []
    monkeypatch.setattr(prefetch, "_UNRESOLVED_GPU_BATCHES", quarantine)

    class Event:
        @property
        def done(self):
            return False

        def synchronize(self):
            assert quarantine == [handle]
            assert handle.actively_awaited is True
            prefetch._ensure_gpu_submissions_safe()

    handle = prefetch._GpuBatchHandle(Event(), ["active"], object())
    in_flight = deque([handle])

    prefetch._wait_for_all_gpu_batches(in_flight)

    assert not in_flight
    assert quarantine == []
    assert handle.actively_awaited is False


@pytest.mark.parametrize("wait_again", ["oldest", "all"])
def test_retrying_gpu_wait_reclaims_quarantined_batch(
    monkeypatch, wait_again
):
    quarantine = []
    waiting = threading.Event()
    resume = threading.Event()
    errors = []
    monkeypatch.setattr(prefetch, "_UNRESOLVED_GPU_BATCHES", quarantine)

    class Event:
        attempts = 0

        @property
        def done(self):
            return False

        def synchronize(self):
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("first wait failed")
            waiting.set()
            assert resume.wait(timeout=5), "retry was not released"

    handle = prefetch._GpuBatchHandle(Event(), ["owner"], object())
    in_flight = deque([handle])
    with pytest.raises(RuntimeError, match="first wait failed"):
        prefetch._wait_for_oldest_gpu_batch(in_flight)
    assert quarantine == [handle]
    assert handle.actively_awaited is False
    with pytest.raises(RuntimeError, match="buffers remain quarantined"):
        prefetch._ensure_gpu_submissions_safe()

    def retry():
        try:
            if wait_again == "oldest":
                prefetch._wait_for_oldest_gpu_batch(in_flight)
            else:
                prefetch._wait_for_all_gpu_batches(in_flight)
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=retry, daemon=True)
    thread.start()
    try:
        assert waiting.wait(timeout=5), "retry did not enter synchronization"
        # A different submitter may proceed while this owner is being waited
        # on, but the owner must remain durably retained through completion.
        prefetch._ensure_gpu_submissions_safe()
        assert quarantine == [handle]
        assert handle.keepalive == ["owner"]
        assert handle.actively_awaited is True
    finally:
        resume.set()
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert errors == []
    assert not in_flight
    assert quarantine == []
    assert handle.actively_awaited is False


def test_quarantine_probe_preserves_keyboard_interrupt(monkeypatch):
    interrupt = KeyboardInterrupt()

    class Event:
        @property
        def done(self):
            raise interrupt

    handle = prefetch._GpuBatchHandle(Event(), [object()], object())
    quarantine = [handle]
    monkeypatch.setattr(prefetch, "_UNRESOLVED_GPU_BATCHES", quarantine)

    with pytest.raises(KeyboardInterrupt) as exc_info:
        prefetch._ensure_gpu_submissions_safe()

    assert exc_info.value is interrupt
    assert quarantine == [handle]


@pytest.mark.parametrize("use_event", [True, False])
def test_quarantine_probe_releases_confirmed_completion(
    monkeypatch, use_event
):
    active_device = {"id": 0}
    observed_devices = []
    Device = _tracking_cuda_device(active_device)

    class Completion:
        @property
        def done(self):
            observed_devices.append(active_device["id"])
            return True

    handle = prefetch._GpuBatchHandle(
        Completion() if use_event else None,
        [object()],
        Completion(),
        device_id=1,
    )
    quarantine = [handle]
    monkeypatch.setattr(prefetch, "_UNRESOLVED_GPU_BATCHES", quarantine)
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(cuda=SimpleNamespace(Device=Device)),
    )

    prefetch._ensure_gpu_submissions_safe()

    assert quarantine == []
    assert observed_devices == [1]
    assert active_device["id"] == 0


def test_wait_for_all_groups_shared_default_stream_by_device(monkeypatch):
    active_device = {"id": 0}
    trace = []
    quarantine = []
    Device = _tracking_cuda_device(active_device)

    class Event:
        def __init__(self, device_id):
            self.device_id = device_id

        def synchronize(self):
            assert active_device["id"] == self.device_id
            trace.append(f"event-{self.device_id}")
            raise RuntimeError(f"event-{self.device_id} failed")

    class DefaultStream:
        def synchronize(self):
            trace.append(f"stream-{active_device['id']}")

    stream = DefaultStream()
    handles = [
        prefetch._GpuBatchHandle(
            Event(device_id),
            [f"owner-{device_id}"],
            stream,
            device_id=device_id,
        )
        for device_id in (0, 1)
    ]
    in_flight = deque(handles)
    monkeypatch.setattr(prefetch, "_UNRESOLVED_GPU_BATCHES", quarantine)
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(cuda=SimpleNamespace(Device=Device)),
    )

    with pytest.raises(RuntimeError, match="event-0 failed"):
        prefetch._wait_for_all_gpu_batches(in_flight)

    assert trace == ["event-0", "event-1", "stream-0", "stream-1"]
    assert not in_flight
    assert quarantine == []
    assert active_device["id"] == 0


def test_wait_for_all_gpu_batches_releases_only_confirmed_stream(monkeypatch):
    quarantine = []
    trace = []
    first_error = RuntimeError("first event failed")

    class Event:
        def __init__(self, index):
            self.index = index

        @property
        def done(self):
            return False

        def synchronize(self):
            trace.append(f"event-{self.index}")
            if self.index == 0:
                raise first_error
            raise RuntimeError("second event failed")

    class Stream:
        def __init__(self, index, succeeds):
            self.index = index
            self.succeeds = succeeds

        def synchronize(self):
            trace.append(f"stream-{self.index}")
            if not self.succeeds:
                raise RuntimeError("stream completion is unknown")

    handles = [
        prefetch._GpuBatchHandle(Event(0), ["released"], Stream(0, True)),
        prefetch._GpuBatchHandle(Event(1), ["retained"], Stream(1, False)),
    ]
    in_flight = deque(handles)
    monkeypatch.setattr(prefetch, "_UNRESOLVED_GPU_BATCHES", quarantine)

    with pytest.raises(RuntimeError) as exc_info:
        prefetch._wait_for_all_gpu_batches(in_flight)

    assert exc_info.value is first_error
    assert trace == ["event-0", "event-1", "stream-0", "stream-1"]
    assert not in_flight
    assert quarantine == [handles[1]]


def test_python_batches_bound_in_flight_pinned_owners(monkeypatch):
    items = _empty_prefetched_files(3)
    state = {"active": 0, "maximum": 0}
    synchronized = []

    class Event:
        def __init__(self, index):
            self.index = index

        @property
        def done(self):
            return False

        def synchronize(self):
            state["active"] -= 1
            synchronized.append(self.index)

    def submit(group, outs, *, stream, owner, in_flight):
        index = owner[0].file_index
        state["active"] += 1
        state["maximum"] = max(state["maximum"], state["active"])
        handle = prefetch._GpuBatchHandle(Event(index), [owner], object())
        in_flight.append(handle)
        return handle

    producer = _run_python_batches(
        monkeypatch,
        items,
        submit,
        batch_queue_depth=2,
    )

    assert state == {"active": 0, "maximum": 2}
    assert synchronized == [0, 1, 2]
    assert producer.stopped is True
    assert producer.joined is True


def test_backpressure_failure_drains_and_quarantines(monkeypatch):
    items = _empty_prefetched_files(2)
    primary_error = RuntimeError("backpressure event wait failed")
    quarantine = []
    trace = []

    class Event:
        @property
        def done(self):
            return False

        def synchronize(self):
            trace.append("event-wait")
            raise primary_error

    class Stream:
        def synchronize(self):
            trace.append("stream-wait")
            raise RuntimeError("sticky stream failure")

    handle = prefetch._GpuBatchHandle(Event(), [items[0]], Stream())

    def submit(group, outs, *, stream, owner, in_flight):
        trace.append(f"submit-{owner[0].file_index}")
        in_flight.append(handle)

    monkeypatch.setattr(prefetch, "_UNRESOLVED_GPU_BATCHES", quarantine)

    with pytest.raises(RuntimeError) as exc_info:
        _run_python_batches(
            monkeypatch,
            items,
            submit,
            batch_queue_depth=1,
            trace=trace,
        )

    assert exc_info.value is primary_error
    assert trace == [
        "submit-0",
        "event-wait",
        "event-wait",
        "stream-wait",
        "stop",
        "join",
    ]
    assert quarantine == [handle]
    assert handle.keepalive == [items[0]]


@pytest.mark.parametrize("consumer", ["python", "native"])
@pytest.mark.parametrize("interrupt_type", [KeyboardInterrupt, SystemExit])
def test_backpressure_control_signal_retains_all_pending_batches(
    monkeypatch, consumer, interrupt_type
):
    items = _empty_prefetched_files(3)
    interrupt = interrupt_type("backpressure interrupted")
    quarantine = []
    handles = []
    waits = []
    trace = []

    class Event:
        done = False

        def __init__(self, index):
            self.index = index

        def synchronize(self):
            waits.append(self.index)
            raise interrupt

    class Builder:
        def __init__(self, *args):
            self.next_index = 0

        def next_batch(self):
            if self.next_index == len(items):
                return None
            owner = items[self.next_index]
            self.next_index += 1
            return (owner, 0, 0, [])

        def request_stop(self):
            trace.append("builder-stop")

    class Planner:
        error = None

        def __init__(self, *args, **kwargs):
            self.planned_by_index = {}

        def start(self):
            pass

        def request_stop(self):
            trace.append("planner-stop")

        def join(self):
            trace.append("planner-join")

    def submit(group, outs, *, stream, owner, in_flight):
        handle = prefetch._GpuBatchHandle(
            Event(len(handles)), [owner, outs], object()
        )
        handles.append(handle)
        in_flight.append(handle)

    monkeypatch.setattr(prefetch, "_UNRESOLVED_GPU_BATCHES", quarantine)
    monkeypatch.setattr(prefetch, "_submit_prefetched_group", submit)
    monkeypatch.setattr(prefetch, "_NativeBatchPlanner", Planner)
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(
            uint8=np.uint8,
            empty=lambda *args, **kwargs: object(),
            cuda=SimpleNamespace(Device=_CudaDevice),
        ),
    )
    with pytest.raises(interrupt_type) as exc_info:
        if consumer == "python":
            _run_python_batches(
                monkeypatch,
                items,
                submit,
                batch_queue_depth=2,
                trace=trace,
            )
        else:
            prefetch._consume_native_batches(
                [item.path for item in items],
                (),
                [],
                decode_batch_files=1,
                batch_queue_depth=2,
                native_read_threads=1,
                native_plan_threads=1,
                section=None,
                stream=None,
                NativeBatchBuilder=Builder,
                native_plan_files=lambda *args, **kwargs: [],
            )

    assert exc_info.value is interrupt
    assert waits == [0]
    assert len(handles) == 2
    assert quarantine == handles
    for index, handle in enumerate(handles):
        assert handle.actively_awaited is False
        assert handle.keepalive[0][0] is items[index]
        assert handle.keepalive[1] == []
    assert trace == (
        ["stop", "join"]
        if consumer == "python"
        else ["planner-stop", "builder-stop", "planner-join"]
    )

    with pytest.raises(RuntimeError, match="buffers remain quarantined"):
        prefetch._ensure_gpu_submissions_safe()
    handles[0].event.done = True
    with pytest.raises(RuntimeError, match="buffers remain quarantined"):
        prefetch._ensure_gpu_submissions_safe()
    assert quarantine == [handles[1]]
    handles[1].event.done = True
    prefetch._ensure_gpu_submissions_safe()
    assert quarantine == []
    assert waits == [0]


def test_batch_stream_waits_for_python_gpu_completion(monkeypatch):
    items = _empty_prefetched_files(1)
    out_queue = _ScriptedQueue(items)
    producer = _TrackingPrefetcher()
    trace = []

    class Event:
        @property
        def done(self):
            return False

        def synchronize(self):
            trace.append("event-synchronize")

    def submit(group, outs, *, stream, owner, in_flight):
        trace.append("submit")
        in_flight.append(
            prefetch._GpuBatchHandle(Event(), [owner, outs], object())
        )

    monkeypatch.setattr(
        prefetch,
        "queue",
        SimpleNamespace(
            Queue=lambda maxsize: out_queue,
            Empty=queue.Empty,
            Full=queue.Full,
        ),
    )
    monkeypatch.setattr(
        prefetch, "_FilePrefetcher", lambda *args, **kwargs: producer
    )
    monkeypatch.setattr(prefetch, "_submit_prefetched_group", submit)
    monkeypatch.setattr(
        prefetch, "configure_kvikio_parallelism", lambda: None
    )
    monkeypatch.setattr(
        prefetch,
        "_native_batch_components",
        lambda native_batcher: (None, None),
    )
    monkeypatch.setattr(
        prefetch, "_probe_output_specs", lambda *args, **kwargs: []
    )
    monkeypatch.setattr(
        prefetch, "_prepare_output_arrays", lambda out, specs, n_files: []
    )

    result = prefetch.batch_to_device_stream(
        ["image.fits"],
        hdu_indices=(),
        native_batcher=False,
        native_read_threads=1,
        native_plan_threads=1,
    )

    assert result == ()
    assert trace == ["submit", "event-synchronize"]


def test_native_batches_bound_in_flight_device_owners(monkeypatch):
    state = {"active": 0, "maximum": 0}
    trace = []

    class Event:
        def __init__(self, index):
            self.index = index

        @property
        def done(self):
            return False

        def synchronize(self):
            state["active"] -= 1
            trace.append(f"wait-{self.index}")

    class Builder:
        instance = None

        def __init__(self, *args):
            type(self).instance = self
            self.next_index = 0
            self.stopped = False

        def next_batch(self):
            if self.next_index == 3:
                trace.append("next-end")
                return None
            owner = SimpleNamespace(index=self.next_index)
            trace.append(f"next-{self.next_index}")
            self.next_index += 1
            return (owner, 0, 0, [])

        def request_stop(self):
            self.stopped = True

    class Planner:
        def __init__(self, *args, **kwargs):
            self.error = None
            self.planned_by_index = {}

        def start(self):
            pass

        def join(self):
            pass

        def request_stop(self):
            pass

    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(
            uint8=np.uint8,
            empty=lambda *args, **kwargs: object(),
            cuda=SimpleNamespace(Device=lambda: SimpleNamespace(id=0)),
        ),
    )
    monkeypatch.setattr(prefetch, "_NativeBatchPlanner", Planner)

    def submit(items, outs, *, stream, owner, in_flight):
        index = owner[0].index
        trace.append(f"submit-{index}")
        state["active"] += 1
        state["maximum"] = max(state["maximum"], state["active"])
        handle = prefetch._GpuBatchHandle(Event(index), [owner], object())
        in_flight.append(handle)
        return handle

    monkeypatch.setattr(prefetch, "_submit_prefetched_group", submit)

    prefetch._consume_native_batches(
        [],
        (),
        [],
        decode_batch_files=1,
        batch_queue_depth=2,
        native_read_threads=1,
        native_plan_threads=1,
        section=None,
        stream=None,
        NativeBatchBuilder=Builder,
        native_plan_files=lambda *args, **kwargs: [],
    )

    assert state == {"active": 0, "maximum": 2}
    assert trace == [
        "next-0",
        "submit-0",
        "next-1",
        "submit-1",
        "wait-0",
        "next-2",
        "submit-2",
        "wait-1",
        "next-end",
        "wait-2",
    ]
    assert Builder.instance.stopped is True


@pytest.mark.parametrize(
    "control_error",
    [
        None,
        KeyboardInterrupt("cleanup interrupted"),
        SystemExit("cleanup interrupted"),
    ],
    ids=["ordinary-drain-error", "keyboard-interrupt", "system-exit"],
)
def test_native_batches_error_drain_precedence(monkeypatch, control_error):
    primary_error = RuntimeError("second submit failed")
    quarantine = []
    trace = []
    monkeypatch.setattr(prefetch, "_UNRESOLVED_GPU_BATCHES", quarantine)

    class Event:
        @property
        def done(self):
            return False

        def synchronize(self):
            trace.append("event-wait")
            if control_error is not None:
                raise control_error
            raise RuntimeError("asynchronous GPU failure")

    class Stream:
        def synchronize(self):
            if control_error is not None:
                pytest.fail("control signal must skip stream fallback")
            trace.append("stream-sync")
            raise RuntimeError("sticky stream failure")

    class Builder:
        instance = None

        def __init__(self, *args):
            type(self).instance = self
            self.next_index = 0
            self.owners = []

        def next_batch(self):
            if self.next_index == 2:
                return None
            owner = SimpleNamespace(index=self.next_index)
            self.owners.append(owner)
            self.next_index += 1
            return (owner, 0, 0, [])

        def request_stop(self):
            trace.append("builder-stop")

    class Planner:
        def __init__(self, *args, **kwargs):
            self.error = None
            self.planned_by_index = {}

        def start(self):
            pass

        def join(self):
            trace.append("planner-join")

        def request_stop(self):
            trace.append("planner-stop")

    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(
            uint8=np.uint8,
            empty=lambda *args, **kwargs: object(),
            cuda=SimpleNamespace(Device=lambda: SimpleNamespace(id=0)),
        ),
    )
    monkeypatch.setattr(prefetch, "_NativeBatchPlanner", Planner)
    submitted_handles = []

    def submit(items, outs, *, stream, owner, in_flight):
        index = owner[0].index
        trace.append(f"submit-{index}")
        if index == 1:
            raise primary_error
        handle = prefetch._GpuBatchHandle(Event(), [owner], Stream())
        submitted_handles.append(handle)
        in_flight.append(handle)

    monkeypatch.setattr(prefetch, "_submit_prefetched_group", submit)

    expected_error = primary_error if control_error is None else control_error
    with pytest.raises(type(expected_error)) as exc_info:
        prefetch._consume_native_batches(
            [],
            (),
            [],
            decode_batch_files=1,
            batch_queue_depth=2,
            native_read_threads=1,
            native_plan_threads=1,
            section=None,
            stream=None,
            NativeBatchBuilder=Builder,
            native_plan_files=lambda *args, **kwargs: [],
        )

    expected_trace = [
        "submit-0",
        "submit-1",
        "event-wait",
    ]
    if control_error is None:
        expected_trace.append("stream-sync")
    expected_trace.extend(["planner-stop", "builder-stop", "planner-join"])

    assert exc_info.value is expected_error
    assert trace == expected_trace
    assert quarantine == submitted_handles
    assert quarantine[0].keepalive[0][0] is Builder.instance.owners[0]
    assert quarantine[0].actively_awaited is False


def test_python_batches_wait_before_cleanup_after_later_submit_error(
    monkeypatch,
):
    items = _empty_prefetched_files(2)
    trace = []

    class Event:
        @property
        def done(self):
            return False

        def synchronize(self):
            trace.append("wait-0")

    def submit(group, outs, *, stream, owner, in_flight):
        index = owner[0].file_index
        trace.append(f"submit-{index}")
        if index == 1:
            raise RuntimeError("second submit failed")
        handle = prefetch._GpuBatchHandle(Event(), [owner], object())
        in_flight.append(handle)
        return handle

    with pytest.raises(RuntimeError, match="second submit failed"):
        _run_python_batches(
            monkeypatch,
            items,
            submit,
            batch_queue_depth=2,
            trace=trace,
        )

    assert trace == ["submit-0", "submit-1", "wait-0", "stop", "join"]


@pytest.mark.parametrize(
    "control_error",
    [
        None,
        KeyboardInterrupt("cleanup interrupted"),
        SystemExit("cleanup interrupted"),
    ],
    ids=["ordinary-drain-error", "keyboard-interrupt", "system-exit"],
)
def test_python_batches_error_drain_precedence(monkeypatch, control_error):
    items = _empty_prefetched_files(2)
    trace = []
    primary_error = RuntimeError("second submit failed")
    quarantine = []
    monkeypatch.setattr(prefetch, "_UNRESOLVED_GPU_BATCHES", quarantine)

    class Event:
        @property
        def done(self):
            return False

        def synchronize(self):
            trace.append("event-wait")
            if control_error is not None:
                raise control_error
            raise RuntimeError("asynchronous GPU failure")

    class Stream:
        def synchronize(self):
            if control_error is not None:
                pytest.fail("control signal must skip stream fallback")
            trace.append("stream-sync")
            raise RuntimeError("sticky stream failure")

    def submit(group, outs, *, stream, owner, in_flight):
        index = owner[0].file_index
        trace.append(f"submit-{index}")
        if index == 1:
            raise primary_error
        handle = prefetch._GpuBatchHandle(Event(), [owner], Stream())
        in_flight.append(handle)
        return handle

    expected_error = primary_error if control_error is None else control_error
    with pytest.raises(type(expected_error)) as exc_info:
        _run_python_batches(
            monkeypatch,
            items,
            submit,
            batch_queue_depth=2,
            trace=trace,
        )

    expected_trace = [
        "submit-0",
        "submit-1",
        "event-wait",
    ]
    if control_error is None:
        expected_trace.append("stream-sync")
    expected_trace.extend(["stop", "join"])

    assert exc_info.value is expected_error
    assert trace == expected_trace
    assert len(quarantine) == 1


def test_python_batches_cleanup_after_async_event_error(monkeypatch):
    items = _empty_prefetched_files(1)
    trace = []
    quarantine = []
    monkeypatch.setattr(prefetch, "_UNRESOLVED_GPU_BATCHES", quarantine)

    class Event:
        @property
        def done(self):
            return False

        def synchronize(self):
            trace.append("wait")
            raise RuntimeError("asynchronous GPU failure")

    class Stream:
        def synchronize(self):
            trace.append("stream-sync")

    def submit(group, outs, *, stream, owner, in_flight):
        trace.append("submit")
        handle = prefetch._GpuBatchHandle(Event(), [owner], Stream())
        in_flight.append(handle)
        return handle

    with pytest.raises(RuntimeError, match="asynchronous GPU failure"):
        _run_python_batches(
            monkeypatch,
            items,
            submit,
            batch_queue_depth=1,
            trace=trace,
        )

    assert trace == ["submit", "wait", "stream-sync", "stop", "join"]
    assert not quarantine


def test_batch_stream_propagates_error_if_prefetcher_exits_without_item(
    monkeypatch,
):
    class EmptyQueue:
        def __init__(self):
            self.timeouts = []

        def get(self, timeout):
            self.timeouts.append(timeout)
            raise queue.Empty

        def get_nowait(self):
            raise queue.Empty

    class FailedPrefetcher:
        def __init__(self, *args, **kwargs):
            self.error = RuntimeError("prefetch failed")

        def start(self):
            pass

        def is_alive(self):
            return False

        def request_stop(self):
            pass

        def join(self):
            pass

    out_queue = EmptyQueue()
    monkeypatch.setattr(
        prefetch,
        "queue",
        SimpleNamespace(
            Queue=lambda maxsize: out_queue,
            Empty=queue.Empty,
            Full=queue.Full,
        ),
    )
    monkeypatch.setattr(prefetch, "_FilePrefetcher", FailedPrefetcher)
    monkeypatch.setattr(
        prefetch, "configure_kvikio_parallelism", lambda: None
    )
    monkeypatch.setattr(
        prefetch, "_native_batch_components", lambda value: (None, None)
    )
    monkeypatch.setattr(
        prefetch,
        "_probe_output_specs",
        lambda *args, **kwargs: [prefetch._OutputSpec((1,), "f4")],
    )
    monkeypatch.setattr(
        prefetch,
        "_prepare_output_arrays",
        lambda *args, **kwargs: [np.empty((1, 1), dtype=np.float32)],
    )

    with pytest.raises(RuntimeError, match="prefetch failed"):
        prefetch.batch_to_device_stream(["image.fits"], native_batcher=False)

    assert out_queue.timeouts
    assert all(timeout > 0 for timeout in out_queue.timeouts)


def test_batch_stream_does_not_consume_partial_group_after_prefetch_error(
    monkeypatch,
):
    partial = prefetch.PrefetchedFile(
        path="first.fits",
        file_index=0,
        host_buf=np.empty(0, dtype=np.uint8),
        hdus=[],
    )

    class PartialQueue:
        def __init__(self):
            self.items = iter((partial, prefetch._SENTINEL))

        def get(self, timeout):
            return next(self.items)

        def get_nowait(self):
            raise queue.Empty

    class FailedPrefetcher:
        def __init__(self, *args, **kwargs):
            self.error = RuntimeError("second file failed")

        def start(self):
            pass

        def is_alive(self):
            return False

        def request_stop(self):
            pass

        def join(self):
            pass

    output = np.zeros((2, 1), dtype=np.float32)
    consumed = []

    def consume(items, outs, stream):
        consumed.extend(items)
        outs[0][0] = 1.0

    monkeypatch.setattr(
        prefetch,
        "queue",
        SimpleNamespace(
            Queue=lambda maxsize: PartialQueue(),
            Empty=queue.Empty,
            Full=queue.Full,
        ),
    )
    monkeypatch.setattr(prefetch, "_FilePrefetcher", FailedPrefetcher)
    monkeypatch.setattr(
        prefetch, "configure_kvikio_parallelism", lambda: None
    )
    monkeypatch.setattr(
        prefetch, "_native_batch_components", lambda value: (None, None)
    )
    monkeypatch.setattr(
        prefetch,
        "_probe_output_specs",
        lambda *args, **kwargs: [prefetch._OutputSpec((1,), "f4")],
    )
    monkeypatch.setattr(
        prefetch,
        "_prepare_output_arrays",
        lambda *args, **kwargs: [output],
    )
    monkeypatch.setattr(prefetch, "_consume_prefetched_group", consume)

    with pytest.raises(RuntimeError, match="second file failed"):
        prefetch.batch_to_device_stream(
            ["first.fits", "second.fits"],
            decode_batch_files=2,
            native_batcher=False,
        )

    assert consumed == []
    np.testing.assert_array_equal(output, np.zeros_like(output))
