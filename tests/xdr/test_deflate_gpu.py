# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Raw DEFLATE alignment and asynchronous decode ownership."""

from __future__ import annotations

import gzip
import struct
import zlib
from types import SimpleNamespace

import numpy as np
import pytest

from cuphoton.xdr import nvcomp_batch


@pytest.fixture
def cp():
    module = pytest.importorskip("cupy")
    try:
        if module.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("CUDA device unavailable")
    except module.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA runtime unavailable")
    ext = nvcomp_batch._try_get_cpp_ext()
    if ext is None or not hasattr(ext, "batch_deflate_decompress"):
        pytest.skip("native DEFLATE extension unavailable")
    return module


@pytest.mark.parametrize("pooled", [False, True])
@pytest.mark.parametrize("preparsed", [False, True])
def test_native_gzip_decodes_optional_headers(cp, pooled, preparsed):
    payloads, originals, header_sizes = [], [], []
    for flags in range(32):
        original = bytes(range(256)) * 9 + bytes([flags]) * 11
        payload = _gzip_with_flags(original, flags)
        assert gzip.decompress(payload) == original
        payloads.append(payload)
        originals.append(original)
        header_sizes.append(nvcomp_batch._parse_gzip_header_len(payload))
    lengths = np.array([len(payload) for payload in payloads], dtype="i8")
    offsets = np.cumsum(lengths) - lengths
    sizes = np.array([len(value) for value in originals], dtype="i8")
    # Dense tile starts deliberately do not obey raw DEFLATE's alignment.
    assert np.any(offsets % 4 != 0)
    stream = cp.cuda.Stream(non_blocking=True)
    owners = []
    with stream:
        data = cp.asarray(np.frombuffer(b"".join(payloads), dtype="u1"))
        result, out_offsets = nvcomp_batch.gpu_gzip_decompress_batch(
            data,
            offsets,
            lengths,
            sizes,
            use_cpp_helper=True,
            use_native_pool=pooled,
            keepalive=owners,
            header_sizes=header_sizes if preparsed else None,
        )
    stream.synchronize()
    np.testing.assert_array_equal(out_offsets, np.cumsum(sizes) - sizes)
    np.testing.assert_array_equal(
        cp.asnumpy(result), np.frombuffer(b"".join(originals), dtype="u1")
    )


def _gzip_with_flags(data, flags):
    header = b"\x1f\x8b\x08" + bytes([flags]) + bytes(6)
    if flags & 4:
        extra = b"extra\x00payload"
        header += struct.pack("<H", len(extra)) + extra
    if flags & 8:
        header += b"tile.fits\x00"
    if flags & 16:
        header += b"x" * (70000 if flags == 31 else 5) + b"\x00"
    if flags & 2:
        header += struct.pack("<H", zlib.crc32(header) & 0xFFFF)
    return (
        header
        + zlib.compress(data, wbits=-15)
        + struct.pack("<II", zlib.crc32(data), len(data))
    )


@pytest.mark.parametrize("wrapped", [False, True])
def test_deflate_alignment_without_keepalive(cp, wrapped):
    originals = [bytes(range(250)) * 4, b"second payload" * 79]
    payloads = [
        gzip.compress(value) if wrapped else zlib.compress(value, wbits=-15)
        for value in originals
    ]
    lengths = np.array([len(payload) for payload in payloads], dtype="i8")
    offsets = np.cumsum(lengths) - lengths
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        # The view also has an unaligned base pointer, independently of tiles.
        allocation = cp.asarray(
            np.frombuffer(b"x" + b"".join(payloads), dtype="u1")
        )
        data = allocation[1:]
        result, _ = nvcomp_batch.gpu_gzip_decompress_batch(
            data,
            offsets,
            lengths,
            [len(value) for value in originals],
            gzip_wrapped=wrapped,
            use_cpp_helper=True,
        )
    stream.synchronize()
    np.testing.assert_array_equal(
        cp.asnumpy(result), np.frombuffer(b"".join(originals), dtype="u1")
    )


@pytest.mark.parametrize("native", [False, True])
def test_deflate_decoder_on_nondefault_stream(cp, monkeypatch, native):
    extension = nvcomp_batch._try_get_cpp_ext()
    monkeypatch.setattr(
        nvcomp_batch,
        "_try_get_cpp_ext",
        lambda: (
            SimpleNamespace(
                batch_deflate_decompress=extension.batch_deflate_decompress
            )
            if native
            else None
        ),
    )
    original = b"legacy decoder tile" * 123
    payload = gzip.compress(original)
    if not native:
        # Warm the codec so stream ordering cannot rely on setup latency.
        nvcomp_batch._get_codec()
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        data = cp.asarray(np.frombuffer(payload, dtype="u1"))
        result, _ = nvcomp_batch.gpu_gzip_decompress_batch(
            data,
            [0],
            [len(payload)],
            [len(original)],
            use_cpp_helper=native,
            header_sizes=[10],
        )
    stream.synchronize()
    np.testing.assert_array_equal(
        cp.asnumpy(result), np.frombuffer(original, dtype="u1")
    )


@pytest.mark.parametrize("pooled", [False, True])
def test_native_deflate_rejects_unaligned_input(cp, pooled):
    ext = nvcomp_batch._try_get_cpp_ext()
    decode = (
        ext.batch_deflate_decompress_pooled
        if pooled
        else ext.batch_deflate_decompress
    )
    data = cp.empty(32, dtype=cp.uint8)
    output = cp.empty(32, dtype=cp.uint8)

    def table(value):
        return np.array([value], dtype=np.int64)

    with pytest.raises(ValueError, match="4-byte aligned"):
        decode(
            int(data.data.ptr) + 1,
            table(0),
            table(16),
            int(output.data.ptr),
            table(0),
            table(32),
        )


@pytest.mark.parametrize("pooled", [False, True])
@pytest.mark.parametrize("size_delta", [16, -16])
@pytest.mark.parametrize("gzip_wrapped", [False, True])
@pytest.mark.parametrize("deferred", [False, True])
def test_native_decode_rejects_incorrect_expected_size(
    cp, pooled, size_delta, gzip_wrapped, deferred
):
    ext = nvcomp_batch._try_get_cpp_ext()
    if deferred:
        submit = (
            ext.batch_gzip_decompress_deferred
            if gzip_wrapped
            else ext.batch_deflate_decompress_deferred
        )

        def decode(*args):
            completion = submit(
                *args, use_native_pool=pooled, decompression_backend="cuda"
            )
            completion.check()

    elif gzip_wrapped:

        def decode(*args):
            return ext.batch_gzip_decompress(
                *args, use_native_pool=pooled, decompression_backend="cuda"
            )
    else:
        decode = (
            ext.batch_deflate_decompress_pooled
            if pooled
            else ext.batch_deflate_decompress
        )
    original = bytes(range(64)) * 4
    # Both tiles use valid streams and full-sized backing allocations.
    # Balance the declared capacities so the scratch-size bounds still cover
    # the actual total and maximum sizes, including the overflow-status case.
    wbits = 31 if gzip_wrapped else -15
    payload = zlib.compress(original, wbits=wbits)
    assert zlib.decompress(payload, wbits=wbits) == original
    stream = cp.cuda.Stream(non_blocking=True)

    def table(value):
        return np.array([value, value], dtype=np.int64)

    with stream:
        data = cp.asarray(np.frombuffer(payload, dtype=np.uint8))
        stride = len(original) + 16
        output = cp.empty(2 * stride, dtype=cp.uint8)
        args = (
            int(data.data.ptr),
            table(0),
            table(len(payload)),
            int(output.data.ptr),
            np.array([0, stride], dtype=np.int64),
        )
        sizes = np.array(
            [len(original) + size_delta, len(original) - size_delta],
            dtype=np.int64,
        )
        message = (
            "decoded size differs for tile 0"
            if size_delta > 0
            else "decompression failed for tile 0"
        )
        with pytest.raises(RuntimeError, match=message):
            decode(*args, sizes, int(stream.ptr))
        # Completed size/status rejections must leave the stream usable.
        decode(*args, table(len(original)), int(stream.ptr))
    for offset in (0, stride):
        np.testing.assert_array_equal(
            cp.asnumpy(output[offset : offset + len(original)]),
            np.frombuffer(original, dtype=np.uint8),
        )


@pytest.mark.parametrize("pooled", [False, True])
@pytest.mark.parametrize("gzip_wrapped", [False, True])
def test_deferred_decode_owns_expected_sizes(cp, pooled, gzip_wrapped):
    ext = nvcomp_batch._try_get_cpp_ext()
    submit = (
        ext.batch_gzip_decompress_deferred
        if gzip_wrapped
        else ext.batch_deflate_decompress_deferred
    )
    original = bytes(range(64)) * 4
    payload = zlib.compress(original, wbits=31 if gzip_wrapped else -15)
    sizes = np.array([len(original)], dtype=np.int64)
    zero = np.array([0], dtype=np.int64)
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        data = cp.asarray(np.frombuffer(payload, dtype=np.uint8))
        output = cp.empty(len(original), dtype=cp.uint8)
        completion = submit(
            int(data.data.ptr),
            zero,
            np.array([len(payload)], dtype=np.int64),
            int(output.data.ptr),
            zero,
            sizes,
            int(stream.ptr),
            use_native_pool=pooled,
            decompression_backend="cuda",
        )
    sizes[:] = 0
    completion.check()
    completion.check()
    np.testing.assert_array_equal(
        cp.asnumpy(output), np.frombuffer(original, dtype=np.uint8)
    )


def test_deferred_decode_returns_before_stream_completion(cp):
    ext = nvcomp_batch._try_get_cpp_ext()
    original = b"asynchronous decode" * 32
    payload = gzip.compress(original)
    data = cp.asarray(np.frombuffer(payload, dtype=np.uint8))
    output = cp.empty(len(original), dtype=cp.uint8)
    zero = np.array([0], dtype=np.int64)
    sizes = np.array([len(original)], dtype=np.int64)
    lengths = np.array([len(payload)], dtype=np.int64)
    stream = cp.cuda.Stream(non_blocking=True)
    cp.cuda.get_current_stream().synchronize()

    def submit():
        return ext.batch_gzip_decompress_deferred(
            int(data.data.ptr),
            zero,
            lengths,
            int(output.data.ptr),
            zero,
            sizes,
            int(stream.ptr),
            decompression_backend="cuda",
        )

    # Warm both native pools before blocking this stream: a cold cudaMalloc
    # may synchronize even though decode submission itself is asynchronous.
    warmup = submit()
    warmup.check()
    del warmup
    # Use a bounded GPU delay, not a Python stream callback: a CUDA API can
    # wait for a callback while holding the GIL that its timeout needs.
    delay = cp.RawKernel(
        r"""
        extern "C" __global__ void delay(unsigned long long ns) {
            unsigned long long start, now;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
            do {
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
            } while (now - start < ns);
        }
        """,
        "delay",
    )
    with stream:
        delay((1,), (1,), (np.uint64(1),))
    stream.synchronize()
    delay_finished = cp.cuda.Event()
    with stream:
        delay((1,), (1,), (np.uint64(500_000_000),))
        delay_finished.record(stream)
    completion = submit()
    returned_early = not delay_finished.done
    completion.check()
    assert returned_early, "decode submission waited for the stream"
    np.testing.assert_array_equal(
        cp.asnumpy(output), np.frombuffer(original, dtype=np.uint8)
    )


def test_deferred_decode_accepts_empty_batch(cp):
    ext = nvcomp_batch._try_get_cpp_ext()
    empty = np.empty(0, dtype=np.int64)
    for submit in (
        ext.batch_gzip_decompress_deferred,
        ext.batch_deflate_decompress_deferred,
    ):
        completion = submit(0, empty, empty, 0, empty, empty)
        completion.check()
        assert completion.status_ptr == 0
        assert completion.actual_sizes_ptr == 0
        assert completion.expected_sizes_ptr == 0


def test_aligned_deflate_input_stays_zero_copy(cp):
    data = cp.empty(32, dtype=cp.uint8)
    offsets = np.array([0, 16], dtype=np.int64)
    packed, packed_offsets, owners = nvcomp_batch._align_deflate_inputs(
        data, offsets, np.array([13, 15], dtype=np.int64)
    )
    assert packed is data
    assert packed_offsets is offsets
    assert not owners


def test_python_deflate_rejects_incorrect_expected_size(cp):
    original = b"valid synthetic content" * 8
    payload = zlib.compress(original, wbits=-15)
    data = cp.asarray(np.frombuffer(payload, dtype=np.uint8))
    with pytest.raises(RuntimeError, match="decoded size differs for tile 0"):
        nvcomp_batch.gpu_gzip_decompress_batch(
            data,
            [0],
            [len(payload)],
            [len(original) + 16],
            gzip_wrapped=False,
            use_cpp_helper=False,
        )
