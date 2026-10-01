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


def test_aligned_deflate_input_stays_zero_copy(cp):
    data = cp.empty(32, dtype=cp.uint8)
    offsets = np.array([0, 16], dtype=np.int64)
    packed, packed_offsets, owners = nvcomp_batch._align_deflate_inputs(
        data, offsets, np.array([13, 15], dtype=np.int64)
    )
    assert packed is data
    assert packed_offsets is offsets
    assert not owners
