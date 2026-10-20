# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Native gzip tile decoding with standard and optional RFC 1952 headers."""

from __future__ import annotations

import gzip
import struct
import weakref
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
    if ext is None or not hasattr(ext, "batch_gzip_decompress"):
        pytest.skip("native gzip extension unavailable")
    return module


@pytest.mark.parametrize("backend", ["auto", "cuda"])
@pytest.mark.parametrize("decoder", ["auto", "gzip", "deflate"])
@pytest.mark.parametrize("pooled", [False, True])
@pytest.mark.parametrize("preparsed", [False, True])
def test_native_gzip_decodes_optional_headers(
    cp, pooled, preparsed, decoder, backend
):
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
            gzip_decoder=decoder,
            decompression_backend=backend,
            use_native_pool=pooled,
            keepalive=owners,
            header_sizes=header_sizes if preparsed else None,
        )
    stream.synchronize()
    np.testing.assert_array_equal(out_offsets, np.cumsum(sizes) - sizes)
    np.testing.assert_array_equal(
        cp.asnumpy(result), np.frombuffer(b"".join(originals), dtype="u1")
    )


@pytest.mark.parametrize(
    "header,error",
    [
        (b"\x00\x8b\x08\x00" + bytes(6), "wrong magic"),
        (b"\x1f\x8b\x09\x00" + bytes(6), "DEFLATE compression"),
        (b"\x1f\x8b\x08\x80" + bytes(6), "reserved header flags"),
        (b"\x1f\x8b\x08\x08" + bytes(6) + b"name", "truncated gzip"),
    ],
)
def test_native_gzip_rejects_invalid_headers_before_decode(cp, header, error):
    # Ten final bytes reserve the two-byte payload and eight-byte trailer.
    data = cp.asarray(np.frombuffer(header + bytes(10), dtype="u1"))
    with pytest.raises(ValueError, match=error):
        nvcomp_batch.gpu_gzip_decompress_batch(
            data, [0], [data.size], [16], use_cpp_helper=True
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
            gzip_decoder="deflate",
            use_cpp_helper=True,
        )
    input_owner = weakref.ref(allocation)
    del data, allocation
    assert input_owner() is not None
    cp.get_default_memory_pool().free_all_blocks()
    stream.synchronize()
    np.testing.assert_array_equal(
        cp.asnumpy(result), np.frombuffer(b"".join(originals), dtype="u1")
    )
    del result
    assert input_owner() is None


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("non_blocking", [False, True])
def test_auto_decoder_orders_legacy_deflate(
    cp, monkeypatch, native, non_blocking
):
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
    payload = gzip.compress(original, compresslevel=0)
    stale = gzip.compress(bytes(len(original)), compresslevel=0)
    assert len(payload) == len(stale)
    delayed_copy = cp.RawKernel(
        r"""
        extern "C" __global__ void delayed_copy(
            const unsigned char* src, unsigned char* dst, long long size) {
            unsigned long long start, now;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
            do {
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
            } while (now - start < 150000000ULL);
            for (long long i = threadIdx.x; i < size; i += blockDim.x) {
                dst[i] = src[i];
            }
        }
        """,
        "delayed_copy",
    )
    delayed_copy.compile()
    staged = cp.asarray(np.frombuffer(payload, dtype="u1"))
    # Keep the stripped payload aligned and valid even before the producer
    # runs, so a stream-ordering regression returns stale pixels safely.
    allocation = cp.asarray(np.frombuffer(b"xx" + stale, dtype="u1"))
    data = allocation[2:]
    cp.cuda.get_current_stream().synchronize()
    stream = cp.cuda.Stream(non_blocking=non_blocking)
    with stream:
        if not native:
            nvcomp_batch._get_codec()
        delayed_copy((1,), (256,), (staged, data, np.int64(len(payload))))
        result, _ = nvcomp_batch.gpu_gzip_decompress_batch(
            data,
            [0],
            [len(payload)],
            [len(original)],
            use_cpp_helper=native,
            gzip_decoder="auto",
            header_sizes=[10],
        )
    stream.synchronize()
    np.testing.assert_array_equal(
        cp.asnumpy(result), np.frombuffer(original, dtype="u1")
    )


@pytest.mark.parametrize(
    "entrypoint",
    [
        "batch_gzip_decompress",
        "batch_deflate_decompress",
        "batch_deflate_decompress_pooled",
    ],
)
def test_native_decoder_rejects_invalid_backend(cp, entrypoint):
    extension = nvcomp_batch._try_get_cpp_ext()
    empty = np.empty(0, dtype="i8")
    with pytest.raises(ValueError, match="decompression_backend must be"):
        getattr(extension, entrypoint)(
            0, empty, empty, 0, empty, empty, decompression_backend="invalid"
        )


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("decoder", ["gzip", "deflate"])
def test_batch_decode_failure_skips_pixels_and_raises(
    cp, tmp_path, monkeypatch, native, decoder
):
    import kvikio
    import kvikio.defaults
    from astropy.io import fits

    from cuphoton.xdr import batch_to_device_stream
    from cuphoton.xdr.reader import GpuCompImageReader

    original = np.arange(64 * 64, dtype=np.int32).reshape(64, 64)
    path = tmp_path / "valid-tiles.fits"
    fits.HDUList(
        [fits.PrimaryHDU()]
        + [
            fits.CompImageHDU(
                original + index,
                compression_type="GZIP_2",
                tile_shape=original.shape,
            )
            for index in range(2)
        ]
    ).writeto(path)
    inflate = GpuCompImageReader.inflate_device_heap
    checks = []

    def mismatched_size(*args, out_bytes, **kwargs):
        # Keep the compressed streams valid and overallocate the final tile.
        # Earlier tiles/HDU offsets remain unchanged, and the final tile must
        # be rejected using the decoder's actual-size result.
        out_bytes = out_bytes.copy()
        out_bytes[-1] += 4
        result = inflate(*args, out_bytes=out_bytes, **kwargs)
        checks.extend(kwargs["_decode_checks"])
        return result

    monkeypatch.setattr(
        GpuCompImageReader,
        "inflate_device_heap",
        staticmethod(mismatched_size),
    )
    outputs = [cp.full((2, 64, 64), -999, dtype=cp.int32) for _ in range(2)]
    cp.cuda.get_current_stream().synchronize()
    stream = cp.cuda.Stream(non_blocking=True)
    with kvikio.defaults.set("compat_mode", kvikio.CompatMode.ON):
        with pytest.raises(
            RuntimeError, match="decoded size differs for tile 3"
        ):
            batch_to_device_stream(
                [path, path],
                hdu_indices=[1, 2],
                out=outputs,
                stream=stream,
                native_batcher=native,
                decode_batch_files=2,
                gzip_decoder=decoder,
                decompression_backend="cuda",
            )
    assert len(checks) == 1
    assert stream.done
    np.testing.assert_array_equal(cp.asnumpy(outputs[0][0]), original)
    np.testing.assert_array_equal(cp.asnumpy(outputs[0][1]), original)
    np.testing.assert_array_equal(cp.asnumpy(outputs[1][0]), original + 1)
    np.testing.assert_array_equal(cp.asnumpy(outputs[1][1]), -999)
