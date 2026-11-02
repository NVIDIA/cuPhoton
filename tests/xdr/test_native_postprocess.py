# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Ahead-of-time FITS restoration parity without CuPy kernel compilation."""

from __future__ import annotations

import zlib
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from cuphoton.xdr import nvcomp_batch
from cuphoton.xdr.reader import GpuCompImageReader
from tests.xdr.test_postprocess_gpu import _decoded_tiles
from tests.xdr.test_postprocess_gpu import cp as cp


@pytest.fixture
def native_cupy(cp):
    pytest.importorskip("cuphoton.core._imaging_ext")
    return cp


@contextmanager
def _forbid_cupy_compilation(monkeypatch):
    from cupy._core import core
    from cupy.cuda import compiler

    def forbidden(*_args, **_kwargs):
        pytest.fail("native FITS restoration attempted CuPy compilation")

    with monkeypatch.context() as patch:
        patch.setattr(core, "compile_with_cache", forbidden)
        patch.setattr(compiler, "_compile_module_with_cache", forbidden)
        patch.setattr(compiler.nvrtc, "compileProgram", forbidden)
        yield


def test_native_gzip_header_probe_handles_optional_fields(
    native_cupy, monkeypatch
):
    cp = native_cupy
    fixed = b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\xff"
    optional = (
        fixed[:3]
        + b"\x1e"
        + fixed[4:]
        + b"\x03\x00abc"
        + b"tile-name\x00"
        + b"comment" * 39
        + b"\x00"
        + b"\x00\x00"
    )
    headers = [fixed, optional, fixed, optional]
    tiles = [header + b"\x03\x00" + b"\x00" * 8 for header in headers]
    # Exercise disjoint tiles, growing optional-header reads, and bounded
    # groups that cannot contain both long headers at once.
    offsets = np.cumsum([3] + [len(tile) + 5 for tile in tiles[:-1]])
    raw = np.frombuffer(b"pre" + b"gap!!".join(tiles), dtype=np.uint8)
    lengths = np.asarray([len(tile) for tile in tiles], dtype=np.int64)
    monkeypatch.setattr(nvcomp_batch, "_MAX_GZIP_PROBE_GATHER_BYTES", 384)
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        device_raw = cp.asarray(raw)
        with _forbid_cupy_compilation(monkeypatch):
            sizes = nvcomp_batch._compute_header_sizes_from_device(
                device_raw, offsets, lengths, native_preprocessing=True
            )
    np.testing.assert_array_equal(sizes, [len(header) for header in headers])
    np.testing.assert_array_equal(cp.asnumpy(device_raw), raw)


def test_native_deflate_alignment_preserves_payloads(
    native_cupy, monkeypatch
):
    cp = native_cupy
    originals = [b"first payload" * 23, bytes(range(256)) * 3]
    payloads = [zlib.compress(value, wbits=-15) for value in originals]
    offsets = np.asarray([1, len(payloads[0]) + 2], dtype=np.int64)
    raw = np.frombuffer(b"!" + b"!".join(payloads), dtype=np.uint8)
    lengths = np.asarray([len(value) for value in payloads], dtype=np.int64)
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        device_raw = cp.asarray(raw)
        with _forbid_cupy_compilation(monkeypatch):
            packed, packed_offsets, owners = (
                nvcomp_batch._align_deflate_inputs(
                    device_raw, offsets, lengths, native_preprocessing=True
                )
            )
            stream.synchronize()
    assert owners[0] is device_raw
    assert np.all((packed.data.ptr + packed_offsets) % 4 == 0)
    packed_host = cp.asnumpy(packed)
    for offset, payload, original in zip(
        packed_offsets, payloads, originals, strict=True
    ):
        actual = packed_host[offset : offset + len(payload)].tobytes()
        assert actual == payload
        assert zlib.decompress(actual, wbits=-15) == original
    np.testing.assert_array_equal(cp.asnumpy(device_raw), raw)


@pytest.mark.parametrize("dtype", ["u1", "i2", "i4", "i8", "f4", "f8"])
@pytest.mark.parametrize("compression", ["GZIP_1", "GZIP_2"])
def test_native_restoration_matches_fused_for_cropped_tiles(
    native_cupy, monkeypatch, dtype, compression
):
    cp = native_cupy
    rng = np.random.default_rng(42)
    pixels = rng.bytes(17 * 19 * np.dtype(dtype).itemsize)
    image = np.frombuffer(pixels, dtype=dtype).reshape(17, 19)
    # Include partial first tiles, the final row/column, and tile boundaries.
    section = np.s_[3:17, 5:19]
    raw, offsets, plan = _decoded_tiles(image, compression, section)
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        device_raw = cp.asarray(raw)
        out = cp.empty(plan["out_shape"], dtype=dtype)
    with _forbid_cupy_compilation(monkeypatch):
        actual = GpuCompImageReader.postprocess_decoded_tiles(
            device_raw,
            offsets,
            plan,
            out=out,
            stream=stream,
            postprocess="native",
        )
        stream.synchronize()
    expected = GpuCompImageReader.postprocess_decoded_tiles(
        device_raw, offsets, plan, stream=stream, postprocess="fused"
    )
    stream.synchronize()

    assert actual is out
    actual_bytes = cp.asnumpy(actual).view("u1")
    np.testing.assert_array_equal(
        actual_bytes, cp.asnumpy(expected).view("u1")
    )
    np.testing.assert_array_equal(
        actual_bytes, image[section].copy().view("u1")
    )
    np.testing.assert_array_equal(cp.asnumpy(device_raw), raw)


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("compression", ["GZIP_1", "GZIP_2"])
def test_native_restoration_preserves_decode_guards(
    native_cupy, monkeypatch, quantized, compression
):
    cp = native_cupy
    image = np.arange(-150, 173, dtype="i4").reshape(17, 19)
    raw, offsets, plan = _decoded_tiles(image, compression, np.s_[3:17, 5:19])
    count = len(offsets)
    if quantized:
        plan.update(
            quantized=True,
            dtype_char="f4",
            zbitpix=-32,
            sel_zscale=np.linspace(0.1234567, 0.7654321, count),
            sel_zzero=np.linspace(-20.1234567, 80.7654321, count),
        )
    # The leading failed tile belongs to an earlier HDU. Within this HDU,
    # one tile reports decode failure and another has the wrong byte count.
    status = np.zeros(count + 1, dtype="i4")
    status[[0, 2]] = 1
    sizes = np.asarray(np.r_[0, plan["out_bytes"]], dtype="u8")
    actual_sizes = sizes.copy()
    actual_sizes[3] += 1
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        device_raw = cp.asarray(raw)
        device_status = cp.asarray(status)
        device_sizes = cp.asarray(sizes)
        device_actual_sizes = cp.asarray(actual_sizes)
        completion = SimpleNamespace(
            status_ptr=device_status.data.ptr,
            actual_sizes_ptr=device_actual_sizes.data.ptr,
            expected_sizes_ptr=device_sizes.data.ptr,
        )
        initial = np.full(plan["out_shape"], -999, dtype=plan["dtype_char"])
        expected = cp.asarray(initial)
        out = cp.asarray(initial)
    with _forbid_cupy_compilation(monkeypatch):
        GpuCompImageReader.postprocess_decoded_tiles(
            device_raw,
            offsets,
            plan,
            out=out,
            stream=stream,
            postprocess="native",
            _decode_status=completion,
            _decode_tile_start=1,
        )
        stream.synchronize()
    GpuCompImageReader.postprocess_decoded_tiles(
        device_raw,
        offsets,
        plan,
        out=expected,
        stream=stream,
        postprocess="fused",
        _decode_status=completion,
        _decode_tile_start=1,
    )
    stream.synchronize()

    actual = cp.asnumpy(out)
    np.testing.assert_array_equal(
        actual.view("u1"), cp.asnumpy(expected).view("u1")
    )
    for tile in (1, 2):
        row, col = plan["origins_r"][tile], plan["origins_c"][tile]
        height, width = plan["heights"][tile], plan["widths"][tile]
        assert np.all(actual[row : row + height, col : col + width] == -999)
    assert np.any(actual != -999)
    np.testing.assert_array_equal(cp.asnumpy(device_raw), raw)
