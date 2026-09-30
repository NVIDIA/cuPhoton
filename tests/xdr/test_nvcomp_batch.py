# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import threading
import weakref
from types import SimpleNamespace

import numpy as np
import pytest

from cuphoton.xdr import nvcomp_batch


def test_python_codec_cache_tracks_stream_device_and_thread(monkeypatch):
    events, created = [], []
    state = SimpleNamespace(device=0)

    class Stream:
        def __init__(self, ptr):
            self.ptr = ptr

        def __del__(self):
            events.append(("stream", self.ptr))

    class Codec:
        def __init__(self, *, algorithm, bitstream_kind, cuda_stream):
            assert algorithm == "deflate" and bitstream_kind == "raw"
            self.stream = weakref.ref(state.stream)
            self.ptr = cuda_stream
            created.append((cuda_stream, state.device))

        def __del__(self):
            events.append(("codec", self.ptr, self.stream() is not None))

    fake_nvcomp = SimpleNamespace(
        Codec=Codec, BitstreamKind=SimpleNamespace(RAW="raw")
    )
    monkeypatch.setitem(
        sys.modules, "nvidia", SimpleNamespace(nvcomp=fake_nvcomp)
    )
    monkeypatch.setitem(sys.modules, "nvidia.nvcomp", fake_nvcomp)
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(
            cuda=SimpleNamespace(
                get_current_stream=lambda: state.stream,
                runtime=SimpleNamespace(getDevice=lambda: state.device),
            )
        ),
    )
    monkeypatch.setattr(
        nvcomp_batch, "_DEFLATE_CODEC_CACHE", threading.local()
    )
    state.stream = Stream(11)
    assert nvcomp_batch._get_codec() is nvcomp_batch._get_codec()
    old_stream = weakref.ref(state.stream)
    state.stream = Stream(22)
    assert old_stream() is not None
    nvcomp_batch._get_codec()
    assert events[:2] == [("codec", 11, True), ("stream", 11)]
    state.device = 1
    nvcomp_batch._get_codec()
    thread = threading.Thread(target=nvcomp_batch._get_codec)
    thread.start()
    thread.join()
    assert created == [(11, 0), (22, 0), (22, 1), (22, 1)]
    assert all(event[2] for event in events if event[0] == "codec")


@pytest.mark.parametrize("aligned", [False, True])
def test_deflate_repack_uses_int64_tables_and_retains_inputs(
    monkeypatch, aligned
):
    data = SimpleNamespace(data=SimpleNamespace(ptr=100))
    packed = SimpleNamespace()
    calls = []
    offsets = np.array(
        [0, 8, 8, 24] if aligned else [1, 7, 7, 30],
        dtype=np.int64 if aligned else np.int32,
    )
    lengths = np.array([5, 0, 13, 3], dtype=np.int32)

    def allocate(size, dtype):
        assert size == 28 and dtype == np.uint8
        return packed

    monkeypatch.setattr(nvcomp_batch, "_REPACK_DEFLATE_KERNEL", None)
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(
            RawKernel=lambda *a: lambda *args: calls.append(args),
            empty=allocate,
            asarray=np.asarray,
            uint8=np.uint8,
        ),
    )
    result, out_offsets, owners = nvcomp_batch._align_deflate_inputs(
        data, offsets, lengths
    )
    if aligned:
        assert result is data and out_offsets is offsets and owners == ()
        assert not calls
    else:
        assert result is packed
        np.testing.assert_array_equal(out_offsets, [0, 8, 8, 24])
        grid, block, args = calls[0]
        assert grid == (4,) and block == (256,)
        assert args[0] is data and args[-1] is packed
        for actual, expected in zip(
            args[1:-1], (offsets, lengths, out_offsets), strict=True
        ):
            assert actual.dtype == np.int64
            np.testing.assert_array_equal(actual, expected)
        assert len(owners) == 5
        assert owners[0] is data and owners[1] is packed
        assert all(
            a is b for a, b in zip(owners[2:], args[1:-1], strict=True)
        )


@pytest.mark.parametrize("flag", [0x08, 0x10])
def test_parse_gzip_header_rejects_unterminated_optional_string(flag):
    head = bytearray(16)
    head[0:2] = b"\x1f\x8b"
    head[2] = 0x08
    head[3] = flag
    head[10:] = b"abcdef"

    with pytest.raises(ValueError, match="truncated gzip header"):
        nvcomp_batch._parse_gzip_header_len(bytes(head))


@pytest.mark.parametrize(
    ("header", "match"),
    [
        (b"\x1f\x8b\x00\x00" + b"\x00" * 6, "DEFLATE compression"),
        (b"\x1f\x8b\x08\x20" + b"\x00" * 6, "reserved header flags"),
    ],
)
def test_parse_gzip_header_rejects_invalid_fixed_fields(header, match):
    with pytest.raises(ValueError, match=match):
        nvcomp_batch._parse_gzip_header_len(header)


@pytest.mark.parametrize(
    ("buffer_size", "offset", "length", "match"),
    [
        (15, 0, 15, "too short to contain a header"),
        (20, -1, 20, "must be nonnegative"),
        (20, 10, 20, "exceeds containing buffer"),
    ],
)
def test_header_probe_rejects_tile_before_device_gather(
    buffer_size, offset, length, match
):
    class DeviceBuffer:
        size = buffer_size

        def __getitem__(self, key):
            raise AssertionError("device gather must not run")

    with pytest.raises(ValueError, match=match):
        nvcomp_batch._compute_header_sizes_from_device(
            DeviceBuffer(),
            np.array([offset], dtype=np.int64),
            np.array([length], dtype=np.int64),
        )


@pytest.mark.parametrize(
    ("rel_offsets", "lengths", "match"),
    [
        ([0.5], [20], "tile offsets.*integer-valued"),
        ([0], [20.5], "tile lengths.*integer-valued"),
        ([np.nan], [20], "tile offsets.*integer-valued"),
        ([0], [np.inf], "tile lengths.*integer-valued"),
        ([True], [20], "tile offsets.*integer-valued"),
        (
            np.array([1 << 63], dtype=np.uint64),
            [20],
            "tile offsets.*int64-representable",
        ),
    ],
)
def test_header_probe_rejects_nonintegral_tile_metadata(
    rel_offsets, lengths, match
):
    class DeviceBuffer:
        size = 20

        def __getitem__(self, key):
            pytest.fail("invalid metadata must be rejected before a gather")

    with pytest.raises(ValueError, match=match):
        nvcomp_batch._compute_header_sizes_from_device(
            DeviceBuffer(),
            rel_offsets,
            lengths,
        )


@pytest.mark.parametrize(
    ("dtype", "consecutive_limit"),
    [
        (np.float16, 1 << 11),
        (np.float32, 1 << 24),
        (np.float64, 1 << 53),
    ],
)
def test_integer_metadata_enforces_float_precision(dtype, consecutive_limit):
    accepted = nvcomp_batch._integer_values_as_int64(
        np.array([consecutive_limit - 1], dtype=dtype), "metadata"
    )
    np.testing.assert_array_equal(accepted, [consecutive_limit - 1])

    with pytest.raises(ValueError, match="consecutive integer precision"):
        nvcomp_batch._integer_values_as_int64(
            np.array([consecutive_limit + 1], dtype=dtype), "metadata"
        )


def test_device_header_probe_accepts_empty_tile_table():
    class DeviceBuffer:
        size = 0

        def __getitem__(self, key):
            pytest.fail("empty header probe must not gather device bytes")

    sizes = nvcomp_batch._compute_header_sizes_from_device(
        DeviceBuffer(),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
    )

    assert sizes.dtype == np.dtype(np.int64)
    assert sizes.shape == (0,)


def test_device_header_probe_rejects_too_small_gather_budget(monkeypatch):
    class DeviceBuffer:
        size = 20

        def __getitem__(self, key):
            raise AssertionError("device gather must not run")

    monkeypatch.setattr(nvcomp_batch, "_MAX_GZIP_PROBE_GATHER_BYTES", 9)
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace())

    with pytest.raises(RuntimeError, match="aggregate byte budget"):
        nvcomp_batch._compute_header_sizes_from_device(
            DeviceBuffer(),
            np.array([0], dtype=np.int64),
            np.array([20], dtype=np.int64),
        )


def test_device_header_probe_extends_variable_headers(monkeypatch):
    fixed = b"\x1f\x8b\x08\x00" + b"\x00" * 4 + b"\x00\xff"
    named_short = (
        b"\x1f\x8b\x08\x08"
        + b"\x00" * 4
        + b"\x00\xff"
        + b"short-device-header"
        + b"\x00"
    )
    named_long = (
        b"\x1f\x8b\x08\x08"
        + b"\x00" * 4
        + b"\x00\xff"
        + b"long-device-header" * 8
        + b"\x00"
    )
    tiles = [
        fixed + b"\x03\x00" + b"\x00" * 8,
        named_short + b"\x03\x00" + b"\x00" * 8,
        named_long + b"\x03\x00" + b"\x00" * 8,
    ]
    data = np.frombuffer(b"".join(tiles), dtype=np.uint8)
    copied_sizes = []

    class DeviceBuffer:
        size = data.size

        def __getitem__(self, key):
            return data[key]

    def asnumpy(values):
        result = np.asarray(values)
        copied_sizes.append(result.size)
        return result

    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(asarray=np.asarray, asnumpy=asnumpy),
    )

    sizes = nvcomp_batch._compute_header_sizes_from_device(
        DeviceBuffer(),
        np.array(
            [0, len(tiles[0]), len(tiles[0]) + len(tiles[1])],
            dtype=np.int64,
        ),
        np.array([len(tile) for tile in tiles], dtype=np.int64),
    )

    np.testing.assert_array_equal(
        sizes, [10, len(named_short), len(named_long)]
    )
    assert copied_sizes == [30, len(named_short) + 64, len(named_long)]


def test_device_header_probe_bounds_each_gather(monkeypatch):
    headers = [
        b"\x1f\x8b\x08\x08"
        + b"\x00" * 4
        + b"\x00\xff"
        + b"n" * name_length
        + b"\x00"
        for name_length in (20, 21, 22, 23)
    ]
    tiles = [header + b"\x03\x00" + b"\x00" * 8 for header in headers]
    data = np.frombuffer(b"".join(tiles), dtype=np.uint8)
    copied_sizes = []

    class DeviceBuffer:
        size = data.size

        def __getitem__(self, key):
            return data[key]

    def asnumpy(values):
        result = np.asarray(values)
        copied_sizes.append(result.size)
        return result

    monkeypatch.setattr(nvcomp_batch, "_MAX_GZIP_HEADER_BYTES", 64)
    monkeypatch.setattr(nvcomp_batch, "_MAX_GZIP_PROBE_GATHER_BYTES", 64)
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(asarray=np.asarray, asnumpy=asnumpy),
    )

    offsets = np.cumsum([0, *(len(tile) for tile in tiles[:-1])])
    sizes = nvcomp_batch._compute_header_sizes_from_device(
        DeviceBuffer(),
        offsets,
        np.array([len(tile) for tile in tiles], dtype=np.int64),
    )

    np.testing.assert_array_equal(sizes, [len(header) for header in headers])
    assert copied_sizes == [40, 63, 33, 34]
    assert max(copied_sizes) <= 64


def test_device_fixed_header_probe_spans_bounded_groups(monkeypatch):
    header = b"\x1f\x8b\x08\x00" + b"\x00" * 4 + b"\x00\xff"
    tiles = [header + b"\x03\x00" + b"\x00" * 8 for _ in range(3)]
    data = np.frombuffer(b"".join(tiles), dtype=np.uint8)
    copied_sizes = []

    class DeviceBuffer:
        size = data.size

        def __getitem__(self, key):
            return data[key]

    def asnumpy(values):
        result = np.asarray(values)
        copied_sizes.append(result.size)
        return result

    monkeypatch.setattr(nvcomp_batch, "_MAX_GZIP_PROBE_GATHER_BYTES", 20)
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(asarray=np.asarray, asnumpy=asnumpy),
    )

    sizes = nvcomp_batch._compute_header_sizes_from_device(
        DeviceBuffer(),
        np.array([0, 20, 40], dtype=np.int64),
        np.full(3, 20, dtype=np.int64),
    )

    np.testing.assert_array_equal(sizes, [10, 10, 10])
    assert copied_sizes == [20, 10]


def test_device_header_probe_rejects_header_over_safety_limit(monkeypatch):
    header = (
        b"\x1f\x8b\x08\x08"
        + b"\x00" * 4
        + b"\x00\xff"
        + b"long-name" * 8
        + b"\x00"
    )
    tile = header + b"\x03\x00" + b"\x00" * 8
    data = np.frombuffer(tile, dtype=np.uint8)
    copied_sizes = []

    class DeviceBuffer:
        size = data.size

        def __getitem__(self, key):
            return data[key]

    def asnumpy(values):
        result = np.asarray(values)
        copied_sizes.append(result.size)
        return result

    monkeypatch.setattr(nvcomp_batch, "_MAX_GZIP_HEADER_BYTES", 32)
    monkeypatch.setattr(nvcomp_batch, "_MAX_GZIP_PROBE_GATHER_BYTES", 32)
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(asarray=np.asarray, asnumpy=asnumpy),
    )

    with pytest.raises(ValueError, match="exceeds 32-byte safety limit"):
        nvcomp_batch._compute_header_sizes_from_device(
            DeviceBuffer(),
            np.array([0], dtype=np.int64),
            np.array([len(tile)], dtype=np.int64),
        )

    assert copied_sizes == [10, 32]


def test_device_header_probe_stops_before_payload_and_trailer(monkeypatch):
    unterminated = (
        b"\x1f\x8b\x08\x08"
        + b"\x00" * 4
        + b"\x00\xff"
        + b"unterminated-name"
        + b"\x03\x01"
        + b"\x00" * 8
    )
    data = np.frombuffer(unterminated, dtype=np.uint8)
    copied_sizes = []

    class DeviceBuffer:
        size = data.size

        def __getitem__(self, key):
            return data[key]

    def asnumpy(values):
        result = np.asarray(values)
        copied_sizes.append(result.size)
        return result

    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(asarray=np.asarray, asnumpy=asnumpy),
    )

    with pytest.raises(ValueError, match="truncated gzip header"):
        nvcomp_batch._compute_header_sizes_from_device(
            DeviceBuffer(),
            np.array([0], dtype=np.int64),
            np.array([len(unterminated)], dtype=np.int64),
        )

    assert copied_sizes == [10, len(unterminated) - 10]


def test_validate_gzip_header_sizes_accepts_tile_bounded_values():
    values = nvcomp_batch._validate_gzip_header_sizes(
        [10.0, 12.0, float(nvcomp_batch._MAX_GZIP_HEADER_BYTES)],
        np.array(
            [20, 22, nvcomp_batch._MAX_GZIP_HEADER_BYTES + 10],
            dtype=np.int64,
        ),
    )

    np.testing.assert_array_equal(
        values,
        np.array([10, 12, nvcomp_batch._MAX_GZIP_HEADER_BYTES]),
    )
    assert values.dtype == np.dtype(np.int64)


@pytest.mark.parametrize(
    ("header_sizes", "lengths", "match"),
    [
        ([[10]], [20], "1D array"),
        ([10], [20, 20], "one entry per gzip tile"),
        ([-1], [20], "at least 10 bytes"),
        ([9], [20], "at least 10 bytes"),
        ([11], [20], "fewer than 2 DEFLATE payload bytes"),
        ([10], [19], "too short to contain a header"),
        ([10.5], [20], "integer-valued numbers"),
        ([np.nan], [20], "integer-valued numbers"),
        ([np.inf], [20], "integer-valued numbers"),
        ([True], [20], "integer-valued numbers"),
        (["10"], [20], "integer-valued numbers"),
        (
            [nvcomp_batch._MAX_GZIP_HEADER_BYTES + 1],
            [nvcomp_batch._MAX_GZIP_HEADER_BYTES + 11],
            "safety limit",
        ),
        (
            np.array([1 << 63], dtype=np.uint64),
            [20],
            "int64-representable values",
        ),
    ],
)
def test_validate_gzip_header_sizes_rejects_invalid_metadata(
    header_sizes, lengths, match
):
    with pytest.raises(ValueError, match=match):
        nvcomp_batch._validate_gzip_header_sizes(
            header_sizes,
            np.asarray(lengths, dtype=np.int64),
        )


def test_checked_output_offsets_support_empty_and_zero_length_tiles():
    offsets, total = nvcomp_batch._checked_output_offsets(
        np.array([3, 0, 4], dtype=np.int64)
    )
    empty_offsets, empty_total = nvcomp_batch._checked_output_offsets(
        np.empty(0, dtype=np.int64)
    )

    np.testing.assert_array_equal(offsets, [0, 3, 3])
    assert total == 7
    assert empty_offsets.dtype == np.dtype(np.int64)
    assert empty_offsets.shape == (0,)
    assert empty_total == 0


def test_checked_output_offsets_rejects_aggregate_int64_overflow():
    max_int64 = np.iinfo(np.int64).max
    with pytest.raises(ValueError, match="aggregate uncompressed size"):
        nvcomp_batch._checked_output_offsets(
            np.array([max_int64, 1], dtype=np.int64)
        )


def test_gpu_batch_accepts_empty_tile_table(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(empty=np.empty, uint8=np.uint8),
    )

    output, offsets = nvcomp_batch.gpu_gzip_decompress_batch(
        SimpleNamespace(size=0),
        [],
        [],
        [],
        gzip_wrapped=False,
        use_cpp_helper=False,
    )

    assert output.shape == (0,)
    assert offsets.dtype == np.dtype(np.int64)
    assert offsets.shape == (0,)


def test_gpu_batch_empty_tile_table_skips_auto_backend_warning(
    monkeypatch, recwarn
):
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(empty=np.empty, uint8=np.uint8),
    )
    monkeypatch.setattr(
        nvcomp_batch,
        "_try_get_cpp_ext",
        lambda: pytest.fail("empty batch must not select a backend"),
    )

    output, offsets = nvcomp_batch.gpu_gzip_decompress_batch(
        SimpleNamespace(size=0),
        [],
        [],
        [],
        gzip_wrapped=False,
    )

    assert output.shape == (0,)
    assert offsets.shape == (0,)
    assert not recwarn


def test_gpu_batch_empty_tile_table_honors_forced_cpp_helper(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(empty=np.empty, uint8=np.uint8),
    )
    monkeypatch.setattr(nvcomp_batch, "_try_get_cpp_ext", lambda: None)

    with pytest.raises(RuntimeError, match="use_cpp_helper=True"):
        nvcomp_batch.gpu_gzip_decompress_batch(
            SimpleNamespace(size=0),
            [],
            [],
            [],
            gzip_wrapped=False,
            use_cpp_helper=True,
        )


def test_gpu_batch_retains_output_before_native_launch(monkeypatch):
    output = SimpleNamespace(data=SimpleNamespace(ptr=200))
    keepalive = []

    class Extension:
        def batch_deflate_decompress(self, *args):
            assert keepalive == [output]
            raise RuntimeError("native launch failed")

    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(
            cuda=SimpleNamespace(
                get_current_stream=lambda: SimpleNamespace(ptr=300)
            )
        ),
    )
    monkeypatch.setattr(nvcomp_batch, "_try_get_cpp_ext", lambda: Extension())
    monkeypatch.setattr(
        nvcomp_batch,
        "_native_device_empty_uint8",
        lambda size, ext=None: output,
    )

    with pytest.raises(RuntimeError, match="native launch failed"):
        nvcomp_batch.gpu_gzip_decompress_batch(
            SimpleNamespace(size=20, data=SimpleNamespace(ptr=102)),
            [0],
            [20],
            [1],
            use_native_pool=True,
            keepalive=keepalive,
            header_sizes=[10],
        )

    assert keepalive == [output]


@pytest.mark.parametrize(
    "decoder,gzip_available,native_pool,retain,expected_codec",
    [
        ("auto", True, False, True, "gzip"),
        ("gzip", True, True, True, "gzip"),
        ("auto", True, True, False, "gzip"),
        ("deflate", True, True, True, "deflate"),
        ("auto", False, False, True, "deflate"),
    ],
)
def test_native_decoder_receives_framing_and_retains_scratch(
    monkeypatch, decoder, gzip_available, native_pool, retain, expected_codec
):
    output = SimpleNamespace(data=SimpleNamespace(ptr=200))
    scratch = object()
    keepalive = [] if retain else None
    calls = []

    def record(codec, args, pooled):
        if keepalive is not None:
            assert keepalive == [output]
        calls.append((codec, args, pooled))
        return scratch if pooled else None

    extension = SimpleNamespace(
        batch_deflate_decompress=lambda *args: record("deflate", args, False),
        batch_deflate_decompress_pooled=lambda *args: record(
            "deflate", args, True
        ),
    )
    if gzip_available:
        extension.batch_gzip_decompress = lambda *args: record(
            "gzip", args[:7], args[7]
        )
    monkeypatch.setattr(nvcomp_batch, "_try_get_cpp_ext", lambda: extension)
    monkeypatch.setattr(
        nvcomp_batch, "_native_device_empty_uint8", lambda *a, **kw: output
    )
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(
            empty=lambda *a, **kw: output,
            uint8=np.uint8,
            cuda=SimpleNamespace(
                get_current_stream=lambda: SimpleNamespace(ptr=300)
            ),
        ),
    )

    result, offsets = nvcomp_batch.gpu_gzip_decompress_batch(
        SimpleNamespace(size=60, data=SimpleNamespace(ptr=102)),
        [0, 20, 40],
        [20, 20, 20],
        [1, 2, 3],
        header_sizes=[10, 10, 10],
        gzip_decoder=decoder,
        use_native_pool=native_pool,
        keepalive=keepalive,
    )

    assert result is output
    np.testing.assert_array_equal(offsets, [0, 1, 3])
    assert len(calls) == 1
    codec, args, pooled = calls[0]
    assert codec == expected_codec
    assert pooled is (native_pool and retain)
    assert (args[0], args[3], args[6]) == (102, 200, 300)
    expected_offsets = [0, 20, 40] if codec == "gzip" else [10, 30, 50]
    expected_lengths = [20, 20, 20] if codec == "gzip" else [2, 2, 2]
    np.testing.assert_array_equal(args[1], expected_offsets)
    np.testing.assert_array_equal(args[2], expected_lengths)
    np.testing.assert_array_equal(args[4], offsets)
    np.testing.assert_array_equal(args[5], [1, 2, 3])
    if keepalive is not None:
        assert keepalive == ([output, scratch] if pooled else [output])


def test_precomputed_header_sizes_skip_device_probe(monkeypatch):
    class StopAfterHeaderValidation(Exception):
        pass

    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace())
    monkeypatch.setattr(
        nvcomp_batch,
        "_compute_header_sizes_from_device",
        lambda *args: pytest.fail("device header probe must not run"),
    )
    monkeypatch.setattr(
        nvcomp_batch,
        "_try_get_cpp_ext",
        lambda: (_ for _ in ()).throw(StopAfterHeaderValidation()),
    )

    with pytest.raises(StopAfterHeaderValidation):
        nvcomp_batch.gpu_gzip_decompress_batch(
            SimpleNamespace(size=20),
            [0],
            [20],
            [1],
            header_sizes=[10],
        )


def test_gpu_batch_rejects_mismatched_tile_metadata(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace())

    with pytest.raises(ValueError, match="must have the same size"):
        nvcomp_batch.gpu_gzip_decompress_batch(
            SimpleNamespace(size=40),
            [0, 20],
            [20],
            [1, 1],
            header_sizes=[10, 10],
        )


@pytest.mark.parametrize(
    ("uncompressed_sizes", "match"),
    [
        ([1.5], "integer-valued"),
        ([np.nan], "integer-valued"),
        ([True], "integer-valued"),
        (
            np.array([1 << 63], dtype=np.uint64),
            "int64-representable",
        ),
    ],
)
def test_gpu_batch_rejects_nonintegral_output_sizes(
    monkeypatch, uncompressed_sizes, match
):
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace())

    with pytest.raises(ValueError, match=match):
        nvcomp_batch.gpu_gzip_decompress_batch(
            SimpleNamespace(size=2),
            [0],
            [2],
            uncompressed_sizes,
            gzip_wrapped=False,
        )


def test_gpu_batch_rejects_tile_outside_device_buffer(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace())

    with pytest.raises(ValueError, match="exceeds containing buffer"):
        nvcomp_batch.gpu_gzip_decompress_batch(
            SimpleNamespace(size=20),
            [1],
            [20],
            [1],
            header_sizes=[10],
        )


def test_gpu_batch_rejects_headers_for_raw_deflate(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace())

    with pytest.raises(ValueError, match="requires gzip_wrapped=True"):
        nvcomp_batch.gpu_gzip_decompress_batch(
            SimpleNamespace(size=2),
            [0],
            [2],
            [1],
            gzip_wrapped=False,
            header_sizes=[0],
        )


def test_gpu_library_preload_orders_kvikio_dependency_first(
    monkeypatch, tmp_path
):
    package_dirs = {}
    for module, library in (
        ("nvidia.libnvcomp", "libnvcomp.so.5"),
        ("rapids_logger", "librapids_logger.so"),
        ("libkvikio", "libkvikio.so"),
    ):
        package_dir = tmp_path / module.replace(".", "-")
        lib_dir = package_dir / "lib64"
        lib_dir.mkdir(parents=True)
        (lib_dir / library).touch()
        package_dirs[module] = package_dir

    loaded = []
    monkeypatch.setattr(
        nvcomp_batch,
        "_package_dir",
        lambda module: package_dirs.get(module),
    )
    monkeypatch.setattr(
        nvcomp_batch,
        "_load_shared_library",
        lambda path: loaded.append(path.name),
    )

    nvcomp_batch._preload_gpu_package_libraries()

    assert loaded == [
        "libnvcomp.so.5",
        "librapids_logger.so",
        "libkvikio.so",
    ]


@pytest.mark.parametrize("decoder", ["invalid", "", True])
def test_gpu_batch_rejects_invalid_decoder_before_import(decoder):
    with pytest.raises(ValueError, match="gzip_decoder must be"):
        nvcomp_batch.gpu_gzip_decompress_batch(
            None, [], [], [], gzip_decoder=decoder
        )


def test_gpu_batch_rejects_gzip_decoder_for_raw_input():
    with pytest.raises(ValueError, match="requires gzip_wrapped=True"):
        nvcomp_batch.gpu_gzip_decompress_batch(
            None, [], [], [], gzip_wrapped=False, gzip_decoder="gzip"
        )


@pytest.mark.parametrize("extension", [None, SimpleNamespace()])
def test_forced_gzip_rejects_missing_native_capability(
    monkeypatch, extension
):
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace())
    monkeypatch.setattr(nvcomp_batch, "_try_get_cpp_ext", lambda: extension)
    monkeypatch.setattr(
        nvcomp_batch,
        "_warn_python_fallback_once",
        lambda: pytest.fail("a forced decoder must not warn about fallback"),
    )
    with pytest.raises(
        RuntimeError, match="requires a native extension with Gzip"
    ):
        nvcomp_batch.gpu_gzip_decompress_batch(
            SimpleNamespace(size=20),
            [0],
            [20],
            [1],
            header_sizes=[10],
            gzip_decoder="gzip",
        )


def test_auto_decoder_propagates_native_gzip_error(monkeypatch):
    def fail_decode(*args):
        raise RuntimeError("native gzip decode failed")

    extension = SimpleNamespace(
        batch_gzip_decompress=fail_decode,
        batch_deflate_decompress=lambda *args: pytest.fail(
            "must not retry decode"
        ),
    )
    monkeypatch.setattr(nvcomp_batch, "_try_get_cpp_ext", lambda: extension)
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(
            empty=lambda *args, **kwargs: SimpleNamespace(
                data=SimpleNamespace(ptr=200)
            ),
            uint8=np.uint8,
            cuda=SimpleNamespace(
                get_current_stream=lambda: SimpleNamespace(ptr=0)
            ),
        ),
    )
    with pytest.raises(RuntimeError, match="native gzip decode failed"):
        nvcomp_batch.gpu_gzip_decompress_batch(
            SimpleNamespace(size=20, data=SimpleNamespace(ptr=100)),
            [0],
            [20],
            [1],
            header_sizes=[10],
        )


@pytest.mark.parametrize("backend", ["invalid", "hardware", "", True])
def test_gpu_batch_rejects_invalid_backend_before_import(backend):
    with pytest.raises(ValueError, match="decompression_backend must be"):
        nvcomp_batch.gpu_gzip_decompress_batch(
            None, [], [], [], decompression_backend=backend
        )


@pytest.mark.parametrize("extension", [None, SimpleNamespace()])
def test_forced_cuda_rejects_missing_native_capability(
    monkeypatch, extension
):
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace())
    monkeypatch.setattr(nvcomp_batch, "_try_get_cpp_ext", lambda: extension)
    monkeypatch.setattr(
        nvcomp_batch, "_warn_python_fallback_once", lambda: None
    )
    with pytest.raises(RuntimeError, match="requires a rebuilt native"):
        nvcomp_batch.gpu_gzip_decompress_batch(
            SimpleNamespace(size=20),
            [0],
            [20],
            [1],
            header_sizes=[10],
            decompression_backend="cuda",
        )
