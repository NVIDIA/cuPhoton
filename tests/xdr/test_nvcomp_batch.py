# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from cuphoton.xdr import nvcomp_batch


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
            SimpleNamespace(size=20, data=SimpleNamespace(ptr=100)),
            [0],
            [20],
            [1],
            use_native_pool=True,
            keepalive=keepalive,
            header_sizes=[10],
        )

    assert keepalive == [output]


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
