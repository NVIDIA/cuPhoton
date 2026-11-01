# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

import cuphoton.xbeam.adapters.vdif.vdif_index as vdif_index
from cuphoton.xbeam.adapters.vdif.vdif import (
    TruncatedVdifFrameError,
    VdifError,
)
from cuphoton.xbeam.adapters.vdif.vdif_index import (
    VdifOrderedSearchCache,
    VdifOrderedSeekError,
    VdifTimestampKey,
    extract_vdif_headers,
    inspect_vdif,
    locate_ordered_vdif_range,
)


def _frame(
    *,
    seconds: int = 0,
    frame_number: int = 0,
    thread_id: int = 0,
    channels: int = 1,
    complex_data: bool = False,
    invalid: bool = False,
    payload_bytes: int = 8,
    station: str = "AA",
) -> bytes:
    log2_channels = channels.bit_length() - 1
    word0 = seconds | (int(invalid) << 31)
    word1 = frame_number | (40 << 24)
    word2 = ((32 + payload_bytes) // 8) | (log2_channels << 24)
    word3 = (
        (ord(station[0]) << 8)
        | ord(station[1])
        | (thread_id << 16)
        | (1 << 26)
        | (int(complex_data) << 31)
    )
    header = np.asarray((word0, word1, word2, word3, 0, 0, 0, 0), dtype="<u4")
    return header.tobytes() + bytes((frame_number & 0xFF,)) * payload_bytes


def _write_ordered(path: Path, count: int, *, threads: int = 1) -> None:
    path.write_bytes(
        b"".join(
            _frame(frame_number=index // threads, thread_id=index % threads)
            for index in range(count)
        )
    )


def test_inspection_is_bounded_and_json_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "sample.vdif"
    _write_ordered(source, 200, threads=2)
    calls: list[tuple[int, int]] = []
    real_pread = vdif_index.os.pread

    def record_pread(fd: int, size: int, offset: int) -> bytes:
        calls.append((size, offset))
        return real_pread(fd, size, offset)

    monkeypatch.setattr(vdif_index.os, "pread", record_pread)
    report = inspect_vdif(source)

    assert report.frame_size_bytes == 40
    assert report.frame_count == 200
    assert report.trailing_bytes == 0
    assert report.thread_ids == (0, 1)
    assert report.sampled_monotonic
    assert len(report.sampled_frame_indices) == 17
    assert len(calls) <= 19
    assert all(size == 32 for size, _ in calls)
    assert json.loads(json.dumps(report.to_dict()))["frame_count"] == 200


def test_inspection_validates_explicit_frame_size(tmp_path: Path) -> None:
    source = tmp_path / "size.vdif"
    source.write_bytes(_frame())

    with pytest.raises(VdifError, match="declares 40 bytes; expected 48"):
        inspect_vdif(source, frame_size_bytes=48)


def test_edge_block_inspection_coalesces_large_probe_sets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "edge-blocks.vdif"
    _write_ordered(source, 200, threads=2)
    preadv_sizes: list[int] = []
    real_preadv = vdif_index.os.preadv

    def record_preadv(fd: int, buffers, offset: int) -> int:
        preadv_sizes.append(sum(len(item) for item in buffers))
        return real_preadv(fd, buffers, offset)

    monkeypatch.setattr(vdif_index.os, "preadv", record_preadv)
    report = inspect_vdif(
        source,
        sample_count=65,
        sample_layout="edge-blocks",
    )

    assert len(report.sampled_frame_indices) == 65
    assert report.thread_ids == (0, 1)
    assert report.sampled_monotonic
    assert len(preadv_sizes) == 2
    assert all(size > report.frame_size_bytes for size in preadv_sizes)


def test_inspection_rejects_unknown_sample_layout(tmp_path: Path) -> None:
    source = tmp_path / "layout.vdif"
    source.write_bytes(_frame())

    with pytest.raises(ValueError, match="sample_layout"):
        inspect_vdif(source, sample_layout="random")


def test_extract_headers_reads_only_headers_and_seeks_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.vdif"
    frames = [_frame(frame_number=index) for index in range(4)]
    source.write_bytes(b"".join(frames))
    output = tmp_path / "headers.bin"
    read_sizes: list[int] = []
    seek_offsets: list[tuple[int, int]] = []
    real_read = vdif_index.os.read
    real_lseek = vdif_index.os.lseek

    def tracked_read(fd: int, size: int) -> bytes:
        read_sizes.append(size)
        return real_read(fd, size)

    def tracked_lseek(fd: int, offset: int, whence: int) -> int:
        seek_offsets.append((offset, whence))
        return real_lseek(fd, offset, whence)

    monkeypatch.setattr(vdif_index.os, "read", tracked_read)
    monkeypatch.setattr(vdif_index.os, "lseek", tracked_lseek)

    result = extract_vdif_headers(
        source, output, first_frame=1, frame_count=2
    )

    assert output.read_bytes() == frames[1][:32] + frames[2][:32]
    assert read_sizes == [32, 32]
    assert seek_offsets == [
        (40, vdif_index.os.SEEK_SET),
        (8, vdif_index.os.SEEK_CUR),
    ]
    assert result.first_frame == 1
    assert result.frame_count == 2
    assert result.output_bytes == 64
    assert result.to_dict()["header_size_bytes"] == 32
    assert not list(tmp_path.glob(".headers.bin.*.tmp"))

    with pytest.raises(FileExistsError, match="already exists"):
        extract_vdif_headers(source, output)


def test_extract_headers_does_not_publish_a_partial_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "short-read.vdif"
    _write_ordered(source, 2)
    output = tmp_path / "headers.bin"
    real_read = vdif_index.os.read
    calls = 0

    def short_second_read(fd: int, size: int) -> bytes:
        nonlocal calls
        calls += 1
        return b"" if calls == 2 else real_read(fd, size)

    monkeypatch.setattr(vdif_index.os, "read", short_second_read)

    with pytest.raises(TruncatedVdifFrameError, match="got 0 bytes"):
        extract_vdif_headers(source, output)

    assert not output.exists()
    assert not list(tmp_path.glob(".headers.bin.*.tmp"))


def test_inspection_cache_reuses_only_unchanged_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "cached.vdif"
    source.write_bytes(_frame())
    first = inspect_vdif(source)
    original_pread = vdif_index.os.pread
    calls = 0

    def counted_pread(*args):
        nonlocal calls
        calls += 1
        return original_pread(*args)

    monkeypatch.setattr(vdif_index.os, "pread", counted_pread)
    assert inspect_vdif(source) is first
    assert calls == 0

    source.write_bytes(_frame() + _frame(frame_number=1))
    refreshed = inspect_vdif(source)
    assert refreshed is not first
    assert refreshed.frame_count == 2
    assert calls > 0


def test_inspection_reports_trailing_bytes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "trailing.vdif"
    source.write_bytes(_frame() + b"x")

    assert inspect_vdif(source).trailing_bytes == 1


def test_ordered_seek_requires_authorization_and_filters_threads(
    tmp_path: Path,
) -> None:
    source = tmp_path / "seek.vdif"
    _write_ordered(source, 12, threads=2)
    start = VdifTimestampKey(40, 0, 1)
    stop = VdifTimestampKey(40, 0, 4)

    with pytest.raises(VdifOrderedSeekError, match="explicit"):
        locate_ordered_vdif_range(source, start, stop)

    selected = locate_ordered_vdif_range(
        source,
        start,
        stop,
        thread_ids=(1,),
        assume_ordered=True,
    )
    assert selected == (vdif_index.VdifFrameRange(3, 8, 40),)


def test_ordered_seek_vectorizes_large_header_envelope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "large-seek.vdif"
    _write_ordered(source, 5000)
    calls = 0
    real_pread = vdif_index.os.pread

    def count_pread(fd: int, size: int, offset: int) -> bytes:
        nonlocal calls
        calls += 1
        return real_pread(fd, size, offset)

    monkeypatch.setattr(vdif_index.os, "pread", count_pread)

    selected = locate_ordered_vdif_range(
        source,
        VdifTimestampKey(40, 0, 0),
        VdifTimestampKey(40, 0, 5000),
        assume_ordered=True,
    )

    assert selected == (vdif_index.VdifFrameRange(0, 5000, 40),)
    assert calls < 100


def test_ordered_seek_cache_reuses_adjacent_window_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "cached-seek.vdif"
    _write_ordered(source, 4096)
    cache = VdifOrderedSearchCache(inspect_vdif(source))
    reads = 0
    original = vdif_index._read_header

    def count_reads(*args, **kwargs):
        nonlocal reads
        reads += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(vdif_index, "_read_header", count_reads)
    first = locate_ordered_vdif_range(
        source,
        VdifTimestampKey(40, 0, 1000),
        VdifTimestampKey(40, 0, 2000),
        assume_ordered=True,
        validate_all_headers=False,
        search_cache=cache,
    )
    first_reads = reads
    second = locate_ordered_vdif_range(
        source,
        VdifTimestampKey(40, 0, 2000),
        VdifTimestampKey(40, 0, 3000),
        assume_ordered=True,
        validate_all_headers=False,
        search_cache=cache,
    )

    assert first == (vdif_index.VdifFrameRange(1000, 2000, 40),)
    assert second == (vdif_index.VdifFrameRange(2000, 3000, 40),)
    assert cache.target_keys == [
        VdifTimestampKey(40, 0, value).packed for value in (1000, 2000, 3000)
    ]
    assert reads - first_reads < first_reads


def test_ordered_seek_can_defer_exhaustive_header_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "deferred-seek.vdif"
    _write_ordered(source, 12, threads=2)

    def unexpected_validation(*_args, **_kwargs):
        raise AssertionError("exhaustive validation must be deferred")

    monkeypatch.setattr(
        vdif_index, "_validate_ordered_range", unexpected_validation
    )

    selected = locate_ordered_vdif_range(
        source,
        VdifTimestampKey(40, 0, 1),
        VdifTimestampKey(40, 0, 4),
        thread_ids=(1,),
        assume_ordered=True,
        validate_all_headers=False,
    )

    # The deferred path conservatively retains the complete timestamp
    # envelope. The recording source subsequently validates all six headers
    # and filters the unrequested thread.
    assert selected == (vdif_index.VdifFrameRange(2, 8, 40),)


def test_deferred_ordered_seek_requires_external_global_order_evidence(
    tmp_path: Path,
) -> None:
    source = tmp_path / "locally-disordered.vdif"
    sequence = [
        (frame_number, thread_id)
        for frame_number in range(60)
        for thread_id in range(2)
    ]
    sequence[35:47] = [
        (18, 0),
        (19, 0),
        (20, 0),
        (17, 1),
        (21, 0),
        (22, 0),
        (18, 1),
        (19, 1),
        (20, 1),
        (23, 0),
        (21, 1),
        (22, 1),
    ]
    source.write_bytes(
        b"".join(
            _frame(frame_number=frame_number, thread_id=thread_id)
            for frame_number, thread_id in sequence
        )
    )
    start = VdifTimestampKey(40, 0, 18)
    stop = VdifTimestampKey(40, 0, 21)

    assert inspect_vdif(source).sampled_monotonic
    with pytest.raises(VdifOrderedSeekError, match="ordering decreases"):
        locate_ordered_vdif_range(
            source,
            start,
            stop,
            assume_ordered=True,
        )

    selected = locate_ordered_vdif_range(
        source,
        start,
        stop,
        assume_ordered=True,
        validate_all_headers=False,
    )

    assert selected == (vdif_index.VdifFrameRange(39, 44, 40),)
    assert sequence.index((19, 0)) == 36


def test_deferred_ordered_seek_omits_zero_length_range(
    tmp_path: Path,
) -> None:
    source = tmp_path / "deferred-empty.vdif"
    _write_ordered(source, 4)

    selected = locate_ordered_vdif_range(
        source,
        VdifTimestampKey(40, 0, 10),
        VdifTimestampKey(40, 0, 11),
        assume_ordered=True,
        validate_all_headers=False,
    )

    assert selected == ()


def test_ordered_validation_reuses_buffer_across_partial_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "partial-preadv.vdif"
    _write_ordered(source, 8)
    original_preadv = os.preadv
    buffer_ids: list[int] = []

    def partial_preadv(fd, buffers, offset):
        view = buffers[0]
        buffer_ids.append(id(view.obj))
        return original_preadv(fd, [view[: min(len(view), 40)]], offset)

    monkeypatch.setattr(vdif_index, "DEFAULT_INDEX_CHUNK_BYTES", 80)
    monkeypatch.setattr(vdif_index.os, "preadv", partial_preadv)

    selected = locate_ordered_vdif_range(
        source,
        VdifTimestampKey(40, 0, 1),
        VdifTimestampKey(40, 0, 7),
        assume_ordered=True,
    )

    assert selected == (vdif_index.VdifFrameRange(1, 7, 40),)
    assert len(buffer_ids) > 2
    assert len(set(buffer_ids)) == 1


def test_ordered_seek_rejects_range_before_header_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "bounded-seek.vdif"
    _write_ordered(source, 10)

    def unexpected_validation(*_args, **_kwargs):
        raise AssertionError("header envelope must not be validated")

    monkeypatch.setattr(
        vdif_index, "_validate_ordered_range", unexpected_validation
    )

    with pytest.raises(VdifOrderedSeekError, match="max_read_bytes"):
        locate_ordered_vdif_range(
            source,
            VdifTimestampKey(40, 0, 0),
            VdifTimestampKey(40, 0, 10),
            assume_ordered=True,
            max_range_bytes=399,
        )


def test_ordered_seek_rejects_sampled_disorder(tmp_path: Path) -> None:
    source = tmp_path / "disordered.vdif"
    source.write_bytes(
        _frame(frame_number=0)
        + _frame(frame_number=2)
        + _frame(frame_number=1)
    )

    with pytest.raises(VdifOrderedSeekError, match="not monotonically"):
        locate_ordered_vdif_range(
            source,
            VdifTimestampKey(40, 0, 0),
            VdifTimestampKey(40, 0, 3),
            assume_ordered=True,
        )


def test_ordered_seek_rejects_local_thread_format_change(
    tmp_path: Path,
) -> None:
    source = tmp_path / "format-change.vdif"
    source.write_bytes(
        _frame(frame_number=0, channels=1)
        + _frame(frame_number=1, channels=2)
    )

    with pytest.raises(VdifOrderedSeekError, match="format changes"):
        locate_ordered_vdif_range(
            source,
            VdifTimestampKey(40, 0, 0),
            VdifTimestampKey(40, 0, 2),
            assume_ordered=True,
        )


def test_empty_explicit_thread_query_is_bounded(tmp_path: Path) -> None:
    source = tmp_path / "empty-filter.vdif"
    source.write_bytes(_frame())

    assert (
        locate_ordered_vdif_range(
            source,
            VdifTimestampKey(40, 0, 0),
            VdifTimestampKey(40, 0, 1),
            thread_ids=(),
            assume_ordered=True,
        )
        == ()
    )
