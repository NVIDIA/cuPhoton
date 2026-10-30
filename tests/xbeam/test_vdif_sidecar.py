# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import numpy as np
import pytest

from cuphoton.xbeam.adapters.vdif import vdif_sidecar as sidecar
from cuphoton.xbeam.adapters.vdif import vdif_sidecar_builder as builder
from cuphoton.xbeam.adapters.vdif.vdif_index import (
    VdifIndexError,
    VdifTimestampKey,
)
from cuphoton.xbeam.adapters.vdif.vdif_sidecar import (
    SIDECAR_HEADER_BYTES,
    VdifFrameState,
    binary_index_size,
    query_vdif_binary_index,
    query_vdif_recorded_filler_runs,
    read_vdif_binary_frames,
    read_vdif_binary_physical_frame_states,
    read_vdif_binary_physical_frames,
    validate_vdif_binary_index,
)
from cuphoton.xbeam.adapters.vdif.vdif_sidecar_builder import (
    build_vdif_binary_index,
)
from cuphoton.xbeam.buffered_reader.timeline import PhysicalAllocationTimeline


@pytest.mark.parametrize("offset", [0, (1 << 53) - 20000])
@pytest.mark.parametrize("pattern", ["jitter", "gaps", "duplicates"])
def test_bitmap_ranges_match_exact_physical_set(offset, pattern):
    rng = np.random.default_rng(42)
    frames = np.arange(8192, dtype=np.uint64)
    if pattern == "gaps":
        frames = frames[frames % 3 != 0]
    elif pattern == "duplicates":
        frames = np.repeat(frames[::2], 2)
    rng.shuffle(frames)
    frames += offset
    bounds = sidecar._bitmap_frame_bounds(frames)
    assert bounds is not None
    expected = sorted(set(map(int, frames)))
    actual = [value for start, stop in bounds for value in range(start, stop)]
    assert actual == expected
    assert np.all(bounds[:-1, 1] < bounds[1:, 0])


@pytest.mark.parametrize("count", [0, 1, 4095, 4096])
def test_bitmap_minimum_entries(count):
    result = sidecar._bitmap_frame_bounds(np.arange(count, dtype=np.uint64))
    assert (result is not None) == (count >= 4096)


@pytest.mark.parametrize("extra", [0, 1])
def test_bitmap_density_boundary(extra):
    frames = np.arange(4096, dtype=np.uint64)
    frames[-1] = 8191 + extra
    assert (sidecar._bitmap_frame_bounds(frames) is None) == bool(extra)


@pytest.mark.parametrize("extra", [-1, 0, 1])
def test_bitmap_memory_admission(monkeypatch, extra):
    frames = np.arange(4096, dtype=np.uint64)
    estimate = 26 * frames.size + 32
    monkeypatch.setattr(sidecar, "_BITMAP_MAX_TEMP_BYTES", estimate + extra)
    assert (sidecar._bitmap_frame_bounds(frames) is not None) == (extra > 0)


@pytest.mark.parametrize("failure", [None, "zeros", "flatnonzero"])
def test_dense_sidecar_query_and_allocation_fallback(
    tmp_path, monkeypatch, failure
):
    source = tmp_path / "dense.vdif"
    order = np.random.default_rng(42).permutation(8192)
    source.write_bytes(b"".join(_frame(int(value)) for value in order))
    build_vdif_binary_index(source, backend="posix")
    if failure:
        original = getattr(sidecar.np, failure)
        bitmap_bounds = sidecar._bitmap_frame_bounds
        calls = 0

        def fail_once(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise MemoryError("injected bitmap allocation failure")
            return original(*args, **kwargs)

        def failing_bitmap(frames):
            with monkeypatch.context() as patch:
                patch.setattr(sidecar.np, failure, fail_once)
                return bitmap_bounds(frames)

        monkeypatch.setattr(sidecar, "_bitmap_frame_bounds", failing_bitmap)
    ranges = query_vdif_binary_index(
        source,
        VdifTimestampKey(40, 17, 1000),
        VdifTimestampKey(40, 17, 7000),
        thread_ids=(0,),
    )
    actual = [
        value
        for item in ranges
        for value in range(item.start_frame, item.stop_frame)
    ]
    assert actual == np.flatnonzero((order >= 1000) & (order < 7000)).tolist()
    if failure:
        assert calls >= 1


@pytest.mark.parametrize("bitmap", [True, False])
def test_sidecar_query_rejects_duplicate_physical_ordinals(
    tmp_path, monkeypatch, bitmap
):
    source = tmp_path / "duplicate-physical.vdif"
    frame_count = 4096 if bitmap else 4
    order = np.arange(frame_count, dtype=np.int64)[::-1]
    source.write_bytes(b"".join(_frame(int(value)) for value in order))
    summary = build_vdif_binary_index(source, backend="posix")
    entries = np.memmap(
        summary.index_path,
        dtype="<u8",
        mode="r+",
        offset=SIDECAR_HEADER_BYTES,
        shape=(frame_count, 3),
    )
    entries[1, 1] = entries[0, 1]
    entries.flush()
    del entries
    if not bitmap:
        monkeypatch.setattr(
            sidecar, "_bitmap_frame_bounds", lambda frames: None
        )

    with pytest.raises(VdifIndexError, match="duplicate physical"):
        query_vdif_binary_index(
            source,
            VdifTimestampKey(40, 17, 0),
            VdifTimestampKey(40, 17, frame_count),
        )


@pytest.mark.parametrize("stage", ["geometry", "write", "validate"])
def test_failed_forced_rebuild_preserves_sidecar(
    tmp_path, monkeypatch, stage
):
    source = tmp_path / "source.vdif"
    source.write_bytes(_frame(0) + _frame(1))
    summary = build_vdif_binary_index(source, backend="posix")
    original = summary.index_path.read_bytes()

    def fail(*args, **kwargs):
        raise RuntimeError("injected build failure")

    target = {
        "geometry": "_frame_geometry",
        "write": "_write_cpu_index",
        "validate": "_validate_binary_index_path",
    }[stage]
    monkeypatch.setattr(builder, target, fail)
    with pytest.raises(RuntimeError, match="injected"):
        build_vdif_binary_index(source, backend="posix", force=True)
    assert summary.index_path.read_bytes() == original
    assert not list(tmp_path.glob(".*"))


@pytest.mark.parametrize("force", [False, True])
def test_concurrent_builders_own_separate_partial_files(
    tmp_path, monkeypatch, force
):
    source = tmp_path / "source.vdif"
    source.write_bytes(_frame(0) + _frame(1))
    barrier = Barrier(2)
    write = builder._write_cpu_index
    partials = []

    def delayed(*args, **kwargs):
        partials.append(args[1])
        write(*args, **kwargs)
        barrier.wait(timeout=5)

    monkeypatch.setattr(builder, "_write_cpu_index", delayed)
    with ThreadPoolExecutor(2) as pool:
        futures = [
            pool.submit(
                build_vdif_binary_index, source, backend="posix", force=force
            )
            for _ in range(2)
        ]
        summaries = [future.result(timeout=10) for future in futures]
    assert summaries[0].index_path == summaries[1].index_path
    assert len(set(partials)) == 2
    assert not any(path.exists() for path in partials)
    assert validate_vdif_binary_index(source).frame_count == 2


@pytest.mark.parametrize(
    "mutation", [None, "sidecar", "source", "repeated_cleanup"]
)
def test_validation_restarts_after_published_hardlink_cleanup(
    tmp_path, monkeypatch, mutation
):
    source = tmp_path / "source.vdif"
    source.write_bytes(_frame(0) + _frame(1))
    index = build_vdif_binary_index(source, backend="posix").index_path
    partial = tmp_path / "index.partial"
    partial.hardlink_to(index)
    original = index.read_bytes()
    metadata = sidecar._thread_metadata
    calls = 0

    def cleanup_during_validation(document):
        nonlocal calls
        calls += 1
        result = metadata(document)
        if mutation == "repeated_cleanup" and not partial.exists():
            partial.hardlink_to(index)
        if partial.exists():
            index_stat = index.stat()
            partial.unlink()
            if mutation == "sidecar":
                with index.open("r+b") as stream:
                    stream.seek(sidecar.physical_timestamp_run_offset(2) + 16)
                    stream.write((1 << 60).to_bytes(8, "little"))
            elif mutation == "source":
                source_stat = source.stat()
                source.write_bytes(_frame(0) + _frame(2))
                os.utime(
                    source,
                    ns=(
                        source_stat.st_atime_ns,
                        source_stat.st_mtime_ns + 1_000_000_000,
                    ),
                )
            # Filesystems may coalesce ctime updates from hardlink cleanup.
            # Make the metadata change observable without relying on a tick.
            os.utime(
                index,
                ns=(
                    index_stat.st_atime_ns,
                    index_stat.st_mtime_ns + 1_000_000_000,
                ),
            )
        return result

    monkeypatch.setattr(
        sidecar, "_thread_metadata", cleanup_during_validation
    )
    if mutation == "sidecar":
        with pytest.raises(
            VdifIndexError, match="timestamp runs are invalid"
        ):
            validate_vdif_binary_index(source)
        with pytest.raises(
            VdifIndexError, match="timestamp runs are invalid"
        ):
            validate_vdif_binary_index(source)
    elif mutation in {"source", "repeated_cleanup"}:
        with pytest.raises(VdifIndexError, match="changed during validation"):
            validate_vdif_binary_index(source)
        assert calls == (1 if mutation == "source" else 3)
    else:
        summary = validate_vdif_binary_index(source)
        assert summary.frame_count == 2
        assert summary.index_identity == sidecar._identity(index.stat())
        assert index.read_bytes() == original
        assert calls == 2


def _frame(
    frame_number: int,
    *,
    thread_id: int = 0,
    invalid: bool = False,
    station_id: int = (ord("A") << 8) | ord("A"),
    declared_frame_size: int = 40,
    actual_frame_size: int = 40,
) -> bytes:
    word0 = 17 | (int(invalid) << 31)
    word1 = frame_number | (40 << 24)
    word2 = declared_frame_size // 8
    word3 = station_id | (thread_id << 16) | (1 << 26)
    words = np.asarray((word0, word1, word2, word3, 0, 0, 0, 0), dtype="<u4")
    return words.tobytes() + bytes((frame_number & 0xFF,)) * (
        actual_frame_size - 32
    )


def _recorded_filler(
    *,
    exact_payload: bool = True,
    thread_id: int = 0,
    frame_size: int = 40,
    trailer: tuple[int, int] | None = None,
) -> bytes:
    words = np.asarray(
        (
            0x8000_0000,
            0,
            frame_size // 8,
            thread_id << 16,
            0xABAD_DEED,
            0,
            0,
            0,
        ),
        dtype="<u4",
    )
    payload_words = [0x1122_3344] * ((frame_size - 32) // 4)
    if trailer is not None:
        payload_words[-2:] = trailer
    payload = np.asarray(payload_words, dtype="<u4").tobytes()
    if not exact_payload:
        payload = payload[:-1] + b"\x00"
    return words.tobytes() + payload


def test_sidecar_records_only_exact_recorded_filler_runs(tmp_path):
    source = tmp_path / "filler.vdif"
    source.write_bytes(
        _frame(0)
        + _recorded_filler()
        + _recorded_filler()
        + _frame(3)
        + _frame(4)
        + _recorded_filler()
    )
    summary = build_vdif_binary_index(source, backend="posix")
    assert summary.recorded_filler_classifier == (
        "exact-abaddeed-repeating-11223344-v2"
    )
    assert query_vdif_recorded_filler_runs(summary) == ((1, 3), (5, 6))
    assert summary.schema_version == 4
    assert summary.has_physical_timestamp_runs
    runs = np.asarray(sidecar._summary_physical_timestamp_runs(summary))
    assert runs[:, :2].tolist() == [[0, 1], [3, 5]]
    assert (runs[:, 2] & np.uint64(0xFFFFFF)).tolist() == [0, 3]


def test_physical_timestamp_runs_retain_explicit_invalid_frames(tmp_path):
    source = tmp_path / "invalid-in-run.vdif"
    source.write_bytes(_frame(0) + _frame(1, invalid=True) + _frame(2))

    summary = build_vdif_binary_index(source, backend="posix")

    runs = np.asarray(sidecar._summary_physical_timestamp_runs(summary))
    assert runs[:, :2].tolist() == [[0, 3]]
    assert int(runs[0, 2] & np.uint64(0xFFFFFF)) == 0


@pytest.mark.parametrize("allocation_frames", [3, 7, 1206])
def test_file_backed_sample_provenance_crosses_filler_and_timestamp_holes(
    tmp_path, allocation_frames
):
    source = tmp_path / "source.vdif"
    source.write_bytes(
        _frame(0)
        + _frame(1, invalid=True)
        + _recorded_filler()
        + _frame(5)
        + _frame(6)
    )
    summary = build_vdif_binary_index(source, backend="posix")
    runs = sidecar._summary_physical_timestamp_runs(summary)
    model = PhysicalAllocationTimeline.from_runs(
        runs[:, 0],
        runs[:, 1],
        runs[:, 2] & np.uint64(0xFFFFFF),
        allocation_frames=allocation_frames,
    )
    # Eight payload bytes encode 32 two-bit samples. Read the real file via
    # the compact sidecar mapping, including requests that split a frame.
    # This tests provenance; explicit-invalid frame 1 remains in the mapping
    # and its eligibility must be decided separately by the validity owner.
    timestamp_to_physical = {0: 0, 1: 1, 5: 3, 6: 4}
    for start, count in [(0, 224), (17, 179), (55, 113), (161, 31)]:
        observed = [None] * count
        with source.open("rb") as recording:
            for span in model.sample_source_spans(
                start, count, samples_per_frame=32
            ):
                for local in range(span.request_start, span.request_stop):
                    physical_sample = (
                        span.physical_start + local - span.request_start
                    )
                    frame, phase = divmod(physical_sample, 32)
                    recording.seek(frame * 40 + 32 + phase // 4)
                    byte = recording.read(1)[0]
                    observed[local] = (byte >> (2 * (phase % 4))) & 3
        expected = []
        for sample in range(start, start + count):
            timestamp_frame, phase = divmod(sample, 32)
            expected.append(
                (timestamp_frame >> (2 * (phase % 4))) & 3
                if timestamp_frame in timestamp_to_physical
                else None
            )
        assert observed == expected


@pytest.mark.parametrize(
    "invalid_field", ["stop", "timestamp_bit_60", "timestamp_bit_63"]
)
def test_schema_v4_validation_rejects_invalid_physical_run(
    tmp_path, invalid_field
):
    source = tmp_path / "invalid-physical-run.vdif"
    source.write_bytes(_frame(0) + _frame(1))
    summary = build_vdif_binary_index(source, backend="posix")
    runs = np.memmap(
        summary.index_path,
        dtype="<u8",
        mode="r+",
        offset=sidecar.physical_timestamp_run_offset(summary.entry_count),
        shape=(summary.physical_timestamp_run_count, 3),
    )
    if invalid_field == "stop":
        runs[0, 1] = summary.frame_count + 1
    else:
        bit = 60 if invalid_field == "timestamp_bit_60" else 63
        runs[0, 2] |= np.uint64(1 << bit)
    runs.flush()
    del runs

    with pytest.raises(VdifIndexError, match="timestamp runs are invalid"):
        sidecar._validate_binary_index_path(source, summary.index_path)


def test_near_filler_remains_subject_to_strict_validation(tmp_path):
    source = tmp_path / "near-filler.vdif"
    source.write_bytes(_frame(0) + _recorded_filler(exact_payload=False))

    with pytest.raises(VdifIndexError, match="frame 1: .*changes format"):
        build_vdif_binary_index(source, backend="posix")


def test_recorded_filler_does_not_register_phantom_thread(tmp_path):
    source = tmp_path / "filler-thread.vdif"
    source.write_bytes(
        _recorded_filler(thread_id=3)
        + _frame(1, thread_id=7)
        + _recorded_filler(thread_id=3)
        + _frame(3, thread_id=7)
    )

    summary = build_vdif_binary_index(
        source, frame_size_bytes=40, backend="posix"
    )

    assert [item.thread_id for item in summary.thread_metadata] == [7]
    assert query_vdif_recorded_filler_runs(summary) == ((0, 1), (2, 3))


def test_physical_query_preserves_filler_state_without_duplicate_or_phantom(
    tmp_path,
):
    source = tmp_path / "physical-filler.vdif"
    source.write_bytes(
        _frame(0, thread_id=7)
        + _recorded_filler(thread_id=3)
        + _frame(2, thread_id=7)
    )
    summary = build_vdif_binary_index(
        source, frame_size_bytes=40, backend="posix"
    )

    timestamps, threads, physical, states = (
        read_vdif_binary_physical_frame_states(
            source, ((0, 2), (1, 3)), frame_size_bytes=40
        )
    )
    _, _, wrapper_physical, invalid = read_vdif_binary_physical_frames(
        source, ((0, 3),), frame_size_bytes=40
    )

    assert timestamps.size == 3
    assert list(physical) == [0, 1, 2]
    assert list(wrapper_physical) == [0, 1, 2]
    assert list(threads) == [7, 3, 7]
    assert list(states) == [
        VdifFrameState.VALID,
        VdifFrameState.EXPLICIT_INVALID | VdifFrameState.RECORDED_FILLER,
        VdifFrameState.VALID,
    ]
    assert list(invalid) == [False, True, False]
    assert [item.thread_id for item in summary.thread_metadata] == [7]


@pytest.mark.parametrize("ordered", [False, True])
def test_physical_query_rejects_duplicate_ordinal_hiding_missing_frame(
    tmp_path, ordered
):
    source = tmp_path / "duplicate-physical-range.vdif"
    frames = range(6) if ordered else range(5, -1, -1)
    source.write_bytes(b"".join(_frame(frame) for frame in frames))
    summary = build_vdif_binary_index(source, backend="posix")
    entries = np.memmap(
        summary.index_path,
        dtype="<u8",
        mode="r+",
        offset=SIDECAR_HEADER_BYTES,
        shape=(summary.entry_count, 3),
    )
    physical = entries[:, 1] & np.uint64((1 << 53) - 1)
    missing = int(np.flatnonzero(physical == 3)[0])
    duplicate = int(np.flatnonzero(physical == 2)[0])
    entries[missing, 1] = entries[duplicate, 1]
    entries.flush()
    del entries

    # The merged request still selects four entries, but now they address
    # frames 1, 2, 2, 4 instead of covering every requested frame once.
    with pytest.raises(VdifIndexError, match="duplicate physical"):
        read_vdif_binary_physical_frame_states(source, ((1, 3), (2, 5)))


@pytest.mark.parametrize("shuffled", [False, True])
def test_physical_block_query_matches_full_scan(
    tmp_path, monkeypatch, shuffled
):
    monkeypatch.setattr(sidecar, "_PHYSICAL_QUERY_BLOCK_ENTRIES", 16)
    sidecar._physical_query_block_bounds.cache_clear()
    values = np.arange(131)
    if shuffled:
        np.random.default_rng(431).shuffle(values)
    source = tmp_path / "block-query.vdif"
    source.write_bytes(b"".join(_frame(int(value)) for value in values))
    summary = build_vdif_binary_index(source, backend="posix")
    entries = sidecar._summary_entries(summary)
    for ranges in [((15, 18),), ((1, 7), (5, 23), (99, 131)), ((0, 131),)]:
        expected_physical = sorted(
            {i for start, stop in ranges for i in range(start, stop)}
        )
        all_physical = entries[:, 1] & np.uint64((1 << 53) - 1)
        selected = entries[np.isin(all_physical, expected_physical)]
        selected = selected[np.argsort(selected[:, 1] & ((1 << 53) - 1))]
        expected = sidecar._decode_binary_frame_entries(summary, selected)
        actual = read_vdif_binary_physical_frame_states(source, ranges)
        for left, right in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(left, right)
    assert sidecar._physical_query_block_bounds.cache_info().misses == 1
    assert sidecar._physical_query_block_bounds.cache_info().hits == 2
    sidecar._physical_query_block_bounds.cache_clear()


def test_physical_block_query_rebuild_invalidates_bounds(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(sidecar, "_PHYSICAL_QUERY_BLOCK_ENTRIES", 16)
    sidecar._physical_query_block_bounds.cache_clear()
    source = tmp_path / "rebuild-query.vdif"
    source.write_bytes(b"".join(_frame(i) for i in range(40)))
    build_vdif_binary_index(source, backend="posix")
    read_vdif_binary_physical_frame_states(source, ((20, 23),))
    source.write_bytes(b"".join(_frame(i) for i in reversed(range(40))))
    build_vdif_binary_index(source, backend="posix", force=True)
    timestamps, _, physical, _ = read_vdif_binary_physical_frame_states(
        source, ((20, 23),)
    )
    assert list(physical) == [20, 21, 22]
    assert list(timestamps & np.uint64(0xFFFFFF)) == [19, 18, 17]
    assert sidecar._physical_query_block_bounds.cache_info().misses == 2
    sidecar._physical_query_block_bounds.cache_clear()


def test_recorded_filler_accepts_recorder_status_trailer(tmp_path):
    source = tmp_path / "filler-status-trailer.vdif"
    source.write_bytes(
        _frame(
            0,
            thread_id=7,
            declared_frame_size=48,
            actual_frame_size=48,
        )
        + _recorded_filler(
            thread_id=3,
            frame_size=48,
            trailer=(0x0000_08AB, 0x0098_77E8),
        )
    )

    summary = build_vdif_binary_index(source, backend="posix")

    assert [item.thread_id for item in summary.thread_metadata] == [7]
    assert query_vdif_recorded_filler_runs(summary) == ((1, 2),)


@pytest.mark.parametrize("same_thread", [False, True])
def test_ordered_sidecar_preserves_equal_timestamp_physical_order(
    tmp_path: Path, same_thread: bool
) -> None:
    source = tmp_path / "equal-timestamps.vdif"
    source.write_bytes(
        _frame(0, thread_id=0 if same_thread else 1, invalid=same_thread)
        + _frame(0)
        + _frame(1)
    )
    summary = build_vdif_binary_index(source, backend="posix")
    assert summary.source_ordered
    start, stop = VdifTimestampKey(40, 17, 0), VdifTimestampKey(40, 17, 1)

    _, _, frames, _ = read_vdif_binary_frames(source, start, stop)
    np.testing.assert_array_equal(frames, [0, 1])
    ranges = query_vdif_binary_index(source, start, stop)
    assert [(item.start_frame, item.stop_frame) for item in ranges] == [
        (0, 2)
    ]


def test_ordered_query_accepts_permuted_equal_timestamp_entries(
    tmp_path: Path,
) -> None:
    source = tmp_path / "permuted-ties.vdif"
    source.write_bytes(_frame(0, thread_id=1) + _frame(0) + _frame(1))
    summary = build_vdif_binary_index(source, backend="posix")
    entries = np.memmap(
        summary.index_path,
        dtype="<u8",
        mode="r+",
        offset=SIDECAR_HEADER_BYTES,
        shape=(3, 3),
    )
    entries[:2] = entries[[1, 0]]
    entries.flush()
    del entries

    ranges = query_vdif_binary_index(
        source, VdifTimestampKey(40, 17, 0), VdifTimestampKey(40, 17, 1)
    )
    assert [(item.start_frame, item.stop_frame) for item in ranges] == [
        (0, 2)
    ]


@pytest.mark.parametrize(
    "thread_ids,matched", [((0, 1), True), ((1,), False), ((), False)]
)
def test_sidecar_query_matches_any_requested_thread(
    tmp_path: Path, thread_ids: tuple[int, ...], matched: bool
) -> None:
    source = tmp_path / "one-thread.vdif"
    source.write_bytes(_frame(0))
    build_vdif_binary_index(source, backend="posix")

    ranges = query_vdif_binary_index(
        source,
        VdifTimestampKey(40, 17, 0),
        VdifTimestampKey(40, 17, 1),
        thread_ids=thread_ids,
    )

    assert [(item.start_frame, item.stop_frame) for item in ranges] == (
        [(0, 1)] if matched else []
    )


def test_cpu_binary_sidecar_sorts_unordered_physical_frames(
    tmp_path: Path,
) -> None:
    source = tmp_path / "unordered.vdif"
    source.write_bytes(b"".join(_frame(value) for value in (3, 0, 2, 1)))
    index_dir = tmp_path / "indexes"

    summary = build_vdif_binary_index(
        source,
        index_dir=index_dir,
        backend="posix",
    )

    assert not summary.source_ordered
    assert summary.frame_count == 4
    assert summary.index_path.stat().st_size == binary_index_size(
        4,
        physical_timestamp_run_count=summary.physical_timestamp_run_count,
    )
    entries = np.memmap(
        summary.index_path,
        dtype="<u8",
        mode="r",
        offset=SIDECAR_HEADER_BYTES,
        shape=(4, 3),
    )
    assert list(entries[:, 0] & np.uint64(0xFFFFFF)) == [0, 1, 2, 3]
    assert list(entries[:, 1] & np.uint64((1 << 53) - 1)) == [1, 3, 2, 0]

    ranges = query_vdif_binary_index(
        source,
        VdifTimestampKey(40, 17, 1),
        VdifTimestampKey(40, 17, 3),
        thread_ids=(0,),
        index_dir=index_dir,
    )
    assert [(item.start_frame, item.stop_frame) for item in ranges] == [
        (2, 4)
    ]


def test_index_directory_handles_duplicate_source_basenames(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first" / "recording.vdif"
    second = tmp_path / "second" / "recording.vdif"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(_frame(0) + _frame(1))
    second.write_bytes(_frame(2) + _frame(3))
    index_dir = tmp_path / "indexes"

    first_summary = build_vdif_binary_index(
        first, index_dir=index_dir, backend="posix"
    )
    second_summary = build_vdif_binary_index(
        second, index_dir=index_dir, backend="posix"
    )

    assert first_summary.index_path != second_summary.index_path
    assert first_summary.index_path.is_file()
    assert second_summary.index_path.is_file()
    assert (
        validate_vdif_binary_index(first, index_dir=index_dir).index_path
        == first_summary.index_path
    )
    assert (
        validate_vdif_binary_index(second, index_dir=index_dir).index_path
        == second_summary.index_path
    )


def test_valid_legacy_flat_index_is_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source" / "recording.vdif"
    source.parent.mkdir()
    source.write_bytes(_frame(0) + _frame(1))
    index_dir = tmp_path / "indexes"
    summary = build_vdif_binary_index(
        source, index_dir=index_dir, backend="posix"
    )
    legacy = index_dir / f"{source.name}.vdix"
    summary.index_path.replace(legacy)

    def unexpected_rebuild(*args, **kwargs):
        raise AssertionError("valid legacy sidecar was rebuilt")

    monkeypatch.setattr(builder, "_write_cpu_index", unexpected_rebuild)
    reused = build_vdif_binary_index(
        source, index_dir=index_dir, backend="posix"
    )

    assert reused.index_path == legacy


def test_stale_legacy_flat_index_does_not_block_same_named_source(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first" / "recording.vdif"
    second = tmp_path / "second" / "recording.vdif"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(_frame(0) + _frame(1))
    second.write_bytes(_frame(2) + _frame(3))
    index_dir = tmp_path / "indexes"
    first_summary = build_vdif_binary_index(
        first, index_dir=index_dir, backend="posix"
    )
    legacy = index_dir / f"{first.name}.vdix"
    first_summary.index_path.replace(legacy)

    second_summary = build_vdif_binary_index(
        second, index_dir=index_dir, backend="posix"
    )

    assert second_summary.index_path != legacy
    assert (
        validate_vdif_binary_index(first, index_dir=index_dir).index_path
        == legacy
    )
    assert (
        validate_vdif_binary_index(second, index_dir=index_dir).index_path
        == second_summary.index_path
    )


def test_binary_sidecar_vectorizes_disjoint_unordered_ranges(
    tmp_path: Path,
) -> None:
    source = tmp_path / "disjoint.vdif"
    source.write_bytes(b"".join(_frame(value) for value in (0, 2, 1, 3)))
    index_dir = tmp_path / "indexes"
    build_vdif_binary_index(source, index_dir=index_dir, backend="posix")

    ranges = query_vdif_binary_index(
        source,
        VdifTimestampKey(40, 17, 0),
        VdifTimestampKey(40, 17, 2),
        thread_ids=(0,),
        index_dir=index_dir,
    )

    assert [(item.start_frame, item.stop_frame) for item in ranges] == [
        (0, 1),
        (2, 3),
    ]


def test_binary_sidecar_preserves_invalid_frame_state(tmp_path: Path) -> None:
    source = tmp_path / "invalid.vdif"
    source.write_bytes(_frame(0) + _frame(1, invalid=True) + _frame(2))
    index_dir = tmp_path / "indexes"
    build_vdif_binary_index(source, index_dir=index_dir, backend="posix")

    timestamps, threads, frames, invalid = read_vdif_binary_frames(
        source,
        VdifTimestampKey(40, 17, 0),
        VdifTimestampKey(40, 17, 3),
        index_dir=index_dir,
    )

    assert list(timestamps & np.uint64(0xFFFFFF)) == [0, 1, 2]
    assert list(threads) == [0, 0, 0]
    assert list(frames) == [0, 1, 2]
    assert list(invalid) == [False, True, False]


def test_binary_sidecar_rejects_format_mismatch(
    tmp_path: Path,
) -> None:
    source = tmp_path / "format-mismatch.vdif"
    source.write_bytes(
        _frame(0)
        + _frame(1, station_id=(ord("B") << 8) | ord("B"))
        + _frame(2)
    )
    with pytest.raises(VdifIndexError, match="frame 1: .*changes format"):
        build_vdif_binary_index(source, backend="posix")
    assert not source.with_suffix(".vdif.vdix").exists()


def test_binary_sidecar_rejects_header_mismatch(
    tmp_path: Path,
) -> None:
    source = tmp_path / "header-mismatch.vdif"
    source.write_bytes(
        _frame(0) + _frame(1, declared_frame_size=48) + _frame(2)
    )
    with pytest.raises(VdifIndexError, match="frame 1: frame length"):
        build_vdif_binary_index(source, backend="posix")
    assert not source.with_suffix(".vdif.vdix").exists()


def test_binary_sidecar_rejects_changed_source(tmp_path: Path) -> None:
    source = tmp_path / "changed.vdif"
    source.write_bytes(_frame(0) + _frame(1))
    build_vdif_binary_index(source, backend="posix")
    source.write_bytes(_frame(0) + _frame(2))

    try:
        validate_vdif_binary_index(source)
    except ValueError as error:
        assert "stale" in str(error)
    else:
        raise AssertionError("stale binary sidecar was accepted")


@pytest.mark.gpu
def test_gpu_cufile_sidecar_matches_cpu_sidecar(tmp_path: Path) -> None:
    cupy = pytest.importorskip("cupy")
    try:
        device_count = cupy.cuda.runtime.getDeviceCount()
    except cupy.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA runtime unavailable")
    if device_count < 1:
        pytest.skip("no CUDA device")
    source = tmp_path / "gpu-unordered.vdif"
    # Equal-time threads alternate physical order to check stable sorting.
    source.write_bytes(
        b"".join(
            _frame(
                value // 2,
                thread_id=value % 2,
                invalid=value in (4, 5),
            )
            for value in (7, 0, 6, 1, 5, 2, 4, 3)
        )
    )
    cpu_dir = tmp_path / "cpu"
    gpu_dir = tmp_path / "gpu"
    cpu = build_vdif_binary_index(source, index_dir=cpu_dir, backend="posix")
    gpu = build_vdif_binary_index(
        source,
        index_dir=gpu_dir,
        backend="cufile",
        chunk_bytes=160,
    )

    assert gpu.source_ordered == cpu.source_ordered
    cpu_entries = np.memmap(
        cpu.index_path,
        dtype="<u8",
        mode="r",
        offset=SIDECAR_HEADER_BYTES,
        shape=(8, 3),
    )
    gpu_entries = np.memmap(
        gpu.index_path,
        dtype="<u8",
        mode="r",
        offset=SIDECAR_HEADER_BYTES,
        shape=(8, 3),
    )
    np.testing.assert_array_equal(gpu_entries, cpu_entries)
    assert gpu.thread_metadata == cpu.thread_metadata
    assert gpu.rejected_format_frame_count == 0
    assert cpu.rejected_format_frame_count == 0
    assert gpu.rejected_header_frame_count == 0
    assert cpu.rejected_header_frame_count == 0

    filler_source = tmp_path / "gpu-filler.vdif"
    filler_source.write_bytes(
        _frame(0)
        + _recorded_filler()
        + _recorded_filler()
        + _frame(3)
        + _frame(4)
    )
    cpu_filler = build_vdif_binary_index(
        filler_source, index_dir=cpu_dir, backend="posix"
    )
    gpu_filler = build_vdif_binary_index(
        filler_source,
        index_dir=gpu_dir,
        backend="cufile",
        chunk_bytes=80,
    )
    assert query_vdif_recorded_filler_runs(gpu_filler) == ((1, 3),)
    assert query_vdif_recorded_filler_runs(gpu_filler) == (
        query_vdif_recorded_filler_runs(cpu_filler)
    )
    assert gpu_filler.recorded_filler_classifier == (
        cpu_filler.recorded_filler_classifier
    )


def _replace_with_v2(summary):
    """Encode the documented v2 two-word layout without using its reader."""
    import json
    import struct

    document = sidecar._read_document(summary.index_path)
    document["schema_version"] = 2
    document["entry_layout"] = "timestamp-u64,thread-and-physical-frame-u64"
    document.pop("physical_timestamp_run_layout", None)
    document.pop("physical_timestamp_run_count", None)
    document.pop("physical_timestamp_run_offset_bytes", None)
    encoded = json.dumps(document).encode()
    entries = np.array(sidecar._summary_entries(summary)[:, :2], copy=True)
    header = struct.pack("<8sII", sidecar.SIDECAR_MAGIC, 2, len(encoded))
    content = header + encoded
    content += bytes(SIDECAR_HEADER_BYTES - len(content))
    content += entries.tobytes()
    content += bytes(
        binary_index_size(summary.entry_count, schema_version=2)
        - len(content)
    )
    # Atomic replacement also exercises schema-aware mmap cache keys.
    temporary = summary.index_path.with_suffix(".v2.tmp")
    temporary.write_bytes(content)
    temporary.replace(summary.index_path)


def test_v2_reader_preserves_ambiguous_invalidity_and_force_upgrades(
    tmp_path,
):
    source = tmp_path / "v2.vdif"
    source.write_bytes(_frame(0) + _frame(1, invalid=True) + _frame(2))
    summary = build_vdif_binary_index(source, backend="posix")
    _replace_with_v2(summary)
    legacy = validate_vdif_binary_index(source)
    assert legacy.schema_version == 2
    assert legacy.entry_bytes == 16
    assert not legacy.has_frame_states
    assert legacy.to_dict()["schema_version"] == 2
    timestamps, threads, physical, states = (
        sidecar.read_vdif_binary_frame_states(
            source, VdifTimestampKey(40, 17, 0), VdifTimestampKey(40, 17, 3)
        )
    )
    np.testing.assert_array_equal(timestamps & np.uint64(0xFFFFFF), [0, 1, 2])
    np.testing.assert_array_equal(threads, [0, 0, 0])
    np.testing.assert_array_equal(physical, [0, 1, 2])
    np.testing.assert_array_equal(states, [0, 8, 0])
    _, _, _, invalid = read_vdif_binary_frames(
        source, VdifTimestampKey(40, 17, 0), VdifTimestampKey(40, 17, 3)
    )
    np.testing.assert_array_equal(invalid, [False, True, False])
    ranges = query_vdif_binary_index(
        source, VdifTimestampKey(40, 17, 1), VdifTimestampKey(40, 17, 3)
    )
    assert [(item.start_frame, item.stop_frame) for item in ranges] == [
        (1, 3)
    ]
    assert (
        build_vdif_binary_index(source, backend="posix").schema_version == 2
    )
    upgraded = build_vdif_binary_index(source, backend="posix", force=True)
    assert upgraded.schema_version == 4
    _, _, _, states = sidecar.read_vdif_binary_frame_states(
        source, VdifTimestampKey(40, 17, 0), VdifTimestampKey(40, 17, 3)
    )
    np.testing.assert_array_equal(states, [0, 1, 0])


@pytest.mark.parametrize("state_value", [0, 8, 1 << 40])
def test_v3_reader_rejects_inconsistent_or_unknown_state(
    tmp_path, state_value
):
    source = tmp_path / "bad-state.vdif"
    source.write_bytes(_frame(0) + _frame(1, invalid=True))
    summary = build_vdif_binary_index(source, backend="posix")
    entries = np.memmap(
        summary.index_path,
        dtype="<u8",
        mode="r+",
        offset=SIDECAR_HEADER_BYTES,
        shape=(2, 3),
    )
    entries[1, 2] = state_value
    entries.flush()
    with pytest.raises(sidecar.VdifIndexError, match="frame state"):
        sidecar.read_vdif_binary_frame_states(
            source, VdifTimestampKey(40, 17, 0), VdifTimestampKey(40, 17, 2)
        )


def test_reader_rejects_disagreeing_schema_versions(tmp_path):
    import struct

    source = tmp_path / "version.vdif"
    source.write_bytes(_frame(0))
    summary = build_vdif_binary_index(source, backend="posix")
    with summary.index_path.open("r+b") as stream:
        stream.seek(8)
        stream.write(struct.pack("<I", 2))
    sidecar._read_document_cached.cache_clear()
    sidecar._validate_cached.cache_clear()
    with pytest.raises(
        sidecar.VdifIndexError, match="version fields disagree"
    ):
        validate_vdif_binary_index(source)
