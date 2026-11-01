# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

import cuphoton.xbeam.adapters.vdif.recording as recording_module
from cuphoton.xbeam.adapters.radio.models import (
    ClockPolynomial,
    Datastream,
    Frequency,
    Job,
    MJDTime,
    RecordedBand,
    SamplingKind,
    Sideband,
    Station,
)
from cuphoton.xbeam.adapters.vdif.ingest import (
    SampleInterval,
    plan_vdif_ingest,
)
from cuphoton.xbeam.adapters.vdif.recording import (
    VdifFileRecordingSource,
    VdifPayloadFilePlan,
    VdifPayloadPlan,
    VdifSourceRange,
    VdifStreamingPlan,
    VdifThreadWindow,
    materialize_vdif_payload_plans,
)
from cuphoton.xbeam.adapters.vdif.vdif import (
    DuplicatePolicy,
    DuplicateVdifFrameError,
    VdifError,
    VdifHeader,
    absolute_frame_slot,
    build_thread_timelines,
    iter_vdif_frames,
)
from cuphoton.xbeam.adapters.vdif.vdif_sidecar import SIDECAR_HEADER_BYTES
from cuphoton.xbeam.adapters.vdif.vdif_sidecar_builder import (
    build_vdif_binary_index,
)
from cuphoton.xbeam.errors import ConfigurationError
from cuphoton.xbeam.profiles import ANALYTIC_V1, INTEROP_REAL_USB_V1


def _frame(
    *,
    payload: bytes,
    frame_number: int,
    invalid: bool = False,
    complex_data: bool = False,
    channels: int = 2,
    thread_id: int = 5,
    seconds_from_epoch: int = 0,
    station: str = "AA",
    reference_epoch: int = 40,
) -> bytes:
    log2_channels = channels.bit_length() - 1
    word0 = (int(invalid) << 31) | seconds_from_epoch
    word1 = frame_number | (reference_epoch << 24)
    word2 = ((32 + len(payload)) // 8) | (log2_channels << 24)
    word3 = (
        (ord(station[0]) << 8)
        | ord(station[1])
        | (thread_id << 16)
        | ((2 - 1) << 26)
        | (int(complex_data) << 31)
    )
    header = np.asarray((word0, word1, word2, word3, 0, 0, 0, 0), dtype="<u4")
    return header.tobytes() + payload


def test_payload_materialization_reads_only_selected_frame_window(
    tmp_path: Path,
) -> None:
    payloads = [bytes((index,)) * 8 for index in range(5)]
    path = tmp_path / "window.vdif"
    path.write_bytes(b"".join(bytes(32) + payload for payload in payloads))
    mapping = np.asarray([-1, -1, 0, 1, -1], dtype=np.int32)
    plan = VdifPayloadPlan(
        40,
        8,
        2,
        (VdifPayloadFilePlan(path, 5, mapping),),
    )
    (actual,) = materialize_vdif_payload_plans([plan])

    np.testing.assert_array_equal(
        actual.reshape(2, 8),
        np.asarray([list(payloads[2]), list(payloads[3])], dtype=np.uint8),
    )


@pytest.mark.parametrize("premature_eof", [False, True])
def test_payload_materialization_handles_short_reads(
    tmp_path: Path, monkeypatch, premature_eof: bool
) -> None:
    path = tmp_path / "short-read.vdif"
    payloads = [bytes((index,)) * 8 for index in range(3)]
    path.write_bytes(b"".join(bytes(32) + payload for payload in payloads))
    plan = VdifPayloadPlan(
        40,
        8,
        3,
        (VdifPayloadFilePlan(path, 3, np.arange(3, dtype=np.int32)),),
    )
    original_open = Path.open

    class ShortReader:
        def __init__(self, stream):
            self.stream = stream
            self.calls = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def readinto(self, buffer):
            self.calls += 1
            if premature_eof and self.calls == 3:
                return 0
            return self.stream.readinto(buffer[:7])

    def short_open(selected, *args, **kwargs):
        stream = original_open(selected, *args, **kwargs)
        return ShortReader(stream) if selected == path else stream

    monkeypatch.setattr(Path, "open", short_open)
    if premature_eof:
        with pytest.raises(OSError, match="expected 120 bytes, got 14"):
            materialize_vdif_payload_plans([plan])
    else:
        (actual,) = materialize_vdif_payload_plans([plan])
        np.testing.assert_array_equal(
            actual, np.frombuffer(b"".join(payloads), dtype=np.uint8)
        )


def test_presented_sidecar_slots_compact_runs_and_drop_unpublished_frames():
    slots = np.asarray([100, 101, 103, 104, 107], dtype=np.int64)
    intervals = ((100, 102, 200), (103, 105, 202))

    actual = recording_module._presented_sidecar_slots(slots, intervals)

    np.testing.assert_array_equal(actual, [200, 201, 202, 203, -1])


def test_presented_physical_sidecar_slots_ignore_header_order():
    source_indices = np.asarray([0, 0, 0, 1, 1], dtype=np.int32)
    physical_frames = np.asarray([10, 11, 13, 10, 11], dtype=np.int64)
    intervals = ((0, 10, 12, 200), (1, 11, 12, 300))

    actual = recording_module._presented_physical_sidecar_slots(
        source_indices, physical_frames, intervals
    )

    np.testing.assert_array_equal(actual, [200, 201, -1, -1, 300])


def test_presented_physical_sidecar_slots_match_interval_semantics():
    source_indices = np.asarray([1, 0, 1, 0, 0, 1, 0, 1], dtype=np.int32)
    physical_frames = np.asarray([7, 12, 5, 10, 16, 6, 11, 4], dtype=np.int64)
    intervals = (
        (0, 10, 12, 100),
        (0, 14, 16, 104),
        (1, 5, 7, 200),
    )

    actual = recording_module._presented_physical_sidecar_slots(
        source_indices, physical_frames, intervals
    )
    rows, slots = recording_module._presented_physical_sidecar_mapping(
        source_indices, physical_frames, intervals
    )

    # Starts are inclusive, stops are exclusive, gaps remain unmapped, and
    # equal physical coordinates from different files are independent.
    np.testing.assert_array_equal(
        actual,
        [-1, -1, 200, 100, -1, 201, 101, -1],
    )
    np.testing.assert_array_equal(rows, [3, 6, 2, 5])
    np.testing.assert_array_equal(slots, [100, 101, 200, 201])


def test_presented_physical_sidecar_slots_preserve_later_overlap():
    source_indices = np.asarray([0, 0, 0, 0], dtype=np.int32)
    physical_frames = np.asarray([9, 10, 11, 12], dtype=np.int64)
    intervals = ((0, 10, 13, 100), (0, 11, 12, 300))

    actual = recording_module._presented_physical_sidecar_slots(
        source_indices, physical_frames, intervals
    )

    np.testing.assert_array_equal(actual, [-1, 100, 300, 102])


def test_physical_sidecar_overrides_move_and_drop_sparse_frames():
    slots = np.asarray([100, 101, 102, 103, 104], dtype=np.int64)
    source_indices = np.zeros(5, dtype=np.int32)
    physical_frames = np.asarray([10, 11, 12, 13, 14], dtype=np.int64)

    actual = recording_module._apply_physical_sidecar_overrides(
        slots,
        source_indices,
        physical_frames,
        ((0, 10, 12, 102), (0, 12, 14, -1)),
    )

    np.testing.assert_array_equal(actual, [102, 103, -1, -1, 104])


def test_presented_physical_sidecar_slots_empty_plan_is_unmapped():
    source_indices = np.asarray([0, 1, 0], dtype=np.int32)
    physical_frames = np.asarray([10, 10, 11], dtype=np.int64)

    actual = recording_module._presented_physical_sidecar_slots(
        source_indices, physical_frames, ()
    )
    rows, slots = recording_module._presented_physical_sidecar_mapping(
        source_indices, physical_frames, ()
    )

    np.testing.assert_array_equal(actual, [-1, -1, -1])
    np.testing.assert_array_equal(rows, np.empty(0, dtype=np.int64))
    np.testing.assert_array_equal(slots, np.empty(0, dtype=np.int64))


def test_presented_physical_sidecar_mapping_reuses_out_of_order_frames():
    source_indices = np.asarray([0, 0, 0], dtype=np.int32)
    physical_frames = np.asarray([10, 11, 12], dtype=np.int64)
    intervals = (
        (0, 12, 13, 200),
        (0, 10, 12, 201),
        (0, 11, 12, 203),
    )

    rows, slots = recording_module._presented_physical_sidecar_mapping(
        source_indices, physical_frames, intervals
    )

    np.testing.assert_array_equal(rows, [2, 0, 1, 1])
    np.testing.assert_array_equal(slots, [200, 201, 202, 203])


def test_payload_materialization_supports_sparse_repeated_source_frames(
    tmp_path: Path,
) -> None:
    payloads = [bytes((index,)) * 8 for index in range(4)]
    path = tmp_path / "sparse-repeat.vdif"
    path.write_bytes(b"".join(bytes(32) + payload for payload in payloads))
    plan = VdifPayloadPlan(
        40,
        8,
        4,
        (
            VdifPayloadFilePlan(
                path,
                4,
                np.asarray([0, 1, 2, 3], dtype=np.int32),
                source_frame_indices=np.asarray([3, 1, 3, 0], dtype=np.int64),
            ),
        ),
    )

    (actual,) = materialize_vdif_payload_plans([plan], target_chunk_bytes=40)

    np.testing.assert_array_equal(
        actual.reshape(4, 8),
        np.asarray(
            [
                list(payloads[3]),
                list(payloads[1]),
                list(payloads[3]),
                list(payloads[0]),
            ],
            dtype=np.uint8,
        ),
    )


def test_physical_payload_mapping_densifies_injective_sources() -> None:
    destinations, sources = recording_module._canonical_payload_file_mapping(
        4,
        np.asarray([2, 0, 3], dtype=np.int64),
        np.asarray([0, 1, 2], dtype=np.int32),
    )

    np.testing.assert_array_equal(destinations, [1, -1, 0, 2])
    assert sources is None


def test_physical_payload_mapping_keeps_reused_sources_sparse() -> None:
    source_frames = np.asarray([2, 0, 2], dtype=np.int64)
    destination_slots = np.asarray([0, 1, 2], dtype=np.int32)

    destinations, sources = recording_module._canonical_payload_file_mapping(
        4, source_frames, destination_slots
    )

    np.testing.assert_array_equal(destinations, destination_slots)
    np.testing.assert_array_equal(sources, source_frames)


@pytest.mark.parametrize(
    "destination_slots",
    [
        np.asarray([-2], dtype=np.int32),
        np.asarray([2], dtype=np.int32),
        np.asarray([0.0], dtype=np.float64),
    ],
)
def test_payload_materialization_rejects_invalid_destination_slots(
    tmp_path: Path, destination_slots: np.ndarray
) -> None:
    file_plan = VdifPayloadFilePlan(
        tmp_path / "input.vdif", 1, destination_slots
    )
    plan = VdifPayloadPlan(40, 8, 2, (file_plan,))

    with pytest.raises((TypeError, ValueError), match="destination slot"):
        materialize_vdif_payload_plans([plan])


def test_payload_materialization_rejects_duplicate_destination_slots(
    tmp_path: Path,
) -> None:
    files = (
        VdifPayloadFilePlan(
            tmp_path / "input-a.vdif",
            1,
            np.asarray([0], dtype=np.int32),
        ),
        VdifPayloadFilePlan(
            tmp_path / "input-b.vdif",
            1,
            np.asarray([0], dtype=np.int32),
        ),
    )
    plan = VdifPayloadPlan(40, 8, 2, files)

    with pytest.raises(ValueError, match="duplicate VDIF destination slot"):
        materialize_vdif_payload_plans([plan])


def test_bulk_materializer_rejects_streaming_plan_before_file_access(
    tmp_path: Path,
) -> None:
    template = VdifHeader.from_bytes(_frame(payload=bytes(8), frame_number=0))
    validity = np.zeros(1, dtype=np.bool_)
    plan = VdifPayloadPlan(
        40,
        8,
        1,
        (VdifPayloadFilePlan(tmp_path / "missing.vdif", 1, None),),
        VdifStreamingPlan(
            5,
            1,
            absolute_frame_slot(template, 1),
            template,
            DuplicatePolicy.FIRST_VALID,
            validity,
        ),
    )

    with pytest.raises(RuntimeError, match="GPU uploader"):
        materialize_vdif_payload_plans([plan])


def test_streaming_source_rejects_more_bands_than_packed_channels(
    tmp_path: Path,
) -> None:
    path = tmp_path / "one-channel.vdif"
    raw = _frame(payload=bytes(8), frame_number=0, channels=1)
    path.write_bytes(raw)
    template = VdifHeader.from_bytes(raw)
    start_slot = absolute_frame_slot(template, 1)
    datastream = Datastream(
        identifier=0,
        station_id=0,
        sampling=SamplingKind.REAL,
        quantization_bits=2,
        frame_size_bytes=40,
        source_files=(str(path),),
        recorded_bands=(
            RecordedBand(0, 0, "R", thread_id=5),
            RecordedBand(1, 0, "L", thread_id=5),
        ),
        sample_rate_hz=32.0,
    )
    stat = path.stat()
    source = VdifFileRecordingSource(
        _job(datastream, 32.0),
        ANALYTIC_V1,
        source_ranges={
            0: (
                VdifSourceRange(
                    path,
                    0,
                    1,
                    stat.st_size,
                    stat.st_mtime_ns,
                    stat.st_dev,
                    stat.st_ino,
                ),
            )
        },
        thread_windows={
            0: {
                5: VdifThreadWindow(
                    template,
                    1,
                    start_slot,
                    start_slot + 1,
                )
            }
        },
        defer_ordered_headers=True,
    )

    with pytest.raises(ConfigurationError, match="maps 2 logical bands"):
        list(source.batches(datastream))


def _job(datastream: Datastream, sample_rate_hz: float) -> Job:
    frequency = Frequency(0, 0.0, sample_rate_hz, Sideband.USB, 4)
    return Job(
        start_time=MJDTime(58849, Decimal(0)),
        execution_duration_s=Decimal(1),
        integration_duration_s=Decimal(1),
        subintegration_duration_s=Decimal(1),
        fringe_approximation_order=1,
        stations=(
            Station(
                0,
                "AA",
                ClockPolynomial(MJDTime(58849, Decimal(0)), (0.0,)),
            ),
        ),
        frequencies=(frequency,),
        datastreams=(datastream,),
        baselines=(),
        scans=(),
    )


def test_file_source_reuses_headers_for_identical_extent(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "shared.vdif"
    path.write_bytes(_frame(payload=bytes(8), frame_number=0))
    datastream = Datastream(
        identifier=0,
        station_id=0,
        sampling=SamplingKind.REAL,
        quantization_bits=2,
        frame_size_bytes=40,
        source_files=(str(path),),
        recorded_bands=(RecordedBand(0, 0, "R", thread_id=5),),
        sample_rate_hz=16.0,
    )
    source = VdifFileRecordingSource(_job(datastream, 16.0), ANALYTIC_V1)
    original = recording_module._bulk_vdif_frames
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(recording_module, "_bulk_vdif_frames", counted)

    assert len(list(source.batches(datastream))) == 1
    assert len(list(source.batches(replace(datastream, identifier=1)))) == 1
    assert calls == 1


@pytest.mark.parametrize(
    "profile_policy", ("first_valid", "last_valid", "first", "last")
)
@pytest.mark.parametrize("order", ["unordered", "ordered", "permuted-ties"])
def test_binary_sidecar_direct_plan_matches_generic_header_plan(
    tmp_path: Path, profile_policy: str, order: str
) -> None:
    path = tmp_path / "unordered-duplicates.vdif"
    path.write_bytes(
        b"".join(
            (
                _frame(
                    payload=bytes((0x11,)) * 8,
                    frame_number=1,
                    invalid=True,
                ),
                _frame(payload=bytes((0x20,)) * 8, frame_number=0),
                _frame(payload=bytes((0x31,)) * 8, frame_number=1),
                _frame(payload=bytes((0x42,)) * 8, frame_number=2),
            )
        )
    )
    if order != "unordered":
        raw = path.read_bytes()
        frames = [
            raw[offset : offset + 40] for offset in range(0, len(raw), 40)
        ]
        path.write_bytes(
            b"".join(
                sorted(
                    frames,
                    key=lambda frame: (
                        VdifHeader.from_bytes(frame).frame_number
                    ),
                )
            )
        )
    datastream = Datastream(
        identifier=0,
        station_id=0,
        sampling=SamplingKind.REAL,
        quantization_bits=2,
        frame_size_bytes=40,
        source_files=(str(path),),
        recorded_bands=(RecordedBand(0, 0, "R", thread_id=5),),
        sample_rate_hz=64.0,
    )
    job = _job(datastream, 64.0)
    profile = replace(ANALYTIC_V1, duplicate_policy=profile_policy)
    stat = path.stat()
    source_ranges = {
        0: (
            VdifSourceRange(
                path,
                0,
                4,
                stat.st_size,
                stat.st_mtime_ns,
                stat.st_dev,
                stat.st_ino,
            ),
        )
    }
    template = VdifHeader.from_bytes(_frame(payload=bytes(8), frame_number=0))
    start_slot = absolute_frame_slot(template, 4)
    windows = {
        0: {5: VdifThreadWindow(template, 4, start_slot, start_slot + 4)}
    }
    index_dir = tmp_path / "sidecars"
    summary = build_vdif_binary_index(
        path, index_dir=index_dir, backend="posix"
    )
    if order == "permuted-ties":
        entries = np.memmap(
            summary.index_path,
            dtype="<u8",
            mode="r+",
            offset=SIDECAR_HEADER_BYTES,
            shape=(4, summary.entry_bytes // 8),
        )
        entries[1:3] = entries[[2, 1]]
        entries.flush()
        del entries

    expected = list(
        VdifFileRecordingSource(
            job,
            profile,
            source_ranges=source_ranges,
            thread_windows=windows,
        ).batches(datastream)
    )
    actual = list(
        VdifFileRecordingSource(
            job,
            profile,
            source_ranges=source_ranges,
            thread_windows=windows,
            binary_sidecars=True,
            binary_index_dir=index_dir,
        ).batches(datastream)
    )

    assert len(actual) == len(expected) == 1
    assert actual[0].start_time == expected[0].start_time
    assert actual[0].packed_layout == expected[0].packed_layout
    np.testing.assert_array_equal(actual[0].validity, expected[0].validity)
    np.testing.assert_array_equal(
        np.asarray(actual[0].samples), np.asarray(expected[0].samples)
    )


def test_binary_sidecar_presents_physical_frames_on_destination_grid(
    tmp_path: Path,
) -> None:
    path = tmp_path / "timestamp-gap.vdif"
    path.write_bytes(
        b"".join(
            (
                _frame(payload=bytes((0x10,)) * 8, frame_number=0),
                _frame(payload=bytes((0x11,)) * 8, frame_number=1),
                _frame(payload=bytes((0x13,)) * 8, frame_number=3),
                _frame(
                    payload=bytes((0x14,)) * 8,
                    frame_number=0,
                    seconds_from_epoch=1,
                ),
            )
        )
    )
    datastream = Datastream(
        identifier=0,
        station_id=0,
        sampling=SamplingKind.REAL,
        quantization_bits=2,
        frame_size_bytes=40,
        source_files=(str(path),),
        recorded_bands=(RecordedBand(0, 0, "R", thread_id=5),),
        sample_rate_hz=64.0,
    )
    job = _job(datastream, 64.0)
    stat = path.stat()
    source_ranges = {
        0: (
            VdifSourceRange(
                path,
                0,
                4,
                stat.st_size,
                stat.st_mtime_ns,
                stat.st_dev,
                stat.st_ino,
            ),
        )
    }
    template = VdifHeader.from_bytes(_frame(payload=bytes(8), frame_number=0))
    start_slot = absolute_frame_slot(template, 4)
    destination_windows = {
        0: {5: VdifThreadWindow(template, 4, start_slot, start_slot + 4)}
    }
    source_windows = {
        0: {5: VdifThreadWindow(template, 4, start_slot, start_slot + 5)}
    }
    index_dir = tmp_path / "sidecars"
    build_vdif_binary_index(path, index_dir=index_dir, backend="posix")

    (actual,) = list(
        VdifFileRecordingSource(
            job,
            ANALYTIC_V1,
            source_ranges=source_ranges,
            thread_windows=destination_windows,
            source_thread_windows=source_windows,
            binary_sidecars=True,
            binary_index_dir=index_dir,
            frame_presentation_plans={
                0: (
                    (start_slot, start_slot + 2, start_slot),
                    (start_slot + 3, start_slot + 5, start_slot + 2),
                )
            },
        ).batches(datastream)
    )

    np.testing.assert_array_equal(actual.validity, [True] * 4)
    np.testing.assert_array_equal(
        np.asarray(actual.samples).reshape(4, 8),
        np.asarray(
            [
                [0x10] * 8,
                [0x11] * 8,
                [0x13] * 8,
                [0x14] * 8,
            ],
            dtype=np.uint8,
        ),
    )


def test_binary_sidecar_physical_query_loads_native_out_of_window_frame_once(
    tmp_path: Path,
) -> None:
    path = tmp_path / "physical-out-of-window.vdif"
    native = _frame(payload=bytes((0x10,)) * 8, frame_number=0)
    destination = _frame(
        payload=bytes((0x20,)) * 8,
        frame_number=0,
        seconds_from_epoch=1,
    )
    path.write_bytes(native + destination)
    datastream = Datastream(
        identifier=0,
        station_id=0,
        sampling=SamplingKind.REAL,
        quantization_bits=2,
        frame_size_bytes=40,
        source_files=(str(path),),
        recorded_bands=(RecordedBand(0, 0, "R", thread_id=5),),
        sample_rate_hz=64.0,
    )
    stat = path.stat()
    requested = VdifSourceRange(
        path,
        0,
        1,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_dev,
        stat.st_ino,
    )
    template = VdifHeader.from_bytes(destination)
    destination_slot = absolute_frame_slot(template, 4)
    windows = {
        0: {
            5: VdifThreadWindow(
                template, 4, destination_slot, destination_slot + 1
            )
        }
    }
    index_dir = tmp_path / "sidecars"
    build_vdif_binary_index(path, index_dir=index_dir, backend="posix")

    (actual,) = list(
        VdifFileRecordingSource(
            _job(datastream, 64.0),
            ANALYTIC_V1,
            source_ranges={0: (requested,)},
            thread_windows=windows,
            source_thread_windows=windows,
            binary_sidecars=True,
            binary_index_dir=index_dir,
            physical_frame_presentation_plans={
                0: ((0, 0, 1, destination_slot),)
            },
        ).batches(datastream)
    )

    file_plan = actual.samples.files[0]
    np.testing.assert_array_equal(file_plan.destination_slots, [0])
    assert file_plan.source_frame_indices is None
    np.testing.assert_array_equal(actual.validity, [True])
    np.testing.assert_array_equal(
        np.asarray(actual.samples).reshape(1, 8),
        np.asarray([[0x10] * 8], dtype=np.uint8),
    )


def test_binary_sidecar_physical_presentation_reuses_out_of_order_frames(
    tmp_path: Path,
) -> None:
    path = tmp_path / "physical-reuse.vdif"
    path.write_bytes(
        b"".join(
            _frame(
                payload=bytes((0x10 + frame_number,)) * 8,
                frame_number=frame_number,
            )
            for frame_number in range(3)
        )
    )
    datastream = Datastream(
        identifier=0,
        station_id=0,
        sampling=SamplingKind.REAL,
        quantization_bits=2,
        frame_size_bytes=40,
        source_files=(str(path),),
        recorded_bands=(RecordedBand(0, 0, "R", thread_id=5),),
        sample_rate_hz=64.0,
    )
    stat = path.stat()
    template = VdifHeader.from_bytes(path.read_bytes()[:40])
    destination = absolute_frame_slot(template, 4)
    windows = {
        0: {5: VdifThreadWindow(template, 4, destination, destination + 4)}
    }
    index_dir = tmp_path / "sidecars"
    build_vdif_binary_index(path, index_dir=index_dir, backend="posix")

    (actual,) = list(
        VdifFileRecordingSource(
            _job(datastream, 64.0),
            ANALYTIC_V1,
            source_ranges={
                0: (
                    VdifSourceRange(
                        path,
                        0,
                        3,
                        stat.st_size,
                        stat.st_mtime_ns,
                        stat.st_dev,
                        stat.st_ino,
                    ),
                )
            },
            thread_windows=windows,
            source_thread_windows=windows,
            binary_sidecars=True,
            binary_index_dir=index_dir,
            physical_frame_presentation_plans={
                0: (
                    (0, 2, 3, destination),
                    (0, 0, 2, destination + 1),
                    (0, 1, 2, destination + 3),
                )
            },
        ).batches(datastream)
    )

    file_plan = actual.samples.files[0]
    np.testing.assert_array_equal(file_plan.destination_slots, [0, 1, 2, 3])
    np.testing.assert_array_equal(
        file_plan.source_frame_indices, [2, 0, 1, 1]
    )
    np.testing.assert_array_equal(actual.validity, [True] * 4)
    np.testing.assert_array_equal(
        np.asarray(actual.samples).reshape(4, 8),
        np.asarray(
            [[0x12] * 8, [0x10] * 8, [0x11] * 8, [0x11] * 8],
            dtype=np.uint8,
        ),
    )


def test_binary_sidecar_physical_query_requires_plan_coverage(
    tmp_path: Path,
) -> None:
    path = tmp_path / "physical-uncovered.vdif"
    frame = _frame(payload=bytes(8), frame_number=0)
    path.write_bytes(frame)
    build_vdif_binary_index(path, backend="posix")
    template = VdifHeader.from_bytes(frame)
    start_slot = absolute_frame_slot(template, 4)
    stat = path.stat()

    with pytest.raises(VdifError, match="does not cover"):
        recording_module._binary_sidecar_frames(
            Datastream(
                identifier=0,
                station_id=0,
                sampling=SamplingKind.REAL,
                quantization_bits=2,
                frame_size_bytes=40,
                source_files=(str(path),),
                recorded_bands=(RecordedBand(0, 0, "R", thread_id=5),),
                sample_rate_hz=64.0,
            ),
            (
                VdifSourceRange(
                    path,
                    0,
                    1,
                    stat.st_size,
                    stat.st_mtime_ns,
                    stat.st_dev,
                    stat.st_ino,
                ),
            ),
            {5: VdifThreadWindow(template, 4, start_slot, start_slot + 1)},
            None,
            ((1, 0, 1, start_slot),),
        )


def test_binary_sidecar_plan_accepts_threads_in_separate_files(
    tmp_path: Path,
) -> None:
    paths = (tmp_path / "thread-5.vdif", tmp_path / "thread-6.vdif")
    for thread_id, path in zip((5, 6), paths, strict=True):
        path.write_bytes(
            _frame(
                payload=bytes((thread_id,)) * 8,
                frame_number=0,
                thread_id=thread_id,
                channels=1,
            )
        )
        build_vdif_binary_index(path, backend="posix")
    datastream = Datastream(
        identifier=0,
        station_id=0,
        sampling=SamplingKind.REAL,
        quantization_bits=2,
        frame_size_bytes=40,
        source_files=tuple(map(str, paths)),
        recorded_bands=(
            RecordedBand(0, 0, "R", thread_id=5),
            RecordedBand(1, 0, "L", thread_id=6),
        ),
        sample_rate_hz=32.0,
    )
    job = _job(datastream, 32.0)
    plan = plan_vdif_ingest(
        job,
        intervals={(0, band): SampleInterval(0, 32) for band in (0, 1)},
        max_read_bytes=1024,
        assume_ordered=False,
        index_dir=None,
    )
    assert {item.path for item in plan.source_ranges[0]} == set(paths)
    batches = list(
        VdifFileRecordingSource(
            job,
            ANALYTIC_V1,
            source_ranges=plan.source_ranges,
            thread_windows=plan.thread_windows,
            binary_sidecars=True,
        ).batches(datastream)
    )
    assert len(batches) == 2
    for thread_id, batch in zip((5, 6), batches, strict=True):
        assert np.count_nonzero(batch.validity) == 1
        payloads = batch.samples.materialize().reshape(-1, 8)
        np.testing.assert_array_equal(
            payloads[batch.validity], [[thread_id] * 8]
        )
        assert not payloads[~batch.validity].any()


@pytest.mark.parametrize(
    ("frame_count", "buffer_size"), [(0, 0), (2, 80), (4, 160), (5, 160)]
)
def test_bulk_header_scan_bounds_and_reuses_buffer_across_partial_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frame_count: int,
    buffer_size: int,
) -> None:
    path = tmp_path / "partial.vdif"
    path.write_bytes(
        b"".join(
            _frame(payload=bytes((index,)) * 8, frame_number=index)
            for index in range(frame_count + 2)
        )
    )
    datastream = Datastream(
        identifier=0,
        station_id=0,
        sampling=SamplingKind.REAL,
        quantization_bits=2,
        frame_size_bytes=40,
        source_files=(str(path),),
        recorded_bands=(RecordedBand(0, 0, "R", thread_id=5),),
        sample_rate_hz=16.0,
    )
    original_pread = recording_module.os.pread
    buffer_ids = []
    buffer_sizes = []

    def partial_preadv(fd, buffers, offset):
        target = buffers[0]
        buffer_ids.append(id(target.obj))
        buffer_sizes.append(len(target.obj))
        received = original_pread(fd, min(17, len(target)), offset)
        target[: len(received)] = received
        return len(received)

    monkeypatch.setattr(recording_module.os, "preadv", partial_preadv)
    monkeypatch.setattr(
        recording_module, "_VDIF_HEADER_SCAN_CHUNK_BYTES", 160
    )

    frames = recording_module._bulk_vdif_frames(
        datastream, (VdifSourceRange(path, 1, frame_count),)
    )

    assert frames.count == frame_count
    if frame_count:
        assert len(buffer_ids) > 1
        assert len(set(buffer_ids)) == 1
        assert set(buffer_sizes) == {buffer_size}
    else:
        assert not buffer_ids
    np.testing.assert_array_equal(
        frames.words[:, 1] & np.uint32(0x00FF_FFFF),
        np.arange(1, frame_count + 1),
    )
    np.testing.assert_array_equal(
        frames.offsets, np.arange(1, frame_count + 1) * 40
    )


def test_bounded_frame_cache_key_includes_declared_sources(
    tmp_path: Path,
) -> None:
    path = tmp_path / "shared.vdif"
    datastream = Datastream(
        identifier=0,
        station_id=0,
        sampling=SamplingKind.REAL,
        quantization_bits=2,
        frame_size_bytes=40,
        source_files=(str(path),),
        recorded_bands=(RecordedBand(0, 0, "R", thread_id=5),),
        sample_rate_hz=16.0,
    )
    requested = (VdifSourceRange(path, 0, 1),)

    first = recording_module._bulk_frame_cache_key(datastream, requested)
    second = recording_module._bulk_frame_cache_key(
        replace(datastream, source_files=(str(tmp_path / "other.vdif"),)),
        requested,
    )

    assert first != second


@pytest.mark.parametrize("complex_data", [False, True])
@pytest.mark.parametrize(
    ("profile_policy", "stream_policy"),
    [
        ("first_valid", DuplicatePolicy.FIRST_VALID),
        ("last_valid", DuplicatePolicy.LAST_VALID),
        ("last-valid", DuplicatePolicy.LAST_VALID),
        ("reject", DuplicatePolicy.REJECT),
        ("unknown", None),
        ("first", DuplicatePolicy.FIRST),
        ("last", DuplicatePolicy.LAST),
    ],
)
def test_bulk_file_source_matches_streaming_timeline(
    tmp_path: Path,
    complex_data: bool,
    profile_policy: str,
    stream_policy: DuplicatePolicy | None,
) -> None:
    payload_bytes = 8
    first_payload = bytes((0x10,)) * payload_bytes
    invalid_payload = bytes((0x20,)) * payload_bytes
    valid_payload = bytes((0x30,)) * payload_bytes
    last_payload = bytes((0x40,)) * payload_bytes
    first_file = tmp_path / "first.vdif"
    second_file = tmp_path / "second.vdif"
    first_file.write_bytes(
        _frame(
            payload=last_payload,
            frame_number=3,
            complex_data=complex_data,
        )
        + _frame(
            payload=invalid_payload,
            frame_number=2,
            invalid=True,
            complex_data=complex_data,
        )
    )
    second_file.write_bytes(
        _frame(
            payload=first_payload,
            frame_number=0,
            complex_data=complex_data,
        )
        + _frame(
            payload=valid_payload,
            frame_number=2,
            complex_data=complex_data,
        )
    )

    samples_per_frame = payload_bytes * 4 // (2 * (2 if complex_data else 1))
    sample_rate_hz = float(samples_per_frame * 4)
    datastream = Datastream(
        identifier=0,
        station_id=0,
        sampling=(
            SamplingKind.COMPLEX if complex_data else SamplingKind.REAL
        ),
        quantization_bits=2,
        frame_size_bytes=32 + payload_bytes,
        source_files=(str(first_file), str(second_file)),
        recorded_bands=(
            RecordedBand(0, 0, "R", thread_id=5),
            RecordedBand(1, 0, "L", thread_id=5),
        ),
        sample_rate_hz=sample_rate_hz,
    )
    profile = replace(ANALYTIC_V1, duplicate_policy=profile_policy)

    if stream_policy is None:
        with pytest.raises(VdifError, match="unknown VDIF duplicate policy"):
            VdifFileRecordingSource(_job(datastream, sample_rate_hz), profile)
        return
    if stream_policy is DuplicatePolicy.REJECT:
        with pytest.raises(DuplicateVdifFrameError, match="duplicate VDIF"):
            list(
                VdifFileRecordingSource(
                    _job(datastream, sample_rate_hz), profile
                ).batches(datastream)
            )
        return
    bulk_batches = list(
        VdifFileRecordingSource(
            _job(datastream, sample_rate_hz), profile
        ).batches(datastream)
    )
    scalar_frames = [
        frame
        for path in (first_file, second_file)
        for frame in iter_vdif_frames(path)
    ]
    timeline = build_thread_timelines(
        scalar_frames,
        frames_per_second=4,
        duplicate_policy=stream_policy,
    )[5]
    expected_payload = b"".join(
        bytes(payload_bytes) if frame is None else frame.payload
        for frame in timeline.frames
    )

    assert len(bulk_batches) == 2
    for channel_index, batch in enumerate(bulk_batches):
        assert isinstance(batch.samples, VdifPayloadPlan)
        np.testing.assert_array_equal(
            batch.samples.materialize(),
            np.frombuffer(expected_payload, dtype=np.uint8),
        )
        np.testing.assert_array_equal(batch.validity, timeline.frame_validity)
        assert batch.packed_layout is not None
        assert batch.packed_layout.channel_index == channel_index
        assert batch.packed_layout.channel_count == 2
        assert batch.packed_layout.complex_data is complex_data
    assert bulk_batches[0].start_time == bulk_batches[1].start_time


def test_bulk_file_source_rejects_sparse_timestamp_expansion(
    tmp_path: Path,
) -> None:
    payload = bytes((0x10,)) * 8
    source = tmp_path / "sparse.vdif"
    source.write_bytes(
        _frame(payload=payload, frame_number=0)
        + _frame(
            payload=payload,
            frame_number=0,
            seconds_from_epoch=100,
        )
    )
    sample_rate_hz = 64.0
    datastream = Datastream(
        identifier=0,
        station_id=0,
        sampling=SamplingKind.REAL,
        quantization_bits=2,
        frame_size_bytes=40,
        source_files=(str(source),),
        recorded_bands=(RecordedBand(0, 0, "R", thread_id=5),),
        sample_rate_hz=sample_rate_hz,
    )

    with pytest.raises(ConfigurationError, match="VDIF timestamp span"):
        list(
            VdifFileRecordingSource(
                _job(datastream, sample_rate_hz), ANALYTIC_V1
            ).batches(datastream)
        )


def test_bulk_file_source_rejects_station_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "wrong-station.vdif"
    source.write_bytes(
        _frame(payload=bytes((0x10,)) * 8, frame_number=0, station="BB")
    )
    sample_rate_hz = 64.0
    datastream = Datastream(
        identifier=0,
        station_id=0,
        sampling=SamplingKind.REAL,
        quantization_bits=2,
        frame_size_bytes=40,
        source_files=(str(source),),
        recorded_bands=(RecordedBand(0, 0, "R", thread_id=5),),
        sample_rate_hz=sample_rate_hz,
    )

    with pytest.raises(
        ConfigurationError, match="does not match job station"
    ):
        list(
            VdifFileRecordingSource(
                _job(datastream, sample_rate_hz), ANALYTIC_V1
            ).batches(datastream)
        )


def test_interop_source_accepts_distinct_telescope_and_vdif_codes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "vdif-code.vdif"
    source.write_bytes(
        _frame(payload=bytes((0x10,)) * 8, frame_number=0, station="BB")
    )
    sample_rate_hz = 64.0
    datastream = Datastream(
        identifier=0,
        station_id=0,
        sampling=SamplingKind.REAL,
        quantization_bits=2,
        frame_size_bytes=40,
        source_files=(str(source),),
        recorded_bands=(RecordedBand(0, 0, "R", thread_id=5),),
        sample_rate_hz=sample_rate_hz,
    )

    batches = list(
        VdifFileRecordingSource(
            _job(datastream, sample_rate_hz), INTEROP_REAL_USB_V1
        ).batches(datastream)
    )

    assert len(batches) == 1


def test_bulk_file_source_rejects_mixed_stations_across_threads(
    tmp_path: Path,
) -> None:
    source = tmp_path / "mixed-stations.vdif"
    payload = bytes((0x10,)) * 8
    source.write_bytes(
        _frame(payload=payload, frame_number=0, thread_id=5, station="AA")
        + _frame(
            payload=payload,
            frame_number=0,
            thread_id=6,
            station="BB",
        )
    )
    sample_rate_hz = 64.0
    datastream = Datastream(
        identifier=0,
        station_id=0,
        sampling=SamplingKind.REAL,
        quantization_bits=2,
        frame_size_bytes=40,
        source_files=(str(source),),
        recorded_bands=(
            RecordedBand(0, 0, "R", thread_id=5),
            RecordedBand(1, 0, "L", thread_id=6),
        ),
        sample_rate_hz=sample_rate_hz,
    )
    job = _job(datastream, sample_rate_hz)
    job = replace(job, stations=(replace(job.stations[0], name="STATION_A"),))

    with pytest.raises(VdifError, match="different station identifiers"):
        list(VdifFileRecordingSource(job, ANALYTIC_V1).batches(datastream))


def test_physical_sidecar_overrides_preserve_later_moves_and_drops():
    slots = np.arange(100, 105, dtype=np.int64)
    actual = recording_module._apply_physical_sidecar_overrides(
        slots,
        np.zeros(5, dtype=np.int32),
        np.arange(10, 15, dtype=np.int64),
        ((0, 10, 14, 200), (0, 11, 13, 300), (0, 12, 13, -1)),
    )
    np.testing.assert_array_equal(actual, [200, 300, -1, 203, 104])
    np.testing.assert_array_equal(slots, np.arange(100, 105))


def test_missing_sidecar_names_supported_remedies(tmp_path):
    from cuphoton.xbeam.adapters.vdif.vdif_index import VdifIndexError
    from cuphoton.xbeam.adapters.vdif.vdif_sidecar import (
        validate_vdif_binary_index,
    )

    source = tmp_path / "unindexed.vdif"
    source.write_bytes(_frame(payload=bytes(8), frame_number=0))
    with pytest.raises(VdifIndexError) as error:
        validate_vdif_binary_index(source)
    assert "build-vdif-sidecar" in str(error.value)
    assert (
        "--assume-ordered only for independently verified ordered input"
        in str(error.value)
    )


@pytest.mark.parametrize(
    "epoch,seconds",
    [(40, 31 * 86400), (39, 215 * 86400), (0, 7336 * 86400 + 5)],
)
@pytest.mark.parametrize("mode", ["unbounded", "ordered", "sidecar"])
def test_recorded_epoch_does_not_shift_utc_selection_or_start(
    tmp_path, epoch, seconds, mode
):
    path = tmp_path / "older-epoch.vdif"
    path.write_bytes(
        b"".join(
            _frame(
                payload=bytes([number + 1]) * 8,
                frame_number=number,
                channels=1,
                reference_epoch=epoch,
                seconds_from_epoch=seconds,
            )
            for number in range(4)
        )
    )
    stream = Datastream(
        0,
        0,
        SamplingKind.REAL,
        2,
        40,
        (str(path),),
        (RecordedBand(0, 0, "R", thread_id=5),),
        sample_rate_hz=128,
    )
    job = replace(_job(stream, 128), start_time=MJDTime(58880, Decimal(0)))
    kwargs = {}
    if mode != "unbounded":
        if mode == "sidecar":
            build_vdif_binary_index(path, backend="posix")
        plan = plan_vdif_ingest(
            job,
            intervals={(0, 0): SampleInterval(0, 128)},
            max_read_bytes=1024,
            assume_ordered=mode == "ordered",
            index_dir=None,
        )
        assert plan.selected_source_bytes == 160
        kwargs = dict(
            source_ranges=plan.source_ranges,
            thread_windows=plan.thread_windows,
            binary_sidecars=mode == "sidecar",
        )
    (batch,) = VdifFileRecordingSource(job, ANALYTIC_V1, **kwargs).batches(
        stream
    )
    assert np.count_nonzero(batch.validity) == 4
    values = batch.samples.materialize().reshape(-1, 8)
    np.testing.assert_array_equal(
        values[batch.validity], [[i] * 8 for i in range(1, 5)]
    )
    assert batch.start_time == (
        MJDTime(58880, Decimal(0))
        if mode == "unbounded"
        else MJDTime(58879, Decimal("86399.75"))
    )


@pytest.mark.parametrize("mode", ["ordered", "sidecar"])
@pytest.mark.parametrize("separate_files", [False, True])
def test_bounded_source_selects_each_recorded_epoch(
    tmp_path, mode, separate_files
):
    paths = []
    for epoch, seconds in [
        (0, 7336 * 86400 + 5),
        (39, 215 * 86400),
        (40, 31 * 86400),
    ]:
        path = tmp_path / f"epoch-{epoch}.vdif"
        path.write_bytes(
            b"".join(
                _frame(
                    payload=bytes([frame + 1]) * 8,
                    frame_number=frame,
                    channels=1,
                    reference_epoch=epoch,
                    seconds_from_epoch=seconds,
                )
                for frame in range(4)
            )
        )
        if mode == "sidecar":
            build_vdif_binary_index(path, backend="posix")
        paths.append(str(path))
    if not separate_files:
        combined = tmp_path / "combined.vdif"
        combined.write_bytes(
            b"".join(Path(path).read_bytes() for path in paths)
        )
        paths = [str(combined)]
        if mode == "sidecar":
            build_vdif_binary_index(combined, backend="posix")
    stream = Datastream(
        0,
        0,
        SamplingKind.REAL,
        2,
        40,
        tuple(paths),
        (RecordedBand(0, 0, "R", thread_id=5),),
        sample_rate_hz=128,
    )
    job = replace(_job(stream, 128), start_time=MJDTime(58880, Decimal(0)))
    plan = plan_vdif_ingest(
        job,
        intervals={(0, 0): SampleInterval(0, 128)},
        max_read_bytes=4096,
        assume_ordered=mode == "ordered",
        index_dir=None,
    )
    assert plan.selected_source_bytes == 480
    (batch,) = VdifFileRecordingSource(
        job,
        ANALYTIC_V1,
        source_ranges=plan.source_ranges,
        thread_windows=plan.thread_windows,
        binary_sidecars=mode == "sidecar",
    ).batches(stream)
    assert np.count_nonzero(batch.validity) == 4
    np.testing.assert_array_equal(
        batch.samples.materialize().reshape(-1, 8)[batch.validity],
        [[i] * 8 for i in range(1, 5)],
    )
