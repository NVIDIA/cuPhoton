# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from decimal import Decimal

import numpy as np
import pytest

from cuphoton.xbeam.adapters.radio.models import (
    BufferedReaderGeometry,
    Datastream,
    MJDTime,
    RecordedBand,
    SamplingKind,
)
from cuphoton.xbeam.adapters.vdif.recording import VdifThreadWindow
from cuphoton.xbeam.adapters.vdif.vdif import VdifHeader, absolute_frame_slot
from cuphoton.xbeam.adapters.vdif.vdif_index import VdifTimestampKey
from cuphoton.xbeam.adapters.vdif.vdif_sidecar_builder import (
    build_vdif_binary_index,
)
from cuphoton.xbeam.buffered_reader.adapter import (
    job_frame_slot,
    packed_timestamp_slots,
    raw_sidecar_context,
)
from cuphoton.xbeam.buffered_reader.geometry import RawResidualOwner
from tests.xbeam.test_recording import _frame, _job
from tests.xbeam.test_vdif_sidecar import _recorded_filler


def recording(tmp_path):
    frames = [
        _frame(
            payload=bytes([p]) * 8,
            frame_number=(p if p < 5 else p + 2) % 4,
            seconds_from_epoch=(p if p < 5 else p + 2) // 4,
            channels=1,
            invalid=p == 3,
        )
        for p in range(40)
    ]
    path = tmp_path / "source.vdif"
    path.write_bytes(b"".join(frames))
    ds = Datastream(
        7,
        0,
        SamplingKind.REAL,
        2,
        40,
        (str(path),),
        (RecordedBand(0, 0, "R", thread_id=5),),
        sample_rate_hz=128.0,
    )
    job = replace(
        _job(ds, 128.0),
        buffered_reader=BufferedReaderGeometry(24, 8, 20),
    )
    frequency = replace(job.frequencies[0], channel_count=16)
    header = VdifHeader.from_bytes(frames[0])
    first = absolute_frame_slot(header, 4)
    window = VdifThreadWindow(header, 4, first, first + 42)
    summary = build_vdif_binary_index(
        path, index_dir=tmp_path / "indexes", backend="posix"
    )
    return job, ds, frequency, window, summary


def context(args, **kwargs):
    return raw_sidecar_context(
        *args, alignment_bytes=8, word_bytes=2, control_word_bits=30, **kwargs
    )


def test_explicit_clock_owner_uses_last_trustworthy_frame_across_hole(
    tmp_path,
):
    args = list(recording(tmp_path))
    source = tmp_path / "source.vdif"
    source.write_bytes(
        b"".join(
            _recorded_filler(frame_size=40)
            if 17 <= p < 21
            else _frame(
                payload=bytes([p]) * 8,
                frame_number=(p if p < 21 else p + 2) % 4,
                seconds_from_epoch=(p if p < 21 else p + 2) // 4,
                channels=1,
                invalid=p == 3,
            )
            for p in range(40)
        )
    )
    args[-1] = build_vdif_binary_index(
        source, index_dir=tmp_path / "indexes", backend="posix", force=True
    )
    physical = context(args)
    clock = context(args, residual_owner=RawResidualOwner.CLOCK_FRAME)
    assert physical.reader_geometry.allocation_frames == 18
    assert (
        physical.validity_state.source_state.request_state.event_allocations
        == {1}
    )
    assert (
        clock.validity_state.source_state.request_state.event_allocations
        == {0}
    )
    np.testing.assert_array_equal(
        physical.validity_state.source_state.request_state.anchor_samples,
        clock.validity_state.source_state.request_state.anchor_samples,
    )
    assert physical.validity_state.physical_valid_spans == (
        (0, 17 * 32),
        (21 * 32, 40 * 32),
    )
    assert physical.validity_state.physical_valid_spans == (
        clock.validity_state.physical_valid_spans
    )


def test_real_sidecar_preserves_raw_ordinals_and_explicit_invalid_clock(
    tmp_path,
):
    args = recording(tmp_path)
    state = context(args)
    timeline = state.validity_state.source_state.timeline
    native = absolute_frame_slot(args[3].template, 4)
    assert state.job_start_slot == native
    assert state.request_geometry.ffts_per_request == 4
    assert state.reader_geometry.allocation_frames == 18
    assert state.request_geometry.samples_per_byte == 4
    # Packed timestamp keys split at second boundaries; those storage splits
    # must not become physical holes or extra timestamp discontinuities.
    assert int(timeline.physical_starts[0]) == 0
    assert int(timeline.physical_stops[-1]) == 40
    assert timeline.discontinuity_count == 1
    event = timeline.discontinuities[0]
    assert event.physical_start == 5 and event.timestamp_gap_frames == 2
    assert event.physical_gap_frames == 0
    assert state.validity_state.physical_valid_spans == ((0, 40 * 32),)
    assert (
        state.validity_state.source_state.request_state.timestamp_origin_frame
        == native
    )
    # Explicit invalid frame3 retains its clock; no authoritative payload
    # validity or exposure is manufactured by this context.
    assert state.validity_state.physical_valid_spans[0][1] > 4 * 32
    # The timestamp jump increases the logical span to 42 frames, but the
    # physical file still contains exactly 40 frames. EOF is a byte-extent
    # fact, not a timestamp-derived duration.
    assert state.validity_state.source_state.recorded_source_stop == 40 * 32
    assert args[3].stop_slot - args[3].start_slot == 42


def test_job_origin_uses_job_time_instead_of_first_recording_header(tmp_path):
    args = list(recording(tmp_path))
    args[0] = replace(args[0], start_time=MJDTime(58849, Decimal(3)))
    state = context(args)
    assert (
        state.job_start_slot == absolute_frame_slot(args[3].template, 4) + 12
    )
    assert (
        state.validity_state.source_state.request_state.anchor_samples[0]
        == -384
    )


def test_non_frame_aligned_job_origin_rejects(tmp_path):
    job, _, _, window, _ = recording(tmp_path)
    job = replace(job, start_time=MJDTime(58849, Decimal("0.1")))
    with pytest.raises(ValueError, match="frame aligned"):
        job_frame_slot(job, window)


def test_sparse_timestamp_epoch_changes_use_the_ordinary_vdif_grid(tmp_path):
    header = recording(tmp_path)[3].template
    keys = np.asarray(
        [
            VdifTimestampKey(39, 5, 1).packed,
            VdifTimestampKey(40, 7, 3).packed,
        ],
        dtype=np.uint64,
    )
    slots = packed_timestamp_slots(keys, frames_per_second=4, template=header)
    expected = [
        absolute_frame_slot(
            replace(
                header,
                reference_epoch=epoch,
                seconds_from_epoch=second,
                frame_number=frame,
            ),
            4,
        )
        for epoch, second, frame in [(39, 5, 1), (40, 7, 3)]
    ]
    np.testing.assert_array_equal(slots, expected)
    cycle = packed_timestamp_slots(
        keys, frames_per_second=4, template=header, epoch_cycle=1
    )
    assert np.all(cycle > slots)


@pytest.mark.parametrize(
    "keys, message",
    [
        (np.array([-1]), "60-bit"),
        (np.array([1 << 60], dtype=np.uint64), "60-bit"),
        (np.array([4], dtype=np.uint64), "frame number"),
        (np.array([], dtype=np.uint64), "nonempty"),
    ],
)
def test_bad_sparse_timestamps_fail_explicitly(tmp_path, keys, message):
    header = recording(tmp_path)[3].template
    with pytest.raises(ValueError, match=message):
        packed_timestamp_slots(keys, frames_per_second=4, template=header)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"rejected_header_frame_count": 1}, "rejected header/format"),
        ({"rejected_format_frame_count": 1}, "rejected header/format"),
        ({"schema_version": 3}, "schema-v4"),
        ({"thread_metadata": ()}, "one physical"),
        ({"frame_count": 1}, "recording frame count"),
    ],
)
def test_invalid_summary_contracts_fail_before_publishing_a_plan(
    tmp_path, changes, message
):
    args = list(recording(tmp_path))
    args[-1] = replace(args[-1], **changes)
    with pytest.raises(ValueError, match=message):
        context(args)


def test_old_sidecar_reports_a_configuration_error_with_rebuild_hint(
    tmp_path,
):
    from cuphoton.xbeam.errors import ConfigurationError

    args = list(recording(tmp_path))
    args[-1] = replace(args[-1], schema_version=3)
    with pytest.raises(ConfigurationError, match="schema-v4.*--force"):
        context(args)


def test_requires_explicit_reader_and_matching_physical_thread(tmp_path):
    args = list(recording(tmp_path))
    ordinary_job = replace(args[0], buffered_reader=None)
    with pytest.raises(ValueError, match="explicit buffered-reader"):
        context([ordinary_job, *args[1:]])
    wrong_window = replace(
        args[3], template=replace(args[3].template, thread_id=0)
    )
    with pytest.raises(ValueError, match="differs from ingest window"):
        context([*args[:3], wrong_window, args[4]])


def test_sidecar_identity_change_is_detected_before_mapping(tmp_path):
    args = recording(tmp_path)
    with args[-1].index_path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="changed before mapping"):
        context(args)
