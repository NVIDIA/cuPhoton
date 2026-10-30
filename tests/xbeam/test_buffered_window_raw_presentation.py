# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import numpy as np
import pytest

from cuphoton.xbeam.adapters.radio.engine import (
    _extend_physical_presentation_ranges,
)
from cuphoton.xbeam.adapters.radio.models import (
    Datastream,
    RecordedBand,
    SamplingKind,
)
from cuphoton.xbeam.adapters.vdif.ingest import (
    SampleInterval,
    plan_vdif_ingest,
)
from cuphoton.xbeam.adapters.vdif.recording import (
    VdifFileRecordingSource,
    VdifSourceRange,
    VdifThreadWindow,
)
from cuphoton.xbeam.adapters.vdif.vdif import (
    VdifHeader,
    absolute_frame_slot,
    decode_2bit,
)
from cuphoton.xbeam.adapters.vdif.vdif_sidecar_builder import (
    build_vdif_binary_index,
)
from cuphoton.xbeam.buffered_reader.addressing import (
    LatchedRawSourceCompilation,
    LatchedRawSourceSpan,
    UnresolvedRawSourceSpan,
)
from cuphoton.xbeam.buffered_reader.geometry import RawBufferedRequestState
from cuphoton.xbeam.buffered_reader.plans import (
    OrdinaryRawSourceSpan,
    compile_raw_replay_plan,
    compile_raw_replay_window,
    with_canonical_raw_linear_presentation,
)
from cuphoton.xbeam.buffered_reader.timeline import PhysicalAllocationTimeline
from cuphoton.xbeam.buffered_reader.validity import (
    RawValidityCompilation,
    RawValidityState,
    split_latched_raw_source_validity,
)
from cuphoton.xbeam.profiles import ANALYTIC_V1
from tests.xbeam.test_recording import _frame, _job


def generated_recording(tmp_path, *, source_index=0):
    path = tmp_path / "canonical.vdif"
    payloads = [
        bytes((p * 11 + b) % 256 for b in range(8)) for p in range(12)
    ]
    # All later native headers are outside the destination job window.
    frames = []
    for physical, payload in enumerate(payloads):
        native = physical if physical < 2 else physical + 20
        frames.append(
            _frame(
                payload=payload,
                frame_number=native % 4,
                seconds_from_epoch=native // 4,
                channels=1,
                invalid=physical == 1,
            )
        )
    path.write_bytes(b"".join(frames))
    source_files = (str(path),)
    if source_index:
        decoy = tmp_path / "decoy.vdif"
        decoy.write_bytes(frames[0])
        source_files = (str(decoy), str(path))
    stream = Datastream(
        identifier=0,
        station_id=0,
        sampling=SamplingKind.REAL,
        quantization_bits=2,
        frame_size_bytes=40,
        source_files=source_files,
        recorded_bands=(RecordedBand(0, 0, "R", thread_id=5),),
        sample_rate_hz=128.0,
    )
    template = VdifHeader.from_bytes(frames[0])
    slot = absolute_frame_slot(template, 4)
    indexes = tmp_path / "indexes"
    build_vdif_binary_index(path, index_dir=indexes, backend="posix")
    timeline = PhysicalAllocationTimeline.from_runs(
        np.array([0, 2]),
        np.array([1, 12]),
        np.array([0, 22]),
        allocation_frames=1206,
    )
    requests = RawBufferedRequestState.from_timeline(
        timeline, samples_per_frame=32
    )
    return (
        path,
        payloads,
        stream,
        template,
        slot,
        indexes,
        RawValidityState.from_timeline(timeline, requests),
    )


def read_presented(path, stream, template, slot, indexes, plan):
    _, first, stop, destination = plan.source_physical_frame_intervals[0]
    assert destination == slot + first
    stat = path.stat()
    windows = {
        0: {5: VdifThreadWindow(template, 4, destination, slot + stop)}
    }
    (batch,) = list(
        VdifFileRecordingSource(
            _job(stream, 128.0),
            ANALYTIC_V1,
            source_ranges={
                0: (
                    VdifSourceRange(
                        path,
                        first,
                        stop - first,
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
            binary_index_dir=indexes,
            physical_frame_presentation_plans={
                0: plan.source_physical_frame_intervals
            },
        ).batches(stream)
    )
    return batch, first, stop


@pytest.mark.parametrize("first_fft,stop_fft", [(0, 6), (2, 6)])
def test_real_posix_sidecars_read_canonical_payload_and_preserve_invalidity(
    tmp_path, first_fft, stop_fft
):
    path, payloads, stream, template, slot, indexes, state = (
        generated_recording(tmp_path)
    )
    addresses = LatchedRawSourceCompilation(
        (LatchedRawSourceSpan(first_fft, stop_fft, 24 + first_fft * 16),), ()
    )
    validity = split_latched_raw_source_validity(
        state, addresses, fft_samples=16
    )
    # Coarse-delay changes are ordinary source facts; canonical offsets cancel
    # them to recover the supplied physical identities.
    ordinary = tuple(
        OrdinaryRawSourceSpan(first, stop, first * 16 - coarse)
        for first, stop, coarse in ((first_fft, 3, 0), (3, stop_fft, 1))
    )
    scoped = compile_raw_replay_window(
        addresses,
        validity,
        first_fft=first_fft,
        stop_fft=stop_fft,
        fft_samples=16,
        ordinary_source_spans=ordinary,
        ineligible_intervals=(),
    )
    presented = with_canonical_raw_linear_presentation(
        scoped,
        addresses,
        fft_samples=16,
        samples_per_frame=32,
        job_start_slot=slot,
        source_index=0,
    )
    plan = presented.plan_for_scope(first_fft, stop_fft)
    batch, first_frame, stop_frame = read_presented(
        path, stream, template, slot, indexes, plan
    )
    np.testing.assert_array_equal(
        np.asarray(batch.samples),
        np.frombuffer(
            b"".join(payloads[first_frame:stop_frame]), dtype=np.uint8
        ),
    )
    np.testing.assert_array_equal(
        batch.validity, [p != 1 for p in range(first_frame, stop_frame)]
    )
    assert batch.start_time.mjd == _job(stream, 128.0).start_time.mjd
    assert batch.start_time.seconds_of_day == first_frame / 4
    decoded, _ = decode_2bit(np.asarray(batch.samples).tobytes())
    decoded = decoded[:, 0]
    sample_validity = np.repeat(batch.validity, 32)
    raw_truth, _ = decode_2bit(b"".join(payloads))
    raw_truth = raw_truth[:, 0]
    for fft in range(first_fft, stop_fft):
        coarse = 0 if fft < 3 else 1
        ordinary_start = fft * 16 - coarse
        offset, cutover = plan.source_replay(fft)
        assert cutover == 0
        start, stop = plan.source_validity(fft) or (0, 16)
        actual = np.zeros(16, np.float32)
        for local in range(start, stop):
            source = ordinary_start + offset + local
            position = source - first_frame * 32
            if sample_validity[position]:
                actual[local] = decoded[position]
        source_indices = np.arange(24 + fft * 16, 24 + (fft + 1) * 16)
        expected = np.where(
            source_indices // 32 != 1, raw_truth[source_indices], 0
        )
        np.testing.assert_array_equal(actual, expected)
    assert not plan.has_authoritative_source_validity
    assert plan.source_weight_override_events == ()
    assert plan.source_weight_credit_intervals == ()
    for name in (
        "ineligible_intervals",
        "source_replay_intervals",
        "source_validity_events",
    ):
        assert getattr(plan, name) == getattr(
            scoped.plan_for_scope(first_fft, stop_fft), name
        )


def test_exact_frame_rounding_large_addresses_and_nonmonotonic_reads():
    physical = 2**55 + 3
    addresses = LatchedRawSourceCompilation(
        (
            LatchedRawSourceSpan(0, 2, physical),
            LatchedRawSourceSpan(2, 3, physical - 20),
        ),
        (),
    )
    validity = RawValidityCompilation(addresses.spans, (), (), ())
    plan = compile_raw_replay_plan(
        addresses,
        validity,
        timeline_fft_count=3,
        fft_samples=8,
        ordinary_sample_origin=0,
        ineligible_intervals=(),
    )
    presented = with_canonical_raw_linear_presentation(
        plan,
        addresses,
        fft_samples=8,
        samples_per_frame=7,
        job_start_slot=2**55 + 1,
        source_index=4,
    )
    first = (physical - 20) // 7
    stop = (physical + 16 + 6) // 7
    assert presented.source_physical_frame_intervals == (
        (4, first, stop, 2**55 + 1 + first),
    )


def test_reject_unresolved_conflicting_or_incomplete_address_scope():
    addresses = LatchedRawSourceCompilation(
        (LatchedRawSourceSpan(0, 2, 0),), ()
    )
    validity = RawValidityCompilation(addresses.spans, (), (), ())
    plan = compile_raw_replay_plan(
        addresses,
        validity,
        timeline_fft_count=2,
        fft_samples=8,
        ordinary_sample_origin=0,
        ineligible_intervals=(),
    )
    arguments = dict(
        fft_samples=8, samples_per_frame=4, job_start_slot=100, source_index=0
    )
    with pytest.raises(ValueError, match="unresolved"):
        with_canonical_raw_linear_presentation(
            plan,
            replace(
                addresses,
                unresolved=(UnresolvedRawSourceSpan(0, 1, "unknown"),),
            ),
            **arguments,
        )
    with pytest.raises(ValueError, match="cover every accepted"):
        with_canonical_raw_linear_presentation(
            plan,
            replace(addresses, spans=(LatchedRawSourceSpan(0, 1, 0),)),
            **arguments,
        )
    with pytest.raises(ValueError, match="conflicts"):
        with_canonical_raw_linear_presentation(
            replace(plan, source_physical_frame_intervals=((0, 0, 4, 100),)),
            addresses,
            **arguments,
        )


def test_real_interpreter_nonzero_source_index_reads_selected_file(tmp_path):
    path, _, stream, template, slot, indexes, state = generated_recording(
        tmp_path, source_index=1
    )
    addresses = LatchedRawSourceCompilation(
        (LatchedRawSourceSpan(0, 2, 0),), ()
    )
    validity = split_latched_raw_source_validity(
        state, addresses, fft_samples=16
    )
    plan = compile_raw_replay_plan(
        addresses,
        validity,
        timeline_fft_count=2,
        fft_samples=16,
        ordinary_sample_origin=0,
        ineligible_intervals=(),
    )
    presented = with_canonical_raw_linear_presentation(
        plan,
        addresses,
        fft_samples=16,
        samples_per_frame=32,
        job_start_slot=slot,
        source_index=1,
    )
    batch, first, stop = read_presented(
        path, stream, template, slot, indexes, presented
    )
    assert first == 0 and stop == 1
    assert batch.samples.files[0].path == path
    assert len(batch.samples.files) == 1
    np.testing.assert_array_equal(
        np.asarray(batch.samples),
        np.frombuffer(path.read_bytes()[32:40], dtype=np.uint8),
    )


def test_actual_bounded_planner_expands_physical_range_without_timestamp_fit(
    tmp_path,
):
    path, payloads, stream, _, slot, indexes, state = generated_recording(
        tmp_path
    )
    addresses = LatchedRawSourceCompilation(
        (LatchedRawSourceSpan(0, 6, 24),), ()
    )
    validity = split_latched_raw_source_validity(
        state, addresses, fft_samples=16
    )
    replay = compile_raw_replay_plan(
        addresses,
        validity,
        timeline_fft_count=6,
        fft_samples=16,
        ordinary_sample_origin=0,
        ineligible_intervals=(),
    )
    canonical = with_canonical_raw_linear_presentation(
        replay,
        addresses,
        fft_samples=16,
        samples_per_frame=32,
        job_start_slot=slot,
        source_index=0,
    )
    job = _job(stream, 128.0)
    initial = plan_vdif_ingest(
        job,
        intervals={(0, 0): SampleInterval(0, 128)},
        max_read_bytes=1 << 20,
        assume_ordered=False,
        index_dir=indexes,
    )
    updated = _extend_physical_presentation_ranges(
        job,
        initial,
        {("scope", 0, 0): canonical},
        max_read_bytes=1 << 20,
    )
    assert tuple(
        (item.first_frame, item.stop_frame)
        for item in updated.source_ranges[0]
    ) == ((0, 4),)
    (batch,) = list(
        VdifFileRecordingSource(
            job,
            ANALYTIC_V1,
            source_ranges=updated.source_ranges,
            thread_windows=updated.thread_windows,
            source_thread_windows=updated.source_thread_windows,
            binary_sidecars=True,
            binary_index_dir=indexes,
            physical_frame_presentation_plans={
                0: canonical.source_physical_frame_intervals
            },
        ).batches(stream)
    )
    window = updated.thread_windows[0][5]
    relative_frames = range(window.start_slot - slot, window.stop_slot - slot)
    expected = b"".join(
        payloads[physical] if 0 <= physical < 4 else bytes(8)
        for physical in relative_frames
    )
    np.testing.assert_array_equal(
        np.asarray(batch.samples), np.frombuffer(expected, dtype=np.uint8)
    )
    np.testing.assert_array_equal(
        batch.validity,
        [0 <= physical < 4 and physical != 1 for physical in relative_frames],
    )
    start_seconds = (
        (batch.start_time.mjd - job.start_time.mjd) * 86400
        + batch.start_time.seconds_of_day
        - job.start_time.seconds_of_day
    )
    assert start_seconds * 128 == (window.start_slot - slot) * 32
    assert batch.samples.files[0].path == path
