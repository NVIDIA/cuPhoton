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
from cuphoton.xbeam.adapters.vdif.recording import VdifFileRecordingSource
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
)
from cuphoton.xbeam.buffered_reader.geometry import RawBufferedRequestState
from cuphoton.xbeam.buffered_reader.plans import (
    OrdinaryRawSourceSpan,
    RawLinearStorage,
    compile_raw_replay_window,
    compile_translated_raw_replay_window,
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


def actual_source(tmp_path):
    payloads = [
        bytes((p * 11 + b) % 256 for b in range(8)) for p in range(80)
    ]
    path = tmp_path / "actual-file-one.vdif"
    frames = []
    for physical, payload in enumerate(payloads):
        native = physical if physical < 20 else physical + 20
        frames.append(
            _frame(
                payload=payload,
                frame_number=native % 4,
                seconds_from_epoch=native // 4,
                channels=1,
                invalid=physical == 21,
            )
        )
    path.write_bytes(b"".join(frames))
    decoy = tmp_path / "decoy-file-zero.vdif"
    decoy.write_bytes(
        _frame(payload=bytes([0xEE]) * 8, frame_number=0, channels=1)
    )
    stream = Datastream(
        identifier=0,
        station_id=0,
        sampling=SamplingKind.REAL,
        quantization_bits=2,
        frame_size_bytes=40,
        source_files=(str(decoy), str(path)),
        recorded_bands=(RecordedBand(0, 0, "R", thread_id=5),),
        sample_rate_hz=128.0,
    )
    indexes = tmp_path / "indexes"
    for source in (decoy, path):
        build_vdif_binary_index(source, index_dir=indexes, backend="posix")
    template = VdifHeader.from_bytes(frames[0])
    slot = absolute_frame_slot(template, 4)
    timeline = PhysicalAllocationTimeline.from_runs(
        np.array([0]),
        np.array([80]),
        np.array([slot]),
        allocation_frames=1206,
    )
    requests = RawBufferedRequestState.from_timeline(
        timeline, samples_per_frame=32, timestamp_origin_frame=slot
    )
    return (
        path,
        payloads,
        stream,
        indexes,
        slot,
        RawValidityState.from_timeline(timeline, requests),
    )


def planned_batch(stream, indexes, plan, *, first_fft, stop_fft):
    job = _job(stream, 128.0)
    ordinary = SampleInterval(first_fft * 16, stop_fft * 16 + 32)
    minimum, maximum = plan.source_offset_bounds_for(first_fft, stop_fft)
    selected = SampleInterval(
        ordinary.first_sample + minimum, ordinary.stop_sample + maximum
    )
    # A shared preparation may preserve both ordinary and replay source spans.
    # Use the real bounded planner on their union, not a mocked extent.
    requested = ordinary.union(selected)
    initial = plan_vdif_ingest(
        job,
        intervals={(0, 0): requested},
        max_read_bytes=1 << 20,
        assume_ordered=False,
        index_dir=indexes,
    )
    prepared = _extend_physical_presentation_ranges(
        job, initial, {("scope", 0, 0): plan}, max_read_bytes=1 << 20
    )
    (batch,) = list(
        VdifFileRecordingSource(
            job,
            ANALYTIC_V1,
            source_ranges=prepared.source_ranges,
            thread_windows=prepared.thread_windows,
            source_thread_windows=prepared.source_thread_windows,
            binary_sidecars=True,
            binary_index_dir=indexes,
            physical_frame_presentation_plans={
                0: plan.source_physical_frame_intervals
            },
        ).batches(stream)
    )
    start_seconds = (batch.start_time.mjd - job.start_time.mjd) * 86400 + (
        batch.start_time.seconds_of_day - job.start_time.seconds_of_day
    )
    return batch, prepared, int(start_seconds * 128)


@pytest.mark.parametrize("first_fft", [0, 2])
def test_actual_source_one_partial_invalid_and_compact_materialization(
    tmp_path, first_fft
):
    path, payloads, stream, indexes, slot, state = actual_source(tmp_path)
    addresses = LatchedRawSourceCompilation(
        (LatchedRawSourceSpan(first_fft, 6, 20 * 32 + 8 + first_fft * 16),),
        (),
    )
    # Clock coverage retains explicit-invalid frames. Its independent sidecar
    # validity must still suppress physical frame21 at execution.
    validity = split_latched_raw_source_validity(
        state, addresses, fft_samples=16
    )
    ordinary = (
        OrdinaryRawSourceSpan(first_fft, 3, first_fft * 16),
        OrdinaryRawSourceSpan(3, 6, 3 * 16 - 1),
    )
    storage = RawLinearStorage.align_to_ordinary_window(
        datastream_id=0,
        source_index=1,
        job_start_slot=slot,
        samples_per_frame=32,
        raw_first_frame=20,
        raw_stop_frame=24,
        ordinary_min_sample=0,
    )
    translated = compile_translated_raw_replay_window(
        storage,
        addresses,
        validity,
        first_fft=first_fft,
        stop_fft=6,
        fft_samples=16,
        ordinary_gpu_source_spans=ordinary,
        ineligible_intervals=(),
    )
    legacy = compile_raw_replay_window(
        addresses,
        validity,
        first_fft=first_fft,
        stop_fft=6,
        fft_samples=16,
        ordinary_source_spans=ordinary,
        ineligible_intervals=(),
    )
    canonical = with_canonical_raw_linear_presentation(
        legacy,
        addresses,
        fft_samples=16,
        samples_per_frame=32,
        job_start_slot=slot,
        source_index=1,
    )
    translated_plan = translated.plan_for_scope(first_fft, 6)
    canonical_plan = canonical.plan_for_scope(first_fft, 6)
    compact, prepared, start = planned_batch(
        stream, indexes, translated_plan, first_fft=first_fft, stop_fft=6
    )
    wide, wide_prepared, _ = planned_batch(
        stream, indexes, canonical_plan, first_fft=first_fft, stop_fft=6
    )
    assert compact.samples.slot_count + 10 < wide.samples.slot_count
    assert (
        prepared.estimated_materialized_bytes
        < wide_prepared.estimated_materialized_bytes
    )
    # Common metadata can retain one extra guard frame in a sliced window;
    # raw I/O remains bounded by the explicitly declared shared extent.
    assert (
        prepared.selected_source_bytes
        <= (storage.raw_stop_frame - storage.raw_first_frame)
        * stream.frame_size_bytes
    )
    assert all(file.path == path for file in compact.samples.files)
    decoded, _ = decode_2bit(np.asarray(compact.samples).tobytes())
    decoded = decoded[:, 0]
    sample_validity = np.repeat(compact.validity, 32)
    truth, _ = decode_2bit(b"".join(payloads))
    truth = truth[:, 0]
    for fft in range(first_fft, 6):
        coarse = 0 if fft < 3 else 1
        offset, cutover = translated_plan.source_replay(fft)
        assert cutover == 0
        actual = np.zeros(16, np.float32)
        for local in range(16):
            position = fft * 16 - coarse + offset + local - start
            if sample_validity[position]:
                actual[local] = decoded[position]
        physical = np.arange(
            20 * 32 + 8 + fft * 16, 20 * 32 + 8 + (fft + 1) * 16
        )
        expected = np.where(physical // 32 == 21, 0, truth[physical])
        np.testing.assert_array_equal(actual, expected)
    assert translated_plan.source_replay_intervals == (
        (first_fft, 3, 8),
        (3, 6, 9),
    )
    assert not translated_plan.has_authoritative_source_validity
    assert translated_plan.source_weight_credit_intervals == ()
    assert storage.physical_presentation == ((1, 20, 24, slot),)
    assert (
        addresses.spans[0].physical_sample_start
        == 20 * 32 + 8 + first_fft * 16
    )
    assert ordinary[0].ordinary_sample_start == first_fft * 16


@pytest.mark.parametrize("ordinary_min", [-(2**55 + 3), 2**55 + 3])
def test_exact_signed_origins_shared_windows_and_frame_rounding(ordinary_min):
    raw_min = 2**55 + 7
    storage = RawLinearStorage.align_to_ordinary_window(
        datastream_id=5,
        source_index=1,
        job_start_slot=2**55 + 3,
        samples_per_frame=7,
        raw_first_frame=raw_min,
        raw_stop_frame=raw_min + 10,
        ordinary_min_sample=ordinary_min,
    )
    assert (raw_min + storage.translation_frames) * 7 == ordinary_min // 7 * 7
    plans = []
    for first, stop in ((0, 2), (2, 4)):
        addresses = LatchedRawSourceCompilation(
            (LatchedRawSourceSpan(first, stop, raw_min * 7 + first * 8),), ()
        )
        validity = RawValidityCompilation(addresses.spans, (), (), ())
        window = compile_translated_raw_replay_window(
            storage,
            addresses,
            validity,
            first_fft=first,
            stop_fft=stop,
            fft_samples=8,
            ordinary_gpu_source_spans=(
                OrdinaryRawSourceSpan(first, stop, ordinary_min + first * 8),
            ),
            ineligible_intervals=(),
        )
        plan = window.plan_for_scope(first, stop)
        plans.append(plan)
        offset, cutover = plan.source_replay(first)
        assert cutover == 0
        assert ordinary_min + first * 8 + offset == (
            raw_min * 7 + first * 8 + storage.translation_samples
        )
    assert (
        plans[0].source_physical_frame_intervals
        == plans[1].source_physical_frame_intervals
    )
    assert (
        plans[0].source_physical_frame_intervals
        == storage.physical_presentation
    )


def test_reject_source_outside_explicit_common_storage_extent():
    storage = RawLinearStorage(0, 0, 100, 8, 20, 21, -20)
    addresses = LatchedRawSourceCompilation(
        (LatchedRawSourceSpan(0, 2, 160),), ()
    )
    validity = RawValidityCompilation(addresses.spans, (), (), ())
    with pytest.raises(ValueError, match="outside common storage"):
        compile_translated_raw_replay_window(
            storage,
            addresses,
            validity,
            first_fft=0,
            stop_fft=2,
            fft_samples=8,
            ordinary_gpu_source_spans=(OrdinaryRawSourceSpan(0, 2, 0),),
            ineligible_intervals=(),
        )
    with pytest.raises(TypeError, match="signed integer"):
        replace(storage, translation_frames=1.0)


def test_reject_noninteger_gpu_basis_before_translation():
    storage = RawLinearStorage(0, 0, 100, 8, 20, 22, -20)
    addresses = LatchedRawSourceCompilation(
        (LatchedRawSourceSpan(0, 2, 160),), ()
    )
    validity = RawValidityCompilation(addresses.spans, (), (), ())
    with pytest.raises(TypeError, match="signed integers"):
        compile_translated_raw_replay_window(
            storage,
            addresses,
            validity,
            first_fft=0,
            stop_fft=2,
            fft_samples=8,
            ordinary_gpu_source_spans=(OrdinaryRawSourceSpan(0, 2, True),),
            ineligible_intervals=(),
        )
