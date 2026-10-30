# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from decimal import Decimal

import numpy as np
import pytest

from cuphoton.xbeam.adapters.radio.models import BufferedReaderGeometry
from cuphoton.xbeam.adapters.vdif.recording import VdifThreadWindow
from cuphoton.xbeam.adapters.vdif.vdif import VdifHeader, absolute_frame_slot
from cuphoton.xbeam.adapters.vdif.vdif_sidecar_builder import (
    build_vdif_binary_index,
)
from cuphoton.xbeam.buffered_reader.adapter import raw_sidecar_context
from cuphoton.xbeam.buffered_reader.requests import (
    evaluate_raw_job_requests,
)
from tests.xbeam.test_buffered_window_job_requests import model_job
from tests.xbeam.test_recording import _frame

RATE, FFT, SPF, REQUEST = 4_096_000_000, 262144, 32768, 50


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "independent-startup.vdif"
    valid = []
    delay_samples = 2 * REQUEST * FFT + FFT // 2 + SPF // 3
    cache = {}

    def codes(epoch):
        if epoch not in cache:
            cache[epoch] = np.random.default_rng(20261008 + epoch).integers(
                0, 4, FFT, dtype=np.uint8
            )
        return cache[epoch]

    def payload(physical):
        clock = physical * SPF + delay_samples
        epoch, offset = divmod(clock, FFT)
        take = min(SPF, FFT - offset)
        values = codes(epoch)[offset : offset + take]
        if take < SPF:
            values = np.concatenate((values, codes(epoch + 1)[: SPF - take]))
        words = values.reshape(-1, 4)
        packed = (
            words[:, 0]
            | (words[:, 1] << 2)
            | (words[:, 2] << 4)
            | (words[:, 3] << 6)
        )
        return packed.tobytes()

    with path.open("wb") as stream:
        for physical in range(1400):
            data = _frame(
                payload=payload(physical), frame_number=physical, channels=1
            )
            stream.write(data)
            assert not VdifHeader.from_bytes(data).invalid
            valid.append(physical)
    summary = build_vdif_binary_index(
        path, index_dir=tmp_path / "index", backend="posix"
    )
    delay = (2 * REQUEST * FFT + FFT // 2 + SPF // 3) / RATE
    job, ds, band, frequency, scan, geometry = model_job(
        (delay,), rate=RATE, fft=FFT
    )
    band = replace(band, thread_id=5)
    ds = replace(ds, source_files=(str(path),), recorded_bands=(band,))
    job = replace(
        job,
        datastreams=(ds,),
        buffered_reader=BufferedReaderGeometry(24, 8, 20),
        subintegration_duration_s=Decimal("0.0032"),
        integration_duration_s=Decimal("0.4"),
    )
    header = VdifHeader.from_bytes(path.read_bytes()[:32])
    origin = absolute_frame_slot(header, 125000)
    window = VdifThreadWindow(header, 125000, origin, origin + 1400)
    context = raw_sidecar_context(
        job,
        ds,
        frequency,
        window,
        summary,
        alignment_bytes=8192,
        word_bytes=4,
        control_word_bits=30,
    )
    # This coverage comes from actual generated header validity, separately
    # from the index's clock runs (which can preserve invalid frames).
    assert valid == list(range(1400))
    return (
        context,
        (job, ds, band, frequency, scan, geometry),
        path,
        ((0, 1400),),
    )


def returned_inputs(source, *, first=0, stop=200, timeline=200):
    from cuphoton.xbeam.buffered_reader.requests import (
        ReadClockConvention,
    )

    context, args, _path, _valid = source
    return evaluate_raw_job_requests(
        *args,
        first_fft=first,
        stop_fft=stop,
        timeline_fft_count=timeline,
        job_start_slot=context.job_start_slot,
        timestamp_origin_frame=context.job_start_slot,
        read_clock_convention=ReadClockConvention.ENDPOINT_MEAN,
    )


def bind_returned(source, request):
    from cuphoton.xbeam.buffered_reader.geometry import (
        BufferedNormalFullExtent,
    )
    from cuphoton.xbeam.buffered_reader.requests import (
        bind_initial_retained_window_contract,
    )

    state = source[0].validity_state
    extent = BufferedNormalFullExtent(8224, 8192, 402, 3, 4)
    return bind_initial_retained_window_contract(
        request,
        state,
        extent,
        normal_contract_factory=lambda suffix: (
            suffix.published_normal_full_window_contract(
                state, extent, control_word_bits=30
            )
        ),
    )


def test_initial_return_matches_independent_labelled_startup(source):
    request = returned_inputs(source, stop=150)
    contract = bind_returned(source, request)
    facts = request.compile(
        source[0].validity_state,
        frame_bytes=8224,
        control_word_bits=30,
        window_clock_contract=contract,
    )
    # Generated payload labels establish these addresses independently of
    # the planner: label k begins at k * FFT minus the known source delay.
    delay_samples = 2 * REQUEST * FFT + FFT // 2 + SPF // 3
    first_read = -(-delay_samples // FFT)
    assert facts.complete
    assert facts.ineligible_intervals == ((0, first_read),)
    assert [
        (span.first_fft, span.stop_fft, span.physical_sample_start)
        for span in facts.addresses.spans
    ] == [(first_read, 150, first_read * FFT - delay_samples)]
    sample = facts.addresses.spans[0].physical_sample_start
    frame, local = divmod(sample, SPF)
    with source[2].open("rb") as stream:
        stream.seek(frame * 8224)
        payloads = [stream.read(8224)[32:] for _ in range(10)]
    packed = np.frombuffer(b"".join(payloads), dtype=np.uint8)
    actual = ((packed[:, None] >> (2 * np.arange(4))) & 3).reshape(-1)[
        local : local + FFT
    ]
    expected = np.random.default_rng(20261008 + first_read).integers(
        0, 4, FFT, dtype=np.uint8
    )
    np.testing.assert_array_equal(actual, expected)
    assert facts.source_timing_events == ()
    assert not np.any(contract.publication_eligible[:100])
    assert np.all(contract.physical_head_samples == 0)


def test_initial_return_preserves_normal_suffix_and_scope_clipping(source):
    whole_inputs = returned_inputs(source)
    whole = bind_returned(source, whole_inputs)
    clipped_inputs = returned_inputs(source, first=113, stop=180)
    clipped = bind_returned(source, clipped_inputs)
    expected = whole.for_scope(113, 180)
    for name in (
        "aligned_head_samples",
        "physical_head_samples",
        "publication_eligible",
        "decoder_clamps_negative_offsets",
    ):
        np.testing.assert_array_equal(
            getattr(clipped, name), getattr(expected, name)
        )
    assert np.all(whole.publication_eligible[150:])


def test_initial_return_unknown_clock_stays_an_error(source):
    from cuphoton.xbeam.buffered_reader.geometry import (
        BufferedNormalFullExtent,
    )
    from cuphoton.xbeam.buffered_reader.requests import (
        bind_initial_retained_window_contract,
    )

    state = source[0].validity_state
    timeline = state.source_state.timeline
    broken = replace(timeline, timestamp_starts=timeline.timestamp_starts + 1)
    state = replace(
        state, source_state=replace(state.source_state, timeline=broken)
    )
    with pytest.raises(ValueError, match="clock is not contiguous"):
        bind_initial_retained_window_contract(
            returned_inputs(source),
            state,
            BufferedNormalFullExtent(8224, 8192, 402, 3, 4),
            normal_contract_factory=lambda _: pytest.fail(
                "unknown clock cannot reach normal factory"
            ),
        )
