# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from decimal import Decimal

import numpy as np
import pytest

import cuphoton.xbeam.buffered_reader.requests as requests
from cuphoton.xbeam.adapters.radio.models import (
    ClockPolynomial,
    Datastream,
    DelayCoordinateModel,
    Frequency,
    Job,
    MJDTime,
    RecordedBand,
    SamplingKind,
    Scan,
    Sideband,
    Station,
)
from cuphoton.xbeam.buffered_reader.geometry import (
    BufferedRequestGeometry,
    RawBufferedRequestState,
)
from cuphoton.xbeam.buffered_reader.timeline import PhysicalAllocationTimeline
from cuphoton.xbeam.buffered_reader.validity import RawValidityState


def model_job(coefficients=(0.0,), *, rate=256.0, fft=256):
    epoch = MJDTime(58849, Decimal(0))
    band = RecordedBand(0, 0, "R")
    ds = Datastream(
        0,
        0,
        SamplingKind.REAL,
        2,
        8224,
        ("unused.vdif",),
        (band,),
        sample_rate_hz=rate,
    )
    freq = Frequency(0, 0.0, rate / 2, Sideband.USB, fft // 2)
    polynomial = DelayCoordinateModel(
        0, Decimal(0), coefficients, (0.0,), (0.0,), (0.0,)
    )
    scan = Scan(
        "ordinary", Decimal(0), Decimal(10000), "0", "0", (polynomial,)
    )
    job = Job(
        epoch,
        Decimal(10000),
        Decimal(1),
        Decimal(1),
        1,
        (Station(0, "test", ClockPolynomial(epoch, (0.0,))),),
        (freq,),
        (ds,),
        (),
        (scan,),
    )
    return (
        job,
        ds,
        band,
        freq,
        scan,
        BufferedRequestGeometry(4, fft, 50, 8192, 4),
    )


def evaluate(args, first=0, stop=50, timeline=1000, **kwargs):
    return requests.evaluate_raw_job_requests(
        *args,
        first_fft=first,
        stop_fft=stop,
        timeline_fft_count=timeline,
        job_start_slot=17,
        timestamp_origin_frame=17,
        **kwargs,
    )


def test_endpoint_and_centre_delay_have_distinct_rounding_contracts():
    result = evaluate(model_job((0.25 / 256, 2 / 256)))
    assert result.desired_sample_starts.tolist() == [0]
    # First FFT centre is 255/512 seconds. Endpoint rounds zero; centre
    # polynomial is 1.24609375 samples and nearest-even rounds one.
    assert result.ordinary_source_starts[0, 0] == -1
    assert result.endpoint_drifts.tolist() == [100 / 256]
    assert result.ordinary_source_starts[0, 1] == 256 - 3


@pytest.mark.parametrize(
    "delay, desired, ordinary", [(-1.5, 1, 2), (2.5, -3, -2)]
)
def test_half_sample_endpoint_rounding_is_not_coarse_nearest_even(
    delay, desired, ordinary
):
    result = evaluate(model_job((delay / 256,)))
    assert result.desired_sample_starts[0] == desired
    assert result.ordinary_source_starts[0, 0] == ordinary


def test_nonrequest_aligned_window_keeps_full_context_and_final_counts():
    args = model_job((0.0, 1 / 256))
    whole = evaluate(args, stop=109, timeline=109)
    window = evaluate(args, first=53, stop=105, timeline=109)
    assert window.request_fft_starts.tolist() == [50, 100]
    assert window.request_fft_counts.tolist() == [50, 9]
    np.testing.assert_array_equal(
        window.ordinary_source_starts, whole.ordinary_source_starts[1:]
    )
    np.testing.assert_array_equal(
        window.desired_sample_starts, whole.desired_sample_starts[1:]
    )
    assert np.all(np.diff(window.ordinary_source_starts[0]) == 255)
    assert window.endpoint_drifts.tolist() == [50 / 256, 50 / 256]
    # Arrays are immutable even if a caller tries to restore WRITEABLE.
    for array in (
        window.request_fft_starts,
        window.request_fft_counts,
        window.desired_sample_starts,
        window.endpoint_drifts,
        window.ordinary_source_starts,
    ):
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_request_limit_and_origin_fail_before_array_allocation(monkeypatch):
    args = model_job()

    def forbidden(*args, **kwargs):
        raise AssertionError("array allocation before bound validation")

    monkeypatch.setattr(requests.np, "arange", forbidden)
    with pytest.raises(ValueError, match="maximum requests"):
        evaluate(args, stop=101, maximum_requests=2)
    with pytest.raises(ValueError, match="timestamp origin"):
        requests.evaluate_raw_job_requests(
            *args,
            first_fft=0,
            stop_fft=50,
            timeline_fft_count=100,
            job_start_slot=17,
            timestamp_origin_frame=18,
        )
    with pytest.raises(ValueError, match="within the FFT timeline"):
        evaluate(args, stop=1001)


def test_geometry_fft_mismatch_rejects_model_inputs():
    args = list(model_job())
    args[-1] = BufferedRequestGeometry(4, 512, 50, 8192, 4)
    with pytest.raises(ValueError, match="FFT length differs"):
        evaluate(args)


def test_default_has_no_optional_read_clock_evaluation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("optional model evaluation on default path")

    monkeypatch.setattr(requests, "_endpoint_model_delays", forbidden)
    result = evaluate(model_job())
    assert result.read_clock_convention is None
    assert result.centre_delays_seconds is None
    assert result.output_sample_starts is None
    assert result.read_average_delays_seconds is None


def test_endpoint_mean_preserves_ordinary_and_leading_clocks():
    args = model_job((0.25 / 256, 2 / 256))
    default = evaluate(args)
    result = evaluate(
        args, read_clock_convention=requests.ReadClockConvention.ENDPOINT_MEAN
    )
    t = np.arange(50, dtype=np.float64)
    np.testing.assert_array_equal(
        result.read_average_delays_seconds[0], (0.25 + 2 * (t + 0.5)) / 256
    )
    np.testing.assert_array_equal(
        result.centre_delays_seconds[0],
        (0.25 + 2 * (t + 255 / 512)) / 256,
    )
    np.testing.assert_array_equal(result.output_sample_starts[0], t * 256)
    for name in (
        "ordinary_source_starts",
        "desired_sample_starts",
        "endpoint_drifts",
    ):
        np.testing.assert_array_equal(
            getattr(result, name), getattr(default, name)
        )
    np.testing.assert_array_equal(
        result.leading_head_inputs.first_centre_delays_seconds,
        default.leading_head_inputs.first_centre_delays_seconds,
    )
    for name in (
        "centre_delays_seconds",
        "output_sample_starts",
        "read_average_delays_seconds",
    ):
        with pytest.raises(ValueError):
            getattr(result, name).setflags(write=True)


def test_endpoint_mean_nonlinear_is_not_integral_or_midpoint():
    result = evaluate(
        model_job((0.0, 0.0, 1 / 256)),
        read_clock_convention=requests.ReadClockConvention.ENDPOINT_MEAN,
    )
    t = np.arange(50, dtype=np.float64)
    expected = (t * t + (t + 1) ** 2) / (2 * 256)
    np.testing.assert_array_equal(
        result.read_average_delays_seconds[0], expected
    )
    assert result.read_average_delays_seconds[0, 0] == 0.5 / 256
    assert result.read_average_delays_seconds[0, 0] != (1 / 3) / 256
    assert result.read_average_delays_seconds[0, 0] != 0.25 / 256


@pytest.mark.parametrize("boundary", [Decimal("0.75"), Decimal("1")])
def test_each_endpoint_selects_its_active_model(boundary):
    args = list(model_job((2 / 256,)))
    first = args[4].station_models[0]
    second = replace(
        first, epoch_offset_s=boundary, delay_coefficients_s=(6 / 256,)
    )
    args[4] = replace(args[4], station_models=(first, second))
    result = evaluate(
        args, read_clock_convention=requests.ReadClockConvention.ENDPOINT_MEAN
    )
    # First centre still uses the first model; its right endpoint uses the
    # second model, including an endpoint exactly on a model epoch.
    assert result.centre_delays_seconds[0, 0] == 2 / 256
    assert result.read_average_delays_seconds[0, 0] == 4 / 256
    assert result.read_average_delays_seconds[0, 1] == 6 / 256
    assert result.ordinary_source_starts[0, 0] == -2


def test_read_endpoint_clock_includes_station_and_band_offsets():
    args = list(model_job((2 / 256,)))
    station = replace(
        args[0].stations[0],
        clock=ClockPolynomial(args[0].start_time, (3 / 256,)),
    )
    args[0] = replace(args[0], stations=(station,))
    args[2] = replace(args[2], clock_offset_s=5 / 256)
    result = evaluate(
        args, read_clock_convention=requests.ReadClockConvention.ENDPOINT_MEAN
    )
    np.testing.assert_array_equal(
        result.read_average_delays_seconds, 10 / 256
    )


def test_optional_clock_rejects_unknown_convention_and_nonfinite_endpoints():
    with pytest.raises(ValueError, match="unsupported read clock"):
        evaluate(model_job(), read_clock_convention="endpoint-mean")
    result = evaluate(
        model_job(),
        read_clock_convention=requests.ReadClockConvention.ENDPOINT_MEAN,
    )
    with pytest.raises(ValueError, match="finite typed"):
        replace(result, read_average_delays_seconds=np.full((1, 50), np.inf))
    with pytest.raises(ValueError, match="match ordinary shape"):
        replace(result, centre_delays_seconds=np.zeros((1, 49)))


@pytest.mark.parametrize(
    "replacement",
    [
        None,
        [[0.0] * 50],
        np.zeros((1, 49)),
        np.zeros((1, 50), dtype=np.int64),
        np.full((1, 50), np.nan),
    ],
)
def test_malformed_optional_clock_matrix_rejected(replacement):
    result = evaluate(
        model_job(),
        read_clock_convention=requests.ReadClockConvention.ENDPOINT_MEAN,
    )
    with pytest.raises(ValueError, match="read clock matrices"):
        replace(result, read_average_delays_seconds=replacement)
    with pytest.raises(ValueError, match="explicit convention"):
        replace(result, read_clock_convention=None)
    with pytest.raises(ValueError, match="finite typed"):
        replace(result, output_sample_starts=np.zeros((1, 50)))


def test_unknown_convention_rejected_before_allocation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("array allocation before convention validation")

    monkeypatch.setattr(requests.np, "arange", forbidden)
    with pytest.raises(ValueError, match="unsupported read clock"):
        evaluate(model_job(), read_clock_convention="integral")


def test_nonfinite_endpoint_fails_before_retaining_read_matrices(monkeypatch):
    actual = requests._delay_values

    def endpoint_only_nonfinite(job, model, station, times, band, xp):
        values = actual(job, model, station, times, band, xp)
        # The final FFT right endpoint is 50 seconds, beyond all centres and
        # left endpoints. Ordinary evaluation remains finite.
        return np.where(times == 50.0, np.inf, values)

    monkeypatch.setattr(requests, "_delay_values", endpoint_only_nonfinite)
    with pytest.raises(ValueError, match="read endpoint model delays"):
        evaluate(
            model_job(),
            read_clock_convention=requests.ReadClockConvention.ENDPOINT_MEAN,
        )


def test_optional_clocks_preserve_complete_request_context():
    args = model_job((0.0, 1 / 256))
    convention = requests.ReadClockConvention.ENDPOINT_MEAN
    whole = evaluate(
        args, stop=109, timeline=109, read_clock_convention=convention
    )
    part = evaluate(
        args,
        first=53,
        stop=105,
        timeline=109,
        read_clock_convention=convention,
    )
    assert part.request_fft_counts.tolist() == [50, 9]
    for name in (
        "centre_delays_seconds",
        "output_sample_starts",
        "read_average_delays_seconds",
    ):
        np.testing.assert_array_equal(
            getattr(part, name), getattr(whole, name)[1:]
        )


def window_state():
    runs = PhysicalAllocationTimeline.from_runs(
        np.array([0, 6000], dtype=np.int64),
        np.array([6000, 120000], dtype=np.int64),
        np.array([17, 6022], dtype=np.int64),
        allocation_frames=1206,
    )
    return RawValidityState.from_timeline(
        runs,
        RawBufferedRequestState.from_timeline(
            runs, samples_per_frame=32768, timestamp_origin_frame=17
        ),
    )


def window_inputs(first=1250, stop=1399, timeline=1399, clocks=True):
    return evaluate(
        model_job((4e-6, -0.007), rate=4096000000, fft=262144),
        first=first,
        stop=stop,
        timeline=timeline,
        read_clock_convention=requests.ReadClockConvention.ENDPOINT_MEAN
        if clocks
        else None,
    )


def window_extent():
    from cuphoton.xbeam.buffered_reader.geometry import (
        BufferedNormalFullExtent,
    )

    return BufferedNormalFullExtent(8224, 8192, 402, 3, 4)


def paired_return_inputs(
    hole=None, *, delay_us=9023.69685655, first=7300, stop=8000
):
    """Independent physical/timestamp inputs, including pre-job padding."""
    capacity, padding, frames, jump = 1206, 2002, 129004, 51 * 1206 + 903
    holes = [] if hole is None else [hole]
    edges = sorted({0, frames, jump, *(x for span in holes for x in span)})
    runs = [
        (a, b)
        for a, b in zip(edges, edges[1:], strict=False)
        if not any(lo <= a < hi for lo, hi in holes)
    ]
    starts, stops = map(
        lambda x: np.asarray(x, dtype=np.int64), zip(*runs, strict=True)
    )
    timestamps = starts + np.where(starts >= jump, capacity + 9, 0)
    timeline = PhysicalAllocationTimeline.from_runs(
        starts, stops, timestamps, allocation_frames=capacity
    )
    cache = RawValidityState.from_timeline(
        timeline,
        RawBufferedRequestState.from_timeline(
            timeline, samples_per_frame=32768, timestamp_origin_frame=padding
        ),
    )
    inputs = requests.evaluate_raw_job_requests(
        *model_job((delay_us / 1e6,), rate=4096000000, fft=262144),
        first_fft=first,
        stop_fft=stop,
        timeline_fft_count=15625,
        job_start_slot=padding,
        timestamp_origin_frame=padding,
        read_clock_convention=requests.ReadClockConvention.ENDPOINT_MEAN,
    )
    return inputs, cache


def test_published_return_is_invariant_to_scope_clipping():
    inputs, cache = paired_return_inputs((61502, 61511))
    whole = inputs.published_normal_full_window_contract(
        cache, window_extent(), control_word_bits=30
    )
    part_inputs, part_cache = paired_return_inputs(
        (61502, 61511), first=7703, stop=7769
    )
    part = part_inputs.published_normal_full_window_contract(
        part_cache, window_extent(), control_word_bits=30
    )
    expected = whole.for_scope(7703, 7769)
    for field in (
        "publication_eligible",
        "physical_head_samples",
        "aligned_head_samples",
    ):
        np.testing.assert_array_equal(
            getattr(part, field), getattr(expected, field)
        )


def clean_clock_return_inputs(
    phase_samples, drift_samples_per_request, first=7300, stop=7700
):
    rate, fft, padding = 4096000000, 262144, 2002
    centre = 7500 * fft / rate + (fft - 1) / (2 * rate)
    slope = drift_samples_per_request / (50 * fft)
    delay = 8e-6 - phase_samples / rate - slope * centre
    timeline = PhysicalAllocationTimeline.from_runs(
        np.asarray([0], dtype=np.int64),
        np.asarray([129004], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        allocation_frames=1206,
    )
    state = RawValidityState.from_timeline(
        timeline,
        RawBufferedRequestState.from_timeline(
            timeline,
            samples_per_frame=32768,
            timestamp_origin_frame=padding,
        ),
    )
    inputs = requests.evaluate_raw_job_requests(
        *model_job((delay, slope), rate=rate, fft=fft),
        first_fft=first,
        stop_fft=stop,
        timeline_fft_count=15625,
        job_start_slot=padding,
        timestamp_origin_frame=padding,
        read_clock_convention=requests.ReadClockConvention.ENDPOINT_MEAN,
    )
    return inputs, state


@pytest.mark.parametrize("drift,missing", [(-4, 7450), (4, 7550)])
def test_explicit_ordinary_nanosecond_head_rejects_drifting_clean_start(
    drift, missing
):
    from cuphoton.xbeam.buffered_reader.clocks import LeadingRawHeadContract

    inputs, state = clean_clock_return_inputs(0, drift)
    contract = inputs.published_normal_full_window_contract(
        state,
        window_extent(),
        control_word_bits=30,
        ordinary_head_contract=LeadingRawHeadContract(4096000000),
    )
    facts = inputs.compile(
        state,
        frame_bytes=8224,
        control_word_bits=30,
        window_clock_contract=contract,
    )
    assert facts.complete
    assert facts.ineligible_intervals == ((missing, missing + 1),)
    assert not contract.decoder_clamps_negative_offsets.flags.writeable
    part = contract.for_scope(7401, 7601)
    np.testing.assert_array_equal(
        part.decoder_clamps_negative_offsets,
        contract.decoder_clamps_negative_offsets[101:301],
    )


def test_ordinary_head_policy_remains_explicit_and_rate_checked():
    from cuphoton.xbeam.buffered_reader.clocks import LeadingRawHeadContract

    inputs, state = clean_clock_return_inputs(0, 4)
    default = inputs.published_normal_full_window_contract(
        state, window_extent(), control_word_bits=30
    )
    assert default.decoder_clamps_negative_offsets is None
    with pytest.raises(ValueError, match="sample rate differs"):
        inputs.published_normal_full_window_contract(
            state,
            window_extent(),
            control_word_bits=30,
            ordinary_head_contract=LeadingRawHeadContract(256),
        )


def test_explicit_ordinary_head_clamps_accepted_negative_decoder_offset():
    from cuphoton.xbeam.buffered_reader.clocks import LeadingRawHeadContract

    inputs, state = clean_clock_return_inputs(-2, 0)
    contract = inputs.published_normal_full_window_contract(
        state,
        window_extent(),
        control_word_bits=30,
        ordinary_head_contract=LeadingRawHeadContract(4096000000),
    )
    facts = inputs.compile(
        state,
        frame_bytes=8224,
        control_word_bits=30,
        window_clock_contract=contract,
    )
    assert facts.complete
    assert not facts.ineligible_intervals
    assert not facts.validity.partial_events
    sources = {
        i: s.physical_sample_start + (i - s.first_fft) * 262144
        for s in facts.addresses.spans
        for i in range(s.first_fft, s.stop_fft)
    }
    for i in range(7300, 7700, 50):
        assert sources[i] == 2002 * 32768 + i * 262144 - 32768


@pytest.mark.parametrize(
    "clamp", [np.zeros(400, dtype=np.int64), np.zeros(399, dtype=np.bool_)]
)
def test_decoder_clamp_vector_is_typed_and_scope_bound(clamp):
    inputs, state = clean_clock_return_inputs(0, 0)
    contract = inputs.published_normal_full_window_contract(
        state, window_extent(), control_word_bits=30
    )
    with pytest.raises(TypeError, match="matching bool vector"):
        replace(contract, decoder_clamps_negative_offsets=clamp)


def test_request_endpoints_select_models_across_epochs():
    job, ds, band, freq, scan, geometry = model_job()
    later = replace(
        scan.station_models[0],
        epoch_offset_s=Decimal(120),
        delay_coefficients_s=(1.0,),
    )
    scan = replace(scan, station_models=(*scan.station_models, later))
    job = replace(job, scans=(scan,))
    result = evaluate((job, ds, band, freq, scan, geometry), stop=250)
    np.testing.assert_array_equal(
        result.leading_head_inputs.endpoint_delays_seconds, [0, 0, 0, 1, 1]
    )
    np.testing.assert_array_equal(result.endpoint_drifts, [0, 0, 1, 0, 0])
    np.testing.assert_array_equal(
        result.desired_sample_starts, np.array([0, 50, 100, 149, 199]) * 256
    )
