# SPDX-License-Identifier: Apache-2.0
"""Compare native and Python returned-window ownership."""

import importlib.util
import os
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from cuphoton.xbeam.adapters.radio.models import BufferedReaderGeometry
from cuphoton.xbeam.buffered_reader.clocks import (
    LeadingRawHeadContract,
    LeadingRawHeadInputs,
    compile_leading_raw_head_coordinates,
)
from cuphoton.xbeam.buffered_reader.geometry import BufferedWindowGeometry
from cuphoton.xbeam.buffered_reader.replay import bind_known_source_end
from tests.xbeam.test_buffered_window_job_requests import (
    clean_clock_return_inputs,
    paired_return_inputs,
    window_extent,
)


@pytest.fixture(scope="module")
def binding():
    directory = os.environ.get("XBEAM_NATIVE_BINDING_DIR")
    if not directory:
        pytest.skip("set XBEAM_NATIVE_BINDING_DIR")
    files = list(Path(directory).rglob("_binding*.so"))
    spec = importlib.util.spec_from_file_location("_binding", files[0])
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    return native


@pytest.mark.parametrize(
    "allowance", [0.0, 1e-9, 0.00249, 0.0025, np.nextafter(0.0025, 0.0)]
)
@pytest.mark.parametrize("ratio", [1, 3, 8])
def test_native_buffer_sizing(binding, allowance, ratio):
    reader = BufferedReaderGeometry(ratio * 8, 8, 20)
    expected = BufferedWindowGeometry.from_reader(
        reader,
        samples_per_frame=32768,
        fft_samples=262144,
        sample_rate=4096000000,
        subintegration_s=Decimal("0.0032"),
        reference_rate_allowance=allowance,
    )
    actual = binding.window_geometry(
        32768, 262144, 4096000000, 32, 10000, ratio * 8, 8, 20, allowance
    )
    assert actual[:5] == (
        expected.frames_per_fft,
        expected.ffts_per_request,
        expected.buffered_ffts,
        expected.ring_segments,
        expected.allocation_frames,
    )
    assert actual[5] * ratio == actual[4]


@pytest.mark.parametrize("origin", [0, 2**53 + 17, -(2**53) - 31])
@pytest.mark.parametrize("delay", [-0.006064, 0.012096, 0.0, -0.0000000009])
def test_native_leading_head(binding, origin, delay):
    inputs = LeadingRawHeadInputs(
        np.asarray([origin], np.int64),
        np.asarray([delay]),
        np.asarray([delay]),
    )
    expected = compile_leading_raw_head_coordinates(
        inputs, LeadingRawHeadContract(4096000000, 1), samples_per_frame=32768
    )
    assert (
        binding.leading_frame_head(origin, delay, 4096000000, 32768, 1)
        == expected.head_sample_starts[0]
    )


def compare_return(binding, inputs, cache):
    extent = window_extent()
    source = cache.source_state
    timeline = source.timeline
    stop = source.raw_source_stop
    cache = replace(
        cache, source_state=replace(source, recorded_source_stop=stop)
    )
    expected = inputs.published_normal_full_window_contract(
        cache,
        extent,
        control_word_bits=30,
        ordinary_head_contract=LeadingRawHeadContract(
            int(inputs.sample_rate), 1
        ),
    )
    expected = bind_known_source_end(expected, stop)
    runs = list(
        zip(
            timeline.physical_starts.tolist(),
            timeline.physical_stops.tolist(),
            timeline.timestamp_starts.tolist(),
            strict=True,
        )
    )
    native_timeline = binding.NativeTimeline(
        runs, source.request_state.allocation_frames
    )
    # The fixture uses the physical-frame event owner, which is explicit here.
    state = binding.NativeRawState(
        native_timeline,
        source.request_state.samples_per_frame,
        inputs.timestamp_origin_frame,
        False,
    )
    data = {
        key: getattr(inputs, key)
        for key in (
            "first_fft",
            "stop_fft",
            "timeline_fft_count",
            "timestamp_origin_frame",
            "sample_rate",
            "request_fft_starts",
            "request_fft_counts",
            "desired_sample_starts",
            "endpoint_drifts",
            "ordinary_source_starts",
            "output_sample_starts",
            "centre_delays_seconds",
        )
    }
    data["left_delays"] = inputs.leading_head_inputs.endpoint_delays_seconds
    data = {
        k: v.tolist() if isinstance(v, np.ndarray) else v
        for k, v in data.items()
    }
    geometry = inputs.geometry
    packing = geometry.samples_per_byte
    actual = binding.returned_window_contract(
        native_timeline,
        state,
        data,
        (
            packing.numerator,
            packing.denominator,
            geometry.fft_samples,
            geometry.ffts_per_request,
            geometry.alignment_bytes,
            geometry.word_bytes,
        ),
        (
            extent.frame_bytes,
            extent.payload_bytes,
            extent.request_frames,
            extent.read_buffer_ratio,
            packing.numerator,
            packing.denominator,
        ),
        stop,
    )
    facts = inputs.compile(
        cache,
        frame_bytes=extent.frame_bytes,
        control_word_bits=30,
        window_clock_contract=expected,
    )
    assert facts.complete
    expected_sources = {
        k: span.physical_sample_start
        + (k - span.first_fft) * geometry.fft_samples
        for span in facts.addresses.spans
        for k in range(span.first_fft, span.stop_fft)
    }
    timing = dict(facts.source_timing_events)
    replay = binding.replay_rows(
        actual, geometry.fft_samples, int(inputs.sample_rate), stop
    )
    for k, physical, offset, _left, _right, accepted in replay:
        assert accepted == (k in expected_sources)
        if accepted:
            assert physical == expected_sources[k]
            assert offset == timing.get(k, 0)
    fields = (
        "output_samples",
        "aligned_head_samples",
        "physical_head_samples",
        "centre_delays_seconds",
        "framed_extent_samples",
        "decoded_extent_samples",
        "publication_eligible",
        "decoder_clamps_negative_offsets",
    )
    for column, key in enumerate(fields, 1):
        target = getattr(expected, key)
        if target is None:
            target = np.zeros(inputs.stop_fft - inputs.first_fft, bool)
        np.testing.assert_array_equal(
            [row[column] for row in actual], target, err_msg=key
        )


@pytest.mark.parametrize("holes", [None, (61502, 61511), (61499, 61530)])
@pytest.mark.parametrize("delay", [-6064, 0, 5904, 12096])
@pytest.mark.parametrize("first,stop", [(7300, 7900), (7703, 7769)])
def test_native_returned_source_ownership(binding, holes, delay, first, stop):
    inputs, cache = paired_return_inputs(
        holes, delay_us=delay, first=first, stop=stop
    )
    compare_return(binding, inputs, cache)


@pytest.mark.parametrize("phase", [-1.5, -0.5, 0, 0.5, 1.5, 32767.5])
@pytest.mark.parametrize("drift", [-2.0, 0.0, 2.0])
def test_native_ordinary_leading_clock(binding, phase, drift):
    inputs, cache = clean_clock_return_inputs(phase, drift)
    compare_return(binding, inputs, cache)


@pytest.mark.parametrize("epoch", range(64))
@pytest.mark.parametrize("fps", [1, 125000])
def test_native_vdif_time_grid(binding, epoch, fps):
    from cuphoton.xbeam.adapters.vdif.vdif import (
        VdifHeader,
        absolute_frame_slot,
    )
    from tests.xbeam.test_recording import _frame

    header = replace(
        VdifHeader.from_bytes(_frame(payload=bytes(8192), frame_number=0)),
        reference_epoch=epoch,
        seconds_from_epoch=12345,
        frame_number=fps - 1,
    )
    key = (epoch << 54) | (12345 << 24) | (fps - 1)
    assert binding.packed_timestamp_slot(key, fps) == absolute_frame_slot(
        header, fps
    )


def test_cpu_inspection_binding_declines_gpu_execution(binding):
    if binding.cuda_available():
        pytest.skip("this check is for the CPU-only optional binding")
    with pytest.raises(
        RuntimeError, match="native CUDA execution was not built"
    ):
        binding.correlate(
            "unused.input",
            "unused-output",
            "unused-index",
            "scan",
            integration_count=1,
            profile="interop-real-usb-buffered-window-replay-v1",
        )


@pytest.mark.parametrize("missing", ["integration_count", "profile"])
def test_binding_requires_explicit_execution_scope(binding, missing):
    options = {
        "integration_count": 1,
        "profile": "interop-real-usb-buffered-window-replay-v1",
    }
    del options[missing]
    with pytest.raises(TypeError, match=missing):
        binding.correlate(
            "unused.input", "unused-output", "unused-index", "scan", **options
        )


def test_binding_rejects_missing_native_objects(binding):
    with pytest.raises(TypeError, match="timeline"):
        binding.NativeRawState(None, 1, 0, False)
    timeline = binding.NativeTimeline([(0, 1, 0)], 1)
    state = binding.NativeRawState(timeline, 1, 0, False)
    for timeline_arg, state_arg, missing in (
        (None, state, "timeline"),
        (timeline, None, "state"),
    ):
        with pytest.raises(TypeError, match=missing):
            binding.returned_window_contract(
                timeline_arg, state_arg, {}, (), (), 0
            )
