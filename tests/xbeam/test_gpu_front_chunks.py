# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Qualification of parallel real-input front rotation."""

from types import SimpleNamespace

import numpy as np
import pytest

from cuphoton.xbeam import gpu_kernels


@pytest.mark.parametrize(
    ("length", "rows", "bands", "complex_input", "selected"),
    [
        (262144, 7, 2, False, False),
        (262144, 8, 1, False, True),
        (262144, 8, 2, False, True),
        (262144, 1600, 1, False, True),
        (262144, 1600, 2, False, True),
        (262144, 1601, 1, False, False),
        (262144, 1601, 2, False, False),
        (262144, 4350, 1, False, False),
        (262144, 4350, 2, False, False),
        (262144, 8, 1, True, False),
        (16384, 8, 1, False, False),
    ],
)
def test_front_chunk_selection(length, rows, bands, complex_input, selected):
    counts = gpu_kernels._front_chunk_counts(
        fft_length=length,
        block_count=rows,
        band_count=bands,
        complex_input=complex_input,
        xp=np,
    )
    assert (counts is not None) == selected


def _fixture(cp, layout, authoritative):
    n, rows, frame_samples = 262144, 8, 1024
    bands = 2 if layout == "bands" else 1
    frames = rows * n // frame_samples + 1
    validity = np.ones(frames, dtype=np.uint8)
    validity[[3, 4, 257, 1023]] = 0
    offsets = np.array([0, 1, -1, 2, -2, 1, 0, -1], dtype=np.int64)
    cutovers = np.array([0, 4095, 4096, 4097, n - 1, n, 0, 5])
    starts = np.array([0, 4095, 4096, 4097, 0, 0, 7, 0])
    stops = np.array([n, n, n - 1, n, 4095, 0, n, n])
    host = np.zeros((bands, rows), dtype=gpu_kernels.BLOCK_PARAMETERS_DTYPE)
    host["output_index"] = np.arange(rows, dtype=np.int64) * n
    host["coarse_sample"] = np.array([13, -2, 1, 0, 0, 0, 0, -3])
    host["block_start"] = np.arange(rows) * n / 4096000000.0
    host["delay_zero"] = 0.0123456789012345
    host["delay_step"] = 7.125e-7 / 4096000000.0
    host["delay_quadratic"] = np.linspace(-1e-23, 1e-23, rows)
    host["source_offset"] = offsets
    host["source_cutover"] = cutovers
    host["valid_start"] = starts
    host["valid_stop"] = stops
    if bands == 2:
        host[1, :]["delay_zero"] *= -1
        host[1, :]["source_offset"] *= -1
    common = dict(
        source_validity=cp.asarray(validity),
        validity_stride=frame_samples,
        block_parameters=cp.asarray(host.view(np.uint8).reshape(-1)),
        fft_length=n,
        sample_rate_hz=4096000000.0,
        xp=cp,
    )
    if layout == "decoded":
        source = np.tile(
            np.array([-3.3359, -1, 1, 3.3359], np.float32),
            frames * frame_samples // 4,
        )
        common["source"] = cp.asarray(source)
    else:
        common.update(
            source=cp.full(
                frames * frame_samples * bands // 4, 0xE4, dtype=cp.uint8
            ),
            source_sample_count=frames * frame_samples,
            packed_samples_per_frame=frame_samples,
            packed_payload_bytes=frame_samples * bands // 4,
            packed_channel_count=bands,
        )
    if bands == 2:
        common.update(
            source_start_samples=[-11, 17],
            fringe_reference_hz=[210000000000.0, 210000123000.0],
            lo_offset_hz=[0.5, -0.25],
            wall_lo_cycles=[0.125, -0.375],
            packed_channel_indices=[1, 0],
        )
        invoke = gpu_kernels.front_rotate_packed_real_bands
    else:
        common.update(
            authoritative_validity=authoritative,
            source_start_sample=-11,
            fringe_reference_hz=210000000000.0,
            lo_offset_hz=0.5,
            wall_lo_cycles=0.125,
            complex_input=False,
            _fft_batch_rows=10,
        )
        if layout == "packed":
            common["packed_channel_index"] = 0
        invoke = gpu_kernels.front_rotate
    expected_weights = np.empty((bands, rows), dtype=np.float64)
    samples = np.arange(n, dtype=np.int64)
    for band in range(bands):
        origin = common.get("source_start_samples", [-11])[band]
        for row, parameter in enumerate(host[band]):
            indices = (
                parameter["output_index"]
                - parameter["coarse_sample"]
                - origin
                + samples
                + (samples >= parameter["source_cutover"])
                * parameter["source_offset"]
            )
            valid = (
                (indices >= 0)
                & (indices < frames * frame_samples)
                & (samples >= parameter["valid_start"])
                & (samples < parameter["valid_stop"])
            )
            if not authoritative or bands == 2:
                valid[valid] &= validity[indices[valid] // frame_samples] != 0
            expected_weights[band, row] = np.count_nonzero(valid) / n
    return SimpleNamespace(
        invoke=invoke,
        kwargs=common,
        weights=expected_weights.reshape(
            (-1,) if bands == 1 else (bands, rows)
        ),
    )


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("layout", "authoritative"),
    [
        ("decoded", False),
        ("decoded", True),
        ("packed", False),
        ("packed", True),
        ("bands", False),
    ],
)
@pytest.mark.parametrize("scratch_oom", [False, True])
def test_chunked_front_preserves_bits_and_validity(
    monkeypatch, layout, authoritative, scratch_oom
):
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.Device(0).use()
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(str(exc))
    fixture = _fixture(cp, layout, authoritative)
    with monkeypatch.context() as original:
        original.setattr(gpu_kernels, "_front_chunk_counts", lambda **_: None)
        expected, expected_weights = fixture.invoke(**fixture.kwargs)
    if scratch_oom:
        empty = cp.empty

        def fail_counts(shape, *args, **kwargs):
            if kwargs.get("dtype") == cp.uint32:
                raise cp.cuda.memory.OutOfMemoryError(2048, 2048)
            return empty(shape, *args, **kwargs)

        monkeypatch.setattr(cp, "empty", fail_counts)
    actual, weights = fixture.invoke(**fixture.kwargs)
    np.testing.assert_array_equal(
        cp.asnumpy(actual).view(np.uint32),
        cp.asnumpy(expected).view(np.uint32),
    )
    np.testing.assert_array_equal(
        cp.asnumpy(weights).view(np.uint64),
        cp.asnumpy(expected_weights).view(np.uint64),
    )
    np.testing.assert_array_equal(cp.asnumpy(weights), fixture.weights)
    if layout != "bands":
        np.testing.assert_array_equal(
            cp.asnumpy(actual[8:]).view(np.uint32), 0
        )
