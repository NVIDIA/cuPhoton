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


def _record_front_launches(monkeypatch, *, execute):
    calls = []

    def install(name):
        original = getattr(gpu_kernels, name)

        def factory(*factory_args):
            kernel = original(*factory_args) if execute else None

            def launch(grid, block, arguments):
                calls.append((name, factory_args, grid, block, arguments))
                if execute:
                    return kernel(grid, block, arguments)

            return launch

        monkeypatch.setattr(gpu_kernels, name, factory)

    for name in (
        "_front_kernel",
        "_front_chunked_kernel",
        "_front_packed_real_bands_kernel",
        "_front_chunked_real_bands_kernel",
        "_front_weight_kernel",
    ):
        install(name)
    return calls


def _assert_front_dispatch(calls, layout, scratch_oom, rotated, weights):
    bands, rows = (2 if layout == "bands" else 1), 8
    if scratch_oom:
        name = (
            "_front_packed_real_bands_kernel"
            if layout == "bands"
            else "_front_kernel"
        )
        assert len(calls) == 1
        assert calls[0][0] == name
        assert calls[0][2] == (
            (rows, bands) if layout == "bands" else (rows,)
        )
        assert calls[0][3] == (256,)
        assert calls[0][4][-1] is weights
    else:
        name = (
            "_front_chunked_real_bands_kernel"
            if layout == "bands"
            else "_front_chunked_kernel"
        )
        assert [call[0] for call in calls] == [name, "_front_weight_kernel"]
        assert calls[0][2:4] == ((rows, bands, 64), (256,))
        counts = calls[0][4][-1]
        assert counts.shape == (bands, rows, 64)
        assert counts.dtype == np.dtype(np.uint32)
        assert calls[1][2:4] == ((rows * bands,), (64,))
        assert calls[1][4][0] is counts
        assert tuple(int(value) for value in calls[1][4][1:3]) == (
            64,
            262144,
        )
        assert calls[1][4][-1] is weights
    if layout != "bands":
        assert calls[0][1] == (
            (False, layout == "packed")
            if scratch_oom
            else (layout == "packed",)
        )
    assert calls[0][4][-2] is rotated


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
def test_front_dispatch_launches_chunked_kernels_and_weight_reducer_on_cpu(
    monkeypatch, layout, authoritative, scratch_oom
):
    class Backend:
        cuda = SimpleNamespace(
            memory=SimpleNamespace(OutOfMemoryError=MemoryError)
        )

        def __getattr__(self, name):
            return getattr(np, name)

        def empty(self, shape, *, dtype):
            if scratch_oom and dtype == np.uint32:
                raise MemoryError("chunk count allocation failed")
            return np.empty(shape, dtype=dtype)

    fixture = _fixture(Backend(), layout, authoritative)
    calls = _record_front_launches(monkeypatch, execute=False)
    rotated, weights = fixture.invoke(**fixture.kwargs)
    _assert_front_dispatch(calls, layout, scratch_oom, rotated, weights)


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
    calls = _record_front_launches(monkeypatch, execute=True)
    actual, weights = fixture.invoke(**fixture.kwargs)
    _assert_front_dispatch(calls, layout, scratch_oom, actual, weights)
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
