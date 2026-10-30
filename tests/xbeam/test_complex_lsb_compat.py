# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from cuphoton.xbeam.dsp import complex_lsb_channels

FIXTURES = [
    ([1 + 101j], [1 + 101j], [-1600.0]),
    ([1 + 101j, 2 + 102j], [1 + 101j, 2 - 102j], [-1600.0, -800.0]),
    (
        [1 + 101j, 2 + 102j, 3 + 103j, 4 + 104j],
        [1 + 101j, 4 - 104j, 3 - 103j, 2 - 102j],
        [-1600.0, -1200.0, -800.0, -400.0],
    ),
    (
        [1 + 101j, 2 + 102j, 3 + 103j, 4 + 104j, 5 + 105j],
        [1 + 101j, 5 - 105j, 4 - 104j, 3 - 103j, 2 - 102j],
        [-1600.0, -1280.0, -960.0, -640.0, -320.0],
    ),
    (
        [1 + 101j, 2 + 102j, 3 + 103j, 4 + 104j, 5 + 105j, 6 + 106j],
        [1 + 101j, 6 - 106j, 5 - 105j, 4 - 104j, 3 - 103j, 2 - 102j],
        [
            -1600.0,
            -1333.3333333333335,
            -1066.6666666666667,
            -800.0,
            -533.3333333333334,
            -266.6666666666667,
        ],
    ),
]


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("fixture", FIXTURES)
def test_labeled_bins_dtype_frequency_and_aliasing(dtype, fixture):
    values, expected, frequencies = fixture
    x = np.array(values, dtype=dtype)
    before = x.copy()
    result = complex_lsb_channels(x, 1600.0, np)
    assert isinstance(result, tuple) and len(result) == 2
    y, f = result
    np.testing.assert_array_equal(y, np.array(expected, dtype=dtype))
    np.testing.assert_array_equal(f, frequencies)
    np.testing.assert_array_equal(x, before)
    assert y.dtype == dtype and f.dtype == np.float64
    assert y.shape == x.shape and f.shape == (x.size,)
    assert np.shares_memory(x, y) == (x.size == 1)
    assert not np.shares_memory(x, f)


@pytest.mark.parametrize("shape", [(2, 5), (2, 3, 5), (0, 5), (2, 0, 5)])
def test_batches(shape):
    values, expected, frequencies = FIXTURES[3]
    x = np.broadcast_to(np.array(values), shape).copy()
    # Different row scales make row mixing visible.
    scales = np.arange(1, np.prod(shape[:-1]) + 1).reshape(shape[:-1] + (1,))
    x *= scales
    before = x.copy()
    y, f = complex_lsb_channels(x, 1600.0, np)
    np.testing.assert_array_equal(
        y, np.broadcast_to(expected, shape) * scales
    )
    np.testing.assert_array_equal(f, frequencies)
    np.testing.assert_array_equal(x, before)
    assert y.shape == shape and y.dtype == x.dtype


@pytest.mark.parametrize("negative_stride", [False, True])
def test_strided_inputs(negative_stride):
    values, expected, frequencies = FIXTURES[3]
    storage = np.full((2, 10), 999 + 888j)
    x = storage[:, ::-2] if negative_stride else storage[:, ::2]
    x[:] = values
    before = storage.copy()
    y, f = complex_lsb_channels(x, 1600.0, np)
    np.testing.assert_array_equal(y, [expected, expected])
    np.testing.assert_array_equal(f, frequencies)
    np.testing.assert_array_equal(storage, before)


@pytest.mark.parametrize("shape", [(), (0,), (3, 0)])
def test_invalid_shape(shape):
    x = np.zeros(shape, dtype=np.complex128)
    before = x.copy()
    with pytest.raises(ValueError, match="at least one FFT bin"):
        complex_lsb_channels(x, 1600.0, np)
    np.testing.assert_array_equal(x, before)


@pytest.mark.parametrize(
    "dtype", [np.float32, np.float64, np.int16, np.int64]
)
def test_observed_real_input_compatibility(dtype):
    x = np.array([1, 2, 3, 4, 5, 6], dtype=dtype)
    y, _ = complex_lsb_channels(x, 1600.0, np)
    np.testing.assert_array_equal(y, [1, 6, 5, 4, 3, 2])
    np.testing.assert_array_equal(x, [1, 2, 3, 4, 5, 6])
    assert y.dtype == dtype


def test_nonfinite_components():
    x = np.empty(6, dtype=np.complex128)
    x.real = [np.nan, 2, np.inf, 4, 5, 7]
    x.imag = [1, np.nan, 3, -np.inf, 6, 8]
    before = x.copy()
    y, _ = complex_lsb_channels(x, 1600.0, np)
    np.testing.assert_array_equal(y.real, [np.nan, 7, 5, 4, np.inf, 2])
    np.testing.assert_array_equal(y.imag, [1, -8, -6, np.inf, -3, np.nan])
    np.testing.assert_array_equal(x.real, before.real)
    np.testing.assert_array_equal(x.imag, before.imag)


@pytest.mark.parametrize(
    "rate,expected",
    [
        (
            48000.0,
            [-48000.0, -40000.0, -32000.0, -24000.0, -16000.0, -8000.0],
        ),
        (0.0, [0.0] * 6),
        (
            -1600.0,
            [
                1600.0,
                1333.3333333333335,
                1066.6666666666667,
                800.0,
                533.3333333333334,
                266.6666666666667,
            ],
        ),
        (np.inf, [-np.inf] * 6),
        (np.nan, [np.nan] * 6),
    ],
)
def test_observed_rate_boundaries(rate, expected):
    values, channels, _ = FIXTURES[4]
    y, f = complex_lsb_channels(np.array(values), rate, np)
    np.testing.assert_array_equal(y, channels)
    np.testing.assert_array_equal(f, expected)
    assert f.dtype == np.float64


def analytic_tone(n, k):
    """DFT orthogonality gives one spectral line."""
    amplitude = 0.75 + 0.5j
    signal = amplitude * np.exp(2j * np.pi * k * np.arange(n) / n)
    transformed = np.fft.fft(signal)
    expected = np.zeros(n, dtype=np.complex128)
    # The observed tone destinations are the negative harmonic modulo N.
    destination = (-k) % n
    expected[destination] = n * (
        amplitude if k == 0 else amplitude.conjugate()
    )
    return transformed, expected


@pytest.mark.parametrize("n,k", [(n, k) for n in (5, 8) for k in range(n)])
def test_analytic_dft_tones(n, k):
    x, expected = analytic_tone(n, k)
    y, _ = complex_lsb_channels(x, 1600.0, np)
    np.testing.assert_allclose(y, expected, rtol=2e-13, atol=2e-13)


@pytest.fixture(scope="module")
def cp():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("No CUDA devices")
        cp.zeros(1).sum().item()
    except (cp.cuda.runtime.CUDARuntimeError, OSError) as error:
        pytest.skip(f"CUDA runtime unavailable: {error}")
    return cp


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("fixture", FIXTURES)
def test_cupy_labeled_bins(cp, dtype, fixture):
    values, expected, frequencies = fixture
    x = cp.asarray(values, dtype=dtype)
    y, f = complex_lsb_channels(x, 1600.0, cp)
    np.testing.assert_array_equal(cp.asnumpy(y), expected)
    np.testing.assert_array_equal(cp.asnumpy(f), frequencies)
    np.testing.assert_array_equal(cp.asnumpy(x), values)
    assert y.dtype == dtype and f.dtype == np.float64
    assert cp.shares_memory(x, y) == (len(values) == 1)


@pytest.mark.gpu
def test_cupy_empty_outer_batch(cp):
    x = cp.empty((0, 8), dtype=cp.complex64)
    y, f = complex_lsb_channels(x, 1600.0, cp)
    assert y.shape == (0, 8) and y.dtype == cp.complex64
    assert f.dtype == cp.float64
    np.testing.assert_array_equal(
        cp.asnumpy(f),
        [-1600.0, -1400.0, -1200.0, -1000.0, -800.0, -600.0, -400.0, -200.0],
    )


@pytest.mark.gpu
@pytest.mark.parametrize("n", [1, 5, 8, 257])
@pytest.mark.parametrize("residual", [0.0, 0.000137])
def test_cuda_channelizer_phase(cp, n, residual):
    from cuphoton.xbeam.cupy_kernels import (
        channelize_and_apply_residual,
        prepare_block_parameters,
    )

    harmonics = sorted({0, n // 2, n - 1})
    tones = [analytic_tone(n, k) for k in harmonics]
    x = np.stack([pair[0] for pair in tones]).astype(np.complex64)
    expected = np.stack([pair[1] for pair in tones])
    rows = len(harmonics)
    residuals = np.full(rows, residual)
    params = prepare_block_parameters(
        output_indices=np.zeros(rows, dtype=np.int64),
        coarse_samples=np.zeros(rows, dtype=np.int64),
        block_starts=np.zeros(rows),
        delay_zero=np.zeros(rows),
        delay_step=np.zeros(rows),
        delay_quadratic=np.zeros(rows),
        sample_rate_hz=1.0,
        centre_delay=-residuals,
        xp=cp,
    )
    device_x = cp.asarray(x)
    y = channelize_and_apply_residual(
        device_x,
        params,
        mapping="complex_lsb",
        channel_count=n,
        baseband_bin_offset=-n,
        channel_spacing_hz=1600.0 / n,
        xp=cp,
    )
    frequencies = np.linspace(-1600.0, -1600.0 / n, n)
    expected *= np.exp(2j * np.pi * residuals[:, None] * frequencies)
    np.testing.assert_allclose(cp.asnumpy(y), expected, rtol=3e-6, atol=3e-5)
    np.testing.assert_array_equal(cp.asnumpy(device_x), x)
    assert y.shape == x.shape and y.dtype == np.complex64
