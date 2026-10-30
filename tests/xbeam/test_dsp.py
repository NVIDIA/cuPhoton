# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from cuphoton.xbeam import dsp


def test_polynomial_uses_ascending_coefficients():
    result = dsp.evaluate_polynomial(
        [1.0, 2.0, 3.0], np.array([0.0, 2.0]), np
    )
    np.testing.assert_allclose(result, [1.0, 17.0])


def test_fringe_and_lo_phase_signs():
    phasor = dsp.fringe_phasor(
        delay_s=np.array([0.25]),
        time_s=np.array([0.0]),
        reference_frequency_hz=1.0,
        lo_offset_hz=0.0,
        xp=np,
    )
    np.testing.assert_allclose(phasor, [-1j], atol=1e-15)
    lo = dsp.fringe_phasor(
        delay_s=np.array([0.0]),
        time_s=np.array([0.25]),
        reference_frequency_hz=0.0,
        lo_offset_hz=1.0,
        xp=np,
    )
    np.testing.assert_allclose(lo, [1j], atol=1e-15)


def test_real_channelization_omits_nyquist_and_orders_usb():
    samples = np.ones(8, dtype=np.float32)
    result = dsp.channelize(
        samples,
        sample_rate_hz=8.0,
        sky_reference_hz=100.0,
        sideband="U",
        sampling_kind="real",
        xp=np,
    )
    assert result.spectrum.shape == (1, 4)
    np.testing.assert_array_equal(
        result.sky_frequencies_hz, [100, 101, 102, 103]
    )
    np.testing.assert_allclose(result.spectrum[0], [8, 0, 0, 0])


def test_lsb_channelization_is_in_increasing_sky_frequency():
    samples = np.ones(8, dtype=np.float32)
    result = dsp.channelize(
        samples,
        sample_rate_hz=8.0,
        sky_reference_hz=100.0,
        sideband="L",
        sampling_kind="real",
        xp=np,
    )
    np.testing.assert_array_equal(
        result.sky_frequencies_hz, [97, 98, 99, 100]
    )
    np.testing.assert_allclose(result.spectrum[0], [0, 0, 0, 8])


def test_complex_tone_lands_in_predicted_increasing_sky_channel():
    count = 8
    samples = np.exp(2j * np.pi * 2 * np.arange(count) / count).astype(
        np.complex64
    )
    result = dsp.channelize(
        samples,
        sample_rate_hz=8.0,
        sky_reference_hz=100.0,
        sideband="U",
        sampling_kind="complex",
        xp=np,
    )
    assert np.all(np.diff(result.sky_frequencies_hz) > 0)
    peak = int(np.argmax(np.abs(result.spectrum[0])))
    assert result.sky_frequencies_hz[peak] == 102.0
    np.testing.assert_allclose(abs(result.spectrum[0, peak]), 8.0, rtol=1e-7)


def test_analytic_complex_lsb_convention():
    station_a = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j])
    station_b = np.array([1 + 1j, 1 - 1j, -1 - 1j, -1 + 1j])
    raw_a = np.fft.fft(station_a)
    raw_b = np.fft.fft(station_b)
    spectrum_b, relative_hz = dsp.complex_lsb_channels(raw_b, 4e6, np)

    np.testing.assert_allclose(raw_a, [0, 4 + 4j, 0, 0], atol=1e-12)
    np.testing.assert_allclose(raw_b, [0, 0, 0, 4 + 4j], atol=1e-12)
    np.testing.assert_allclose(spectrum_b, [0, 4 - 4j, 0, 0], atol=1e-12)
    np.testing.assert_array_equal(relative_hz, [-4e6, -3e6, -2e6, -1e6])
    np.testing.assert_allclose(
        dsp.cross_power(raw_a, spectrum_b, np), [0, 32j, 0, 0], atol=1e-12
    )

    dc = np.fft.fft(np.full(4, 1 + 1j))
    dc_lsb, _ = dsp.complex_lsb_channels(dc, 4e6, np)
    np.testing.assert_allclose(dc_lsb, [4 + 4j, 0, 0, 0], atol=1e-12)

    reference = dsp.fringe_phasor(
        np.array([1.2377450980392156e-7]), np.array([0.0]), 204e6, 0.0, np
    )
    np.testing.assert_allclose(reference, [-1j], atol=1e-12)
    fringed_raw_b = np.fft.fft(station_b * reference[0])
    fringed_spectrum_b, _ = dsp.complex_lsb_channels(fringed_raw_b, 4e6, np)
    np.testing.assert_allclose(fringed_spectrum_b[1], 4 + 4j, atol=1e-12)
    corrected = dsp.apply_fractional_delay(
        fringed_spectrum_b[None, :], relative_hz, 1.25e-7, np
    )
    np.testing.assert_allclose(corrected[0, 1], -4j * np.sqrt(2), atol=1e-12)
    np.testing.assert_allclose(
        dsp.cross_power(raw_a, corrected[0], np)[1],
        -16 * np.sqrt(2) + 16j * np.sqrt(2),
        atol=1e-12,
    )


def test_fractional_delay_is_linear_phase():
    spectrum = np.ones((1, 4), np.complex64)
    corrected = dsp.apply_fractional_delay(
        spectrum, np.array([0.0, 1.0, 2.0, 3.0]), 0.125, np
    )
    expected = np.exp(2j * np.pi * np.arange(4) * 0.125)
    np.testing.assert_allclose(corrected[0], expected, atol=1e-15)


def test_cross_power_and_channel_averaging():
    first = np.array([[1 + 1j, 2 + 0j, 3 + 0j, 4 + 0j]])
    second = np.ones_like(first) * (1 + 1j)
    product = dsp.cross_power(first, second, np)
    np.testing.assert_allclose(product[0, 0], 2 + 0j)
    averaged = dsp.average_channels(product, 2, np)
    np.testing.assert_allclose(averaged, product.reshape(1, 2, 2).mean(-1))


def test_midnight_normalization_and_empty_window_indices():
    assert dsp.mjd_seconds(60000, 86400.25) == (60001, 0.25)
    assert dsp.mjd_seconds(60000, -0.25) == (59999, 86399.75)
    assert dsp.group_midpoints([0.25, 2.25], 0.0, 1.0) == {0: [0], 2: [1]}


@pytest.mark.parametrize("size", [5, 6])
def test_real_channelization_retains_last_bin_without_a_nyquist_bin(size):
    samples = np.arange(size, dtype=float)
    result = dsp.channelize(
        samples,
        sample_rate_hz=float(size),
        sky_reference_hz=100.0,
        sideband="U",
        sampling_kind="real",
        xp=np,
    )
    expected = np.fft.rfft(samples)
    if size % 2 == 0:
        expected = expected[:-1]
    np.testing.assert_allclose(result.spectrum[0], expected)
    np.testing.assert_array_equal(
        result.sky_frequencies_hz, 100 + np.arange(3)
    )
