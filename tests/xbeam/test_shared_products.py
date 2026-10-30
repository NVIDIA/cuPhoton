# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""One batched XMAC implementation for radio and generic series products."""

import numpy as np
import pytest

from cuphoton.xbeam import ArraySource, CorrelationSpec, correlate_series
from cuphoton.xbeam.core import gpu_products


def _cupy():
    cp = pytest.importorskip("cupy")
    if not cp.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return cp


@pytest.mark.parametrize(
    "backend", ["numpy", pytest.param("cupy", marks=pytest.mark.gpu)]
)
@pytest.mark.parametrize(
    "dtype", [np.float32, np.float64, np.complex64, np.complex128]
)
@pytest.mark.parametrize("mode", ["linear", "circular"])
def test_tiled_products_match_independent_fp64_sums(backend, dtype, mode):
    if backend == "cupy":
        _cupy()
    rng = np.random.default_rng(730)
    arrays = {
        key: (
            rng.normal(size=261) + 1j * rng.normal(size=261)
            if np.issubdtype(dtype, np.complexfloating)
            else rng.normal(size=261)
        ).astype(dtype)
        for key in ("a", "b")
    }
    masks = {key: rng.random(261) > 0.2 for key in arrays}
    source = ArraySource(arrays, sample_rate_hz=1.0, validity=masks)
    spec = CorrelationSpec(
        (("a", "b"), ("a", "a")),
        23,
        hop_samples=17,
        windows_per_result=5,
        batch_windows=3,
        result_groups_per_tile=2,
        output="both",
        lag_mode=mode,
        backend=backend,
    )
    for result in correlate_series(source, spec):
        for pair, (a, b) in enumerate(spec.pairs):
            spectra, lags, coverages = [], [], []
            for start in result.window_starts:
                selection = slice(start, start + spec.window_samples)
                ma, mb = masks[a][selection], masks[b][selection]
                x = np.where(ma, arrays[a][selection], 0).astype(
                    np.complex128
                )
                y = np.where(mb, arrays[b][selection], 0).astype(
                    np.complex128
                )
                spectra.append(
                    np.fft.fft(x, spec.fft_samples)
                    * np.conj(np.fft.fft(y, spec.fft_samples))
                )
                lag, coverage = [], []
                for shift in result.lag_samples:
                    indices = np.arange(spec.window_samples)
                    other = indices + shift
                    if mode == "linear":
                        valid = (other >= 0) & (other < spec.window_samples)
                        indices, other = indices[valid], other[valid]
                    else:
                        other %= spec.window_samples
                    lag.append(np.sum(x[indices] * np.conj(y[other])))
                    coverage.append(np.sum(ma[indices] & mb[other]))
                lags.append(lag)
                coverages.append(coverage)
            tolerance = (
                2e-6 if result.spectrum.dtype == np.complex64 else 2e-13
            )
            np.testing.assert_allclose(
                result.spectrum[pair],
                np.mean(spectra, axis=0),
                rtol=tolerance,
                atol=tolerance,
            )
            np.testing.assert_allclose(
                result.lag[pair],
                np.mean(lags, axis=0),
                rtol=tolerance,
                atol=tolerance,
            )
            np.testing.assert_array_equal(
                result.lag_valid_samples[pair], np.sum(coverages, axis=0)
            )


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("averaging", [1, 4, 32, 5])
@pytest.mark.parametrize("accumulate", [False, True])
def test_shared_batched_kernel_precision_and_accumulation(
    dtype, averaging, accumulate
):
    cp = _cupy()

    rng = np.random.default_rng(42)
    a = (rng.normal(size=(9, 160)) + 1j * rng.normal(size=(9, 160))).astype(
        dtype
    )
    b = (rng.normal(size=(9, 160)) + 1j * rng.normal(size=(9, 160))).astype(
        dtype
    )
    initial = np.full(2 * 160 // averaging + 7, 3 + 2j, dtype=dtype)
    output = cp.asarray(initial)
    weight = np.linspace(0.1, 0.9, 9)
    weights = cp.empty(2, dtype=cp.float64)
    offsets = np.array([2, 5 + 160 // averaging], dtype=np.int64)
    gpu_products.batched_visibility_product(
        cp.asarray(a),
        cp.asarray(b),
        normalization=3.0,
        channels_to_average=averaging,
        weight_a=cp.asarray(weight),
        group_starts=np.array([0, 5]),
        group_lengths=np.array([5, 4]),
        output_offsets=offsets,
        weight_offsets=np.arange(2),
        output=output,
        weight_sum=weights,
        xp=cp,
        accumulate=accumulate,
    )
    expected = initial.copy()
    for group, (start, stop) in enumerate([(0, 5), (5, 9)]):
        value = np.sum(a[start:stop] * np.conj(b[start:stop]), axis=0)
        value = value.reshape(-1, averaging).mean(axis=1) / 3.0
        target = slice(offsets[group], offsets[group] + len(value))
        expected[target] = value + (initial[target] if accumulate else 0)
    tolerance = 2e-6 if dtype == np.complex64 else 2e-14
    np.testing.assert_allclose(
        cp.asnumpy(output), expected, rtol=tolerance, atol=tolerance
    )
    np.testing.assert_allclose(
        cp.asnumpy(weights), [weight[:5].sum(), weight[5:].sum()], atol=1e-15
    )


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_grouped_adapter_preserves_other_pairs_and_cached_metadata(dtype):
    cp = _cupy()

    rng = np.random.default_rng(18)
    a = (
        rng.normal(size=(3, 19, 67)) + 1j * rng.normal(size=(3, 19, 67))
    ).astype(dtype)
    b = (
        rng.normal(size=(3, 19, 67)) + 1j * rng.normal(size=(3, 19, 67))
    ).astype(dtype)
    output = cp.full((3, 3, 67), 2 + 1j, dtype=dtype)
    cache = {}
    for _ in range(2):
        gpu_products.accumulate_grouped_cross_power(
            cp.asarray(a), cp.asarray(b), output, 1, cache=cache, xp=cp
        )
    assert len(cache) == 1
    actual = cp.asnumpy(output)
    np.testing.assert_array_equal(actual[:, 0], 2 + 1j)
    np.testing.assert_array_equal(actual[:, 2], 2 + 1j)
    expected = np.sum(a * np.conj(b), axis=1)
    expected = (2 + 1j + expected) + expected
    tolerance = 3e-6 if dtype == np.complex64 else 3e-14
    np.testing.assert_allclose(
        actual[:, 1], expected, rtol=tolerance, atol=tolerance
    )


@pytest.mark.gpu
@pytest.mark.parametrize("masked", [False, True])
def test_series_dispatches_through_same_batched_xmac_as_radio(
    monkeypatch, masked
):
    cp = _cupy()

    rng = np.random.default_rng(812)
    arrays = {k: rng.normal(size=4096).astype("f4") for k in ("a", "b")}
    validity = {"a": rng.random(4096) > 0.2} if masked else None
    source = ArraySource(arrays, sample_rate_hz=1.0, validity=validity)
    spec = CorrelationSpec(
        (("a", "b"), ("b", "b")),
        33,
        windows_per_result=7,
        batch_windows=3,
        result_groups_per_tile=4,
        output="both",
    )
    expected = list(correlate_series(source, spec))
    from dataclasses import replace

    calls = []
    original = gpu_products.batched_visibility_product

    def tracked(a, b, **kwargs):
        calls.append((a.dtype, kwargs["accumulate"]))
        return original(a, b, **kwargs)

    monkeypatch.setattr(gpu_products, "batched_visibility_product", tracked)
    actual = list(correlate_series(source, replace(spec, backend="cupy")))
    assert calls and all(accumulate for _, accumulate in calls)
    assert any(dtype == cp.complex64 for dtype, _ in calls)
    if masked:
        assert any(dtype == cp.complex128 for dtype, _ in calls)
    for x, y in zip(actual, expected, strict=True):
        np.testing.assert_allclose(
            x.spectrum, y.spectrum, rtol=3e-6, atol=2e-5
        )
        np.testing.assert_allclose(x.lag, y.lag, rtol=3e-6, atol=2e-6)
        np.testing.assert_array_equal(
            x.simultaneous_valid_samples, y.simultaneous_valid_samples
        )
        np.testing.assert_array_equal(
            x.lag_valid_samples, y.lag_valid_samples
        )
