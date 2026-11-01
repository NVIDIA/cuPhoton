# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Scheduling experiments must not change arithmetic or output addressing."""

import numpy as np
import pytest

from cuphoton.xbeam import gpu_kernels, kernel_compilation, runtime_tuning
from cuphoton.xbeam.core import gpu_products


@pytest.mark.parametrize("value", ["-1", "garbage", "1.5"])
def test_invalid_pack_limit(monkeypatch, value):
    monkeypatch.setenv("CUPHOTON_XBEAM_XMAC_PACKS_PER_LAUNCH", value)
    with pytest.raises(ValueError, match="non-negative"):
        runtime_tuning.xmac_packs_per_launch()


def test_priority_configuration(monkeypatch):
    name = "CUPHOTON_XBEAM_CUFILE_HIGH_PRIORITY"
    monkeypatch.delenv(name, raising=False)
    assert not runtime_tuning.cufile_high_priority("posix")
    monkeypatch.setenv(name, "1")
    assert runtime_tuning.cufile_high_priority("cufile")
    with pytest.raises(ValueError, match="cuFile"):
        runtime_tuning.cufile_high_priority("posix")
    monkeypatch.setenv(name, "yes")
    with pytest.raises(ValueError, match="0 or 1"):
        runtime_tuning.cufile_high_priority("cufile")


@pytest.mark.gpu
@pytest.mark.parametrize("averaging", [1, 4, 32])
def test_bounded_packs_are_identical(monkeypatch, averaging):
    cp = pytest.importorskip("cupy")
    if not cp.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    rng = np.random.default_rng(41)
    # Padded rows, two groups, mixed cross/auto products and sparse offsets.
    spectra = [
        cp.asarray(
            (
                rng.normal(size=(17, 160)) + 1j * rng.normal(size=(17, 160))
            ).astype(np.complex64)
        )[:, :128]
        for _ in range(2)
    ]
    weights = [cp.asarray(rng.uniform(size=17)) for _ in range(2)]
    count = 131
    packs = [
        {
            "spectra": spectra,
            "product_bands": [(0, 1), (0, 0)],
            "normalizations": [2.0, 3.0],
            "weight_pairs": [
                (weights[0], weights[1]),
                (weights[0], weights[0]),
            ],
            "multiply_weights": [True, False],
            "product_rows": [2 * i, 2 * i + 1],
        }
        for i in range(count)
    ]
    channels = 128 // averaging
    offsets = np.arange(count * 4, dtype=np.int64).reshape(count * 2, 2)
    kwargs = dict(
        channels_to_average=averaging,
        group_starts=np.array([0, 8], dtype=np.int32),
        group_lengths=np.array([8, 9], dtype=np.int32),
        output_offsets=offsets * (channels + 3),
        weight_offsets=offsets * 2,
        xp=cp,
    )
    original = kernel_compilation._multi_visibility_kernel(2, 2)
    launches = []

    def tracked(grid, block, args):
        launches.append((grid[2], int(args[-3])))
        return original(grid, block, args)

    monkeypatch.setattr(
        gpu_products, "_multi_visibility_kernel", lambda *bounds: tracked
    )
    results = []
    for limit in [0, 1, 32, 64, 128, 256]:
        monkeypatch.setenv("CUPHOTON_XBEAM_XMAC_PACKS_PER_LAUNCH", str(limit))
        output = cp.full(count * 4 * (channels + 3), -999, dtype=cp.complex64)
        weight_sum = cp.full(count * 8, -999, dtype=cp.float64)
        launches.clear()
        gpu_kernels.batched_visibility_product_packs(
            packs, output=output, weight_sum=weight_sum, **kwargs
        )
        results.append((cp.asnumpy(output), cp.asnumpy(weight_sum)))
        step = min(limit or count, count)
        assert launches == [
            (min(step, count - start), start)
            for start in range(0, count, step)
        ]
        for expected, actual in zip(results[0], results[-1], strict=True):
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.gpu
def test_priority_stream(monkeypatch):
    cp = pytest.importorskip("cupy")
    if not cp.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    name = "CUPHOTON_XBEAM_CUFILE_HIGH_PRIORITY"
    monkeypatch.setenv(name, "0")
    normal = runtime_tuning.cufile_strip_stream(cp)
    monkeypatch.setenv(name, "1")
    urgent = runtime_tuning.cufile_strip_stream(cp)
    assert urgent.priority <= normal.priority


@pytest.mark.gpu
@pytest.mark.parametrize(
    "band_count,product_count", [(1, 2), (2, 4), (4, 8), (8, 8)]
)
@pytest.mark.parametrize("averaging", [1, 4, 32])
def test_specialized_packs_match_generic_exactly(
    monkeypatch, band_count, product_count, averaging
):
    cp = pytest.importorskip("cupy")
    if not cp.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    rng = np.random.default_rng(2026)
    spectra = [
        cp.asarray(
            (
                rng.normal(size=(19, 160)) + 1j * rng.normal(size=(19, 160))
            ).astype(np.complex64)
        )[:, :128]
        for _ in range(band_count)
    ]
    weights = [cp.asarray(rng.uniform(size=19)) for _ in spectra]
    packs = []
    for count in (2, product_count):
        pairs = [(i % band_count, (i + 1) % band_count) for i in range(count)]
        first_row = sum(len(p["product_bands"]) for p in packs)
        packs.append(
            dict(
                spectra=spectra,
                product_bands=pairs,
                normalizations=[2.0 + i for i in range(count)],
                weight_pairs=[(weights[a], weights[b]) for a, b in pairs],
                multiply_weights=[bool(i % 2) for i in range(count)],
                product_rows=list(range(first_row, first_row + count)),
            )
        )
    rows = product_count + 2
    channels = 128 // averaging
    offsets = np.arange(rows * 2, dtype=np.int64).reshape(rows, 2)
    results = []
    selector = kernel_compilation._multi_visibility_kernel
    for specialized in (False, True):
        monkeypatch.setattr(
            gpu_products,
            "_multi_visibility_kernel",
            selector if specialized else lambda *bounds: selector(),
        )
        output = cp.full(rows * 2 * (channels + 3), -999, dtype=cp.complex64)
        weight_sum = cp.full(rows * 4, -999, dtype=cp.float64)
        gpu_kernels.batched_visibility_product_packs(
            packs,
            channels_to_average=averaging,
            group_starts=np.array([0, 8], dtype=np.int32),
            group_lengths=np.array([8, 11], dtype=np.int32),
            output_offsets=offsets * (channels + 3),
            weight_offsets=offsets * 2,
            output=output,
            weight_sum=weight_sum,
            xp=cp,
        )
        results.append((cp.asnumpy(output), cp.asnumpy(weight_sum)))
    for generic, specialized in zip(*results, strict=True):
        np.testing.assert_array_equal(specialized, generic)
