# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Repeated native solves retain stream, shape and thread isolation."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from typing import Any

import numpy as np
import pytest

from cuphoton.xpois.ois import (
    GaussianBasisComponent,
    solve_constant_kernel_device,
)


@pytest.fixture
def native_cupy() -> Any:
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("native xPois requires an actual CUDA device")
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    pytest.importorskip("cuphoton.core._imaging_ext")
    return cp


def _solve_and_compare(cp, shape, seed, sparse=False):
    rng = np.random.default_rng(seed)
    reference = cp.asarray(rng.normal(size=shape))
    target = cp.asarray(rng.normal(size=shape))
    mask = np.ones(shape, dtype=bool)
    if sparse:
        mask[::2, :] = False
    options = dict(
        kernel_shape=(5, 7),
        fit_mask=cp.asarray(mask),
        background_degree=seed % 2,
    )
    components = [GaussianBasisComponent(1.3, 1)]
    actual = solve_constant_kernel_device(
        reference, target, components, backend="native", **options
    )
    expected = solve_constant_kernel_device(
        reference, target, components, backend="cupy", **options
    )
    assert actual.fit_pixel_count == expected.fit_pixel_count
    assert actual.dof == expected.dof
    assert actual.chi2 == pytest.approx(expected.chi2, rel=5e-12)
    np.testing.assert_array_equal(
        cp.asnumpy(actual.fit_mask), cp.asnumpy(expected.fit_mask)
    )
    for name in ("kernel", "matched", "residual", "background"):
        np.testing.assert_allclose(
            cp.asnumpy(getattr(actual, name)),
            cp.asnumpy(getattr(expected, name)),
            rtol=5e-12,
            atol=5e-12,
        )
    return actual


def test_native_workspace_reuses_shapes_and_recovers_after_error(native_cupy):
    cp = native_cupy
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(2)]
    retained = None
    snapshot = None
    for index, shape in enumerate(((41, 43), (41, 43), (61, 49), (41, 43))):
        with streams[index % 2]:
            if index == 1:
                image = cp.asarray(np.ones(shape))
                with pytest.raises(ValueError, match="at least one interior"):
                    solve_constant_kernel_device(
                        image,
                        image,
                        [GaussianBasisComponent(1.3, 1)],
                        kernel_shape=(5, 7),
                        fit_mask=cp.asarray(np.zeros(shape, dtype=bool)),
                        backend="native",
                    )
            result = _solve_and_compare(cp, shape, index, sparse=index == 1)
            if retained is None:
                retained = result
                snapshot = cp.asnumpy(result.matched)
    # Returned arrays belong to the caller, never to the reused scratch.
    np.testing.assert_array_equal(cp.asnumpy(retained.matched), snapshot)


def test_native_workspace_isolated_between_threads(native_cupy):
    cp = native_cupy
    device = cp.cuda.runtime.getDevice()
    barrier = Barrier(2)

    def worker(seed):
        with cp.cuda.Device(device), cp.cuda.Stream(non_blocking=True):
            for iteration in range(3):
                barrier.wait(timeout=30)
                _solve_and_compare(
                    cp,
                    (37 + seed * 4, 45 + iteration * 2),
                    seed + iteration,
                    sparse=bool(seed),
                )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(worker, seed) for seed in range(2)]
        for future in futures:
            future.result(timeout=60)
