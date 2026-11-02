# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Scientific and compilation coverage for native source selection."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from cuphoton.xpois import build_compact_source_stamp_mask


def _gpu():
    if importlib.util.find_spec("cuphoton.core._imaging_ext") is None:
        pytest.skip("native imaging extension is not built")
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("CUDA device is unavailable")
    return cp


@pytest.mark.parametrize(
    "max_stamps,dtype",
    [(5, np.float64), (10000, np.float64), (5, np.float16)],
)
def test_native_selection_preserves_masks_order_and_scores(
    max_stamps, dtype, monkeypatch
):
    cp = _gpu()
    rng = np.random.default_rng(823)
    image = rng.normal(100.0, 2.0, (183, 211))
    yy, xx = np.mgrid[-5:6, -5:6]
    for y in range(20, 170, 15):
        for x in range(20, 196, 15):
            image[y - 5 : y + 6, x - 5 : x + 6] += rng.uniform(
                80, 200
            ) * np.exp(-(xx * xx + yy * yy) / 5.0)
    image[5:10] = np.nan
    image = image.astype(dtype)
    variance = np.full(image.shape, 4.0)
    variance[70:73, 40:60] = np.inf
    mask = np.ones(image.shape, dtype=dtype)
    mask[112, 20:80] = 0
    options = dict(
        stamp_size=11,
        max_stamps=max_stamps,
        background_filter_size=7,
        peak_filter_size=5,
        peak_percentile=97.0,
        min_separation=14,
    )
    from cupy._core import core
    from cupy.cuda import compiler

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "native selection attempted CuPy kernel compilation"
        )

    with monkeypatch.context() as patch, cp.cuda.Stream(non_blocking=True):
        patch.setattr(core, "compile_with_cache", forbidden)
        patch.setattr(compiler, "_compile_module_with_cache", forbidden)
        patch.setattr(compiler, "compile_using_nvrtc", forbidden)
        actual = build_compact_source_stamp_mask(
            cp.asarray(image[:, ::-1])[:, ::-1],
            variance=cp.asarray(variance),
            valid_mask=cp.asarray(mask),
            backend="native",
            **options,
        )
    expected = build_compact_source_stamp_mask(
        cp.asarray(image),
        variance=cp.asarray(variance),
        valid_mask=cp.asarray(mask),
        **options,
    )
    assert actual.centers == expected.centers
    assert actual.stamps == expected.stamps
    assert actual.threshold == expected.threshold
    np.testing.assert_array_equal(
        cp.asnumpy(actual.mask), cp.asnumpy(expected.mask)
    )
    for name in ("scores", "compact_fractions", "radial_moments"):
        np.testing.assert_allclose(
            getattr(actual, name), getattr(expected, name), rtol=1e-12, atol=0
        )


def test_native_selection_rejects_nonbinary_masks():
    cp = _gpu()
    image = cp.asarray(np.ones((21, 21)))
    mask = np.ones((21, 21))
    mask[4, 4] = 0.5
    with pytest.raises(ValueError, match="0/1"):
        build_compact_source_stamp_mask(
            image, valid_mask=cp.asarray(mask), backend="native"
        )
