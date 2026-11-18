# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Numerical contracts that could invalidate scene inference."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scene import (
    Source,
    generate,
    kernels,
    pixel_gaussian,
    render,
    write_bundle,
)


def test_integrated_flux_centroid_and_width():
    image = pixel_gaussian(101, 50, 50, 2)
    y, x = np.indices(image.shape)
    assert image.sum() == pytest.approx(1, abs=1e-13)
    assert (image * x).sum() == pytest.approx(50)
    assert (image * y).sum() == pytest.approx(50)
    assert (image * (x - 50) ** 2).sum() == pytest.approx(
        4 + 1 / 12, abs=1e-6
    )


def test_independent_convolution_against_quadrature():
    from scipy.integrate import quad

    source = Source(18.3, 17.8, 100, 1.2, "control")
    image = render([source], sigma=1.4, size=40, background=0)
    width = np.hypot(1.2, 1.4)

    def integral(center, pixel):
        return quad(
            lambda t: (
                np.exp(-((t - center) ** 2) / (2 * width**2))
                / (np.sqrt(2 * np.pi) * width)
            ),
            pixel - 0.5,
            pixel + 0.5,
        )[0]

    assert image[18, 19] == pytest.approx(
        100 * integral(source.x, 19) * integral(source.y, 18), rel=1e-12
    )


def test_bundle_axes_mask_polarity_and_noise_pairing():
    arrays, _ = generate(101, case="outliers")
    assert arrays["images"].shape == (1, 120, 120, 12)
    assert arrays["psfs"].shape == (15, 15, 1, 12)
    assert arrays["masks"].dtype == bool
    assert np.all(arrays["variances"] > 0)
    clean, _ = generate(101)
    good = arrays["masks"]
    np.testing.assert_array_equal(
        arrays["images"][good], clean["images"][good]
    )
    assert np.all(arrays["images"][~good] - clean["images"][~good] > 99)
    np.testing.assert_allclose(arrays["psfs"].sum(axis=(0, 1)), 1)


def test_real_border_context_and_refuse_overwrite(tmp_path):
    arrays, _ = generate(5)
    # The 96-pixel fitting ROI retains real support for the radius-7 PSF.
    assert (120 - 96) // 2 > kernels([1.7]).shape[0] // 2
    assert np.all(arrays["images"][:, :7].std(axis=-1) > 0)
    destination = tmp_path / "bundle"
    write_bundle(destination, arrays)
    with pytest.raises(FileExistsError):
        write_bundle(destination, arrays)


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
def test_invalid_variance_refused_before_writes(tmp_path, bad):
    arrays, _ = generate(5)
    arrays["variances"][0, 20, 20, 0] = bad
    destination = tmp_path / "invalid"
    with pytest.raises(ValueError, match="finite and positive"):
        write_bundle(destination, arrays)
    assert not destination.exists()
