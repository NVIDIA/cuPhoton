# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from cuphoton.xrep import BBox, ReprojectionSpec, reproject_array


def test_cpu_bilinear_identity_preserves_source() -> None:
    source = np.arange(25, dtype=np.float64).reshape(5, 5)
    spec = ReprojectionSpec(
        mapping=lambda coords: np.asarray(coords, dtype=np.float64),
        output_bbox=BBox(min_x=0, min_y=0, width=5, height=5),
        interpolation="bilinear",
        mapping_grid_step=1,
    )

    result = reproject_array(source, spec, backend="cpu")

    assert np.allclose(result.image, source)
    assert result.mask is not None
    assert not np.any(result.mask)


def test_cpu_lanczos_identity_preserves_source() -> None:
    source = np.arange(25, dtype=np.float64).reshape(5, 5)
    spec = ReprojectionSpec(
        mapping=lambda coords: np.asarray(coords, dtype=np.float64),
        output_bbox=BBox(min_x=0, min_y=0, width=5, height=5),
        interpolation="lanczos3",
        mapping_grid_step=1,
    )

    result = reproject_array(source, spec, backend="cpu")

    assert np.allclose(result.image, source)
    assert result.mask is not None
    assert not np.any(result.mask)


@pytest.mark.parametrize("interpolation", ["bilinear", "lanczos3"])
def test_cpu_zero_weight_nonfinite_neighbors_do_not_poison_identity(
    interpolation: str,
) -> None:
    source = np.array([[5.0, np.nan], [np.inf, -np.inf]])
    spec = ReprojectionSpec(
        mapping=lambda coords: np.asarray(coords, dtype=np.float64),
        output_bbox=BBox(min_x=0, min_y=0, width=1, height=1),
        interpolation=interpolation,
        mapping_grid_step=1,
    )

    with np.errstate(invalid="raise"):
        result = reproject_array(source, spec, backend="cpu")

    assert result.image[0, 0] == pytest.approx(5.0)


def test_cpu_lanczos_mixed_coords_skip_zero_weight_nonfinite() -> None:
    source = np.array([[5.0, np.inf], [np.nan, -np.inf]])
    spec = ReprojectionSpec(
        mapping=lambda coords: (
            np.asarray(coords, dtype=np.float64) * np.array([0.5, 0.0])
        ),
        output_bbox=BBox(min_x=0, min_y=0, width=2, height=1),
        interpolation="lanczos3",
        mapping_grid_step=1,
        area_scaling=False,
    )

    with np.errstate(invalid="raise"):
        result = reproject_array(source, spec, backend="cpu")

    assert result.image[0, 0] == pytest.approx(5.0)
    assert np.isinf(result.image[0, 1])


def test_cpu_mask_and_invalid_region_propagation() -> None:
    source = np.arange(25, dtype=np.float64).reshape(5, 5)
    source_mask = np.zeros((5, 5), dtype=bool)
    source_mask[1, 1] = True
    spec = ReprojectionSpec(
        mapping=lambda coords: np.asarray(coords, dtype=np.float64),
        output_bbox=BBox(min_x=0, min_y=0, width=7, height=7),
        interpolation="bilinear",
        mapping_grid_step=1,
    )

    result = reproject_array(
        source, spec, source_mask=source_mask, backend="cpu"
    )

    assert np.isnan(result.image[0:5, 5:]).all()
    assert np.isnan(result.image[5:, :]).all()
    assert result.mask is not None
    assert bool(result.mask[1, 1]) is True
    assert bool(result.mask[6, 6]) is True
