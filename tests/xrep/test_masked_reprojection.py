# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from cuphoton.xrep import (
    BBox,
    MaskedReprojectionResult,
    ReprojectionSpec,
    prepare_reprojection,
    reproject_array,
    reproject_masked_array,
)


def _offset_spec(
    *,
    offset: tuple[float, float] = (0.0, 0.0),
    interpolation: str = "bilinear",
    area_scaling: bool = True,
    shape: tuple[int, int] = (1, 1),
) -> ReprojectionSpec:
    return ReprojectionSpec(
        mapping=lambda coords: (
            np.asarray(coords, dtype=np.float64)
            + np.asarray(offset, dtype=np.float64)
        ),
        output_bbox=BBox(
            min_x=0,
            min_y=0,
            width=shape[1],
            height=shape[0],
        ),
        interpolation=interpolation,
        mapping_grid_step=1,
        area_scaling=area_scaling,
    )


@pytest.mark.parametrize("interpolation", ["bilinear", "lanczos3"])
def test_masked_identity_preserves_all_planes(interpolation: str) -> None:
    image = np.arange(36, dtype=np.float64).reshape(6, 6)
    variance = np.linspace(1.0, 4.0, 36).reshape(6, 6)
    mask = np.arange(36, dtype=np.uint64).reshape(6, 6)
    spec = _offset_spec(interpolation=interpolation, shape=image.shape)

    result = reproject_masked_array(
        image,
        variance,
        mask,
        spec,
        backend="cpu",
    )

    assert isinstance(result, MaskedReprojectionResult)
    assert np.allclose(result.image, image)
    assert np.allclose(result.variance, variance)
    assert np.array_equal(result.mask, mask)
    assert result.mask.dtype == np.uint64
    assert result.metadata["variance_propagation"] == "diagonal"
    assert result.metadata["variance_covariance_propagated"] is False


def test_bilinear_uses_squared_weights_and_exact_mask_or() -> None:
    image = np.array([[0.0, 2.0], [4.0, 6.0]])
    variance = np.full((2, 2), 4.0)
    mask = np.array([[1, 2], [4, 8]], dtype=np.uint64)
    spec = _offset_spec(offset=(0.5, 0.5))

    result = reproject_masked_array(
        image,
        variance,
        mask,
        spec,
        backend="cpu",
    )

    assert result.image[0, 0] == pytest.approx(3.0)
    assert result.variance[0, 0] == pytest.approx(1.0)
    assert result.mask[0, 0] == 15


def test_bilinear_edge_mask_ors_in_bounds_contributors_and_invalid_bit() -> (
    None
):
    image = np.ones((2, 2), dtype=np.float64)
    variance = np.ones_like(image)
    mask = np.array([[1, 2], [4, 8]], dtype=np.uint16)
    spec = ReprojectionSpec(
        mapping=lambda coords: np.broadcast_to(
            np.array([-0.25, 0.5]),
            np.asarray(coords).shape,
        ),
        output_bbox=BBox(min_x=0, min_y=0, width=1, height=1),
        interpolation="bilinear",
        mapping_grid_step=1,
        area_scaling=False,
        invalid_mask_value=16,
    )

    result = reproject_masked_array(
        image,
        variance,
        mask,
        spec,
        backend="cpu",
    )

    assert result.mask[0, 0] == 1 | 4 | 16


def test_lanczos_mask_or_includes_contributor_outside_bilinear() -> None:
    image = np.ones((9, 9), dtype=np.float64)
    variance = np.ones_like(image)
    mask = np.zeros(image.shape, dtype=np.uint64)
    mask[2, 2] = np.uint64(1) << np.uint64(40)
    offset = np.array([4.25, 4.375], dtype=np.float64)

    def make_spec(interpolation: str) -> ReprojectionSpec:
        return ReprojectionSpec(
            mapping=lambda coords: np.broadcast_to(
                offset,
                np.asarray(coords).shape,
            ),
            output_bbox=BBox(min_x=0, min_y=0, width=1, height=1),
            interpolation=interpolation,
            mapping_grid_step=1,
            area_scaling=False,
        )

    bilinear = reproject_masked_array(
        image,
        variance,
        mask,
        make_spec("bilinear"),
        backend="cpu",
    )
    lanczos = reproject_masked_array(
        image,
        variance,
        mask,
        make_spec("lanczos3"),
        backend="cpu",
    )
    legacy = reproject_array(
        image,
        make_spec("lanczos3"),
        source_mask=mask,
        backend="cpu",
    )

    assert bilinear.mask[0, 0] == 0
    assert legacy.mask[0, 0] == 0
    assert lanczos.mask[0, 0] == np.uint64(1) << np.uint64(40)


@pytest.mark.parametrize("interpolation", ["bilinear", "lanczos3"])
def test_zero_weight_nonfinite_neighbors_do_not_poison_identity(
    interpolation: str,
) -> None:
    image = np.array([[5.0, np.nan], [np.inf, -np.inf]])
    variance = np.array([[2.0, np.inf], [np.inf, np.inf]])
    mask = np.zeros((2, 2), dtype=np.uint16)
    spec = _offset_spec(shape=(1, 1), interpolation=interpolation)

    with np.errstate(invalid="raise"):
        result = reproject_masked_array(
            image,
            variance,
            mask,
            spec,
            backend="cpu",
        )

    assert result.image[0, 0] == pytest.approx(5.0)
    assert result.variance[0, 0] == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("variance_fill_value", "expected"),
    [(float("nan"), float("nan")), (123.0, 123.0)],
)
def test_variance_fill_is_independent_from_image_fill(
    variance_fill_value: float,
    expected: float,
) -> None:
    image = np.ones((2, 2), dtype=np.float64)
    variance = np.ones_like(image)
    mask = np.zeros(image.shape, dtype=np.uint16)
    spec = ReprojectionSpec(
        mapping=lambda coords: np.asarray(coords, dtype=np.float64) - 10.0,
        output_bbox=BBox(min_x=0, min_y=0, width=1, height=1),
        interpolation="bilinear",
        mapping_grid_step=1,
        fill_value=-99.0,
    )

    result = reproject_masked_array(
        image,
        variance,
        mask,
        spec,
        backend="cpu",
        variance_fill_value=variance_fill_value,
    )

    assert result.image[0, 0] == -99.0
    if np.isnan(expected):
        assert np.isnan(result.variance[0, 0])
    else:
        assert result.variance[0, 0] == expected


def test_lanczos_variance_uses_squared_normalized_weights() -> None:
    image = np.ones((9, 9), dtype=np.float64)
    variance = np.full((9, 9), 4.0)
    mask = np.zeros((9, 9), dtype=np.uint16)
    offset = (4.25, 4.375)
    spec = ReprojectionSpec(
        mapping=lambda coords: np.broadcast_to(
            np.asarray(offset, dtype=np.float64),
            np.asarray(coords).shape,
        ),
        output_bbox=BBox(min_x=0, min_y=0, width=1, height=1),
        interpolation="lanczos3",
        mapping_grid_step=1,
        area_scaling=False,
    )

    result = reproject_masked_array(
        image,
        variance,
        mask,
        spec,
        backend="cpu",
    )

    def sincpi(value: np.ndarray) -> np.ndarray:
        return np.sinc(value)

    offsets = np.arange(-2, 4, dtype=np.float64)
    weight_x = sincpi(offsets - 0.25) * sincpi((offsets - 0.25) / 3.0)
    weight_y = sincpi(offsets - 0.375) * sincpi((offsets - 0.375) / 3.0)
    weights = np.outer(weight_y, weight_x)
    normalized = weights / weights.sum()
    expected_variance = 4.0 * np.square(normalized).sum()

    assert result.image[0, 0] == pytest.approx(1.0)
    assert result.variance[0, 0] == pytest.approx(
        expected_variance,
        rel=1e-12,
    )
    assert result.variance[0, 0] < 4.0


def test_relative_area_is_squared_for_variance() -> None:
    image = np.ones((3, 3), dtype=np.float64)
    variance = np.full((3, 3), 2.0)
    mask = np.zeros((3, 3), dtype=np.uint16)
    spec = ReprojectionSpec(
        mapping=lambda coords: 2.0 * np.asarray(coords, dtype=np.float64),
        output_bbox=BBox(min_x=0, min_y=0, width=1, height=1),
        interpolation="bilinear",
        mapping_grid_step=1,
        area_scaling=True,
    )

    result = reproject_masked_array(
        image,
        variance,
        mask,
        spec,
        backend="cpu",
    )

    assert result.image[0, 0] == pytest.approx(4.0)
    assert result.variance[0, 0] == pytest.approx(32.0)
    assert result.metadata["variance_area_scaling_power"] == 2


def test_masked_reproject_reuses_supplied_prepared_geometry() -> None:
    enabled = True

    def mapping(coords):
        if not enabled:
            raise AssertionError("mapping was unexpectedly reevaluated")
        return np.asarray(coords, dtype=np.float64)

    image = np.ones((3, 4), dtype=np.float64)
    variance = np.ones_like(image)
    mask = np.zeros(image.shape, dtype=np.uint16)
    spec = ReprojectionSpec(
        mapping=mapping,
        output_bbox=BBox(min_x=0, min_y=0, width=4, height=3),
        interpolation="bilinear",
        mapping_grid_step=1,
    )
    prepared = prepare_reprojection(spec, source_shape=image.shape)
    enabled = False

    result = reproject_masked_array(
        image,
        variance,
        mask,
        spec,
        backend="cpu",
        prepared=prepared,
    )

    assert np.allclose(result.image, image)


def test_masked_reproject_validates_plane_shapes_and_mask_dtype() -> None:
    image = np.ones((2, 2), dtype=np.float64)
    spec = _offset_spec(shape=(2, 2))

    with pytest.raises(ValueError, match="variance shape"):
        reproject_masked_array(
            image,
            np.ones((1, 2)),
            np.zeros((2, 2), dtype=np.uint16),
            spec,
            backend="cpu",
        )
    with pytest.raises(ValueError, match="mask shape"):
        reproject_masked_array(
            image,
            np.ones((2, 2)),
            np.zeros((1, 2), dtype=np.uint16),
            spec,
            backend="cpu",
        )
    with pytest.raises(TypeError, match="integer or boolean"):
        reproject_masked_array(
            image,
            np.ones((2, 2)),
            np.zeros((2, 2), dtype=np.float64),
            spec,
            backend="cpu",
        )


def test_masked_reproject_rejects_negative_variance_but_allows_no_data() -> (
    None
):
    image = np.ones((2, 2), dtype=np.float64)
    mask = np.zeros(image.shape, dtype=np.uint16)
    spec = _offset_spec(shape=(1, 1))

    with pytest.raises(ValueError, match="negative"):
        reproject_masked_array(
            image,
            np.array([[1.0, -0.5], [np.nan, np.inf]]),
            mask,
            spec,
            backend="cpu",
        )

    result = reproject_masked_array(
        image,
        np.array([[1.0, np.nan], [np.nan, np.inf]]),
        mask,
        spec,
        backend="cpu",
    )
    assert result.variance[0, 0] == pytest.approx(1.0)
