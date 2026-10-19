# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from cuphoton.xrep import (
    BBox,
    ReprojectionSpec,
    ResamplingWeights,
    prepare_reprojection,
    reproject_array,
    resampling_weights_at,
)
from cuphoton.xrep.interpolation import (
    lanczos_offsets,
    sample_bilinear_array,
    sample_lanczos_array,
)

SAMPLERS = {
    "lanczos3": sample_lanczos_array,
    "bilinear": sample_bilinear_array,
}


def _shift(dx: float, dy: float):
    delta = np.array([dx, dy], dtype=np.float64)
    return lambda coords: np.asarray(coords, dtype=np.float64) + delta


def _rotation_matrix(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    return np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )


def _rotation(degrees: float):
    matrix = _rotation_matrix(degrees)
    return lambda coords: np.asarray(coords, dtype=np.float64) @ matrix.T


def _scale(factor: float):
    return lambda coords: np.asarray(coords, dtype=np.float64) * factor


def _warp(coords):
    coords = np.asarray(coords, dtype=np.float64)
    return coords + 1e-4 * coords**2


def _seam(jump: float, column: float):
    def mapping(coords):
        out = np.asarray(coords, dtype=np.float64).copy()
        out[out[:, 0] >= column, 0] += jump
        return out

    return mapping


def _gather(source: np.ndarray, result: ResamplingWeights) -> np.ndarray:
    base = np.floor(np.nan_to_num(result.source)).astype(np.int64)
    rows = base[:, 1, None] + result.offsets[None, :]
    cols = base[:, 0, None] + result.offsets[None, :]
    rows = np.clip(rows, 0, source.shape[0] - 1)
    cols = np.clip(cols, 0, source.shape[1] - 1)
    window = source[rows[:, :, None], cols[:, None, :]]
    return (result.weights * window).sum(axis=(1, 2))


def _grid_positions(bbox: BBox) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0 : bbox.height, 0 : bbox.width]
    return xx.ravel().astype(np.float64), yy.ravel().astype(np.float64)


@pytest.mark.parametrize("interpolation", ["lanczos3", "bilinear"])
@pytest.mark.parametrize("shift", [(0.0, 0.0), (0.5, 0.5), (3.0, 0.5)])
def test_weights_sum_to_one_and_match_interpolation_module(
    interpolation: str, shift: tuple[float, float]
) -> None:
    rng = np.random.default_rng(3)
    source = rng.normal(size=(16, 16))
    x = np.array([5.0, 6.0, 7.4, 8.6, 9.0])
    y = np.array([4.0, 9.0, 5.6, 6.4, 7.0])

    result = resampling_weights_at(
        _shift(*shift), x, y, interpolation=interpolation
    )

    size = 6 if interpolation == "lanczos3" else 2
    assert result.weights.shape == (5, size, size)
    assert result.jacobian.shape == (5, 2, 2)
    assert result.interpolation == interpolation
    assert np.allclose(result.weights.sum(axis=(1, 2)), 1.0)
    assert np.allclose(result.phase, np.mod(shift, 1.0))
    assert result.is_translation.all()
    assert np.allclose(result.jacobian, np.eye(2))
    destination = np.column_stack([np.rint(x), np.rint(y)])
    assert np.array_equal(result.destination, destination)
    expected = SAMPLERS[interpolation](
        source, destination[:, 0] + shift[0], destination[:, 1] + shift[1]
    )
    assert np.allclose(_gather(source, result), expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("interpolation", ["lanczos3", "bilinear"])
def test_source_shape_renormalizes_at_edges_like_the_sampler(
    interpolation: str,
) -> None:
    rng = np.random.default_rng(4)
    source = rng.normal(size=(10, 12))
    x = np.array([0.0, 1.0, 10.0, 6.0, -8.0])
    y = np.array([0.0, 8.0, 8.0, 0.0, 5.0])

    result = resampling_weights_at(
        _shift(0.5, 0.25),
        x,
        y,
        interpolation=interpolation,
        source_shape=source.shape,
    )
    expected = SAMPLERS[interpolation](source, x + 0.5, y + 0.25)

    total = result.weights.sum(axis=(1, 2))
    assert np.allclose(total[:4], 1.0)
    assert total[4] == 0.0
    assert np.isnan(expected[4])
    assert np.allclose(_gather(source, result)[:4], expected[:4], atol=1e-12)


@pytest.mark.parametrize("interpolation", ["lanczos3", "bilinear"])
@pytest.mark.parametrize("form", ["callable", "prepared", "spec_prepared"])
def test_matches_reproject_array_including_the_edge_strip(
    interpolation: str, form: str
) -> None:
    rng = np.random.default_rng(5)
    source = rng.normal(size=(10, 12))
    spec = ReprojectionSpec(
        mapping=_shift(-0.5, 0.0),
        output_bbox=BBox(min_x=0, min_y=0, width=13, height=10),
        interpolation=interpolation,
        mapping_grid_step=1,
        area_scaling=False,
    )
    image = reproject_array(source, spec, backend="cpu").image
    x, y = _grid_positions(spec.output_bbox)
    prepared = prepare_reprojection(spec, source_shape=source.shape)
    if form == "callable":
        result = resampling_weights_at(
            spec.mapping,
            x,
            y,
            interpolation=interpolation,
            source_shape=source.shape,
        )
    elif form == "prepared":
        result = resampling_weights_at(
            prepared,
            x,
            y,
            interpolation=interpolation,
            source_shape=source.shape,
        )
    else:
        result = resampling_weights_at(
            spec, x, y, prepared=prepared, source_shape=source.shape
        )

    filled = np.isnan(image.ravel())
    assert filled.sum() == 2 * source.shape[0]
    total = result.weights.sum(axis=(1, 2))
    assert np.all(total[filled] == 0.0)
    assert np.allclose(total[~filled], 1.0)
    assert np.allclose(
        _gather(source, result)[~filled], image.ravel()[~filled], atol=1e-12
    )


def test_prepared_mapping_matches_reproject_array_on_a_coarse_grid() -> None:
    rng = np.random.default_rng(6)
    source = rng.normal(size=(300, 300))
    spec = ReprojectionSpec(
        mapping=_warp,
        output_bbox=BBox(min_x=0, min_y=0, width=256, height=256),
        interpolation="lanczos3",
        area_scaling=False,
    )
    assert spec.mapping_grid_step == 100
    image = reproject_array(source, spec, backend="cpu").image
    prepared = prepare_reprojection(spec, source_shape=source.shape)
    x = np.array([50.0, 120.0, 199.0, 250.0])
    y = np.array([50.0, 77.0, 150.0, 3.0])

    from_prepared = resampling_weights_at(
        spec, x, y, prepared=prepared, source_shape=source.shape
    )
    from_spec = resampling_weights_at(spec, x, y, source_shape=source.shape)

    expected = image[y.astype(int), x.astype(int)]
    assert np.allclose(_gather(source, from_prepared), expected, atol=1e-12)
    assert not np.allclose(_gather(source, from_spec), expected, atol=1e-6)
    assert np.abs(from_prepared.phase - from_spec.phase).max() > 0.1


@pytest.mark.parametrize(
    ("mapping", "expected"),
    [
        (_shift(0.3, -2.7), True),
        (_rotation(10.0), False),
        (_scale(1.2), False),
    ],
)
def test_is_translation_flags_rotation_and_scale(mapping, expected) -> None:
    result = resampling_weights_at(
        mapping,
        np.array([20.0, 31.0]),
        np.array([25.0, 8.0]),
        interpolation="lanczos3",
    )
    assert result.is_translation.tolist() == [expected, expected]
    assert np.allclose(result.weights.sum(axis=(1, 2)), 1.0)


def test_jacobian_is_the_rotation_matrix_under_a_rotation() -> None:
    result = resampling_weights_at(
        _rotation(10.0),
        np.array([20.0, 31.0]),
        np.array([25.0, 8.0]),
        interpolation="bilinear",
    )
    expected = np.broadcast_to(_rotation_matrix(10.0), (2, 2, 2))
    assert np.allclose(result.jacobian, expected, atol=1e-12)


@pytest.mark.parametrize("form", ["callable", "prepared"])
def test_seam_between_tiles_is_not_read_as_a_rotation(form: str) -> None:
    mapping = _seam(23.0, 30.0)
    x = np.array([29.0, 30.0, 10.0])
    y = np.array([5.0, 5.0, 5.0])
    if form == "prepared":
        spec = ReprojectionSpec(
            mapping=mapping,
            output_bbox=BBox(min_x=0, min_y=0, width=40, height=10),
            mapping_grid_step=1,
        )
        mapping = prepare_reprojection(spec, source_shape=(20, 80))

    kept = resampling_weights_at(mapping, x, y, interpolation="lanczos3")
    averaged = resampling_weights_at(
        mapping, x, y, interpolation="lanczos3", seam_tolerance=50.0
    )

    assert kept.is_translation.tolist() == [True, True, True]
    assert np.allclose(kept.jacobian, np.eye(2))
    assert averaged.is_translation.tolist() == [False, False, True]
    assert np.allclose(averaged.jacobian[:2, 0, 0], 12.5)


@pytest.mark.parametrize("mapping", [_shift(0.3, -2.7), _rotation(10.0)])
def test_prepared_mapping_matches_callable(mapping) -> None:
    spec = ReprojectionSpec(
        mapping=mapping,
        output_bbox=BBox(min_x=0, min_y=0, width=40, height=40),
        interpolation="lanczos3",
        mapping_grid_step=1,
    )
    prepared = prepare_reprojection(spec, source_shape=(64, 64))
    x = np.array([0.0, 12.0, 20.4, 39.0])
    y = np.array([0.0, 7.0, 25.6, 39.0])

    from_prepared = resampling_weights_at(
        prepared, x, y, interpolation="lanczos3"
    )
    from_callable = resampling_weights_at(
        mapping, x, y, interpolation="lanczos3", source_shape=(64, 64)
    )

    assert np.allclose(from_prepared.source, from_callable.source)
    assert np.allclose(from_prepared.phase, from_callable.phase)
    assert np.allclose(from_prepared.weights, from_callable.weights)
    assert np.allclose(from_prepared.jacobian, from_callable.jacobian)
    assert np.array_equal(
        from_prepared.is_translation, from_callable.is_translation
    )


def test_spec_form_reads_the_kernel_options_from_the_spec() -> None:
    spec = ReprojectionSpec(
        mapping=_shift(10.3, 20.6),
        output_bbox=BBox(min_x=0, min_y=0, width=8, height=8),
        interpolation="lanczos3",
        lanczos_a=2,
        two_a_footprint=False,
        source_origin=(10, 20),
        mapping_grid_step=1,
    )
    x = np.array([2.0, 5.0])
    y = np.array([3.0, 6.0])

    from_spec = resampling_weights_at(spec, x, y)
    explicit = resampling_weights_at(
        spec.mapping,
        x,
        y,
        interpolation="lanczos3",
        lanczos_a=2,
        two_a_footprint=False,
        source_origin=(10, 20),
    )

    assert from_spec.weights.shape == (2, 5, 5)
    assert np.array_equal(from_spec.offsets, lanczos_offsets(2, False))
    assert np.allclose(from_spec.source, [[2.3, 3.6], [5.3, 6.6]])
    assert np.allclose(from_spec.weights, explicit.weights)
    for option in (
        {"interpolation": "lanczos3"},
        {"lanczos_a": 2},
        {"two_a_footprint": False},
        {"source_origin": (10, 20)},
    ):
        with pytest.raises(
            ValueError, match="come from the ReprojectionSpec"
        ):
            resampling_weights_at(spec, x, y, **option)


def test_prepared_mapping_rejects_positions_outside_its_grid() -> None:
    spec = ReprojectionSpec(
        mapping=_shift(0.0, 0.0),
        output_bbox=BBox(min_x=0, min_y=0, width=4, height=4),
        mapping_grid_step=1,
    )
    prepared = prepare_reprojection(spec, source_shape=(8, 8))
    with pytest.raises(ValueError, match="inside the prepared output grid"):
        resampling_weights_at(
            prepared,
            np.array([4.0]),
            np.array([0.0]),
            interpolation="bilinear",
        )


def test_source_origin_shifts_the_source_coordinates() -> None:
    result = resampling_weights_at(
        _shift(10.25, 20.5),
        np.array([2.0]),
        np.array([3.0]),
        interpolation="bilinear",
        source_origin=(10, 20),
    )
    assert np.allclose(result.source, [[2.25, 3.5]])
    assert np.allclose(result.phase, [[0.25, 0.5]])


def test_nonfinite_mapping_gives_zero_weights_and_nan_phase() -> None:
    def mapping(coords):
        out = np.asarray(coords, dtype=np.float64).copy()
        out[out[:, 0] > 5.0] = np.nan
        return out

    result = resampling_weights_at(
        mapping,
        np.array([2.0, 8.0]),
        np.array([2.0, 2.0]),
        interpolation="lanczos3",
    )
    assert result.weights[0].sum() == pytest.approx(1.0)
    assert np.all(result.weights[1] == 0.0)
    assert np.isnan(result.phase[1]).all()
    assert result.is_translation.tolist() == [True, False]


def test_argument_validation() -> None:
    x = np.array([1.0])
    spec = ReprojectionSpec(
        mapping=_shift(0.0, 0.0),
        output_bbox=BBox(min_x=0, min_y=0, width=4, height=4),
    )
    with pytest.raises(ValueError, match="interpolation"):
        resampling_weights_at(_shift(0.0, 0.0), x, x, interpolation="cubic")
    with pytest.raises(ValueError, match="interpolation"):
        resampling_weights_at(_shift(0.0, 0.0), x, x)
    with pytest.raises(ValueError, match="same shape"):
        resampling_weights_at(
            _shift(0.0, 0.0),
            x,
            np.array([1.0, 2.0]),
            interpolation="bilinear",
        )
    with pytest.raises(ValueError, match="finite"):
        resampling_weights_at(
            _shift(0.0, 0.0),
            x,
            np.array([np.nan]),
            interpolation="bilinear",
        )
    with pytest.raises(TypeError, match="callable"):
        resampling_weights_at(object(), x, x, interpolation="bilinear")
    prepared = prepare_reprojection(spec, source_shape=(8, 8))
    with pytest.raises(ValueError, match="only with a ReprojectionSpec"):
        resampling_weights_at(
            spec.mapping, x, x, interpolation="bilinear", prepared=prepared
        )


def test_cupy_prepared_mapping_sets_the_array_namespace() -> None:
    cp = pytest.importorskip("cupy")
    spec = ReprojectionSpec(
        mapping=_shift(0.3, -2.7),
        output_bbox=BBox(min_x=0, min_y=0, width=40, height=40),
        interpolation="lanczos3",
        mapping_grid_step=1,
    )
    x = np.array([0.0, 12.0, 20.4, 39.0])
    y = np.array([0.0, 7.0, 25.6, 39.0])
    host = resampling_weights_at(
        spec, x, y, prepared=prepare_reprojection(spec, source_shape=(64, 64))
    )
    device_prepared = prepare_reprojection(spec, source_shape=(64, 64), xp=cp)

    from_device = resampling_weights_at(spec, x, y, prepared=device_prepared)
    from_device_positions = resampling_weights_at(
        device_prepared,
        cp.asarray(x),
        cp.asarray(y),
        interpolation="lanczos3",
    )
    from_callable = resampling_weights_at(
        spec, cp.asarray(x), cp.asarray(y), source_shape=(64, 64)
    )

    for result in (from_device, from_device_positions, from_callable):
        for name in ("weights", "phase", "is_translation", "jacobian"):
            assert isinstance(getattr(result, name), cp.ndarray)
            assert np.allclose(
                cp.asnumpy(getattr(result, name)),
                getattr(host, name),
                equal_nan=True,
            )
