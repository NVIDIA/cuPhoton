# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np
import pytest

from cuphoton.xray.detector_artifacts import (
    _detector_artifact_config_hash,
    _publish_detector_artifact_outputs,
    _write_amp_sum_filtered,
    compare_detector_artifacts,
    detector_artifact_resume_identity,
)
from cuphoton.xray.detector_storage import (
    SPECTRAL_LAYOUT_FILE,
    TileRowArray,
    create_detector_spectra,
    load_detector_array,
)
from cuphoton.xray.phonon_viz import _detector_source
from cuphoton.xray.workflow_viz import _detector_filtered_phonon_data


def _write_pair(tmp_path):
    dense, compact = tmp_path / "dense", tmp_path / "compact"
    dense.mkdir()
    compact.mkdir()
    shape = (3, 7, 5)
    spectra = create_detector_spectra(
        compact, shape, x_edges=np.asarray([0, 3, 7])
    )
    for index, (name, array) in enumerate(spectra.items()):
        array[:] = np.arange(30).reshape(3, 2, 5) + index
        array[0, 0] = np.array([0.0, -0.0, np.nan, np.inf, -np.inf])
        array.flush()
        logical = np.repeat(array, [3, 4], axis=1)
        np.save(dense / f"{name}.npy", logical)
    for root in (dense, compact):
        _write_amp_sum_filtered(
            root / "amp_all_sum_filtered.npy",
            load_detector_array(root / "amp_all.npy"),
            amp_threshold=1.6,
        ).flush()
        np.save(root / "fit_status.npy", np.ones(shape[:2], dtype=np.uint8))
        (root / "manifest.json").write_text(json.dumps({"roi_lower": [0, 0]}))
    return dense, compact


def test_tile_row_slices_preserve_bytes_and_dense_consumers(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "cuphoton.xray.phonon_viz._detector_phonon_figure",
        lambda **kwargs: None,
    )
    dense, compact = _write_pair(tmp_path)
    array = load_detector_array(compact / "amp_all.npy")
    reference = load_detector_array(dense / "amp_all.npy")
    assert isinstance(array, TileRowArray)
    assert len(array) == reference.shape[0]
    assert isinstance(array.values, np.memmap)
    assert array.values.shape == (3, 2, 5)
    for key in (
        slice(None),
        (slice(1, 3), slice(2, 6), slice(1, 4)),
        (slice(None), 3, slice(None)),
        (1, slice(None, None, -1), 2),
        (1, -1, -1),
        (Ellipsis, 2),
        (slice(None, None, -1), slice(None, None, -2), slice(None)),
    ):
        assert (
            np.asarray(array[key]).tobytes()
            == np.asarray(reference[key]).tobytes()
        )
    with pytest.raises(IndexError):
        array[:, 7, :]
    comparison = compare_detector_artifacts(
        reference_dir=dense, candidate_dir=compact
    )
    for stats in comparison["arrays"].values():
        assert stats["max_abs_diff"] == 0
        assert stats["nonfinite_value_mismatch_count"] == 0
    for root in (dense, compact):
        assert (
            _detector_source(
                root,
                x_value=3,
                y_start=0,
                y_end=3,
                max_points=100,
                amp_threshold=1.6,
            )["row_count"]
            == 3
        )
    left = _detector_filtered_phonon_data(
        dense,
        x_value=3,
        amp_threshold=1.6,
        max_points=100,
    )
    right = _detector_filtered_phonon_data(
        compact,
        x_value=3,
        amp_threshold=1.6,
        max_points=100,
    )
    assert left.keys() == right.keys()
    for key in left:
        np.testing.assert_array_equal(left[key], right[key])


@pytest.mark.parametrize(
    "operation", [np.asarray, np.nanmax, np.count_nonzero]
)
def test_compact_reader_requires_explicit_slice_for_numpy(
    tmp_path, operation
):
    _dense, compact = _write_pair(tmp_path)
    array = load_detector_array(compact / "amp_all.npy")
    with pytest.raises(TypeError, match="bounded slice"):
        operation(array)


@pytest.mark.parametrize(
    "key",
    [True, (slice(None), True, slice(None)), (Ellipsis, np.bool_(False))],
)
def test_compact_reader_rejects_boolean_indexing(tmp_path, key):
    _dense, compact = _write_pair(tmp_path)
    array = load_detector_array(compact / "amp_all.npy")
    with pytest.raises(IndexError, match="not boolean"):
        array[key]


@pytest.mark.parametrize(
    "edges", [[0, 3], [0, 4, 3, 7], [False, True], [[0, 3, 7]]]
)
def test_compact_writer_rejects_invalid_edges_before_writing(tmp_path, edges):
    with pytest.raises(ValueError, match="shape or x edges"):
        create_detector_spectra(
            tmp_path, (3, 7, 5), x_edges=np.asarray(edges)
        )
    assert not list(tmp_path.iterdir())


def test_publish_missing_pixel_map_reports_its_dense_filename(tmp_path):
    _dense, compact = _write_pair(tmp_path)
    missing = compact / "amp_all_sum_filtered.npy"
    missing.unlink()
    with pytest.raises(FileNotFoundError) as error:
        _publish_detector_artifact_outputs(compact, tmp_path / "published")
    assert error.value.filename == str(missing)


def test_republishing_clears_old_storage_layout(tmp_path):
    dense, compact = _write_pair(tmp_path)
    target = tmp_path / "published"
    _publish_detector_artifact_outputs(compact, target)
    assert (target / SPECTRAL_LAYOUT_FILE).exists()
    assert isinstance(
        load_detector_array(target / "amp_all.npy"), TileRowArray
    )
    _publish_detector_artifact_outputs(dense, target)
    assert not (target / SPECTRAL_LAYOUT_FILE).exists()
    assert not list(target.glob("*.tile-rows.npy"))
    assert isinstance(load_detector_array(target / "amp_all.npy"), np.memmap)


def test_compact_layout_identity_and_metadata_validation(tmp_path):
    _dense, compact = _write_pair(tmp_path)
    for identity in (
        _detector_artifact_config_hash,
        detector_artifact_resume_identity,
    ):
        assert identity({}) == identity({"artifact_layout": "dense"})
        assert identity({}) != identity({"artifact_layout": "tile-rows"})
    path = compact / SPECTRAL_LAYOUT_FILE
    payload = json.loads(path.read_text())
    payload["x_edges"] = [0, 4, 3, 7]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="x edges"):
        load_detector_array(compact / "amp_all.npy")
