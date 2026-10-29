# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cuphoton.ximage.data import bundle as bundle_data
from cuphoton.ximage.data.bundle import (
    XImageBundle,
    detect_ximage_store,
    load_ximage_bundle,
    normalize_exposure_indices,
)


def _bundle() -> XImageBundle:
    arrays = {
        "images": np.zeros((1, 8, 8, 3), dtype=np.float64),
        "masks": np.ones((1, 8, 8, 3), dtype=bool),
        "sky": np.zeros((1, 8, 8, 3), dtype=np.uint8),
        "variances": np.zeros((1, 8, 8, 3), dtype=np.float64),
        "psfs": np.zeros((5, 5, 1, 3), dtype=np.float64),
        "mcgauss": np.zeros((3, 3, 3), dtype=np.float64),
        "exp_times": np.array([270.0, 300.0, 300.0]),
    }
    return XImageBundle(
        store_format="npy",
        store_path=Path("."),
        arrays=arrays,
        metadata={},
    )


def test_loader_preserves_solver_artifact_order(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "ximage_npy"
    store_path.mkdir()
    (store_path / "metadata.json").write_text("{}")
    for key in bundle_data.ARTIFACTS:
        np.save(store_path / f"{key}.npy", np.zeros(1))

    neutral = load_ximage_bundle(store_path)

    assert list(neutral) == [
        "images",
        "masks",
        "sky",
        "variances",
        "psfs",
        "exp_times",
        "mcgauss",
    ]


def test_load_ximage_bundle_supports_artifact_selection(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "ximage_npy"
    store_path.mkdir()
    (store_path / "metadata.json").write_text("{}")
    np.save(store_path / "images.npy", np.zeros((1, 2, 2, 1)))

    bundle = load_ximage_bundle(store_path, artifacts=("images",))

    assert type(bundle) is XImageBundle
    assert list(bundle) == ["images"]


def test_select_exposures_subsets_all_exposure_axes() -> None:
    bundle = _bundle()
    subset = bundle.select_exposures([1])
    assert subset.images.shape == (1, 8, 8, 1)
    assert subset.masks.shape == (1, 8, 8, 1)
    assert subset.psfs.shape == (5, 5, 1, 1)
    assert subset.mcgauss is not None
    assert subset.mcgauss.shape == (1, 3, 3)
    assert subset.exp_times.shape == (1,)
    assert subset.metadata["selected_exposures"] == [1]


def test_reconstruction_inputs_use_documented_solver_order() -> None:
    bundle = _bundle()
    expected = (
        bundle.images,
        bundle.psfs,
        bundle.masks,
        bundle.sky,
        bundle.variances,
        bundle.exp_times,
        bundle.mcgauss,
    )
    assert all(
        actual is expected_item
        for actual, expected_item in zip(
            bundle.reconstruction_inputs(),
            expected,
            strict=True,
        )
    )


def test_bundle_context_closes_owned_resources() -> None:
    class Resource:
        closed = False

        def close(self) -> None:
            self.closed = True

    resource = Resource()
    bundle = _bundle()
    bundle.resources = (resource,)

    with bundle as entered:
        assert entered is bundle

    assert resource.closed is True


def test_reconstruction_input_loader_materializes_and_closes_proxies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle()
    bundle.store_format = "hdf5"
    closed = False

    def close() -> None:
        nonlocal closed
        closed = True

    monkeypatch.setattr(bundle, "close", close)
    monkeypatch.setattr(
        bundle_data,
        "load_ximage_bundle",
        lambda *args, **kwargs: bundle,
    )

    arrays = bundle_data.load_reconstruction_inputs()

    assert all(isinstance(array, np.ndarray) for array in arrays)
    assert closed is True


def test_normalize_exposure_indices_validates_inputs() -> None:
    assert normalize_exposure_indices(
        [0, 2], max_exposures=None, total_exposures=3
    ) == [0, 2]
    assert normalize_exposure_indices(
        None, max_exposures=2, total_exposures=3
    ) == [0, 1]
    with pytest.raises(ValueError):
        normalize_exposure_indices([], max_exposures=None, total_exposures=3)
    with pytest.raises(ValueError):
        normalize_exposure_indices([5], max_exposures=None, total_exposures=3)
    with pytest.raises(ValueError):
        normalize_exposure_indices([0], max_exposures=1, total_exposures=3)


def test_pickle_bundle_is_not_automatically_loaded(tmp_path: Path) -> None:
    (tmp_path / "images.pkl").write_bytes(b"not loaded")

    with pytest.raises(FileNotFoundError, match="Expected an NPY directory"):
        detect_ximage_store(tmp_path)


def test_hdf5_key_resolution_failure_closes_handle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h5py = pytest.importorskip("h5py")
    store_path = tmp_path / "partial.h5"
    with h5py.File(store_path, "w") as handle:
        handle.create_dataset("images", data=np.zeros((1, 2, 2, 1)))

    opened = []
    real_file = h5py.File

    def tracking_file(*args, **kwargs):
        handle = real_file(*args, **kwargs)
        opened.append(handle)
        return handle

    monkeypatch.setattr(h5py, "File", tracking_file)

    with pytest.raises(KeyError):
        load_ximage_bundle(
            store_path,
            prefer=("hdf5",),
            artifacts=("images", "psfs"),
        )

    assert len(opened) == 1
    assert not opened[0].id.valid


@pytest.mark.parametrize("store_format", ["npy", "hdf5", "zarr"])
def test_numeric_store_round_trip(tmp_path: Path, store_format: str) -> None:
    images = np.arange(48, dtype=np.float32).reshape(1, 4, 6, 2)
    path = tmp_path / "images"
    if store_format == "npy":
        path.mkdir()
        (path / "metadata.json").write_text("{}")
        np.save(path / "images.npy", images, allow_pickle=False)
    elif store_format == "hdf5":
        h5py = pytest.importorskip("h5py")
        path = path.with_suffix(".h5")
        with h5py.File(path, "w") as store:
            store.create_dataset("images", data=images)
    else:
        zarr = pytest.importorskip("zarr")
        path = path.with_suffix(".zarr")
        store = zarr.open_group(str(path), mode="w")
        store.create_array("images", data=images)
    with load_ximage_bundle(path, artifacts=["images"]) as bundle:
        assert bundle.store_format == store_format
        np.testing.assert_array_equal(bundle.images[:], images)


@pytest.mark.parametrize(
    "masks,issue",
    [
        (np.zeros((1, 8, 8, 12, 3), dtype=bool), "rank-4"),
        (np.ones((1, 8, 8, 3), dtype=np.uint8), "boolean dtype"),
        (np.ones((1, 7, 8, 3), dtype=bool), "does not match"),
        (np.ones((1, 8, 8, 2), dtype=bool), "does not match"),
    ],
)
def test_validation_rejects_invalid_good_pixel_masks(masks, issue) -> None:
    bundle = _bundle()
    bundle.arrays["masks"] = masks
    assert any(
        issue in found for found in bundle_data.validate_bundle(bundle)
    )


@pytest.mark.parametrize("exposure_time", [0.0, -1.0, np.nan, np.inf])
def test_validation_rejects_invalid_exposure_times(
    exposure_time: float,
) -> None:
    bundle = _bundle()
    bundle.arrays["exp_times"] = np.array([1.0, exposure_time, 2.0])
    assert any(
        "Exposure times must be finite and strictly positive" in issue
        for issue in bundle_data.validate_bundle(bundle)
    )


@pytest.mark.parametrize("kernel_shape", [(4, 4), (3, 5), (0, 0)])
def test_validation_rejects_invalid_psf_geometry(kernel_shape) -> None:
    bundle = _bundle()
    bundle.arrays["psfs"] = np.zeros((*kernel_shape, 1, 3))
    assert any(
        "square, odd spatial dimensions" in issue
        for issue in bundle_data.validate_bundle(bundle)
    )


def test_bundle_validation_preserves_generic_batch_and_channel_axes() -> None:
    bundle = _bundle()
    for key in ("images", "masks", "sky", "variances"):
        bundle.arrays[key] = np.repeat(bundle.arrays[key], 2, axis=0)
    bundle.arrays["psfs"] = np.repeat(bundle.psfs, 2, axis=2)
    assert bundle_data.validate_bundle(bundle) == []


@pytest.mark.parametrize("store_format", ["npy", "hdf5", "zarr"])
def test_minimal_bundle_loads_with_optional_defaults(tmp_path, store_format):
    arrays = {
        "images": np.arange(48, dtype=np.float32).reshape(1, 4, 6, 2),
        "variances": np.ones((1, 4, 6, 2), dtype=np.float32),
        "psfs": np.ones((3, 3, 1, 2), dtype=np.float32) / 9,
    }
    path = tmp_path / "minimal"
    if store_format == "npy":
        path.mkdir()
        for key, values in arrays.items():
            np.save(path / f"{key}.npy", values)
    elif store_format == "hdf5":
        h5py = pytest.importorskip("h5py")
        path = path.with_suffix(".h5")
        with h5py.File(path, "w") as store:
            for key, values in arrays.items():
                store.create_dataset(key, data=values)
    else:
        zarr = pytest.importorskip("zarr")
        path = path.with_suffix(".zarr")
        store = zarr.open_group(str(path), mode="w")
        for key, values in arrays.items():
            store.create_array(key, data=values)

    with load_ximage_bundle(path) as bundle:
        assert bundle_data.validate_bundle(bundle) == []
        assert set(bundle) == set(arrays)
        assert bundle.sky is None
        assert bundle.mcgauss is None
        rois = bundle_data.centered_bundle_rois(bundle, height=2, width=2)
        assert set(rois) == {"images", "variances"}
        np.testing.assert_array_equal(
            rois["images"], arrays["images"][:, 1:3, 2:4]
        )
        assert bundle.masks.dtype == np.dtype(bool)
        np.testing.assert_array_equal(
            bundle.masks, np.ones_like(arrays["images"])
        )
        np.testing.assert_array_equal(bundle.exp_times, [1.0, 1.0])
        assert bundle.axes("images") == ("batch", "y", "x", "frame")
        assert bundle.axes("masks") == bundle.axes("images")
        subset = bundle.select_exposures([1])
        assert subset.masks.shape == (1, 4, 6, 1)
        np.testing.assert_array_equal(subset.exp_times, [1.0])

    inputs = bundle_data.load_reconstruction_inputs(path)
    assert inputs[3] is None
    assert inputs[6] is None
    np.testing.assert_array_equal(inputs[0], arrays["images"])
    np.testing.assert_array_equal(inputs[2], np.ones_like(arrays["images"]))


@pytest.mark.parametrize("key", ["images", "variances", "psfs"])
def test_validation_requires_core_arrays(key):
    bundle = _bundle()
    del bundle.arrays[key]
    assert bundle_data.validate_bundle(bundle) == [
        f"Missing required artifacts: {key}"
    ]
