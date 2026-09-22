# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq
import pytest

from cuphoton.core.artifacts import file_sha256
from cuphoton.core.cli import run_component
from cuphoton.xfit.io import load_xfit_dataset
from cuphoton.xfit.models import GaussianDipoleModel
from cuphoton.xscan import xfit_features
from cuphoton.xscan.xfit_features import (
    FEATURE_NAMES,
    build_xfit_feature_bundle,
    export_xfit_input,
    load_xfit_feature_matrix,
)


def _dataset(tmp_path, *, images=None, candidate_ids=None):
    root = tmp_path / "dataset"
    root.mkdir()
    if images is None:
        images = np.arange(3 * 5 * 7, dtype=np.float32).reshape(3, 5, 7)
        images[2] = images[0]
    if candidate_ids is None:
        candidate_ids = ["b", "a", "b"]
    np.save(root / "difference.npy", images, allow_pickle=False)
    rows = [
        {"candidate_id": value, "split": "train", "split_group": value}
        for value in candidate_ids
    ]
    (root / "metadata.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    return root, images


def _planes(tmp_path, images):
    variance = np.ones(images.shape, dtype=np.float64)
    variance[1] = 4
    mask = np.ones(images.shape, dtype=np.uint16)
    mask[1, 0] = 0
    mask[:, 2, 2] = 4
    variance_path = tmp_path / "variance.npy"
    mask_path = tmp_path / "mask.npy"
    np.save(variance_path, variance)
    np.save(mask_path, mask)
    return variance_path, mask_path, variance, mask


@pytest.mark.parametrize("planes", ["variance", "mask", "both"])
def test_export_preserves_auxiliary_rows_and_dtypes(tmp_path, planes):
    root, images = _dataset(tmp_path)
    variance_path, mask_path, variance, mask = _planes(tmp_path, images)
    options = {}
    if planes in {"variance", "both"}:
        options["variance_path"] = variance_path
    if planes in {"mask", "both"}:
        options["mask_path"] = mask_path
    output = tmp_path / "input.npz"
    summary = export_xfit_input(
        dataset_dir=root, output_path=output, image_unit="electron", **options
    )
    dataset = load_xfit_dataset(output, mode="difference", model="gaussian")
    np.testing.assert_array_equal(dataset.images, images[:2])
    assert dataset.candidate_id.tolist() == ["b", "a"]
    assert summary["image_unit"] == "electron"
    assert summary["source_arrays"]["images"]["sha256"] == file_sha256(
        root / "difference.npy"
    )
    for name, expected, path in (
        ("variance", variance, variance_path),
        ("mask", mask, mask_path),
    ):
        actual = getattr(dataset, name)
        assert summary[f"{name}_present"] == (actual is not None)
        if f"{name}_path" not in options:
            assert actual is None and name not in summary["source_arrays"]
            continue
        np.testing.assert_array_equal(actual, expected[:2])
        assert actual.dtype == expected.dtype
        assert summary["source_arrays"][name] == {
            "name": path.name,
            "sha256": file_sha256(path),
        }
    assert summary["variance_unit"] == (
        "(electron)^2" if "variance_path" in options else None
    )


@pytest.mark.parametrize("source", ["images", "variance", "mask"])
def test_export_rejects_sources_changed_during_archive_copy(
    tmp_path, monkeypatch, source
):
    root, images = _dataset(tmp_path)
    variance_path, mask_path, _, _ = _planes(tmp_path, images)
    source_path = {
        "images": root / "difference.npy",
        "variance": variance_path,
        "mask": mask_path,
    }[source]
    source_hash = file_sha256(source_path)
    write_member = xfit_features._write_indexed_image_member

    def mutate_before_copy(
        archive, values, row_indices, *, name="images.npy"
    ):
        if name == f"{source}.npy":
            changed = np.load(source_path, mmap_mode="r+", allow_pickle=False)
            changed[1, 1, 1] += 1
            changed.flush()
            del changed
            assert file_sha256(source_path) != source_hash
        write_member(archive, values, row_indices, name=name)

    monkeypatch.setattr(
        xfit_features, "_write_indexed_image_member", mutate_before_copy
    )
    output = tmp_path / "input.npz"
    with pytest.raises(ValueError, match="source changed during export"):
        export_xfit_input(
            dataset_dir=root,
            output_path=output,
            variance_path=variance_path,
            mask_path=mask_path,
        )
    assert not output.exists()


@pytest.mark.parametrize("plane", ["variance", "mask"])
def test_export_rejects_conflicting_duplicate_planes(tmp_path, plane):
    root, images = _dataset(tmp_path)
    values = np.ones_like(images)
    values[2, 0, 0] = 2
    path = tmp_path / f"{plane}.npy"
    np.save(path, values)
    output = tmp_path / "input.npz"
    with pytest.raises(ValueError, match=f"conflicting {plane}"):
        export_xfit_input(
            dataset_dir=root, output_path=output, **{f"{plane}_path": path}
        )
    assert not output.exists()


@pytest.mark.parametrize("shape", [(), (5, 7), (2, 5, 7), (3, 7, 5)])
@pytest.mark.parametrize("plane", ["variance", "mask"])
def test_export_rejects_unaligned_planes(tmp_path, shape, plane):
    root, _ = _dataset(tmp_path)
    path = tmp_path / f"{plane}.npy"
    np.save(path, np.ones(shape))
    with pytest.raises(ValueError, match="same shape"):
        export_xfit_input(
            dataset_dir=root,
            output_path=tmp_path / "input.npz",
            **{f"{plane}_path": path},
        )


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, 0, -1])
def test_export_rejects_invalid_included_variance(tmp_path, value):
    root, images = _dataset(tmp_path)
    path, mask_path, variance, _ = _planes(tmp_path, images)
    variance[1, 1, 1] = value
    np.save(path, variance)
    with pytest.raises(ValueError, match="positive at included pixels"):
        export_xfit_input(
            dataset_dir=root,
            output_path=tmp_path / "input.npz",
            variance_path=path,
            mask_path=mask_path,
        )


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_export_rejects_nonfinite_mask(tmp_path, value):
    root, images = _dataset(tmp_path)
    mask = np.ones_like(images)
    mask[1, 0, 0] = value
    path = tmp_path / "mask.npy"
    np.save(path, mask)
    with pytest.raises(ValueError, match="mask must contain only finite"):
        export_xfit_input(
            dataset_dir=root,
            output_path=tmp_path / "input.npz",
            mask_path=path,
        )


@pytest.mark.parametrize("dtype", [complex, object, "U2"])
@pytest.mark.parametrize("plane", ["variance", "mask"])
def test_export_rejects_nonreal_auxiliary_arrays(tmp_path, dtype, plane):
    root, images = _dataset(tmp_path)
    path = tmp_path / f"{plane}.npy"
    np.save(path, np.ones(images.shape, dtype=dtype))
    with pytest.raises(ValueError, match="numeric array"):
        export_xfit_input(
            dataset_dir=root,
            output_path=tmp_path / "input.npz",
            **{f"{plane}_path": path},
        )


def test_export_accepts_masked_nonfinite_images_and_variance(tmp_path):
    root, images = _dataset(tmp_path)
    variance_path, mask_path, variance, mask = _planes(tmp_path, images)
    images[1, 0] = [np.nan, np.inf, -np.inf, np.nan, np.nan, np.nan, np.nan]
    variance[1, 0] = [np.nan, 0, -1, np.inf, -np.inf, 0, 0]
    np.save(root / "difference.npy", images)
    np.save(variance_path, variance)
    output = tmp_path / "input.npz"
    export_xfit_input(
        dataset_dir=root,
        output_path=output,
        variance_path=variance_path,
        mask_path=mask_path,
    )
    dataset = load_xfit_dataset(output, mode="difference", model="gaussian")
    np.testing.assert_array_equal(dataset.images, images[:2])
    np.testing.assert_array_equal(dataset.variance, variance[:2])
    np.testing.assert_array_equal(dataset.mask, mask[:2])


def test_export_bounds_auxiliary_reads_and_cleans_partial_archive(
    tmp_path, monkeypatch
):
    root, images = _dataset(tmp_path)
    variance_path, mask_path, _, _ = _planes(tmp_path, images)
    output = tmp_path / "input.npz"
    monkeypatch.setattr(xfit_features, "_EXPORT_BATCH_BYTES", 1)
    getitem = np.memmap.__getitem__
    reads = []

    def checked_getitem(self, index):
        if isinstance(index, list):
            assert len(index) == 1
            reads.append((self.filename, index))
        return getitem(self, index)

    monkeypatch.setattr(np.memmap, "__getitem__", checked_getitem)
    export_xfit_input(
        dataset_dir=root,
        output_path=output,
        variance_path=variance_path,
        mask_path=mask_path,
    )
    assert len(reads) == 6
    original = xfit_features._write_indexed_image_member

    def fail_on_mask(*args, **kwargs):
        if kwargs.get("name") == "mask.npy":
            raise OSError("test write failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(
        xfit_features, "_write_indexed_image_member", fail_on_mask
    )
    failed = tmp_path / "failed.npz"
    with pytest.raises(OSError, match="test write failure"):
        export_xfit_input(
            dataset_dir=root,
            output_path=failed,
            variance_path=variance_path,
            mask_path=mask_path,
        )
    assert not failed.exists()


def test_export_cli_matches_manual_weighted_fit_and_features(
    tmp_path, capsys
):
    shape = (15, 17)
    parameters = np.asarray(
        [[12, 1.2, 1.6, 0.2, 2, 1, -2, -1]], dtype=np.float64
    )
    model = GaussianDipoleModel(shape)
    images = model.evaluate(parameters, mode="difference")
    images += np.random.default_rng(91).normal(scale=0.03, size=images.shape)
    root, images = _dataset(tmp_path, images=images, candidate_ids=["source"])
    variance = np.linspace(0.0005, 0.005, images.size).reshape(images.shape)
    mask = np.ones_like(images, dtype=bool)
    mask[:, 0] = False
    images[:, 0] = np.nan
    variance[:, 0] = np.nan
    np.save(root / "difference.npy", images)
    variance_path = tmp_path / "variance.npy"
    mask_path = tmp_path / "mask.npy"
    np.save(variance_path, variance)
    np.save(mask_path, mask)
    exported = tmp_path / "exported.npz"
    assert (
        run_component(
            "xscan",
            [
                "data-export-xfit-input",
                "--dataset-dir",
                str(root),
                "--output",
                str(exported),
                "--variance",
                str(variance_path),
                "--mask",
                str(mask_path),
                "--image-unit",
                "electron",
            ],
        )
        == 0
    )
    export_summary = json.loads(capsys.readouterr().out)
    assert export_summary["variance_unit"] == "(electron)^2"
    manual = tmp_path / "manual.npz"
    np.savez_compressed(
        manual,
        candidate_id=np.asarray(["source"]),
        images=images,
        variance=variance,
        mask=mask,
    )
    for label, path in (("exported", exported), ("manual", manual)):
        assert (
            run_component(
                "xfit",
                [
                    "fit-dipoles",
                    "--input",
                    str(path),
                    "--output-dir",
                    str(tmp_path / label),
                    "--model",
                    "gaussian",
                    "--mode",
                    "difference",
                    "--backend",
                    "numpy",
                    "--compute-dtype",
                    "float64",
                ],
            )
            == 0
        )
        summary = json.loads(capsys.readouterr().out)
        assert summary["inputs"]["variance_present"]
    actual = pq.read_table(tmp_path / "exported" / "fits.parquet")
    expected = pq.read_table(tmp_path / "manual" / "fits.parquet")
    assert actual.equals(expected)
    assert actual["converged"].to_pylist() == [True]
    fitted = np.asarray(
        [[actual[name][0].as_py() for name in model.parameter_names]]
    )
    prediction = model.evaluate(fitted, mode="difference")
    chi_square = np.sum(
        (images[mask] - prediction[mask]) ** 2 / variance[mask]
    )
    np.testing.assert_allclose(
        actual["chi_square"][0].as_py(), chi_square, rtol=1e-10
    )
    feature_dir = tmp_path / "features"
    build_xfit_feature_bundle(
        dataset_dir=root,
        xfit_run_dir=tmp_path / "exported",
        output_dir=feature_dir,
    )
    matrix = load_xfit_feature_matrix(
        dataset_dir=root, feature_dir=feature_dir
    )
    assert matrix.values[0, FEATURE_NAMES.index("variance_weighted")] == 1
    for name in ("log_delta_chi_square", "log_reduced_chi_square"):
        assert matrix.values[0, FEATURE_NAMES.index(name)] != 0


@pytest.mark.parametrize("value", [-1, 2])
def test_export_treats_any_nonzero_mask_value_as_included(tmp_path, value):
    root, images = _dataset(tmp_path)
    images[1, 0, 0] = np.nan
    np.save(root / "difference.npy", images)
    mask = np.full_like(images, value)
    path = tmp_path / "mask.npy"
    np.save(path, mask)
    with pytest.raises(ValueError, match="non-finite included"):
        export_xfit_input(
            dataset_dir=root,
            output_path=tmp_path / "input.npz",
            mask_path=path,
        )


@pytest.mark.parametrize("plane", ["variance", "mask"])
def test_export_rejects_npz_disguised_as_npy(tmp_path, plane):
    root, images = _dataset(tmp_path)
    path = tmp_path / f"{plane}.npy"
    with path.open("wb") as handle:
        np.savez(handle, values=np.ones_like(images))
    with pytest.raises(ValueError, match="single .npy"):
        export_xfit_input(
            dataset_dir=root,
            output_path=tmp_path / "input.npz",
            **{f"{plane}_path": path},
        )


def test_export_rejects_boolean_variance(tmp_path):
    root, images = _dataset(tmp_path)
    path = tmp_path / "variance.npy"
    np.save(path, np.ones(images.shape, dtype=bool))
    with pytest.raises(ValueError, match="real numeric"):
        export_xfit_input(
            dataset_dir=root,
            output_path=tmp_path / "input.npz",
            variance_path=path,
        )


@pytest.mark.parametrize("label", ["", "   ", 123])
def test_export_rejects_invalid_unit_labels(tmp_path, label):
    root, _ = _dataset(tmp_path)
    with pytest.raises(ValueError, match="nonempty string"):
        export_xfit_input(
            dataset_dir=root,
            output_path=tmp_path / "input.npz",
            image_unit=label,
        )


def test_export_selects_noncontiguous_unique_rows_for_every_plane(tmp_path):
    images = np.zeros((4, 5, 7), dtype=np.float32)
    images[2] = 3
    root, _ = _dataset(
        tmp_path, images=images, candidate_ids=["z", "z", "a", "z"]
    )
    variance = np.full(images.shape, 2.0)
    variance[2] = 7
    mask = np.ones(images.shape, dtype=bool)
    mask[2, 0] = False
    variance_path = tmp_path / "variance.npy"
    mask_path = tmp_path / "mask.npy"
    np.save(variance_path, variance)
    np.save(mask_path, mask)
    output = tmp_path / "input.npz"
    export_xfit_input(
        dataset_dir=root,
        output_path=output,
        variance_path=variance_path,
        mask_path=mask_path,
    )
    dataset = load_xfit_dataset(output)
    assert dataset.candidate_id.tolist() == ["z", "a"]
    for actual, source in (
        (dataset.images, images),
        (dataset.variance, variance),
        (dataset.mask, mask),
    ):
        np.testing.assert_array_equal(actual, source[[0, 2]])
