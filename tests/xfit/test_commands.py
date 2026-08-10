# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest
import yaml

from cuphoton.core.artifacts import array_sha256, file_sha256
from cuphoton.core.cli import run_component
from cuphoton.xfit.io import load_xfit_dataset, write_fit_artifacts


def _write_gaussian_input(
    path,
    *,
    mode: str = "difference",
    include_auxiliaries: bool = False,
    candidate_id: np.ndarray | None = None,
    dtype=np.float64,
):
    images = np.zeros((2, 5, 7), dtype=dtype)
    if mode == "split":
        images = np.stack((images, images + 1.0, images + 2.0), axis=1)
    images[1] += 0.25
    initial = np.asarray(
        [
            [4.0, 1.2, 1.4, 0.1, 2.0, 2.0, 4.0, 2.5],
            [5.0, 1.3, 1.1, -0.1, 2.1, 1.9, 4.1, 2.6],
        ],
        dtype=dtype,
    )
    if candidate_id is None:
        candidate_id = np.asarray(["candidate-a", "candidate-b"])
    arrays = {
        "candidate_id": candidate_id,
        "images": images,
        "initial": initial,
    }
    if include_auxiliaries:
        arrays["mask"] = np.ones(images.shape[-2:], dtype=bool)
        arrays["variance"] = np.ones(images.shape[-2:], dtype=np.float64)
    np.savez_compressed(path, **arrays)
    return images


def _fake_result(images: np.ndarray):
    parameters = np.asarray(
        [
            [4.0, 1.2, 1.4, 0.1, 2.0, 2.0, 4.0, 2.5],
            [5.0, 1.3, 1.1, -0.1, 2.1, 1.9, 4.1, 2.6],
        ],
        dtype=np.float64,
    )
    parameter_names = (
        "amplitude",
        "sigma_x",
        "sigma_y",
        "theta",
        "x_pos",
        "y_pos",
        "x_neg",
        "y_neg",
    )
    return SimpleNamespace(
        parameters=parameters,
        parameter_names=parameter_names,
        status=np.asarray(["converged-f-tol", "max-evaluations"]),
        converged=np.asarray([True, False]),
        evaluations=np.asarray([7, 20]),
        residual_norm=np.asarray([0.1, 0.5]),
        chi_square=np.asarray([0.01, 0.25]),
        valid_pixel_count=np.asarray([35, 35]),
        valid_pixel_fraction=np.asarray([1.0, 1.0]),
        null_chi_square=np.asarray([1.0, 0.2]),
        delta_chi_square=np.asarray([0.99, -0.05]),
        fractional_null_improvement=np.asarray([0.99, -0.25]),
        degrees_of_freedom=np.asarray([27, 27]),
        reduced_chi_square=np.asarray([0.01 / 27, 0.25 / 27]),
        covariance=np.stack((np.eye(8), np.full((8, 8), np.nan))),
        standard_errors=np.stack((np.ones(8), np.full(8, np.nan))),
        uncertainty_valid=np.asarray([True, False]),
        uncertainty_reason=("", "rank-deficient"),
        residuals=np.zeros_like(images),
        backend="numpy",
        device="cpu",
        dtype="float64",
        model="gaussian",
        mode="difference",
    )


def test_data_inspect_reports_safe_array_inventory(tmp_path, capsys) -> None:
    input_path = tmp_path / "input.npz"
    _write_gaussian_input(input_path)

    rc = run_component("xfit", ["data-inspect", "--input", str(input_path)])
    captured = capsys.readouterr()

    assert rc == 0
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["batch_size"] == 2
    assert payload["mode"] == "difference"
    assert payload["arrays"]["images"] == {
        "dtype": "float64",
        "shape": [2, 5, 7],
    }


def test_data_validate_checks_selected_mode_and_model(
    tmp_path, capsys
) -> None:
    input_path = tmp_path / "split.npz"
    _write_gaussian_input(input_path, mode="split")

    rc = run_component(
        "xfit",
        [
            "data-validate",
            "--input",
            str(input_path),
            "--model",
            "gaussian",
            "--mode",
            "split",
        ],
    )
    captured = capsys.readouterr()

    assert rc == 0
    assert captured.err == ""
    assert json.loads(captured.out)["valid"] is True


def test_input_rejects_npy_files(tmp_path, capsys) -> None:
    input_path = tmp_path / "input.npy"
    np.save(input_path, np.zeros((2, 5, 7)), allow_pickle=False)

    rc = run_component("xfit", ["data-inspect", "--input", str(input_path)])
    captured = capsys.readouterr()

    assert rc == 1
    assert ".npz inputs only" in captured.err


def test_input_rejects_object_and_pickle_backed_arrays(tmp_path) -> None:
    input_path = tmp_path / "object.npz"
    np.savez_compressed(
        input_path,
        candidate_id=np.asarray(["a"], dtype=object),
        images=np.zeros((1, 3, 3)),
    )

    with pytest.raises(ValueError, match="object or pickle-backed"):
        load_xfit_dataset(input_path)


@pytest.mark.parametrize("missing", ["candidate_id", "images"])
def test_input_requires_named_arrays(tmp_path, missing: str) -> None:
    arrays = {
        "candidate_id": np.asarray(["a"]),
        "images": np.zeros((1, 3, 3)),
    }
    del arrays[missing]
    input_path = tmp_path / "missing.npz"
    np.savez_compressed(input_path, **arrays)

    with pytest.raises(ValueError, match="missing required"):
        load_xfit_dataset(input_path)


def test_input_rejects_duplicate_candidate_ids(tmp_path) -> None:
    input_path = tmp_path / "duplicates.npz"
    np.savez_compressed(
        input_path,
        candidate_id=np.asarray(["same", "same"]),
        images=np.zeros((2, 3, 3)),
    )

    with pytest.raises(ValueError, match="must be unique"):
        load_xfit_dataset(input_path)


def test_input_rejects_byte_string_candidate_ids(tmp_path) -> None:
    input_path = tmp_path / "bytes.npz"
    np.savez_compressed(
        input_path,
        candidate_id=np.asarray([b"candidate-a"]),
        images=np.zeros((1, 3, 3)),
    )

    with pytest.raises(ValueError, match="Unicode strings"):
        load_xfit_dataset(input_path)


def test_input_rejects_candidate_ids_above_signed_64_bit(tmp_path) -> None:
    input_path = tmp_path / "uint64.npz"
    np.savez_compressed(
        input_path,
        candidate_id=np.asarray([2**64 - 1], dtype=np.uint64),
        images=np.zeros((1, 3, 3)),
    )

    with pytest.raises(ValueError, match="fit signed 64-bit"):
        load_xfit_dataset(input_path)


def test_input_rejects_unknown_arrays(tmp_path) -> None:
    input_path = tmp_path / "unknown.npz"
    np.savez_compressed(
        input_path,
        candidate_id=np.asarray(["a"]),
        images=np.zeros((1, 3, 3)),
        hidden_pickle=np.asarray(["not part of the contract"]),
    )

    with pytest.raises(ValueError, match="unsupported array"):
        load_xfit_dataset(input_path)


@pytest.mark.parametrize(
    "key,value",
    [
        ("images", np.ones((1, 3, 3), dtype=bool)),
        ("initial", np.ones((1, 8), dtype=bool)),
        ("variance", np.ones((1, 3, 3), dtype=bool)),
        ("stamp_basis", np.ones((3, 3), dtype=bool)),
    ],
)
def test_input_rejects_boolean_scientific_arrays(
    tmp_path,
    key: str,
    value: np.ndarray,
) -> None:
    arrays = {
        "candidate_id": np.asarray(["a"]),
        "images": np.zeros((1, 3, 3)),
    }
    arrays[key] = value
    input_path = tmp_path / f"boolean-{key}.npz"
    np.savez_compressed(input_path, **arrays)

    with pytest.raises(ValueError, match="real numeric array"):
        load_xfit_dataset(input_path)


def test_input_accepts_boolean_mask(tmp_path) -> None:
    input_path = tmp_path / "boolean-mask.npz"
    np.savez_compressed(
        input_path,
        candidate_id=np.asarray(["a"]),
        images=np.zeros((1, 3, 3)),
        mask=np.ones((3, 3), dtype=bool),
    )

    dataset = load_xfit_dataset(input_path)

    assert dataset.mask is not None
    assert dataset.mask.dtype == np.dtype(bool)


def test_input_accepts_masked_nonfinite_image_values(tmp_path) -> None:
    input_path = tmp_path / "masked-nonfinite.npz"
    images = np.ones((1, 3, 3), dtype=np.float64)
    images[:, 0, 0] = np.nan
    mask = np.ones((3, 3), dtype=bool)
    mask[0, 0] = False
    np.savez(
        input_path,
        candidate_id=np.asarray(["candidate-0"]),
        images=images,
        mask=mask,
    )

    dataset = load_xfit_dataset(input_path, mode="difference")
    assert np.isnan(dataset.images[:, 0, 0]).all()


def test_input_rejects_unmasked_nonfinite_image_values(tmp_path) -> None:
    input_path = tmp_path / "unmasked-nonfinite.npz"
    images = np.ones((1, 3, 3), dtype=np.float64)
    images[:, 0, 0] = np.nan
    np.savez(
        input_path,
        candidate_id=np.asarray(["candidate-0"]),
        images=images,
    )

    with pytest.raises(ValueError, match="finite at included pixels"):
        load_xfit_dataset(input_path, mode="difference")


def test_stamp_model_requires_one_numeric_basis(tmp_path) -> None:
    input_path = tmp_path / "stamp.npz"
    np.savez_compressed(
        input_path,
        candidate_id=np.asarray([1]),
        images=np.zeros((1, 5, 7)),
        stamp_basis=np.zeros((2, 3, 3)),
    )

    with pytest.raises(ValueError, match="only one mode"):
        load_xfit_dataset(input_path, model="stamp", mode="difference")


def test_split_auxiliary_arrays_support_documented_broadcasts(
    tmp_path,
) -> None:
    input_path = tmp_path / "broadcast.npz"
    mask = np.ones((2, 5, 7), dtype=bool)
    mask[:, 0, 0] = False
    variance = np.ones((5, 7), dtype=np.float64)
    variance[0, 0] = np.nan
    np.savez_compressed(
        input_path,
        candidate_id=np.asarray([1, 2]),
        images=np.zeros((2, 3, 5, 7)),
        mask=mask,
        variance=variance,
        initial=np.zeros((2, 8)),
    )

    dataset = load_xfit_dataset(
        input_path,
        model="gaussian",
        mode="split",
    )

    assert dataset.mask.shape == (2, 5, 7)
    assert dataset.variance.shape == (5, 7)


def test_three_row_split_accepts_explicit_per_plane_auxiliary(
    tmp_path,
) -> None:
    input_path = tmp_path / "per-plane.npz"
    np.savez_compressed(
        input_path,
        candidate_id=np.asarray([1, 2, 3]),
        images=np.zeros((3, 3, 5, 7)),
        mask=np.ones((1, 3, 5, 7), dtype=bool),
        initial=np.ones((3, 8)),
    )

    dataset = load_xfit_dataset(
        input_path,
        model="gaussian",
        mode="split",
    )

    assert dataset.mask.shape == (1, 3, 5, 7)


@pytest.mark.parametrize(
    "candidate_id",
    [
        pytest.param(
            np.asarray(["candidate-a", "candidate-b"]), id="unicode"
        ),
        pytest.param(np.asarray([101, 202], dtype=np.int64), id="numeric"),
    ],
)
def test_artifacts_preserve_candidate_order_without_pickle(
    tmp_path,
    candidate_id: np.ndarray,
) -> None:
    input_path = tmp_path / "input.npz"
    images = _write_gaussian_input(
        input_path,
        include_auxiliaries=True,
        candidate_id=candidate_id,
    )
    dataset = load_xfit_dataset(
        input_path,
        model="gaussian",
        mode="difference",
    )
    output_dir = tmp_path / "fit"

    summary = write_fit_artifacts(
        output_dir,
        dataset=dataset,
        result=_fake_result(images),
        effective_config={
            "schema_version": 1,
            "backend": "auto",
            "model": "gaussian",
            "mode": "difference",
        },
    )

    assert set(path.name for path in output_dir.iterdir()) == {
        "effective-config.yaml",
        "fit-arrays.npz",
        "fits.parquet",
        "summary.json",
    }
    table = pq.read_table(output_dir / "fits.parquet").to_pylist()
    assert [row["candidate_id"] for row in table] == candidate_id.tolist()
    assert [row["input_image_sha256"] for row in table] == [
        array_sha256(image) for image in images
    ]
    assert table[0]["input_image_sha256"] != table[1]["input_image_sha256"]
    assert table[0]["amplitude"] == 4.0
    assert table[0]["amplitude_standard_error"] == 1.0
    assert table[0]["valid_pixel_count"] == 35
    assert table[0]["valid_pixel_fraction"] == 1.0
    assert table[0]["null_chi_square"] == 1.0
    assert table[0]["delta_chi_square"] == 0.99
    assert table[0]["fractional_null_improvement"] == 0.99
    assert table[1]["delta_chi_square"] == -0.05
    assert table[1]["fractional_null_improvement"] == -0.25
    with np.load(output_dir / "fit-arrays.npz", allow_pickle=False) as arrays:
        assert arrays.files == [
            "candidate_index",
            "candidate_id",
            "covariance",
            "residuals",
        ]
        assert arrays["candidate_index"].tolist() == [0, 1]
        assert np.array_equal(arrays["candidate_id"], candidate_id)
        assert arrays["candidate_id"].dtype.kind in "iuU"
        assert arrays["covariance"].dtype != object
        assert arrays["residuals"].shape == images.shape
    persisted = json.loads((output_dir / "summary.json").read_text())
    assert persisted == summary
    assert persisted["inputs"]["input_archive_sha256"] == (
        dataset.input_archive_sha256
    )
    assert dataset.input_archive_sha256 == file_sha256(input_path)
    assert persisted["inputs"]["mask_present"] is True
    assert persisted["inputs"]["variance_present"] is True
    assert persisted["artifact_sha256"] == {
        "effective_config": file_sha256(output_dir / "effective-config.yaml"),
        "fits": file_sha256(output_dir / "fits.parquet"),
        "fit_arrays": file_sha256(output_dir / "fit-arrays.npz"),
    }
    config = yaml.safe_load(
        (output_dir / "effective-config.yaml").read_text()
    )
    assert config["backend"] == "auto"


def test_artifacts_retain_archive_hash_captured_at_load_time(
    tmp_path,
) -> None:
    input_path = tmp_path / "input.npz"
    images = _write_gaussian_input(input_path, dtype=np.float32)
    dataset = load_xfit_dataset(
        input_path,
        model="gaussian",
        mode="difference",
    )
    loaded_sha256 = dataset.input_archive_sha256
    input_path.write_bytes(b"changed after xFit loaded the archive")

    summary = write_fit_artifacts(
        tmp_path / "fit",
        dataset=dataset,
        result=_fake_result(images),
        effective_config={
            "schema_version": 1,
            "backend": "numpy",
            "model": "gaussian",
            "mode": "difference",
        },
    )

    assert summary["inputs"]["input_archive_sha256"] == loaded_sha256
    assert loaded_sha256 != file_sha256(input_path)


def test_fit_dipoles_command_writes_complete_artifact_set(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    import cuphoton.xfit as xfit

    input_path = tmp_path / "input.npz"
    images = _write_gaussian_input(input_path, dtype=np.float32)
    output_dir = tmp_path / "fit-output"

    class FakeConfig:
        def __init__(self, **kwargs):
            self.values = kwargs
            self.initial_damping = 1.0e-3
            self.damping_increase = 10.0
            self.damping_decrease = 0.3
            self.finite_difference_step = None
            self.use_finite_difference = kwargs["use_finite_difference"]

        def resolved_max_evaluations(self, parameter_count):
            configured = self.values["max_evaluations"]
            return (
                200 * (parameter_count + 1)
                if configured is None
                else configured
            )

    def fake_fit_dipoles(received_images, **kwargs):
        assert np.array_equal(received_images, images)
        assert received_images.dtype == np.float64
        assert kwargs["model"] == "gaussian"
        assert isinstance(kwargs["config"], FakeConfig)
        return _fake_result(received_images)

    monkeypatch.setattr(xfit, "LMConfig", FakeConfig, raising=False)
    monkeypatch.setattr(
        xfit,
        "fit_dipoles",
        fake_fit_dipoles,
        raising=False,
    )

    rc = run_component(
        "xfit",
        [
            "fit-dipoles",
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--model",
            "gaussian",
            "--backend",
            "numpy",
            "--compute-dtype",
            "float64",
        ],
    )
    captured = capsys.readouterr()

    assert rc == 0
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["backend"] == "numpy"
    assert payload["dtype"] == "float64"
    assert payload["inputs"]["images_dtype"] == "float32"
    assert payload["inputs"]["mask_present"] is False
    assert payload["inputs"]["variance_present"] is False
    assert (output_dir / "summary.json").is_file()
    assert (output_dir / "effective-config.yaml").is_file()
    assert (output_dir / "fits.parquet").is_file()
    assert (output_dir / "fit-arrays.npz").is_file()
    effective = yaml.safe_load(
        (output_dir / "effective-config.yaml").read_text(encoding="utf-8")
    )
    assert effective["compute_dtype"] == {
        "requested": "float64",
        "resolved": "float64",
    }


def test_fit_dipoles_command_runs_real_synthetic_gaussian_fit(
    tmp_path,
    capsys,
) -> None:
    from cuphoton.xfit import GaussianDipoleModel

    model = GaussianDipoleModel((9, 11), dtype=np.float64)
    truth = np.asarray([[5.0, 1.4, 1.2, 0.1, -1.0, 0.0, 1.0, 0.0]])
    images = np.asarray(model.evaluate(truth))
    initial = truth.copy()
    initial[:, 0] *= 0.95
    input_path = tmp_path / "synthetic.npz"
    output_dir = tmp_path / "synthetic-fit"
    np.savez_compressed(
        input_path,
        candidate_id=np.asarray(["synthetic-0"]),
        images=images,
        initial=initial,
    )

    rc = run_component(
        "xfit",
        [
            "fit-dipoles",
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--model",
            "gaussian",
            "--backend",
            "numpy",
        ],
    )
    captured = capsys.readouterr()

    assert rc == 0
    assert captured.err == ""
    assert json.loads(captured.out)["metrics"]["converged_count"] == 1
    row = pq.read_table(output_dir / "fits.parquet").to_pylist()[0]
    assert row["candidate_id"] == "synthetic-0"
    assert row["converged"] is True
    assert row["amplitude"] == pytest.approx(5.0, rel=1.0e-5)
    effective = yaml.safe_load(
        (output_dir / "effective-config.yaml").read_text()
    )
    assert effective["solver"]["f_tol"] == pytest.approx(
        np.sqrt(np.finfo(np.float64).eps)
    )
    assert effective["solver"]["max_evaluations"] == 1800


def test_stamp_command_records_resolved_finite_difference_mode(
    tmp_path,
    capsys,
) -> None:
    from cuphoton.xfit import StampDipoleModel

    y, x = np.mgrid[-3:4, -2:3]
    basis = np.exp(-0.5 * ((x / 1.0) ** 2 + (y / 1.2) ** 2))
    model = StampDipoleModel(basis, image_shape=(9, 13), dtype=np.float64)
    truth = np.asarray([[-2.1, 0.6, 2.2, -0.4, 5.0]])
    initial = truth + np.asarray([[0.15, -0.1, -0.12, 0.1, -0.4]])
    input_path = tmp_path / "stamp.npz"
    output_dir = tmp_path / "stamp-fit"
    np.savez_compressed(
        input_path,
        candidate_id=np.asarray(["stamp-0"]),
        images=model.evaluate(truth),
        initial=initial,
        stamp_basis=basis,
    )

    rc = run_component(
        "xfit",
        [
            "fit-dipoles",
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--model",
            "stamp",
            "--backend",
            "numpy",
        ],
    )
    captured = capsys.readouterr()

    assert rc == 0
    assert captured.err == ""
    assert json.loads(captured.out)["metrics"]["converged_count"] == 1
    effective = yaml.safe_load(
        (output_dir / "effective-config.yaml").read_text()
    )
    assert effective["solver"]["use_finite_difference"] is True


def test_fit_command_refuses_to_overwrite_output_directory(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    import cuphoton.xfit as xfit

    input_path = tmp_path / "input.npz"
    images = _write_gaussian_input(input_path)
    output_dir = tmp_path / "existing"
    output_dir.mkdir()

    class FakeConfig:
        def __init__(self, **kwargs):
            self.values = kwargs

    monkeypatch.setattr(xfit, "LMConfig", FakeConfig, raising=False)
    monkeypatch.setattr(
        xfit,
        "fit_dipoles",
        lambda *args, **kwargs: _fake_result(images),
        raising=False,
    )

    rc = run_component(
        "xfit",
        [
            "fit-dipoles",
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--model",
            "gaussian",
        ],
    )
    captured = capsys.readouterr()

    assert rc == 1
    assert "output directory already exists" in captured.err


def test_command_help_exposes_stable_safe_input_options(capsys) -> None:
    rc = run_component("xfit", ["help", "fit-dipoles"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Usage: cuphoton xfit fit-dipoles" in captured.out
    assert "--input" in captured.out
    assert "--output-dir" in captured.out
    assert "--stamp-evaluation" in captured.out
    assert "--compute-dtype" in captured.out
    assert "vignetted" in captured.out
    assert "finite-volume" in captured.out
    assert "--use-finite-difference" in captured.out
