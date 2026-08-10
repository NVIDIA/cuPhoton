# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Late-fusion contracts for optional xFit features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import get_origin, get_type_hints

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import default_collate

from cuphoton.core.artifacts import array_sha256, file_sha256
from cuphoton.core.cli import run_component
from cuphoton.xfit import GaussianDipoleModel
from cuphoton.xscan.config import ModelConfig, TrainingConfig
from cuphoton.xscan.dataset import StampDataset
from cuphoton.xscan.model import StackedInputRealBogusModel
from cuphoton.xscan.training import (
    forward_feature_batch,
    move_feature_batch,
    require_training_label_provenance,
    train_classifier,
)
from cuphoton.xscan.types import CollatedDatasetBatch
from cuphoton.xscan.workflows import evaluate_workflow, infer_workflow
from cuphoton.xscan.xfit_features import (
    FEATURE_NAMES,
    FEATURE_TRANSFORMS,
    TRANSFORM_CONSTANTS,
    load_xfit_feature_matrix,
)


def test_xfit_pipeline_commands_are_registered(capsys) -> None:
    assert run_component("xscan", ["help", "data-export-xfit-input"]) == 0
    captured = capsys.readouterr()
    assert "--dataset-dir" in captured.out
    assert "--output" in captured.out

    assert run_component("xscan", ["help", "data-build-xfit-features"]) == 0
    captured = capsys.readouterr()
    assert "--xfit-run-dir" in captured.out
    assert "--missing-policy" in captured.out

    assert run_component("xscan", ["help", "infer-real-bogus"]) == 0
    assert "--xfit-feature-dir" in capsys.readouterr().out


def _model(
    *,
    input_mode: str,
    with_xfit: bool,
    xfit_modality_dropout: float = 0.0,
):
    return StackedInputRealBogusModel(
        input_mode=input_mode,
        image_size=17,
        depths=[1],
        num_heads=[1],
        embed_dims=[8],
        decoder_embedding_dim=4,
        pos_dim=8,
        output_nc=1,
        drop_rate=0.0,
        attn_drop=0.0,
        xfit_feature_names=(
            ["fit_present", "fit_valid", "separation_over_stamp"]
            if with_xfit
            else []
        ),
        xfit_hidden_dim=4,
        xfit_modality_dropout=xfit_modality_dropout,
    )


def _image_logit(model, images):
    if model.input_mode == "pair":
        return model.inner(images[:, 0:1], images[:, 1:2])
    return model.inner(images[:, 0:1], images[:, 1:2], images[:, 2:3])


@pytest.mark.parametrize(
    ("input_mode", "channels"), [("pair", 2), ("triplet", 3)]
)
def test_zero_initialized_xfit_fusion_preserves_image_logits(
    input_mode: str,
    channels: int,
) -> None:
    model = _model(input_mode=input_mode, with_xfit=True).eval()
    images = torch.randn(3, channels, 17, 17)
    features = torch.tensor(
        [[1.0, 1.0, 0.2], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )

    with torch.no_grad():
        expected = _image_logit(model, images)
        actual = model(images, xfit_features=features)

    assert torch.equal(actual, expected)


def test_missing_fit_gate_is_an_exact_image_only_fallback() -> None:
    model = _model(input_mode="pair", with_xfit=True).eval()
    final = model.xfit_head[-1]
    assert isinstance(final, nn.Linear)
    nn.init.ones_(final.weight)
    nn.init.constant_(final.bias, 0.5)
    images = torch.randn(2, 2, 17, 17)
    features = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 0.5]])

    with torch.no_grad():
        expected = _image_logit(model, images)
        actual = model(images, xfit_features=features)

    assert actual[0] == expected[0]
    assert actual[1] != expected[1]


def test_fusion_head_receives_gradient_from_present_fits() -> None:
    model = _model(input_mode="pair", with_xfit=True).train()
    images = torch.randn(2, 2, 17, 17)
    features = torch.tensor([[1.0, 1.0, 0.4], [1.0, 0.0, 0.1]])

    model(images, xfit_features=features).sum().backward()

    final = model.xfit_head[-1]
    assert isinstance(final, nn.Linear)
    assert final.weight.grad is not None
    assert torch.count_nonzero(final.weight.grad) > 0


def test_fusion_rejects_missing_or_wrong_features() -> None:
    model = _model(input_mode="pair", with_xfit=True).eval()
    images = torch.randn(2, 2, 17, 17)

    with pytest.raises(ValueError, match="requires xFit"):
        model(images)
    with pytest.raises(ValueError, match="configured schema"):
        model(images, xfit_features=torch.ones(2, 2))
    image_only = _model(input_mode="pair", with_xfit=False).eval()
    with pytest.raises(ValueError, match="image-only"):
        image_only(images, xfit_features=torch.ones(2, 3))


def test_xfit_modality_dropout_is_seeded_and_applies_sample_gate() -> None:
    model = _model(
        input_mode="pair",
        with_xfit=True,
        xfit_modality_dropout=0.5,
    ).train()
    assert model.xfit_head is not None
    final = model.xfit_head[-1]
    assert isinstance(final, nn.Linear)
    nn.init.ones_(final.weight)
    nn.init.constant_(final.bias, 0.25)
    images = torch.randn(16, 2, 17, 17)
    features = torch.ones(16, 3)

    with torch.no_grad():
        image_logits = _image_logit(model, images)
        fit_logits = model.xfit_head(features).squeeze(1)
        torch.manual_seed(73)
        expected_keep = (torch.rand(16) >= 0.5).to(fit_logits.dtype)
        torch.manual_seed(73)
        actual = model(images, xfit_features=features)
        torch.manual_seed(73)
        repeated = model(images, xfit_features=features)

    assert torch.equal(actual, image_logits + expected_keep * fit_logits)
    assert torch.equal(repeated, actual)
    assert torch.count_nonzero(expected_keep) not in {0, 16}


def _write_dataset_and_features(root: Path) -> tuple[Path, Path]:
    dataset_dir = root / "dataset"
    feature_dir = root / "xfit-features"
    dataset_dir.mkdir()
    feature_dir.mkdir()
    rng = np.random.default_rng(12)
    sample_count = 6
    search = rng.normal(size=(sample_count, 17, 17)).astype(np.float32)
    template = rng.normal(size=(sample_count, 17, 17)).astype(np.float32)
    labels = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)
    splits = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
    candidate_id = np.asarray(
        [f"candidate-{index}" for index in range(sample_count)]
    )
    np.save(dataset_dir / "search.npy", search, allow_pickle=False)
    np.save(dataset_dir / "template.npy", template, allow_pickle=False)
    difference = search - template
    np.save(dataset_dir / "difference.npy", difference, allow_pickle=False)
    np.save(dataset_dir / "labels.npy", labels, allow_pickle=False)
    np.save(dataset_dir / "split.npy", splits, allow_pickle=False)
    rows = [
        {
            "candidate_id": str(candidate_id[index]),
            "split": ("train", "val", "test")[int(splits[index])],
            "split_group": f"group-{index}",
            "label": int(labels[index]),
            "label_source": "independent_human_review",
            "target_label_available": True,
        }
        for index in range(sample_count)
    ]
    (dataset_dir / "metadata.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    features = np.zeros((sample_count, len(FEATURE_NAMES)), dtype=np.float32)
    for index, name in enumerate(FEATURE_NAMES):
        low, high = FEATURE_TRANSFORMS[name]["range"]
        features[:, index] = rng.uniform(low, high, size=sample_count)
    for name in (
        "fit_present",
        "fit_valid",
        "uncertainty_valid",
        "variance_weighted",
        "gaussian_shape_available",
    ):
        features[:, FEATURE_NAMES.index(name)] = 1.0
    candidate_path = feature_dir / "candidate-id.npy"
    feature_path = feature_dir / "features.npy"
    image_hash_path = feature_dir / "input-image-sha256.npy"
    np.save(candidate_path, candidate_id, allow_pickle=False)
    np.save(feature_path, features, allow_pickle=False)
    np.save(
        image_hash_path,
        np.asarray(
            [array_sha256(stamp).encode("ascii") for stamp in difference],
            dtype="S64",
        ),
        allow_pickle=False,
    )
    schema = {
        "schema_version": 1,
        "artifact": "xfit-xscan-feature-bundle",
        "feature_names": list(FEATURE_NAMES),
        "feature_dtype": "float32",
        "candidate_id_dtype": str(candidate_id.dtype),
        "features": [
            {"index": index, "name": name, **FEATURE_TRANSFORMS[name]}
            for index, name in enumerate(FEATURE_NAMES)
        ],
        "transform_constants": dict(TRANSFORM_CONSTANTS),
        "missing_policy": "error",
        "source": {
            "model": "gaussian",
            "mode": "difference",
            "image_shape": [17, 17],
            "mask_present": False,
            "variance_present": True,
            "input_archive_sha256": "0" * 64,
        },
        "artifacts": {
            "candidate_id": {
                "name": candidate_path.name,
                "sha256": file_sha256(candidate_path),
            },
            "features": {
                "name": feature_path.name,
                "sha256": file_sha256(feature_path),
            },
            "input_image_sha256": {
                "name": image_hash_path.name,
                "sha256": file_sha256(image_hash_path),
            },
            "schema": {"name": "schema.json"},
        },
        "source_artifacts": {
            "summary": {"name": "summary.json", "sha256": "1" * 64},
            "fits": {"name": "fits.parquet", "sha256": "2" * 64},
            "fit_arrays": {
                "name": "fit-arrays.npz",
                "sha256": "3" * 64,
            },
        },
        "join_diagnostics": {
            "dataset_row_count": sample_count,
            "dataset_unique_candidate_count": sample_count,
            "duplicate_dataset_row_count": 0,
            "fit_row_count": sample_count,
            "matched_dataset_row_count": sample_count,
            "missing_dataset_row_count": 0,
            "reused_fit_row_count": 0,
            "extra_fit_row_count": 0,
            "missing_candidate_ids": [],
            "extra_fit_candidate_ids": [],
        },
    }
    (feature_dir / "schema.json").write_text(
        json.dumps(schema, indent=2) + "\n",
        encoding="utf-8",
    )
    return dataset_dir, feature_dir


def test_export_xfit_input_command(tmp_path: Path, capsys) -> None:
    dataset_dir, _ = _write_dataset_and_features(tmp_path)
    output_path = tmp_path / "exported-xfit-input.npz"

    rc = run_component(
        "xscan",
        [
            "data-export-xfit-input",
            "--dataset-dir",
            str(dataset_dir),
            "--output",
            str(output_path),
        ],
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["candidate_count"] == 6
    with np.load(output_path, allow_pickle=False) as archive:
        assert archive["images"].dtype == np.float32
        assert archive["images"].shape == (6, 17, 17)


def test_public_xfit_to_xscan_artifact_pipeline(
    tmp_path: Path, capsys
) -> None:
    dataset_dir, _ = _write_dataset_and_features(tmp_path)
    truths = np.asarray(
        [
            [
                8.0 + index,
                1.8,
                1.4,
                0.15,
                -2.0,
                -0.5,
                2.0,
                0.5,
            ]
            for index in range(6)
        ],
        dtype=np.float32,
    )
    difference = np.asarray(
        GaussianDipoleModel((17, 17), dtype=np.float32).evaluate(truths)
    )
    np.save(dataset_dir / "difference.npy", difference, allow_pickle=False)
    np.save(dataset_dir / "search.npy", difference, allow_pickle=False)
    np.save(
        dataset_dir / "template.npy",
        np.zeros_like(difference),
        allow_pickle=False,
    )
    input_path = tmp_path / "pipeline-input.npz"
    fit_dir = tmp_path / "fit-run"
    feature_dir = tmp_path / "pipeline-features"

    assert (
        run_component(
            "xscan",
            [
                "data-export-xfit-input",
                "--dataset-dir",
                str(dataset_dir),
                "--output",
                str(input_path),
            ],
        )
        == 0
    )
    capsys.readouterr()
    assert (
        run_component(
            "xfit",
            [
                "fit-dipoles",
                "--input",
                str(input_path),
                "--output-dir",
                str(fit_dir),
                "--model",
                "gaussian",
                "--mode",
                "difference",
                "--backend",
                "numpy",
                "--max-evaluations",
                "20",
            ],
        )
        == 0
    )
    capsys.readouterr()
    assert (
        run_component(
            "xscan",
            [
                "data-build-xfit-features",
                "--dataset-dir",
                str(dataset_dir),
                "--xfit-run-dir",
                str(fit_dir),
                "--output-dir",
                str(feature_dir),
            ],
        )
        == 0
    )
    capsys.readouterr()

    dataset = StampDataset(
        dataset_dir,
        input_mode="triplet",
        split="test",
        xfit_feature_dir=feature_dir,
    )
    assert len(dataset) == 2
    assert dataset.xfit_feature_names == FEATURE_NAMES
    assert dataset.xfit_feature_bundle_identity is not None


def test_stamp_dataset_keeps_image_only_contract_and_opt_in_fusion(
    tmp_path: Path,
) -> None:
    dataset_dir, feature_dir = _write_dataset_and_features(tmp_path)
    image_only = StampDataset(dataset_dir, input_mode="pair", split="val")
    fused = StampDataset(
        dataset_dir,
        input_mode="pair",
        split="val",
        xfit_feature_dir=feature_dir,
    )

    image_features, image_label = image_only[0]
    fused_features, fused_label = fused[0]

    assert isinstance(image_features, torch.Tensor)
    assert isinstance(fused_features, tuple)
    assert torch.equal(fused_features[0], image_features)
    assert fused_features[1].shape == (len(FEATURE_NAMES),)
    assert fused.xfit_feature_names == FEATURE_NAMES
    assert fused_features[1][0] == 1.0
    assert fused_label == image_label


def test_split_datasets_share_one_read_only_file_backed_feature_matrix(
    tmp_path: Path,
) -> None:
    dataset_dir, feature_dir = _write_dataset_and_features(tmp_path)
    loaded = load_xfit_feature_matrix(
        dataset_dir=dataset_dir,
        feature_dir=feature_dir,
    )
    train_dataset = StampDataset(
        dataset_dir,
        input_mode="pair",
        split="train",
        xfit_feature_matrix=loaded,
    )
    val_dataset = StampDataset(
        dataset_dir,
        input_mode="pair",
        split="val",
        xfit_feature_matrix=loaded,
    )

    assert isinstance(loaded.values, np.memmap)
    assert train_dataset.xfit_features is val_dataset.xfit_features
    assert train_dataset.xfit_features is loaded.values
    assert loaded.values.flags.writeable is False


def test_shared_feature_matrix_is_bound_to_validated_dataset(
    tmp_path: Path,
) -> None:
    dataset_dir, feature_dir = _write_dataset_and_features(tmp_path)
    loaded = load_xfit_feature_matrix(
        dataset_dir=dataset_dir,
        feature_dir=feature_dir,
    )
    other_root = tmp_path / "other"
    other_root.mkdir()
    other_dataset_dir, _ = _write_dataset_and_features(other_root)

    with pytest.raises(ValueError, match="different XScan dataset"):
        StampDataset(
            other_dataset_dir,
            input_mode="pair",
            split="train",
            xfit_feature_matrix=loaded,
        )


def test_fusion_batch_annotations_are_runtime_resolvable_and_collated_list(
    tmp_path: Path,
) -> None:
    dataset_dir, feature_dir = _write_dataset_and_features(tmp_path)
    dataset = StampDataset(
        dataset_dir,
        input_mode="pair",
        split="train",
        xfit_feature_dir=feature_dir,
    )

    assert get_type_hints(StampDataset.__getitem__)
    assert get_type_hints(move_feature_batch)
    assert get_type_hints(forward_feature_batch)
    assert get_origin(CollatedDatasetBatch) is list
    assert isinstance(default_collate([dataset[0], dataset[1]]), list)


def test_training_config_preserves_positional_api_and_requires_fusion_opt_in(
    tmp_path: Path,
) -> None:
    positional = TrainingConfig("dataset", "old-output-root")
    assert positional.output_root == "old-output-root"
    assert positional.xfit_feature_dir is None
    assert positional.use_xfit_features is False

    dataset_dir, feature_dir = _write_dataset_and_features(tmp_path)
    run_dir = tmp_path / "strict-run"
    run_dir.mkdir()
    with pytest.raises(
        ValueError,
        match="xfit_feature_dir requires use_xfit_features=true",
    ):
        train_classifier(
            TrainingConfig(
                dataset_dir=str(dataset_dir),
                xfit_feature_dir=str(feature_dir),
                device="cpu",
            ),
            run_dir=run_dir,
        )
    with pytest.raises(
        ValueError,
        match="use_xfit_features=true requires top-level xfit_feature_dir",
    ):
        train_classifier(
            TrainingConfig(
                dataset_dir=str(dataset_dir),
                use_xfit_features=True,
                device="cpu",
            ),
            run_dir=run_dir,
        )


def test_evaluation_rejects_material_xfit_calibration_coverage_mismatch(
    tmp_path: Path,
) -> None:
    dataset_dir, feature_dir = _write_dataset_and_features(tmp_path)
    feature_path = feature_dir / "features.npy"
    features = np.load(feature_path, allow_pickle=False)
    features[2:4] = 0.0
    np.save(feature_path, features, allow_pickle=False)
    schema_path = feature_dir / "schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema["missing_policy"] = "indicator"
    schema["artifacts"]["features"]["sha256"] = file_sha256(feature_path)
    schema["join_diagnostics"]["fit_row_count"] = 4
    schema["join_diagnostics"]["matched_dataset_row_count"] = 4
    schema["join_diagnostics"]["missing_dataset_row_count"] = 2
    schema["join_diagnostics"]["missing_candidate_ids"] = [
        "candidate-2",
        "candidate-3",
    ]
    schema_path.write_text(
        json.dumps(schema, indent=2) + "\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "coverage-run"
    run_dir.mkdir()
    config = TrainingConfig(
        dataset_dir=str(dataset_dir),
        epochs=1,
        batch_size=2,
        device="cpu",
        model=ModelConfig(
            input_mode="pair",
            image_size=17,
            depths=[1],
            num_heads=[1],
            embed_dims=[8],
            decoder_embedding_dim=4,
            pos_dim=8,
            output_nc=1,
            drop_rate=0.0,
            attn_drop=0.0,
            xfit_hidden_dim=4,
        ),
        xfit_feature_dir=str(feature_dir),
        use_xfit_features=True,
        val_split="test",
    )
    train_classifier(config, run_dir=run_dir)

    with pytest.raises(ValueError, match="materially mismatched xFit"):
        evaluate_workflow(
            run_dir=run_dir,
            dataset_dir=dataset_dir,
            split="val",
            xfit_feature_dir=feature_dir,
            use_xfit_features=True,
        )

    evaluation = evaluate_workflow(
        run_dir=run_dir,
        dataset_dir=dataset_dir,
        split="val",
        xfit_feature_dir=feature_dir,
        use_xfit_features=True,
        allow_xfit_coverage_mismatch=True,
    )
    coverage = evaluation.summary["xfit_features"]["fit_coverage"]
    assert coverage["validation"]["split"] == "test"
    assert coverage["validation"]["fit_coverage"] == 1.0
    assert coverage["evaluated"]["split"] == "val"
    assert coverage["evaluated"]["fit_coverage"] == 0.0
    assert coverage["absolute_difference"] == 1.0
    assert coverage["material_threshold"] == 0.05
    assert coverage["materially_mismatched"] is True
    assert coverage["mismatch_allowed"] is True


@pytest.mark.parametrize("input_mode", ["pair", "triplet"])
def test_train_checkpoint_and_infer_with_xfit_sidecar(
    tmp_path: Path,
    input_mode: str,
) -> None:
    dataset_dir, feature_dir = _write_dataset_and_features(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    model_config = ModelConfig(
        input_mode=input_mode,
        image_size=17,
        depths=[1],
        num_heads=[1],
        embed_dims=[8],
        decoder_embedding_dim=4,
        pos_dim=8,
        output_nc=1,
        drop_rate=0.0,
        attn_drop=0.0,
        xfit_hidden_dim=4,
    )
    config = TrainingConfig(
        dataset_dir=str(dataset_dir),
        xfit_feature_dir=str(feature_dir),
        use_xfit_features=True,
        epochs=1,
        batch_size=2,
        device="cpu",
        model=model_config,
    )

    summary = train_classifier(config, run_dir=run_dir)

    assert summary["xfit_features"]["enabled"] is True
    assert summary["xfit_features"]["feature_names"] == list(FEATURE_NAMES)
    assert summary["xfit_features"]["fit_coverage_by_split"] == {
        "train": {
            "split": "train",
            "sample_count": 2,
            "fit_present_count": 2,
            "fit_coverage": 1.0,
        },
        "validation": {
            "split": "val",
            "sample_count": 2,
            "fit_present_count": 2,
            "fit_coverage": 1.0,
        },
    }
    checkpoint = torch.load(
        run_dir / "checkpoint.pt", map_location="cpu", weights_only=True
    )
    assert checkpoint["model_config"]["xfit_feature_names"] == list(
        FEATURE_NAMES
    )
    assert (
        checkpoint["xfit_feature_bundle"]
        == summary["xfit_features"]["bundle_identity"]
    )
    assert len(checkpoint["xfit_feature_bundle"]["schema_sha256"]) == 64
    assert len(checkpoint["xfit_feature_bundle"]["feature_sha256"]) == 64
    with pytest.raises(ValueError, match="requires --use-xfit-features"):
        infer_workflow(
            run_dir=run_dir,
            dataset_dir=dataset_dir,
            split="test",
        )
    with pytest.raises(
        ValueError,
        match="--xfit-feature-dir requires --use-xfit-features",
    ):
        infer_workflow(
            run_dir=run_dir,
            dataset_dir=dataset_dir,
            split="test",
            xfit_feature_dir=feature_dir,
        )

    inference = infer_workflow(
        run_dir=run_dir,
        dataset_dir=dataset_dir,
        split="test",
        xfit_feature_dir=feature_dir,
        use_xfit_features=True,
    )
    assert inference.summary["xfit_features"]["enabled"] is True
    probabilities = np.load(
        inference.run_dir / "probabilities.npy", allow_pickle=False
    )
    assert probabilities.shape == (2,)
    assert np.isfinite(probabilities).all()

    schema_path = feature_dir / "schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema_path.write_text(
        json.dumps(schema, sort_keys=True) + "\n", encoding="utf-8"
    )
    second_inference = infer_workflow(
        run_dir=run_dir,
        dataset_dir=dataset_dir,
        split="test",
        xfit_feature_dir=feature_dir,
        use_xfit_features=True,
    )
    assert (
        second_inference.summary["xfit_features"]["bundle_identity"]
        != checkpoint["xfit_feature_bundle"]
    )


@pytest.mark.parametrize(
    ("first_field", "second_field"),
    [
        ("split_group", "split_group"),
        ("candidate_diaObjectId", "candidate_diaObjectId"),
        ("candidate_diaObjectId", "diaObjectId"),
        ("diaObjectId", "dia_object_id"),
    ],
)
def test_training_rejects_cross_split_group_leakage(
    tmp_path: Path,
    first_field: str,
    second_field: str,
) -> None:
    dataset_dir, _ = _write_dataset_and_features(tmp_path)
    metadata_path = dataset_dir / "metadata.jsonl"
    rows = [
        json.loads(line)
        for line in metadata_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0][first_field] = "shared-entity"
    rows[2][second_field] = "shared-entity"
    metadata_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="group leakage"):
        require_training_label_provenance(dataset_dir)


def test_training_rejects_numeric_string_entity_alias_leakage(
    tmp_path: Path,
) -> None:
    dataset_dir, _ = _write_dataset_and_features(tmp_path)
    metadata_path = dataset_dir / "metadata.jsonl"
    rows = [
        json.loads(line)
        for line in metadata_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["candidate_diaObjectId"] = 42
    rows[2]["dia_object_id"] = "42"
    metadata_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="group leakage"):
        require_training_label_provenance(dataset_dir)
