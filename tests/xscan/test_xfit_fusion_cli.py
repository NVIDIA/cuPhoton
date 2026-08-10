# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""CLI integration coverage for explicit xFit late fusion."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cuphoton.core.artifacts import array_sha256, file_sha256
from cuphoton.core.cli import run_component
from cuphoton.xscan.config import ModelConfig, TrainingConfig
from cuphoton.xscan.training import train_classifier
from cuphoton.xscan.xfit_features import (
    CANDIDATE_ID_ARTIFACT_NAME,
    FEATURE_ARTIFACT_NAME,
    FEATURE_NAMES,
    FEATURE_TRANSFORMS,
    INPUT_IMAGE_SHA256_ARTIFACT_NAME,
    TRANSFORM_CONSTANTS,
)


def test_fusion_cli_exposes_explicit_selection_and_evaluation_override(
    capsys,
) -> None:
    assert run_component("xscan", ["help", "infer-real-bogus"]) == 0
    inference_help = capsys.readouterr().out
    assert "--use-xfit-features" in inference_help
    assert "--xfit-feature-dir" in inference_help
    assert "--allow-xfit-coverage-mismatch" not in inference_help

    assert run_component("xscan", ["help", "evaluate-real-bogus"]) == 0
    evaluation_help = capsys.readouterr().out
    assert "--use-xfit-features" in evaluation_help
    assert "--xfit-feature-dir" in evaluation_help
    assert "--allow-xfit-coverage-mismatch" in evaluation_help


def _write_fusion_inputs(root: Path) -> tuple[Path, Path]:
    dataset_dir = root / "dataset"
    feature_dir = root / "xfit-features"
    dataset_dir.mkdir()
    feature_dir.mkdir()

    rng = np.random.default_rng(23)
    sample_count = 6
    search = rng.normal(size=(sample_count, 17, 17)).astype(np.float32)
    template = rng.normal(size=(sample_count, 17, 17)).astype(np.float32)
    difference = search - template
    labels = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)
    splits = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
    candidate_ids = np.asarray(
        [f"candidate-{index}" for index in range(sample_count)]
    )
    np.save(dataset_dir / "search.npy", search, allow_pickle=False)
    np.save(dataset_dir / "template.npy", template, allow_pickle=False)
    np.save(dataset_dir / "difference.npy", difference, allow_pickle=False)
    np.save(dataset_dir / "labels.npy", labels, allow_pickle=False)
    np.save(dataset_dir / "split.npy", splits, allow_pickle=False)
    metadata = [
        {
            "candidate_id": str(candidate_ids[index]),
            "split": ("train", "val", "test")[int(splits[index])],
            "split_group": f"group-{index}",
            "label": int(labels[index]),
            "label_source": "independent_human_review",
            "target_label_available": True,
        }
        for index in range(sample_count)
    ]
    (dataset_dir / "metadata.jsonl").write_text(
        "\n".join(json.dumps(row) for row in metadata) + "\n",
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
    candidate_id_path = feature_dir / CANDIDATE_ID_ARTIFACT_NAME
    feature_path = feature_dir / FEATURE_ARTIFACT_NAME
    input_image_sha256_path = feature_dir / INPUT_IMAGE_SHA256_ARTIFACT_NAME
    np.save(candidate_id_path, candidate_ids, allow_pickle=False)
    np.save(feature_path, features, allow_pickle=False)
    np.save(
        input_image_sha256_path,
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
        "candidate_id_dtype": str(candidate_ids.dtype),
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
        "source_artifacts": {
            "summary": {"name": "summary.json", "sha256": "1" * 64},
            "fits": {"name": "fits.parquet", "sha256": "2" * 64},
            "fit_arrays": {
                "name": "fit-arrays.npz",
                "sha256": "3" * 64,
            },
        },
        "artifacts": {
            "candidate_id": {
                "name": CANDIDATE_ID_ARTIFACT_NAME,
                "sha256": file_sha256(candidate_id_path),
            },
            "features": {
                "name": FEATURE_ARTIFACT_NAME,
                "sha256": file_sha256(feature_path),
            },
            "input_image_sha256": {
                "name": INPUT_IMAGE_SHA256_ARTIFACT_NAME,
                "sha256": file_sha256(input_image_sha256_path),
            },
            "schema": {"name": "schema.json"},
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


def test_fused_inference_and_evaluation_cli_end_to_end(
    tmp_path: Path, capsys
) -> None:
    dataset_dir, feature_dir = _write_fusion_inputs(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = TrainingConfig(
        dataset_dir=str(dataset_dir),
        use_xfit_features=True,
        xfit_feature_dir=str(feature_dir),
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
    )
    train_classifier(config, run_dir=run_dir)

    common_args = [
        "--run-dir",
        str(run_dir),
        "--dataset-dir",
        str(dataset_dir),
        "--split",
        "test",
        "--batch-size",
        "2",
        "--use-xfit-features",
        "--xfit-feature-dir",
        str(feature_dir),
    ]
    assert run_component("xscan", ["infer-real-bogus", *common_args]) == 0
    inference = json.loads(capsys.readouterr().out)
    assert inference["xfit_features"]["enabled"] is True
    assert inference["xfit_features"]["feature_dir"] == str(
        feature_dir.resolve()
    )
    assert (run_dir / "inference" / "test" / "probabilities.npy").exists()

    assert run_component("xscan", ["evaluate-real-bogus", *common_args]) == 0
    evaluation = json.loads(capsys.readouterr().out)
    assert evaluation["xfit_features"]["enabled"] is True
    assert evaluation["xfit_features"]["feature_dir"] == str(
        feature_dir.resolve()
    )
    assert "roc_auc" in evaluation
    assert (run_dir / "evaluation" / "test" / "predictions.jsonl").exists()
