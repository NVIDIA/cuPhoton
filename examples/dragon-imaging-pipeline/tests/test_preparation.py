# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for preparation reuse and its real CLI input contracts."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from cuphoton.xpois.batch import load_image_pair_manifest
from cuphoton.xscan.pipeline_executor import load_pipeline_manifest

_SPEC = importlib.util.spec_from_file_location(
    "sc26_preparation", Path(__file__).parents[1] / "prepare.py"
)
assert _SPEC is not None and _SPEC.loader is not None
preparation = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(preparation)


def test_resume_verifies_artifacts_and_does_not_repeat_work(tmp_path):
    output = tmp_path / "artifact.txt"
    calls = []

    def run():
        calls.append(True)
        output.write_text("original", encoding="utf-8")

    preparation.checked_stage(tmp_path, "example", {"seed": 1}, [output], run)
    preparation.checked_stage(tmp_path, "example", {"seed": 1}, [output], run)
    assert calls == [True]
    with pytest.raises(ValueError, match="inputs changed"):
        preparation.checked_stage(
            tmp_path, "example", {"seed": 2}, [output], run
        )
    output.write_text("modified", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact changed"):
        preparation.checked_stage(
            tmp_path, "example", {"seed": 1}, [output], run
        )


def test_split_validation_rejects_shared_background_group():
    rows = [
        {
            "candidate_id": f"{split}-{index}",
            "split": split,
            "split_group": "same-sky-region",
            "label_source": "synthetic_moving_source",
            "label": index % 2,
        }
        for split, count in (("train", 320), ("val", 96), ("test", 96))
        for index in range(count)
    ]
    with pytest.raises(ValueError, match="multiple splits"):
        preparation.validate_splits(rows, training=True)


def test_manifests_load_with_the_product_parsers(tmp_path, monkeypatch):
    scene = {"item_id": "pair-0", "hdu": 1}
    for name in ("reference", "target", "variance", "fit_mask"):
        path = tmp_path / f"{name}.fits"
        dtype = np.uint8 if name == "fit_mask" else np.float32
        fits.HDUList(
            [
                fits.PrimaryHDU(),
                fits.ImageHDU(np.ones((96, 96), dtype=dtype)),
            ]
        ).writeto(path)
        scene[name] = str(path)
    scene["candidates"] = [{"candidate_id": "source-0", "x": 47, "y": 47}]
    model = tmp_path / "training" / "model"
    model.mkdir(parents=True)
    (model / "checkpoint.pt").write_bytes(b"descriptor-only fixture")
    features = tmp_path / "training" / "features"
    features.mkdir()
    (features / "schema.json").write_text("{}", encoding="utf-8")
    masks = tmp_path / "demo" / "subtractions" / "pair-0"
    masks.mkdir(parents=True)
    np.save(masks / "fit-mask.npy", np.ones((96, 96), dtype=bool))

    preparation.make_manifests(tmp_path, [scene], 0.5)

    pairs = load_image_pair_manifest(tmp_path / "pairs.yaml")
    assert len(pairs.pairs) == 1
    assert pairs.pairs[0].fit_mask.suffix == ".npy"
    for name, backend in (
        ("pipeline.json", "cupy"),
        ("pipeline-native.json", "native"),
    ):
        config, items = load_pipeline_manifest(tmp_path / name)
        assert config.xfit.backend == backend
        assert config.xfit.solver_version == "v0.2"
        assert items[0].candidates[0].candidate_id == "source-0"

    preparation.make_transient_chain(
        tmp_path,
        {
            "pairs": [scene],
            "reprojection": {
                "raw_valid_mask": scene["fit_mask"],
                "raw_target": scene["target"],
                "reference": scene["reference"],
            },
        },
    )
    monkeypatch.syspath_prepend(
        str(Path(__file__).parents[2] / "transient-chain")
    )
    from transient_chain.config import load_config
    from transient_chain.surveys.generic import GenericFitsAdapter

    config = load_config(tmp_path / "transient-chain" / "config.yaml")
    adapter = GenericFitsAdapter(config["survey"]["params"])
    assert adapter.template_p["on_target_grid"] is False
    assert adapter.target_p["variance_kind"] == "variance"
    assert config["classify"]["checkpoint_dir"] is None
