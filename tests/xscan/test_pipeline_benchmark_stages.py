# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""CPU rejection tests for file-mediated benchmark stage artifacts."""

from __future__ import annotations

import builtins
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cuphoton.core.artifacts import file_sha256
from cuphoton.xscan.pipeline_benchmark import stages


@pytest.fixture
def stage_files(tmp_path: Path) -> Path:
    """Make two tiny typed records, not a scientific solver fixture."""
    items = [
        {
            "item_id": f"pair-{index}",
            "candidate_ids": [index * 2, index * 2 + 1],
        }
        for index in range(2)
    ]
    for stage in stages.STAGES:
        directory = tmp_path / stage
        directory.mkdir()
        records = []
        for index, item in enumerate(items):
            arrays = {}
            for name in stages.SCIENCE_KEYS:
                if not name.startswith(f"{stage}."):
                    continue
                if name.endswith("converged"):
                    array = np.array([True, False], dtype=np.bool_)
                elif name.endswith("evaluations"):
                    array = np.array([1, 7], dtype=np.int32)
                elif name.endswith(("features", "logits", "probabilities")):
                    array = np.array([index + 0.25, 0.75], dtype=np.float32)
                else:
                    array = np.array([index + 0.5, np.nan], dtype=np.float64)
                arrays[name] = array
            artifacts = stages._write_arrays(
                tmp_path, directory / f"item-{index:04d}", arrays
            )
            records.append({"item": item, "artifacts": artifacts})
        stages._write_json(
            directory / "summary.json",
            {
                "schema": stages.SCHEMA,
                "status": "success",
                "stage": stage,
                "configuration_sha256": "config-fixture",
                "checkpoint_sha256": "checkpoint-fixture",
                "feature_schema_sha256": "schema-fixture",
                "items_sha256": stages._json_hash(items),
                "items": records,
                "upstream_summary_sha256": {
                    previous: file_sha256(
                        tmp_path / previous / "summary.json"
                    )
                    for previous in stages.STAGES[
                        : stages.STAGES.index(stage)
                    ]
                },
            },
        )
    return tmp_path


def _edit_summary(root: Path, stage: str, change) -> None:
    path = root / stage / "summary.json"
    summary = json.loads(path.read_text())
    change(summary)
    path.write_text(json.dumps(summary))


def test_science_loads_all_arrays_with_float64_and_nan_masks(
    stage_files: Path,
) -> None:
    for index in range(2):
        arrays = stages.load_science_arrays(stage_files, index)
        assert len(arrays) == 22
        assert set(arrays) == set(stages.SCIENCE_KEYS)
        for name, values in arrays.items():
            stage = name.split(".")[0]
            summary = stages._read_summary(stage_files, stage)
            artifact = summary["items"][index]["artifacts"][name]
            source = stages._read_array(stage_files, artifact)
            assert source.dtype.str == artifact["dtype"]
            assert values.dtype == np.dtype("float64")
            np.testing.assert_array_equal(values, source.astype(np.float64))
            np.testing.assert_array_equal(np.isnan(values), np.isnan(source))


def test_last_item_prediction_tamper_is_rejected(stage_files: Path) -> None:
    path = stage_files / "xscan/item-0001/xscan.probabilities.npy"
    np.save(path, np.array([0.3, 0.4], dtype=np.float32))
    # Earlier occurrences remain valid; checking only the first hides this.
    stages.load_science_arrays(stage_files, 0)
    with pytest.raises(ValueError, match="artifact hash changed"):
        stages.load_science_arrays(stage_files, 1)


@pytest.mark.parametrize("kind", ["order", "candidate", "config"])
def test_changed_stage_identity_is_rejected(
    stage_files: Path, kind: str
) -> None:
    def change(summary):
        if kind == "order":
            summary["items"].reverse()
        elif kind == "candidate":
            summary["items"][1]["item"]["candidate_ids"].reverse()
        else:
            summary["configuration_sha256"] = "different-config"

    _edit_summary(stage_files, "xfit", change)
    with pytest.raises(ValueError, match="metadata or order|identities"):
        stages.load_science_arrays(stage_files, 0)


def test_completed_upstream_summary_change_is_rejected(
    stage_files: Path,
) -> None:
    _edit_summary(
        stage_files,
        "xpois",
        lambda summary: summary.update(extra_changed_metadata=1),
    )
    with pytest.raises(ValueError, match="upstream summary changed"):
        stages.load_science_arrays(stage_files, 0)


def test_missing_upstream_link_cannot_skip_chain_check(
    stage_files: Path,
) -> None:
    _edit_summary(
        stage_files,
        "xscan",
        lambda summary: summary["upstream_summary_sha256"].pop("xfit"),
    )
    with pytest.raises(ValueError, match="links are incomplete"):
        stages.load_science_arrays(stage_files, 0)


@pytest.mark.parametrize("field,value", [("dtype", "<f8"), ("shape", [1])])
def test_array_contract_checked_in_addition_to_file_hash(
    stage_files: Path, field: str, value
) -> None:
    summary = stages._read_summary(stage_files, "xscan")
    artifact = summary["items"][0]["artifacts"]["xscan.logits"]
    artifact[field] = value
    with pytest.raises(ValueError, match="artifact contract changed"):
        stages._read_array(stage_files, artifact)


def test_synchronized_timer_waits_for_completion(monkeypatch) -> None:
    calls = []
    ticks = iter([10.0, 13.0])
    monkeypatch.setattr(stages.time, "perf_counter", lambda: next(ticks))
    timings = {}
    result = stages._measure(
        timings,
        "compute",
        lambda: calls.append("work") or "result",
        lambda: calls.append("sync"),
    )
    assert result == "result"
    assert calls == ["sync", "work", "sync"]
    assert timings == {"compute": 3.0}


@pytest.mark.parametrize("stage", stages.STAGES)
def test_setup_imports_only_its_numerical_runtime(
    tmp_path, monkeypatch, stage
) -> None:
    from cuphoton.xscan.config import PerformanceConfig

    checkpoint = tmp_path / "checkpoint.pt"
    schema = tmp_path / "schema.json"
    checkpoint.write_bytes(b"checkpoint fixture")
    schema.write_text("{}")
    config = SimpleNamespace(
        checkpoint_dir=str(tmp_path),
        checkpoint_sha256=file_sha256(checkpoint),
        feature_schema_path=str(schema),
        feature_schema_sha256=file_sha256(schema),
        stamp_shape=(17, 17),
        inference_policy=stages.pipeline._strict_inference_policy_payload(),
        device_id=0,
        device="cuda:0",
    )
    for name in (
        "_validate_feature_schema_source",
        "_validate_checkpoint_contract",
        "_validate_loaded_model_contract",
    ):
        monkeypatch.setattr(stages.pipeline, name, lambda *a, **kw: None)
    monkeypatch.setattr(
        stages.pipeline, "_load_feature_schema_contract", lambda *a, **kw: {}
    )
    calls = []
    fake_torch = SimpleNamespace(
        device=lambda name: name,
        set_num_threads=lambda count: calls.append(("threads", count)),
        cuda=SimpleNamespace(
            set_device=lambda index: calls.append(("torch", index)),
            synchronize=lambda device: calls.append(("sync", device)),
        ),
    )
    fake_cupy = SimpleNamespace(
        cuda=SimpleNamespace(
            Device=lambda index: SimpleNamespace(
                use=lambda: calls.append(("cupy", index)),
                synchronize=lambda: calls.append(("sync", index)),
            )
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    monkeypatch.setitem(
        sys.modules,
        "cuphoton.xscan.training",
        SimpleNamespace(
            load_model_from_checkpoint=lambda *a, **kw: (
                object(),
                {},
                PerformanceConfig(**config.inference_policy),
            )
        ),
    )
    original_import = builtins.__import__
    forbidden = (
        {"cupy"} if stage == "xscan" else {"torch", "cuphoton.xscan.training"}
    )

    def guarded_import(name, *args, **kwargs):
        if any(
            name == item or name.startswith(item + ".") for item in forbidden
        ):
            pytest.fail(f"{stage} imported unused runtime {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    runtime = stages._setup(stage, config)
    if stage == "xscan":
        assert runtime["cp"] is None
        assert runtime["torch"] is fake_torch
        assert calls == [("threads", 1), ("torch", 0), ("sync", "cuda:0")]
    else:
        assert runtime["torch"] is None
        assert runtime["cp"] is fake_cupy
        assert calls == [("cupy", 0), ("sync", 0)]
