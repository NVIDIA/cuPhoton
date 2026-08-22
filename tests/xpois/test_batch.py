# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import numpy as np
import pytest

from cuphoton.xpois.batch import (
    BatchFitOptions,
    load_image_pair_manifest,
    preflight_image_pair_manifest,
    run_image_pair_item,
)


def _options(**overrides) -> BatchFitOptions:
    values = {
        "kernel_shape": (9, 9),
        "basis_sigmas": (1.5,),
        "basis_degrees": (0,),
        "backend": "cupy",
    }
    values.update(overrides)
    return BatchFitOptions(**values)


def _write_array(path, shape=(32, 32)) -> None:
    np.save(path, np.zeros(shape, dtype=np.float64), allow_pickle=False)


def test_manifest_resolves_paths_hashes_and_preflights(tmp_path) -> None:
    _write_array(tmp_path / "reference.npy")
    _write_array(tmp_path / "target.npy")
    manifest_path = tmp_path / "pairs.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "cuphoton.xpois.image-pairs/v1",
                "pairs": [
                    {
                        "id": "pair-1",
                        "reference": "reference.npy",
                        "target": "target.npy",
                    }
                ],
            }
        )
    )

    manifest = load_image_pair_manifest(manifest_path)
    preflight_image_pair_manifest(manifest, _options())

    assert manifest.pairs[0].reference == (tmp_path / "reference.npy")
    assert len(manifest.sha256) == 64
    identity = manifest.input_identity_payload()
    assert identity["policy"] == "size-mtime-ns"
    assert identity["content_hashes"] is False
    assert len(identity["files"]) == 2
    assert (
        manifest.work_items()[0].weight_bytes
        == 2 * (tmp_path / "reference.npy").stat().st_size
    )


def test_manifest_work_item_counts_reused_path_once(tmp_path) -> None:
    shared_path = tmp_path / "shared.npy"
    _write_array(shared_path)
    manifest_path = tmp_path / "pairs.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "cuphoton.xpois.image-pairs/v1",
                "pairs": [
                    {
                        "id": "shared",
                        "reference": "shared.npy",
                        "target": "shared.npy",
                    }
                ],
            }
        )
    )

    manifest = load_image_pair_manifest(manifest_path)
    item = manifest.work_items()[0]

    assert item.weight_bytes == shared_path.stat().st_size
    assert len(item.payload["_input_identity"]) == 1


def test_preflight_rejects_input_changed_after_manifest_load(
    tmp_path,
) -> None:
    _write_array(tmp_path / "reference.npy")
    _write_array(tmp_path / "target.npy")
    manifest_path = tmp_path / "pairs.yaml"
    manifest_path.write_text(
        """schema: cuphoton.xpois.image-pairs/v1
pairs:
  - id: changed
    reference: reference.npy
    target: target.npy
"""
    )
    manifest = load_image_pair_manifest(manifest_path)
    with (tmp_path / "target.npy").open("ab") as handle:
        handle.write(b"changed")

    with pytest.raises(RuntimeError, match="changed since manifest load"):
        preflight_image_pair_manifest(manifest, _options())


def test_preflight_ignores_fits_table_before_image(tmp_path) -> None:
    from astropy.io import fits

    reference = tmp_path / "reference.fits"
    target = tmp_path / "target.fits"
    table = fits.BinTableHDU.from_columns(
        [
            fits.Column(
                name="VALUE",
                format="D",
                array=np.arange(3, dtype=np.float64),
            )
        ]
    )
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            table,
            fits.ImageHDU(np.zeros((32, 32), dtype=np.float64)),
        ]
    ).writeto(reference)
    fits.PrimaryHDU(np.zeros((32, 32), dtype=np.float64)).writeto(target)
    manifest_path = tmp_path / "pairs.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "cuphoton.xpois.image-pairs/v1",
                "pairs": [
                    {
                        "id": "table-before-image",
                        "reference": "reference.fits",
                        "target": "target.fits",
                    }
                ],
            }
        )
    )

    manifest = load_image_pair_manifest(manifest_path)

    preflight_image_pair_manifest(manifest, _options())


def test_manifest_rejects_duplicate_json_keys(tmp_path) -> None:
    manifest_path = tmp_path / "pairs.json"
    manifest_path.write_text(
        '{"schema":"cuphoton.xpois.image-pairs/v1",'
        '"schema":"duplicate","pairs":[]}'
    )

    with pytest.raises(ValueError, match="duplicate key"):
        load_image_pair_manifest(manifest_path)


def test_preflight_rejects_shape_mismatch(tmp_path) -> None:
    _write_array(tmp_path / "reference.npy", (32, 32))
    _write_array(tmp_path / "target.npy", (16, 16))
    manifest_path = tmp_path / "pairs.yaml"
    manifest_path.write_text(
        """schema: cuphoton.xpois.image-pairs/v1
pairs:
  - id: mismatch
    reference: reference.npy
    target: target.npy
"""
    )

    manifest = load_image_pair_manifest(manifest_path)
    with pytest.raises(ValueError, match="does not match"):
        preflight_image_pair_manifest(manifest, _options())


def test_batch_options_reject_unknown_backend() -> None:
    with pytest.raises(ValueError, match="unsupported fit backend"):
        _options(backend="unknown")


@pytest.mark.parametrize("sigma", [float("nan"), float("inf"), float("-inf")])
def test_batch_options_reject_nonfinite_basis_sigmas(sigma) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        _options(basis_sigmas=(sigma,))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"auto_stamp_size": 30}, "positive odd integer"),
        ({"auto_stamp_size": True}, "positive odd integer"),
        ({"auto_stamp_count": 0}, "positive integer"),
        ({"auto_stamp_count": True}, "positive integer"),
        ({"auto_peak_percentile": float("nan")}, "finite number"),
        ({"auto_peak_percentile": 101.0}, "finite number"),
    ],
)
def test_batch_options_reject_invalid_auto_stamp_values(
    overrides, message
) -> None:
    with pytest.raises(ValueError, match=message):
        _options(**overrides)


def test_batch_options_round_trip_worker_payload() -> None:
    options = _options(crop_y0=1, crop_x0=2, crop_height=16, crop_width=17)

    assert BatchFitOptions.from_payload(options.to_payload()) == options


def test_image_pair_item_cpu_smoke(tmp_path) -> None:
    reference = np.zeros((64, 64), dtype=np.float64)
    reference[20:40, 20:40] = 1.0
    target = reference.copy()
    reference_path = tmp_path / "reference.npy"
    target_path = tmp_path / "target.npy"
    np.save(reference_path, reference, allow_pickle=False)
    np.save(target_path, target, allow_pickle=False)
    manifest_path = tmp_path / "cpu-pairs.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "cuphoton.xpois.image-pairs/v1",
                "pairs": [
                    {
                        "id": "cpu-pair",
                        "reference": str(reference_path),
                        "target": str(target_path),
                    }
                ],
            }
        )
    )
    item = load_image_pair_manifest(manifest_path).work_items()[0]
    output_dir = tmp_path / "items" / "cpu-pair"
    output_dir.parent.mkdir()

    result = run_image_pair_item(
        item,
        output_dir,
        _options(backend="cpu"),
    )

    assert result["backend"] == "cpu"
    assert result["timings_sec"]["input_read_sec"] >= 0
    assert result["timings_sec"]["solve_sec"] >= 0
    assert result["timings_sec"]["artifact_write_sec"] >= 0
    assert result["wall_sec"]["item_runner"] >= sum(
        result["timings_sec"][field]
        for field in ("input_read_sec", "solve_sec", "artifact_write_sec")
    )
    assert result["wall_sec"]["workflow_before_summary_write"] >= sum(
        result["timings_sec"].values()
    )
    assert "item_runner_sec" not in result["timings_sec"]
    assert "workflow_before_summary_write_sec" not in result["timings_sec"]
    assert (output_dir / "summary.json").is_file()
    assert (output_dir / "artifacts" / "residual.npy").is_file()


def test_item_preserves_workflow_error_and_postcheck_failure(
    monkeypatch, tmp_path
) -> None:
    from cuphoton.xpois import workflows

    reference_path = tmp_path / "reference.npy"
    target_path = tmp_path / "target.npy"
    _write_array(reference_path)
    _write_array(target_path)
    manifest_path = tmp_path / "pairs.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "cuphoton.xpois.image-pairs/v1",
                "pairs": [
                    {
                        "id": "mutated-failure",
                        "reference": str(reference_path),
                        "target": str(target_path),
                    }
                ],
            }
        )
    )
    item = load_image_pair_manifest(manifest_path).work_items()[0]

    def fail_after_mutation(**kwargs):
        del kwargs
        with target_path.open("ab") as handle:
            handle.write(b"changed")
        raise ValueError("injected workflow failure")

    monkeypatch.setattr(
        workflows, "run_constant_kernel_fit", fail_after_mutation
    )

    with pytest.raises(ValueError, match="injected workflow failure") as info:
        run_image_pair_item(
            item,
            tmp_path / "items" / item.item_id,
            _options(backend="cpu"),
        )

    assert any(
        "post-item input identity verification failed" in note
        and "input changed since manifest load" in note
        for note in info.value.__notes__
    )
