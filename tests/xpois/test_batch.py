# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os

import numpy as np
import pytest
import yaml

from cuphoton.xpois.batch import (
    BatchFitOptions,
    ImagePairManifest,
    ImagePairSpec,
    InputFileIdentity,
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


def test_public_manifest_objects_preserve_v1_defaults(tmp_path) -> None:
    pair = ImagePairSpec(
        item_id="legacy",
        reference=tmp_path / "reference.npy",
        target=tmp_path / "target.npy",
    )
    manifest = ImagePairManifest(
        source_path=tmp_path / "pairs.json",
        pairs=(pair,),
        input_identities=(),
        sha256="legacy-hash",
    )

    assert manifest.schema == "cuphoton.xpois.image-pairs/v1"
    assert "fit_positions" not in pair.to_payload()


@pytest.mark.parametrize("schema", ["unsupported", None, []])
def test_programmatic_manifest_rejects_unsupported_schema(
    tmp_path, schema
) -> None:
    pair = ImagePairSpec(
        item_id="pair",
        reference=tmp_path / "reference.npy",
        target=tmp_path / "target.npy",
    )

    with pytest.raises(ValueError, match="manifest schema must be one of"):
        ImagePairManifest(
            source_path=tmp_path / "pairs.json",
            pairs=(pair,),
            input_identities=(),
            sha256="unused",
            schema=schema,
        )


@pytest.mark.parametrize("schema", [None, "cuphoton.xpois.image-pairs/v1"])
def test_programmatic_v1_manifest_rejects_fit_positions(
    tmp_path, schema
) -> None:
    pair = ImagePairSpec(
        item_id="positions",
        reference=tmp_path / "reference.npy",
        target=tmp_path / "target.npy",
        fit_positions=tmp_path / "positions.npy",
    )
    kwargs = {} if schema is None else {"schema": schema}

    with pytest.raises(ValueError, match="fit_positions require.*v2"):
        ImagePairManifest(
            source_path=tmp_path / "pairs.json",
            pairs=(pair,),
            input_identities=(),
            sha256="unused",
            **kwargs,
        )


def test_programmatic_v2_manifest_retains_fit_positions(tmp_path) -> None:
    pair = ImagePairSpec(
        item_id="positions",
        reference=tmp_path / "reference.npy",
        target=tmp_path / "target.npy",
        fit_positions=tmp_path / "positions.npy",
    )
    paths = (pair.reference, pair.target, pair.fit_positions)
    for path in paths:
        _write_array(path)
    manifest = ImagePairManifest(
        source_path=tmp_path / "pairs.json",
        pairs=(pair,),
        input_identities=tuple(InputFileIdentity.capture(p) for p in paths),
        sha256="unused",
        schema="cuphoton.xpois.image-pairs/v2",
    )

    payload = manifest.canonical_payload()["pairs"][0]
    item = manifest.work_items()[0]
    assert payload["fit_positions"] == str(pair.fit_positions)
    assert item.payload["fit_positions"] == str(pair.fit_positions)
    assert item.weight_bytes == sum(path.stat().st_size for path in paths)


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
    assert manifest.schema == "cuphoton.xpois.image-pairs/v1"
    assert "fit_positions" not in manifest.canonical_payload()["pairs"][0]
    assert "fit_positions" not in manifest.work_items()[0].payload
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


def test_manifest_hash_includes_captured_input_identity(tmp_path) -> None:
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
    first = load_image_pair_manifest(manifest_path)
    target = tmp_path / "target.npy"
    stat_result = target.stat()
    os.utime(
        target,
        ns=(stat_result.st_atime_ns, stat_result.st_mtime_ns + 1_000_000_000),
    )

    second = load_image_pair_manifest(manifest_path)

    assert first.canonical_payload() == second.canonical_payload()
    assert first.input_identity_payload() != second.input_identity_payload()
    assert first.sha256 != second.sha256


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


def test_manifest_rejects_duplicate_yaml_keys(tmp_path) -> None:
    manifest_path = tmp_path / "pairs.yaml"
    manifest_path.write_text(
        """schema: cuphoton.xpois.image-pairs/v1
schema: duplicate
pairs: []
"""
    )

    with pytest.raises(ValueError, match="duplicate key"):
        load_image_pair_manifest(manifest_path)


def test_manifest_yaml_merge_allows_explicit_override(tmp_path) -> None:
    _write_array(tmp_path / "reference.npy")
    _write_array(tmp_path / "target.npy")
    manifest_path = tmp_path / "pairs.yaml"
    manifest_path.write_text(
        """schema: cuphoton.xpois.image-pairs/v1
pairs:
  - &base
    id: base
    reference: reference.npy
    target: target.npy
  - <<: *base
    id: override
"""
    )

    manifest = load_image_pair_manifest(manifest_path)

    assert [pair.item_id for pair in manifest.pairs] == ["base", "override"]
    assert manifest.pairs[1].reference == tmp_path / "reference.npy"
    assert manifest.pairs[1].target == tmp_path / "target.npy"


@pytest.mark.parametrize(
    ("pairs_yaml", "expected_ids"),
    [
        (
            """  - &defaults
    id: defaults
    reference: reference.npy
    target: target.npy
  - &base
    <<: *defaults
    id: base
    target: override.npy
  - <<: *base
    id: child
""",
            ["defaults", "base", "child"],
        ),
        (
            """  - <<: &base
      <<: &defaults
        id: defaults
        reference: reference.npy
        target: target.npy
      id: base
      target: override.npy
    id: child
  - *base
  - *defaults
""",
            ["child", "base", "defaults"],
        ),
    ],
    ids=["parent-before-child", "child-before-parent"],
)
def test_manifest_yaml_chained_merges_preserve_overrides(
    tmp_path, pairs_yaml, expected_ids
) -> None:
    for name in ("reference.npy", "target.npy", "override.npy"):
        _write_array(tmp_path / name)
    manifest_path = tmp_path / "pairs.yaml"
    manifest_path.write_text(
        "schema: cuphoton.xpois.image-pairs/v1\npairs:\n" + pairs_yaml
    )

    manifest = load_image_pair_manifest(manifest_path)

    assert [pair.item_id for pair in manifest.pairs] == expected_ids
    for pair in manifest.pairs:
        assert pair.reference == tmp_path / "reference.npy"
        target = (
            "target.npy" if pair.item_id == "defaults" else "override.npy"
        )
        assert pair.target == tmp_path / target


def test_manifest_rejects_duplicate_yaml_keys_across_merged_mappings(
    tmp_path,
) -> None:
    manifest_path = tmp_path / "pairs.yaml"
    manifest_path.write_text(
        """schema: cuphoton.xpois.image-pairs/v1
pairs:
  - &first
    id: first
    reference: reference.npy
    target: target.npy
  - &second
    id: second
    reference: reference.npy
    target: target.npy
  - <<: [*first, *second]
    id: merged
"""
    )

    with pytest.raises(ValueError, match="across merged mappings"):
        load_image_pair_manifest(manifest_path)


def test_yaml_merge_allows_disjoint_merged_mappings() -> None:
    from cuphoton.xpois.batch import _UniqueKeySafeLoader

    document = """
first: &first {reference: reference.npy}
second: &second {target: target.npy}
merged:
  <<: [*first, *second]
  id: merged
"""

    loaded = yaml.load(document, Loader=_UniqueKeySafeLoader)

    assert loaded["merged"] == {
        "reference": "reference.npy",
        "target": "target.npy",
        "id": "merged",
    }


def test_manifest_rejects_duplicate_yaml_merge_keys(tmp_path) -> None:
    manifest_path = tmp_path / "pairs.yaml"
    manifest_path.write_text(
        """schema: cuphoton.xpois.image-pairs/v1
pairs:
  - &base
    id: base
    reference: reference.npy
    target: target.npy
  - <<: *base
    <<: *base
"""
    )

    with pytest.raises(ValueError, match="duplicate merge key"):
        load_image_pair_manifest(manifest_path)


def test_manifest_rejects_duplicate_yaml_keys_inside_merge(tmp_path) -> None:
    manifest_path = tmp_path / "pairs.yaml"
    manifest_path.write_text(
        """schema: cuphoton.xpois.image-pairs/v1
pairs:
  - <<: &base
      id: first
      id: duplicate
      reference: reference.npy
      target: target.npy
"""
    )

    with pytest.raises(ValueError, match="duplicate key 'id'"):
        load_image_pair_manifest(manifest_path)


def test_v1_manifest_rejects_fit_positions_field(tmp_path) -> None:
    _write_array(tmp_path / "reference.npy")
    _write_array(tmp_path / "target.npy")
    np.save(
        tmp_path / "positions.npy",
        np.asarray([[4, 5]], dtype=np.int32),
        allow_pickle=False,
    )
    manifest_path = tmp_path / "pairs.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "cuphoton.xpois.image-pairs/v1",
                "pairs": [
                    {
                        "id": "legacy",
                        "reference": "reference.npy",
                        "target": "target.npy",
                        "fit_positions": "positions.npy",
                    }
                ],
            }
        )
    )

    with pytest.raises(ValueError, match="unknown field.*fit_positions"):
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


@pytest.mark.parametrize("full_image_mask", [False, True])
def test_batch_cropped_fit_accepts_full_or_cropped_mask(
    tmp_path, full_image_mask
) -> None:
    reference = np.random.default_rng(20260921).normal(size=(64, 64))
    np.save(tmp_path / "reference.npy", reference)
    np.save(tmp_path / "target.npy", reference)
    full_mask = np.zeros(reference.shape, dtype=bool)
    full_mask[20:40, 20:40] = True
    full_mask[25, 26] = False
    cropped_mask = full_mask[8:56, 8:56]
    np.save(
        tmp_path / "fit-mask.npy",
        full_mask if full_image_mask else cropped_mask,
    )
    manifest_path = tmp_path / "pairs.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "cuphoton.xpois.image-pairs/v1",
                "pairs": [
                    {
                        "id": "cropped",
                        "reference": "reference.npy",
                        "target": "target.npy",
                        "fit_mask": "fit-mask.npy",
                    }
                ],
            }
        )
    )
    manifest = load_image_pair_manifest(manifest_path)
    options = _options(
        backend="cpu",
        crop_y0=8,
        crop_x0=8,
        crop_height=48,
        crop_width=48,
    )

    preflight_image_pair_manifest(manifest, options)
    output_dir = tmp_path / "items" / "cropped"
    output_dir.parent.mkdir()
    run_image_pair_item(manifest.work_items()[0], output_dir, options)

    np.testing.assert_array_equal(
        np.load(output_dir / "artifacts" / "fit_mask.npy"), cropped_mask
    )
    assert np.load(output_dir / "artifacts" / "residual.npy").shape == (
        48,
        48,
    )


@pytest.mark.parametrize("mask_policy", ["strict", "masklite"])
@pytest.mark.parametrize("missing_mask", ["reference_mask", "target_mask"])
def test_preflight_rejects_missing_npy_image_mask(
    tmp_path, mask_policy, missing_mask, monkeypatch
) -> None:
    from cuphoton.xpois import batch

    for name in ("reference.npy", "target.npy", "mask.npy"):
        _write_array(tmp_path / name)
    pair = {
        "id": "maskless",
        "reference": "reference.npy",
        "target": "target.npy",
        "reference_mask": "mask.npy",
        "target_mask": "mask.npy",
    }
    del pair[missing_mask]
    manifest_path = tmp_path / "pairs.json"
    manifest_path.write_text(
        json.dumps(
            {"schema": "cuphoton.xpois.image-pairs/v1", "pairs": [pair]}
        )
    )
    manifest = load_image_pair_manifest(manifest_path)
    mask_probes = []
    probe = batch._probe_array_shape

    def record_probe(path, hdu, *, kind):
        if kind == "mask":
            mask_probes.append(path.name)
        return probe(path, hdu, kind=kind)

    monkeypatch.setattr(batch, "_probe_array_shape", record_probe)
    with pytest.raises(ValueError, match=f"{missing_mask}.*required.*NPY"):
        preflight_image_pair_manifest(
            manifest, _options(mask_policy=mask_policy)
        )
    assert set(mask_probes) <= {"mask.npy"}


def test_batch_options_reject_unknown_backend() -> None:
    with pytest.raises(ValueError, match="unsupported fit backend"):
        _options(backend="unknown")


def test_batch_options_reject_unknown_solver() -> None:
    with pytest.raises(ValueError, match="unsupported solver"):
        _options(solver="unknown")


@pytest.mark.parametrize("backend", ["cutile", "numba-cuda"])
def test_batch_options_reject_unsupported_spatial_backend(backend) -> None:
    with pytest.raises(ValueError, match="supports only auto, cpu, and cupy"):
        _options(solver="spatial-als", backend=backend)


@pytest.mark.parametrize(
    "overrides",
    [
        {"spatial_degree": 2},
        {"als_iterations": 30},
        {"als_tolerance": 1e-6},
        {"als_regularization": 1e-6},
    ],
)
def test_batch_options_reject_spatial_option_for_constant_solver(
    overrides,
) -> None:
    with pytest.raises(ValueError, match="require solver='spatial-als'"):
        _options(**overrides)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"spatial_degree": -1}, "non-negative"),
        ({"spatial_degree": True}, "must be an integer"),
        ({"als_iterations": 0}, "must be positive"),
        ({"als_iterations": True}, "must be an integer"),
        ({"als_tolerance": float("nan")}, "must be finite"),
        ({"als_regularization": -1.0}, "must be non-negative"),
    ],
)
def test_batch_options_reject_invalid_spatial_values(
    overrides, message
) -> None:
    with pytest.raises(ValueError, match=message):
        _options(solver="spatial-als", **overrides)


@pytest.mark.parametrize("sigma", [float("nan"), float("inf"), float("-inf")])
def test_batch_options_reject_nonfinite_basis_sigmas(sigma) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        _options(basis_sigmas=(sigma,))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"auto_stamp_size": 30}, "positive odd integer"),
        ({"auto_stamp_size": True}, "positive odd integer"),
        ({"auto_stamp_mask": True, "auto_stamp_size": 1}, "at least 3"),
        ({"auto_stamp_count": 0}, "positive integer"),
        ({"auto_stamp_count": True}, "positive integer"),
        ({"auto_peak_percentile": float("nan")}, "finite number"),
        ({"auto_peak_percentile": 101.0}, "finite number"),
        ({"auto_stamp_mask": True, "auto_peak_percentile": 0}, "strictly"),
        ({"auto_stamp_mask": True, "auto_peak_percentile": 100}, "strictly"),
    ],
)
def test_batch_options_reject_invalid_auto_stamp_values(
    overrides, message
) -> None:
    with pytest.raises(ValueError, match=message):
        _options(**overrides)


@pytest.mark.parametrize("enabled, size", [(False, 1), (True, 3)])
def test_batch_options_accept_minimum_stamp_size(enabled, size) -> None:
    options = _options(auto_stamp_mask=enabled, auto_stamp_size=size)

    assert BatchFitOptions.from_payload(options.to_payload()) == options


def test_batch_options_round_trip_worker_payload() -> None:
    options = _options(
        crop_y0=1,
        crop_x0=2,
        crop_height=16,
        crop_width=17,
        solver="spatial-als",
        spatial_degree=3,
        als_iterations=17,
        als_tolerance=2e-7,
        als_regularization=4e-5,
    )

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


@pytest.mark.parametrize("use_defaults", [False, True])
def test_image_pair_item_spatial_als_cpu_smoke(
    tmp_path, use_defaults
) -> None:
    rng = np.random.default_rng(20260905)
    reference = rng.normal(size=(41, 43))
    coordinates = np.arange(9, dtype=np.float64) - 4.0
    line = np.exp(-(coordinates**2) / (2.0 * 1.5**2))
    line /= line.sum()
    kernel = np.outer(line, line)
    patches = np.lib.stride_tricks.sliding_window_view(reference, (9, 9))
    target = np.zeros_like(reference)
    target[4:-4, 4:-4] = np.einsum(
        "yxvu,vu->yx",
        patches,
        kernel[::-1, ::-1],
        optimize=True,
    )
    reference_path = tmp_path / "reference.npy"
    target_path = tmp_path / "target.npy"
    fit_positions_path = tmp_path / "fit-positions.npy"
    np.save(reference_path, reference, allow_pickle=False)
    np.save(target_path, target, allow_pickle=False)
    crop_shape = (39, 40)
    all_positions = np.argwhere(np.ones(crop_shape, dtype=bool))
    interior = all_positions[
        (all_positions[:, 0] >= 4)
        & (all_positions[:, 0] < crop_shape[0] - 4)
        & (all_positions[:, 1] >= 4)
        & (all_positions[:, 1] < crop_shape[1] - 4)
    ][::4]
    fit_positions = np.concatenate((interior, interior[:13])).astype(np.int32)
    np.save(fit_positions_path, fit_positions, allow_pickle=False)
    manifest_path = tmp_path / "spatial-pairs.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "cuphoton.xpois.image-pairs/v2",
                "pairs": [
                    {
                        "id": "spatial-pair",
                        "reference": str(reference_path),
                        "target": str(target_path),
                        "fit_positions": str(fit_positions_path),
                    }
                ],
            }
        )
    )
    manifest = load_image_pair_manifest(manifest_path)
    with pytest.raises(ValueError, match="require solver='spatial-als'"):
        preflight_image_pair_manifest(manifest, _options(backend="cpu"))
    options = _options(
        backend="cpu",
        solver="spatial-als",
        **(
            {}
            if use_defaults
            else {
                "spatial_degree": 1,
                "als_iterations": 3,
                "als_tolerance": 0.0,
                "als_regularization": 4e-5,
            }
        ),
        crop_y0=1,
        crop_x0=2,
        crop_height=crop_shape[0],
        crop_width=crop_shape[1],
    )
    preflight_image_pair_manifest(manifest, options)
    item = manifest.work_items()[0]
    output_dir = tmp_path / "items" / "spatial-pair"
    output_dir.parent.mkdir()

    result = run_image_pair_item(
        item,
        output_dir,
        options,
    )

    summary = json.loads((output_dir / "summary.json").read_text())
    assert result["solver"] == "spatial-als"
    assert result["backend"] == "cpu"
    assert summary["solver"] == "spatial-als"
    assert summary["image_shape"] == list(crop_shape)
    assert summary["crop"] == {
        "y0": 1,
        "x0": 2,
        "height": crop_shape[0],
        "width": crop_shape[1],
    }
    assert summary["spatial_als"]["spatial_degree"] == (
        2 if use_defaults else 1
    )
    assert summary["spatial_als"]["max_iterations"] == (
        30 if use_defaults else 3
    )
    assert summary["spatial_als"]["tolerance"] == (
        1e-6 if use_defaults else 0.0
    )
    assert summary["spatial_als"]["regularization"] == pytest.approx(
        1e-6 if use_defaults else 4e-5
    )
    assert summary["fit_region"] == {
        "kind": "explicit_positions",
        "coordinate_order": "y,x",
        "duplicates_preserved": True,
        "row_count": fit_positions.shape[0],
        "pixel_count": interior.shape[0],
        "duplicate_row_count": 13,
    }
    assert (output_dir / "artifacts" / "kernel_center.npy").is_file()
    assert (
        output_dir / "artifacts" / "horizontal_coefficients.npy"
    ).is_file()
    np.testing.assert_array_equal(
        np.load(
            output_dir / "artifacts" / "fit_positions.npy",
            allow_pickle=False,
        ),
        fit_positions,
    )


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
