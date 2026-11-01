# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

import cuphoton.xbeam.sharding as sharding_module
from cuphoton.xbeam.adapters.radio.output import (
    METADATA_FILENAME,
    SPECTRA_FILENAME,
    OutputRecord,
    XBeamV1Writer,
    iter_xbeam_v1,
)
from cuphoton.xbeam.adapters.radio.swin import SwinWriter
from cuphoton.xbeam.integrity import execution_contract
from cuphoton.xbeam.sharding import (
    load_ordered_shard_outputs,
    merge_native_shards,
    merge_swin_shards,
    plan_time_shards,
    select_time_shard,
)
from cuphoton.xbeam.windowing import CorrelationWindow


def _record(second: int) -> OutputRecord:
    return OutputRecord(
        antenna_a="AA",
        antenna_b="BB",
        mjd=60000,
        seconds_of_day=Decimal(second),
        source_id="source",
        configuration_id="config",
        output_frequency_id="frequency",
        polarization_a="R",
        polarization_b="R",
        weight=1.0,
        u_m=0.0,
        v_m=0.0,
        w_m=0.0,
        spectrum=np.asarray([second + 1j, second + 2j], dtype=np.complex64),
    )


def _vdif_frame(station: str) -> bytes:
    payload = bytes([0xAA]) * 8
    word1 = 40 << 24
    word2 = (32 + len(payload)) // 8
    station_id = (ord(station[0]) << 8) | ord(station[1])
    word3 = station_id | (1 << 26)
    header = np.asarray((0, word1, word2, word3, 0, 0, 0, 0), dtype="<u4")
    return header.tobytes() + payload


def _write_shard(
    root: Path,
    shard,
    seconds: tuple[int, ...],
    *,
    fingerprint: str = "job-fingerprint",
    source_path: str = "/recording/AA.vdif",
    source_mtime_ns: int = 123,
    source_device: int = 1,
    source_inode: int = 2,
) -> Path:
    root.mkdir()
    with XBeamV1Writer(root) as writer:
        for second in seconds:
            writer.write(_record(second))
    (root / "ingest-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "job_fingerprint": fingerprint,
                "profile": "analytic-v1",
                "backend": "numpy",
                "autocorrelation_mode": "average",
                "execution_contract": execution_contract(
                    fingerprint, "analytic-v1", "numpy", "average"
                ),
                "output_integrity": writer.integrity,
                "source_snapshot": [
                    {
                        "datastream_id": 0,
                        "source_index": 0,
                        "path": source_path,
                        "size_bytes": 40,
                        "mtime_ns": source_mtime_ns,
                        "device": source_device,
                        "inode": source_inode,
                    }
                ],
                "shard": shard.to_dict(),
            }
        ),
        encoding="utf-8",
    )
    return root


def test_plan_time_shards_balances_contiguous_integrations() -> None:
    parent = CorrelationWindow("scan", 7, 10)

    shards = plan_time_shards(parent, 3)

    assert tuple(
        (
            item.index,
            item.window.first_integration,
            item.window.integration_count,
        )
        for item in shards
    ) == ((0, 7, 4), (1, 11, 3), (2, 14, 3))
    assert all(item.parent == parent and item.count == 3 for item in shards)
    assert select_time_shard(parent, 3, 1) == shards[1]


@pytest.mark.parametrize("count", (0, 11))
def test_plan_time_shards_rejects_empty_shards(count: int) -> None:
    with pytest.raises(ValueError):
        plan_time_shards(CorrelationWindow("scan", 0, 10), count)


def test_select_time_shard_validates_count_before_bounds() -> None:
    with pytest.raises(TypeError, match="shard_count must be an integer"):
        select_time_shard(
            CorrelationWindow("scan", 0, 10),
            "2",
            0,  # type: ignore[arg-type]
        )


def test_native_shards_validate_and_merge_in_time_order(
    tmp_path: Path,
) -> None:
    plan = plan_time_shards(CorrelationWindow("scan", 0, 4), 2)
    first = _write_shard(tmp_path / "first", plan[0], (0, 1))
    second = _write_shard(tmp_path / "second", plan[1], (2, 3))

    ordered, actual_plan = load_ordered_shard_outputs(
        (second, first), expected_job_fingerprint="job-fingerprint"
    )
    output = tmp_path / "merged"
    output.mkdir()
    count = merge_native_shards(ordered, output)

    assert ordered == (first.resolve(), second.resolve())
    assert actual_plan == plan
    assert count == 4
    records = list(iter_xbeam_v1(output))
    assert [record.seconds_of_day for record in records] == [
        Decimal(index) for index in range(4)
    ]
    for index, record in enumerate(records):
        np.testing.assert_array_equal(
            record.spectrum, _record(index).spectrum
        )


@pytest.mark.parametrize("filename", (METADATA_FILENAME, SPECTRA_FILENAME))
def test_native_shard_merge_rejects_existing_final_artifact(
    tmp_path: Path, filename: str
) -> None:
    plan = plan_time_shards(CorrelationWindow("scan", 0, 1), 1)
    shard = _write_shard(tmp_path / "shard", plan[0], (0,))
    output = tmp_path / "merged"
    output.mkdir()
    destination = output / filename
    destination.write_bytes(b"preserve")

    with pytest.raises(FileExistsError, match="already exists"):
        merge_native_shards((shard,), output)

    assert destination.read_bytes() == b"preserve"


def test_native_shard_merge_rejects_empty_set_without_artifacts(
    tmp_path: Path,
) -> None:
    output = tmp_path / "merged"
    output.mkdir()

    with pytest.raises(ValueError, match="at least one shard"):
        merge_native_shards((), output)

    assert list(output.iterdir()) == []


def test_native_shard_merge_rejects_non_object_metadata(
    tmp_path: Path,
) -> None:
    plan = plan_time_shards(CorrelationWindow("scan", 0, 1), 1)
    shard = _write_shard(tmp_path / "shard", plan[0], (0,))
    (shard / METADATA_FILENAME).write_text("[]\n", encoding="utf-8")
    output = tmp_path / "merged"
    output.mkdir()

    with pytest.raises(ValueError, match="invalid metadata object"):
        merge_native_shards((shard,), output)

    assert not (output / METADATA_FILENAME).exists()
    assert not (output / SPECTRA_FILENAME).exists()
    assert not (output / f"{METADATA_FILENAME}.partial").exists()
    assert not (output / f"{SPECTRA_FILENAME}.partial").exists()


def test_native_shard_merge_rolls_back_partial_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = plan_time_shards(CorrelationWindow("scan", 0, 1), 1)
    shard = _write_shard(tmp_path / "shard", plan[0], (0,))
    output = tmp_path / "merged"
    output.mkdir()
    replace = sharding_module.os.link
    calls = 0

    def fail_second_replace(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("publish failed")
        replace(source, destination)

    monkeypatch.setattr(sharding_module.os, "link", fail_second_replace)

    with pytest.raises(OSError, match="publish failed"):
        merge_native_shards((shard,), output)

    assert not (output / METADATA_FILENAME).exists()
    assert not (output / SPECTRA_FILENAME).exists()
    assert not (output / f"{METADATA_FILENAME}.partial").exists()
    assert not (output / f"{SPECTRA_FILENAME}.partial").exists()


def test_shard_validation_rejects_mixed_jobs(tmp_path: Path) -> None:
    plan = plan_time_shards(CorrelationWindow("scan", 0, 2), 2)
    first = _write_shard(tmp_path / "first", plan[0], (0,))
    second = _write_shard(
        tmp_path / "second", plan[1], (1,), fingerprint="other-job"
    )

    with pytest.raises(ValueError, match="different jobs"):
        load_ordered_shard_outputs((first, second))


def test_shard_validation_rejects_mixed_source_snapshots(
    tmp_path: Path,
) -> None:
    plan = plan_time_shards(CorrelationWindow("scan", 0, 2), 2)
    first = _write_shard(tmp_path / "first", plan[0], (0,))
    second = _write_shard(
        tmp_path / "second", plan[1], (1,), source_mtime_ns=456
    )

    with pytest.raises(ValueError, match="source snapshots"):
        load_ordered_shard_outputs((first, second))


def test_shard_validation_rejects_non_object_manifest(tmp_path: Path) -> None:
    plan = plan_time_shards(CorrelationWindow("scan", 0, 1), 1)
    shard = _write_shard(tmp_path / "shard", plan[0], (0,))
    (shard / "ingest-manifest.json").write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="manifest must be an object"):
        load_ordered_shard_outputs((shard,))


def test_shard_validation_allows_node_local_stat_identity(
    tmp_path: Path,
) -> None:
    plan = plan_time_shards(CorrelationWindow("scan", 0, 2), 2)
    first = _write_shard(tmp_path / "first", plan[0], (0,))
    second = _write_shard(
        tmp_path / "second",
        plan[1],
        (1,),
        source_device=99,
        source_inode=100,
    )

    ordered, actual_plan = load_ordered_shard_outputs((first, second))

    assert ordered == (first.resolve(), second.resolve())
    assert actual_plan == plan


def test_shard_validation_allows_node_local_source_paths(
    tmp_path: Path,
) -> None:
    plan = plan_time_shards(CorrelationWindow("scan", 0, 2), 2)
    first = _write_shard(tmp_path / "first", plan[0], (0,))
    second = _write_shard(
        tmp_path / "second",
        plan[1],
        (1,),
        source_path="/node-local/alias/AA.vdif",
    )

    ordered, actual_plan = load_ordered_shard_outputs((first, second))

    assert ordered == (first.resolve(), second.resolve())
    assert actual_plan == plan


def test_shard_validation_rejects_incomplete_set(tmp_path: Path) -> None:
    plan = plan_time_shards(CorrelationWindow("scan", 0, 2), 2)
    first = _write_shard(tmp_path / "first", plan[0], (0,))

    with pytest.raises(ValueError, match="incomplete"):
        load_ordered_shard_outputs((first,))


def test_shard_validation_bounds_declared_count(tmp_path: Path) -> None:
    plan = plan_time_shards(CorrelationWindow("scan", 0, 1), 1)
    shard = _write_shard(tmp_path / "shard", plan[0], (0,))
    manifest_path = shard / "ingest-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["shard"]["count"] = 10**12
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="incomplete"):
        load_ordered_shard_outputs((shard,))


def test_shard_validation_rejects_out_of_bounds_index(
    tmp_path: Path,
) -> None:
    plan = plan_time_shards(CorrelationWindow("scan", 0, 1), 1)
    shard = _write_shard(tmp_path / "shard", plan[0], (0,))
    manifest_path = shard / "ingest-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["shard"]["index"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid shard index"):
        load_ordered_shard_outputs((shard,))


def test_swin_shards_are_concatenated_in_validated_order(
    tmp_path: Path,
) -> None:
    output = tmp_path / "merged"
    output.mkdir()
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    from dataclasses import replace

    for index, directory in enumerate((first, second)):
        with SwinWriter(
            directory / "output.swin", {"AA": 0, "BB": 1}
        ) as writer:
            writer.write(
                replace(
                    _record(index),
                    source_id=0,
                    configuration_id=0,
                    output_frequency_id=0,
                )
            )
        (directory / "ingest-manifest.json").write_text(
            json.dumps({"output_integrity": writer.integrity})
        )
    expected = (first / "output.swin").read_bytes() + (
        second / "output.swin"
    ).read_bytes()

    size = merge_swin_shards((first, second), output, "output.swin")

    assert size == len(expected)
    assert (output / "output.swin").read_bytes() == expected


def test_swin_shard_merge_rejects_empty_set_without_artifacts(
    tmp_path: Path,
) -> None:
    output = tmp_path / "merged"
    output.mkdir()

    with pytest.raises(ValueError, match="at least one shard"):
        merge_swin_shards((), output, "output.swin")

    assert list(output.iterdir()) == []


def test_swin_shards_reject_inconsistent_frequency_geometry(tmp_path):
    from dataclasses import replace

    shards = [tmp_path / "first", tmp_path / "second"]
    output = tmp_path / "merged"
    output.mkdir()
    for count, directory in zip((2, 3), shards, strict=True):
        directory.mkdir()
        with SwinWriter(
            directory / "output.swin", {"AA": 0, "BB": 1}
        ) as writer:
            writer.write(
                replace(
                    _record(0),
                    source_id=0,
                    configuration_id=0,
                    output_frequency_id=0,
                    spectrum=np.ones(count, dtype=np.complex64),
                )
            )
        (directory / "ingest-manifest.json").write_text(
            json.dumps({"output_integrity": writer.integrity})
        )
    with pytest.raises(ValueError, match="geometry disagrees between shards"):
        merge_swin_shards(shards, output, "output.swin")
    assert list(output.iterdir()) == []


@pytest.mark.parametrize("start", [1, 4])
def test_reject_overlapping_or_gapped_declared_shard_windows(tmp_path, start):
    plan = plan_time_shards(CorrelationWindow("scan", 0, 5), 2)
    first = _write_shard(tmp_path / "first", plan[0], (0, 1, 2))
    # The second planned window starts at3; select an overlap or a gap.
    wrong = replace(plan[1], window=CorrelationWindow("scan", start, 1))
    second = _write_shard(tmp_path / "second", wrong, (start,))
    with pytest.raises(ValueError, match="declared plan"):
        load_ordered_shard_outputs((first, second))
