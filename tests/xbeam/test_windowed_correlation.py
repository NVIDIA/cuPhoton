# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from cuphoton.core.cli import run_component
from cuphoton.xbeam.adapters.radio.engine import (
    correlate,
    prepare_correlation,
)
from cuphoton.xbeam.adapters.radio.models import (
    Baseline,
    BaselineProduct,
    ClockPolynomial,
    Datastream,
    DelayCoordinateModel,
    Frequency,
    Job,
    MJDTime,
    RecordedBand,
    SamplingKind,
    Scan,
    Sideband,
    Station,
    dump_job,
)
from cuphoton.xbeam.adapters.radio.output import iter_xbeam_v1
from cuphoton.xbeam.commands import _job_fingerprint
from cuphoton.xbeam.errors import ConfigurationError
from cuphoton.xbeam.windowing import CorrelationWindow


def _frame(station: str) -> bytes:
    payload = bytes([0xAA]) * 8
    word0 = 0
    word1 = 40 << 24  # 2020-01-01
    word2 = (32 + len(payload)) // 8
    station_id = (ord(station[0]) << 8) | ord(station[1])
    word3 = station_id | ((2 - 1) << 26)
    return (
        np.asarray(
            (word0, word1, word2, word3, 0, 0, 0, 0), dtype="<u4"
        ).tobytes()
        + payload
    )


def _job(tmp_path: Path, *, scan_duration: str = "1") -> Job:
    paths = []
    for name in ("AA", "BB"):
        path = tmp_path / f"{name}.vdif"
        path.write_bytes(_frame(name))
        paths.append(path)

    epoch = MJDTime(58849, Decimal(0))
    stations = tuple(
        Station(index, name, ClockPolynomial(epoch, (0.0,)))
        for index, name in enumerate(("AA", "BB"))
    )
    frequency = Frequency(0, 0.0, 16.0, Sideband.USB, 4)
    streams = tuple(
        Datastream(
            index,
            index,
            SamplingKind.REAL,
            2,
            40,
            (str(paths[index]),),
            (RecordedBand(0, 0, "R"),),
            sample_rate_hz=32.0,
        )
        for index in range(2)
    )

    def zero_model(station_id: int) -> DelayCoordinateModel:
        return DelayCoordinateModel(
            station_id,
            Decimal(0),
            (0.0,),
            (float(station_id),),
            (0.0,),
            (0.0,),
        )

    return Job(
        start_time=epoch,
        execution_duration_s=Decimal(1),
        integration_duration_s=Decimal("0.5"),
        subintegration_duration_s=Decimal("0.25"),
        fringe_approximation_order=1,
        stations=stations,
        frequencies=(frequency,),
        datastreams=streams,
        baselines=(Baseline(0, 0, 1, (BaselineProduct(0, 0, 0),)),),
        scans=(
            Scan(
                "scan",
                Decimal(0),
                Decimal(scan_duration),
                "0",
                "0",
                (zero_model(0), zero_model(1)),
            ),
        ),
        write_autocorrelations=True,
    )


def _assert_same_record(actual, expected) -> None:
    assert replace(actual, spectrum=None) == replace(expected, spectrum=None)
    np.testing.assert_array_equal(actual.spectrum, expected.spectrum)


def test_windowed_output_matches_corresponding_unwindowed_group(tmp_path):
    job = _job(tmp_path)

    unwindowed = list(correlate(job, backend="numpy", assume_ordered=True))
    windowed = list(
        correlate(
            job,
            backend="numpy",
            window=CorrelationWindow("scan", 1, 1),
            assume_ordered=True,
        )
    )

    assert len(unwindowed) == 6
    assert len(windowed) == 3
    for actual, expected in zip(windowed, unwindowed[3:], strict=True):
        _assert_same_record(actual, expected)


def test_segmented_model_matches_equivalent_scan_boundary_split(tmp_path):
    job = _job(tmp_path)
    first_models = job.scans[0].station_models
    second_models = tuple(
        replace(
            model,
            epoch_offset_s=Decimal("0.5"),
            delay_coefficients_s=(
                0.0 if model.station_id == 0 else 1.0 / 32.0,
            ),
            u_coefficients_m=(float(model.station_id + 10),),
        )
        for model in first_models
    )
    segmented_scan = replace(
        job.scans[0], station_models=first_models + second_models
    )
    segmented = replace(job, scans=(segmented_scan,))
    split = replace(
        job,
        scans=(
            replace(
                job.scans[0],
                identifier="first",
                duration_s=Decimal("0.5"),
                station_models=first_models,
            ),
            replace(
                job.scans[0],
                identifier="second",
                start_offset_s=Decimal("0.5"),
                duration_s=Decimal("0.5"),
                station_models=second_models,
            ),
        ),
    )

    actual = list(correlate(segmented, backend="numpy", assume_ordered=True))
    expected = list(correlate(split, backend="numpy", assume_ordered=True))

    assert len(actual) == len(expected) == 6
    for actual_record, expected_record in zip(actual, expected, strict=True):
        _assert_same_record(actual_record, expected_record)


def test_prepare_correlation_preflights_missing_source_eagerly(tmp_path):
    job = _job(tmp_path)
    missing_streams = tuple(
        replace(
            stream,
            source_files=(
                str(tmp_path / f"missing-{stream.identifier}.vdif"),
            ),
        )
        for stream in job.datastreams
    )

    with pytest.raises((ConfigurationError, OSError)):
        prepare_correlation(
            replace(job, datastreams=missing_streams),
            backend="numpy",
            window=CorrelationWindow("scan", 0, 1),
            assume_ordered=True,
        )


def test_job_fingerprint_ignores_node_local_source_paths(tmp_path):
    job = _job(tmp_path)
    aliased = replace(
        job,
        datastreams=tuple(
            replace(
                datastream,
                source_files=tuple(
                    f"/node-local/{datastream.identifier}/{source_index}"
                    for source_index, _source in enumerate(
                        datastream.source_files
                    )
                ),
            )
            for datastream in job.datastreams
        ),
    )

    assert _job_fingerprint(aliased) == _job_fingerprint(job)


def test_job_fingerprint_preserves_topology_and_configuration(tmp_path):
    job = _job(tmp_path)
    first, *remaining = job.datastreams
    changed_topology = replace(
        job,
        datastreams=(
            replace(first, source_files=(*first.source_files, "extra.vdif")),
            *remaining,
        ),
    )
    changed_configuration = replace(
        job, integration_duration_s=Decimal("0.25")
    )

    fingerprint = _job_fingerprint(job)
    assert _job_fingerprint(changed_topology) != fingerprint
    assert _job_fingerprint(changed_configuration) != fingerprint


def test_job_first_cli_is_rejected(tmp_path, capsys):
    job_path = tmp_path / "job.json"
    output = tmp_path / "output"
    dump_job(_job(tmp_path), job_path)

    status = run_component("xbeam", [str(job_path), "--output", str(output)])

    assert status == 2
    assert capsys.readouterr().err == (
        f"ERROR: Unknown command '{job_path}'\n"
    )
    assert not output.exists()


def test_distributed_time_shards_merge_to_unsharded_output(tmp_path, capsys):
    job = _job(tmp_path)
    job_path = tmp_path / "job.json"
    dump_job(job, job_path)
    shard_outputs = (tmp_path / "shard-0", tmp_path / "shard-1")

    for index in (1, 0):
        assert (
            run_component(
                "xbeam",
                [
                    "correlate",
                    str(job_path),
                    "--output",
                    str(shard_outputs[index]),
                    "--backend",
                    "numpy",
                    "--scan",
                    "scan",
                    "--first-integration",
                    "0",
                    "--integration-count",
                    "2",
                    "--shard-count",
                    "2",
                    "--shard-index",
                    str(index),
                    "--assume-ordered",
                ],
            )
            == 0
        )
    selected_manifest = json.loads(
        (shard_outputs[1] / "ingest-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert selected_manifest["shard"]["index"] == 1
    assert selected_manifest["shard"]["window"] == {
        "scan_id": "scan",
        "first_integration": 1,
        "integration_count": 1,
    }

    merged = tmp_path / "merged"
    assert (
        run_component(
            "xbeam",
            [
                "merge-shards",
                str(job_path),
                str(shard_outputs[1]),
                str(shard_outputs[0]),
                "--output",
                str(merged),
            ],
        )
        == 0
    )
    assert capsys.readouterr().err == ""
    actual = list(iter_xbeam_v1(merged))
    unsharded = tmp_path / "unsharded"
    assert (
        run_component(
            "xbeam",
            [
                "correlate",
                str(job_path),
                "--output",
                str(unsharded),
                "--backend",
                "numpy",
                "--assume-ordered",
            ],
        )
        == 0
    )
    expected = list(iter_xbeam_v1(unsharded))
    assert len(actual) == len(expected) == 6
    for actual_record, expected_record in zip(actual, expected, strict=True):
        _assert_same_record(actual_record, expected_record)
    merged_manifest = json.loads(
        (merged / "ingest-manifest.json").read_text(encoding="utf-8")
    )
    assert merged_manifest["status"] == "complete"
    assert merged_manifest["mode"] == "distributed-assembly"
    assert merged_manifest["assembled_metric"] == 6
    assert merged_manifest["assembled_metric_unit"] == "records"


def test_distributed_shards_reject_mixed_source_snapshots(tmp_path, capsys):
    job = _job(tmp_path)
    job_path = tmp_path / "job.json"
    dump_job(job, job_path)
    common = [
        "correlate",
        str(job_path),
        "--backend",
        "numpy",
        "--scan",
        "scan",
        "--first-integration",
        "0",
        "--integration-count",
        "2",
        "--shard-count",
        "2",
        "--assume-ordered",
    ]
    first = tmp_path / "shard-0"
    second = tmp_path / "shard-1"
    assert (
        run_component(
            "xbeam", [*common, "--shard-index", "0", "--output", str(first)]
        )
        == 0
    )

    source = Path(job.datastreams[0].source_files[0])
    before = source.stat()
    changed = bytearray(source.read_bytes())
    changed[-1] ^= 1
    source.write_bytes(changed)
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000))

    assert (
        run_component(
            "xbeam", [*common, "--shard-index", "1", "--output", str(second)]
        )
        == 0
    )
    merged = tmp_path / "merged"
    assert (
        run_component(
            "xbeam",
            [
                "merge-shards",
                str(job_path),
                str(first),
                str(second),
                "--output",
                str(merged),
            ],
        )
        == 1
    )
    assert "different recording source snapshots" in capsys.readouterr().err
    assert not merged.exists()


def test_distributed_merge_rejects_changed_local_job_source(tmp_path, capsys):
    job = _job(tmp_path)
    job_path = tmp_path / "job.json"
    dump_job(job, job_path)
    common = [
        "correlate",
        str(job_path),
        "--backend",
        "numpy",
        "--scan",
        "scan",
        "--first-integration",
        "0",
        "--integration-count",
        "2",
        "--shard-count",
        "2",
        "--assume-ordered",
    ]
    shards = (tmp_path / "shard-0", tmp_path / "shard-1")
    for index, shard in enumerate(shards):
        assert (
            run_component(
                "xbeam",
                [
                    *common,
                    "--shard-index",
                    str(index),
                    "--output",
                    str(shard),
                ],
            )
            == 0
        )

    source = Path(job.datastreams[0].source_files[0])
    before = source.stat()
    changed = bytearray(source.read_bytes())
    changed[-1] ^= 1
    source.write_bytes(changed)
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000))

    merged = tmp_path / "merged"
    assert (
        run_component(
            "xbeam",
            [
                "merge-shards",
                str(job_path),
                *map(str, shards),
                "--output",
                str(merged),
            ],
        )
        == 1
    )
    assert "do not match the assembly job" in capsys.readouterr().err
    assert not merged.exists()


@pytest.mark.parametrize(
    "shard_args",
    (
        ("--shard-count", "2"),
        ("--shard-index", "0"),
        ("--shard-count", "2", "--shard-index", "0"),
    ),
)
def test_sharding_requires_complete_worker_and_window_selection(
    tmp_path, capsys, shard_args
):
    job_path = tmp_path / "job.json"
    output = tmp_path / "output"
    dump_job(_job(tmp_path), job_path)

    status = run_component(
        "xbeam",
        [
            "correlate",
            str(job_path),
            "--output",
            str(output),
            *shard_args,
        ],
    )

    assert status == 1
    assert capsys.readouterr().err.startswith("ERROR:")
    assert not output.exists()


@pytest.mark.parametrize(
    ("selectors", "scan_duration"),
    [
        (("--scan", "scan"), "1"),
        (
            (
                "--scan",
                "scan",
                "--first-integration",
                "-1",
                "--integration-count",
                "1",
            ),
            "1",
        ),
        (
            (
                "--scan",
                "scan",
                "--first-integration",
                "0",
                "--integration-count",
                "0",
            ),
            "1",
        ),
        (
            (
                "--scan",
                "scan",
                "--first-integration",
                "1",
                "--integration-count",
                "1",
            ),
            "0.75",
        ),
    ],
    ids=("incomplete-selector", "negative-index", "zero-count", "partial"),
)
def test_invalid_or_partial_cli_window_is_configuration_failure(
    tmp_path, capsys, selectors, scan_duration
):
    job_path = tmp_path / "job.json"
    output = tmp_path / "output"
    dump_job(_job(tmp_path, scan_duration=scan_duration), job_path)

    status = run_component(
        "xbeam",
        [
            "correlate",
            str(job_path),
            "--output",
            str(output),
            *selectors,
        ],
    )

    assert status == 1
    assert capsys.readouterr().err.startswith("ERROR:")
    assert not output.exists()
