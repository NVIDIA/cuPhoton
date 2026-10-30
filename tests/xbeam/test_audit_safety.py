# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Validation and bounded-memory behavior."""

import json
import os
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from cuphoton.xbeam.adapters.radio import gpu_execution
from cuphoton.xbeam.adapters.vdif import recording, vdif_sidecar
from cuphoton.xbeam.adapters.vdif.vdif_index import (
    VdifIndexError,
    VdifTimestampKey,
)
from cuphoton.xbeam.adapters.vdif.vdif_sidecar_builder import (
    build_vdif_binary_index,
)
from cuphoton.xbeam.integrity import execution_contract
from cuphoton.xbeam.sharding import (
    load_ordered_shard_outputs,
    merge_native_shards,
    merge_swin_shards,
    plan_time_shards,
)
from cuphoton.xbeam.validation import validate_job
from cuphoton.xbeam.windowing import CorrelationWindow
from tests.xbeam.test_engine import _job
from tests.xbeam.test_sharding import _write_shard
from tests.xbeam.test_vdif_sidecar import _frame


def _source(tmp_path):
    path = tmp_path / "source.vdif"
    path.write_bytes(_frame(0) + _frame(1) + _frame(2))
    return path


def _read(path):
    return vdif_sidecar.read_vdif_binary_frames(
        path, VdifTimestampKey(40, 17, 0), VdifTimestampKey(40, 17, 100)
    )


def test_interior_change_rejects_stale_sidecar(tmp_path):
    path = _source(tmp_path)
    build_vdif_binary_index(path, backend="posix")
    _read(path)
    before = path.stat()
    with path.open("r+b") as stream:
        stream.seek(40)
        stream.write(_frame(99))
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1000000))
    with pytest.raises(VdifIndexError, match="provenance"):
        _read(path)


def test_sidecar_replacement_preserving_size_mtime_invalidates_mmap(tmp_path):
    path = _source(tmp_path)
    summary = build_vdif_binary_index(path, backend="posix")
    before = summary.index_path.stat()
    old = _read(path)[0]
    with path.open("r+b") as stream:
        stream.seek(40)
        stream.write(_frame(99))
    build_vdif_binary_index(path, backend="posix", force=True)
    os.utime(summary.index_path, ns=(before.st_atime_ns, before.st_mtime_ns))
    new = _read(path)[0]
    assert old.tolist() != new.tolist()
    assert new[-1] == VdifTimestampKey(40, 17, 99).packed


def test_legacy_provenance_requires_rebuild(tmp_path):
    path = _source(tmp_path)
    summary = build_vdif_binary_index(path, backend="posix")
    document = vdif_sidecar._read_document(summary.index_path)
    del document["source_provenance"]
    with summary.index_path.open("r+b") as stream:
        stream.write(vdif_sidecar.encode_binary_index_header(document))
    vdif_sidecar._read_document_cached.cache_clear()
    vdif_sidecar._validate_cached.cache_clear()
    with pytest.raises(VdifIndexError, match="rebuild"):
        _read(path)


def test_unordered_query_once_for_many_ranges(tmp_path, monkeypatch):
    path = tmp_path / "unordered.vdif"
    path.write_bytes(
        b"".join(_frame(n) for j in range(20) for n in (j, 100 + j))
    )
    build_vdif_binary_index(path, backend="posix")
    bounds = (VdifTimestampKey(40, 17, 0), VdifTimestampKey(40, 17, 20))

    def query_bounds(windows, epochs):
        assert epochs == (40,)
        return (bounds,)

    monkeypatch.setattr(recording, "_sidecar_query_bounds", query_bounds)
    calls = []
    read = recording.read_vdif_binary_frames

    def counted(*args, **kwargs):
        result = read(*args, **kwargs)
        calls.append(len(result[0]))
        return result

    monkeypatch.setattr(recording, "read_vdif_binary_frames", counted)
    ranges = tuple(
        recording.VdifSourceRange(path, j * 2, 1) for j in range(20)
    )
    # Include overlap: it must emit independently without querying again.
    frames = recording._binary_sidecar_frames(
        SimpleNamespace(frame_size_bytes=40), ranges + ranges[:1], {}, None
    )
    assert calls == [20]
    np.testing.assert_array_equal(
        frames.physical_frames, list(range(0, 40, 2)) + [0]
    )
    np.testing.assert_array_equal(frames.source_indices, range(21))


def test_replacement_between_validation_and_cached_mapping(
    tmp_path, monkeypatch
):
    path = _source(tmp_path)
    summary = build_vdif_binary_index(path, backend="posix")
    _read(path)
    replacement = tmp_path / "replacement"
    replacement.write_bytes(summary.index_path.read_bytes())
    os.replace(replacement, summary.index_path)
    monkeypatch.setattr(
        vdif_sidecar, "validate_vdif_binary_index", lambda *a, **k: summary
    )
    with pytest.raises(VdifIndexError, match="changed before mapping"):
        _read(path)


def test_mutation_during_build_preserves_existing_sidecar(
    tmp_path, monkeypatch
):
    from cuphoton.xbeam.adapters.vdif import vdif_sidecar_builder as builder

    path = _source(tmp_path)
    summary = build_vdif_binary_index(path, backend="posix")
    original = summary.index_path.read_bytes()
    write = builder._write_cpu_index

    def mutate(*args, **kwargs):
        write(*args, **kwargs)
        stat = path.stat()
        with path.open("r+b") as stream:
            stream.seek(40)
            stream.write(_frame(99))
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000))

    monkeypatch.setattr(builder, "_write_cpu_index", mutate)
    with pytest.raises(OSError, match="changed while building"):
        build_vdif_binary_index(path, backend="posix", force=True)
    assert summary.index_path.read_bytes() == original


@pytest.mark.parametrize("clock", [False, True])
def test_order_two_rejects_unsupported_higher_terms(tmp_path, clock):
    job = replace(_job(tmp_path), fringe_approximation_order=2)
    if clock:
        station = job.stations[0]
        station = replace(
            station,
            clock=replace(
                station.clock, coefficients_s=(0.0, 0.0, 0.0, 1e-6)
            ),
        )
        job = replace(job, stations=(station,) + job.stations[1:])
    else:
        scan = job.scans[0]
        model = replace(
            scan.station_models[0], delay_coefficients_s=(0.0, 0.0, 0.0, 1e-6)
        )
        job = replace(
            job,
            scans=(
                replace(
                    scan, station_models=(model,) + scan.station_models[1:]
                ),
            ),
        )
    with pytest.raises(ValueError, match="not equivalent"):
        validate_job(job)


def test_tiling_retains_one_scratch_for_100_tiles(monkeypatch):
    seen = []

    def compute(*args, visibility_layout=None):
        arena = args[8]
        seen.append(id(arena))
        arena.spectra.fill(1)
        arena.weights.fill(1)
        return [object()]

    monkeypatch.setattr(gpu_execution, "_run_chunked_gpu", compute)
    _, arena, sizes, retries = gpu_execution._run_subintegration_tiled_gpu(
        None,
        np,
        (0, None, range(100)),
        {},
        {},
        {},
        {},
        {},
        1,
        8,
        None,
        1024,
        None,
        None,
        1,
    )
    assert len(set(seen)) == len(arena.retained) == 1
    assert sizes == [1] * 100 and retries == 0
    np.testing.assert_array_equal(arena.spectra, 100)


def test_tiled_retry_resets_failed_scratch(monkeypatch):
    class OOM(RuntimeError):
        pass

    xp = SimpleNamespace(
        empty=np.empty,
        uint8=np.uint8,
        complex64=np.complex64,
        float64=np.float64,
        cuda=SimpleNamespace(memory=SimpleNamespace(OutOfMemoryError=OOM)),
        get_default_memory_pool=lambda: SimpleNamespace(
            free_all_blocks=lambda: None
        ),
    )
    calls = []

    def compute(*args, visibility_layout=None):
        selected = args[2][0][2]
        arena = args[8]
        calls.append(id(arena))
        arena.spectra.fill(1000)
        arena.weights.fill(1000)
        if len(selected) > 4:
            raise OOM()
        arena.spectra.fill(len(selected))
        arena.weights.fill(len(selected))
        return [object()]

    monkeypatch.setattr(gpu_execution, "_run_chunked_gpu", compute)
    _, arena, sizes, retries = gpu_execution._run_subintegration_tiled_gpu(
        None,
        xp,
        (0, None, range(8)),
        {},
        {},
        {},
        {},
        {},
        1,
        8,
        None,
        1024,
        None,
        None,
        8,
    )
    assert len(set(calls)) == 1 and retries == 1 and sizes == [4, 4]
    np.testing.assert_array_equal(arena.spectra, 8)
    np.testing.assert_array_equal(arena.weights, 8)


def test_mixed_autocorrelation_modes_rejected(tmp_path):
    plan = plan_time_shards(CorrelationWindow("scan", 0, 2), 2)
    shards = [
        _write_shard(tmp_path / str(i), part, (i,))
        for i, part in enumerate(plan)
    ]
    path = shards[1] / "ingest-manifest.json"
    document = json.loads(path.read_text())
    document["autocorrelation_mode"] = "truncation"
    document["execution_contract"] = execution_contract(
        "job-fingerprint", "analytic-v1", "numpy", "truncation"
    )
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="different numerical"):
        load_ordered_shard_outputs(shards)


def test_shard_missing_contract_rejected(tmp_path):
    plan = plan_time_shards(CorrelationWindow("scan", 0, 1), 1)
    shard = _write_shard(tmp_path / "shard", plan[0], (0,))
    path = shard / "ingest-manifest.json"
    document = json.loads(path.read_text())
    del document["execution_contract"]
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="regenerate"):
        load_ordered_shard_outputs([shard])


def test_native_record_count_mismatch_rejected(tmp_path):
    plan = plan_time_shards(CorrelationWindow("scan", 0, 1), 1)
    shard = _write_shard(tmp_path / "shard", plan[0], (0,))
    path = shard / "ingest-manifest.json"
    document = json.loads(path.read_text())
    document["output_integrity"]["record_count"] = 2
    path.write_text(json.dumps(document))
    output = tmp_path / "out"
    output.mkdir()
    with pytest.raises(ValueError, match="record count"):
        merge_native_shards([shard], output)
    assert not list(output.iterdir())


def test_merge_preserves_competing_lock_and_partial(tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    lock = output / ".assembly.lock"
    lock.write_bytes(b"owner")
    partial = output / "spectra.c64.partial"
    partial.write_bytes(b"other writer")
    with pytest.raises(FileExistsError):
        merge_native_shards([], output)
    assert lock.read_bytes() == b"owner"
    assert partial.read_bytes() == b"other writer"


def test_corrupted_native_spectra_never_published(tmp_path):
    plan = plan_time_shards(CorrelationWindow("scan", 0, 1), 1)
    shard = _write_shard(tmp_path / "shard", plan[0], (0,))
    (shard / "spectra.c64").write_bytes(bytes(16))
    output = tmp_path / "out"
    output.mkdir()
    with pytest.raises(ValueError, match="integrity"):
        merge_native_shards([shard], output)
    assert list(output.iterdir()) == []


def test_arbitrary_swin_bytes_rejected(tmp_path):
    shard = tmp_path / "shard"
    shard.mkdir()
    (shard / "data").write_bytes(b"arbitrary truncated bytes")
    (shard / "ingest-manifest.json").write_text(
        json.dumps(
            {
                "output_integrity": {
                    "version": 1,
                    "record_count": 1,
                    "files": {},
                }
            }
        )
    )
    output = tmp_path / "out"
    output.mkdir()
    with pytest.raises(ValueError, match="truncated"):
        merge_swin_shards([shard], output, "data")
    assert list(output.iterdir()) == []
