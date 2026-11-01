# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cuphoton.xbeam import (
    ingest_scheduler as scheduler,
)
from cuphoton.xbeam import upload_resources
from cuphoton.xbeam.adapters.radio import engine
from cuphoton.xbeam.adapters.vdif.recording import (
    VdifPayloadFilePlan,
    VdifPayloadPlan,
)


def test_sparse_chunk_capacity_handles_repeated_physical_indices():
    assert (
        upload_resources._sparse_chunk_mapping_bytes(
            np.array([], np.int64), 4
        )
        == 0
    )
    assert (
        upload_resources._sparse_chunk_mapping_bytes(np.arange(100), 4) == 32
    )
    assert (
        upload_resources._sparse_chunk_mapping_bytes(
            np.repeat(np.arange(100), 3), 4
        )
        == 96
    )


@pytest.mark.parametrize("trusted", [False, True])
def test_sparse_frontiers_use_physical_indices(trusted):
    plan = VdifPayloadPlan(
        40,
        8,
        8,
        (
            VdifPayloadFilePlan(
                Path("source"),
                80,
                np.arange(8, dtype=np.int32),
                source_frame_indices=np.arange(0, 80, 10, dtype=np.int64),
            ),
        ),
    )
    result = scheduler._mapped_frontier_upload_tasks(
        [[plan]],
        {id(plan): np.zeros(64, np.uint8)},
        {id(plan): np.ones(8, np.uint8)},
        {id(plan): scheduler._PayloadProgress(8)},
        [440],
        2,
        trust_slices=trusted,
    )
    tasks, _ = result
    assert [
        (t.plans[0].files[0].first_frame, t.plans[0].files[0].frame_count)
        for t in tasks
    ] == [(0, 31), (40, 31)]
    for task in tasks:
        np.testing.assert_array_equal(
            task.plans[0].files[0].source_frame_indices, [0, 10, 20, 30]
        )


def test_posix_queue_fails_pending_tasks_and_closes(monkeypatch):
    from contextlib import nullcontext

    errors = []
    progress = SimpleNamespace(finish=errors.append)
    tasks = [
        scheduler._FrontierUploadTask(i, [], [], [], [progress], 64)
        for i in range(3)
    ]
    queue = scheduler._FrontierUploadQueue(tasks)
    closed = []
    monkeypatch.setattr(
        scheduler, "_close_vdif_posix_upload_state", closed.append
    )

    def fail(*args, **kwargs):
        raise OSError("read failed")

    monkeypatch.setattr(scheduler, "_upload_vdif_plan_group", fail)
    xp = SimpleNamespace(cuda=SimpleNamespace(Device=lambda _: nullcontext()))
    with pytest.raises(OSError, match="read failed"):
        scheduler._run_frontier_upload_worker(queue, xp, 0, None, 4096, None)
    assert len(errors) == 2
    assert queue.take() is None
    assert len(closed) == 1


@pytest.mark.gpu
@pytest.mark.parametrize("unordered", [False, True])
@pytest.mark.parametrize("tiled", [False, True])
def test_posix_output_overlap_matches_synchronous(
    tmp_path, monkeypatch, unordered, tiled
):
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.Device(0).use()
    except Exception as exc:
        pytest.skip(str(exc))
    from tests.xbeam.test_gpu_backend import _packed_two_station_source

    base, source = _packed_two_station_source(tmp_path)
    job = replace(
        base,
        integration_duration_s=Decimal(2),
        subintegration_duration_s=Decimal("0.5"),
        frequencies=(replace(base.frequencies[0], channel_count=16),),
        write_autocorrelations=True,
    )
    if unordered:
        batches = {}
        for ds in job.datastreams:
            batch = next(source.batches(ds))
            files = []
            for f in batch.samples.files:
                order = np.arange(f.frame_count)
                order[1], order[2] = order[2], order[1]
                rows = np.fromfile(f.path, np.uint8).reshape(-1, 40)
                f.path.write_bytes(rows[order].tobytes())
                files.append(
                    replace(f, destination_slots=f.destination_slots[order])
                )
            batches[ds.identifier] = replace(
                batch, samples=replace(batch.samples, files=tuple(files))
            )
        source = SimpleNamespace(
            batches=lambda ds: iter([batches[ds.identifier]])
        )
    opts = dict(
        backend="cupy",
        recording_source=source,
        vdif_io="posix",
        vdif_upload_workers=1,
    )
    if tiled:
        opts["max_tile_subintegrations"] = 3
    monkeypatch.setenv("CUPHOTON_XBEAM_OUTPUT_PIPELINE", "0")
    expected = list(engine.correlate(job, **opts))
    monkeypatch.setenv("CUPHOTON_XBEAM_OUTPUT_PIPELINE", "1")
    run = engine.prepare_correlation(job, **opts)
    actual = list(run.records())
    assert run.ingest_manifest["output_pipeline"]["effective"]
    assert run.ingest_manifest["vdif_upload"]["frontier_scheduling"]
    assert len(actual) == len(expected) == 12
    for got, want in zip(actual, expected, strict=True):
        assert replace(got, spectrum=None) == replace(want, spectrum=None)
        np.testing.assert_allclose(
            got.spectrum, want.spectrum, rtol=3e-5, atol=3e-5
        )


@pytest.mark.gpu
def test_sparse_posix_reads_preserve_pairs_across_chunks(tmp_path):
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.Device(0).use()
    except Exception as exc:
        pytest.skip(str(exc))
    rows = np.zeros((50, 40), np.uint8)
    rows[:, 32:] = np.arange(50, dtype=np.uint8)[:, None]
    path = tmp_path / "sparse.vdif"
    path.write_bytes(rows.tobytes())
    sources = np.array([49, 3, 4, 3, 25, 1], np.int64)
    mapping = np.array([0, 1, 2, 3, 4, -1], np.int32)
    plan = VdifPayloadPlan(
        40,
        8,
        5,
        (
            VdifPayloadFilePlan(
                path,
                50,
                mapping,
                source_frame_indices=sources,
            ),
        ),
    )
    (out,) = upload_resources.load_vdif_payload_plans(
        [plan],
        xp=cp,
        io_backend="posix",
        target_chunk_bytes=88,
    )
    np.testing.assert_array_equal(
        cp.asnumpy(out).reshape(5, 8), rows[sources[:5], 32:]
    )


@pytest.mark.gpu
def test_posix_pipeline_generator_can_close_early(tmp_path, monkeypatch):
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.Device(0).use()
    except Exception as exc:
        pytest.skip(str(exc))
    from tests.xbeam.test_gpu_backend import _packed_two_station_source

    base, source = _packed_two_station_source(tmp_path)
    job = replace(
        base,
        integration_duration_s=Decimal(2),
        subintegration_duration_s=Decimal("0.5"),
        frequencies=(replace(base.frequencies[0], channel_count=16),),
    )
    monkeypatch.setenv("CUPHOTON_XBEAM_OUTPUT_PIPELINE", "1")
    run = engine.prepare_correlation(
        job,
        backend="cupy",
        recording_source=source,
        vdif_io="posix",
        vdif_upload_workers=1,
    )
    records = run.records()
    next(records)
    records.close()
    cp.cuda.get_current_stream().synchronize()
    assert run.ingest_manifest["output_pipeline"]["effective"]


@pytest.mark.parametrize("mappings", [([0, 0, 2, 3],), ([0, 1], [1, 2, 3])])
def test_untrusted_ordered_frontiers_reject_duplicate_destinations(mappings):
    plan = VdifPayloadPlan(
        40,
        8,
        4,
        tuple(
            VdifPayloadFilePlan(
                Path(str(index)), len(mapping), np.asarray(mapping, np.int32)
            )
            for index, mapping in enumerate(mappings)
        ),
    )
    with pytest.raises(ValueError, match="duplicate VDIF destination"):
        scheduler._mapped_frontier_upload_tasks(
            [[plan]],
            {id(plan): np.zeros(32, np.uint8)},
            {id(plan): np.ones(4, np.uint8)},
            {id(plan): scheduler._PayloadProgress(4)},
            [440],
            2,
            trust_slices=False,
        )
