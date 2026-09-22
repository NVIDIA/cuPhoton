# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Barrier, Event
from types import SimpleNamespace

import numpy as np
import pytest

from cuphoton.core.bulk import WorkItem
from cuphoton.xpois import mpi
from cuphoton.xpois.batch import BatchFitOptions


class _SharedComm:
    def __init__(self, rank: int) -> None:
        self.rank = rank
        self.freed = False

    def Get_rank(self) -> int:
        return self.rank

    def Free(self) -> None:
        self.freed = True


class _SingletonComm:
    def __init__(self, local_rank: int = 0) -> None:
        self.local_rank = local_rank
        self.split_calls: list[tuple[int, int]] = []
        self.gather_calls = 0
        self.shared: _SharedComm | None = None

    def Get_rank(self) -> int:
        return 0

    def Get_size(self) -> int:
        return 1

    def gather(self, value, root: int):
        assert root == 0
        self.gather_calls += 1
        return [value]

    def bcast(self, value, root: int):
        assert root == 0
        return value

    def Split_type(self, split_type: int, key: int) -> _SharedComm:
        self.split_calls.append((split_type, key))
        self.shared = _SharedComm(self.local_rank)
        return self.shared


class _NonrootAggregateComm:
    def __init__(self, decision) -> None:
        self.decision = decision

    def gather(self, value, root: int):
        assert root == 0
        return None

    def bcast(self, value, root: int):
        assert root == 0
        assert value is None
        return self.decision


def _options(**overrides) -> BatchFitOptions:
    values = {
        "kernel_shape": (3, 3),
        "basis_sigmas": (1.0,),
        "basis_degrees": (0,),
        "backend": "cupy",
    }
    values.update(overrides)
    return BatchFitOptions(**values)


def _gpu(uuid: str = "GPU-physical-0") -> dict[str, object]:
    return {
        "backend": "cupy",
        "device_index": 0,
        "uuid": uuid,
        "pci_bus_id": "0000:01:00.0",
        "identity_error": None,
    }


def _write_manifest(tmp_path: Path, count: int = 1) -> Path:
    pairs = []
    for index in range(count):
        reference = tmp_path / f"reference-{index}.npy"
        target = tmp_path / f"target-{index}.npy"
        np.save(reference, np.ones((8, 8), dtype=np.float32))
        np.save(target, np.ones((8, 8), dtype=np.float32))
        pairs.append(
            {
                "id": f"pair-{index}",
                "reference": str(reference),
                "target": str(target),
            }
        )
    path = tmp_path / "pairs.json"
    path.write_text(
        json.dumps(
            {"schema": "cuphoton.xpois.image-pairs/v1", "pairs": pairs}
        ),
        encoding="utf-8",
    )
    return path


def _item_runner(
    item: WorkItem, output: Path, options: BatchFitOptions
) -> dict[str, object]:
    output.mkdir()
    summary = output / "summary.json"
    summary.write_text("{}\n", encoding="utf-8")
    return {
        "run_dir": str(output),
        "summary_path": str(summary),
        "requested_backend": options.backend,
        "backend": options.backend,
        "device": "cuda:0",
        "runtime": {},
        "timings_sec": {"solve": 0.1},
        "wall_sec": {"item_runner": 0.1},
    }


def _audit_gpu_identities(
    tmp_path: Path,
    identities: tuple[dict[str, object], ...],
    hosts: tuple[str, ...],
    backend: str = "cupy",
) -> dict[str, object]:
    run_dir = tmp_path / "run"
    (run_dir / "items").mkdir(parents=True)
    (run_dir / "records").mkdir()
    (run_dir / "ranks").mkdir()
    shards = tuple(
        (WorkItem(f"item-{rank}", {"id": f"item-{rank}"}, 1),)
        for rank in range(len(identities))
    )
    results = []
    for rank, identity in enumerate(identities):
        results.append(
            mpi._execute_rank(
                run_id="gpu-identity",
                run_dir=run_dir,
                manifest_sha256="manifest",
                context=mpi._RankContext(
                    rank,
                    rank,
                    len(identities),
                    hosts[rank],
                    "test",
                    launch_id="test-launch",
                ),
                items=shards[rank],
                options=_options(backend=backend),
                visibility=str(rank),
                setup_error=None,
                item_runner=_item_runner,
                gpu_identity_loader=lambda backend, value=identity: value,
            )
        )
    records, errors = mpi._read_mappings(run_dir / "records")
    assert errors == []
    return mpi._audit_ranks(
        "gpu-identity",
        "manifest",
        backend,
        shards,
        results,
        records,
    )


def _stage_file_rank(
    tmp_path: Path,
    run_id: str,
    attempt_id: str,
    item_runner=_item_runner,
    *,
    count: int = 1,
):
    case_dir = tmp_path / run_id
    case_dir.mkdir()
    options = _options()
    manifest = mpi.load_image_pair_manifest(
        _write_manifest(case_dir, count=count)
    )
    run_dir = case_dir / "runs" / run_id
    context = mpi._RankContext(
        0, 0, 1, "host", "test", launch_id="test-launch"
    )
    mpi._file_prepare_run(
        context,
        run_dir,
        run_id,
        attempt_id,
        0.1,
        mpi.timestamp_utc(),
        manifest,
        options,
    )
    shards = mpi.partition_byte_balanced(manifest.work_items(), 1)
    rank_dir = mpi._file_rank_staging_dir(run_dir, run_id, attempt_id, 0)
    result = mpi._execute_rank(
        run_id=run_id,
        run_dir=rank_dir,
        manifest_sha256=manifest.sha256,
        context=context,
        items=shards[0],
        options=options,
        visibility="GPU-0",
        setup_error=None,
        item_runner=item_runner,
        gpu_identity_loader=lambda backend: _gpu(),
        attempt_id=attempt_id,
    )
    mpi._complete_file_rank(
        rank_dir,
        run_id,
        attempt_id,
        manifest.sha256,
        0,
        result,
    )
    return manifest, run_dir, rank_dir, shards, result


@pytest.mark.parametrize(
    ("decision", "error"),
    [
        ({"status": "success", "error": None}, None),
        ({"status": "failed", "error": None}, "MPI batch 'aggregate' failed"),
        (
            {
                "status": "failed",
                "error": {"type": "OSError", "message": "cannot write"},
            },
            "cannot persist MPI aggregate",
        ),
    ],
)
def test_collective_nonroot_honors_terminal_decision(
    decision, error, tmp_path: Path
) -> None:
    comm = _NonrootAggregateComm(decision)
    context = mpi._RankContext(
        1, 1, 2, "host", "test", launch_id="test-launch"
    )
    arguments = (
        comm,
        context,
        tmp_path / "run",
        "aggregate",
        SimpleNamespace(),
        _options(),
        ((), ()),
        {"rank": 1},
        mpi.timestamp_utc(),
        time.perf_counter(),
        mpi._MPIAPI(None, comm, "4.test", "Test MPI"),
    )

    if error is None:
        assert mpi._mpi_aggregate(*arguments) is None
    else:
        with pytest.raises(RuntimeError, match=error):
            mpi._mpi_aggregate(*arguments)


def _mpi_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_launcher_environment(monkeypatch)
    for key, value in {
        "CUDA_VISIBLE_DEVICES": "GPU-0",
        "CUPHOTON_ALLOCATED_CUDA_VISIBLE_DEVICES": "GPU-0",
        "OMPI_COMM_WORLD_RANK": "0",
        "OMPI_COMM_WORLD_SIZE": "1",
        "OMPI_COMM_WORLD_LOCAL_RANK": "0",
    }.items():
        monkeypatch.setenv(key, value)


def _files_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_launcher_environment(monkeypatch)
    for key, value in {
        "CUDA_VISIBLE_DEVICES": "GPU-0",
        "SLURM_PROCID": "0",
        "SLURM_NTASKS": "1",
        "SLURM_LOCALID": "0",
        "SLURM_JOB_ID": "123",
        "SLURM_STEP_ID": "0",
    }.items():
        monkeypatch.setenv(key, value)


def _clear_launcher_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "CUPHOTON_MPI_LAUNCH_ID",
        "PMIX_NAMESPACE",
        "SLURM_JOB_ID",
        "SLURM_STEP_ID",
        "CUPHOTON_ALLOCATED_CUDA_VISIBLE_DEVICES",
        "OMPI_COMM_WORLD_RANK",
        "OMPI_COMM_WORLD_SIZE",
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "PMIX_RANK",
        "PMIX_SIZE",
        "PMIX_LOCAL_RANK",
        "PMI_RANK",
        "PMI_SIZE",
        "PMI_LOCAL_RANK",
        "MPI_LOCALRANKID",
        "SLURM_PROCID",
        "SLURM_NTASKS",
        "SLURM_LOCALID",
    ):
        monkeypatch.delenv(key, raising=False)


def _remove_cuda_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in mpi._CUDA_MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)


@pytest.mark.parametrize("mode", ["auto", "local", "dragon"])
def test_arguments_reject_implicit_or_non_mpi_modes(mode: str) -> None:
    with pytest.raises(ValueError, match="files, mpi"):
        mpi._validate_arguments(mode, 600.0, None, None, None, _options())


def test_collective_timeout_and_file_identity_contracts() -> None:
    with pytest.raises(ValueError, match="only valid for files"):
        mpi._validate_arguments("mpi", 600.0, 10.0, None, None, _options())
    with pytest.raises(ValueError, match="explicit run_id and attempt_id"):
        mpi._validate_arguments("files", 600.0, 10.0, None, None, _options())
    with pytest.raises(ValueError, match="positive rank_timeout_sec"):
        mpi._validate_arguments(
            "files",
            600.0,
            None,
            "run-one",
            "attempt-one",
            _options(),
        )


@pytest.mark.parametrize("timeout", [True, 0.0, float("inf")])
def test_arguments_require_positive_setup_timeout(timeout) -> None:
    with pytest.raises(ValueError, match="positive rank_setup_timeout_sec"):
        mpi._validate_arguments("mpi", timeout, None, None, None, _options())


def test_identity_mapping_wait_retries_missing_and_stale_entries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    responses = iter(
        (
            FileNotFoundError("not visible yet"),
            {"run_id": "stale", "attempt_id": None},
            {"run_id": "fresh", "attempt_id": None, "status": "ready"},
        )
    )

    def read(path):
        del path
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(mpi, "_read_regular_mapping", read)
    monkeypatch.setattr(mpi.time, "sleep", lambda seconds: None)

    result = mpi._wait_identity_mapping(
        tmp_path / ".ready.json",
        1.0,
        {"run_id": "fresh", "attempt_id": None},
    )

    assert result["status"] == "ready"


def test_identity_mapping_wait_bounds_stale_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        mpi,
        "_read_regular_mapping",
        lambda path: {"run_id": "stale"},
    )

    with pytest.raises(TimeoutError, match="identity differs in: run_id"):
        mpi._wait_identity_mapping(
            tmp_path / ".ready.json", 0.001, {"run_id": "fresh"}
        )


def test_batch_rejects_more_ranks_than_manifest_items(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _files_environment(monkeypatch)
    _remove_cuda_modules(monkeypatch)
    monkeypatch.setattr(
        mpi,
        "_environment_context",
        lambda environ: mpi._RankContext(
            0, 0, 2, "host", "test", launch_id="test-launch"
        ),
    )

    with pytest.raises(RuntimeError, match="world size cannot exceed"):
        mpi.run_mpi_image_pair_batch(
            manifest_path=_write_manifest(tmp_path),
            output_root=tmp_path / "runs",
            run_id="too-many-ranks",
            aggregation_mode="files",
            rank_timeout_sec=1.0,
            attempt_id="attempt-one",
            options=_options(),
        )


def test_file_attempt_claim_is_exclusive_for_concurrent_roots(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = mpi.load_image_pair_manifest(_write_manifest(tmp_path))
    run_dir = tmp_path / "runs" / "claimed-run"
    barrier = Barrier(2)
    original_claim = mpi._claim_file_attempt

    def synchronized_claim(marker: Path) -> None:
        barrier.wait(timeout=2)
        original_claim(marker)

    monkeypatch.setattr(mpi, "_claim_file_attempt", synchronized_claim)

    def prepare() -> str:
        try:
            mpi._file_prepare_run(
                mpi._RankContext(
                    0, 0, 1, "host", "test", launch_id="test-launch"
                ),
                run_dir,
                "claimed-run",
                "same-attempt",
                1.0,
                mpi.timestamp_utc(),
                manifest,
                _options(),
            )
        except RuntimeError as exc:
            return str(exc)
        return "ready"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: prepare(), range(2)))

    assert outcomes.count("ready") == 1
    assert sum("attempt_id already exists" in item for item in outcomes) == 1
    marker = mpi._attempt_path(run_dir, "claimed-run", "same-attempt")
    assert mpi.read_json_mapping(marker)["status"] == "ready"


@pytest.mark.parametrize("peer_published", [False, True])
def test_competing_launch_cannot_change_winning_rank_preflight(
    peer_published: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = mpi.load_image_pair_manifest(
        _write_manifest(tmp_path, count=2)
    )
    run_dir = tmp_path / "runs" / "competing"
    root_published = Event()
    winner_published = Event()
    release_root = Event()
    original_publish = mpi._publish_file_preflight

    def publish(context, *args, **kwargs):
        original_publish(context, *args, **kwargs)
        if context.rank == 0:
            root_published.set()
            assert release_root.wait(5)
        else:
            winner_published.set()

    monkeypatch.setattr(mpi, "_publish_file_preflight", publish)

    def prepare(rank, launch):
        return mpi._file_prepare_run(
            mpi._RankContext(rank, rank, 2, "host", "test", launch),
            run_dir,
            "competing",
            "same-attempt",
            5.0,
            mpi.timestamp_utc(),
            manifest,
            _options(),
        )

    marker = mpi._attempt_path(run_dir, "competing", "same-attempt")
    staging = marker.with_suffix("")
    with ThreadPoolExecutor(max_workers=4) as pool:
        root = pool.submit(prepare, 0, "winner")
        try:
            assert root_published.wait(5)
            peer = (
                pool.submit(prepare, 1, "winner") if peer_published else None
            )
            if peer_published:
                assert winner_published.wait(5)
            before = {
                path: path.read_bytes()
                for path in (marker, *staging.rglob("*.json"))
            }
            simultaneous = Barrier(2)

            def compete(rank):
                simultaneous.wait(5)
                with pytest.raises(mpi._FileOwnershipError):
                    prepare(rank, "loser")

            list(pool.map(compete, (0, 1)))
            assert {path: path.read_bytes() for path in before} == before
            assert set(staging.rglob("*.json")) == set(before) - {marker}
            if peer is None:
                assert not (staging / "rank-0001").exists()
                peer = pool.submit(prepare, 1, "winner")
                assert winner_published.wait(5)
        finally:
            release_root.set()
        assert root.result() is None
        assert peer.result() is None

    preflight = mpi.read_json_mapping(
        staging / "preflight" / "rank-0001.json"
    )
    assert preflight["status"] == "success"
    assert preflight["launch_id"] == "winner"
    before = {path: path.read_bytes() for path in staging.rglob("*.json")}
    with pytest.raises(mpi._FileOwnershipError, match="already claimed"):
        prepare(1, "winner")
    assert {path: path.read_bytes() for path in before} == before


def test_duplicate_nonroot_cannot_replace_preflight_while_starting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = mpi.load_image_pair_manifest(
        _write_manifest(tmp_path, count=2)
    )
    run_dir = tmp_path / "runs" / "claimed-peer"
    marker = mpi._attempt_path(run_dir, "claimed-peer", "attempt")
    mpi.atomic_write_json(
        marker,
        {
            "run_id": "claimed-peer",
            "attempt_id": "attempt",
            "launch_id": "same-launch",
            "status": "starting",
            "terminal": False,
        },
    )
    context = mpi._RankContext(1, 1, 2, "host", "test", "same-launch")
    staging = marker.with_suffix("")
    (staging / "preflight").mkdir(parents=True)
    mpi._prepare_file_rank_for_preflight(
        context, run_dir, "claimed-peer", "attempt", None, 1.0
    )
    mpi._publish_file_preflight(
        context,
        run_dir,
        "claimed-peer",
        "attempt",
        manifest,
        _options(),
        None,
        1.0,
        1.0,
    )
    before = {path: path.read_bytes() for path in staging.rglob("*.json")}
    with pytest.raises(mpi._FileOwnershipError, match="already claimed"):
        mpi._file_prepare_run(
            context,
            run_dir,
            "claimed-peer",
            "attempt",
            1.0,
            mpi.timestamp_utc(),
            manifest,
            _options(),
        )
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize("directory", ["records", "ranks"])
def test_final_mapping_scan_rejects_symlinks(
    directory: str, tmp_path: Path
) -> None:
    records = tmp_path / directory
    records.mkdir()
    external = tmp_path / "external.json"
    mpi.atomic_write_json(external, {"status": "success"})
    (records / "linked.json").symlink_to(external)
    mpi.atomic_write_json(records / "regular.json", {"status": "failed"})
    loaded, errors = mpi._read_mappings(records)
    assert loaded == [{"status": "failed"}]
    assert len(errors) == 1
    assert errors[0]["path"] == "linked.json"
    assert "not a regular file" in errors[0]["message"]


def test_file_launch_identity_distinguishes_steps_and_nested_launches() -> (
    None
):
    environment = {"SLURM_JOB_ID": "123", "SLURM_STEP_ID": "0"}
    first = mpi._file_launch_id(environment, "slurm")
    assert first != mpi._file_launch_id(
        {**environment, "SLURM_STEP_ID": "1"}, "slurm"
    )
    assert mpi._file_launch_id(
        {**environment, "PMIX_NAMESPACE": "first"}, "openmpi"
    ) != mpi._file_launch_id(
        {**environment, "PMIX_NAMESPACE": "second"}, "openmpi"
    )
    assert mpi._file_launch_id(
        {"PMIX_NAMESPACE": "first"}, "openmpi"
    ) == mpi._file_launch_id(
        {**environment, "PMIX_NAMESPACE": "first"}, "openmpi"
    )


@pytest.mark.parametrize(
    "environment,launcher",
    [
        ({}, "openmpi"),
        ({"SLURM_JOB_ID": "123", "SLURM_STEP_ID": "0"}, "openmpi"),
        ({"SLURM_JOB_ID": "123"}, "slurm"),
        ({"SLURM_JOB_ID": "123", "SLURM_STEP_ID": "batch"}, "slurm"),
        ({"CUPHOTON_MPI_LAUNCH_ID": ""}, "pmi"),
    ],
)
def test_file_launch_identity_requires_a_real_invocation(
    environment, launcher
) -> None:
    with pytest.raises(RuntimeError, match="CUPHOTON_MPI_LAUNCH_ID"):
        mpi._file_launch_id(environment, launcher)
    explicit = {**environment, "CUPHOTON_MPI_LAUNCH_ID": "fresh-token"}
    assert mpi._file_launch_id(explicit, launcher)


@pytest.mark.parametrize("rank", [0, 1])
def test_file_attempt_rejects_redirected_attempt_root_before_rank_io(
    rank: int, tmp_path: Path
) -> None:
    manifest = mpi.load_image_pair_manifest(
        _write_manifest(tmp_path, count=2)
    )
    run_id = "redirected-attempt-root"
    attempt_id = "attempt-one"
    run_dir = tmp_path / "runs" / run_id
    output_root = run_dir.parent
    output_root.mkdir()
    redirected = tmp_path / "redirected-attempts"
    redirected.mkdir()
    attempt_root = output_root / ".mpi-attempts"
    attempt_root.symlink_to(redirected, target_is_directory=True)
    marker = mpi._attempt_path(run_dir, run_id, attempt_id)
    redirected_marker = redirected / marker.name
    mpi.atomic_write_json(
        redirected_marker,
        {
            "run_id": run_id,
            "attempt_id": attempt_id,
            "manifest_sha256": manifest.sha256,
            "status": "starting",
            "terminal": False,
        },
    )
    redirected_staging = redirected / marker.with_suffix("").name
    (redirected_staging / "preflight").mkdir(parents=True)
    before = {
        path.relative_to(redirected): path.read_bytes()
        for path in redirected.rglob("*")
        if path.is_file()
    }

    with pytest.raises(
        RuntimeError, match="file-attempt path is not a real directory"
    ):
        mpi._file_prepare_run(
            mpi._RankContext(
                rank, rank, 2, "host", "test", launch_id="test-launch"
            ),
            run_dir,
            run_id,
            attempt_id,
            1.0,
            mpi.timestamp_utc(),
            manifest,
            _options(),
        )

    after = {
        path.relative_to(redirected): path.read_bytes()
        for path in redirected.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert attempt_root.is_symlink()
    assert not run_dir.exists()


@pytest.mark.parametrize("failed_rank", [0, 1])
def test_file_preflight_failure_reaches_both_ranks_without_run(
    failed_rank: int, tmp_path: Path
) -> None:
    manifest = mpi.load_image_pair_manifest(
        _write_manifest(tmp_path, count=2)
    )
    run_id = f"preflight-failure-{failed_rank}"
    attempt_id = "attempt-one"
    run_dir = tmp_path / "runs" / run_id
    failure = {"type": "ValueError", "message": "synthetic preflight"}

    def prepare(rank: int) -> str:
        try:
            mpi._file_prepare_run(
                mpi._RankContext(
                    rank, rank, 2, "host", "test", launch_id="test-launch"
                ),
                run_dir,
                run_id,
                attempt_id,
                2.0,
                mpi.timestamp_utc(),
                None if rank == failed_rank else manifest,
                _options(),
                failure if rank == failed_rank else None,
            )
        except RuntimeError as exc:
            return str(exc)
        return "unexpected success"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(prepare, range(2)))

    assert all("synthetic preflight" in outcome for outcome in outcomes)
    marker = mpi.read_json_mapping(
        mpi._attempt_path(run_dir, run_id, attempt_id)
    )
    assert marker["status"] == "failed"
    assert marker["terminal"] is True
    assert marker["error"]["rank"] == failed_rank
    assert not run_dir.exists()
    assert not (run_dir / "run.json").exists()


@pytest.mark.parametrize("failed_rank", [0, 1])
def test_file_rank_staging_failure_is_terminal_preflight_evidence(
    failed_rank: int, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = mpi.load_image_pair_manifest(
        _write_manifest(tmp_path, count=2)
    )
    run_id = f"rank-staging-failure-{failed_rank}"
    attempt_id = "attempt-one"
    run_dir = tmp_path / "runs" / run_id
    original_prepare = mpi._prepare_file_rank_staging

    def prepare_rank(
        rank_dir: Path, *, parent_visibility_timeout_sec: float = 0.0
    ) -> None:
        if rank_dir.name == f"rank-{failed_rank:04d}":
            raise OSError("synthetic rank staging failure")
        original_prepare(
            rank_dir,
            parent_visibility_timeout_sec=parent_visibility_timeout_sec,
        )

    monkeypatch.setattr(mpi, "_prepare_file_rank_staging", prepare_rank)

    def prepare(rank: int) -> str:
        try:
            mpi._file_prepare_run(
                mpi._RankContext(
                    rank, rank, 2, "host", "test", launch_id="test-launch"
                ),
                run_dir,
                run_id,
                attempt_id,
                2.0,
                mpi.timestamp_utc(),
                manifest,
                _options(),
            )
        except RuntimeError as exc:
            return str(exc)
        return "unexpected success"

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(prepare, range(2)))

    assert time.perf_counter() - start < 1.0
    assert all("synthetic rank staging failure" in item for item in outcomes)
    marker = mpi.read_json_mapping(
        mpi._attempt_path(run_dir, run_id, attempt_id)
    )
    assert marker["status"] == "failed"
    assert marker["terminal"] is True
    assert marker["error"] == {
        "rank": failed_rank,
        "type": "OSError",
        "message": "synthetic rank staging failure",
    }
    failed_preflight = mpi.read_json_mapping(
        mpi._file_preflight_path(run_dir, run_id, attempt_id, failed_rank)
    )
    assert failed_preflight["status"] == "failed"
    assert failed_preflight["error"] == {
        "type": "OSError",
        "message": "synthetic rank staging failure",
    }
    assert not run_dir.exists()


def test_file_rank_staging_retries_delayed_parent_visibility(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rank_dir = tmp_path / "attempt-staging" / "rank-0001"
    sleeps: list[float] = []

    def publish_parent(delay: float) -> None:
        sleeps.append(delay)
        rank_dir.parent.mkdir()

    monkeypatch.setattr(mpi.time, "sleep", publish_parent)

    mpi._prepare_file_rank_staging(
        rank_dir,
        parent_visibility_timeout_sec=1.0,
    )

    assert sleeps == pytest.approx([mpi._POLL_INITIAL_SEC])
    assert {path.name for path in rank_dir.iterdir()} == {
        "items",
        "ranks",
        "records",
    }
    with pytest.raises(FileExistsError):
        mpi._prepare_file_rank_staging(
            rank_dir,
            parent_visibility_timeout_sec=1.0,
        )


def test_file_preflight_consensus_creates_run_after_all_ranks(
    tmp_path: Path,
) -> None:
    manifest = mpi.load_image_pair_manifest(
        _write_manifest(tmp_path, count=2)
    )
    run_id = "preflight-consensus"
    attempt_id = "attempt-one"
    run_dir = tmp_path / "runs" / run_id

    def prepare(rank: int):
        return mpi._file_prepare_run(
            mpi._RankContext(
                rank, rank, 2, "host", "test", launch_id="test-launch"
            ),
            run_dir,
            run_id,
            attempt_id,
            2.0,
            mpi.timestamp_utc(),
            manifest,
            _options(),
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(prepare, range(2)))

    assert outcomes == [None, None]
    assert (run_dir / "run.json").is_file()
    preflights = [
        mpi.read_json_mapping(
            mpi._file_preflight_path(run_dir, run_id, attempt_id, rank)
        )
        for rank in range(2)
    ]
    assert [item["status"] for item in preflights] == ["success", "success"]
    assert all(
        item["schema"] == "cuphoton.xpois.mpi-file-preflight/v2"
        and item["resolved_run_dir"] == str(run_dir)
        and item["attempt_marker"]
        == str(mpi._attempt_path(run_dir, run_id, attempt_id))
        and item["rank_setup_timeout_sec"] == 2.0
        and item["rank_timeout_sec"] == 2.0
        for item in preflights
    )
    marker = mpi.read_json_mapping(
        mpi._attempt_path(run_dir, run_id, attempt_id)
    )
    assert marker["status"] == "ready"
    assert marker["terminal"] is False
    for rank in range(2):
        rank_dir = mpi._file_rank_staging_dir(
            run_dir, run_id, attempt_id, rank
        )
        assert rank_dir.is_dir()
        assert all(
            (rank_dir / name).is_dir()
            for name in ("items", "ranks", "records")
        )

    divergent = {**preflights[1], "resolved_run_dir": "/other/run"}
    path_error = mpi._file_preflight_record_error(
        divergent,
        1,
        run_id,
        attempt_id,
        run_dir,
        manifest,
        _options(),
        2,
        launch_id="test-launch",
    )
    assert path_error is not None
    assert "resolved_run_dir" in path_error["message"]

    timeout_divergent = {**preflights[1], "rank_timeout_sec": 3.0}
    timeout_error = mpi._file_preflight_record_error(
        timeout_divergent,
        1,
        run_id,
        attempt_id,
        run_dir,
        manifest,
        _options(),
        2,
        launch_id="test-launch",
        rank_setup_timeout_sec=2.0,
        rank_timeout_sec=2.0,
    )
    assert timeout_error is not None
    assert "rank_timeout_sec" in timeout_error["message"]


def test_file_ready_transition_failure_reaches_both_ranks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = mpi.load_image_pair_manifest(
        _write_manifest(tmp_path, count=2)
    )
    run_id = "ready-transition-failure"
    attempt_id = "attempt-one"
    run_dir = tmp_path / "runs" / run_id
    marker = mpi._attempt_path(run_dir, run_id, attempt_id)
    original_write = mpi.atomic_write_json

    def write(path: Path, payload, **kwargs) -> None:
        if path == marker and payload.get("status") == "ready":
            raise OSError("synthetic ready transition failure")
        original_write(path, payload, **kwargs)

    monkeypatch.setattr(mpi, "atomic_write_json", write)

    def prepare(rank: int) -> str:
        try:
            mpi._file_prepare_run(
                mpi._RankContext(
                    rank, rank, 2, "host", "test", launch_id="test-launch"
                ),
                run_dir,
                run_id,
                attempt_id,
                2.0,
                mpi.timestamp_utc(),
                manifest,
                _options(),
            )
        except RuntimeError as exc:
            return str(exc)
        return "unexpected success"

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(prepare, range(2)))

    assert time.perf_counter() - start < 1.0
    assert all(
        "synthetic ready transition failure" in item for item in outcomes
    )
    attempt = mpi.read_json_mapping(marker)
    assert attempt["status"] == "failed"
    assert attempt["terminal"] is True
    assert (run_dir / "run.json").is_file()


def test_file_preflight_peer_failure_does_not_wait_for_missing_rank(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = mpi.load_image_pair_manifest(
        _write_manifest(tmp_path, count=3)
    )
    run_id = "prompt-peer-failure"
    attempt_id = "attempt-one"
    run_dir = tmp_path / "runs" / run_id
    failure = {"type": "ValueError", "message": "peer preflight failed"}
    published = Barrier(2)
    original_publish = mpi._publish_file_preflight

    def publish(*args, **kwargs) -> None:
        original_publish(*args, **kwargs)
        published.wait(timeout=2)

    monkeypatch.setattr(mpi, "_publish_file_preflight", publish)

    def prepare(rank: int) -> str:
        try:
            mpi._file_prepare_run(
                mpi._RankContext(
                    rank, rank, 3, "host", "test", launch_id="test-launch"
                ),
                run_dir,
                run_id,
                attempt_id,
                5.0,
                mpi.timestamp_utc(),
                None if rank == 1 else manifest,
                _options(),
                failure if rank == 1 else None,
            )
        except RuntimeError as exc:
            return str(exc)
        return "unexpected success"

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(prepare, (0, 1)))

    assert time.perf_counter() - start < 1.0
    assert all("peer preflight failed" in outcome for outcome in outcomes)
    marker = mpi.read_json_mapping(
        mpi._attempt_path(run_dir, run_id, attempt_id)
    )
    assert marker["error"]["rank"] == 1
    assert marker["terminal"] is True
    assert not run_dir.exists()


def test_file_peer_rejects_terminal_success_without_wait_or_mutation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = mpi.load_image_pair_manifest(
        _write_manifest(tmp_path, count=2)
    )
    run_id = "terminal-peer"
    attempt_id = "attempt-one"
    run_dir = tmp_path / "runs" / run_id
    marker = mpi._attempt_path(run_dir, run_id, attempt_id)
    mpi.atomic_write_json(
        marker,
        {
            "run_id": run_id,
            "attempt_id": attempt_id,
            "manifest_sha256": manifest.sha256,
            "status": "success",
            "terminal": True,
            "error": None,
        },
    )
    preserved = marker.read_bytes()
    monkeypatch.setattr(
        mpi.time,
        "sleep",
        lambda seconds: pytest.fail("terminal peer attempted to wait"),
    )

    with pytest.raises(RuntimeError, match="already terminal"):
        mpi._file_prepare_run(
            mpi._RankContext(
                1, 1, 2, "host", "test", launch_id="test-launch"
            ),
            run_dir,
            run_id,
            attempt_id,
            2.0,
            mpi.timestamp_utc(),
            manifest,
            _options(),
        )

    assert marker.read_bytes() == preserved
    assert not run_dir.exists()
    assert not mpi._attempt_staging_dir(run_dir, run_id, attempt_id).exists()


def test_file_attempt_directs_post_setup_failure_to_summary() -> None:
    with pytest.raises(RuntimeError, match="inspect summary.json"):
        mpi._validate_file_attempt(
            {
                "run_id": "failed-run",
                "attempt_id": "attempt-one",
                "status": "failed",
                "terminal": True,
                "error": None,
            },
            "failed-run",
            "attempt-one",
            "test-launch",
        )


def test_prebound_visibility_validates_helper_allocation() -> None:
    environment = {
        "CUDA_VISIBLE_DEVICES": "MIG-b",
        "CUPHOTON_ALLOCATED_CUDA_VISIBLE_DEVICES": "GPU-a,MIG-b",
        "OMPI_COMM_WORLD_LOCAL_RANK": "1",
    }

    assert mpi._prebound_visibility(environment) == "MIG-b"

    environment["CUDA_VISIBLE_DEVICES"] = "GPU-a"
    with pytest.raises(RuntimeError, match="does not match"):
        mpi._prebound_visibility(environment)
    with pytest.raises(RuntimeError, match="must see one CUDA device"):
        mpi._prebound_visibility({"CUDA_VISIBLE_DEVICES": "0,1"})
    with pytest.raises(
        RuntimeError, match="CUDA_VISIBLE_DEVICES is unset.*rank-exec"
    ):
        mpi._prebound_visibility({})
    with pytest.raises(RuntimeError, match="unset it when launching"):
        mpi._prebound_visibility(
            {
                "CUDA_VISIBLE_DEVICES": "0",
                "CUPHOTON_ALLOCATED_CUDA_VISIBLE_DEVICES": "0,1",
                "SLURM_LOCALID": "0",
            }
        )


def test_numba_gpu_identity_uses_normalized_pci_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = SimpleNamespace(
        id=0,
        uuid="GPU-one",
        PCI_DOMAIN_ID=0,
        PCI_BUS_ID=0xC1,
        PCI_DEVICE_ID=0,
    )
    fake_cuda = SimpleNamespace(get_current_device=lambda: device)
    monkeypatch.setitem(sys.modules, "numba", SimpleNamespace(cuda=fake_cuda))

    identity = mpi._gpu_identity("numba-cuda")

    assert identity == {
        "backend": "numba-cuda",
        "device_index": 0,
        "name": "unknown",
        "uuid": "GPU-one",
        "pci_bus_id": "00000000:C1:00.0",
        "identity_error": None,
        "identity_warnings": [],
    }


def test_cupy_gpu_identity_uses_normalized_pci_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = SimpleNamespace(
        getDevice=lambda: 0,
        getDeviceProperties=lambda index: {"uuid": "GPU-one"},
        deviceGetPCIBusId=lambda index: "0000:c1:00.0",
    )
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(cuda=SimpleNamespace(runtime=runtime)),
    )

    identity = mpi._gpu_identity("cupy")

    assert identity["pci_bus_id"] == "00000000:C1:00.0"
    assert identity["identity_warnings"] == []


def test_cupy_gpu_identity_preserves_uuid_on_pci_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(index: int) -> str:
        raise RuntimeError("pci unavailable")

    runtime = SimpleNamespace(
        getDevice=lambda: 0,
        getDeviceProperties=lambda index: {"uuid": "GPU-one"},
        deviceGetPCIBusId=unavailable,
    )
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(cuda=SimpleNamespace(runtime=runtime)),
    )

    identity = mpi._gpu_identity("cupy")

    assert identity["uuid"] == "GPU-one"
    assert identity["pci_bus_id"] is None
    assert identity["identity_error"] is None
    assert identity["identity_warnings"] == [
        "pci_bus_id: RuntimeError: pci unavailable"
    ]


def test_numba_gpu_identity_preserves_missing_uuid_for_pci_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    device = SimpleNamespace(
        id=0,
        uuid=None,
        PCI_DOMAIN_ID=0,
        PCI_BUS_ID=1,
        PCI_DEVICE_ID=0,
    )
    fake_cuda = SimpleNamespace(get_current_device=lambda: device)
    monkeypatch.setitem(sys.modules, "numba", SimpleNamespace(cuda=fake_cuda))

    identity = mpi._gpu_identity("numba-cuda")

    assert identity["uuid"] is None
    assert identity["pci_bus_id"] == "00000000:01:00.0"
    assert identity["identity_error"] is None
    assert identity["identity_warnings"] == []
    audit = _audit_gpu_identities(
        tmp_path,
        (identity, identity),
        ("same-host", "same-host"),
        "numba-cuda",
    )
    assert audit["duplicate_physical_gpu_ranks"] == [0, 1]


@pytest.mark.parametrize("broken_field", ["uuid", "pci_bus_id"])
def test_numba_gpu_identity_preserves_partial_identity(
    monkeypatch: pytest.MonkeyPatch, broken_field: str
) -> None:
    class PartialIdentity:
        id = 0
        PCI_BUS_ID = 1
        PCI_DEVICE_ID = 0

        @property
        def uuid(self):
            if broken_field == "uuid":
                raise RuntimeError("uuid unavailable")
            return "GPU-one"

        @property
        def PCI_DOMAIN_ID(self):
            if broken_field == "pci_bus_id":
                raise RuntimeError("pci unavailable")
            return 0

    fake_cuda = SimpleNamespace(get_current_device=PartialIdentity)
    monkeypatch.setitem(sys.modules, "numba", SimpleNamespace(cuda=fake_cuda))

    identity = mpi._gpu_identity("numba-cuda")

    assert identity["identity_error"] is None
    assert len(identity["identity_warnings"]) == 1
    assert identity["identity_warnings"][0].startswith(broken_field)
    assert identity["uuid"] is not None or identity["pci_bus_id"] is not None


def test_numba_gpu_identity_reports_identity_lookup_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenIdentity:
        id = 0

        @property
        def uuid(self):
            raise RuntimeError("uuid unavailable")

        @property
        def PCI_DOMAIN_ID(self):
            raise RuntimeError("pci unavailable")

    fake_cuda = SimpleNamespace(get_current_device=BrokenIdentity)
    monkeypatch.setitem(sys.modules, "numba", SimpleNamespace(cuda=fake_cuda))

    identity = mpi._gpu_identity("numba-cuda")

    assert identity["uuid"] is None
    assert identity["pci_bus_id"] is None
    assert identity["identity_error"] == (
        "uuid: RuntimeError: uuid unavailable; "
        "pci_bus_id: RuntimeError: pci unavailable"
    )
    assert identity["identity_warnings"] == [
        "uuid: RuntimeError: uuid unavailable",
        "pci_bus_id: RuntimeError: pci unavailable",
    ]


def test_mpi_context_rejects_launcher_disagreement() -> None:
    comm = _SingletonComm()
    environment = {
        "OMPI_COMM_WORLD_RANK": "1",
        "OMPI_COMM_WORLD_SIZE": "2",
        "OMPI_COMM_WORLD_LOCAL_RANK": "0",
    }

    with pytest.raises(RuntimeError, match="disagrees"):
        mpi._mpi_context(
            SimpleNamespace(COMM_TYPE_SHARED=1), comm, environment
        )


def test_mpi_context_accepts_openmpi_with_rank_only_pmix() -> None:
    comm = _SingletonComm()
    context = mpi._mpi_context(
        SimpleNamespace(COMM_TYPE_SHARED=1),
        comm,
        {
            "OMPI_COMM_WORLD_RANK": "0",
            "OMPI_COMM_WORLD_SIZE": "1",
            "OMPI_COMM_WORLD_LOCAL_RANK": "0",
            "PMIX_RANK": "0",
        },
    )

    assert context.rank == 0
    assert context.world_size == 1
    assert context.launcher == "openmpi"
    assert comm.split_calls == [(1, 0)]
    assert comm.shared is not None and comm.shared.freed is True


def test_mpi_context_prefers_openmpi_over_inherited_slurm() -> None:
    comm = _SingletonComm(local_rank=0)
    context = mpi._mpi_context(
        SimpleNamespace(COMM_TYPE_SHARED=1),
        comm,
        {
            "OMPI_COMM_WORLD_RANK": "0",
            "OMPI_COMM_WORLD_SIZE": "1",
            "OMPI_COMM_WORLD_LOCAL_RANK": "0",
            "SLURM_PROCID": "7",
            "SLURM_NTASKS": "8",
            "SLURM_LOCALID": "6",
        },
    )

    assert context.rank == 0
    assert context.local_rank == 0
    assert context.world_size == 1
    assert context.launcher == "openmpi"


def test_launcher_and_local_rank_precedence() -> None:
    environment = {
        "PMIX_RANK": "1",
        "PMIX_LOCAL_RANK": "1",
        "PMI_RANK": "1",
        "PMI_SIZE": "2",
        "PMI_LOCAL_RANK": "1",
        "SLURM_PROCID": "7",
        "SLURM_NTASKS": "8",
        "SLURM_LOCALID": "6",
    }

    assert mpi._launcher_context(environment) == ("pmi", 1, 2)
    assert mpi._local_rank(environment, "pmi") == 1


def test_pmi_launcher_accepts_matching_direct_slurm_local_rank() -> None:
    context = mpi._environment_context(
        {
            "PMI_RANK": "1",
            "PMI_SIZE": "2",
            "SLURM_PROCID": "1",
            "SLURM_NTASKS": "2",
            "SLURM_LOCALID": "1",
            "SLURM_JOB_ID": "123",
            "SLURM_STEP_ID": "0",
        }
    )

    assert context.rank == 1
    assert context.local_rank == 1
    assert context.world_size == 2
    assert context.launcher == "pmi"


def test_pmi_launcher_rejects_disagreeing_slurm_topology() -> None:
    with pytest.raises(RuntimeError, match="PMI and Slurm.*disagree"):
        mpi._environment_context(
            {
                "PMI_RANK": "1",
                "PMI_SIZE": "2",
                "SLURM_PROCID": "0",
                "SLURM_NTASKS": "2",
                "SLURM_LOCALID": "1",
                "SLURM_JOB_ID": "123",
                "SLURM_STEP_ID": "0",
            }
        )


def test_launcher_local_rank_ignores_stale_metadata() -> None:
    environment = {
        "OMPI_COMM_WORLD_LOCAL_RANK": "1",
        "PMIX_LOCAL_RANK": "7",
        "SLURM_LOCALID": "6",
    }

    assert mpi._local_rank(environment, "openmpi") == 1


def test_slurm_local_rank_rejects_disagreeing_pmix_metadata() -> None:
    with pytest.raises(RuntimeError, match="slurm.*local-rank.*disagree"):
        mpi._local_rank(
            {
                "SLURM_LOCALID": "1",
                "SLURM_JOB_ID": "123",
                "SLURM_STEP_ID": "0",
                "PMIX_LOCAL_RANK": "7",
            },
            "slurm",
        )


def test_environment_context_accepts_direct_slurm_pmix_metadata() -> None:
    context = mpi._environment_context(
        {
            "PMIX_RANK": "1",
            "PMIX_LOCAL_RANK": "1",
            "SLURM_PROCID": "1",
            "SLURM_NTASKS": "2",
            "SLURM_LOCALID": "1",
            "SLURM_JOB_ID": "123",
            "SLURM_STEP_ID": "0",
        }
    )

    assert context.rank == 1
    assert context.local_rank == 1
    assert context.world_size == 2
    assert context.launcher == "slurm"


def test_environment_context_rejects_local_rank_outside_world() -> None:
    with pytest.raises(RuntimeError, match="local rank must be smaller"):
        mpi._environment_context(
            {
                "OMPI_COMM_WORLD_RANK": "0",
                "OMPI_COMM_WORLD_SIZE": "1",
                "OMPI_COMM_WORLD_LOCAL_RANK": "1",
            }
        )


def test_mpi_context_rejects_launcher_local_rank_mismatch() -> None:
    comm = _SingletonComm(local_rank=0)

    with pytest.raises(RuntimeError, match="local rank disagrees"):
        mpi._mpi_context(
            SimpleNamespace(COMM_TYPE_SHARED=1),
            comm,
            {
                "OMPI_COMM_WORLD_RANK": "0",
                "OMPI_COMM_WORLD_SIZE": "1",
                "OMPI_COMM_WORLD_LOCAL_RANK": "1",
            },
        )

    assert comm.split_calls == [(1, 0)]
    assert comm.gather_calls == 1


def test_mpi_context_rejects_peer_error_by_consensus() -> None:
    class PeerFailureComm(_SingletonComm):
        def Get_size(self) -> int:
            return 2

        def gather(self, value, root: int):
            assert root == 0
            self.gather_calls += 1
            return [
                value,
                {
                    "rank": 1,
                    "error": {
                        "type": "RuntimeError",
                        "message": "peer local rank disagrees",
                    },
                },
            ]

    comm = PeerFailureComm()

    with pytest.raises(RuntimeError, match="peer local rank disagrees"):
        mpi._mpi_context(
            SimpleNamespace(COMM_TYPE_SHARED=1),
            comm,
            {
                "OMPI_COMM_WORLD_RANK": "0",
                "OMPI_COMM_WORLD_SIZE": "2",
                "OMPI_COMM_WORLD_LOCAL_RANK": "0",
            },
        )

    assert comm.gather_calls == 1


def test_mpi_startup_consensus_rejects_peer_error() -> None:
    class PeerFailureComm(_SingletonComm):
        def Get_size(self) -> int:
            return 2

        def gather(self, value, root: int):
            assert root == 0
            self.gather_calls += 1
            return [
                value,
                {
                    "rank": 1,
                    "error": {
                        "type": "RuntimeError",
                        "message": "peer visibility is invalid",
                    },
                },
            ]

    comm = PeerFailureComm()

    with pytest.raises(
        RuntimeError,
        match="rank 1: RuntimeError: peer visibility is invalid",
    ):
        mpi._mpi_failure_consensus(
            comm,
            "MPI rank-startup validation",
            None,
        )

    assert comm.gather_calls == 1


def test_mpi_failure_consensus_broadcasts_persistence_error() -> None:
    events: list[str] = []

    class PeerFailureComm(_SingletonComm):
        def Get_size(self) -> int:
            return 2

        def gather(self, value, root: int):
            assert root == 0
            return [
                value,
                {
                    "rank": 1,
                    "error": {
                        "type": "RuntimeError",
                        "message": "peer cannot read setup evidence",
                    },
                },
            ]

        def bcast(self, value, root: int):
            assert root == 0
            events.append("broadcast")
            return value

    def fail_persistence(error) -> None:
        assert "peer cannot read setup evidence" in error["message"]
        events.append("persist")
        raise OSError("summary write failed")

    with pytest.raises(
        RuntimeError,
        match=(
            "peer cannot read setup evidence.*failed to persist terminal "
            "setup evidence: OSError: summary write failed"
        ),
    ):
        mpi._mpi_failure_consensus(
            PeerFailureComm(),
            "MPI shared run-directory validation",
            None,
            root_failure_handler=fail_persistence,
        )

    assert events == ["persist", "broadcast"]


def test_mpi_startup_consensus_rejects_malformed_peer_error() -> None:
    class MalformedComm(_SingletonComm):
        def Get_size(self) -> int:
            return 2

        def gather(self, value, root: int):
            assert root == 0
            return [value, {"rank": 1, "error": {"type": "broken"}}]

    with pytest.raises(RuntimeError, match="rank 1: invalid error payload"):
        mpi._mpi_failure_consensus(
            MalformedComm(),
            "MPI rank-startup validation",
            None,
        )


def test_mpi_context_ignores_lower_precedence_partial_launcher() -> None:
    context = mpi._mpi_context(
        SimpleNamespace(COMM_TYPE_SHARED=1),
        _SingletonComm(),
        {
            "OMPI_COMM_WORLD_RANK": "0",
            "OMPI_COMM_WORLD_SIZE": "1",
            "OMPI_COMM_WORLD_LOCAL_RANK": "0",
            "PMIX_RANK": "1",
        },
    )

    assert context.launcher == "openmpi"


def test_environment_context_rejects_partial_selected_launcher() -> None:
    with pytest.raises(RuntimeError, match="openmpi launcher rank and size"):
        mpi._environment_context(
            {
                "OMPI_COMM_WORLD_RANK": "0",
                "PMIX_RANK": "0",
                "PMIX_SIZE": "1",
                "PMIX_LOCAL_RANK": "0",
            }
        )


def test_batch_coordinates_cuda_import_guard_after_mpi_loading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _mpi_environment(monkeypatch)
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace())
    comm = _SingletonComm()
    loaded = []

    def load_api():
        loaded.append(True)
        return mpi._MPIAPI(None, comm, "4.test", "Test MPI")

    monkeypatch.setattr(
        mpi,
        "_load_mpi_api",
        load_api,
    )
    monkeypatch.setattr(
        mpi,
        "_mpi_context",
        lambda *args: pytest.fail("context entered after startup failure"),
    )

    with pytest.raises(
        RuntimeError,
        match="rank-startup validation.*before MPI rank binding: cupy",
    ):
        mpi.run_mpi_image_pair_batch(
            manifest_path=tmp_path / "unused.json",
            output_root=tmp_path / "runs",
            run_id=None,
            aggregation_mode="mpi",
            rank_timeout_sec=None,
            attempt_id=None,
            options=_options(),
        )

    assert loaded == [True]
    assert comm.gather_calls == 1


def test_mpi_consensus_rejects_rank_option_mismatch() -> None:
    class DivergentComm:
        def gather(self, value, root: int):
            assert root == 0
            other = dict(value)
            other["rank"] = 1
            other["options"] = {**value["options"], "backend": "cutile"}
            return [value, other]

        def bcast(self, value, root: int):
            assert root == 0
            return value

    with pytest.raises(RuntimeError, match="options differ"):
        mpi._mpi_manifest_consensus(
            DivergentComm(),
            mpi._RankContext(
                0, 0, 2, "host", "test", launch_id="test-launch"
            ),
            SimpleNamespace(sha256="manifest"),
            None,
            "run",
            Path("/shared/run"),
            _options(),
            "4.test",
            "Test MPI",
        )


def test_mpi_consensus_preserves_manifest_failure_detail() -> None:
    comm = _SingletonComm()

    with pytest.raises(
        RuntimeError,
        match=(
            "manifest preflight failed: rank 0: FileNotFoundError: "
            "manifest does not exist"
        ),
    ):
        mpi._mpi_manifest_consensus(
            comm,
            mpi._RankContext(
                0, 0, 1, "host", "test", launch_id="test-launch"
            ),
            None,
            {
                "type": "FileNotFoundError",
                "message": "manifest does not exist",
            },
            "run",
            Path("/shared/run"),
            _options(),
            "4.test",
            "Test MPI",
        )


def test_mpi_consensus_broadcasts_malformed_peer_failure() -> None:
    class MalformedComm:
        broadcast = False

        def gather(self, value, root: int):
            assert root == 0
            return [value, {"rank": 1}]

        def bcast(self, value, root: int):
            assert root == 0
            self.broadcast = True
            return value

    comm = MalformedComm()
    with pytest.raises(RuntimeError, match="invalid MPI manifest consensus"):
        mpi._mpi_manifest_consensus(
            comm,
            mpi._RankContext(
                0, 0, 2, "host", "test", launch_id="test-launch"
            ),
            SimpleNamespace(sha256="manifest"),
            None,
            "run",
            Path("/shared/run"),
            _options(),
            "4.test",
            "Test MPI",
        )

    assert comm.broadcast is True


def test_mpi_consensus_rejects_manifest_identity_mismatch() -> None:
    class DivergentComm:
        def gather(self, value, root: int):
            assert root == 0
            other = dict(value)
            other["rank"] = 1
            other["manifest_sha256"] = "different-input-identity"
            return [value, other]

        def bcast(self, value, root: int):
            assert root == 0
            return value

    with pytest.raises(
        RuntimeError,
        match="manifest, requested run ID, or resolved run directory",
    ):
        mpi._mpi_manifest_consensus(
            DivergentComm(),
            mpi._RankContext(
                0, 0, 2, "host", "test", launch_id="test-launch"
            ),
            SimpleNamespace(sha256="manifest-and-input-identity"),
            None,
            "run",
            Path("/shared/run"),
            _options(),
            "4.test",
            "Test MPI",
        )


@pytest.mark.parametrize(
    ("field", "peer_value", "message"),
    [
        ("mpi4py_version", "4.other", "mpi4py versions differ"),
        ("mpi_library_version", "Other MPI", "MPI library versions differ"),
        (
            "rank_setup_timeout_sec",
            601.0,
            "setup timeouts differ",
        ),
    ],
)
def test_mpi_consensus_rejects_version_mismatch(
    field: str, peer_value: object, message: str
) -> None:
    class DivergentComm:
        def gather(self, value, root: int):
            assert root == 0
            other = dict(value)
            other["rank"] = 1
            other[field] = peer_value
            return [value, other]

        def bcast(self, value, root: int):
            assert root == 0
            return value

    with pytest.raises(RuntimeError, match=message):
        mpi._mpi_manifest_consensus(
            DivergentComm(),
            mpi._RankContext(
                0, 0, 2, "host", "test", launch_id="test-launch"
            ),
            SimpleNamespace(sha256="manifest"),
            None,
            "run",
            Path("/shared/run"),
            _options(),
            "4.test",
            "Test MPI\x00\n",
        )


def test_mpi_consensus_rejects_resolved_run_directory_mismatch() -> None:
    class DivergentComm:
        def gather(self, value, root: int):
            assert root == 0
            other = {**value, "rank": 1, "resolved_run_dir": "/other/run"}
            return [value, other]

        def bcast(self, value, root: int):
            assert root == 0
            return value

    with pytest.raises(RuntimeError, match="resolved run directory"):
        mpi._mpi_manifest_consensus(
            DivergentComm(),
            mpi._RankContext(
                0, 0, 2, "host", "test", launch_id="test-launch"
            ),
            SimpleNamespace(sha256="manifest"),
            None,
            "run",
            Path("/shared/run"),
            _options(),
            "4.test",
            "Test MPI",
        )


def test_mpi_library_version_strips_trailing_nul(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mpi = SimpleNamespace(
        COMM_WORLD=object(),
        Get_library_version=lambda: "Test MPI\x00\x00\n",
    )
    monkeypatch.setitem(sys.modules, "mpi4py", SimpleNamespace(MPI=fake_mpi))
    monkeypatch.setattr(mpi, "_distribution_version", lambda name: "4.test")

    api = mpi._load_mpi_api()

    assert api.library_version == "Test MPI"


def test_execute_rank_continues_and_persists_terminal_records(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(
        "CUPHOTON_ALLOCATED_CUDA_VISIBLE_DEVICES", raising=False
    )
    run_dir = tmp_path / "run"
    (run_dir / "items").mkdir(parents=True)
    (run_dir / "records").mkdir()
    (run_dir / "ranks").mkdir()
    items = (
        WorkItem("bad", {"id": "bad"}, 10),
        WorkItem("good", {"id": "good"}, 20),
    )

    def runner(item, output, options):
        if item.item_id == "bad":
            error = ValueError("synthetic failure")
            error.add_note("input identity changed after preflight")
            raise error
        return {
            **_item_runner(item, output, options),
            "started_at_utc": "1900-01-01T00:00:00+00:00",
        }

    result = mpi._execute_rank(
        run_id="test-run",
        run_dir=run_dir,
        manifest_sha256="manifest",
        context=mpi._RankContext(
            0, 0, 1, "host", "test", launch_id="test-launch"
        ),
        items=items,
        options=_options(),
        visibility="GPU-0",
        setup_error=None,
        item_runner=runner,
        gpu_identity_loader=lambda backend: _gpu(),
    )

    assert result["status"] == "failed"
    assert result["success_count"] == 1
    assert result["failed_count"] == 1
    assert sorted(path.name for path in (run_dir / "records").iterdir()) == [
        "bad.json",
        "good.json",
    ]
    failed_record = json.loads((run_dir / "records" / "bad.json").read_text())
    good_record = json.loads((run_dir / "records" / "good.json").read_text())
    assert failed_record["status"] == "failed"
    assert failed_record["error"]["notes"] == (
        "input identity changed after preflight"
    )
    assert good_record["started_at_utc"] != "1900-01-01T00:00:00+00:00"
    records, errors = mpi._read_mappings(run_dir / "records")
    assert errors == []
    audit = mpi._audit_ranks(
        "test-run",
        "manifest",
        "cupy",
        (items,),
        (result,),
        records,
    )
    assert audit["ok"] is True
    assert audit["failed_ranks"] == [0]
    assert audit["setup_failed_ranks"] == []
    assert audit["mismatched_ranks"] == []


def test_rank_result_sort_key_places_malformed_evidence_last() -> None:
    results = [
        {"rank": 1},
        {"rank": "invalid"},
        {},
        {"rank": 0},
    ]

    assert sorted(results, key=mpi._rank_result_sort_key) == [
        {"rank": 0},
        {"rank": 1},
        {"rank": "invalid"},
        {},
    ]


def test_rank_audit_classifies_trustworthy_setup_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(
        "CUPHOTON_ALLOCATED_CUDA_VISIBLE_DEVICES", raising=False
    )
    run_dir = tmp_path / "run"
    (run_dir / "items").mkdir(parents=True)
    (run_dir / "records").mkdir()
    (run_dir / "ranks").mkdir()
    items = (WorkItem("one", {"id": "one"}, 1),)
    setup_error = {
        "type": "RunSetupMismatchError",
        "message": "file setup differs in: options",
    }
    result = mpi._execute_rank(
        run_id="setup-failure",
        run_dir=run_dir,
        manifest_sha256="manifest",
        context=mpi._RankContext(
            0, 0, 1, "host", "test", launch_id="test-launch"
        ),
        items=items,
        options=_options(),
        visibility="GPU-0",
        setup_error=setup_error,
        item_runner=lambda *args: pytest.fail("setup failure ran an item"),
        gpu_identity_loader=lambda backend: pytest.fail(
            "setup failure initialized a GPU"
        ),
    )
    records, errors = mpi._read_mappings(run_dir / "records")
    assert errors == []

    audit = mpi._audit_ranks(
        "setup-failure",
        "manifest",
        "cupy",
        (items,),
        (result,),
        records,
    )

    assert audit["ok"] is True
    assert audit["failed_ranks"] == []
    assert audit["setup_failed_ranks"] == [0]
    assert audit["mismatched_ranks"] == []

    for malformed in (None, {}, {"type": "", "message": "bad"}):
        malformed_result = {**result, "error": malformed}
        malformed_audit = mpi._audit_ranks(
            "setup-failure",
            "manifest",
            "cupy",
            (items,),
            (malformed_result,),
            records,
        )
        assert malformed_audit["setup_failed_ranks"] == []
        assert malformed_audit["failed_ranks"] == []
        assert malformed_audit["mismatched_ranks"][0]["rank"] == 0
        fields = malformed_audit["mismatched_ranks"][0]["fields"]
        assert "provenance" in fields
        if malformed is not None:
            assert "error" in fields

    gpu_result = {
        **result,
        "provenance": {**result["provenance"], "gpu": _gpu()},
    }
    gpu_audit = mpi._audit_ranks(
        "setup-failure",
        "manifest",
        "cupy",
        (items,),
        (gpu_result,),
        records,
    )
    assert gpu_audit["setup_failed_ranks"] == []
    assert gpu_audit["failed_ranks"] == []
    assert "provenance" in gpu_audit["mismatched_ranks"][0]["fields"]


def test_rank_audit_does_not_infer_setup_failure_from_item_error_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(
        "CUPHOTON_ALLOCATED_CUDA_VISIBLE_DEVICES", raising=False
    )
    run_dir = tmp_path / "run"
    (run_dir / "items").mkdir(parents=True)
    (run_dir / "records").mkdir()
    (run_dir / "ranks").mkdir()
    items = (WorkItem("one", {"id": "one"}, 1),)

    def fail_with_setup_prefix(*args):
        raise RuntimeError("rank setup failed: legitimate item failure")

    result = mpi._execute_rank(
        run_id="workload-failure",
        run_dir=run_dir,
        manifest_sha256="manifest",
        context=mpi._RankContext(
            0, 0, 1, "host", "test", launch_id="test-launch"
        ),
        items=items,
        options=_options(),
        visibility="GPU-0",
        setup_error=None,
        item_runner=fail_with_setup_prefix,
        gpu_identity_loader=lambda backend: _gpu(),
    )
    records, errors = mpi._read_mappings(run_dir / "records")
    assert errors == []

    audit = mpi._audit_ranks(
        "workload-failure",
        "manifest",
        "cupy",
        (items,),
        (result,),
        records,
    )

    assert audit["ok"] is True
    assert audit["failed_ranks"] == [0]
    assert audit["setup_failed_ranks"] == []
    assert audit["mismatched_ranks"] == []


def test_execute_rank_rejects_success_without_real_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(
        "CUPHOTON_ALLOCATED_CUDA_VISIBLE_DEVICES", raising=False
    )
    run_dir = tmp_path / "run"
    (run_dir / "items").mkdir(parents=True)
    (run_dir / "records").mkdir()
    (run_dir / "ranks").mkdir()

    def missing_output(item, output, options):
        return {
            "run_dir": str(output),
            "summary_path": str(output / "summary.json"),
            "requested_backend": options.backend,
            "backend": options.backend,
            "device": "cuda:0",
            "runtime": {},
            "timings_sec": {"solve": 0.1},
            "wall_sec": {"item_runner": 0.1},
        }

    result = mpi._execute_rank(
        run_id="missing-output",
        run_dir=run_dir,
        manifest_sha256="manifest",
        context=mpi._RankContext(
            0, 0, 1, "host", "test", launch_id="test-launch"
        ),
        items=(WorkItem("one", {"id": "one"}, 1),),
        options=_options(),
        visibility="GPU-0",
        setup_error=None,
        item_runner=missing_output,
        gpu_identity_loader=lambda backend: _gpu(),
    )

    record = mpi.read_json_mapping(run_dir / "records" / "one.json")
    assert result["status"] == "failed"
    assert record["status"] == "failed"
    assert "real output directory" in record["error"]["message"]


def test_rank_audit_rejects_duplicate_physical_gpu(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(
        "CUPHOTON_ALLOCATED_CUDA_VISIBLE_DEVICES", raising=False
    )
    run_dir = tmp_path / "run"
    (run_dir / "items").mkdir(parents=True)
    (run_dir / "records").mkdir()
    (run_dir / "ranks").mkdir()
    shards = (
        (WorkItem("one", {"id": "one"}, 1),),
        (WorkItem("two", {"id": "two"}, 1),),
    )
    results = []
    for rank in range(2):
        results.append(
            mpi._execute_rank(
                run_id="duplicate-gpu",
                run_dir=run_dir,
                manifest_sha256="manifest",
                context=mpi._RankContext(
                    rank,
                    rank,
                    2,
                    "same-host",
                    "test",
                    launch_id="test-launch",
                ),
                items=shards[rank],
                options=_options(),
                visibility=str(rank),
                setup_error=None,
                item_runner=_item_runner,
                gpu_identity_loader=lambda backend: _gpu(),
            )
        )
    records, errors = mpi._read_mappings(run_dir / "records")
    assert errors == []

    audit = mpi._audit_ranks(
        "duplicate-gpu",
        "manifest",
        "cupy",
        shards,
        results,
        records,
    )

    assert audit["ok"] is False
    assert audit["duplicate_physical_gpu_ranks"] == [0, 1]


def test_rank_audit_does_not_compare_duplicate_rank_to_itself(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(
        "CUPHOTON_ALLOCATED_CUDA_VISIBLE_DEVICES", raising=False
    )
    run_dir = tmp_path / "run"
    (run_dir / "items").mkdir(parents=True)
    (run_dir / "records").mkdir()
    (run_dir / "ranks").mkdir()
    shards = ((WorkItem("one", {"id": "one"}, 1),),)
    result = mpi._execute_rank(
        run_id="duplicate-rank",
        run_dir=run_dir,
        manifest_sha256="manifest",
        context=mpi._RankContext(
            0, 0, 1, "same-host", "test", launch_id="test-launch"
        ),
        items=shards[0],
        options=_options(),
        visibility="0",
        setup_error=None,
        item_runner=_item_runner,
        gpu_identity_loader=lambda backend: _gpu(),
    )
    records, errors = mpi._read_mappings(run_dir / "records")
    assert errors == []

    audit = mpi._audit_ranks(
        "duplicate-rank",
        "manifest",
        "cupy",
        shards,
        (result, result),
        records,
    )

    assert audit["ok"] is False
    assert audit["duplicate_ranks"] == [0]
    assert audit["duplicate_physical_gpu_ranks"] == []
    assert audit["incomparable_physical_gpu_ranks"] == []
    assert audit["mismatched_ranks"] == []


def test_rank_audit_rejects_duplicate_uuid_across_hosts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(
        "CUPHOTON_ALLOCATED_CUDA_VISIBLE_DEVICES", raising=False
    )
    run_dir = tmp_path / "run"
    (run_dir / "items").mkdir(parents=True)
    (run_dir / "records").mkdir()
    (run_dir / "ranks").mkdir()
    shards = (
        (WorkItem("one", {"id": "one"}, 1),),
        (WorkItem("two", {"id": "two"}, 1),),
    )
    results = []
    for rank in range(2):
        identity = {
            **_gpu("GPU-global-duplicate"),
            "pci_bus_id": f"0000:0{rank + 1}:00.0",
        }
        results.append(
            mpi._execute_rank(
                run_id="duplicate-uuid",
                run_dir=run_dir,
                manifest_sha256="manifest",
                context=mpi._RankContext(
                    rank,
                    0,
                    2,
                    f"host-{rank}",
                    "test",
                    launch_id="test-launch",
                ),
                items=shards[rank],
                options=_options(),
                visibility=str(rank),
                setup_error=None,
                item_runner=_item_runner,
                gpu_identity_loader=lambda backend, value=identity: value,
            )
        )
    records, errors = mpi._read_mappings(run_dir / "records")
    assert errors == []

    audit = mpi._audit_ranks(
        "duplicate-uuid",
        "manifest",
        "cupy",
        shards,
        results,
        records,
    )

    assert audit["ok"] is False
    assert audit["duplicate_physical_gpu_ranks"] == [0, 1]

    shared_pci_results = []
    for rank, result in enumerate(results):
        provenance = result["provenance"]
        shared_pci_results.append(
            {
                **result,
                "provenance": {
                    **provenance,
                    "gpu": {
                        **provenance["gpu"],
                        "uuid": f"GPU-unique-{rank}",
                        "pci_bus_id": "0000:01:00.0",
                    },
                },
            }
        )
    pci_audit = mpi._audit_ranks(
        "duplicate-uuid",
        "manifest",
        "cupy",
        shards,
        shared_pci_results,
        records,
    )
    assert pci_audit["ok"] is True


def test_rank_audit_allows_distinct_mig_uuids_sharing_pci(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(
        "CUPHOTON_ALLOCATED_CUDA_VISIBLE_DEVICES", raising=False
    )
    identities = tuple(
        {
            **_gpu(f"MIG-GPU-instance-{rank}"),
            "pci_bus_id": "0000:01:00.0",
        }
        for rank in range(2)
    )

    audit = _audit_gpu_identities(
        tmp_path, identities, ("same-host", "same-host")
    )

    assert audit["ok"] is True
    assert audit["duplicate_physical_gpu_ranks"] == []


def test_rank_audit_rejects_incomparable_same_host_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(
        "CUPHOTON_ALLOCATED_CUDA_VISIBLE_DEVICES", raising=False
    )
    identities = (
        {**_gpu("GPU-only"), "pci_bus_id": None},
        {**_gpu(), "uuid": None, "pci_bus_id": "0000:02:00.0"},
    )

    audit = _audit_gpu_identities(
        tmp_path, identities, ("same-host", "same-host")
    )

    assert audit["ok"] is False
    assert audit["duplicate_physical_gpu_ranks"] == []
    assert audit["incomparable_physical_gpu_ranks"] == [0, 1]
    assert audit["mismatched_ranks"] == [
        {"rank": 0, "fields": ["provenance"]},
        {"rank": 1, "fields": ["provenance"]},
    ]


@pytest.mark.parametrize(
    ("hosts", "uuids", "pci_bus_ids", "expected_duplicates"),
    (
        (
            ("same-host", "same-host"),
            (None, None),
            ("0000:01:00.0", "0000:01:00.0"),
            [0, 1],
        ),
        (
            ("same-host", "same-host"),
            ("MIG-GPU-instance-0", None),
            ("0000:01:00.0", "0000:01:00.0"),
            [0, 1],
        ),
        (
            ("host-0", "host-1"),
            (None, None),
            ("0000:01:00.0", "0000:01:00.0"),
            [],
        ),
        (
            ("same-host", "same-host"),
            (None, None),
            ("0000:01:00.0", "0000:02:00.0"),
            [],
        ),
    ),
    ids=(
        "both-uuids-missing-same-pci",
        "one-uuid-missing-same-pci",
        "both-uuids-missing-different-hosts",
        "both-uuids-missing-different-pci",
    ),
)
def test_rank_audit_uses_pci_when_uuid_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    hosts: tuple[str, str],
    uuids: tuple[str | None, str | None],
    pci_bus_ids: tuple[str, str],
    expected_duplicates: list[int],
) -> None:
    monkeypatch.delenv(
        "CUPHOTON_ALLOCATED_CUDA_VISIBLE_DEVICES", raising=False
    )
    identities = tuple(
        {
            **_gpu(),
            "uuid": uuid,
            "pci_bus_id": pci_bus_id,
        }
        for uuid, pci_bus_id in zip(uuids, pci_bus_ids, strict=True)
    )

    audit = _audit_gpu_identities(tmp_path, identities, hosts)

    if expected_duplicates:
        assert audit["ok"] is False
    else:
        assert audit["ok"] is True
    assert audit["duplicate_physical_gpu_ranks"] == expected_duplicates


def test_rank_audit_rejects_malformed_counts_timing_and_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(
        "CUPHOTON_ALLOCATED_CUDA_VISIBLE_DEVICES", raising=False
    )
    run_dir = tmp_path / "run"
    (run_dir / "items").mkdir(parents=True)
    (run_dir / "records").mkdir()
    (run_dir / "ranks").mkdir()
    shard = (WorkItem("one", {"id": "one"}, 1),)
    result = mpi._execute_rank(
        run_id="malformed-rank",
        run_dir=run_dir,
        manifest_sha256="manifest",
        context=mpi._RankContext(
            0, 0, 1, "host", "test", launch_id="test-launch"
        ),
        items=shard,
        options=_options(),
        visibility="GPU-0",
        setup_error=None,
        item_runner=_item_runner,
        gpu_identity_loader=lambda backend: _gpu(),
    )
    records, errors = mpi._read_mappings(run_dir / "records")
    assert errors == []
    malformed = {
        **result,
        "success_count": 2,
        "failed_count": 1,
        "rank_wall_sec": float("nan"),
        "error": {"type": "RuntimeError", "message": "contradiction"},
        "provenance": {
            **result["provenance"],
            "launcher": "wrong-launcher",
            "cuda_visible_devices": "0,1",
            "gpu": {**result["provenance"]["gpu"], "device_index": 1},
        },
    }

    audit = mpi._audit_ranks(
        "malformed-rank",
        "manifest",
        "cupy",
        (shard,),
        (malformed,),
        records,
    )

    assert audit["ok"] is False
    fields = audit["mismatched_ranks"][0]["fields"]
    for field in (
        "error",
        "failed_count",
        "provenance",
        "rank_wall_sec",
        "success_count",
    ):
        assert field in fields

    bad_context = {**result, "context": {**result["context"], "rank": 1}}
    context_audit = mpi._audit_ranks(
        "malformed-rank",
        "manifest",
        "cupy",
        (shard,),
        (bad_context,),
        records,
    )
    context_fields = context_audit["mismatched_ranks"][0]["fields"]
    assert "context" in context_fields
    assert "provenance" in context_fields

    bad_local_rank = {
        **result,
        "context": {**result["context"], "local_rank": 1},
        "provenance": {**result["provenance"], "local_rank": 1},
    }
    local_rank_audit = mpi._audit_ranks(
        "malformed-rank",
        "manifest",
        "cupy",
        (shard,),
        (bad_local_rank,),
        records,
    )
    local_rank_fields = local_rank_audit["mismatched_ranks"][0]["fields"]
    assert "context" in local_rank_fields
    assert "provenance" in local_rank_fields


def test_terminal_record_audit_checks_execution_identity() -> None:
    shard = (WorkItem("one", {"id": "one"}, 7),)
    record = {
        "schema": "wrong",
        "run_id": "wrong",
        "manifest_sha256": "wrong",
        "item_id": "one",
        "rank": 1,
        "weight_bytes": 8,
        "requested_backend": "numba-cuda",
        "backend": "numba-cuda",
        "status": "success",
        "error": {"type": "RuntimeError", "message": "contradiction"},
    }

    errors = mpi._terminal_record_errors(
        "expected-run", "expected-manifest", "cupy", (shard,), (record,)
    )

    assert len(errors) == 1
    for field in (
        "schema",
        "run_id",
        "manifest_sha256",
        "rank",
        "weight_bytes",
        "requested_backend",
        "backend",
        "completed_at_utc",
        "device",
        "error",
        "run_dir",
        "runtime",
        "started_at_utc",
        "summary_path",
        "timings_sec",
        "wall_sec",
        "worker_seconds",
    ):
        assert field in errors[0]["message"]


@pytest.mark.parametrize(
    ("field", "value"),
    (("rank", 1), ("weight_bytes", 8)),
)
def test_terminal_record_audit_reports_assignment_fields_independently(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    value: int,
) -> None:
    monkeypatch.delenv(
        "CUPHOTON_ALLOCATED_CUDA_VISIBLE_DEVICES", raising=False
    )
    run_dir = tmp_path / "run"
    (run_dir / "items").mkdir(parents=True)
    (run_dir / "records").mkdir()
    (run_dir / "ranks").mkdir()
    shard = (WorkItem("one", {"id": "one"}, 7),)
    mpi._execute_rank(
        run_id="assignment",
        run_dir=run_dir,
        manifest_sha256="manifest",
        context=mpi._RankContext(
            0, 0, 1, "host", "test", launch_id="test-launch"
        ),
        items=shard,
        options=_options(),
        visibility="GPU-0",
        setup_error=None,
        item_runner=_item_runner,
        gpu_identity_loader=lambda backend: _gpu(),
    )
    record = mpi.read_json_mapping(run_dir / "records" / "one.json")
    record[field] = value

    errors = mpi._terminal_record_errors(
        "assignment", "manifest", "cupy", (shard,), (record,)
    )

    assert errors == [
        {
            "item_id": "one",
            "type": "InvalidTerminalRecord",
            "message": f"record 0 has invalid field(s): {field}",
        }
    ]


def test_singleton_collective_run_is_lazy_root_only_and_audited(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _mpi_environment(monkeypatch)
    _remove_cuda_modules(monkeypatch)
    manifest = _write_manifest(tmp_path)
    comm = _SingletonComm()
    load_seen = False

    def load_api():
        nonlocal load_seen
        load_seen = True
        assert mpi._prebound_visibility(dict(mpi.os.environ)) == "GPU-0"
        return mpi._MPIAPI(
            SimpleNamespace(COMM_TYPE_SHARED=1),
            comm,
            "4.test",
            "Test MPI",
        )

    monkeypatch.setattr(mpi, "_load_mpi_api", load_api)
    monkeypatch.setattr(mpi, "run_image_pair_item", _item_runner)
    monkeypatch.setattr(mpi, "_gpu_identity", lambda backend: _gpu())

    result = mpi.run_mpi_image_pair_batch(
        manifest_path=manifest,
        output_root=tmp_path / "runs",
        run_id="mpi-success",
        aggregation_mode="mpi",
        rank_timeout_sec=None,
        attempt_id=None,
        options=_options(),
    )

    assert load_seen is True
    assert result is not None
    assert result.status == "success"
    assert result.to_dict()["executor"] == "mpi"
    assert result.summary["terminal_record_audit"]["ok"] is True
    assert result.summary["rank_result_audit"]["ok"] is True
    run_record = json.loads((result.run_dir / "run.json").read_text())
    assert "rank_timeout_sec" not in run_record
    assert run_record["rank_setup_timeout_sec"] == 600.0
    assert result.summary["rank_setup_timeout_sec"] == 600.0
    assert run_record["mpi4py_version"] == "4.test"


def test_collective_generated_run_uses_fully_resolved_output_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _mpi_environment(monkeypatch)
    _remove_cuda_modules(monkeypatch)
    manifest = _write_manifest(tmp_path)
    real_root = tmp_path / "real-results"
    real_root.mkdir()
    alias_root = tmp_path / "results-alias"
    alias_root.symlink_to(real_root, target_is_directory=True)
    monkeypatch.setattr(mpi, "new_run_id", lambda prefix: "generated-run")
    monkeypatch.setattr(
        mpi,
        "_load_mpi_api",
        lambda: mpi._MPIAPI(
            SimpleNamespace(COMM_TYPE_SHARED=1),
            _SingletonComm(),
            "4.test",
            "Test MPI",
        ),
    )
    monkeypatch.setattr(mpi, "run_image_pair_item", _item_runner)
    monkeypatch.setattr(mpi, "_gpu_identity", lambda backend: _gpu())

    result = mpi.run_mpi_image_pair_batch(
        manifest_path=manifest,
        output_root=alias_root,
        run_id=None,
        aggregation_mode="mpi",
        rank_timeout_sec=None,
        attempt_id=None,
        options=_options(),
    )

    assert result is not None
    assert result.run_id == "generated-run"
    assert result.run_dir == (real_root / "generated-run").resolve()
    ready = mpi.read_json_mapping(result.run_dir / ".ready.json")
    assert ready["resolved_run_dir"] == str(result.run_dir)
    assert isinstance(ready["shared_nonce"], str)
    assert ready["shared_nonce"]


def test_collective_rejects_dangling_run_directory_symlink(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _mpi_environment(monkeypatch)
    _remove_cuda_modules(monkeypatch)
    manifest = _write_manifest(tmp_path)
    output_root = tmp_path / "runs"
    output_root.mkdir()
    redirected_run = tmp_path / "redirected" / "escaped-run"
    run_link = output_root / "known-run"
    run_link.symlink_to(redirected_run, target_is_directory=True)
    monkeypatch.setattr(
        mpi,
        "_load_mpi_api",
        lambda: mpi._MPIAPI(
            SimpleNamespace(COMM_TYPE_SHARED=1),
            _SingletonComm(),
            "4.test",
            "Test MPI",
        ),
    )

    with pytest.raises(
        RuntimeError, match="cannot prepare MPI run: FileExistsError"
    ):
        mpi.run_mpi_image_pair_batch(
            manifest_path=manifest,
            output_root=output_root,
            run_id="known-run",
            aggregation_mode="mpi",
            rank_timeout_sec=None,
            attempt_id=None,
            options=_options(),
        )

    assert run_link.is_symlink()
    assert not redirected_run.exists()


def test_file_run_rejects_dangling_run_directory_symlink(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _files_environment(monkeypatch)
    _remove_cuda_modules(monkeypatch)
    manifest = _write_manifest(tmp_path)
    output_root = tmp_path / "runs"
    output_root.mkdir()
    redirected_run = tmp_path / "redirected" / "escaped-run"
    run_link = output_root / "known-run"
    run_link.symlink_to(redirected_run, target_is_directory=True)

    with pytest.raises(
        RuntimeError, match="cannot prepare MPI file run: FileExistsError"
    ):
        mpi.run_mpi_image_pair_batch(
            manifest_path=manifest,
            output_root=output_root,
            run_id="known-run",
            aggregation_mode="files",
            rank_timeout_sec=1.0,
            attempt_id="symlink-attempt",
            options=_options(),
        )

    marker = mpi._attempt_path(
        output_root / "known-run", "known-run", "symlink-attempt"
    )
    assert marker.parent == output_root / ".mpi-attempts"
    assert mpi.read_json_mapping(marker)["status"] == "failed"
    assert run_link.is_symlink()
    assert not redirected_run.exists()


def test_collective_labels_run_directory_resolution_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FailingOutputRoot:
        def expanduser(self):
            return self

        def __truediv__(self, value):
            return self

        def resolve(self, *, strict: bool):
            assert strict is False
            raise OSError("synthetic output path failure")

    _mpi_environment(monkeypatch)
    _remove_cuda_modules(monkeypatch)
    manifest = _write_manifest(tmp_path)
    monkeypatch.setattr(
        mpi,
        "_load_mpi_api",
        lambda: mpi._MPIAPI(
            SimpleNamespace(COMM_TYPE_SHARED=1),
            _SingletonComm(),
            "4.test",
            "Test MPI",
        ),
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "MPI run-directory resolution failed: rank 0: OSError: "
            "synthetic output path failure"
        ),
    ):
        mpi.run_mpi_image_pair_batch(
            manifest_path=manifest,
            output_root=FailingOutputRoot(),
            run_id="path-failure",
            aggregation_mode="mpi",
            rank_timeout_sec=None,
            attempt_id=None,
            options=_options(),
        )


def test_collective_prepare_rejects_peer_without_shared_run_directory(
    tmp_path: Path,
) -> None:
    class MissingPeerComm(_SingletonComm):
        def Get_size(self) -> int:
            return 2

        def gather(self, value, root: int):
            assert root == 0
            return [
                value,
                {
                    "rank": 1,
                    "error": {
                        "type": "FileNotFoundError",
                        "message": "peer cannot read .ready.json",
                    },
                },
            ]

    manifest = mpi.load_image_pair_manifest(
        _write_manifest(tmp_path, count=2)
    )
    shards = mpi.partition_byte_balanced(manifest.work_items(), 2)
    run_dir = tmp_path / "runs" / "not-shared"
    context = mpi._RankContext(
        0, 0, 2, "host", "test", launch_id="test-launch"
    )
    api = mpi._MPIAPI(None, MissingPeerComm(), "4.test", "Test MPI")

    with pytest.raises(
        RuntimeError,
        match="shared run-directory validation.*peer cannot read",
    ):
        mpi._mpi_prepare_run(
            api.comm,
            context,
            run_dir,
            "not-shared",
            mpi.timestamp_utc(),
            manifest,
            _options(),
            api,
            shards=shards,
            start=time.perf_counter(),
        )

    assert (run_dir / ".ready.json").is_file()
    summary = mpi.read_json_mapping(run_dir / "summary.json")
    assert summary["status"] == "failed"
    assert summary["rank_result_audit"]["missing_ranks"] == [0, 1]
    assert summary["terminal_record_audit"]["missing_item_ids"] == [
        "pair-0",
        "pair-1",
    ]
    assert len(summary["aggregation_errors"]) == 1
    assert summary["aggregation_errors"][0]["phase"] == "setup"
    assert (
        "peer cannot read .ready.json"
        in summary["aggregation_errors"][0]["message"]
    )


def test_collective_partial_initialization_persists_terminal_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = mpi.load_image_pair_manifest(_write_manifest(tmp_path))
    shards = mpi.partition_byte_balanced(manifest.work_items(), 1)
    run_dir = tmp_path / "runs" / "partial-setup"
    context = mpi._RankContext(
        0, 0, 1, "host", "test", launch_id="test-launch"
    )
    api = mpi._MPIAPI(None, _SingletonComm(), "4.test", "Test MPI")
    original_write = mpi.atomic_write_json

    def fail_input_identity(path: Path, payload, **kwargs) -> None:
        if path.name == "input-identity.json":
            raise OSError("synthetic input identity failure")
        original_write(path, payload, **kwargs)

    monkeypatch.setattr(mpi, "atomic_write_json", fail_input_identity)

    with pytest.raises(
        RuntimeError,
        match="cannot prepare MPI run: OSError: synthetic input identity",
    ):
        mpi._mpi_prepare_run(
            api.comm,
            context,
            run_dir,
            "partial-setup",
            mpi.timestamp_utc(),
            manifest,
            _options(),
            api,
            shards=shards,
            start=time.perf_counter(),
        )

    summary = mpi.read_json_mapping(run_dir / "summary.json")
    assert summary["status"] == "failed"
    assert summary["aggregation_errors"] == [
        {
            "phase": "setup",
            "type": "OSError",
            "message": "synthetic input identity failure",
        }
    ]
    summary_bytes = (run_dir / "summary.json").read_bytes()

    with pytest.raises(
        RuntimeError, match="cannot prepare MPI run: FileExistsError"
    ):
        mpi._mpi_prepare_run(
            api.comm,
            context,
            run_dir,
            "partial-setup",
            mpi.timestamp_utc(),
            manifest,
            _options(),
            api,
            shards=shards,
            start=time.perf_counter(),
        )

    assert (run_dir / "summary.json").read_bytes() == summary_bytes


def test_failed_collective_root_persists_failed_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _mpi_environment(monkeypatch)
    _remove_cuda_modules(monkeypatch)
    manifest = _write_manifest(tmp_path)
    monkeypatch.setattr(
        mpi,
        "_load_mpi_api",
        lambda: mpi._MPIAPI(
            SimpleNamespace(COMM_TYPE_SHARED=1),
            _SingletonComm(),
            "4.test",
            "Test MPI",
        ),
    )
    monkeypatch.setattr(mpi, "_gpu_identity", lambda backend: _gpu())

    def fail(item, output, options):
        raise ValueError("synthetic failure")

    monkeypatch.setattr(mpi, "run_image_pair_item", fail)

    result = mpi.run_mpi_image_pair_batch(
        manifest_path=manifest,
        output_root=tmp_path / "runs",
        run_id="mpi-failed",
        aggregation_mode="mpi",
        rank_timeout_sec=None,
        attempt_id=None,
        options=_options(),
    )

    assert result is not None
    assert result.status == "failed"
    assert result.summary_path.is_file()
    assert result.summary["terminal_record_audit"]["failed_item_ids"] == [
        "pair-0"
    ]


def test_file_setup_validates_nonroot_configuration(
    tmp_path: Path,
) -> None:
    options = _options()
    manifest = mpi.load_image_pair_manifest(
        _write_manifest(tmp_path, count=2)
    )
    run_id = "identity-consensus"
    attempt_id = "attempt-one"
    timeout = 1.0
    run_dir = tmp_path / "runs" / run_id
    root = mpi._RankContext(0, 0, 2, "host", "test", launch_id="test-launch")
    peer = mpi._RankContext(1, 1, 2, "host", "test", launch_id="test-launch")

    def prepare(context: mpi._RankContext):
        return mpi._file_prepare_run(
            context,
            run_dir,
            run_id,
            attempt_id,
            timeout,
            mpi.timestamp_utc(),
            manifest,
            options,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        setup_results = list(pool.map(prepare, (root, peer)))

    assert setup_results == [None, None]
    assert (
        mpi.read_json_mapping(run_dir / "input-identity.json")
        == manifest.input_identity_payload()
    )
    divergent_options = replace(options, backend="numba-cuda")
    setup_error = mpi._file_ready_error(
        peer,
        run_dir,
        run_id,
        attempt_id,
        timeout,
        manifest,
        divergent_options,
    )

    assert setup_error is not None
    assert setup_error["type"] == "RunSetupMismatchError"
    assert setup_error["message"] == "file setup differs in: options"

    completion_timeout_error = mpi._file_ready_error(
        peer,
        run_dir,
        run_id,
        attempt_id,
        timeout,
        manifest,
        options,
        rank_timeout_sec=2.0,
    )
    assert completion_timeout_error is not None
    assert completion_timeout_error["message"] == (
        "file setup differs in: rank_timeout_sec"
    )

    setup_timeout_error = mpi._file_ready_error(
        peer,
        run_dir,
        run_id,
        attempt_id,
        2.0,
        manifest,
        options,
        rank_timeout_sec=timeout,
    )
    assert setup_timeout_error is not None
    assert setup_timeout_error["message"] == (
        "file setup differs in: rank_setup_timeout_sec"
    )

    rank_dir = mpi._file_rank_staging_dir(
        run_dir, run_id, attempt_id, peer.rank
    )
    shards = mpi.partition_byte_balanced(
        manifest.work_items(), peer.world_size
    )
    rank_result = mpi._execute_rank(
        run_id=run_id,
        run_dir=rank_dir,
        manifest_sha256=manifest.sha256,
        context=peer,
        items=shards[peer.rank],
        options=divergent_options,
        visibility="GPU-1",
        setup_error=setup_error,
        item_runner=lambda item, output, batch_options: pytest.fail(
            "item execution started after file-setup mismatch"
        ),
        gpu_identity_loader=lambda backend: pytest.fail(
            "GPU initialization started after file-setup mismatch"
        ),
        attempt_id=attempt_id,
    )

    assert rank_result["status"] == "failed"
    assert rank_result["error"] == setup_error


def test_file_setup_exception_publishes_nonroot_completion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _files_environment(monkeypatch)
    _remove_cuda_modules(monkeypatch)
    run_id = "ready-validation-failure"
    attempt_id = "attempt-one"
    manifest_path = _write_manifest(tmp_path, count=2)
    manifest = mpi.load_image_pair_manifest(manifest_path)
    output_root = tmp_path / "runs"
    run_dir = output_root / run_id
    for name in ("items", "ranks", "records"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    mpi._attempt_staging_dir(run_dir, run_id, attempt_id).mkdir(parents=True)
    rank_dir = mpi._file_rank_staging_dir(run_dir, run_id, attempt_id, 1)
    mpi._prepare_file_rank_staging(rank_dir)
    mpi.atomic_write_json(
        mpi._attempt_path(run_dir, run_id, attempt_id),
        {
            "run_id": run_id,
            "attempt_id": attempt_id,
            "launch_id": "test-launch",
            "manifest_sha256": manifest.sha256,
            "status": "ready",
            "terminal": False,
            "error": None,
        },
    )
    monkeypatch.setattr(
        mpi,
        "_environment_context",
        lambda environ: mpi._RankContext(
            1, 1, 2, "host", "test", launch_id="test-launch"
        ),
    )

    def fail_file_setup(*args, **kwargs):
        raise OSError("synthetic ready validation failure")

    monkeypatch.setattr(mpi, "_file_prepare_run", fail_file_setup)
    monkeypatch.setattr(
        mpi,
        "run_image_pair_item",
        lambda *args, **kwargs: pytest.fail(
            "item execution started after file setup failure"
        ),
    )
    monkeypatch.setattr(
        mpi,
        "_gpu_identity",
        lambda backend: pytest.fail(
            "GPU initialization started after file setup failure"
        ),
    )

    result = mpi.run_mpi_image_pair_batch(
        manifest_path=manifest_path,
        output_root=output_root,
        run_id=run_id,
        aggregation_mode="files",
        rank_timeout_sec=2.0,
        attempt_id=attempt_id,
        options=_options(),
        rank_setup_timeout_sec=1.0,
    )

    rank_result = mpi.read_json_mapping(rank_dir / "ranks" / "rank-0001.json")
    assert result is None
    assert rank_result["status"] == "failed"
    assert rank_result["error"] == {
        "type": "OSError",
        "message": "synthetic ready validation failure",
    }
    assert (rank_dir / ".complete.json").is_file()


@pytest.mark.parametrize(
    "marker_update",
    [
        None,
        {"run_id": "other"},
        {"attempt_id": "other"},
        {"manifest_sha256": "other"},
        {"status": "starting"},
        {"status": "failed", "terminal": True},
    ],
)
def test_file_setup_exception_rejects_inactive_attempt(
    marker_update, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _files_environment(monkeypatch)
    _remove_cuda_modules(monkeypatch)
    run_id = "inactive-setup-failure"
    attempt_id = "attempt-one"
    manifest_path = _write_manifest(tmp_path, count=2)
    manifest = mpi.load_image_pair_manifest(manifest_path)
    output_root = tmp_path / "runs"
    run_dir = output_root / run_id
    for name in ("items", "ranks", "records"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    mpi._attempt_staging_dir(run_dir, run_id, attempt_id).mkdir(parents=True)
    if marker_update is not None:
        marker = {
            "run_id": run_id,
            "attempt_id": attempt_id,
            "launch_id": "test-launch",
            "manifest_sha256": manifest.sha256,
            "status": "ready",
            "terminal": False,
            "error": None,
        }
        marker.update(marker_update)
        mpi.atomic_write_json(
            mpi._attempt_path(run_dir, run_id, attempt_id), marker
        )
    monkeypatch.setattr(
        mpi,
        "_environment_context",
        lambda environ: mpi._RankContext(
            1, 1, 2, "host", "test", launch_id="test-launch"
        ),
    )

    def fail_file_setup(*args, **kwargs):
        raise OSError("synthetic ready validation failure")

    monkeypatch.setattr(mpi, "_file_prepare_run", fail_file_setup)

    with pytest.raises(OSError, match="synthetic ready validation failure"):
        mpi.run_mpi_image_pair_batch(
            manifest_path=manifest_path,
            output_root=output_root,
            run_id=run_id,
            aggregation_mode="files",
            rank_timeout_sec=2.0,
            attempt_id=attempt_id,
            options=_options(),
            rank_setup_timeout_sec=1.0,
        )

    rank_dir = mpi._file_rank_staging_dir(run_dir, run_id, attempt_id, 1)
    assert not rank_dir.exists()


@pytest.mark.parametrize(
    "artifact",
    [
        "completion",
        "record",
        "rank",
        "item",
        "nested",
        "dangling-completion",
        "dangling-item",
        "dangling-items",
        "dangling-rank-dir",
    ],
)
def test_file_promotion_rejects_symlink_artifacts(
    artifact: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_id = f"symlink-{artifact}"
    attempt_id = "attempt-one"
    manifest, run_dir, rank_dir, shards, _ = _stage_file_rank(
        tmp_path, run_id, attempt_id
    )
    item_id = shards[0][0].item_id
    paths = {
        "completion": rank_dir / ".complete.json",
        "record": rank_dir / "records" / f"{item_id}.json",
        "rank": rank_dir / "ranks" / "rank-0000.json",
        "item": rank_dir / "items" / item_id,
        "dangling-completion": rank_dir / ".complete.json",
        "dangling-item": rank_dir / "items" / item_id,
        "dangling-items": rank_dir / "items",
        "dangling-rank-dir": rank_dir,
    }
    if artifact == "nested":
        (rank_dir / "items" / item_id / "nested-link").symlink_to(tmp_path)
    else:
        source = paths[artifact]
        target_root = (
            rank_dir.parent if artifact == "dangling-rank-dir" else rank_dir
        )
        target = target_root / f"{artifact}-target"
        source.rename(target)
        link_target = (
            target_root / f"{artifact}-missing"
            if artifact.startswith("dangling-")
            else target
        )
        source.symlink_to(
            link_target,
            target_is_directory=artifact
            in {
                "item",
                "dangling-item",
                "dangling-items",
                "dangling-rank-dir",
            },
        )

    monkeypatch.setattr(
        mpi.time,
        "sleep",
        lambda delay: pytest.fail(
            f"deterministic invalid evidence slept for {delay} seconds"
        ),
    )

    ranks, errors = mpi._publish_completed_file_ranks(
        run_dir,
        run_id,
        attempt_id,
        manifest.sha256,
        shards,
        0.01,
    )

    assert ranks == []
    assert errors
    assert errors[0]["type"] == "ValueError"
    for name in ("items", "ranks", "records"):
        assert list((run_dir / name).iterdir()) == []


@pytest.mark.parametrize(
    "case", ["malformed-record-missing-rank", "malformed-rank-missing-item"]
)
def test_file_promotion_validates_visible_json_before_retry(
    case: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_id = f"visible-invalid-{case}"
    manifest, run_dir, rank_dir, shards, _ = _stage_file_rank(
        tmp_path, run_id, "attempt-one"
    )
    item_id = shards[0][0].item_id
    rank_path = rank_dir / "ranks" / "rank-0000.json"
    record_path = rank_dir / "records" / f"{item_id}.json"
    if case == "malformed-record-missing-rank":
        record_path.write_text("{", encoding="utf-8")
        rank_path.rename(rank_path.with_suffix(".missing"))
    else:
        rank_path.write_text("{", encoding="utf-8")
        item_dir = rank_dir / "items" / item_id
        item_dir.rename(item_dir.with_name(f"{item_id}-missing"))

    monkeypatch.setattr(
        mpi.time,
        "sleep",
        lambda delay: pytest.fail(
            f"visible invalid evidence slept for {delay} seconds"
        ),
    )

    ranks, errors = mpi._publish_completed_file_ranks(
        run_dir,
        run_id,
        "attempt-one",
        manifest.sha256,
        shards,
        0.01,
        artifact_visibility_timeout=10.0,
    )

    assert ranks == []
    assert errors
    assert errors[0]["type"] in {"JSONDecodeError", "ValueError"}
    for name in ("items", "ranks", "records"):
        assert list((run_dir / name).iterdir()) == []


def test_file_promotion_matches_completion_and_preserves_failed_evidence(
    tmp_path: Path,
) -> None:
    def fail(item, output, options):
        raise ValueError("synthetic failure")

    manifest, run_dir, _, shards, _ = _stage_file_rank(
        tmp_path, "failed-rank", "attempt-failed", fail
    )
    ranks, errors = mpi._publish_completed_file_ranks(
        run_dir,
        "failed-rank",
        "attempt-failed",
        manifest.sha256,
        shards,
        0.01,
    )
    assert errors == []
    assert ranks[0]["status"] == "failed"
    assert (run_dir / "records" / "pair-0.json").is_file()
    assert (run_dir / "ranks" / "rank-0000.json").is_file()

    manifest, run_dir, rank_dir, shards, _ = _stage_file_rank(
        tmp_path, "mismatched-rank", "attempt-mismatch", fail
    )
    completion_path = rank_dir / ".complete.json"
    completion = mpi.read_json_mapping(completion_path)
    completion["status"] = "success"
    mpi.atomic_write_json(completion_path, completion)
    ranks, errors = mpi._publish_completed_file_ranks(
        run_dir,
        "mismatched-rank",
        "attempt-mismatch",
        manifest.sha256,
        shards,
        0.01,
    )
    assert ranks == []
    assert any(error["type"] == "InvalidRankCompletion" for error in errors)
    for name in ("items", "ranks", "records"):
        assert list((run_dir / name).iterdir()) == []


def test_file_promotion_skips_declared_record_write_failure_without_wait(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    original_write = mpi.atomic_write_json

    def fail_one_record(path: Path, payload, **kwargs) -> None:
        if path.parent.name == "records" and path.name == "pair-0.json":
            raise OSError("synthetic record write failure")
        original_write(path, payload, **kwargs)

    monkeypatch.setattr(mpi, "atomic_write_json", fail_one_record)
    manifest, run_dir, rank_dir, shards, result = _stage_file_rank(
        tmp_path,
        "declared-record-write-failure",
        "attempt-write-failure",
        count=2,
    )
    monkeypatch.setattr(
        mpi.time,
        "sleep",
        lambda delay: pytest.fail(
            f"declared permanent write failure slept for {delay} seconds"
        ),
    )

    ranks, errors = mpi._publish_completed_file_ranks(
        run_dir,
        "declared-record-write-failure",
        "attempt-write-failure",
        manifest.sha256,
        shards,
        600.0,
    )

    assert result["status"] == "failed"
    assert result["record_write_errors"] == [
        {
            "item_id": "pair-0",
            "type": "OSError",
            "message": "synthetic record write failure",
        }
    ]
    assert ranks == [result]
    assert errors == [
        {
            "rank": 0,
            "path": "records/pair-0.json",
            "type": "OSError",
            "message": "synthetic record write failure",
        }
    ]
    assert not (run_dir / "items" / "pair-0").exists()
    assert not (run_dir / "records" / "pair-0.json").exists()
    assert (rank_dir / "items" / "pair-0").is_dir()
    assert (run_dir / "items" / "pair-1" / "summary.json").is_file()
    assert (run_dir / "records" / "pair-1.json").is_file()
    assert (run_dir / "ranks" / "rank-0000.json").is_file()


@pytest.mark.parametrize("record_failure", [False, True])
def test_file_promotion_uses_receipt_when_rank_artifact_write_fails(
    record_failure: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original_write = mpi.atomic_write_json

    def fail_rank_and_record(path: Path, payload, **kwargs) -> None:
        if (
            record_failure
            and path.parent.name == "records"
            and path.name == "pair-0.json"
        ):
            raise OSError("synthetic record write failure")
        if path.parent.name == "ranks" and path.name == "rank-0000.json":
            raise OSError("synthetic rank write failure")
        original_write(path, payload, **kwargs)

    monkeypatch.setattr(mpi, "atomic_write_json", fail_rank_and_record)
    run_id = f"rank-artifact-write-failure-{record_failure}"
    manifest, run_dir, rank_dir, shards, result = _stage_file_rank(
        tmp_path,
        run_id,
        "attempt-write-failure",
        count=2,
    )
    completion = mpi.read_json_mapping(rank_dir / ".complete.json")
    monkeypatch.setattr(
        mpi.time,
        "sleep",
        lambda delay: pytest.fail(
            f"declared permanent write failure slept for {delay} seconds"
        ),
    )

    ranks, errors = mpi._publish_completed_file_ranks(
        run_dir,
        run_id,
        "attempt-write-failure",
        manifest.sha256,
        shards,
        600.0,
    )

    assert result["status"] == "failed"
    assert completion["artifact_error"] == {
        "type": "OSError",
        "message": "synthetic rank write failure",
    }
    expected_record_errors = (
        [
            {
                "item_id": "pair-0",
                "type": "OSError",
                "message": "synthetic record write failure",
            }
        ]
        if record_failure
        else []
    )
    assert completion["record_write_errors"] == expected_record_errors
    assert ranks == []
    expected_errors = [
        {
            "rank": 0,
            "path": "ranks/rank-0000.json",
            "type": "OSError",
            "message": "synthetic rank write failure",
        }
    ]
    if record_failure:
        expected_errors.append(
            {
                "rank": 0,
                "path": "records/pair-0.json",
                "type": "OSError",
                "message": "synthetic record write failure",
            }
        )
    assert errors == expected_errors
    for name in ("items", "ranks", "records"):
        assert list((run_dir / name).iterdir()) == []


def test_file_receipt_retains_record_error_when_rank_artifact_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    original_write = mpi.atomic_write_json

    def fail_one_record(path: Path, payload, **kwargs) -> None:
        if path.parent.name == "records" and path.name == "pair-0.json":
            raise OSError("synthetic record write failure")
        original_write(path, payload, **kwargs)

    monkeypatch.setattr(mpi, "atomic_write_json", fail_one_record)
    manifest, run_dir, rank_dir, shards, _ = _stage_file_rank(
        tmp_path,
        "missing-rank-after-receipt",
        "attempt-write-failure",
    )
    (rank_dir / "ranks" / "rank-0000.json").unlink()
    monkeypatch.setattr(
        mpi.time,
        "sleep",
        lambda delay: pytest.fail(f"missing rank slept for {delay} seconds"),
    )

    ranks, errors = mpi._publish_completed_file_ranks(
        run_dir,
        "missing-rank-after-receipt",
        "attempt-write-failure",
        manifest.sha256,
        shards,
        0.01,
        artifact_visibility_timeout=0.0,
    )

    assert ranks == []
    assert {
        (error.get("path"), error["type"], error["message"])
        for error in errors
    } == {
        (
            "records/pair-0.json",
            "OSError",
            "synthetic record write failure",
        ),
        (
            None,
            "FileNotFoundError",
            "completed rank artifacts did not become visible: "
            "ranks/rank-0000.json",
        ),
    }


def test_invalid_completion_retains_declared_record_write_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    original_write = mpi.atomic_write_json

    def fail_one_record(path: Path, payload, **kwargs) -> None:
        if path.parent.name == "records" and path.name == "pair-0.json":
            raise OSError("synthetic record write failure")
        original_write(path, payload, **kwargs)

    monkeypatch.setattr(mpi, "atomic_write_json", fail_one_record)
    manifest, run_dir, rank_dir, shards, _ = _stage_file_rank(
        tmp_path,
        "invalid-completion-write-failure",
        "attempt-invalid-completion",
        count=2,
    )
    completion_path = rank_dir / ".complete.json"
    completion = mpi.read_json_mapping(completion_path)
    completion["status"] = "success"
    mpi.atomic_write_json(completion_path, completion)

    ranks, errors = mpi._publish_completed_file_ranks(
        run_dir,
        "invalid-completion-write-failure",
        "attempt-invalid-completion",
        manifest.sha256,
        shards,
        0.01,
    )

    assert ranks == []
    assert {(error["type"], error["message"]) for error in errors} == {
        ("InvalidRankCompletion", "invalid field(s): status"),
        ("OSError", "synthetic record write failure"),
    }
    for name in ("items", "ranks", "records"):
        assert list((run_dir / name).iterdir()) == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", []),
        ("artifact_error", {"type": "", "message": "bad"}),
        (
            "record_write_errors",
            [
                {
                    "item_id": "outside-shard",
                    "type": "OSError",
                    "message": "bad",
                }
            ],
        ),
    ],
)
def test_file_promotion_rejects_invalid_completion_receipt_without_wait(
    field: str,
    value,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest, run_dir, rank_dir, shards, _ = _stage_file_rank(
        tmp_path, f"invalid-completion-{field}", "attempt-invalid"
    )
    completion_path = rank_dir / ".complete.json"
    completion = mpi.read_json_mapping(completion_path)
    completion[field] = value
    mpi.atomic_write_json(completion_path, completion)
    rank_path = rank_dir / "ranks" / "rank-0000.json"
    rank_path.rename(rank_path.with_suffix(".missing"))
    monkeypatch.setattr(
        mpi.time,
        "sleep",
        lambda delay: pytest.fail(
            f"invalid completion evidence slept for {delay} seconds"
        ),
    )

    ranks, errors = mpi._publish_completed_file_ranks(
        run_dir,
        f"invalid-completion-{field}",
        "attempt-invalid",
        manifest.sha256,
        shards,
        600.0,
    )

    assert ranks == []
    completion_errors = [
        error for error in errors if error["type"] == "InvalidRankCompletion"
    ]
    assert len(completion_errors) == 1
    assert field in completion_errors[0]["message"]
    for name in ("items", "ranks", "records"):
        assert list((run_dir / name).iterdir()) == []


def test_file_completion_compares_write_errors_by_json_type(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    original_write = mpi.atomic_write_json

    def fail_one_record(path: Path, payload, **kwargs) -> None:
        if path.parent.name == "records" and path.name == "pair-0.json":
            raise OSError("synthetic record write failure")
        original_write(path, payload, **kwargs)

    monkeypatch.setattr(mpi, "atomic_write_json", fail_one_record)
    manifest, run_dir, rank_dir, shards, _ = _stage_file_rank(
        tmp_path,
        "type-sensitive-completion",
        "attempt-type-sensitive",
    )
    completion_path = rank_dir / ".complete.json"
    completion = mpi.read_json_mapping(completion_path)
    completion["record_write_errors"][0]["code"] = False
    mpi.atomic_write_json(completion_path, completion)
    rank_path = rank_dir / "ranks" / "rank-0000.json"
    rank_result = mpi.read_json_mapping(rank_path)
    rank_result["record_write_errors"][0]["code"] = 0
    mpi.atomic_write_json(rank_path, rank_result)

    ranks, errors = mpi._publish_completed_file_ranks(
        run_dir,
        "type-sensitive-completion",
        "attempt-type-sensitive",
        manifest.sha256,
        shards,
        0.01,
    )

    assert ranks == []
    assert any(
        error["type"] == "InvalidRankCompletion"
        and "record_write_errors" in error["message"]
        for error in errors
    )
    for name in ("items", "ranks", "records"):
        assert list((run_dir / name).iterdir()) == []


def test_file_completion_retains_both_mismatched_write_error_reports(
    tmp_path: Path,
) -> None:
    manifest, run_dir, rank_dir, shards, _ = _stage_file_rank(
        tmp_path,
        "mismatched-write-error-reports",
        "attempt-mismatched-reports",
        count=2,
    )
    shared_error = {
        "item_id": "pair-0",
        "type": "OSError",
        "message": "shared write failure",
    }
    completion_path = rank_dir / ".complete.json"
    completion = mpi.read_json_mapping(completion_path)
    completion["status"] = "failed"
    completion["record_write_errors"] = [
        shared_error,
        {
            "item_id": "pair-1",
            "type": "OSError",
            "message": "receipt write failure",
        },
    ]
    mpi.atomic_write_json(completion_path, completion)
    rank_path = rank_dir / "ranks" / "rank-0000.json"
    rank_result = mpi.read_json_mapping(rank_path)
    rank_result["status"] = "failed"
    rank_result["record_write_errors"] = [
        shared_error,
        {
            "item_id": "pair-1",
            "type": "OSError",
            "message": "rank artifact write failure",
        },
    ]
    mpi.atomic_write_json(rank_path, rank_result)

    ranks, errors = mpi._publish_completed_file_ranks(
        run_dir,
        "mismatched-write-error-reports",
        "attempt-mismatched-reports",
        manifest.sha256,
        shards,
        0.01,
    )

    assert ranks == []
    assert errors == [
        {
            "rank": 0,
            "path": "records/pair-0.json",
            "type": "OSError",
            "message": "shared write failure",
            "evidence_sources": [
                "completion_receipt",
                "rank_artifact",
            ],
        },
        {
            "rank": 0,
            "path": "records/pair-1.json",
            "type": "OSError",
            "message": "receipt write failure",
            "evidence_sources": ["completion_receipt"],
        },
        {
            "rank": 0,
            "path": "records/pair-1.json",
            "type": "OSError",
            "message": "rank artifact write failure",
            "evidence_sources": ["rank_artifact"],
        },
        {
            "rank": 0,
            "type": "InvalidRankCompletion",
            "message": "invalid field(s): record_write_errors",
        },
    ]
    for name in ("items", "ranks", "records"):
        assert list((run_dir / name).iterdir()) == []


def test_file_completion_reconciles_disjoint_missing_write_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest, run_dir, rank_dir, shards, _ = _stage_file_rank(
        tmp_path,
        "disjoint-missing-write-failures",
        "attempt-disjoint-missing",
        count=2,
    )
    completion_path = rank_dir / ".complete.json"
    completion = mpi.read_json_mapping(completion_path)
    completion["status"] = "failed"
    completion["record_write_errors"] = [
        {
            "item_id": "pair-0",
            "type": "OSError",
            "message": "receipt write failure",
        }
    ]
    mpi.atomic_write_json(completion_path, completion)
    rank_path = rank_dir / "ranks" / "rank-0000.json"
    rank_result = mpi.read_json_mapping(rank_path)
    rank_result["status"] = "failed"
    rank_result["record_write_errors"] = [
        {
            "item_id": "pair-1",
            "type": "OSError",
            "message": "rank artifact write failure",
        }
    ]
    mpi.atomic_write_json(rank_path, rank_result)
    for item_id in ("pair-0", "pair-1"):
        (rank_dir / "records" / f"{item_id}.json").unlink()
    monkeypatch.setattr(
        mpi.time,
        "sleep",
        lambda delay: pytest.fail(
            f"declared missing record slept for {delay} seconds"
        ),
    )

    ranks, errors = mpi._publish_completed_file_ranks(
        run_dir,
        "disjoint-missing-write-failures",
        "attempt-disjoint-missing",
        manifest.sha256,
        shards,
        600.0,
    )

    assert ranks == []
    assert errors == [
        {
            "rank": 0,
            "path": "records/pair-0.json",
            "type": "OSError",
            "message": "receipt write failure",
            "evidence_sources": ["completion_receipt"],
        },
        {
            "rank": 0,
            "path": "records/pair-1.json",
            "type": "OSError",
            "message": "rank artifact write failure",
            "evidence_sources": ["rank_artifact"],
        },
        {
            "rank": 0,
            "type": "InvalidRankCompletion",
            "message": "invalid field(s): record_write_errors",
        },
    ]
    assert rank_path.is_file()
    for name in ("items", "ranks", "records"):
        assert list((run_dir / name).iterdir()) == []


@pytest.mark.parametrize(
    "write_errors",
    [
        None,
        [None],
        [{"item_id": "other", "type": "OSError", "message": "bad"}],
        [
            {
                "item_id": "pair-0",
                "type": "OSError",
                "message": "first",
            },
            {
                "item_id": "pair-0",
                "type": "OSError",
                "message": "second",
            },
        ],
        [{"item_id": "pair-0", "type": "", "message": "bad"}],
    ],
)
def test_file_promotion_rejects_invalid_record_write_failures_without_wait(
    write_errors, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest, run_dir, rank_dir, shards, _ = _stage_file_rank(
        tmp_path, "invalid-record-write-errors", "attempt-invalid"
    )
    rank_path = rank_dir / "ranks" / "rank-0000.json"
    rank_result = mpi.read_json_mapping(rank_path)
    rank_result["record_write_errors"] = write_errors
    mpi.atomic_write_json(rank_path, rank_result)
    monkeypatch.setattr(
        mpi.time,
        "sleep",
        lambda delay: pytest.fail(
            f"invalid record-write evidence slept for {delay} seconds"
        ),
    )

    ranks, errors = mpi._publish_completed_file_ranks(
        run_dir,
        "invalid-record-write-errors",
        "attempt-invalid",
        manifest.sha256,
        shards,
        600.0,
    )

    assert ranks == []
    assert len(errors) == 1
    assert errors[0]["type"] == "ValueError"
    assert "InvalidRecordWrite" in errors[0]["message"]
    for name in ("items", "ranks", "records"):
        assert list((run_dir / name).iterdir()) == []


def test_file_promotion_stops_rank_after_first_rename_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest, run_dir, rank_dir, shards, _ = _stage_file_rank(
        tmp_path, "rename-failure", "attempt-rename"
    )
    attempted: list[tuple[Path, Path]] = []

    def fail_replace(source: Path, destination: Path) -> Path:
        attempted.append((source, destination))
        raise OSError("synthetic rename failure")

    monkeypatch.setattr(Path, "replace", fail_replace)

    ranks, errors = mpi._publish_completed_file_ranks(
        run_dir,
        "rename-failure",
        "attempt-rename",
        manifest.sha256,
        shards,
        0.01,
    )

    assert ranks == []
    assert len(errors) == 1
    assert errors[0]["type"] == "OSError"
    assert len(attempted) == 1
    item_id = shards[0][0].item_id
    assert attempted[0] == (
        rank_dir / "items" / item_id,
        run_dir / "items" / item_id,
    )
    assert (rank_dir / "records" / f"{item_id}.json").is_file()
    assert (rank_dir / "ranks" / "rank-0000.json").is_file()
    assert not (run_dir / "records" / f"{item_id}.json").exists()
    assert not (run_dir / "ranks" / "rank-0000.json").exists()


def test_file_promotion_retries_transient_staging_visibility(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest, run_dir, rank_dir, shards, _ = _stage_file_rank(
        tmp_path, "visibility-retry", "attempt-visibility"
    )
    original = mpi._missing_staged_rank_paths
    calls = 0

    def transient(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return ["ranks/rank-0000.json"]
        return original(*args, **kwargs)

    monkeypatch.setattr(mpi, "_missing_staged_rank_paths", transient)

    ranks, errors = mpi._publish_completed_file_ranks(
        run_dir,
        "visibility-retry",
        "attempt-visibility",
        manifest.sha256,
        shards,
        0.1,
        artifact_visibility_timeout=0.1,
    )

    assert errors == []
    assert len(ranks) == 1
    assert calls == 2
    assert not (rank_dir / "ranks" / "rank-0000.json").exists()
    assert (run_dir / "ranks" / "rank-0000.json").is_file()


def test_shared_filesystem_wait_uses_bounded_backoff(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clock = [0.0]
    sleeps: list[float] = []

    monkeypatch.setattr(mpi.time, "monotonic", lambda: clock[0])

    def advance(delay: float) -> None:
        sleeps.append(delay)
        clock[0] += delay

    monkeypatch.setattr(mpi.time, "sleep", advance)

    with pytest.raises(TimeoutError):
        mpi._wait_mapping(tmp_path / "missing.json", 3.0, {"ready"})

    assert sleeps[:5] == pytest.approx([0.05, 0.1, 0.2, 0.4, 0.8])
    assert max(sleeps) == 1.0
    assert sum(sleeps) == pytest.approx(3.0)


def test_collective_artifact_wait_absorbs_visibility_delay(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "collective"
    for name in ("items", "ranks", "records"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    shard = (WorkItem("pair-0", {"id": "pair-0"}, 1),)

    def publish() -> None:
        time.sleep(0.02)
        item_dir = run_dir / "items" / "pair-0"
        item_dir.mkdir()
        mpi.atomic_write_json(item_dir / "summary.json", {})
        mpi.atomic_write_json(
            run_dir / "records" / "pair-0.json", {"status": "success"}
        )
        mpi.atomic_write_json(run_dir / "ranks" / "rank-0000.json", {})

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(publish)
        assert mpi._wait_collective_artifacts(run_dir, (shard,), 0.5) == []
        future.result()


def test_collective_artifact_wait_reports_missing_paths(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "collective"
    for name in ("items", "ranks", "records"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    shard = (WorkItem("pair-0", {"id": "pair-0"}, 1),)

    errors = mpi._wait_collective_artifacts(run_dir, (shard,), 0.001)

    assert {error["message"] for error in errors} == {
        "collective artifact did not become visible: ranks/rank-0000.json",
        "collective artifact did not become visible: records/pair-0.json",
    }


@pytest.mark.parametrize("kind", ["symlink", "dangling-symlink", "directory"])
@pytest.mark.parametrize(
    "label",
    [
        "ranks/rank-0000.json",
        "records/pair-0.json",
        "items/pair-0/summary.json",
    ],
)
def test_collective_artifact_wait_rejects_nonregular_paths_immediately(
    kind: str, label: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_dir = tmp_path / "collective"
    for name in ("items/pair-0", "ranks", "records"):
        (run_dir / name).mkdir(parents=True)
    shard = (WorkItem("pair-0", {"id": "pair-0"}, 1),)
    reported = {"rank": 0, "record_write_errors": []}
    mpi.atomic_write_json(run_dir / "ranks/rank-0000.json", reported)
    mpi.atomic_write_json(
        run_dir / "records/pair-0.json", {"status": "success"}
    )
    mpi.atomic_write_json(run_dir / "items/pair-0/summary.json", {})
    path = run_dir / label
    path.unlink()
    target = tmp_path / "target.json"
    if kind == "directory":
        path.mkdir()
    else:
        if kind == "symlink":
            target.write_text("{}", encoding="utf-8")
        path.symlink_to(target)
    # Invalid rank/item records are permanent even while peers are missing.
    if label.startswith("ranks/"):
        (run_dir / "records/pair-0.json").unlink()
    elif label.startswith("records/"):
        (run_dir / "ranks/rank-0000.json").unlink()
    monkeypatch.setattr(
        mpi.time,
        "sleep",
        lambda delay: pytest.fail(
            f"non-regular collective artifact slept for {delay} seconds"
        ),
    )

    errors = mpi._wait_collective_artifacts(
        run_dir, (shard,), 600.0, rank_results=(reported,)
    )

    assert errors == [
        {
            "path": label,
            "type": "InvalidArtifact",
            "message": "collective artifact is not a regular file",
        }
    ]
    if kind == "symlink":
        assert target.read_text(encoding="utf-8") == "{}"


@pytest.mark.parametrize(
    "artifact", ["mismatched", "malformed", "type-confused"]
)
def test_collective_artifact_wait_validates_durable_rank_result(
    artifact: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_dir = tmp_path / "collective"
    for name in ("items", "ranks", "records"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    shard = (WorkItem("pair-0", {"id": "pair-0"}, 1),)
    reported = {"rank": 0, "record_write_errors": []}
    mpi.atomic_write_json(
        run_dir / "records" / "pair-0.json", {"status": "failed"}
    )
    rank_path = run_dir / "ranks" / "rank-0000.json"
    if artifact == "mismatched":
        mpi.atomic_write_json(
            rank_path, {"rank": 0, "record_write_errors": [], "extra": True}
        )
    elif artifact == "type-confused":
        mpi.atomic_write_json(
            rank_path, {"rank": False, "record_write_errors": []}
        )
    else:
        rank_path.write_text("{", encoding="utf-8")
    monkeypatch.setattr(
        mpi.time,
        "sleep",
        lambda delay: pytest.fail(
            f"invalid durable rank evidence slept for {delay} seconds"
        ),
    )

    errors = mpi._wait_collective_artifacts(
        run_dir, (shard,), 600.0, rank_results=(reported,)
    )

    assert len(errors) == 1
    assert errors[0]["rank"] == 0
    assert errors[0]["path"] == "ranks/rank-0000.json"
    assert errors[0]["type"] == (
        "JSONDecodeError"
        if artifact == "malformed"
        else "InvalidRankArtifact"
    )


def test_collective_artifact_wait_accepts_matching_durable_rank_result(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "collective"
    for name in ("items", "ranks", "records"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    shard = (WorkItem("pair-0", {"id": "pair-0"}, 1),)
    reported = {"rank": 0, "record_write_errors": []}
    mpi.atomic_write_json(
        run_dir / "records" / "pair-0.json", {"status": "failed"}
    )
    mpi.atomic_write_json(run_dir / "ranks" / "rank-0000.json", reported)

    assert (
        mpi._wait_collective_artifacts(
            run_dir, (shard,), 0.01, rank_results=(reported,)
        )
        == []
    )


def test_collective_artifact_wait_retries_transient_rank_read_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_dir = tmp_path / "collective"
    for name in ("items", "ranks", "records"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    shard = (WorkItem("pair-0", {"id": "pair-0"}, 1),)
    reported = {"rank": 0, "record_write_errors": []}
    mpi.atomic_write_json(
        run_dir / "records" / "pair-0.json", {"status": "failed"}
    )
    rank_path = run_dir / "ranks" / "rank-0000.json"
    mpi.atomic_write_json(rank_path, reported)
    original_lstat = Path.lstat
    rank_lstat_attempts = 0
    transient_failures = 0

    def transient_lstat(path: Path):
        nonlocal rank_lstat_attempts, transient_failures
        if path == rank_path:
            rank_lstat_attempts += 1
        if path == rank_path and rank_lstat_attempts == 2:
            transient_failures += 1
            raise OSError("rank artifact is not visible yet")
        return original_lstat(path)

    monkeypatch.setattr(Path, "lstat", transient_lstat)
    monkeypatch.setattr(mpi.time, "sleep", lambda delay: None)

    assert (
        mpi._wait_collective_artifacts(
            run_dir, (shard,), 0.5, rank_results=(reported,)
        )
        == []
    )
    assert transient_failures == 1


def test_collective_artifact_wait_does_not_poll_declared_write_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_dir = tmp_path / "collective"
    for name in ("items", "ranks", "records"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    shards = ((WorkItem("pair-0", {"id": "pair-0"}, 1),),)
    result = {
        "rank": 0,
        "artifact_error": {
            "type": "OSError",
            "message": "rank write failed",
        },
        "record_write_errors": [
            {
                "item_id": "pair-0",
                "type": "OSError",
                "message": "record write failed",
            }
        ],
    }
    monkeypatch.setattr(
        mpi.time,
        "sleep",
        lambda delay: pytest.fail(
            f"declared permanent write failure slept for {delay} seconds"
        ),
    )

    errors = mpi._wait_collective_artifacts(
        run_dir, shards, 600.0, rank_results=(result,)
    )

    assert {(error.get("path"), error["message"]) for error in errors} == {
        ("ranks/rank-0000.json", "rank write failed"),
        ("records/pair-0.json", "record write failed"),
    }


def test_collective_artifact_wait_retries_only_reported_successes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_dir = tmp_path / "collective"
    for name in ("items", "ranks", "records"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    shards = (
        (
            WorkItem("failed-record", {"id": "failed-record"}, 1),
            WorkItem("successful-record", {"id": "successful-record"}, 1),
        ),
    )
    result = {
        "rank": 0,
        "record_write_errors": [
            {
                "item_id": "failed-record",
                "type": "OSError",
                "message": "record write failed",
            }
        ],
    }
    sleeps: list[float] = []

    def publish(delay: float) -> None:
        sleeps.append(delay)
        item_dir = run_dir / "items" / "successful-record"
        item_dir.mkdir()
        mpi.atomic_write_json(item_dir / "summary.json", {})
        mpi.atomic_write_json(
            run_dir / "records" / "successful-record.json",
            {"status": "success"},
        )
        mpi.atomic_write_json(run_dir / "ranks" / "rank-0000.json", result)

    monkeypatch.setattr(mpi.time, "sleep", publish)

    errors = mpi._wait_collective_artifacts(
        run_dir, shards, 1.0, rank_results=(result,)
    )

    assert len(sleeps) == 1
    assert [(error.get("path"), error["message"]) for error in errors] == [
        ("records/failed-record.json", "record write failed")
    ]


def test_wait_mapping_does_not_follow_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text('{"status": "ready"}\n', encoding="utf-8")
    marker = tmp_path / "marker.json"
    marker.symlink_to(target)

    with pytest.raises(TimeoutError):
        mpi._wait_mapping(marker, 0.001, {"ready"})


def test_file_promotion_records_partial_rank_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest, run_dir, rank_dir, shards, _ = _stage_file_rank(
        tmp_path, "partial-promotion", "attempt-partial"
    )
    original_replace = Path.replace
    attempted: list[tuple[Path, Path]] = []

    def fail_second(source: Path, destination: Path) -> Path:
        attempted.append((source, destination))
        if len(attempted) == 2:
            raise OSError("synthetic mid-promotion failure")
        return original_replace(source, destination)

    monkeypatch.setattr(Path, "replace", fail_second)

    ranks, errors = mpi._publish_completed_file_ranks(
        run_dir,
        "partial-promotion",
        "attempt-partial",
        manifest.sha256,
        shards,
        0.01,
    )

    assert ranks == []
    assert len(errors) == 1
    assert errors[0]["type"] == "PartialRankPromotion"
    assert errors[0]["promoted_paths"] == ["items/pair-0"]
    assert errors[0]["failed_path"] == "records/pair-0.json"
    assert errors[0]["cause"]["type"] == "OSError"
    assert (run_dir / "items" / "pair-0").is_dir()
    assert (rank_dir / "records" / "pair-0.json").is_file()
    assert (rank_dir / "ranks" / "rank-0000.json").is_file()


def test_file_promotion_rejects_nonfinite_staged_json(tmp_path: Path) -> None:
    manifest, run_dir, rank_dir, shards, _ = _stage_file_rank(
        tmp_path, "nonfinite-json", "attempt-nonfinite"
    )
    rank_path = rank_dir / "ranks" / "rank-0000.json"
    rank = mpi.read_json_mapping(rank_path)
    rank["rank_wall_sec"] = float("nan")
    rank_path.write_text(json.dumps(rank), encoding="utf-8")

    ranks, errors = mpi._publish_completed_file_ranks(
        run_dir,
        "nonfinite-json",
        "attempt-nonfinite",
        manifest.sha256,
        shards,
        0.01,
    )

    assert ranks == []
    assert len(errors) == 1
    assert errors[0]["type"] == "ValueError"
    assert "JSON-compatible" in errors[0]["message"]
    for name in ("items", "ranks", "records"):
        assert list((run_dir / name).iterdir()) == []


def test_singleton_file_run_records_timeout_and_is_immutable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _files_environment(monkeypatch)
    _remove_cuda_modules(monkeypatch)
    manifest = _write_manifest(tmp_path)
    monkeypatch.setattr(mpi, "run_image_pair_item", _item_runner)
    monkeypatch.setattr(mpi, "_gpu_identity", lambda backend: _gpu())
    monkeypatch.setattr(
        mpi,
        "_load_mpi_api",
        lambda: pytest.fail("file aggregation imported mpi4py"),
    )

    result = mpi.run_mpi_image_pair_batch(
        manifest_path=manifest,
        output_root=tmp_path / "runs",
        run_id="files-success",
        aggregation_mode="files",
        rank_timeout_sec=0.5,
        attempt_id="attempt-one",
        options=_options(),
        rank_setup_timeout_sec=0.25,
    )

    assert result is not None
    assert result.status == "success"
    run_record = json.loads((result.run_dir / "run.json").read_text())
    assert run_record["rank_setup_timeout_sec"] == 0.25
    assert run_record["rank_timeout_sec"] == 0.5
    assert result.summary["attempt_id"] == "attempt-one"
    assert result.summary["rank_setup_timeout_sec"] == 0.25
    assert result.summary["rank_timeout_sec"] == 0.5
    assert (
        json.loads((result.run_dir / "records" / "pair-0.json").read_text())[
            "attempt_id"
        ]
        == "attempt-one"
    )
    assert (
        json.loads((result.run_dir / "ranks" / "rank-0000.json").read_text())[
            "attempt_id"
        ]
        == "attempt-one"
    )
    attempt = mpi.read_json_mapping(
        mpi._attempt_path(result.run_dir, "files-success", "attempt-one")
    )
    assert attempt["status"] == "success"
    assert attempt["terminal"] is True
    preserved = result.summary_path.read_bytes()

    with pytest.raises(RuntimeError, match="attempt_id already exists"):
        mpi.run_mpi_image_pair_batch(
            manifest_path=manifest,
            output_root=tmp_path / "runs",
            run_id="files-success",
            aggregation_mode="files",
            rank_timeout_sec=0.5,
            attempt_id="attempt-one",
            options=_options(),
            rank_setup_timeout_sec=0.25,
        )
    with pytest.raises(RuntimeError, match="cannot prepare"):
        mpi.run_mpi_image_pair_batch(
            manifest_path=manifest,
            output_root=tmp_path / "runs",
            run_id="files-success",
            aggregation_mode="files",
            rank_timeout_sec=0.5,
            attempt_id="attempt-two",
            options=_options(),
            rank_setup_timeout_sec=0.25,
        )
    assert result.summary_path.read_bytes() == preserved


def test_singleton_file_run_records_rank_artifact_write_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _files_environment(monkeypatch)
    _remove_cuda_modules(monkeypatch)
    manifest = _write_manifest(tmp_path)
    run_id = "files-rank-artifact-failure"
    attempt_id = "attempt-one"
    original_write = mpi.atomic_write_json

    def fail_rank(path: Path, payload, **kwargs) -> None:
        if path.parent.name == "ranks" and path.name == "rank-0000.json":
            raise OSError("synthetic rank write failure")
        original_write(path, payload, **kwargs)

    monkeypatch.setattr(mpi, "atomic_write_json", fail_rank)
    monkeypatch.setattr(mpi, "run_image_pair_item", _item_runner)
    monkeypatch.setattr(mpi, "_gpu_identity", lambda backend: _gpu())
    monkeypatch.setattr(
        mpi.time,
        "sleep",
        lambda delay: pytest.fail(
            f"declared permanent write failure slept for {delay} seconds"
        ),
    )

    result = mpi.run_mpi_image_pair_batch(
        manifest_path=manifest,
        output_root=tmp_path / "runs",
        run_id=run_id,
        aggregation_mode="files",
        rank_timeout_sec=600.0,
        attempt_id=attempt_id,
        options=_options(),
        rank_setup_timeout_sec=600.0,
    )

    assert result is not None
    assert result.status == "failed"
    assert result.summary["aggregation_errors"] == [
        {
            "rank": 0,
            "path": "ranks/rank-0000.json",
            "type": "OSError",
            "message": "synthetic rank write failure",
        }
    ]
    assert result.summary["rank_result_audit"]["missing_ranks"] == [0]
    assert list((result.run_dir / "ranks").iterdir()) == []
    assert list((result.run_dir / "records").iterdir()) == []
    marker = mpi.read_json_mapping(
        mpi._attempt_path(result.run_dir, run_id, attempt_id)
    )
    assert marker["status"] == "failed"
    assert marker["terminal"] is True


def test_file_rank_zero_completion_failure_terminalizes_attempt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _files_environment(monkeypatch)
    _remove_cuda_modules(monkeypatch)
    manifest_path = _write_manifest(tmp_path)
    run_id = "completion-write-failure"
    attempt_id = "attempt-one"
    output_root = tmp_path / "runs"
    monkeypatch.setattr(mpi, "run_image_pair_item", _item_runner)
    monkeypatch.setattr(mpi, "_gpu_identity", lambda backend: _gpu())
    monkeypatch.setattr(
        mpi,
        "_complete_file_rank",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            OSError("synthetic completion write failure")
        ),
    )

    with pytest.raises(
        RuntimeError, match="cannot publish MPI rank-zero completion"
    ):
        mpi.run_mpi_image_pair_batch(
            manifest_path=manifest_path,
            output_root=output_root,
            run_id=run_id,
            aggregation_mode="files",
            rank_setup_timeout_sec=0.1,
            rank_timeout_sec=0.1,
            attempt_id=attempt_id,
            options=_options(),
        )

    run_dir = output_root / run_id
    marker = mpi.read_json_mapping(
        mpi._attempt_path(run_dir, run_id, attempt_id)
    )
    assert marker["status"] == "failed"
    assert marker["terminal"] is True
    assert marker["error"]["type"] == "OSError"
    aggregate_error = mpi.read_json_mapping(run_dir / "aggregate-error.json")
    assert aggregate_error["error"]["message"] == (
        "synthetic completion write failure"
    )
    assert not (run_dir / "summary.json").exists()


@pytest.mark.parametrize("failed_summary", [False, True])
def test_exact_retry_repairs_only_committed_attempt_marker(
    failed_summary: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fail_item(item, output, options):
        raise ValueError("synthetic item failure")

    run_id = "repair-failed" if failed_summary else "repair-success"
    attempt_id = "attempt-repair"
    item_runner = fail_item if failed_summary else _item_runner
    manifest, run_dir, _, shards, _ = _stage_file_rank(
        tmp_path, run_id, attempt_id, item_runner
    )
    context = mpi._RankContext(
        0, 0, 1, "host", "test", launch_id="test-launch"
    )
    original_terminal = mpi._set_attempt_terminal

    def fail_terminal(*args, **kwargs) -> None:
        raise OSError("synthetic terminal marker failure")

    monkeypatch.setattr(mpi, "_set_attempt_terminal", fail_terminal)
    with pytest.raises(
        RuntimeError, match="summary is committed.*terminal marker is not"
    ):
        mpi._file_aggregate(
            context,
            run_dir,
            run_id,
            attempt_id,
            manifest,
            _options(),
            shards,
            0.1,
            mpi.timestamp_utc(),
            time.perf_counter(),
            rank_setup_timeout_sec=0.1,
        )

    expected_status = "failed" if failed_summary else "success"
    summary = mpi.read_json_mapping(run_dir / "summary.json")
    assert summary["status"] == expected_status
    marker_path = mpi._attempt_path(run_dir, run_id, attempt_id)
    marker_before = marker_path.read_bytes()
    marker = mpi.read_json_mapping(marker_path)
    assert marker["status"] == "ready"
    assert marker["terminal"] is False
    run_evidence = {
        str(path.relative_to(run_dir)): path.read_bytes()
        for path in run_dir.rglob("*")
        if path.is_file()
    }

    monkeypatch.setattr(mpi, "_set_attempt_terminal", original_terminal)
    with pytest.raises(
        RuntimeError, match="marker differs in: manifest_sha256"
    ):
        mpi._file_prepare_run(
            context,
            run_dir,
            run_id,
            attempt_id,
            0.1,
            mpi.timestamp_utc(),
            replace(manifest, sha256="0" * 64),
            _options(),
        )
    assert marker_path.read_bytes() == marker_before
    assert {
        str(path.relative_to(run_dir)): path.read_bytes()
        for path in run_dir.rglob("*")
        if path.is_file()
    } == run_evidence

    with pytest.raises(
        RuntimeError, match="committed evidence differs in: run.options"
    ):
        mpi._file_prepare_run(
            context,
            run_dir,
            run_id,
            attempt_id,
            0.1,
            mpi.timestamp_utc(),
            manifest,
            _options(backend="numba-cuda"),
        )
    assert marker_path.read_bytes() == marker_before
    assert {
        str(path.relative_to(run_dir)): path.read_bytes()
        for path in run_dir.rglob("*")
        if path.is_file()
    } == run_evidence

    with pytest.raises(RuntimeError, match="repaired the terminal marker"):
        mpi._file_prepare_run(
            context,
            run_dir,
            run_id,
            attempt_id,
            0.1,
            mpi.timestamp_utc(),
            manifest,
            _options(),
        )
    repaired = mpi.read_json_mapping(marker_path)
    assert repaired["status"] == expected_status
    assert repaired["terminal"] is True
    assert {
        str(path.relative_to(run_dir)): path.read_bytes()
        for path in run_dir.rglob("*")
        if path.is_file()
    } == run_evidence


def test_late_file_rank_cannot_mutate_terminal_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(
        "CUPHOTON_ALLOCATED_CUDA_VISIBLE_DEVICES", raising=False
    )
    options = _options()
    manifest = mpi.load_image_pair_manifest(
        _write_manifest(tmp_path, count=2)
    )
    run_id = "late-rank"
    attempt_id = "attempt-late"
    run_dir = tmp_path / "runs" / run_id
    started_at = mpi.timestamp_utc()
    root = mpi._RankContext(0, 0, 2, "host", "test", launch_id="test-launch")
    peer = mpi._RankContext(1, 1, 2, "host", "test", launch_id="test-launch")
    setup_timeout = 2.0
    timeout = 0.2

    def prepare(context: mpi._RankContext):
        return mpi._file_prepare_run(
            context,
            run_dir,
            run_id,
            attempt_id,
            setup_timeout,
            started_at,
            manifest,
            options,
            rank_timeout_sec=timeout,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(prepare, (root, peer))) == [None, None]
    shards = mpi.partition_byte_balanced(manifest.work_items(), 2)
    rank_dirs = [
        mpi._file_rank_staging_dir(run_dir, run_id, attempt_id, rank)
        for rank in range(2)
    ]

    rank_zero = mpi._execute_rank(
        run_id=run_id,
        run_dir=rank_dirs[0],
        manifest_sha256=manifest.sha256,
        context=root,
        items=shards[0],
        options=options,
        visibility="GPU-0",
        setup_error=None,
        item_runner=_item_runner,
        gpu_identity_loader=lambda backend: _gpu("GPU-0"),
        attempt_id=attempt_id,
    )
    mpi._complete_file_rank(
        rank_dirs[0],
        run_id,
        attempt_id,
        manifest.sha256,
        0,
        rank_zero,
    )
    result = mpi._file_aggregate(
        root,
        run_dir,
        run_id,
        attempt_id,
        manifest,
        options,
        shards,
        timeout,
        started_at,
        time.perf_counter(),
        rank_setup_timeout_sec=setup_timeout,
    )
    assert result is not None and result.status == "failed"
    attempt_marker = mpi._attempt_path(run_dir, run_id, attempt_id)
    terminal_marker = attempt_marker.read_bytes()

    def tree(path: Path) -> dict[str, bytes | None]:
        return {
            str(item.relative_to(path)): (
                item.read_bytes() if item.is_file() else None
            )
            for item in path.rglob("*")
        }

    terminal_tree = tree(run_dir)
    staging_before = tree(rank_dirs[1])
    rank_one = mpi._execute_rank(
        run_id=run_id,
        run_dir=rank_dirs[1],
        manifest_sha256=manifest.sha256,
        context=mpi._RankContext(
            1, 1, 2, "host", "test", launch_id="test-launch"
        ),
        items=shards[1],
        options=options,
        visibility="GPU-1",
        setup_error=None,
        item_runner=_item_runner,
        gpu_identity_loader=lambda backend: _gpu("GPU-1"),
        attempt_id=attempt_id,
    )
    mpi._complete_file_rank(
        rank_dirs[1],
        run_id,
        attempt_id,
        manifest.sha256,
        1,
        rank_one,
    )

    assert tree(rank_dirs[1]) != staging_before
    assert (rank_dirs[1] / ".complete.json").is_file()
    assert tree(run_dir) == terminal_tree
    assert attempt_marker.read_bytes() == terminal_marker
    assert json.loads((run_dir / "summary.json").read_text())["status"] == (
        "failed"
    )


def test_file_nonroot_does_not_wait_for_root_aggregation(
    tmp_path: Path,
) -> None:
    start = time.perf_counter()

    result = mpi._file_aggregate(
        mpi._RankContext(1, 1, 2, "host", "test", launch_id="test-launch"),
        tmp_path / "absent-run",
        "nonroot-return",
        "attempt-one",
        SimpleNamespace(sha256="manifest"),
        _options(),
        (),
        0.001,
        mpi.timestamp_utc(),
        start,
    )

    assert result is None
    assert time.perf_counter() - start < 0.1
