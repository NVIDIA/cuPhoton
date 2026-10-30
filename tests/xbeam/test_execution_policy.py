# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Configuration is frozen before resources exist and follows owned work."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from threading import Event

import pytest

from cuphoton.xbeam.adapters.radio import engine
from cuphoton.xbeam.runtime.policy import (
    ExecutionPolicy,
    current_policy,
    policy_records,
    resolve_execution_policy,
    submit_with_policy,
)
from cuphoton.xbeam.upload_resources import _join_cufile_submissions
from tests.xbeam.test_engine import _job

PREFIX = "CUPHOTON_XBEAM_"
SETTINGS = [
    ("CUFILE_FRONTIER", "yes", "1", "0"),
    ("CUFILE_FRONTIER_PLANNER", " GPU ", "gpu", "cpu"),
    ("CUFILE_SLOTS", " 4 ", "04", "2"),
    ("CUFILE_HIGH_PRIORITY", " 1 ", "1", "0"),
    ("XMAC_PACKS_PER_LAUNCH", " 64 ", "064", "0"),
    ("OUTPUT_PIPELINE", "true", "on", "off"),
    ("GLOBAL_GDS_READS", "16", "016", "1"),
]


@pytest.mark.parametrize("suffix,value,equivalent,different", SETTINGS)
def test_equivalent_setting_values(suffix, value, equivalent, different):
    name = PREFIX + suffix
    stable = resolve_execution_policy("cufile", environ={name: value})
    alternate = resolve_execution_policy("cufile", environ={name: equivalent})
    changed = resolve_execution_policy("cufile", environ={name: different})
    assert stable == alternate
    assert stable != changed


@pytest.mark.parametrize(
    "suffix,value",
    [
        ("CUFILE_FRONTIER", "sometimes"),
        ("CUFILE_FRONTIER_PLANNER", "invalid"),
        ("CUFILE_SLOTS", "1"),
        ("CUFILE_SLOTS", "5"),
        ("CUFILE_HIGH_PRIORITY", "2"),
        ("XMAC_PACKS_PER_LAUNCH", "-1"),
        ("XMAC_PACKS_PER_LAUNCH", "1.5"),
        ("OUTPUT_PIPELINE", "invalid"),
        ("GLOBAL_GDS_READS", "0"),
        ("GLOBAL_GDS_READS", "65"),
    ],
)
def test_invalid_values(suffix, value):
    name = PREFIX + suffix
    with pytest.raises(ValueError, match=name):
        resolve_execution_policy("cufile", environ={name: value})


def test_defaults_are_immutable_and_cap_is_optional():
    policy = resolve_execution_policy("cufile", environ={})
    assert policy.cufile_frontier and policy.cufile_high_priority
    assert policy.cufile_slots == policy.cufile_upload_workers == 4
    assert policy.xmac_packs_per_launch == 64 and policy.output_pipeline
    assert policy.global_gds_reads is None
    with pytest.raises(FrozenInstanceError):
        policy.cufile_slots = 2
    assert "read_token_fds" not in policy.to_manifest()
    assert resolve_execution_policy("posix", environ={})


@pytest.mark.parametrize(
    "suffix,value",
    [
        ("CUFILE_FRONTIER", "1"),
        ("CUFILE_HIGH_PRIORITY", "1"),
        ("CUFILE_SLOTS", "4"),
        ("GLOBAL_GDS_READS", "16"),
    ],
)
def test_explicit_cufile_settings_reject_posix(suffix, value):
    with pytest.raises(ValueError, match="requires the cuFile backend"):
        resolve_execution_policy("posix", environ={PREFIX + suffix: value})


@pytest.mark.parametrize("suffix", [setting[0] for setting in SETTINGS])
def test_preparation_validates_before_backend_initialization(
    monkeypatch, tmp_path, suffix
):
    monkeypatch.setenv(PREFIX + suffix, "invalid")

    def forbidden(*args):
        pytest.fail("backend initialized before policy validation")

    monkeypatch.setattr(engine, "get_backend", forbidden)
    with pytest.raises(ValueError, match=PREFIX + suffix):
        engine.prepare_correlation(
            _job(tmp_path), backend="cupy", vdif_io="cufile"
        )


def test_prepared_iterator_ignores_environment_changes(monkeypatch, tmp_path):
    monkeypatch.setenv(PREFIX + "XMAC_PACKS_PER_LAUNCH", "32")
    run = engine.prepare_correlation(_job(tmp_path))
    expected = list(run.records())
    monkeypatch.setenv(PREFIX + "XMAC_PACKS_PER_LAUNCH", "invalid")
    actual = list(run.records())
    assert run.execution_policy.xmac_packs_per_launch == 32
    assert [(r.seconds_of_day, r.weight) for r in actual] == [
        (r.seconds_of_day, r.weight) for r in expected
    ]
    import numpy as np

    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a.spectrum, b.spectrum)


def test_iterators_and_worker_submissions_keep_separate_policies(monkeypatch):
    finished = []
    first = replace(ExecutionPolicy(), xmac_packs_per_launch=1)
    second = replace(first, xmac_packs_per_launch=32)
    monkeypatch.setenv(PREFIX + "XMAC_PACKS_PER_LAUNCH", "7")
    with ThreadPoolExecutor(max_workers=1) as executor:

        def records():
            try:
                for _ in range(2):
                    yield submit_with_policy(
                        executor, current_policy
                    ).result()
            finally:
                finished.append(current_policy())

        a, b = policy_records(first, records), policy_records(second, records)
        assert next(a) is first
        assert current_policy().xmac_packs_per_launch == 7
        assert next(b) is second
        assert next(a) is first
        b.close()
        a.close()
    assert finished == [second, first]
    assert current_policy().xmac_packs_per_launch == 7


def test_cancelled_submission_finishes_join_without_deadlock():
    started, release = Event(), Event()
    with ThreadPoolExecutor(max_workers=1) as executor:

        def occupy():
            started.set()
            release.wait(2)

        busy = executor.submit(occupy)
        assert started.wait(2)
        cancelled = executor.submit(lambda: None)
        assert cancelled.cancel()
        joined = _join_cufile_submissions([busy, cancelled])
        release.set()
        from concurrent.futures import CancelledError

        with pytest.raises(CancelledError):
            joined.result(timeout=2)


def test_standalone_upload_snapshots_policy_before_entering_loader(
    monkeypatch,
):
    from cuphoton.xbeam import upload_resources

    name = PREFIX + "CUFILE_SLOTS"
    monkeypatch.setenv(name, "4")

    def load(*args, **kwargs):
        first = current_policy()
        monkeypatch.setenv(name, "invalid")
        assert current_policy() is first
        assert first.cufile_slots == 4
        return []

    monkeypatch.setattr(upload_resources, "_load_vdif_payload_plans", load)
    assert (
        upload_resources.load_vdif_payload_plans(
            [], xp=None, io_backend="cufile"
        )
        == []
    )
    with pytest.raises(ValueError, match="must be 2, 3, or 4"):
        current_policy()


def test_launcher_validates_before_sidecars_or_output(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from cuphoton.xbeam import commands
    from cuphoton.xbeam.windowing import CorrelationWindow

    monkeypatch.setenv(PREFIX + "GLOBAL_GDS_READS", "invalid")

    def forbidden(*args):
        pytest.fail("sidecar preparation ran before validation")

    monkeypatch.setattr(commands, "_prepare_local_shard_indexes", forbidden)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="GLOBAL_GDS_READS"):
        commands._run_local_shards(
            command=SimpleNamespace(vdif_io="cufile"),
            job_path=tmp_path / "job.json",
            job=None,
            profile="analytic-v1",
            parent_window=CorrelationWindow("scan", 0, 4),
            devices=("0", "1"),
            output=output,
        )
    assert not output.exists()
