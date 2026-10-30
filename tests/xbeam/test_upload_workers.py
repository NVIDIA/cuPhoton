# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Common worker limits, ordered concurrent reads and independent buffers."""

from threading import Event, Lock

import numpy as np
import pytest

from cuphoton.xbeam import CorrelationSpec, Hdf5Source, correlate_series
from cuphoton.xbeam.runtime.prefetch import BatchPrefetch
from cuphoton.xbeam.runtime.uploads import (
    bounded_upload_workers,
    normalize_upload_workers,
    resolve_upload_workers,
)


@pytest.mark.parametrize("value", [0, 17, True, 1.5, "bad", "2.5"])
def test_worker_validation_is_common_to_hdf5_and_radio(value, tmp_path):
    with pytest.raises(ValueError, match="upload_workers"):
        normalize_upload_workers(value)
    with pytest.raises(ValueError, match="upload_workers"):
        Hdf5Source(
            tmp_path / "not-opened.h5", {"a": "a"}, upload_workers=value
        )


def test_auto_and_available_work_bound_the_limit():
    assert normalize_upload_workers(" AUTO ") is None
    assert normalize_upload_workers("4") == 4
    assert bounded_upload_workers("auto", automatic=4, available=2) == 2
    assert bounded_upload_workers(1, automatic=4, available=4) == 1
    assert resolve_upload_workers(None, alias="4") == 4
    assert resolve_upload_workers("4", alias=4) == 4
    with pytest.raises(ValueError, match="conflicts"):
        resolve_upload_workers(2, alias=4)


def test_concurrent_prefetch_is_bounded_and_keeps_request_order():
    started = [Event() for _ in range(3)]
    release = [Event() for _ in range(2)]
    lock = Lock()
    active = maximum = 0

    def read(index):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        try:
            if index < 3:
                started[index].set()
            if index < 2:
                assert release[index].wait(3)
            return index
        finally:
            with lock:
                active -= 1

    with BatchPrefetch(read, range(7), workers=2, ahead=3) as reader:
        try:
            assert started[0].wait(3) and started[1].wait(3)
            assert not started[2].is_set()
            release[1].set()
            assert started[2].wait(3)  # Later results may finish first.
            assert len(reader._futures) == 3
            release[0].set()
            assert next(reader) == 0
            assert len(reader._futures) == 3
            assert list(reader) == list(range(1, 7))
        finally:
            for event in release:
                event.set()
    assert maximum == 2
    assert not reader._futures and reader._pool is None


def test_parallel_read_failure_surfaces_at_its_request():
    def read(index):
        if index == 1:
            raise OSError("second request failed")
        return index

    with BatchPrefetch(read, range(5), workers=3, ahead=3) as reader:
        assert next(reader) == 0
        with pytest.raises(OSError, match="second request failed"):
            next(reader)
    assert reader._pool is None


@pytest.mark.parametrize("workers", [1, 2, 4, "auto"])
def test_hdf5_posix_workers_preserve_products_and_masks(tmp_path, workers):
    from tests.xbeam.test_series_prefetch import write_input

    path = tmp_path / "signals.h5"
    write_input(path)
    spec = CorrelationSpec(
        (("a", "b"),), 17, windows_per_result=3, output="both"
    )
    options = dict(
        dataset_paths={"a": "a", "b": "b"}, validity_paths={"b": "mask"}
    )
    with Hdf5Source(path, **options) as source:
        expected = list(correlate_series(source, spec))
    with Hdf5Source(
        path, **options, upload_workers=workers, io_queue_depth=4
    ) as source:
        actual = list(correlate_series(source, spec))
        assert source.upload_worker_limit == min(
            3, 1 if workers == "auto" else workers
        )
        assert not source._readers
    for first, second in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(first.spectrum, second.spectrum)
        np.testing.assert_array_equal(first.lag, second.lag)
        np.testing.assert_array_equal(
            first.lag_valid_samples, second.lag_valid_samples
        )


@pytest.mark.parametrize(
    "flag", ["--upload-workers", "--vdif-upload-workers"]
)
def test_radio_cli_alias_targets_the_common_option(
    tmp_path, monkeypatch, flag
):
    from cuphoton.core.cli import run_component
    from cuphoton.xbeam.commands import CorrelateCommand

    job = tmp_path / "job.json"
    job.write_text("{}")
    seen = []
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(
        CorrelateCommand,
        "_run_correlation",
        lambda self: seen.append(self.upload_workers),
    )
    assert (
        run_component(
            "xbeam",
            [
                "correlate",
                str(job),
                "--output",
                str(tmp_path / "out"),
                flag,
                "4",
            ],
        )
        == 0
    )
    assert seen == ["4"]
