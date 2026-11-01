# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Read-ahead ownership, boundedness, errors, and scheduling equivalence."""

from dataclasses import fields, replace
from threading import Event, Thread

import numpy as np
import pytest

from cuphoton.xbeam import CorrelationSpec, Hdf5Source, correlate_series
from cuphoton.xbeam.runtime.prefetch import BatchPrefetch


def test_read_ahead_overlaps_consumer_and_stays_one_batch_ahead():
    started, release = Event(), Event()
    calls = []

    def read(i):
        calls.append(i)
        if i == 1:
            started.set()
            assert release.wait(3)
        return np.full(8, i)

    with BatchPrefetch(read, range(4)) as reader:
        current = next(reader)
        try:
            assert started.wait(3)
            assert calls == [0, 1]  # The next read runs during consumption.
            np.testing.assert_array_equal(current, 0)
        finally:
            release.set()
        remaining = list(reader)
    assert calls == [0, 1, 2, 3]
    assert [v[0] for v in remaining] == [1, 2, 3]
    np.testing.assert_array_equal(current, 0)  # No slot is overwritten.
    reader.close()


def test_read_failure_propagates_in_order_and_joins_worker():
    def read(i):
        if i == 1:
            raise OSError("synthetic read failure")
        return i

    with pytest.raises(OSError, match="synthetic read failure"):
        with BatchPrefetch(read, range(3)) as reader:
            assert next(reader) == 0
            next(reader)
    assert reader._pool is None
    assert list(reader) == []


def write_input(path):
    h5py = pytest.importorskip("h5py")
    rng = np.random.default_rng(31)
    with h5py.File(path, "w") as f:
        for key in ["a", "b"]:
            ds = f.create_dataset(
                key, data=rng.normal(size=257), compression="gzip"
            )
            ds.attrs.update(sample_rate_hz=32.0, origin_seconds=5.0)
        f.create_dataset("mask", data=rng.random(257) > 0.2)


@pytest.mark.parametrize("hop", [5, 17, 39])
@pytest.mark.parametrize("mode", ["linear", "circular"])
def test_prefetched_shards_match_synchronous_results(tmp_path, hop, mode):
    path = tmp_path / "input.h5"
    write_input(path)
    spec = CorrelationSpec(
        (("a", "b"),),
        17,
        hop_samples=hop,
        windows_per_result=3,
        batch_windows=2,
        output="both",
        lag_mode=mode,
    )
    with Hdf5Source(
        path, {"a": "a", "b": "b"}, validity_paths={"b": "mask"}
    ) as source:
        expected = list(
            correlate_series(source, replace(spec, prefetch_batches=0))
        )
        actual = []
        for shard in range(7):  # Includes empty shards and overlap halos.
            actual.extend(
                correlate_series(
                    source, replace(spec, shard_count=7, shard_index=shard)
                )
            )
        assert not source._readers
    for x, y in zip(actual, expected, strict=True):
        for field in fields(x):
            xv, yv = getattr(x, field.name), getattr(y, field.name)
            if isinstance(xv, np.ndarray):
                np.testing.assert_array_equal(xv, yv)
            else:
                assert xv == yv


def test_source_close_waits_for_inflight_read_before_closing_file(tmp_path):
    path = tmp_path / "input.h5"
    write_input(path)
    source = Hdf5Source(path, {"a": "a"})
    started, release, closing, closed = Event(), Event(), Event(), Event()
    original = source._read_batch

    def blocked(request):
        if request[1][0] > 0:
            started.set()
            assert release.wait(3)
        return original(request)

    source._read_batch = blocked
    iterator = correlate_series(source, CorrelationSpec((("a", "a"),), 17))
    next(iterator)
    assert started.wait(3)

    def close():
        closing.set()
        source.close()
        closed.set()

    thread = Thread(target=close)
    thread.start()
    try:
        assert closing.wait(3)
        assert not closed.is_set()
        assert source._file.id.valid
    finally:
        release.set()
        thread.join(3)
        iterator.close()
        source.close()
    assert closed.is_set() and not thread.is_alive()
    assert not source._file.id.valid
    assert not source._readers


def test_generator_close_releases_reader_and_read_errors_escape(tmp_path):
    path = tmp_path / "input.h5"
    write_input(path)
    with Hdf5Source(path, {"a": "a"}) as source:
        spec = CorrelationSpec((("a", "a"),), 17)
        iterator = correlate_series(source, spec)
        next(iterator)
        iterator.close()
        assert not source._readers
        original = source._read_batch

        def fail(request):
            if request[1][0] > 0:
                raise OSError("prefetch error")
            return original(request)

        source._read_batch = fail
        iterator = correlate_series(source, spec)
        next(iterator)
        with pytest.raises(OSError, match="prefetch error"):
            next(iterator)
        assert not source._readers


@pytest.mark.parametrize("value", [-1, 2, True, 1.5])
def test_reject_invalid_prefetch_bound(value):
    with pytest.raises(ValueError, match="prefetch_batches"):
        CorrelationSpec((("a", "a"),), 8, prefetch_batches=value)


def test_consuming_after_source_close_reports_lifetime_error(tmp_path):
    path = tmp_path / "input.h5"
    write_input(path)
    with Hdf5Source(path, {"a": "a"}) as source:
        iterator = correlate_series(
            source, CorrelationSpec((("a", "a"),), 17)
        )
        next(iterator)
        source.close()
        with pytest.raises(RuntimeError, match="prefetched source closed"):
            next(iterator)
        assert not source._readers
