# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Independent byte-stream fixtures for bounded parallel shard assembly."""

import hashlib
import json
from decimal import Decimal
from threading import Event, get_ident
from types import SimpleNamespace as NS

import pytest

from cuphoton.xbeam.adapters.radio.swin import (
    FORMAT_VERSION,
    HEADER,
    SYNC_WORD,
)
from cuphoton.xbeam.sharding import merge_swin_shards


def record(seconds, frequency=0, channels=2, **overrides):
    fields = dict(
        sync=SYNC_WORD,
        version=FORMAT_VERSION,
        baseline=258,
        mjd=58000,
        seconds=seconds,
        config=0,
        source=0,
        frequency=frequency,
        polarization=b"RR",
        pulsar=0,
        weight=1.0,
        u=0.0,
        v=0.0,
        w=0.0,
    )
    fields.update(overrides)
    return HEADER.pack(*fields.values()) + bytes(channels * 8)


def shard(root, index, raw, count=2, channels=None):
    path = root / f"shard-{index}"
    path.mkdir()
    (path / "data").write_bytes(raw)
    (path / "ingest-manifest.json").write_text(
        json.dumps(
            {
                "shard": {
                    "window": {
                        "scan_id": "s",
                        "first_integration": index,
                        "integration_count": 1,
                    }
                },
                "output_integrity": {
                    "version": 1,
                    "record_count": count,
                    "channels_by_frequency": channels or {"0": 2},
                    "files": {
                        "data": {
                            "size_bytes": len(raw),
                            "sha256": hashlib.sha256(raw).hexdigest(),
                        }
                    },
                },
            }
        )
    )
    return path


def job():
    return NS(
        frequencies=[
            NS(identifier=0, channel_count=2, channels_to_average=1)
        ],
        scans=[NS(identifier="s", start_offset_s=Decimal(0))],
        start_time=NS(mjd=58000, seconds_of_day=Decimal(0)),
        integration_duration_s=Decimal(".32"),
    )


@pytest.mark.parametrize("workers", [1, 4])
def test_merge_order_short_writes_and_integrity(
    tmp_path, monkeypatch, workers
):
    import cuphoton.xbeam.sharding as module

    original = module.os.pwrite
    monkeypatch.setattr(
        module.os,
        "pwrite",
        lambda fd, data, offset: original(fd, data[:31], offset),
    )
    inputs = [
        shard(tmp_path, i, record(0.16 + i * 0.32) * 2) for i in range(4)
    ]
    output = tmp_path / "out"
    output.mkdir()
    merge_swin_shards(inputs, output, "data", job=job(), workers=workers)
    expected = b"".join((p / "data").read_bytes() for p in inputs)
    assert (output / "data").read_bytes() == expected
    integrity = json.loads((output / ".assembly-integrity.json").read_text())
    assert integrity["record_count"] == 8
    assert (
        integrity["files"]["data"]["sha256"]
        == hashlib.sha256(expected).hexdigest()
    )


@pytest.mark.parametrize(
    "changes",
    [
        dict(sync=0),
        dict(version=2),
        dict(baseline=256),
        dict(baseline=65536),
        dict(polarization=b"\xffR"),
        dict(frequency=9),
        dict(seconds=0.17),
        dict(seconds=0.48),
    ],
)
def test_parallel_validation_failure_never_publishes(tmp_path, changes):
    args = dict(seconds=0.16)
    args.update(changes)
    bad = shard(tmp_path, 0, record(**args), count=1)
    good = shard(tmp_path, 1, record(0.48), count=1)
    output = tmp_path / "out"
    output.mkdir()
    with pytest.raises(ValueError):
        merge_swin_shards([bad, good], output, "data", job=job())
    assert list(output.iterdir()) == []


@pytest.mark.parametrize(
    "failure", ["truncated", "count", "checksum", "write"]
)
def test_parallel_io_failures_clean_up(tmp_path, monkeypatch, failure):
    raw = record(0.16)
    path = shard(
        tmp_path,
        0,
        raw[:-1] if failure == "truncated" else raw,
        count=2 if failure == "count" else 1,
    )
    if failure == "checksum":
        (path / "data").write_bytes(raw[:-1] + b"x")
    if failure == "write":
        monkeypatch.setattr(
            "cuphoton.xbeam.sharding.os.pwrite", lambda *args: 0
        )
    output = tmp_path / "out"
    output.mkdir()
    with pytest.raises((ValueError, OSError)):
        merge_swin_shards([path], output, "data", job=job())
    assert list(output.iterdir()) == []


def test_variable_channels_keep_serial_support(tmp_path):
    raw = record(0.16) + record(0.16, frequency=1, channels=3)
    path = shard(tmp_path, 0, raw, channels={"0": 2, "1": 3})
    output = tmp_path / "out"
    output.mkdir()
    merge_swin_shards([path], output, "data")
    assert (output / "data").read_bytes() == raw


def test_unused_parent_bands_do_not_disable_block_merge(
    tmp_path, monkeypatch
):
    raw = record(0.16)
    path = shard(tmp_path, 0, raw, count=1)
    output = tmp_path / "out"
    output.mkdir()
    model = job()
    model.frequencies.append(
        NS(identifier=99, channel_count=262144, channels_to_average=1)
    )

    def unexpected(*args):
        raise AssertionError(
            "fixed output must not decode individual spectra"
        )

    monkeypatch.setattr("cuphoton.xbeam.sharding.read_record", unexpected)
    merge_swin_shards([path], output, "data", job=model)
    assert (output / "data").read_bytes() == raw


def test_failure_cancels_queued_and_active_copies(tmp_path, monkeypatch):
    import cuphoton.xbeam.sharding as module

    inputs = [shard(tmp_path, i, record(0.16), count=1) for i in range(8)]
    entered = Event()
    exited = Event()
    visited = []
    threads = set()

    def copy(path, filename, fd, offset, integrity, channels, job, cancelled):
        visited.append(path)
        threads.add(get_ident())
        if path == inputs[1]:
            assert entered.wait(5)
            raise ValueError("bad second shard")
        entered.set()
        assert cancelled.wait(5), "merge did not cancel a running copy"
        exited.set()
        return 0

    monkeypatch.setattr(module, "_fixed_swin_copy", copy)
    output = tmp_path / "out"
    output.mkdir()
    with pytest.raises(ValueError, match="bad second shard"):
        merge_swin_shards(inputs, output, "data", workers=2)
    assert exited.is_set()
    assert len(threads) == 2
    assert len(visited) < len(inputs)
    assert list(output.iterdir()) == []


def test_parallel_merge_crosses_block_boundary_with_uniform_frequencies(
    tmp_path,
):
    pair = record(0.16) + record(0.16, frequency=1)
    pairs = (8 << 20) // len(pair) + 3
    raw = pair * pairs
    inputs = [
        shard(tmp_path, 0, raw, count=2 * pairs, channels={"0": 2, "1": 2})
    ]
    inputs.extend(
        shard(tmp_path, i, pair, channels={"0": 2, "1": 2})
        for i in range(1, 6)
    )
    output = tmp_path / "out"
    output.mkdir()
    size = merge_swin_shards(inputs, output, "data", workers=2)
    expected = raw + pair * 5
    assert size == len(expected)
    assert (output / "data").read_bytes() == expected


def test_header_error_identifies_shard_field_and_offset_after_first_block(
    tmp_path,
):
    good = record(0.16)
    count = (8 << 20) // len(good)
    path = shard(
        tmp_path, 0, good * count + record(0.16, sync=0), count=count + 1
    )
    output = tmp_path / "out"
    output.mkdir()
    with pytest.raises(ValueError) as caught:
        merge_swin_shards([path], output, "data")
    assert str(path / "data") in str(caught.value)
    assert f"invalid SWIN sync word at byte {count * len(good)}" in str(
        caught.value
    )
    assert list(output.iterdir()) == []


@pytest.mark.parametrize("serial", [False, True])
@pytest.mark.parametrize("mismatch", ["missing-frequency", "channel-count"])
def test_both_merge_paths_reject_inconsistent_geometry(
    tmp_path, monkeypatch, serial, mismatch
):
    import cuphoton.xbeam.sharding as module

    model = job()
    model.frequencies.append(
        NS(identifier=1, channel_count=2, channels_to_average=1)
    )
    if mismatch == "missing-frequency":
        path = shard(tmp_path, 0, record(0.16, frequency=1), count=1)
    else:
        path = shard(tmp_path, 0, record(0.16), count=1, channels={"0": 3})
    if serial:
        monkeypatch.setattr(module, "_fixed_swin_plan", lambda *args: None)
    output = tmp_path / "out"
    output.mkdir()
    with pytest.raises(ValueError, match="SWIN channel geometry"):
        merge_swin_shards([path], output, "data", job=model)
    assert list(output.iterdir()) == []


def test_unknown_scan_identifies_shard(tmp_path):
    path = shard(tmp_path, 0, record(0.16), count=1)
    model = job()
    model.scans = []
    output = tmp_path / "out"
    output.mkdir()
    with pytest.raises(ValueError, match="unknown scan 's'") as caught:
        merge_swin_shards([path], output, "data", job=model)
    assert str(path) in str(caught.value)
    assert list(output.iterdir()) == []
