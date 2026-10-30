# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest

from cuphoton.xbeam import profiling


class _FakeNvtx:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    def RangePush(self, message: str) -> None:
        self.calls.append(("push", message))

    def RangePop(self) -> None:
        self.calls.append(("pop", None))


def test_nvtx_range_is_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(profiling, "_enabled", False)
    monkeypatch.setattr(profiling, "_host_timing_enabled", False)
    monkeypatch.setattr(profiling, "_nvtx", None)
    monkeypatch.setattr(profiling, "_nvtx_import_attempted", False)

    with profiling.nvtx_range("disabled"):
        pass

    assert profiling._nvtx_import_attempted is False


def test_host_timing_records_nested_self_time(monkeypatch):
    ticks = iter((100, 120, 150, 200))
    monkeypatch.setattr(profiling, "_enabled", False)
    monkeypatch.setattr(profiling, "_host_timing_enabled", True)
    monkeypatch.setattr(profiling, "_host_timings", {})
    monkeypatch.setattr(
        profiling.time, "perf_counter_ns", lambda: next(ticks)
    )

    with profiling.nvtx_range("outer"):
        with profiling.nvtx_range("inner"):
            pass

    rows = {
        row["name"]: row for row in profiling.host_timing_report()["ranges"]
    }
    assert rows["inner"]["inclusive_seconds"] == pytest.approx(30e-9)
    assert rows["inner"]["self_seconds"] == pytest.approx(30e-9)
    assert rows["outer"]["inclusive_seconds"] == pytest.approx(100e-9)
    assert rows["outer"]["self_seconds"] == pytest.approx(70e-9)


def test_host_timing_report_can_be_written_as_json(tmp_path, monkeypatch):
    output = tmp_path / "timing.json"
    monkeypatch.setattr(profiling, "_host_timing_enabled", True)
    monkeypatch.setattr(profiling, "_host_timing_written", False)
    monkeypatch.setattr(
        profiling,
        "_host_timings",
        {"stage": profiling._HostTiming(2, 8_000_000, 7_000_000, 3, 5)},
    )

    profiling.write_host_timing_report(output)

    report = json.loads(output.read_text())
    assert report["format"] == "cuphoton-xbeam-host-timing-v1"
    assert report["ranges"][0]["name"] == "stage"
    assert report["ranges"][0]["calls"] == 2


def test_host_timing_report_retries_after_write_failure(
    tmp_path, monkeypatch
):
    output = tmp_path / "timing.json"
    monkeypatch.setattr(profiling, "_host_timing_enabled", True)
    monkeypatch.setattr(profiling, "_host_timing_written", False)
    original_write_text = profiling.Path.write_text
    calls = 0

    def fail_once(path, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("temporary write failure")
        return original_write_text(path, *args, **kwargs)

    monkeypatch.setattr(profiling.Path, "write_text", fail_once)

    with pytest.raises(OSError, match="temporary write failure"):
        profiling.write_host_timing_report(output)
    assert profiling._host_timing_written is False

    profiling.write_host_timing_report(output)

    assert profiling._host_timing_written is True
    assert json.loads(output.read_text())["format"] == (
        "cuphoton-xbeam-host-timing-v1"
    )


def test_nvtx_range_is_balanced_on_failure(monkeypatch):
    fake = _FakeNvtx()
    monkeypatch.setattr(profiling, "_enabled", True)
    monkeypatch.setattr(profiling, "_nvtx", fake)
    monkeypatch.setattr(profiling, "_nvtx_import_attempted", True)

    with pytest.raises(RuntimeError, match="failure"):
        with profiling.nvtx_range("stage"):
            raise RuntimeError("failure")

    assert fake.calls == [("push", "stage"), ("pop", None)]
