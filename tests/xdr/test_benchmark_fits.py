# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from cuphoton.core.cli import run_component


def _load_benchmark_module():
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    import cuphoton.xdr.benchmark_fits as benchmark

    return benchmark


def test_native_plan_summary_counts_hdus_and_bytes():
    bench = _load_benchmark_module()
    planned = bench.PlannedFileSummary(
        path=Path("a.fits"),
        file_index=0,
        hdu_indices=(1, 2, 3),
        image_hdus=1,
        compressed_hdus=2,
        raw_bytes=4096,
        data_mb=12.5,
        spans=7,
    )

    assert (
        planned.note()
        == "3 HDUs: 1 image, 2 compressed, 7 spans, 4.0 KiB raw"
    )


def test_failed_phase_from_exception_marks_not_implemented():
    bench = _load_benchmark_module()

    result = bench.failed_phase_from_exception(
        "batch_to_device",
        NotImplementedError("RICE_1 is not supported on the GPU path"),
        note="load FITS files",
    )

    assert result.phase == "batch_to_device"
    assert result.ok is False
    assert result.error.startswith("NotImplementedError:")
    assert result.note == "load FITS files"


def test_parse_native_batcher_values():
    bench = _load_benchmark_module()

    assert bench.parse_native_batcher("auto") == "auto"
    assert bench.parse_native_batcher("on") is True
    assert bench.parse_native_batcher("off") is False


def test_resolve_paths_accepts_explicit_files(tmp_path):
    bench = _load_benchmark_module()
    first = tmp_path / "first.fits"
    second = tmp_path / "second.fit"
    first.touch()
    second.touch()

    assert bench.resolve_paths(
        [str(first), second],
        scan_dir=None,
        max_files=None,
    ) == [first, second]


def test_resolve_paths_rejects_mixed_and_missing_inputs(tmp_path):
    bench = _load_benchmark_module()
    scan_dir = tmp_path / "fits"
    scan_dir.mkdir()

    with pytest.raises(ValueError, match="either positional"):
        bench.resolve_paths(
            ["one.fits"],
            scan_dir=scan_dir,
            max_files=None,
        )
    with pytest.raises(FileNotFoundError, match="missing FITS file"):
        bench.resolve_paths(
            [tmp_path / "missing.fits"],
            scan_dir=None,
            max_files=None,
        )
    with pytest.raises(ValueError, match="provide FITS files"):
        bench.resolve_paths([], scan_dir=None, max_files=None)


def test_run_benchmark_rejects_unavailable_gpu_before_work(
    monkeypatch,
    tmp_path,
):
    bench = _load_benchmark_module()
    path = tmp_path / "input.fits"
    path.touch()
    monkeypatch.setattr(bench, "gpu_available", lambda: False)

    with pytest.raises(RuntimeError, match="GPU / kvikio"):
        bench.run_benchmark([path], out=lambda _line: None)


def test_benchmark_command_help_uses_the_shared_cli(capsys):
    rc = run_component("xdr", ["help", "benchmark-fits"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "usage: cuphoton xdr benchmark-fits" in captured.out
    assert "--hdu-indices" in captured.out
    assert "--native-batcher" in captured.out
    assert "fits_file" in captured.out
    assert captured.err == ""


def test_benchmark_invalid_hdu_indices_preserve_parser_error(capsys):
    rc = run_component(
        "xdr",
        ["benchmark-fits", "--hdu-indices", "nope"],
    )
    captured = capsys.readouterr()

    assert rc == 2
    assert captured.out == ""
    assert captured.err.startswith("usage: cuphoton xdr benchmark-fits")
    assert captured.err.endswith(
        "cuphoton xdr benchmark-fits: error: argument "
        "--hdu-indices: --hdu-indices must be comma-separated integers\n"
    )


def test_benchmark_command_preserves_ordered_positionals_and_options(
    monkeypatch,
    capsys,
):
    import cuphoton.xdr.commands as commands

    received = {}

    def fake_run_benchmark(fits_files, **kwargs):
        received["fits_files"] = list(fits_files)
        received.update(kwargs)
        return []

    monkeypatch.setattr(commands, "run_benchmark", fake_run_benchmark)
    rc = run_component(
        "xdr",
        [
            "benchmark-fits",
            "--hdu-indices",
            "2,1,2",
            "--native-batcher",
            "off",
            "--mock-storage",
            "host",
            "--skip-gds-read",
            "first.fits",
            "second.fits",
            "first.fits",
        ],
    )
    captured = capsys.readouterr()

    assert rc == 0
    assert received["fits_files"] == [
        "first.fits",
        "second.fits",
        "first.fits",
    ]
    assert received["hdu_indices"] == [2, 1, 2]
    assert received["native_batcher"] == "off"
    assert received["mock_storage_kind"] == "host"
    assert received["skip_gds_read"] is True
    assert captured.err == ""


def test_benchmark_command_reports_no_input_as_command_error(capsys):
    rc = run_component("xdr", ["benchmark-fits"])
    captured = capsys.readouterr()

    assert rc == 1
    assert captured.out == ""
    assert "provide FITS files or --dir" in captured.err


def test_gds_read_drains_before_submit_queue_fills(monkeypatch):
    bench = _load_benchmark_module()

    class FakeNativeBatchBuilder:
        instances = []

        def __init__(
            self,
            decode_batch_files,
            batch_queue_depth,
            native_read_threads,
            device_id,
        ):
            self.capacity = int(batch_queue_depth) + int(native_read_threads)
            self.outstanding = 0
            self.max_outstanding = 0
            self.completed = []
            self.closed = False
            self.stopped = False
            self.__class__.instances.append(self)

        def submit_batch(self, batch_id, plans):
            if self.outstanding >= self.capacity:
                raise AssertionError("submit_batch called with full queue")
            plans = tuple(plans)
            device_nbytes = sum(int(item.total_bytes) for item in plans)
            native_files = tuple(object() for _ in plans)
            self.completed.append((object(), 0, device_nbytes, native_files))
            self.outstanding += 1
            self.max_outstanding = max(self.max_outstanding, self.outstanding)

        def close_input(self):
            self.closed = True

        def next_batch(self):
            if self.completed:
                self.outstanding -= 1
                return self.completed.pop(0)
            if self.closed:
                return None
            return None

        def io_stats(self):
            return {}

        def request_stop(self):
            self.stopped = True

    planned_files = [
        SimpleNamespace(total_bytes=1024, file_index=index)
        for index in range(8)
    ]
    fake_cuda = SimpleNamespace(
        Device=lambda: SimpleNamespace(id=0),
    )

    monkeypatch.setattr(bench.storage_cache, "active", lambda: False)
    monkeypatch.setattr(
        bench,
        "get_native_batch_builder",
        lambda required: FakeNativeBatchBuilder,
    )
    monkeypatch.setattr(
        bench,
        "_native_file_plans",
        lambda group: tuple(group),
    )
    monkeypatch.setattr(bench, "cp", SimpleNamespace(cuda=fake_cuda))

    result = bench.bench_gds_read(
        planned_files,
        iterations=1,
        decode_batch_files=1,
        batch_queue_depth=1,
        native_read_threads=1,
    )

    assert result.ok, result.error
    assert all(
        instance.max_outstanding <= instance.capacity
        for instance in FakeNativeBatchBuilder.instances
    )
    assert all(
        instance.stopped for instance in FakeNativeBatchBuilder.instances
    )
