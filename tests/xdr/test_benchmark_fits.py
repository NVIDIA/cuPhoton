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
            "--output-json",
            "report.json",
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
    assert received["output_json"] == Path("report.json")
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


@pytest.fixture
def json_benchmark(monkeypatch, tmp_path):
    import cuphoton.xdr.nvcomp_batch as nvcomp_batch

    bench = _load_benchmark_module()
    path = tmp_path / "input.fits"
    path.write_bytes(b"synthetic payload")
    monkeypatch.setattr(bench, "gpu_available", lambda: True)
    monkeypatch.setattr(bench, "cp", SimpleNamespace(__version__="test-cupy"))
    monkeypatch.setattr(
        bench, "kvikio", SimpleNamespace(__version__="test-kvikio")
    )
    monkeypatch.setattr(bench, "_nvcomp_version", lambda: "test-nvcomp")
    monkeypatch.setattr(bench.storage_cache, "_enabled", False)
    monkeypatch.setattr(bench.storage_cache, "preload", lambda _path: None)
    monkeypatch.setattr(bench, "cpp_helper_available", lambda: True)
    monkeypatch.setattr(
        nvcomp_batch,
        "get_native_batch_builder",
        lambda required=False: object,
    )
    monkeypatch.setattr(bench, "is_gds_active", lambda: True)
    phases = []

    def helper_probe():
        assert phases[-1].phase == "batch_to_device_stream"
        return True

    monkeypatch.setattr(bench, "cpp_helper_available", helper_probe)

    def plan(paths, hdus, iterations, threads):
        result = bench.make_result("plan_native", [1.0, 3.0], 0.5, "plan")
        phases.append(result)
        summary = bench.PlannedFileSummary(
            path=paths[0],
            file_index=0,
            hdu_indices=tuple(hdus),
            image_hdus=1,
            compressed_hdus=1,
            raw_bytes=4096,
            data_mb=0.5,
            spans=2,
        )
        return [object()], [summary], result

    def read(*args, **kwargs):
        result = bench.make_result("gds_read", [2.0, 4.0], 0.5, "read")
        phases.append(result)
        return result

    def load(*args, **kwargs):
        phase = (
            "batch_to_device_stream"
            if kwargs["use_stream"]
            else "batch_to_device"
        )
        result = bench.make_result(phase, [3.0, 5.0], 0.5, "load")
        phases.append(result)
        return result

    monkeypatch.setattr(bench, "bench_native_plan", plan)
    monkeypatch.setattr(bench, "bench_gds_read", read)
    monkeypatch.setattr(bench, "bench_batch_load", load)
    return bench, path, phases


@pytest.mark.parametrize("mode", ["real", "host", "device"])
def test_json_report_matches_returned_and_printed_phases(
    json_benchmark, tmp_path, mode
):
    import json
    from dataclasses import asdict

    bench, path, phases = json_benchmark
    output = tmp_path / "benchmark.json"
    printed = []
    results = bench.run_benchmark(
        [path, path],
        hdu_indices=(2, 1),
        iterations=2,
        prefetch_depth=3,
        decode_batch_files=2,
        batch_queue_depth=4,
        native_read_threads=5,
        native_plan_threads=6,
        native_batcher="auto",
        mock_storage_kind=None if mode == "real" else mode,
        skip_gds_read=(mode != "real"),
        output_json=output,
        out=printed.append,
    )
    report = json.loads(output.read_bytes())
    assert results == phases
    assert report["schema_version"] == 1
    assert report["ok"] is True
    assert report["phases"] == [asdict(phase) for phase in results]
    assert report["storage"]["mode"] == mode
    assert report["capabilities"] == {
        "cpp_helper_available": True,
        "gds_active": True,
        "gds_probe": "cuphoton.xdr.is_gds_active",
    }
    assert report["options"] == {
        "hdu_indices": [2, 1],
        "iterations": 2,
        "prefetch_depth": 3,
        "decode_batch_files": 2,
        "batch_queue_depth": 4,
        "native_read_threads": 5,
        "native_plan_threads": 6,
        "native_batcher": "auto",
        "native_batcher_enabled": mode == "real",
        "native_batcher_error": None,
        "skip_gds_read": mode != "real",
        "max_files": None,
    }
    assert report["workload"] == {
        "files": [str(path.resolve()), str(path.resolve())],
        "file_count": 2,
        "planned_file_count": 1,
        "raw_bytes": 4096,
        "data_mb": 0.5,
    }
    assert report["versions"]["cuphoton"] == bench.__version__
    assert report["versions"]["cupy"] == "test-cupy"
    assert report["versions"]["kvikio"] == "test-kvikio"
    assert report["versions"]["nvcomp"] == "test-nvcomp"
    expected_printed = []
    bench.print_results(
        results, env_note=printed[2][5:], out=expected_printed.append
    )
    assert printed == expected_printed
    assert not list(tmp_path.glob(".benchmark.json.*"))


@pytest.mark.parametrize("native_batcher", ["auto", "off", "on"])
def test_json_reports_missing_helper_and_failed_phases(
    json_benchmark, monkeypatch, tmp_path, native_batcher
):
    import json

    import cuphoton.xdr.nvcomp_batch as nvcomp_batch

    bench, path, _ = json_benchmark
    monkeypatch.setattr(bench, "cpp_helper_available", lambda: False)
    monkeypatch.setattr(bench, "is_gds_active", lambda: False)

    def missing_builder(required=False):
        if required:
            raise RuntimeError("native builder unavailable")
        return None

    monkeypatch.setattr(
        nvcomp_batch, "get_native_batch_builder", missing_builder
    )

    def failed_plan(*args):
        return (
            [],
            [],
            bench.failed_phase_from_exception(
                "plan_native",
                RuntimeError("native helper unavailable"),
                note="plan",
            ),
        )

    monkeypatch.setattr(bench, "bench_native_plan", failed_plan)
    output = tmp_path / "failed.json"
    results = bench.run_benchmark(
        [path],
        native_batcher=native_batcher,
        output_json=output,
        out=lambda _: None,
    )
    report = json.loads(output.read_bytes())
    assert report["ok"] is False
    assert report["capabilities"]["cpp_helper_available"] is False
    assert report["capabilities"]["gds_active"] is False
    assert report["phases"][0]["ok"] is False
    assert report["phases"][0]["error"] == results[0].error
    assert report["workload"]["planned_file_count"] == 0
    assert report["options"]["native_batcher_enabled"] == (
        None if native_batcher == "on" else False
    )
    assert report["options"]["native_batcher_error"] == (
        "native builder unavailable" if native_batcher == "on" else None
    )


def test_json_reports_ambient_mock_storage(
    json_benchmark, monkeypatch, tmp_path
):
    import json

    bench, path, _ = json_benchmark
    monkeypatch.setattr(bench.storage_cache, "_enabled", True)
    monkeypatch.setattr(bench.storage_cache, "_location", "host")
    output = tmp_path / "ambient.json"
    printed = []
    bench.run_benchmark(
        [path], skip_gds_read=True, output_json=output, out=printed.append
    )
    report = json.loads(output.read_bytes())
    assert report["storage"] == {
        "mode": "host",
        "requested_mock_storage": None,
    }
    assert report["options"]["native_batcher_enabled"] is False
    assert "mock-storage=host" in printed[2]


@pytest.mark.parametrize(
    "destination", ["input", "directory", "missing-parent"]
)
def test_json_destination_is_checked_before_benchmark_phases(
    json_benchmark, tmp_path, destination
):
    bench, path, phases = json_benchmark
    output = {
        "input": path,
        "directory": tmp_path,
        "missing-parent": tmp_path / "missing" / "report.json",
    }[destination]
    with pytest.raises((ValueError, OSError)):
        bench.run_benchmark([path], output_json=output, out=lambda _: None)
    assert phases == []
    assert path.read_bytes() == b"synthetic payload"


def test_json_is_not_published_after_preflight_failure(
    json_benchmark, monkeypatch, tmp_path
):
    bench, path, phases = json_benchmark
    output = tmp_path / "report.json"
    monkeypatch.setattr(bench, "gpu_available", lambda: False)
    with pytest.raises(RuntimeError, match="GPU / kvikio"):
        bench.run_benchmark([path], output_json=output, out=lambda _: None)
    assert not output.exists()
    assert phases == []


def test_json_atomic_write_preserves_previous_report_on_exception(
    json_benchmark, monkeypatch, tmp_path
):
    bench, path, phases = json_benchmark
    output = tmp_path / "report.json"
    output.write_text("previous report")

    def preload_failure(_path):
        raise OSError("preload failed")

    monkeypatch.setattr(bench.storage_cache, "preload", preload_failure)
    with pytest.raises(OSError, match="preload failed"):
        bench.run_benchmark(
            [path],
            mock_storage_kind="host",
            output_json=output,
            out=lambda _: None,
        )
    assert output.read_text() == "previous report"
    assert phases == []
    assert not list(tmp_path.glob(".report.json.*"))


def test_json_preserves_previous_report_when_terminal_output_fails(
    json_benchmark, tmp_path
):
    bench, path, phases = json_benchmark
    output = tmp_path / "report.json"
    output.write_text("previous report")

    def closed_output(_line):
        raise BrokenPipeError("terminal output closed")

    with pytest.raises(BrokenPipeError, match="terminal output closed"):
        bench.run_benchmark([path], output_json=output, out=closed_output)

    assert len(phases) == 4
    assert output.read_text() == "previous report"
    assert not list(tmp_path.glob(".report.json.*"))


def test_benchmark_without_json_keeps_return_and_text_output(
    json_benchmark, monkeypatch
):
    bench, path, phases = json_benchmark

    def unexpected_metadata():
        pytest.fail("optional report probes must not run without output_json")

    monkeypatch.setattr(bench, "cpp_helper_available", unexpected_metadata)
    printed = []
    assert bench.run_benchmark([path], out=printed.append) == phases
    assert any("batch_to_device_stream" in line for line in printed)


def test_json_reporting_preserves_cold_native_plan_timings(
    monkeypatch, tmp_path
):
    import json

    import cuphoton.xdr.nvcomp_batch as nvcomp_batch

    bench = _load_benchmark_module()
    path = tmp_path / "input.fits"
    path.touch()
    monkeypatch.setattr(bench, "gpu_available", lambda: True)
    monkeypatch.setattr(bench, "cp", SimpleNamespace(__version__="test"))
    monkeypatch.setattr(bench, "is_gds_active", lambda: False)
    monkeypatch.setattr(bench.storage_cache, "_enabled", False)
    state = {"clock": 0.0, "loaded": False, "imports": 0}

    def native_plan(*args):
        state["clock"] += 0.005
        return []

    extension = SimpleNamespace(
        plan_native_files=native_plan, NativeBatchBuilder=object
    )

    def load_extension():
        if not state["loaded"]:
            state["clock"] += 0.125
            state["loaded"] = True
            state["imports"] += 1
        return extension

    monkeypatch.setattr(nvcomp_batch, "_try_get_cpp_ext", load_extension)
    monkeypatch.setattr(bench.time, "perf_counter", lambda: state["clock"])
    monkeypatch.setattr(
        bench,
        "bench_batch_load",
        lambda *args, **kwargs: bench.make_result("batch", [1.0], 0, ""),
    )
    measured = []
    output = tmp_path / "report.json"
    for destination in (None, output):
        state.update(clock=0.0, loaded=False, imports=0)
        results = bench.run_benchmark(
            [path],
            iterations=2,
            output_json=destination,
            out=lambda _line: None,
        )
        measured.append(results[0].elapsed_ms)
        assert state["imports"] == 1
        assert results[0].ok
    assert measured[0] == pytest.approx(67.5)
    assert measured[1] == pytest.approx(measured[0])
    report = json.loads(output.read_text())
    assert report["phases"][0]["elapsed_ms"] == measured[1]
    assert report["capabilities"]["cpp_helper_available"] is True
    assert report["options"]["native_batcher_enabled"] is True


def test_benchmark_cli_reports_invalid_json_destination(
    json_benchmark, tmp_path, capsys
):
    _, path, phases = json_benchmark
    rc = run_component(
        "xdr",
        [
            "benchmark-fits",
            "--output-json",
            str(tmp_path / "missing" / "report.json"),
            str(path),
        ],
    )
    assert rc == 1
    assert "No such file or directory" in capsys.readouterr().err
    assert phases == []
