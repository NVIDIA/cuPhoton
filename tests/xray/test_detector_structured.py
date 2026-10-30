# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
import threading
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from cuphoton.core.cli import run_component
from cuphoton.xray import _detector_structured as scheduler
from cuphoton.xray import _structured_gpu, _structured_native, structured
from cuphoton.xray import detector_artifacts as artifacts
from cuphoton.xray.detector_mask import AxisRange


def _tile(start, height=16, *, skipped=()):
    skip = np.zeros(height, dtype=np.bool_)
    skip[list(skipped)] = True
    return artifacts._PreparedDetectorTile(
        bounds=(0, 16, start, start + height),
        skip_rows=skip,
        traces_gpu=np.broadcast_to(
            np.arange(start + 1, start + height + 1, dtype=np.float64),
            (798, height),
        ).copy(),
        time_gpu=np.arange(798, dtype=np.float64),
    )


@pytest.fixture
def stages(monkeypatch):
    state = SimpleNamespace(
        prepared=[],
        finished=[],
        batches=[],
        owners=[],
        events=[],
        prepare_hook=None,
        finish_hook=None,
        solve_hook=None,
        close_hook=None,
    )

    def prepare(time, trace):
        marker = float(trace[0])
        state.prepared.append(marker)
        if state.prepare_hook is not None:
            state.prepare_hook(marker)
        return SimpleNamespace(
            coefficients=np.full(597, marker), marker=marker
        )

    def finish(prepared, roots, *, root_device, fit_diagnostics):
        marker = prepared.marker
        state.finished.append(marker)
        if state.finish_hook is not None:
            state.finish_hook(marker)
        np.testing.assert_array_equal(roots, np.full(597, marker))
        return SimpleNamespace(
            frequency=np.array([1.0, 2.0]),
            spectrum_components=np.array([[0.0], [1.0]]),
            amplitude=np.array([marker]),
        )

    def solve(polynomial, **kwargs):
        markers = -polynomial[:, 1]
        state.batches.append(markers.tolist())
        state.events.append("solve-start")
        roots = np.broadcast_to(markers[:, None], (len(markers), 597)).astype(
            np.complex128
        )
        status = np.zeros((len(markers), 8), dtype=np.int32)
        status[:, 1] = 597
        if state.solve_hook is not None:
            state.solve_hook(markers, status)
        state.events.append("solve-end")
        return roots, status, np.zeros((len(markers), 256), dtype=np.uint8)

    class FakeGPU:
        def __init__(self, cp):
            self.owner = threading.get_ident()
            state.owners.append(self.owner)

        def solve(self, polynomial):
            assert threading.get_ident() == self.owner
            return solve(polynomial)

        def close(self):
            assert threading.get_ident() == self.owner
            state.events.append("close")
            if state.close_hook is not None:
                state.close_hook()

    def fft(cp, time, traces):
        markers = traces[..., 0]
        values = np.stack((markers, markers + 0.5), axis=-1)
        return np.array([0.0, 1.0]), values

    monkeypatch.setattr(structured, "prepare_fit", prepare)
    monkeypatch.setattr(structured, "finish_fit", finish)
    monkeypatch.setattr(_structured_native, "solve_roots", solve)
    monkeypatch.setattr(_structured_native, "require_native", lambda: None)
    monkeypatch.setattr(_structured_gpu, "StructuredRootsGPU", FakeGPU)
    monkeypatch.setattr(artifacts, "_tdsfft_cupy", fft)
    monkeypatch.setattr(
        scheduler,
        "runtime_metadata",
        lambda **kwargs: {
            "package_version": "test",
            "device": None,
            **kwargs,
        },
    )
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace())
    return state


def _fit(tiles, *, root_device="cpu", schedule="serial", provenance=None):
    return scheduler.fit_structured_tiles(
        cp=np,
        tiles=iter(tiles),
        fit_options={"padded_length": 4},
        root_device=root_device,
        host_workers=2,
        root_workers=1,
        batch_target=16,
        schedule=schedule,
        provenance={} if provenance is None else provenance,
    )


@pytest.mark.parametrize(
    ("root_device", "schedule"),
    [("cpu", "serial"), ("gpu", "serial"), ("gpu", "overlap")],
)
def test_scheduler_preserves_order_masks_and_scalar_edge(
    stages, root_device, schedule
):
    tiles = [
        _tile(0),
        _tile(16, skipped=range(16)),
        _tile(32, 2, skipped=(1,)),
    ]
    provenance = {}
    with _fit(
        tiles,
        root_device=root_device,
        schedule=schedule,
        provenance=provenance,
    ) as results:
        fitted = list(results)

    assert [tile.bounds for tile in fitted] == [tile.bounds for tile in tiles]
    assert [batch["active_rows"] for batch in provenance["batches"]] == [
        16,
        0,
        1,
    ]
    assert list(map(len, stages.batches)) == [16, 1]
    assert sorted(stages.prepared) == list(range(1, 17)) + [33]
    assert sorted(stages.finished) == sorted(stages.prepared)
    assert fitted[1].rows == (None,) * 16
    assert fitted[2].rows[1] is None
    assert [tile.batched for tile in fitted] == [True, False, False]
    for source, result in zip(tiles, fitted, strict=True):
        for row in np.flatnonzero(~source.skip_rows):
            marker = source.traces_gpu[0, row]
            np.testing.assert_array_equal(
                result.rows[row]["amp"], [marker, 0, 0, 0]
            )
            np.testing.assert_array_equal(
                result.rows[row]["fft"], [marker, marker + 0.5, 0, 0]
            )
            np.testing.assert_array_equal(
                result.rows[row]["fft_freq"], [0, 1, 0, 0]
            )
    np.testing.assert_array_equal(fitted[2].rows[0]["freq"], [2, 0, 0, 0])
    if root_device == "gpu":
        assert len(stages.owners) == 1
        assert stages.owners[0] != threading.get_ident()
        assert stages.events[-1] == "close"


@pytest.mark.parametrize("stage", ["prepare", "roots", "finish"])
def test_scheduler_returns_row_failures_without_retry(stages, stage):
    def fail(marker):
        if marker == 1:
            raise ValueError("rejected row")

    if stage == "roots":

        def reject(markers, status):
            status[0, 0] = 5

        stages.solve_hook = reject
    else:
        setattr(stages, stage + "_hook", fail)
    with _fit([_tile(0, 2)]) as results:
        rows = next(results).rows
    assert isinstance(rows[0], ValueError)
    np.testing.assert_array_equal(rows[1]["amp"], [2, 0, 0, 0])
    assert len(stages.batches) == 1
    assert stages.prepared.count(1) == 1
    assert stages.finished.count(1) == (stage == "finish")


def test_overlap_observes_failed_gpu_before_preparing_next_tile(stages):
    failed = threading.Event()

    def fail(markers, status):
        failed.set()
        raise RuntimeError("GPU batch failed")

    stages.solve_hook = fail

    def tiles():
        yield _tile(0)
        assert failed.wait(5), "GPU worker did not run"
        yield _tile(16)
        pytest.fail("read beyond the failed batch")

    with pytest.raises(RuntimeError, match="GPU batch failed"):
        with _fit(tiles(), root_device="gpu", schedule="overlap") as results:
            list(results)
    assert sorted(stages.prepared) == list(range(1, 17))
    assert len(stages.batches) == 1
    assert stages.events[-1] == "close"


@pytest.mark.parametrize("schedule", ["serial", "overlap"])
@pytest.mark.parametrize("close_fails", [False, True])
def test_gpu_failure_preserves_original_and_distinct_cleanup_errors(
    stages, schedule, close_fails
):
    original = RuntimeError("GPU batch failed")
    close_error = RuntimeError("GPU close failed")

    def fail_solve(markers, status):
        raise original

    def fail_close():
        raise close_error

    stages.solve_hook = fail_solve
    if close_fails:
        stages.close_hook = fail_close
    with pytest.raises(RuntimeError, match="GPU batch failed") as caught:
        with _fit(
            [_tile(0)], root_device="gpu", schedule=schedule
        ) as results:
            list(results)
    assert caught.value is original
    assert getattr(original, "__notes__", []) == (
        [f"Structured fitting cleanup: {close_error!r}"]
        if close_fails
        else []
    )
    assert stages.events[-1] == "close"


@pytest.mark.parametrize("stage", ["prepare", "roots"])
def test_overlap_yields_known_row_error_before_next_gpu_solve(stages, stage):
    def fail(marker):
        if marker == 1:
            raise ValueError("rejected row")

    def reject(markers, status):
        status[0, 0] = 5

    if stage == "prepare":
        stages.prepare_hook = fail
    else:
        stages.solve_hook = reject
    with _fit(
        [_tile(0), _tile(16)], root_device="gpu", schedule="overlap"
    ) as results:
        first = next(results)
        assert isinstance(first.rows[0], ValueError)
        assert len(stages.batches) == 1
    assert len(stages.batches) == 1
    assert stages.events[-1] == "close"


def test_consumer_failure_drains_inflight_batch_on_its_owner(stages):
    second_started = threading.Event()
    release_second = threading.Event()
    consumer_stopped = threading.Event()
    finished = threading.Event()
    failures = []

    def hold_second(markers, status):
        if markers[0] == 17:
            second_started.set()
            assert release_second.wait(5), (
                "test did not release the GPU worker"
            )

    stages.solve_hook = hold_second

    def consume():
        try:
            with _fit(
                [_tile(0), _tile(16), _tile(32)],
                root_device="gpu",
                schedule="overlap",
            ) as results:
                next(results)
                consumer_stopped.set()
                raise RuntimeError("consumer stopped")
        except BaseException as error:
            failures.append(error)
        finally:
            finished.set()

    thread = threading.Thread(target=consume)
    thread.start()
    try:
        assert consumer_stopped.wait(5)
        assert second_started.wait(5)
        assert not finished.is_set()
        assert "close" not in stages.events
    finally:
        release_second.set()
        thread.join(5)
    assert not thread.is_alive()
    assert len(failures) == 1
    assert str(failures[0]) == "consumer stopped"
    assert stages.events[-2:] == ["solve-end", "close"]
    assert sorted(stages.prepared) == list(range(1, 33))
    assert len(stages.batches) == 2


@pytest.mark.parametrize(
    "options",
    [
        {"components": 29},
        {"fit_method": "iterative"},
        {"roots_backend": "auto"},
        {"p2_ridge_alpha": 0.1},
        {"fit_diagnostics": "summary"},
        {"batch_rows": False},
        {"tile_shape": (17, 16)},
        {"tile_shape": (16, 17)},
        {"shard_index": 0, "shard_count": 2},
        {"structured_host_workers": 0},
        {"structured_schedule": "overlap"},
    ],
)
def test_unsupported_structured_options_fail_before_native_or_input(
    tmp_path, monkeypatch, options
):
    monkeypatch.setattr(
        _structured_native,
        "require_native",
        lambda: pytest.fail("unsupported settings loaded native code"),
    )
    with pytest.raises(ValueError, match="structured"):
        artifacts.build_detector_artifacts_cupy(
            h5dir=tmp_path,
            fon="missing-on.h5",
            foff="missing-off.h5",
            output_dir=tmp_path / "output",
            fit_backend="structured-cpu",
            **options,
        )
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "identity",
    [
        artifacts._detector_artifact_config_hash,
        artifacts.detector_artifact_resume_identity,
    ],
)
def test_resume_identity_distinguishes_solver_refinement_and_execution(
    identity, detector_result
):
    manifest = json.loads(detector_result.manifest_path.read_text())
    metadata = manifest["structured"]
    dense = {**manifest, "backend": "cupy", "roots_backend": "eigvals"}
    del dense["structured"]
    assert identity(dense) != identity(manifest)
    for key, value in metadata.items():
        changed = value + 1 if isinstance(value, int) else f"changed-{key}"
        candidate = {
            **manifest,
            "structured": {**metadata, key: changed},
        }
        if key in {"runtime", "batches"}:
            assert identity(candidate) == identity(manifest), key
        else:
            assert identity(candidate) != identity(manifest), key


def test_serial_fitting_stops_before_unrequested_rows_and_tiles(monkeypatch):
    read_tiles = []
    fitted_rows = []

    def tiles():
        for start in (0, 16):
            read_tiles.append(start)
            yield _tile(start)

    def fail_row(*, trace_gpu, **kwargs):
        fitted_rows.append(float(trace_gpu[0]))
        raise ValueError("rejected row")

    monkeypatch.setattr(artifacts, "_fit_detector_row", fail_row)
    with artifacts._fit_detector_tiles(
        cp=np, tiles=tiles(), fit_options={"batch_rows": False}
    ) as results:
        assert read_tiles == []
        first = next(results)
        assert read_tiles == [0]
        assert fitted_rows == []
        error = next(iter(first.rows))
        assert isinstance(error, ValueError)
        assert str(error) == "rejected row"
    assert fitted_rows == [1]
    assert read_tiles == [0]


@pytest.mark.parametrize("explicit", [False, True])
def test_cli_forwards_structured_options_and_preserves_default(
    tmp_path, monkeypatch, capsys, explicit
):
    captured = {}

    def build(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(to_dict=lambda: {"complete": True})

    monkeypatch.setattr(artifacts, "build_detector_artifacts_cupy", build)
    arguments = [
        "detector-artifacts",
        "--h5dir",
        str(tmp_path),
        "--fon",
        "on.h5",
        "--foff",
        "off.h5",
        "--output-dir",
        str(tmp_path / "output"),
        "--json",
    ]
    if explicit:
        arguments += [
            "--fit-backend",
            "structured-gpu",
            "--structured-host-workers",
            "4",
            "--structured-root-workers",
            "2",
            "--structured-batch-rows",
            "64",
            "--structured-schedule",
            "overlap",
        ]
    assert run_component("xray", arguments) == 0
    assert json.loads(capsys.readouterr().out) == {"complete": True}
    assert captured["fit_backend"] == (
        "structured-gpu" if explicit else "cupy"
    )
    assert captured["structured_host_workers"] == (4 if explicit else 8)
    assert captured["structured_root_workers"] == (2 if explicit else 1)
    assert captured["structured_batch_rows"] == (64 if explicit else 33024)
    assert captured["structured_schedule"] == (
        "overlap" if explicit else "serial"
    )


@pytest.fixture
def detector_result(tmp_path, stages):
    samples, height, width = 799, 3, 2
    off = np.ones((samples, height, width))
    on = off + np.linspace(0.01, 0.03, samples)[:, None, None]
    for filename, image in (("on.h5", on), ("off.h5", off)):
        with h5py.File(tmp_path / filename, "w") as handle:
            handle["ROI"] = np.ones((height, width))
            for name in ("bin_count", "i0", "i0_ipm3"):
                handle[name] = np.ones(samples)
            handle["imgs"] = image
            handle["scan_var"] = np.arange(samples, dtype=np.float64)
    result = artifacts.build_detector_artifacts_cupy(
        h5dir=tmp_path,
        fon="on.h5",
        foff="off.h5",
        output_dir=tmp_path / "output",
        tile_shape=(2, 2),
        exclude_y=(AxisRange(1, 2),),
        drop_leading=0,
        zero_offset_index=0,
        integrate_pixels=0,
        fit_backend="structured-cpu",
        structured_host_workers=2,
        structured_batch_rows=16,
        artifact_layout="tile-rows",
    )
    return result


def test_detector_publishes_structured_masks_and_provenance(detector_result):
    result = detector_result
    assert result.raw_fits == 2
    assert result.failures == 0
    assert result.skipped_fits == 1
    status = np.load(result.output_dir / artifacts.FIT_STATUS_FILE)
    np.testing.assert_array_equal(status[1], artifacts.FIT_STATUS_SKIPPED)
    np.testing.assert_array_equal(status[[0, 2]], artifacts.FIT_STATUS_OK)
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["backend"] == "structured-cpu"
    assert manifest["roots_backend"] == "structured-rds"
    assert (
        manifest["structured"]["refinement"] == structured.REFINEMENT_VERSION
    )
    assert artifacts.detector_artifact_complete(result.output_dir)
