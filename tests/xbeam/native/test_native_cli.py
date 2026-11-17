# SPDX-License-Identifier: Apache-2.0
"""Native CLI dispatch must not enter Python radio planning or execution."""

from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from cuphoton.core.cli import run_component
from cuphoton.xbeam import commands, native


@pytest.fixture(autouse=True)
def isolated_cli_state(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))


def _arguments(tmp_path):
    job = tmp_path / "job.input"
    job.write_text("parsed only by C++")
    return [
        "native-correlate",
        str(job),
        "--output",
        str(tmp_path / "output"),
        "--index-dir",
        str(tmp_path),
        "--scan",
        "scan",
        "--integration-count",
        "3",
        "--profile",
        "interop-real-usb-buffered-window-replay-v1",
    ]


def test_native_cli_bypasses_python_execution(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("native CLI entered the Python data plane")

    for name in ("load_interop_job", "load_job", "prepare_correlation"):
        monkeypatch.setattr(commands, name, forbidden)
    monkeypatch.setattr(native, "cuda_available", lambda: True)
    calls = []

    def execute(*args, **kwargs):
        calls.append((args, kwargs))
        return {"records": 1, "backend": "native-cuda"}

    monkeypatch.setattr(native, "correlate", execute)
    error = StringIO()
    status = run_component(
        "xbeam",
        _arguments(tmp_path)
        + [
            "--vdif-io",
            "cufile",
            "--device",
            "2",
            "--fft-tile",
            "400",
            "--upload-workers",
            "2",
            "--prefetch",
            "off",
        ],
        err=error,
    )
    assert status == 0, error.getvalue()
    assert len(calls) == 1
    args, options = calls[0]
    assert str(args[0]) == str(tmp_path / "job.input")
    assert args[1:] == (tmp_path / "output", tmp_path, "scan")
    assert options == dict(
        first_integration=0,
        integration_count=3,
        fft_tile=400,
        device=2,
        vdif_io="cufile",
        upload_workers=2,
        prefetch=False,
        profile="interop-real-usb-buffered-window-replay-v1",
    )


def test_native_cli_missing_cuda_is_actionable(tmp_path, monkeypatch):
    monkeypatch.setattr(native, "cuda_available", lambda: False)
    error = StringIO()
    assert run_component("xbeam", _arguments(tmp_path), err=error) != 0
    assert "CUPHOTON_XBEAM_NATIVE_CUDA=1" in error.getvalue()
    assert not (tmp_path / "output").exists()


def test_native_cli_expands_quoted_home_job_path(tmp_path, monkeypatch):
    monkeypatch.setattr(native, "cuda_available", lambda: True)
    calls = []

    def execute(job, *args, **kwargs):
        calls.append(job)
        return {"records": 1}

    monkeypatch.setattr(native, "correlate", execute)
    with TemporaryDirectory(dir=Path.home()) as directory:
        job = Path(directory) / "job.input"
        job.write_text("parsed only by C++")
        arguments = _arguments(tmp_path)
        arguments[1] = "~/" + str(job.relative_to(Path.home()))
        error = StringIO()
        assert run_component("xbeam", arguments, err=error) == 0, (
            error.getvalue()
        )
        assert calls == [str(job)]


def test_native_cli_requires_explicit_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(native, "cuda_available", lambda: True)
    monkeypatch.setattr(
        native, "correlate", lambda *a, **k: pytest.fail("executed")
    )
    error = StringIO()
    assert run_component("xbeam", _arguments(tmp_path)[:-2], err=error) != 0
    assert "profile" in error.getvalue()
