# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check execution guards and receipts without a reference study."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import nbformat
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reference
import run
import run_study
import study


def git(repo, *args):
    return subprocess.check_output(
        ["git", *args], cwd=repo, text=True, stderr=subprocess.PIPE
    ).strip()


@pytest.fixture
def source_repo(tmp_path):
    repo = tmp_path / "repo"
    example = repo / "examples/model-alerts"
    example.mkdir(parents=True)
    (repo / "src").mkdir()
    (repo / "src/package.py").write_text("VALUE = 1\n")
    (repo / "LICENSE").write_text("test license\n")
    (example / "reference.py").write_text("# example\n")
    nbformat.write(nbformat.v4.new_notebook(), example / "model-alerts.ipynb")
    git(repo, "init", "--initial-branch=main")
    git(repo, "config", "user.name", "Test")
    git(repo, "config", "user.email", "test@example.org")
    git(repo, "config", "commit.gpgsign", "false")
    git(repo, "config", "core.hooksPath", "/dev/null")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "Initial source")
    return repo


@pytest.mark.parametrize("override", [None, "a" * 40, "invalid"])
def test_reference_resolves_revision_before_experiment(
    source_repo, tmp_path, monkeypatch, override
):
    monkeypatch.setattr(
        reference,
        "__file__",
        str(source_repo / "examples/model-alerts/reference.py"),
    )
    monkeypatch.delenv("CUPHOTON_MODEL_ALERTS_SOURCE_REPO", raising=False)
    monkeypatch.delenv("CUPHOTON_MODEL_ALERTS_REVISION", raising=False)
    if override is not None:
        monkeypatch.setenv("CUPHOTON_MODEL_ALERTS_REVISION", override)
    monkeypatch.chdir(tmp_path)
    called = []
    monkeypatch.setattr(
        reference, "_experiment", lambda root, **kwargs: called.append(kwargs)
    )
    output = tmp_path / "reference"
    if override == "invalid":
        with pytest.raises(ValueError, match="complete Git commit SHA"):
            reference.execute_reference(output, device="cpu")
        assert not output.exists()
        assert not called
    else:
        reference.execute_reference(output, device="cpu")
        assert called[0]["revision"] == (
            override or git(source_repo, "rev-parse", "HEAD")
        )


@pytest.mark.parametrize("change", [None, "package", "revision"])
def test_runner_rejects_source_changes_before_passed_receipt(
    source_repo, tmp_path, monkeypatch, change
):
    initial_revision = git(source_repo, "rev-parse", "HEAD")
    output = tmp_path / "result"
    monkeypatch.setattr(run, "REPO", source_repo)
    monkeypatch.setattr(run, "HERE", source_repo / "examples/model-alerts")
    monkeypatch.setattr(
        sys, "argv", ["run.py", "--output", str(output), "--device", "cpu"]
    )
    monkeypatch.setattr(
        run,
        "AsyncKernelManager",
        lambda **kwargs: SimpleNamespace(
            has_kernel=False, kernel_spec=SimpleNamespace(argv=[])
        ),
    )

    def execute(**kwargs):
        assert (
            kwargs["env"]["CUPHOTON_MODEL_ALERTS_REVISION"]
            == initial_revision
        )
        if change == "package":
            (source_repo / "src/package.py").write_text("VALUE = 2\n")
        elif change == "revision":
            git(source_repo, "commit", "--allow-empty", "-m", "New revision")
        (output / "study").mkdir()
        (output / "study/report.json").write_text('{"design": {}}')
        (output / "study/freeze.json").write_text("{}")

    monkeypatch.setattr(
        run,
        "NotebookClient",
        lambda *args, **kwargs: SimpleNamespace(execute=execute),
    )
    monkeypatch.setattr(
        run.importlib.metadata, "version", lambda name: "test"
    )
    if change:
        with pytest.raises(RuntimeError, match="source changed"):
            run.main()
        assert not (output / "execution.json").exists()
    else:
        run.main()
        receipt = json.loads((output / "execution.json").read_text())
        assert receipt["status"] == "passed"
        assert receipt["cuphoton_revision"] == initial_revision


def test_runner_times_out_stalled_cell_and_releases_kernel(
    source_repo, tmp_path, monkeypatch
):
    from nbclient.exceptions import CellTimeoutError

    example = source_repo / "examples/model-alerts"
    notebook = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell("import time; time.sleep(30)")]
    )
    nbformat.write(notebook, example / "model-alerts.ipynb")
    git(source_repo, "add", ".")
    git(source_repo, "commit", "-m", "Stalled notebook")
    output = tmp_path / "timeout"
    monkeypatch.setattr(run, "REPO", source_repo)
    monkeypatch.setattr(run, "HERE", example)
    monkeypatch.setattr(
        sys, "argv", ["run.py", "--output", str(output), "--device", "cpu"]
    )
    managers = []
    notebook_client = run.NotebookClient

    def timed_client(*args, **kwargs):
        kwargs["timeout"] = 1
        client = notebook_client(*args, **kwargs)
        managers.append(kwargs["km"])
        return client

    monkeypatch.setattr(run, "NotebookClient", timed_client)
    with pytest.raises(CellTimeoutError):
        run.main()
    assert not managers[0].has_kernel
    assert (output / "model-alerts.executed.ipynb").is_file()
    assert not (output / "execution.json").exists()


@pytest.mark.parametrize(
    "cuda,with_lock", [(True, True), (True, False), (False, False)]
)
def test_auto_device_obeys_gpu_lock(tmp_path, monkeypatch, cuda, with_lock):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    args = [
        "run_study.py",
        "--output",
        str(tmp_path / "pilot"),
        "--device",
        "auto",
    ]
    if with_lock:
        args.extend(["--gpu-lock", str(tmp_path / "gpu.lock")])
    monkeypatch.setattr(sys, "argv", args)
    events = []
    monkeypatch.setattr(
        run_study.fcntl, "flock", lambda *args: events.append("lock")
    )
    monkeypatch.setattr(
        run_study,
        "pilot",
        lambda *args, **kwargs: events.append(kwargs["device"]),
    )
    if cuda and not with_lock:
        with pytest.raises(SystemExit):
            run_study.main()
        assert events == []
    else:
        run_study.main()
        assert events == (["lock", "cuda"] if cuda else ["cpu"])


def test_auto_device_is_resolved_for_training_and_validation(
    tmp_path, monkeypatch
):
    import torch

    from cuphoton.xscan import dataset, training

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    data = tmp_path / "data"
    data.mkdir()
    np.save(data / "search.npy", np.zeros((2, 31, 31)))
    devices = []

    def train(config, **kwargs):
        devices.append(config.device)
        return dict(epochs_completed=1, early_stopping={})

    def load(*args, device):
        devices.append(str(device))
        return object(), None, None

    def predict(**kwargs):
        devices.append(str(kwargs["device"]))
        return dict(
            labels=np.array([0, 1]),
            probabilities=np.array([0.1, 0.9]),
            metadata_rows=[dict(field=1), dict(field=2)],
        )

    monkeypatch.setattr(training, "train_classifier", train)
    monkeypatch.setattr(training, "load_model_from_checkpoint", load)
    monkeypatch.setattr(training, "predict_dataset", predict)
    monkeypatch.setattr(
        dataset, "StampDataset", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        study, "paired_field_bootstrap", lambda *args: {"a": [0, 1]}
    )
    result = study.train_development(
        data,
        tmp_path / "train",
        mode="pair",
        seed=11,
        epochs=1,
        device="auto",
    )
    assert devices == ["cpu", "cpu", "cpu"]
    assert result["epochs_completed"] == 1
    assert (tmp_path / "train/validation.npz").is_file()
