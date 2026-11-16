# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check preview destinations and real notebook timeout cleanup."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pytest
from nbclient.exceptions import CellTimeoutError

HERE = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(
        f"model_sky_{name}", HERE / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preview_rejects_output_inside_guide(tmp_path, monkeypatch, capsys):
    preview = load_script("preview")
    repo = tmp_path / "repo"
    output = repo / "docs/guide/local-preview"
    output.mkdir(parents=True)
    monkeypatch.setattr(preview, "REPO", repo)
    monkeypatch.setattr(
        sys,
        "argv",
        ["preview.py", "--executed", "unused.ipynb", "--output", str(output)],
    )
    with pytest.raises(SystemExit) as exc:
        preview.main()
    assert exc.value.code == 2
    assert "outside the guide source" in capsys.readouterr().err
    assert list(output.iterdir()) == []


def test_stalled_notebook_times_out_and_stops_kernel(tmp_path, monkeypatch):
    runner = load_script("run")
    source = tmp_path / "source"
    source.mkdir()
    cells = [
        nbformat.v4.new_code_cell(f"{flag} = False") for flag in runner.FLAGS
    ]
    cells += [nbformat.v4.new_code_cell("import time; time.sleep(60)")]
    cells += [nbformat.v4.new_code_cell("pass") for _ in range(17)]
    nbformat.write(
        nbformat.v4.new_notebook(cells=cells), source / "model-sky.ipynb"
    )
    output = tmp_path / "output"
    monkeypatch.setattr(runner, "HERE", source)
    monkeypatch.setattr(sys, "argv", ["run.py", "--output", str(output)])
    # Isolate input preparation; notebook execution and cleanup remain real.
    monkeypatch.setattr(runner, "sha256", lambda path: "unused")
    monkeypatch.setattr(
        runner, "shutil", SimpleNamespace(copy2=lambda *args: None)
    )
    monkeypatch.setattr(
        runner,
        "subprocess",
        SimpleNamespace(run=lambda *args, **kwargs: None),
    )
    manager = runner.AsyncKernelManager(kernel_name="python3")
    monkeypatch.setattr(
        runner, "AsyncKernelManager", lambda **kwargs: manager
    )
    client_class = runner.NotebookClient

    def short_timeout(notebook, **kwargs):
        kwargs["timeout"] = 1
        return client_class(notebook, **kwargs)

    monkeypatch.setattr(runner, "NotebookClient", short_timeout)
    with pytest.raises(CellTimeoutError):
        runner.main()
    assert not manager.has_kernel
    executed = nbformat.read(
        output / "model-sky.executed.ipynb", as_version=4
    )
    assert executed.cells[0].execution_count is not None
    assert executed.cells[-1].execution_count is None
