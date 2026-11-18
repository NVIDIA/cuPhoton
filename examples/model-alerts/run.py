# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Snapshot and freshly execute the complete Model Alerts notebook."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import nbformat
from jupyter_client import AsyncKernelManager
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_files():
    return sorted(
        p
        for p in HERE.rglob("*")
        if p.is_file()
        and "__pycache__" not in p.parts
        and ".pytest_cache" not in p.parts
        and p.suffix
        in (".py", ".js", ".css", ".html", ".md", ".ipynb", ".txt")
    )


def source_status():
    return subprocess.check_output(
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            "src",
            "examples/model-alerts",
            ".gitattributes",
        ],
        cwd=REPO,
        text=True,
    ).strip()


def source_revision():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=["quick-smoke", "full-reference"],
        default="quick-smoke",
    )
    parser.add_argument("--gpu-lock", type=Path)
    parser.add_argument(
        "--device", choices=["cpu", "cuda:0"], default="cuda:0"
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output == REPO or output.is_relative_to(REPO):
        parser.error(
            "--output must be a new directory outside example sources"
        )
    if args.device.startswith("cuda") and args.gpu_lock is None:
        parser.error("CUDA execution requires --gpu-lock")
    if args.gpu_lock is not None and not args.gpu_lock.parent.is_dir():
        parser.error("shared GPU lock parent must already exist")
    revision = source_revision()
    if source_status():
        parser.error(
            "commit example and cuPhoton source edits before execution"
        )
    files = source_files()
    hashes = {str(p.relative_to(HERE)): digest(p) for p in files}
    output.mkdir(parents=True)
    snapshot = output / "source"
    for p in files:
        destination = snapshot / p.relative_to(HERE)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, destination)
    with zipfile.ZipFile(
        output / "model-alerts-source.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for p in files:
            archive.write(p, "model-alerts/" + str(p.relative_to(HERE)))
        archive.write(REPO / "LICENSE", "model-alerts/LICENSE")
    notebook = nbformat.read(snapshot / "model-alerts.ipynb", as_version=4)
    manager = AsyncKernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [
        sys.executable,
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ]
    client = NotebookClient(
        notebook,
        timeout=None if args.profile == "full-reference" else 7200,
        km=manager,
        resources={"metadata": {"path": str(snapshot)}},
        record_timing=True,
    )
    environment = os.environ.copy()
    environment.update(
        CUPHOTON_MODEL_ALERTS_OUTPUT=str(output / "study"),
        CUPHOTON_MODEL_ALERTS_SOURCE_REPO=str(REPO),
        CUPHOTON_MODEL_ALERTS_PROFILE=args.profile,
        CUPHOTON_MODEL_ALERTS_GPU_LOCK=str(args.gpu_lock.resolve())
        if args.gpu_lock is not None
        else "",
        CUPHOTON_MODEL_ALERTS_DEVICE=args.device,
        PYTHONPATH=os.pathsep.join(
            [
                str(REPO / "src"),
                str(snapshot),
                *[
                    str(Path(p).resolve())
                    for p in environment.get("PYTHONPATH", "").split(
                        os.pathsep
                    )
                    if p
                ],
            ]
        ),
        CUPHOTON_MODEL_ALERTS_REVISION=revision,
    )
    started = time.monotonic()
    try:
        client.execute(env=environment, cwd=str(snapshot), cleanup_kc=True)
    finally:
        # Preserve warnings and numbers; replace only machine-local prefixes,
        # following the Model Sky execution workflow.
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            for item in cell.outputs:
                replacements = [
                    (str(output), "<reference-output>"),
                    (str(REPO), "<cuphoton-source>"),
                    (sys.prefix, "<python-environment>"),
                    (str(Path(sys.prefix).resolve()), "<python-environment>"),
                    (str(Path.home()), "<user-home>"),
                ]

                def portable(text, replacements=tuple(replacements)):
                    for prefix, label in sorted(
                        replacements, key=lambda p: -len(p[0])
                    ):
                        text = text.replace(prefix, label)
                    return text

                if item.output_type == "stream":
                    item.text = portable(item.text)
                elif item.output_type in ("display_data", "execute_result"):
                    for mime in ("text/plain", "text/html", "text/markdown"):
                        if isinstance(item.data.get(mime), str):
                            item.data[mime] = portable(item.data[mime])
        nbformat.write(notebook, output / "model-alerts.executed.ipynb")
    errors = [
        o
        for c in notebook.cells
        if c.cell_type == "code"
        for o in c.outputs
        if o.output_type == "error"
    ]
    if errors:
        raise RuntimeError("notebook execution contains errors")
    if source_status() or source_revision() != revision:
        raise RuntimeError(
            "cuPhoton source changed during reference execution"
        )
    if any(
        digest(HERE / name) != expected for name, expected in hashes.items()
    ):
        raise RuntimeError("source changed during reference execution")
    report = json.loads((output / "study/report.json").read_text())
    receipt = dict(
        status="passed",
        profile=args.profile,
        elapsed_seconds=time.monotonic() - started,
        source_sha256=hashes,
        cuphoton_revision=revision,
        dependencies={
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "scipy",
                "torch",
                "bokeh",
                "nbclient",
                "nbformat",
            )
        },
        output_sha256={
            name: digest(output / name)
            for name in (
                "model-alerts.executed.ipynb",
                "model-alerts-source.zip",
                "study/report.json",
                "study/freeze.json",
            )
        },
        code_cells=sum(c.cell_type == "code" for c in notebook.cells),
        study_profile=report["design"],
        artifact_sha256={
            str(p.relative_to(output)): digest(p)
            for p in output.rglob("*")
            if p.is_file()
            and "__pycache__" not in p.parts
            and p.suffix
            in (
                ".npy",
                ".npz",
                ".pt",
                ".parquet",
                ".jsonl",
                ".json",
                ".ipynb",
                ".zip",
            )
        },
    )
    (output / "execution.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
