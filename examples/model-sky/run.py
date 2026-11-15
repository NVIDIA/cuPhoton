# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare Model Sky inputs and execute the notebook and both exercises.

All generated files and a copy of the example sources go under --output. The
directory must be absent or empty; existing runs are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
FLAGS = ("RUN_BASELINE_EXERCISE", "RUN_BLEND_EXTENSION")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_receipt(path: Path, receipt: dict) -> None:
    path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("model-sky-output"),
        help="Absent or empty directory for inputs, results and notebook",
    )
    options = parser.parse_args()
    output = options.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(
            "--output must be absent or empty; keep existing runs intact"
        )

    source = HERE / "model-sky.ipynb"
    notebook = nbformat.read(source, as_version=4)
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    if len(code) != 20 or any(
        cell.outputs or cell.execution_count is not None for cell in code
    ):
        raise ValueError("Expected the unexecuted 20-cell Model Sky source")
    for flag in FLAGS:
        matches = [cell for cell in code if f"{flag} = False" in cell.source]
        if len(matches) != 1:
            raise ValueError(f"Expected one disabled source flag: {flag}")
        matches[0].source = matches[0].source.replace(
            f"{flag} = False", f"{flag} = True"
        )
    notebook.metadata["reference_execution"] = {
        "enabled_optional_flags": list(FLAGS),
        "source_defaults_unchanged": True,
        "relative_data": "data",
        "relative_output": "results",
    }

    sources = [
        source,
        *(
            HERE / name
            for name in (
                "sources.md",
                "requirements.txt",
                "model_sky.py",
                "scene.py",
                "pipeline.py",
                "workshop.py",
                "views.py",
                "blend.py",
                "notebook_style.py",
                "run.py",
            )
        ),
        *(
            HERE / "assets" / f"story-{name}.svg"
            for name in ("field", "compare", "motion", "transit")
        ),
    ]
    source_hashes = {
        path.relative_to(HERE).as_posix(): sha256(path) for path in sources
    }
    output.mkdir(parents=True, exist_ok=True)
    for path in sources:
        destination = output / path.relative_to(HERE)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)

    env = os.environ.copy()
    env["PATH"] = (
        str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    )
    env["CUPHOTON_MODEL_SKY_DATA"] = "data"
    env["CUPHOTON_MODEL_SKY_OUTPUT"] = "results"
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    receipt_path = output / "execution.json"
    receipt = {
        "schema_version": 1,
        "status": "incomplete",
        "created_utc": datetime.now(UTC).isoformat(),
        "source_sha256": source_hashes,
        "optional_flags_enabled": list(FLAGS),
    }
    write_receipt(receipt_path, receipt)

    print("Preparing Model Sky inputs", flush=True)
    subprocess.run(
        [sys.executable, "model_sky.py", "--output", "data"],
        cwd=output,
        env=env,
        check=True,
    )
    inputs = sorted(
        path for path in (output / "data").rglob("*") if path.is_file()
    )
    input_hashes = {
        path.relative_to(output).as_posix(): sha256(path) for path in inputs
    }
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [
        sys.executable,
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ]

    def cell_start(cell, cell_index, **kwargs):
        if cell.cell_type == "code":
            print(
                f"Cell {cell_index + 1}/{len(notebook.cells)}: "
                f"{cell.source.splitlines()[0][:80]}",
                flush=True,
            )

    client = NotebookClient(
        notebook,
        km=manager,
        timeout=600,
        allow_errors=False,
        record_timing=True,
        on_cell_start=cell_start,
    )
    executed = output / "model-sky.executed.ipynb"
    started = time.monotonic()
    try:
        client.execute(cwd=str(output), env=env)
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
        # Preserve every output, including warnings. Only machine-local path
        # prefixes in streams are replaced; figures and numbers are untouched.
        for cell in code:
            for item in cell.outputs:
                if item.output_type == "stream":
                    for prefix in {
                        sys.prefix,
                        str(Path(sys.prefix).resolve()),
                    }:
                        item.text = item.text.replace(
                            prefix, "<python-environment>"
                        )
                    item.text = item.text.replace(str(output), ".")
        nbformat.write(notebook, executed)
    elapsed = time.monotonic() - started
    if any(cell.execution_count is None for cell in code):
        raise RuntimeError("Notebook execution did not finish")
    bokeh_count = sum(
        "application/vnd.bokehjs_exec.v0+json" in item.get("data", {})
        for cell in code
        for item in cell.outputs
    )
    if bokeh_count != 6:
        raise RuntimeError(
            f"Expected six interactive figures, got {bokeh_count}"
        )
    for name, expected in source_hashes.items():
        if (
            sha256(HERE / name) != expected
            or sha256(output / name) != expected
        ):
            raise RuntimeError(f"Source changed during execution: {name}")
    for name, expected in input_hashes.items():
        if sha256(output / name) != expected:
            raise RuntimeError(f"Input changed during execution: {name}")

    environment = {
        "python": platform.python_version(),
        "architecture": platform.machine(),
        "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"],
        "packages": {
            name: version(name)
            for name in (
                "cuphoton",
                "cupy-cuda13x",
                "numpy",
                "scipy",
                "astropy",
                "batman-package",
                "bokeh",
                "nbclient",
                "nbformat",
                "ipykernel",
            )
        },
    }
    cuda = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "import json, cupy as cp; "
            "p = cp.cuda.runtime.getDeviceProperties(0); "
            "print(json.dumps({'gpu': p['name'].decode(), "
            "'compute_capability': [p['major'], p['minor']], "
            "'cuda_runtime': cp.cuda.runtime.runtimeGetVersion(), "
            "'cuda_driver': cp.cuda.runtime.driverGetVersion()}))",
        ],
        env=env,
        text=True,
    )
    environment.update(json.loads(cuda))
    products = sorted(
        path for path in (output / "results").rglob("*") if path.is_file()
    )
    receipt.update(
        status="passed",
        code_cells_executed=len(code),
        bokeh_documents=bokeh_count,
        elapsed_seconds=elapsed,
        input_sha256=input_hashes,
        output_sha256={
            path.relative_to(output).as_posix(): sha256(path)
            for path in [executed, *products]
        },
        environment=environment,
        environment_sha256=hashlib.sha256(
            json.dumps(environment, sort_keys=True).encode()
        ).hexdigest(),
        stream_path_normalization=(
            "Python environment and output directory prefixes only"
        ),
    )
    write_receipt(receipt_path, receipt)
    print(
        "Executed 20 cells and six interactive figures; see execution.json",
        flush=True,
    )


if __name__ == "__main__":
    main()
