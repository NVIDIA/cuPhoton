# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run a fresh study and notebook, retaining outputs outside Git."""

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient
from study import execute_study

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--profile", choices=["quick", "reference"], default="quick"
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--gpu-lock", type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("Use an absent output directory; preserve earlier runs")
    if args.device == "cuda" and args.gpu_lock is None:
        parser.error("CUDA execution requires the coordinated --gpu-lock")
    protocol_path = HERE / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if args.profile == "reference" and not protocol["frozen"]:
        parser.error("Reference execution requires a frozen protocol")
    source_hashes = {
        p.relative_to(HERE).as_posix(): digest(p)
        for p in HERE.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    for name, expected in protocol.get("science_source_hashes", {}).items():
        if digest(HERE / name) != expected:
            parser.error(f"Frozen scientific source changed: {name}")
    output = args.output.resolve()
    output.mkdir(parents=True)
    for name in source_hashes:
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(HERE / name, target)
    env = os.environ.copy()
    env["CUPHOTON_MODEL_SCENE_OUTPUT"] = str(output)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(HERE), env.get("PYTHONPATH", "")]
    )
    receipt = dict(
        profile=args.profile,
        device=args.device,
        source_hashes=source_hashes,
        protocol_sha256=digest(protocol_path),
        python=platform.python_version(),
        architecture=platform.machine(),
        runtime_cuphoton_version=__import__("cuphoton").__version__,
        source_revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=HERE, text=True
        ).strip(),
        packages={
            name: version(name)
            for name in (
                "cuphoton",
                "torch",
                "numpy",
                "scipy",
                "bokeh",
                "nbclient",
            )
        },
    )
    (output / "execution-profile.json").write_text(
        json.dumps(
            {
                "profile": args.profile,
                "full_study_requested": args.profile == "reference",
            }
        )
    )
    lock_context = (
        open(args.gpu_lock, "a")
        if args.gpu_lock
        else contextlib.nullcontext()
    )
    with lock_context as lock:
        if args.device == "cuda":
            fcntl.flock(lock, fcntl.LOCK_EX)
        import torch

        torch.set_num_threads(4)
        with (output / "execution.log").open("w") as log:
            with (
                contextlib.redirect_stdout(log),
                contextlib.redirect_stderr(log),
            ):
                execute_study(
                    output / "study",
                    protocol,
                    device=args.device,
                    quick=args.profile == "quick",
                )
        notebook = nbformat.read(HERE / "model-scene.ipynb", as_version=4)
        manager = KernelManager(kernel_name="python3")
        manager.kernel_spec.argv = [
            sys.executable,
            "-m",
            "ipykernel_launcher",
            "-f",
            "{connection_file}",
        ]
        client = NotebookClient(
            notebook,
            km=manager,
            timeout=1200,
            allow_errors=False,
            record_timing=True,
        )
        try:
            client.execute(cwd=str(HERE), env=env)
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
            nbformat.write(notebook, output / "model-scene.executed.ipynb")
    for name, expected in source_hashes.items():
        if digest(HERE / name) != expected:
            raise RuntimeError(f"Source changed during execution: {name}")
    receipt["completed"] = True
    receipt["artifact_hashes"] = {
        p.relative_to(output).as_posix(): digest(p)
        for p in output.rglob("*")
        if p.is_file() and "source" not in p.relative_to(output).parts
    }
    (output / "execution.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(f"Completed {args.profile} execution: {output}")


if __name__ == "__main__":
    main()
