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
import numpy as np
from controls import asymmetric_operator_control, numerical_controls
from jupyter_client import AsyncKernelManager
from nbclient import NotebookClient
from scene import science_window
from study import execute_study

import cuphoton

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_source(repo, receipt):
    """Reject a changed revision or uncommitted reference inputs."""
    receipt["source_tree_clean"] = not subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo, text=True
    ).strip()
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    if revision != receipt["source_revision"]:
        raise RuntimeError("Source revision changed during execution")
    if receipt["profile"] == "reference" and not receipt["source_tree_clean"]:
        raise RuntimeError("Source tree changed during reference execution")


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
    repo = HERE.parents[1]
    if args.output.resolve().is_relative_to(repo):
        parser.error("Build outside the source checkout")
    clean = not subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo, text=True
    ).strip()
    if args.profile == "reference" and not clean:
        parser.error("Reference execution requires a clean committed source")
    import_path = Path(cuphoton.__file__).resolve()
    if not import_path.is_relative_to(repo / "src"):
        parser.error("cuPhoton must import from this exact source checkout")
    protocol_path = HERE / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if args.profile == "reference" and not protocol["frozen"]:
        parser.error("Reference execution requires a frozen protocol")
    if args.profile == "reference" and not protocol.get(
        "science_source_hashes"
    ):
        parser.error(
            "Reference protocol must include scientific source hashes"
        )
    tracked = (
        subprocess.check_output(["git", "ls-files", "-z", "."], cwd=HERE)
        .decode()
        .split("\0")
    )
    source_hashes = {name: digest(HERE / name) for name in tracked if name}
    for name, expected in protocol.get("science_source_hashes", {}).items():
        if digest(HERE / name) != expected:
            parser.error(f"Frozen scientific source changed: {name}")
    output = args.output.resolve()
    output.mkdir(parents=True)
    for name in source_hashes:
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(HERE / name, target)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env = os.environ.copy()
    env["CUPHOTON_MODEL_SCENE_OUTPUT"] = str(output)
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(HERE), env.get("PYTHONPATH", "")]
    )
    quick = args.profile == "quick"
    science_start = (
        protocol["development_seeds"][0]
        if quick
        else protocol["test_seed_start"]
    )
    calibration_start = 2000 if quick else protocol["calibration_seed_start"]
    receipt = dict(
        completed=False,
        source_tree_clean=clean,
        source_import=import_path.relative_to(repo).as_posix(),
        visible_devices=env["CUDA_VISIBLE_DEVICES"],
        science_seeds=list(
            range(
                science_start,
                science_start
                + (2 if quick else protocol["noise_realizations"]),
            )
        ),
        calibration_seeds=list(
            range(calibration_start, calibration_start + (2 if quick else 50))
        ),
        profile=args.profile,
        device=args.device,
        source_hashes=source_hashes,
        protocol_sha256=digest(protocol_path),
        python=platform.python_version(),
        architecture=platform.machine(),
        runtime_cuphoton_version=cuphoton.__version__,
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
    try:
        lock_context = (
            open(args.gpu_lock, "a")
            if args.gpu_lock
            else contextlib.nullcontext()
        )
        with lock_context as lock:
            if args.device == "cuda":
                fcntl.flock(lock, fcntl.LOCK_EX)
            check_source(repo, receipt)
            import torch

            torch.set_num_threads(4)
            if args.device == "cuda":
                receipt["gpu"] = {
                    "model": torch.cuda.get_device_name(0),
                    "capability": list(torch.cuda.get_device_capability(0)),
                    "cuda_runtime": torch.version.cuda,
                }
            controls = {
                "cpu": numerical_controls(output / "controls-cpu"),
                "asymmetric": asymmetric_operator_control(
                    output / "controls-asymmetric"
                ),
            }
            if args.device == "cuda":
                controls["cuda"] = numerical_controls(
                    output / "controls-cuda", device="cuda"
                )
                cpu = np.load(
                    output / "controls-cpu/runs/rates/artifacts/image.npy"
                ).squeeze()
                gpu = np.load(
                    output / "controls-cuda/runs/rates/artifacts/image.npy"
                ).squeeze()
                np.testing.assert_allclose(
                    science_window(cpu),
                    science_window(gpu),
                    rtol=1e-8,
                    atol=1e-8,
                )
                controls["cpu_cuda_max_abs"] = float(
                    np.max(np.abs(science_window(cpu) - science_window(gpu)))
                )
            (output / "controls.json").write_text(
                json.dumps(controls, indent=2) + "\n"
            )
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
                km=manager,
                timeout=1200,
                allow_errors=False,
                record_timing=True,
            )
            try:
                client.execute(cwd=str(HERE), env=env, cleanup_kc=True)
            finally:
                nbformat.write(
                    notebook, output / "model-scene.executed.ipynb"
                )
        check_source(repo, receipt)
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
    except BaseException as exc:
        receipt.update(
            completed=False, error_type=type(exc).__name__, error=str(exc)
        )
        (output / "execution.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )
        raise
    print(f"Completed {args.profile} execution: {output}")


if __name__ == "__main__":
    main()
