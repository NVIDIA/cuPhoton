# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Development-only workflow exercise; never produces test-study claims."""

import argparse
import fcntl
import json
from pathlib import Path

import numpy as np
from experiment import calibrate, reconstruct
from scene import generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--gpu-lock", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("Use a fresh output destination")
    args.output.mkdir(parents=True)
    # Retain this file descriptor for the entire CUDA workload.
    with open(args.gpu_lock, "a") as lock:
        if args.device == "cuda":
            fcntl.flock(lock, fcntl.LOCK_EX)
        import torch

        torch.set_num_threads(4)
        arrays, truth = generate(1001, development=True, matched=True)
        results = {}
        for name, algorithm in [("l2", "l2"), ("huber", "rho")]:
            image, run, secondary = reconstruct(
                args.output,
                arrays,
                name=name,
                algorithm=algorithm,
                device=args.device,
                iterations=75,
            )
            results[name] = {
                "shape": list(image.shape),
                "finite": bool(np.isfinite(image).all()),
                "run": run.run_dir.name,
            }
        for method in ("mm", "adam"):
            run, measured = calibrate(
                args.output, method=method, device=args.device
            )
            results[method] = {"run": run.run_dir.name}
        (args.output / "development.json").write_text(
            json.dumps(results, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
