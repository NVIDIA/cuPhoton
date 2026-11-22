# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Reproduce the development iteration sweep before freezing."""

import argparse
import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np
from experiment import reconstruct
from measure import evaluate
from scene import generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--gpu-lock", type=Path, required=True)
    parser.add_argument(
        "--iterations", type=int, nargs="+", default=[25, 75, 150, 300]
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[1005, 1006, 1007]
    )
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("Use a fresh output destination")
    args.output.mkdir(parents=True)
    sources = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path(__file__).parent.glob("*.py")
    }
    rows = []
    with open(args.gpu_lock, "a") as lock:
        if args.device == "cuda":
            fcntl.flock(lock, fcntl.LOCK_EX)
        import torch

        torch.set_num_threads(4)
        for seed in args.seeds:
            arrays, truth = generate(seed, development=True)
            for iterations in args.iterations:
                images = {}
                for name, algorithm in [("l2", "l2"), ("huber", "rho")]:
                    image, _, _ = reconstruct(
                        args.output,
                        arrays,
                        name=f"{seed}-{iterations}-{name}",
                        algorithm=algorithm,
                        device=args.device,
                        iterations=iterations,
                    )
                    images[name] = image
                measured, products, _ = evaluate(arrays, truth, images)
                rows.append(
                    dict(
                        seed=seed,
                        iterations=iterations,
                        results=measured,
                        ranges={
                            name: [float(im.min()), float(im.max())]
                            for name, im in products.items()
                        },
                    )
                )
                (args.output / "development.json").write_text(
                    json.dumps(
                        dict(
                            source_hashes=sources,
                            development_only=True,
                            rows=rows,
                        ),
                        indent=2,
                    )
                    + "\n"
                )
                print(
                    seed,
                    iterations,
                    {
                        name: float(
                            np.mean(
                                [
                                    abs(s["fractional_flux_error"])
                                    for s in result["sources"]
                                ]
                            )
                        )
                        for name, result in measured.items()
                    },
                    flush=True,
                )


if __name__ == "__main__":
    main()
