# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Measure warmed end-to-end xFit Gaussian fits."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from typing import Any

import numpy as np

from cuphoton.xfit import GaussianDipoleModel, fit_dipoles


def _fixture(
    batch: int,
    image_shape: tuple[int, int],
    dtype: np.dtype[Any],
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    base = np.asarray([8.0, 1.7, 2.3, 0.2, 1.1, -0.7, -1.5, 0.9], dtype=dtype)
    truth = np.repeat(base[None, :], batch, axis=0)
    row = np.arange(batch, dtype=np.float64)
    truth[:, 0] *= (1.0 + 0.05 * np.sin(0.17 * row)).astype(dtype)
    truth[:, 3] += (0.08 * np.sin(0.11 * row)).astype(dtype)
    truth[:, 4] += (0.15 * np.sin(0.07 * row)).astype(dtype)
    truth[:, 6] += (0.12 * np.cos(0.09 * row)).astype(dtype)
    model = GaussianDipoleModel(image_shape, dtype=dtype)
    images = np.asarray(model.evaluate(truth, mode=mode))
    initial = truth + np.asarray(
        [0.2, 0.1, -0.1, 0.03, 0.1, -0.1, -0.1, 0.1], dtype=dtype
    )
    return images, initial


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", action="append", choices=("numpy", "cupy", "cutile")
    )
    parser.add_argument("--batch", type=int, action="append")
    parser.add_argument(
        "--dtype", choices=("float32", "float64"), default="float64"
    )
    parser.add_argument(
        "--mode", choices=("difference", "split"), default="difference"
    )
    parser.add_argument("--height", type=int, default=17)
    parser.add_argument("--width", type=int, default=21)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    if args.repeat <= 0:
        parser.error("--repeat must be positive")
    backends = args.backend or ["cupy", "cutile"]
    batches = args.batch or [1, 16, 256, 4096]
    dtype = np.dtype(args.dtype)

    for batch in batches:
        images, initial = _fixture(
            batch, (args.height, args.width), dtype, args.mode
        )
        for backend in backends:
            for _ in range(args.warmup):
                result = fit_dipoles(
                    images,
                    model="gaussian",
                    initial=initial,
                    mode=args.mode,
                    backend=backend,
                )
            samples = []
            for _ in range(args.repeat):
                start = time.perf_counter()
                result = fit_dipoles(
                    images,
                    model="gaussian",
                    initial=initial,
                    mode=args.mode,
                    backend=backend,
                )
                samples.append(1.0e3 * (time.perf_counter() - start))
            print(
                json.dumps(
                    {
                        "backend": backend,
                        "batch": batch,
                        "dtype": dtype.name,
                        "mode": args.mode,
                        "image_shape": [args.height, args.width],
                        "milliseconds_min": min(samples),
                        "milliseconds_median": float(np.median(samples)),
                        "milliseconds_max": max(samples),
                        "status": dict(Counter(result.status.tolist())),
                        "evaluations_median": float(
                            np.median(result.evaluations)
                        ),
                    },
                    sort_keys=True,
                )
            )


if __name__ == "__main__":
    main()
