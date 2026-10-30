# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Create a deterministic xImage bundle from synthetic point sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import convolve2d


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")

    size, count, kernel_size = 64, 3, 9
    scene = np.full((size, size), 0.1, dtype=np.float64)
    scene[20, 22], scene[35, 40], scene[44, 18] = 30.0, 50.0, 20.0
    y, x = np.indices((kernel_size, kernel_size), dtype=np.float64)
    radius2 = (y - kernel_size // 2) ** 2 + (x - kernel_size // 2) ** 2
    psfs = []
    images = []
    rng = np.random.default_rng(0)
    noise_sigma = 0.01
    for sigma in (0.8, 1.0, 1.2):
        psf = np.exp(-radius2 / (2 * sigma**2))
        psf /= psf.sum()
        psfs.append(psf)
        blurred = convolve2d(scene, psf, mode="same", boundary="fill")
        images.append(blurred + rng.normal(0, noise_sigma, scene.shape))

    image_stack = np.stack(images, axis=-1)[None, ...]
    arrays = {
        "images": image_stack,
        "variances": np.full_like(image_stack, noise_sigma**2),
        "psfs": np.stack(psfs, axis=-1)[:, :, None, :],
    }
    args.output.mkdir(parents=True)
    for name, array in arrays.items():
        np.save(args.output / f"{name}.npy", array, allow_pickle=False)
    (args.output / "metadata.json").write_text(
        json.dumps({"source": "synthetic", "seed": 0}) + "\n",
        encoding="utf-8",
    )
    print(f"Created {count} synthetic frames in {args.output}")


if __name__ == "__main__":
    main()
