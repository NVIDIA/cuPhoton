# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""xImage store inspection and read benchmark helpers."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from cuphoton.ximage.data import (
    ARTIFACTS,
    Artifact,
    load_ximage_bundle,
)
from cuphoton.ximage.data.bundle import (
    spatial_roi_slices,
)


def resolve_artifacts(spec: str) -> list[Artifact]:
    if spec == "all":
        return [ARTIFACTS[key] for key in ARTIFACTS]
    keys = [key.strip() for key in spec.split(",") if key.strip()]
    unknown = [key for key in keys if key not in ARTIFACTS]
    if unknown:
        raise ValueError(f"Unknown artifact keys: {', '.join(unknown)}")
    return [ARTIFACTS[key] for key in keys]


def benchmark_store(
    *,
    store: Path,
    fmt: str,
    artifact: Artifact,
    roi_height: int,
    roi_width: int,
    repeats: int,
) -> dict[str, Any]:
    try:
        bundle = load_ximage_bundle(
            store, prefer=(fmt,), artifacts=(artifact.key,)
        )
    except FileNotFoundError as exc:
        raise ValueError(
            f"Store {store} is not a detectable {fmt} store."
        ) from exc
    with bundle:
        array = bundle[artifact.key]
        roi = roi_slice(array.shape, artifact, roi_height, roi_width)
        sample = np.array(array[roi], copy=True)
        out_shape = sample.shape
        out_bytes = int(sample.nbytes)
        del sample
        timings = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            block = np.array(array[roi], copy=True)
            timings.append(time.perf_counter() - t0)
            del block
        return {
            "format": fmt,
            "store": str(store),
            "artifact": artifact.key,
            "roi": repr(roi),
            "shape": list(out_shape),
            "bytes": out_bytes,
            "timings_sec": timings,
            "mean_sec": sum(timings) / len(timings),
            "throughput_mb_s": (out_bytes / 1_000_000)
            / (sum(timings) / len(timings)),
        }


def roi_slice(
    shape: tuple[int, ...],
    artifact: Artifact,
    roi_height: int,
    roi_width: int,
) -> tuple[slice | int, ...]:
    return spatial_roi_slices(
        shape, artifact, height=roi_height, width=roi_width
    )
