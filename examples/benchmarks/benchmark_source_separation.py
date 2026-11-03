# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Measure ordered source separation on deterministic CPU populations."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
from statistics import median
from time import perf_counter

import numpy as np

from cuphoton.xpois.ois import _select_separated_sources

DISTANCE = 8


def _population(size, crowded):
    """Keep density constant as the image area and candidate count grow."""
    rng = np.random.default_rng(8721)
    sites = (size + 1) // 2 if crowded else size
    width = int(np.ceil(np.sqrt(sites)))
    site = np.arange(size, dtype=np.int64) % sites
    ys, xs = (site // width) * DISTANCE + 2, (site % width) * DISTANCE + 2
    if crowded:
        ys += rng.integers(-1, 2, size=size)
        xs += rng.integers(-1, 2, size=size)
    order = rng.permutation(size)
    records = np.rec.fromarrays(
        [ys[order], xs[order], np.arange(size, dtype=np.int64)],
        names="y,x,source_index",
    )
    coordinates = [
        (int(y), int(x)) for y, x in zip(records.y, records.x, strict=True)
    ]
    shape = (int(ys.max()) + 1, int(xs.max()) + 1)
    return records, coordinates, shape


def _tuple_select(coordinates, max_stamps):
    """Straightforward greedy oracle with prepared Python coordinates."""
    selected, indices = [], []
    distance_squared = DISTANCE * DISTANCE
    for index, (y, x) in enumerate(coordinates):
        if any(
            (y - other_y) ** 2 + (x - other_x) ** 2 < distance_squared
            for other_y, other_x in selected
        ):
            continue
        selected.append((y, x))
        indices.append(index)
        if len(selected) >= max_stamps:
            break
    return indices


def _record_select(records, max_stamps):
    """Historical quadratic NumPy record-scalar loop, for negative control."""
    selected = []
    distance_squared = DISTANCE * DISTANCE
    for candidate in records:
        if any(
            (candidate.y - other.y) ** 2 + (candidate.x - other.x) ** 2
            < distance_squared
            for other in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= max_stamps:
            break
    return selected


def _measure(implementation, records, coordinates, shape, cap):
    start = perf_counter()
    if implementation == "tuple":
        result = _tuple_select(coordinates, cap)
    elif implementation == "record":
        result = _record_select(records, cap)
    else:
        result = _select_separated_sources(records, shape, DISTANCE, cap)
    elapsed = perf_counter() - start
    indices = (
        result
        if implementation == "tuple"
        else [int(source.source_index) for source in result]
    )
    return elapsed, indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs=2, default=(512, 2048))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--implementation",
        choices=("current", "tuple", "record"),
        default="current",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    small, large = args.sizes
    if small < 6 or large <= small or args.repeats < 1:
        parser.error("require 6 <= small < large and repeats >= 1")
    if args.check and (large < 1024 or args.repeats < 5):
        parser.error("--check requires large >= 1024 and repeats >= 5")

    rows, checks = [], []
    for population in ("separated", "crowded"):
        uncapped = []
        for size in args.sizes:
            records, coordinates, shape = _population(
                size, population == "crowded"
            )
            for cap in (5, size):
                timings = {"oracle": [], "implementation": []}
                for repeat in range(args.repeats):
                    order = ("oracle", "implementation")
                    if repeat % 2:
                        order = order[::-1]
                    indices = {}
                    for name in order:
                        implementation = (
                            "tuple"
                            if name == "oracle"
                            else args.implementation
                        )
                        seconds, indices[name] = _measure(
                            implementation, records, coordinates, shape, cap
                        )
                        timings[name].append(seconds)
                    if indices["oracle"] != indices["implementation"]:
                        raise RuntimeError("ordered selections differ")
                row = {
                    "population": population,
                    "candidates": size,
                    "max_stamps": cap,
                    "selected": len(indices["oracle"]),
                    "input_sha256": hashlib.sha256(
                        records.tobytes()
                    ).hexdigest(),
                    "selected_indices_sha256": hashlib.sha256(
                        np.asarray(
                            indices["oracle"], dtype=np.int64
                        ).tobytes()
                    ).hexdigest(),
                    "seconds": timings,
                    "median_seconds": {
                        name: median(values)
                        for name, values in timings.items()
                    },
                    "median_paired_ratio": median(
                        measured / reference
                        for measured, reference in zip(
                            timings["implementation"],
                            timings["oracle"],
                            strict=True,
                        )
                    ),
                }
                rows.append(row)
                if cap == size:
                    uncapped.append(row)
        growth = (
            uncapped[1]["median_seconds"]["implementation"]
            / uncapped[0]["median_seconds"]["implementation"]
        )
        ratio = uncapped[1]["median_paired_ratio"]
        checks.append(
            {
                "population": population,
                "large_to_small_ratio": growth,
                "maximum_growth": 2 * large / small,
                "large_relative_to_tuple": ratio,
                "maximum_relative_to_tuple": 0.25,
                "passed": growth <= 2 * large / small and ratio <= 0.25,
            }
        )

    print(
        json.dumps(
            {
                "implementation": args.implementation,
                "python": platform.python_version(),
                "numpy": np.__version__,
                "selector_source_sha256": hashlib.sha256(
                    inspect.getsource(_select_separated_sources).encode()
                ).hexdigest(),
                "min_separation": DISTANCE,
                "ordered_indices_exact": True,
                "measurements": rows,
                "checks": checks,
                "check_requested": args.check,
            },
            indent=2,
        )
    )
    return int(args.check and not all(check["passed"] for check in checks))


if __name__ == "__main__":
    raise SystemExit(main())
