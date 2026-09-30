# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Measure warmed xFit fits with a fixed candidate cohort and CPU budget.

Use --compare-native for interleaved CuPy/native rounds.
--execution processes uses spawn; MPS, if wanted, must be managed externally.
Wall timing includes scheduling and result IPC; worker timing measures the
fit call through NumPy result materialization. Validation is outside timing.
Changing --workers also changes fit batch sizes, which can change numerical
trajectories and evaluation counts. Backend comparisons at a fixed worker
count use identical batches. Output records worker_batch_sizes in worker
order.
Use --timeout-seconds to allow longer rounds when cold compilation needs it.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import fields
from typing import Any

import numpy as np

from cuphoton.xfit import GaussianDipoleModel, fit_dipoles

_worker = threading.local()


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


def _initialize(images, initial, mode, workers, counter, barrier, gpu):
    with counter.get_lock():
        index = counter.value
        counter.value += 1
    start, stop = (
        index * len(images) // workers,
        (index + 1) * len(images) // workers,
    )
    _worker.index = index
    _worker.images = images[start:stop].copy()
    _worker.initial = initial[start:stop].copy()
    _worker.variance = np.ones_like(_worker.images)
    _worker.mode = mode
    _worker.barrier = barrier
    _worker.stream = None
    _worker.pool = None
    if gpu:
        import cupy as cp

        _worker.cp = cp
        _worker.stream = cp.cuda.Stream(non_blocking=True)
        _worker.pool = cp.cuda.MemoryPool()


def _run(treatment):
    # Exactly one task per worker participates in each round. A missing or
    # failed worker breaks the bounded wait instead of hanging the pool.
    _worker.barrier.wait(timeout=60)
    if treatment is None:
        return _worker.index
    if treatment == "close":
        native = sys.modules.get("cuphoton.xfit._native")
        if native is not None:
            native.release_workspace()
        if _worker.stream is not None:
            _worker.stream.synchronize()
            _worker.pool.free_all_blocks()
        return _worker.index
    backend = treatment
    context = _worker.stream if _worker.stream is not None else nullcontext()
    allocator = (
        _worker.cp.cuda.using_allocator(_worker.pool.malloc)
        if _worker.pool is not None
        else nullcontext()
    )
    with context, allocator:
        start = time.perf_counter()
        result = fit_dipoles(
            _worker.images,
            model="gaussian",
            initial=_worker.initial,
            variance=_worker.variance,
            mode=_worker.mode,
            backend=backend,
        )
        seconds = time.perf_counter() - start
    if result.backend != backend:
        raise AssertionError(
            f"requested {backend}, received {result.backend}"
        )
    gil = getattr(sys, "_is_gil_enabled", lambda: True)()
    native = sys.modules.get("cuphoton.xfit._native")
    timings = native.last_timings() if backend == "native" else None
    return _worker.index, result, seconds, gil, timings


def _round(pool, treatment, workers, timeout_seconds=180):
    start = time.perf_counter()
    futures = [pool.submit(_run, treatment) for _ in range(workers)]
    deadline = time.monotonic() + timeout_seconds
    results, failure = [], None
    for future in futures:
        try:
            results.append(
                future.result(timeout=max(0, deadline - time.monotonic()))
            )
        except Exception as exc:
            # Drain the remaining synchronous calls before scheduling any
            # cleanup; retain the first failure as the reported cause.
            failure = exc if failure is None else failure
    if failure is not None:
        raise failure
    return sorted(results), time.perf_counter() - start


def _validate(actual, expected, dtype):
    # Existing cross-kernel scientific tolerances, fixed before measuring.
    tolerance = 3e-5 if dtype == np.dtype("float32") else 5e-12
    for field in fields(expected):
        if field.name in {"backend", "device"}:
            continue
        value, reference = (
            getattr(actual, field.name),
            getattr(expected, field.name),
        )
        if isinstance(reference, np.ndarray):
            if (
                value.shape != reference.shape
                or value.dtype != reference.dtype
            ):
                raise AssertionError(
                    f"array shape/dtype mismatch: {field.name}"
                )
            if reference.dtype.kind == "f":
                np.testing.assert_allclose(
                    value,
                    reference,
                    rtol=tolerance,
                    atol=tolerance,
                    equal_nan=True,
                    err_msg=field.name,
                )
            else:
                np.testing.assert_array_equal(
                    value, reference, err_msg=field.name
                )
        elif value != reference:
            raise AssertionError(f"scientific mismatch: {field.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        action="append",
        choices=("numpy", "cupy", "cutile", "native", "numba-cuda-mlir"),
    )
    parser.add_argument("--compare-native", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--execution", choices=("threads", "processes"), default="threads"
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=None,
        help="Restrict all workers together to this many inherited CPUs.",
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
    parser.add_argument("--repeat", "--rounds", type=int, default=5)
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=180,
        help="Per-round wait, including cold compilation (default: 180).",
    )
    args = parser.parse_args()
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive and finite")
    if args.repeat <= 0:
        parser.error("--repeat must be positive")
    if args.batch and any(batch <= 0 for batch in args.batch):
        parser.error("--batch must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    if args.workers < 1 or args.height < 1 or args.width < 1:
        parser.error("workers and image dimensions must be positive")
    if args.compare_native and args.backend:
        parser.error("--compare-native cannot be combined with --backend")
    backends = args.backend or ["cupy", "cutile"]
    batches = args.batch or [1, 16, 256, 4096]
    if any(batch < args.workers for batch in batches):
        parser.error("every batch must contain at least --workers candidates")
    allowed = (
        sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None
    )
    if args.cpus is not None:
        if allowed is None or not 1 <= args.cpus <= len(allowed):
            parser.error(
                "--cpus requires a positive count within inherited "
                "Linux affinity"
            )
        os.sched_setaffinity(0, allowed[: args.cpus])
        allowed = sorted(os.sched_getaffinity(0))
    treatments = ["cupy", "native"] if args.compare_native else backends
    # Native comparisons always validate against CuPy, including when only
    # native timings were requested. Reference work is outside measurements.
    reference = "cupy" if "native" in treatments else treatments[0]
    dtype = np.dtype(args.dtype)
    context = multiprocessing.get_context("spawn")
    for batch in batches:
        images, initial = _fixture(
            batch, (args.height, args.width), dtype, args.mode
        )
        executor = (
            ThreadPoolExecutor
            if args.execution == "threads"
            else ProcessPoolExecutor
        )
        executor_options = (
            {} if args.execution == "threads" else {"mp_context": context}
        )
        start = time.perf_counter()
        with executor(
            max_workers=args.workers,
            initializer=_initialize,
            initargs=(
                images,
                initial,
                args.mode,
                args.workers,
                context.Value("i", 0),
                context.Barrier(args.workers),
                any(backend != "numpy" for backend in treatments),
            ),
            **executor_options,
        ) as pool:
            _round(pool, None, args.workers, args.timeout_seconds)
            startup = time.perf_counter() - start
            warmups, samples, last = {}, {t: [] for t in treatments}, {}
            try:
                expected, reference_seconds = _round(
                    pool, reference, args.workers, args.timeout_seconds
                )
                for treatment in treatments:
                    start = time.perf_counter()
                    for _ in range(args.warmup):
                        results, _ = _round(
                            pool,
                            treatment,
                            args.workers,
                            args.timeout_seconds,
                        )
                        for actual, baseline in zip(
                            results, expected, strict=True
                        ):
                            _validate(actual[1], baseline[1], dtype)
                    warmups[treatment] = time.perf_counter() - start
                orders = []
                for round_index in range(args.repeat):
                    offset = round_index % len(treatments)
                    order = treatments[offset:] + treatments[:offset]
                    orders.append(order)
                    for treatment in order:
                        results, wall = _round(
                            pool,
                            treatment,
                            args.workers,
                            args.timeout_seconds,
                        )
                        for actual, baseline in zip(
                            results, expected, strict=True
                        ):
                            _validate(actual[1], baseline[1], dtype)
                        samples[treatment].append(
                            {
                                "wall_milliseconds": 1e3 * wall,
                                "worker_materialized_milliseconds": [
                                    1e3 * row[2] for row in results
                                ],
                                "native_iteration_timings": [
                                    row[4] for row in results
                                ],
                            }
                        )
                        last[treatment] = results
            finally:
                original_error = sys.exception()
                try:
                    _round(pool, "close", args.workers, args.timeout_seconds)
                except Exception as cleanup_error:
                    if original_error is None:
                        raise
                    original_error.add_note(
                        f"worker cleanup also failed: {cleanup_error!r}"
                    )
        for treatment in treatments:
            rows = last[treatment]
            wall_ms = [
                sample["wall_milliseconds"] for sample in samples[treatment]
            ]
            print(
                json.dumps(
                    {
                        "backend": treatment,
                        "execution": args.execution,
                        "workers": args.workers,
                        "batch": batch,
                        "worker_batch_sizes": [
                            len(row[1].status) for row in rows
                        ],
                        "dtype": dtype.name,
                        "mode": args.mode,
                        "image_shape": [args.height, args.width],
                        "cpu_affinity": allowed,
                        "memory_pool_policy": "private-per-worker",
                        "worker_gil_enabled": [row[3] for row in rows],
                        "python": sys.version,
                        "mps_pipe_configured": bool(
                            os.environ.get("CUDA_MPS_PIPE_DIRECTORY")
                        ),
                        "startup_seconds": startup,
                        "reference_seconds": reference_seconds,
                        "warmup_seconds": warmups[treatment],
                        "warmup_rounds": args.warmup,
                        "timeout_seconds": args.timeout_seconds,
                        "round_order": orders,
                        "samples": samples[treatment],
                        "milliseconds_min": min(wall_ms),
                        "milliseconds_median": float(np.median(wall_ms)),
                        "milliseconds_max": max(wall_ms),
                        "status": dict(
                            Counter(
                                value
                                for row in rows
                                for value in row[1].status.tolist()
                            )
                        ),
                        "evaluations_median": float(
                            np.median(
                                np.concatenate(
                                    [row[1].evaluations for row in rows]
                                )
                            )
                        ),
                        "validation": "all scientific fields match reference",
                        "reference_backend": reference,
                    },
                    sort_keys=True,
                )
            )


if __name__ == "__main__":
    main()
