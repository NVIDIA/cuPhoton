# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Bounded whole-tile execution for the opt-in structured detector fitter."""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Event
from typing import Any

import numpy as np

from cuphoton.core.runtime import runtime_metadata


def validate_structured_options(
    *,
    fit_backend,
    components,
    fit_method,
    p2_ridge_alpha,
    roots_backend,
    batch_rows,
    fit_diagnostics,
    tile_shape,
    host_workers,
    root_workers,
    batch_target,
    schedule,
    shard_index,
):
    if fit_backend not in {"cupy", "structured-cpu", "structured-gpu"}:
        raise ValueError(
            "fit_backend must be cupy, structured-cpu or structured-gpu"
        )
    if fit_backend == "cupy":
        if (host_workers, root_workers, batch_target, schedule) != (
            8,
            1,
            33024,
            "serial",
        ):
            raise ValueError(
                "structured execution options require "
                "a structured fit backend"
            )
        return
    if not (
        components == 30
        and fit_method == "linear-prediction"
        and p2_ridge_alpha == 0
        and roots_backend == "eigvals"
        and batch_rows
        and fit_diagnostics == "none"
        and shard_index is None
        and all(0 < size <= 16 for size in tile_shape)
    ):
        raise ValueError(
            "structured fitting requires LP with 30 components, zero ridge, "
            "default roots selection, batched rows, no fit diagnostics or "
            "sharding, and tiles no larger than 16 by 16"
        )
    for name, value, low, high in (
        ("structured_host_workers", host_workers, 1, 64),
        ("structured_root_workers", root_workers, 1, 64),
        ("structured_batch_rows", batch_target, 16, 33024),
    ):
        if type(value) is not int or not low <= value <= high:
            raise ValueError(
                f"{name} must be an integer from {low} through {high}"
            )
    if schedule not in {"serial", "overlap"}:
        raise ValueError("structured_schedule must be serial or overlap")
    if fit_backend == "structured-cpu" and schedule != "serial":
        raise ValueError(
            "structured CPU fitting requires the serial schedule"
        )


def _drain(future):
    """Finish owned work even if the caller is interrupted while waiting."""
    interrupted = None
    while not future.done():
        try:
            wait([future])
        except BaseException as error:
            if interrupted is None:
                interrupted = error
    if interrupted is not None:
        raise interrupted
    return future.result()


def _shutdown(pool):
    interrupted = None
    while True:
        try:
            pool.shutdown(wait=True, cancel_futures=True)
            break
        except (KeyboardInterrupt, SystemExit) as error:
            if interrupted is None:
                interrupted = error
    if interrupted is not None:
        raise interrupted


@dataclass
class _Chunk:
    tiles: list[Any]
    targets: list[Any]
    prepared: list[Any]
    polynomial: np.ndarray
    errors: list[Exception | None]


@contextmanager
def fit_structured_tiles(
    *,
    cp,
    tiles,
    fit_options,
    root_device,
    host_workers,
    root_workers,
    batch_target,
    schedule,
    provenance,
):
    """Yield compact results with two host chunks and one GPU batch."""
    from . import detector_artifacts as da
    from . import structured
    from ._structured_native import solve_roots

    if cp is not np:
        raise ValueError("structured fitting requires NumPy preparation")
    cancelled = Event()
    host = ThreadPoolExecutor(
        max_workers=host_workers, thread_name_prefix="xray-host"
    )
    driver = None
    solver = None
    root_future = None
    results = None
    common_time = None
    provenance.update(
        solver=structured.BACKEND_VERSION,
        refinement=structured.REFINEMENT_VERSION,
        preparation="numpy-scipy",
        root_device=root_device,
        host_workers=host_workers,
        root_workers=root_workers,
        batch_rows=batch_target,
        schedule=schedule,
        max_active_host_chunks=2 if schedule == "overlap" else 1,
        max_device_batches=1 if root_device == "gpu" else 0,
        batches=[],
    )

    def initialize():
        nonlocal solver
        import cupy

        from ._structured_gpu import StructuredRootsGPU

        solver = StructuredRootsGPU(cupy)
        provenance["runtime"] = runtime_metadata(
            backend="cupy", dtype="float64"
        )

    def prepare_tile(tile):
        nonlocal common_time
        x0, x1, y0, y1 = tile.bounds
        skip = np.asarray(tile.skip_rows, dtype=np.bool_)
        if not (
            0 < x1 - x0 <= 16
            and 0 < y1 - y0 <= 16
            and tile.traces_gpu.shape == (798, y1 - y0)
            and tile.traces_gpu.dtype == np.float64
            and tile.time_gpu.shape == (798,)
            and tile.time_gpu.dtype == np.float64
            and skip.shape == (y1 - y0,)
        ):
            raise ValueError(
                "structured tiles require 798 FP64 samples "
                "and at most 16 by 16 pixels"
            )
        if common_time is None:
            common_time = tile.time_gpu.copy()
        elif tile.time_gpu.tobytes() != common_time.tobytes():
            raise ValueError(
                "structured tiles must share identical fitted times"
            )
        active = np.flatnonzero(~skip)
        traces = np.ascontiguousarray(tile.traces_gpu.T[active])
        padded = fit_options["padded_length"]
        fft = np.zeros((len(active), 2, padded), dtype=np.float64)
        if len(active) > 1:
            frequency, values = da._tdsfft_cupy(np, common_time, traces)
            size = min(padded, int(values.shape[-1]))
            fft[:, 0, :size] = np.abs(values[:, :size])
            fft[:, 1, :size] = np.abs(frequency[None, :size])
        elif len(active):
            frequency, values = da._tdsfft_cupy(
                np, common_time, tile.traces_gpu[:, int(active[0])]
            )
            fft[0, 0] = da._pad_abs(values, padded)
            fft[0, 1] = da._pad_abs(frequency, padded)
        return dict(
            bounds=tile.bounds,
            skip=skip,
            active=active,
            traces=traces,
            fft=fft,
            rows=[None] * len(skip),
        )

    def groups():
        group, active, physical = [], 0, 0
        for tile in tiles:
            if root_future is not None and root_future.done():
                root_future.result()
            item = prepare_tile(tile)
            group.append(item)
            active += len(item["active"])
            physical += len(item["skip"])
            if (
                active >= batch_target
                or physical >= 2 * batch_target
                or active == 0
            ):
                yield group
                group, active, physical = [], 0, 0
        if group:
            yield group

    def prepare(group):
        targets = [
            (tile, int(row), k)
            for tile in group
            for k, row in enumerate(tile["active"])
        ]
        errors: list[Exception | None] = [None] * len(targets)

        def p1(index):
            tile, _, k = targets[index]
            if cancelled.is_set():
                return None
            try:
                return structured.prepare_fit(common_time, tile["traces"][k])
            except Exception as error:
                errors[index] = error.with_traceback(None)
                return None

        prepared = list(host.map(p1, range(len(targets))))
        polynomial = np.full((len(targets), 598), np.nan, dtype=np.float64)
        polynomial[:, 0] = 1.0
        for index, value in enumerate(prepared):
            if value is not None:
                polynomial[index, 1:] = -value.coefficients
        provenance["batches"].append(
            dict(active_rows=len(targets), tiles=len(group))
        )
        return _Chunk(group, targets, prepared, polynomial, errors)

    def roots(chunk):
        if solver is None:
            return solve_roots(chunk.polynomial, workers=root_workers)
        return solver.solve(chunk.polynomial)

    def check_roots(chunk, value):
        values, status, messages = value
        for index in range(len(chunk.targets)):
            if chunk.errors[index] is None and (
                status[index, 0] != 0
                or status[index, 1] != 597
                or status[index, 2] != 0
                or not np.isfinite(values[index]).all()
            ):
                message = (
                    bytes(messages[index])
                    .split(b"\0", 1)[0]
                    .decode(errors="replace")
                )
                chunk.errors[index] = ValueError(
                    f"structured roots rejected: {status[index].tolist()}: "
                    f"{message}"
                )
        return values

    def finish(chunk, values):
        def suffix(index):
            tile, local, k = chunk.targets[index]
            if cancelled.is_set():
                return
            if chunk.errors[index] is not None:
                tile["rows"][local] = chunk.errors[index]
                return
            try:
                result = structured.finish_fit(
                    chunk.prepared[index],
                    values[index],
                    root_device=root_device,
                    fit_diagnostics=True,
                )
                tile["rows"][local] = dict(
                    freq=da._pad_abs(
                        da._frequency_centers(
                            result.frequency, result.spectrum_components
                        ),
                        fit_options["padded_length"],
                    ),
                    amp=da._pad_abs(
                        result.amplitude, fit_options["padded_length"]
                    ),
                    fft=tile["fft"][k, 0],
                    fft_freq=tile["fft"][k, 1],
                )
            except Exception as error:
                tile["rows"][local] = error.with_traceback(None)

        # Consume all workers before publishing this whole chunk.
        for _ in host.map(suffix, range(len(chunk.targets))):
            pass

    def emit(group):
        for tile in group:
            yield da._FittedDetectorTile(
                tile["bounds"],
                tile["skip"],
                tuple(tile["rows"]),
                len(tile["active"]) > 1,
            )

    def ordered():
        nonlocal root_future
        source = iter(groups())

        def advance():
            masked = []
            for group in source:
                chunk = prepare(group)
                if chunk.targets:
                    return masked, chunk
                # Preserve only compact all-masked records between chunks.
                masked.extend(group)
            return masked, None

        if schedule == "serial":
            for group in source:
                chunk = prepare(group)
                if chunk.targets:
                    if driver is None:
                        value = roots(chunk)
                    else:
                        root_future = driver.submit(roots, chunk)
                        value = _drain(root_future)
                        root_future = None
                    finish(chunk, check_roots(chunk, value))
                yield from emit(group)
            return

        masked, current = advance()
        yield from emit(masked)
        if current is not None:
            root_future = driver.submit(roots, current)
        while current is not None:
            prefetched = not any(current.errors)
            between, following = advance() if prefetched else ([], None)
            values = check_roots(current, _drain(root_future))
            root_future = None
            if following is not None and not any(current.errors):
                root_future = driver.submit(roots, following)
            finish(current, values)
            yield from emit(current.tiles)
            yield from emit(between)
            # Release the completed chunk before gathering a third one.
            current = following
            if not prefetched:
                masked, current = advance()
                yield from emit(masked)
            if current is not None and root_future is None:
                root_future = driver.submit(roots, current)

    try:
        if root_device == "gpu":
            driver = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="xray-gpu-owner"
            )
            _drain(driver.submit(initialize))
        else:
            provenance["runtime"] = runtime_metadata(
                backend="cpu", dtype="float64"
            )
        results = ordered()
        yield results
    finally:
        original = sys.exc_info()[1]
        cancelled.set()
        failures = []

        def cleanup(action):
            try:
                action()
            except BaseException as error:
                if error is not original:
                    failures.append(error)

        if root_future is not None:
            cleanup(lambda: _drain(root_future))
        if driver is not None:

            def close_gpu():
                nonlocal solver
                try:
                    if solver is not None:
                        solver.close()
                finally:
                    solver = None

            cleanup(lambda: _drain(driver.submit(close_gpu)))
            cleanup(lambda: _shutdown(driver))
        cleanup(lambda: _shutdown(host))
        if results is not None:
            cleanup(results.close)
        if failures:
            if original is None:
                raise failures[0]
            for error in failures:
                original.add_note(f"Structured fitting cleanup: {error!r}")
