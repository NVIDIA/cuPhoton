# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Benchmark xdr GPU FITS loading.

It uses xDataReader's native CFITSIO planning path instead of Astropy HDU
objects. The benchmark operates on explicit HDU indices. By default it
benchmarks HDU 1; use ``--hdu-indices 1,2,3`` for multi-extension FITS files.
"""

from __future__ import annotations

import contextlib
import os
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from cuphoton import xdr
from cuphoton.xdr import (
    batch_to_device,
    batch_to_device_stream,
    gpu_available,
    is_gds_active,
    mock_storage,
    storage_cache,
)
from cuphoton.xdr.nvcomp_batch import get_native_batch_builder
from cuphoton.xdr.planning import plan_native_files
from cuphoton.xdr.prefetch import (
    _native_file_plans,
    _planned_file_from_native_tuple,
)

try:
    import cupy as cp
except Exception:  # pragma: no cover - exercised by real environment checks.
    cp = None

try:
    import kvikio
except Exception:  # pragma: no cover - exercised by real environment checks.
    kvikio = None


@contextlib.contextmanager
def nvtx_range(message: str):
    """Best-effort NVTX annotation without making nvtx a hard dependency."""
    try:
        import nvtx
    except Exception:
        yield
        return

    rng = nvtx.push_range(message, color="red")
    try:
        yield
    finally:
        nvtx.pop_range(rng)


@dataclass
class PhaseResult:
    phase: str
    ok: bool
    iterations: int
    elapsed_ms: float
    min_ms: float
    max_ms: float
    data_mb: float
    throughput_mb_s: float
    note: str
    error: str | None = None


@dataclass
class PlannedFileSummary:
    path: Path
    file_index: int
    hdu_indices: tuple[int, ...]
    image_hdus: int
    compressed_hdus: int
    raw_bytes: int
    data_mb: float
    spans: int

    def note(self) -> str:
        return (
            f"{len(self.hdu_indices)} HDUs: {self.image_hdus} image, "
            f"{self.compressed_hdus} compressed, {self.spans} spans, "
            f"{fmt_bytes(self.raw_bytes)} raw"
        )


def fmt_bytes(n_bytes: int) -> str:
    value = float(n_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    raise AssertionError("unreachable")


def make_result(
    phase: str,
    times_ms: Sequence[float],
    data_mb: float,
    note: str,
    *,
    ok: bool = True,
    error: str | None = None,
) -> PhaseResult:
    if not times_ms:
        times_ms = [0.0]
    avg = statistics.mean(times_ms)
    throughput = data_mb / max(avg / 1000.0, 1e-12) if data_mb > 0 else 0.0
    return PhaseResult(
        phase=phase,
        ok=ok,
        iterations=len(times_ms),
        elapsed_ms=avg,
        min_ms=min(times_ms),
        max_ms=max(times_ms),
        data_mb=data_mb,
        throughput_mb_s=throughput,
        note=note,
        error=error,
    )


def failed_phase_from_exception(
    phase: str, exc: BaseException, *, note: str
) -> PhaseResult:
    return make_result(
        phase,
        [],
        0.0,
        note,
        ok=False,
        error=f"{type(exc).__name__}: {exc}",
    )


def parse_hdu_indices(value: str) -> tuple[int, ...]:
    try:
        indices = tuple(
            int(item.strip()) for item in value.split(",") if item.strip()
        )
    except ValueError as exc:
        raise ValueError(
            "--hdu-indices must be comma-separated integers"
        ) from exc
    if not indices:
        raise ValueError("--hdu-indices must contain at least one HDU index")
    return indices


def parse_native_batcher(value: str) -> str | bool:
    return {"auto": "auto", "on": True, "off": False}[value]


def discover_fits_paths(
    scan_dir: Path, max_files: int | None = None
) -> list[Path]:
    paths: list[Path] = []
    for pattern in ("*.fits", "*.fit", "*.fits.gz", "*.fits.fz"):
        paths.extend(sorted(scan_dir.glob(pattern)))
    paths = sorted(set(paths))
    return paths[:max_files] if max_files is not None else paths


def repeated_paths(paths: Sequence[Path], n_files: int) -> list[Path]:
    if not paths or n_files <= 0:
        return []
    return [paths[i % len(paths)] for i in range(n_files)]


def plan_paths(
    paths: Sequence[Path],
    hdu_indices: Sequence[int],
    native_plan_threads: int,
    section=None,
):
    native_items = plan_native_files(
        paths,
        range(len(paths)),
        hdu_indices,
        native_plan_threads,
        section,
    )
    return [_planned_file_from_native_tuple(item) for item in native_items]


def summarize_planned_file(
    path: Path, planned, hdu_indices: Sequence[int]
) -> PlannedFileSummary:
    image_hdus = 0
    compressed_hdus = 0
    data_bytes = 0
    for hdu in planned.hdus:
        if hdu.kind == "image":
            image_hdus += 1
            itemsize = int(hdu.image_itemsize)
            n_items = 1
            for dim in hdu.image_shape:
                n_items *= int(dim)
            data_bytes += n_items * itemsize
        elif hdu.kind == "comp":
            compressed_hdus += 1
            data_bytes += int(sum(int(v) for v in hdu.plan["out_bytes"]))

    return PlannedFileSummary(
        path=path,
        file_index=int(planned.file_index),
        hdu_indices=tuple(int(i) for i in hdu_indices),
        image_hdus=image_hdus,
        compressed_hdus=compressed_hdus,
        raw_bytes=int(planned.total_bytes),
        data_mb=data_bytes / 1024**2,
        spans=len(planned.spans),
    )


def bench_native_plan(
    paths: Sequence[Path],
    hdu_indices: Sequence[int],
    iterations: int,
    native_plan_threads: int,
) -> tuple[list, list[PlannedFileSummary], PhaseResult]:
    times: list[float] = []
    planned = []
    summaries: list[PlannedFileSummary] = []
    with nvtx_range("xdr.plan_native"):
        try:
            for _ in range(iterations):
                t0 = time.perf_counter()
                planned = plan_paths(paths, hdu_indices, native_plan_threads)
                times.append((time.perf_counter() - t0) * 1000.0)
            summaries = [
                summarize_planned_file(path, item, hdu_indices)
                for path, item in zip(paths, planned)
            ]
        except Exception as exc:
            return (
                [],
                [],
                failed_phase_from_exception(
                    "plan_native",
                    exc,
                    note=(
                        f"{len(paths)} files x {len(tuple(hdu_indices))} HDUs"
                    ),
                ),
            )

    data_mb = sum(summary.raw_bytes for summary in summaries) / 1024**2
    note = (
        f"{len(paths)} files, "
        f"{sum(s.image_hdus for s in summaries)} image HDUs, "
        f"{sum(s.compressed_hdus for s in summaries)} compressed HDUs, "
        f"{sum(s.spans for s in summaries)} spans"
    )
    return (
        planned,
        summaries,
        make_result("plan_native", times, data_mb, note),
    )


def bench_gds_read(
    planned_files: Sequence,
    iterations: int,
    *,
    decode_batch_files: int,
    batch_queue_depth: int,
    native_read_threads: int,
) -> PhaseResult:
    if not planned_files:
        return make_result("gds_read", [0.0], 0.0, "no planned files")
    if storage_cache.active():
        return make_result(
            "gds_read",
            [0.0],
            0.0,
            "NativeBatchBuilder raw read requires real storage",
            ok=False,
            error="mock storage is active",
        )

    total_bytes = sum(int(item.total_bytes) for item in planned_files)
    groups = [
        list(planned_files[i : i + int(decode_batch_files)])
        for i in range(0, len(planned_files), int(decode_batch_files))
    ]

    def run_once() -> dict:
        NativeBatchBuilder = get_native_batch_builder(required=True)
        device_id = int(cp.cuda.Device().id)
        builder = NativeBatchBuilder(
            int(decode_batch_files),
            int(batch_queue_depth),
            int(native_read_threads),
            device_id,
        )
        try:
            ready_batches = 0
            ready_files = 0
            device_bytes = 0
            max_outstanding_batches = int(batch_queue_depth) + int(
                native_read_threads
            )
            in_flight_batches = 0

            def drain_ready_batch() -> bool:
                nonlocal ready_batches, ready_files, device_bytes

                native_batch = builder.next_batch()
                if native_batch is None:
                    return False
                owner, _device_ptr, device_nbytes, native_files = native_batch
                ready_batches += 1
                ready_files += len(native_files)
                device_bytes += int(device_nbytes)
                del owner, native_batch
                return True

            for batch_id, group in enumerate(groups):
                while in_flight_batches >= max_outstanding_batches:
                    if not drain_ready_batch():
                        raise RuntimeError(
                            "native batch builder returned no completed "
                            "batch while the submit queue was full"
                        )
                    in_flight_batches -= 1
                builder.submit_batch(batch_id, _native_file_plans(group))
                in_flight_batches += 1
            builder.close_input()

            while drain_ready_batch():
                in_flight_batches -= 1
            if in_flight_batches != 0:
                raise RuntimeError(
                    "native batch builder finished before all submitted "
                    "batches were returned"
                )
            stats = dict(builder.io_stats())
            stats.update(
                ready_batches=ready_batches,
                ready_files=ready_files,
                device_bytes=device_bytes,
            )
            return stats
        finally:
            builder.request_stop()

    times: list[float] = []
    note = (
        f"{len(planned_files)} files, {fmt_bytes(total_bytes)} raw, "
        f"batch_files={decode_batch_files}, queue={batch_queue_depth}, "
        f"read_threads={native_read_threads}"
    )
    with nvtx_range("xdr.gds_read"):
        try:
            run_once()
            for _ in range(iterations):
                t0 = time.perf_counter()
                stats = run_once()
                times.append((time.perf_counter() - t0) * 1000.0)
                if int(stats.get("device_bytes", total_bytes)) != total_bytes:
                    note = (
                        f"{note}, WARNING "
                        f"device_bytes={stats.get('device_bytes')}"
                    )
        except Exception as exc:
            return failed_phase_from_exception("gds_read", exc, note=note)

    return make_result("gds_read", times, total_bytes / 1024**2, note)


def bench_batch_load(
    paths: Sequence[Path],
    hdu_indices: Sequence[int],
    iterations: int,
    *,
    prefetch_depth: int,
    decode_batch_files: int,
    batch_queue_depth: int,
    native_read_threads: int,
    native_plan_threads: int,
    native_batcher: str | bool,
    use_stream: bool,
    data_mb: float,
) -> PhaseResult:
    fn = batch_to_device_stream if use_stream else batch_to_device
    phase = "batch_to_device_stream" if use_stream else "batch_to_device"
    kwargs = dict(
        hdu_indices=tuple(hdu_indices),
        prefetch_depth=prefetch_depth,
        decode_batch_files=decode_batch_files,
        batch_queue_depth=batch_queue_depth,
        native_read_threads=native_read_threads,
        native_plan_threads=native_plan_threads,
        native_batcher=native_batcher,
    )

    times: list[float] = []
    note = (
        f"{len(paths)} files x {len(tuple(hdu_indices))} HDUs, "
        f"native_batcher={native_batcher}"
    )
    with nvtx_range(f"xdr.{phase}"):
        try:
            fn(paths, **kwargs)
            cp.cuda.Stream.null.synchronize()
            for _ in range(iterations):
                t0 = time.perf_counter()
                fn(paths, **kwargs)
                cp.cuda.Stream.null.synchronize()
                times.append((time.perf_counter() - t0) * 1000.0)
        except Exception as exc:
            return failed_phase_from_exception(phase, exc, note=note)

    return make_result(phase, times, data_mb, note)


def print_results(
    results: Sequence[PhaseResult],
    *,
    env_note: str,
    out: Callable[[str], None] = print,
) -> None:
    out("")
    out("=== xDataReader FITS GPU Benchmark ===")
    out(f"Env: {env_note}")
    out("")
    out(
        f"{'Phase':<24} {'iter':>4}  {'avg_ms':>8}  "
        f"{'min_ms':>8}  {'max_ms':>8}  {'MB/s':>8}  note"
    )
    out(
        f"{'-' * 24}  {'-' * 4}  {'-' * 8}  {'-' * 8}  "
        f"{'-' * 8}  {'-' * 8}  {'-' * 40}"
    )
    for result in results:
        mbps = (
            f"{result.throughput_mb_s:>8.1f}"
            if result.data_mb > 0
            else f"{'--':>8}"
        )
        status = "" if result.ok else f"  [FAILED: {result.error}]"
        out(
            f"{result.phase:<24}  {result.iterations:>4}  "
            f"{result.elapsed_ms:>8.1f}  {result.min_ms:>8.1f}  "
            f"{result.max_ms:>8.1f}  {mbps}  {result.note}{status}"
        )


def resolve_paths(
    fits_files: Sequence[str | Path],
    *,
    scan_dir: Path | None,
    max_files: int | None,
) -> list[Path]:
    if scan_dir is not None and fits_files:
        raise ValueError(
            "use either positional FITS files or --dir, not both"
        )
    if scan_dir is not None:
        if not scan_dir.is_dir():
            raise ValueError(f"--dir is not a directory: {scan_dir}")
        paths = discover_fits_paths(scan_dir, max_files)
    else:
        paths = [Path(item) for item in fits_files]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing FITS file: {missing[0]}")
    if not paths:
        raise ValueError("provide FITS files or --dir")
    return paths


def run_benchmark(
    fits_files: Sequence[str | Path],
    *,
    scan_dir: Path | None = None,
    max_files: int | None = None,
    hdu_indices: Sequence[int] = (1,),
    iterations: int = 5,
    prefetch_depth: int = 2,
    decode_batch_files: int = 1,
    batch_queue_depth: int = 2,
    native_read_threads: int = 4,
    native_plan_threads: int = max(1, os.cpu_count() or 1),
    native_batcher: str = "auto",
    mock_storage_kind: str | None = None,
    skip_gds_read: bool = False,
    out: Callable[[str], None] = print,
) -> list[PhaseResult]:
    """Run the FITS benchmark with already validated command values."""

    paths = resolve_paths(
        fits_files,
        scan_dir=scan_dir,
        max_files=max_files,
    )
    resolved_native_batcher = parse_native_batcher(native_batcher)

    if not gpu_available():
        raise RuntimeError("GPU / kvikio / nvcomp / cupy are not available")
    if cp is None:
        raise RuntimeError("cupy is not importable")

    storage_note = (
        f"mock-storage={mock_storage_kind}"
        if mock_storage_kind
        else "storage=real"
    )
    kvikio_version = (
        getattr(kvikio, "__version__", "<unavailable>")
        if kvikio is not None
        else "<unavailable>"
    )
    package_version = getattr(xdr, "__version__", "<local>")
    env_note = (
        f"xdr={package_version}, "
        f"cupy={cp.__version__}, kvikio={kvikio_version}, "
        f"GDS={'ACTIVE' if is_gds_active() else 'COMPAT-FALLBACK'}, "
        f"{storage_note}"
    )

    ctx = (
        mock_storage(mock_storage_kind)
        if mock_storage_kind
        else contextlib.nullcontext()
    )
    with ctx:
        if mock_storage_kind:
            for path in paths:
                storage_cache.preload(path)

        planned, summaries, plan_result = bench_native_plan(
            paths,
            hdu_indices,
            iterations,
            native_plan_threads,
        )
        total_data_mb = sum(summary.data_mb for summary in summaries)
        results: list[PhaseResult] = [plan_result]

        if planned and not skip_gds_read:
            results.append(
                bench_gds_read(
                    planned,
                    iterations,
                    decode_batch_files=decode_batch_files,
                    batch_queue_depth=batch_queue_depth,
                    native_read_threads=native_read_threads,
                )
            )

        results.append(
            bench_batch_load(
                paths,
                hdu_indices,
                iterations,
                prefetch_depth=prefetch_depth,
                decode_batch_files=decode_batch_files,
                batch_queue_depth=batch_queue_depth,
                native_read_threads=native_read_threads,
                native_plan_threads=native_plan_threads,
                native_batcher=resolved_native_batcher,
                use_stream=False,
                data_mb=total_data_mb,
            )
        )
        results.append(
            bench_batch_load(
                paths,
                hdu_indices,
                iterations,
                prefetch_depth=prefetch_depth,
                decode_batch_files=decode_batch_files,
                batch_queue_depth=batch_queue_depth,
                native_read_threads=native_read_threads,
                native_plan_threads=native_plan_threads,
                native_batcher=resolved_native_batcher,
                use_stream=True,
                data_mb=total_data_mb,
            )
        )

    print_results(results, env_note=env_note, out=out)
    return results
