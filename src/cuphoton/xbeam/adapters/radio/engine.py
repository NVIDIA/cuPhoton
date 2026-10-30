# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Eager preparation, resource orchestration, and correlation iteration."""

from __future__ import annotations

import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Event
from types import MappingProxyType
from typing import Any

import numpy as np

from cuphoton.xbeam.adapters.radio.compute_plan import (
    _automatic_subintegration_tile_limit,
    _required_sample_intervals,
    _sample_rate,
    _timing_groups,
)
from cuphoton.xbeam.adapters.radio.gpu_execution import (
    _build_visibility_layout,
    _run_chunked_gpu,
    _run_subintegration_tiled_gpu,
)
from cuphoton.xbeam.adapters.radio.models import (
    AutocorrelationMode,
    Datastream,
    Job,
    RecordingSource,
    SamplingKind,
    Scan,
    Sideband,
    VisibilityRecord,
    VoltageBatch,
)
from cuphoton.xbeam.adapters.radio.output_records import (
    _flush_gpu_records,
    _flush_pipeline_records,
    _gpu_output_capacity,
    _gpu_output_group_limit,
    _new_gpu_output_arena,
)
from cuphoton.xbeam.adapters.radio.reference_execution import _run_reference
from cuphoton.xbeam.adapters.vdif.ingest import (
    SampleInterval,
    VdifIngestPlan,
    VdifPlanningCache,
    plan_vdif_ingest,
)
from cuphoton.xbeam.adapters.vdif.recording import (
    VdifFileRecordingSource,
    VdifPayloadPlan,
    VdifSourceRange,
    materialize_vdif_payload_plans,
)
from cuphoton.xbeam.errors import (
    ConfigurationError,
)
from cuphoton.xbeam.gpu_output import (
    OutputPipeline,
)
from cuphoton.xbeam.ingest_scheduler import (
    _deduplicate_payload_plan_groups,
    _fail_frontier_tasks,
    _frontier_staging_plan,
    _FrontierUploadQueue,
    _mapped_frontier_upload_tasks,
    _payload_upload_staging_plan,
    _PayloadProgress,
    _PayloadProgressNotifier,
    _reusable_vdif_upload_staging_bytes,
    _reusable_vdif_upload_staging_covers,
    _run_cufile_frontier_upload_worker,
    _run_frontier_upload_worker,
    _upload_vdif_plan_group,
)
from cuphoton.xbeam.kernel_compilation import (
    warm_gpu_pipeline,
)
from cuphoton.xbeam.missing_timestamp_recovery import (
    GlobalFFTEligibilityPlan,
    RecoveryPlanKey,
)
from cuphoton.xbeam.profiles import (
    get_profile,
)
from cuphoton.xbeam.profiling import (
    host_timing_enabled,
    nvtx_range,
)
from cuphoton.xbeam.runtime.backend import get_backend
from cuphoton.xbeam.runtime.policy import (
    ExecutionPolicy,
    current_policy,
    policy_records,
    resolve_execution_policy,
    submit_with_policy,
)
from cuphoton.xbeam.runtime.uploads import (
    MAX_UPLOAD_WORKERS,
    resolve_upload_workers,
)
from cuphoton.xbeam.runtime_tuning import (
    cufile_high_priority,
    cufile_strip_stream,
    vdif_upload_staging_slots,
    xmac_packs_per_launch,
)
from cuphoton.xbeam.runtime_types import (
    _BandSamples,
)
from cuphoton.xbeam.sample_loading import (
    _load_band_samples,
)
from cuphoton.xbeam.upload_resources import (
    CuFileReadStatistics,
    _cufile_upload_state_requirements,
    _retain_exact_cufile_workspaces,
    clear_cached_cufile_workspaces,
    clear_cached_vdif_upload_workspaces,
    reserve_vdif_ingest,
    reusable_vdif_upload_staging_bytes,
    vdif_upload_staging_bytes,
)
from cuphoton.xbeam.validation import (
    validate_job,
)
from cuphoton.xbeam.windowing import (
    CorrelationWindow,
    select_complete_timing_groups,
)

_CUFILE_FRONTIER_ENV = "CUPHOTON_XBEAM_CUFILE_FRONTIER"


_GPU_WARMUP_EXECUTOR = ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="cuphoton-xbeam-gpu-warmup"
)


def _available_gpu_working_bytes(xp: Any, memory_pool: Any) -> int:
    """Budget reusable allocations while retaining device memory headroom."""

    free_bytes, total_bytes = xp.cuda.runtime.memGetInfo()
    # Contexts, FFT workspaces, and non-pool allocations also need memory.
    reserve_bytes = int(total_bytes) // 10
    available = max(
        0, int(free_bytes) + int(memory_pool.free_bytes()) - reserve_bytes
    )
    pool_limit = int(memory_pool.get_limit())
    if pool_limit:
        available = min(
            available, max(0, pool_limit - int(memory_pool.used_bytes()))
        )
    return available


@contextmanager
def _gpu_memory_headroom(xp: Any):
    """Bound pool growth before allocation can exhaust a GPU memory node."""

    pool = xp.get_default_memory_pool()
    previous_limit = int(pool.get_limit())
    _free_bytes, total_bytes = xp.cuda.runtime.memGetInfo()
    limit = int(total_bytes) * 9 // 10
    if previous_limit:
        limit = min(limit, previous_limit)
    cache = xp.fft.config.get_plan_cache()
    previous_cache_limit = int(cache.get_memsize())
    cache_limit = int(total_bytes) // 10
    if previous_cache_limit >= 0:
        cache_limit = min(cache_limit, previous_cache_limit)
    pool.set_limit(size=limit)
    try:
        # Retained cuFFT workspaces otherwise crowd out subsequent tiles.
        cache.set_memsize(cache_limit)
        if int(pool.total_bytes()) > limit:
            pool.free_all_blocks()
        yield
    finally:
        cache.set_memsize(previous_cache_limit)
        pool.set_limit(size=previous_limit)


def _cufile_frontier_enabled(io_backend: str) -> bool:
    return (
        current_policy(io_backend).cufile_frontier and io_backend == "cufile"
    )


def _start_gpu_warmup(
    job: Job, xp: Any, *, packed_input: bool
) -> Future[None]:
    """Start job-specific lazy CUDA initialization during host planning."""

    frequencies = {item.identifier: item for item in job.frequencies}
    complex_inputs = tuple(
        sorted(
            {
                item.sampling is not SamplingKind.REAL
                for item in job.datastreams
            }
        )
    )
    mappings: set[str] = set()
    fft_lengths: set[int] = set()
    for datastream in job.datastreams:
        for band in datastream.recorded_bands:
            frequency = frequencies[band.frequency_id]
            if datastream.sampling is SamplingKind.REAL:
                fft_lengths.add(frequency.channel_count * 2)
                mappings.add(
                    "identity"
                    if frequency.sideband is Sideband.USB
                    else "real_lsb"
                )
            elif datastream.sampling is SamplingKind.COMPLEX:
                fft_lengths.add(frequency.channel_count)
                mappings.add(
                    "identity"
                    if frequency.sideband is Sideband.USB
                    else "complex_lsb"
                )
            else:
                fft_lengths.add(frequency.channel_count)
                mappings.add("complex_dsb")
    device_identifier = int(xp.cuda.Device().id)
    return submit_with_policy(
        _GPU_WARMUP_EXECUTOR,
        warm_gpu_pipeline,
        device_identifier,
        complex_inputs,
        packed_input,
        tuple(sorted(mappings)),
        tuple(sorted(fft_lengths)),
        packed_input,
    )


def _output_pipeline_requested():
    return current_policy().output_pipeline


def _wait_gpu_warmup(future: Future[None] | None) -> None:
    if future is not None:
        with nvtx_range("xbeam:gpu-warmup-wait"):
            future.result()


@contextmanager
def _host_batch_preparation(gpu_warmup: Future[None] | None):
    try:
        with nvtx_range("xbeam:host-batch-plan"):
            yield
    except BaseException as host_error:
        # The planner's executor has drained before reaching this handler.
        # Keep the reservation until warmup also finishes, even on failure.
        try:
            _wait_gpu_warmup(gpu_warmup)
        except BaseException as warmup_error:
            raise host_error from warmup_error
        raise
    else:
        _wait_gpu_warmup(gpu_warmup)


def _merge_source_intervals(
    intervals: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    merged: list[tuple[int, int]] = []
    for start, stop in sorted(set(intervals)):
        if start >= stop:
            raise ValueError("source invalid intervals must be nonempty")
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
        else:
            merged.append((start, stop))
    return tuple(merged)


def _apply_source_invalid_intervals(
    band_data: Mapping[tuple[int, int], Any],
    recovery_plans: Mapping[RecoveryPlanKey, GlobalFFTEligibilityPlan] | None,
) -> None:
    """Mask exact source frames consumed by file-backed recovery."""

    if recovery_plans is None:
        return
    intervals_by_band: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for (_scan, datastream_id, band_id), plan in recovery_plans.items():
        intervals_by_band.setdefault((datastream_id, band_id), []).extend(
            plan.source_invalid_intervals
        )

    shared_masks: dict[int, tuple[tuple[int, int], ...]] = {}
    for key, data in band_data.items():
        intervals = _merge_source_intervals(intervals_by_band.get(key, []))
        if not intervals:
            continue
        if data.payload_plan is not None:
            if data.payload_plan.streaming is not None:
                raise RuntimeError(
                    "source recovery masks require static VDIF payload plans"
                )
            shared = shared_masks.setdefault(id(data.payload_plan), intervals)
            if shared != intervals:
                raise RuntimeError(
                    "bands sharing a VDIF payload plan require identical "
                    "source recovery masks"
                )

        source_count = (
            data.source_sample_count
            if data.source_sample_count is not None
            else data.validity.size
        )
        source_stop = data.start_sample + source_count
        for start, stop in intervals:
            clipped_start = max(start, data.start_sample)
            clipped_stop = min(stop, source_stop)
            if clipped_start >= clipped_stop:
                continue
            local_start = clipped_start - data.start_sample
            local_stop = clipped_stop - data.start_sample
            if data.payload_frame_validity is not None:
                frame_samples = data.packed_samples_per_frame
                if (
                    frame_samples <= 0
                    or local_start % frame_samples
                    or local_stop % frame_samples
                ):
                    raise RuntimeError(
                        "source recovery masks must align to packed frames"
                    )
                first = local_start // frame_samples
                last = local_stop // frame_samples
                data.payload_frame_validity[first:last] = False
                if data.device_validity is not None:
                    data.device_validity[first:last] = 0
                if data.validity.size:
                    data.validity[local_start:local_stop] = False
            else:
                data.validity[local_start:local_stop] = False
                if data.device_validity is not None:
                    data.device_validity[local_start:local_stop] = 0


def _frame_presentation_plans(
    recovery_plans: Mapping[RecoveryPlanKey, GlobalFFTEligibilityPlan],
) -> dict[int, tuple[tuple[int, int, int], ...]]:
    """Return one scan-invariant physical presentation per datastream."""

    result: dict[int, tuple[tuple[int, int, int], ...]] = {}
    for (_scan_id, datastream_id, _band_id), plan in recovery_plans.items():
        if not plan.source_frame_intervals:
            continue
        previous = result.setdefault(
            datastream_id, plan.source_frame_intervals
        )
        if previous != plan.source_frame_intervals:
            raise RuntimeError(
                "bands and scans sharing a datastream require one physical "
                "frame presentation"
            )
    return result


def _physical_frame_presentation_plans(
    recovery_plans: Mapping[RecoveryPlanKey, GlobalFFTEligibilityPlan],
) -> dict[int, tuple[tuple[int, int, int, int], ...]]:
    """Return one scan-invariant file-order presentation per datastream."""

    result: dict[int, tuple[tuple[int, int, int, int], ...]] = {}
    for (_scan_id, datastream_id, _band_id), plan in recovery_plans.items():
        if not plan.source_physical_frame_intervals:
            continue
        previous = result.setdefault(
            datastream_id, plan.source_physical_frame_intervals
        )
        if previous != plan.source_physical_frame_intervals:
            raise RuntimeError(
                "bands and scans sharing a datastream require one physical "
                "file-order presentation"
            )
    return result


def _physical_frame_override_plans(
    recovery_plans: Mapping[RecoveryPlanKey, GlobalFFTEligibilityPlan],
) -> dict[int, tuple[tuple[int, int, int, int], ...]]:
    """Return one scan-invariant sparse physical override per datastream."""

    result: dict[int, tuple[tuple[int, int, int, int], ...]] = {}
    for (_scan_id, datastream_id, _band_id), plan in recovery_plans.items():
        if not plan.source_physical_frame_overrides:
            continue
        previous = result.setdefault(
            datastream_id, plan.source_physical_frame_overrides
        )
        if previous != plan.source_physical_frame_overrides:
            raise RuntimeError(
                "bands and scans sharing a datastream require one physical "
                "frame override"
            )
    return result


def _extend_physical_presentation_ranges(
    job: Job,
    ingest_plan: VdifIngestPlan,
    recovery_plans: Mapping[RecoveryPlanKey, GlobalFFTEligibilityPlan] | None,
    *,
    max_read_bytes: int,
) -> VdifIngestPlan:
    """Ensure calibrated physical frame presentations are readable."""

    if (
        not isinstance(ingest_plan, VdifIngestPlan)
        or not recovery_plans
        or ingest_plan.source_ranges is None
    ):
        return ingest_plan
    requested: dict[int, dict[int, list[tuple[int, int]]]] = {}
    for (_scan_id, datastream_id, _band_id), plan in recovery_plans.items():
        thread_windows = (
            None
            if ingest_plan.thread_windows is None
            else ingest_plan.thread_windows.get(datastream_id)
        )
        if not thread_windows:
            raise ConfigurationError(
                f"datastream[{datastream_id}]",
                "source_files",
                "physical presentation lacks a bounded thread window",
            )
        destination_windows = tuple(
            sorted(
                (window.start_slot, window.stop_slot)
                for window in thread_windows.values()
            )
        )
        for (
            source_index,
            start,
            stop,
            destination,
        ) in plan.source_physical_frame_intervals:
            by_source = requested.setdefault(datastream_id, {})
            destination_stop = destination + stop - start
            for window_start, window_stop in destination_windows:
                clipped_start = max(destination, window_start)
                clipped_stop = min(destination_stop, window_stop)
                if clipped_start >= clipped_stop:
                    continue
                source_start = start + clipped_start - destination
                source_stop = start + clipped_stop - destination
                by_source.setdefault(source_index, []).append(
                    (source_start, source_stop)
                )
    if not requested:
        return ingest_plan

    datastreams = {item.identifier: item for item in job.datastreams}
    updated = dict(ingest_plan.source_ranges)
    for datastream_id, bounds_by_source in requested.items():
        datastream = datastreams[datastream_id]
        # A physical presentation is the complete file-order source for this
        # datastream. Retaining the timestamp-selected ranges would read the
        # same logical window twice and can double materialization at a gap.
        ranges = []
        for source_index, requested_bounds in sorted(
            bounds_by_source.items()
        ):
            if source_index >= len(datastream.source_files):
                raise ConfigurationError(
                    f"datastream[{datastream_id}]",
                    "source_files",
                    "physical presentation refers to an unavailable source",
                )
            path = Path(datastream.source_files[source_index]).expanduser()
            path = path.resolve(strict=True)
            stat = path.stat()
            file_frames, remainder = divmod(
                stat.st_size, datastream.frame_size_bytes
            )
            if remainder:
                raise ConfigurationError(
                    f"datastream[{datastream_id}]",
                    "source_files",
                    "physical presentation source has a partial frame",
                )
            for required_start, required_stop in sorted(requested_bounds):
                if required_stop > file_frames:
                    raise ConfigurationError(
                        f"datastream[{datastream_id}]",
                        "source_files",
                        "physical presentation exceeds the source file",
                    )
                ranges.append(
                    VdifSourceRange(
                        path,
                        required_start,
                        required_stop - required_start,
                        stat.st_size,
                        stat.st_mtime_ns,
                        stat.st_dev,
                        stat.st_ino,
                    )
                )

        merged: list[VdifSourceRange] = []
        for item in sorted(
            ranges, key=lambda value: (value.path, value.first_frame)
        ):
            if (
                merged
                and merged[-1].path == item.path
                and item.first_frame <= merged[-1].stop_frame
            ):
                previous = merged[-1]
                merged[-1] = VdifSourceRange(
                    previous.path,
                    previous.first_frame,
                    max(previous.stop_frame, item.stop_frame)
                    - previous.first_frame,
                    previous.expected_size_bytes,
                    previous.expected_mtime_ns,
                    previous.expected_device,
                    previous.expected_inode,
                )
            else:
                merged.append(item)
        updated[datastream_id] = tuple(merged)

    old_bytes = 0
    new_bytes = 0
    for datastream_id, source_ranges in ingest_plan.source_ranges.items():
        frame_size = datastreams[datastream_id].frame_size_bytes
        old_bytes += (
            sum(item.frame_count for item in source_ranges) * frame_size
        )
    for datastream_id, source_ranges in updated.items():
        frame_size = datastreams[datastream_id].frame_size_bytes
        new_bytes += (
            sum(item.frame_count for item in source_ranges) * frame_size
        )
    selected_source_bytes = (
        ingest_plan.selected_source_bytes - old_bytes + new_bytes
    )
    if selected_source_bytes > max_read_bytes:
        raise ConfigurationError(
            "correlate",
            "max_read_bytes",
            "physical presentation source extents require "
            f"{selected_source_bytes} bytes; limit is {max_read_bytes}",
        )
    manifest = dict(ingest_plan.manifest)
    manifest["physical_presentation_source_bytes"] = new_bytes
    return replace(
        ingest_plan,
        source_ranges=updated,
        selected_source_bytes=selected_source_bytes,
        manifest=manifest,
    )


def _execute_correlation(
    job: Job,
    *,
    xp: Any,
    source: RecordingSource,
    timing_groups: list[tuple[int, Scan, Sequence[int]]],
    batch_blocks: int | None,
    vdif_io: str,
    autocorrelation_mode: AutocorrelationMode,
    subintegration_tiling: str,
    max_tile_subintegrations: int | None,
    tiling_manifest: dict[str, Any],
    upload_workers: int | None = None,
    ingest_manifest: dict[str, Any] | None = None,
    reservation_token: object | None = None,
    ingest_replay: dict[str, Any] | None = None,
    gpu_warmup: Future[None] | None = None,
    planned_sample_coverage: bool = True,
    recovery_plans: Mapping[RecoveryPlanKey, GlobalFFTEligibilityPlan]
    | None = None,
) -> Iterator[VisibilityRecord]:
    frequencies = {item.identifier: item for item in job.frequencies}
    datastreams = {item.identifier: item for item in job.datastreams}
    stations = {item.identifier: item for item in job.stations}

    def collect_source_batches(
        datastream: Datastream,
    ) -> dict[int, list[VoltageBatch]]:
        result: dict[int, list[VoltageBatch]] = {}
        recorded_band_ids = {
            item.identifier for item in datastream.recorded_bands
        }
        with nvtx_range(
            f"xbeam:host-ingest-plan:datastream={datastream.identifier}"
        ):
            for batch in source.batches(datastream):
                if batch.band_id not in recorded_band_ids:
                    raise ConfigurationError(
                        f"datastream[{datastream.identifier}]",
                        "recorded_bands",
                        "recording source yielded unknown band "
                        f"{batch.band_id}",
                    )
                result.setdefault(batch.band_id, []).append(batch)
        return result

    # Only the built-in file source guarantees CPU-only batch preparation.
    if type(source) is not VdifFileRecordingSource or xp.__name__ != "cupy":
        _wait_gpu_warmup(gpu_warmup)
        gpu_warmup = None
    with _host_batch_preparation(gpu_warmup):
        source_batches_by_datastream: dict[
            int, dict[int, list[VoltageBatch]]
        ] = {}
        parallel_file_planning = (
            isinstance(source, VdifFileRecordingSource)
            and len(job.datastreams) > 1
        )
        if parallel_file_planning:
            assert isinstance(source, VdifFileRecordingSource)
            representatives: dict[tuple[Any, ...], Datastream] = {}
            for datastream in job.datastreams:
                representatives.setdefault(
                    source.batch_cache_key(datastream), datastream
                )
            worker_count = min(len(representatives), MAX_UPLOAD_WORKERS)
            with ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="cuphoton-xbeam-vdif-plan",
            ) as plan_executor:
                plan_futures = {
                    datastream.identifier: submit_with_policy(
                        plan_executor, collect_source_batches, datastream
                    )
                    for datastream in representatives.values()
                }
                for datastream in job.datastreams:
                    future = plan_futures.get(datastream.identifier)
                    source_batches_by_datastream[datastream.identifier] = (
                        future.result()
                        if future is not None
                        else collect_source_batches(datastream)
                    )
        else:
            source_batches_by_datastream = {
                datastream.identifier: collect_source_batches(datastream)
                for datastream in job.datastreams
            }

    if ingest_manifest is not None and xp.__name__ == "cupy":
        ingest_manifest.setdefault("output_pipeline", {})["reason"] = (
            "requires-file-backed-ingest"
            if current_policy().output_pipeline
            else "disabled"
        )
        ingest_manifest.setdefault("vdif_upload", {})["frontier_reason"] = (
            "requires-file-backed-ingest"
        )

    band_data: dict[tuple[int, int], _BandSamples] = {}
    device_cache: dict[tuple[str, int], tuple[Any, Any]] = {}
    file_plan_groups: list[list[VdifPayloadPlan]] = []
    for datastream in job.datastreams:
        source_batches = source_batches_by_datastream[datastream.identifier]
        file_plans: list[VdifPayloadPlan] = []
        seen_plans: set[int] = set()
        for batches in source_batches.values():
            for batch in batches:
                if (
                    isinstance(batch.samples, VdifPayloadPlan)
                    and id(batch.samples) not in seen_plans
                ):
                    seen_plans.add(id(batch.samples))
                    file_plans.append(batch.samples)
        if file_plans:
            if xp.__name__ == "cupy":
                file_plan_groups.append(file_plans)
            else:
                host_payloads = materialize_vdif_payload_plans(file_plans)
                for plan, payload in zip(
                    file_plans, host_payloads, strict=True
                ):
                    device_cache[("host_samples", id(plan))] = (
                        plan,
                        payload,
                    )
        for band in datastream.recorded_bands:
            band_data[(datastream.identifier, band.identifier)] = (
                _load_band_samples(
                    job,
                    datastream,
                    band,
                    frequencies[band.frequency_id],
                    source_batches.get(band.identifier, []),
                    xp,
                    device_cache,
                    defer_file_payload=xp.__name__ == "cupy",
                    reservation_token=reservation_token,
                    planned_sample_coverage=planned_sample_coverage,
                )
            )
    _apply_source_invalid_intervals(band_data, recovery_plans)
    device_cache.clear()

    if xp.__name__ == "cupy" and file_plan_groups:
        file_plan_groups, payload_aliases = _deduplicate_payload_plan_groups(
            file_plan_groups
        )
        for data in band_data.values():
            if data.payload_plan is not None:
                data.payload_plan = payload_aliases[id(data.payload_plan)]

    payload_outputs: dict[int, Any] = {}
    payload_validity_outputs: dict[int, Any] = {}
    payload_progresses: dict[int, _PayloadProgress] = {}
    upload_progress_notifier = _PayloadProgressNotifier()
    payload_outputs_ready: Any = None
    if xp.__name__ == "cupy":
        for plans in file_plan_groups:
            for plan in plans:
                payload_outputs[id(plan)] = xp.zeros(
                    int(plan.byte_count), dtype=xp.uint8
                )
                static_validity = next(
                    (
                        data.payload_frame_validity
                        for data in band_data.values()
                        if data.payload_plan is plan
                    ),
                    None,
                )
                if plan.streaming is None and static_validity is None:
                    raise RuntimeError(
                        "mapped VDIF payload plan requires frame validity"
                    )
                payload_progresses[id(plan)] = _PayloadProgress(
                    int(plan.slot_count),
                    dynamic_validity=plan.streaming is not None,
                    static_validity=(
                        None
                        if plan.streaming is not None
                        else static_validity
                    ),
                    notifier=upload_progress_notifier,
                )
        for data in band_data.values():
            if data.payload_plan is not None:
                data.device_payload = payload_outputs[id(data.payload_plan)]
                data.payload_progress = payload_progresses[
                    id(data.payload_plan)
                ]
                existing_validity = payload_validity_outputs.setdefault(
                    id(data.payload_plan), data.device_validity
                )
                if existing_validity is not data.device_validity:
                    raise RuntimeError(
                        "bands sharing a VDIF payload plan must share "
                        "device validity"
                    )
        if payload_outputs:
            payload_outputs_ready = xp.cuda.Event()
            payload_outputs_ready.record(xp.cuda.get_current_stream())

    requested_auto_bands: dict[int, dict[int, set[int]]] = {}
    for baseline in job.baselines:
        for product in baseline.products:
            requested_auto_bands.setdefault(
                baseline.datastream_a_id, {}
            ).setdefault(product.output_frequency_id, set()).add(
                product.band_a_id
            )
            requested_auto_bands.setdefault(
                baseline.datastream_b_id, {}
            ).setdefault(product.output_frequency_id, set()).add(
                product.band_b_id
            )

    if xp.__name__ == "cupy":
        visibility_layout = _build_visibility_layout(
            job, band_data, frequencies, datastreams, requested_auto_bands
        )
        records_per_group, spectra_per_group = _gpu_output_capacity(
            job, 1, frequencies, requested_auto_bands
        )
        bytes_per_group = (
            spectra_per_group * np.dtype(np.complex64).itemsize
            + records_per_group * np.dtype(np.float64).itemsize
        )
        device_identifier = int(xp.cuda.Device().id)
        free_bytes, total_bytes = xp.cuda.runtime.memGetInfo()
        memory_pool = xp.get_default_memory_pool()
        uncached_available_bytes = _available_gpu_working_bytes(
            xp, memory_pool
        )
        cached_staging_bytes = _reusable_vdif_upload_staging_bytes(
            vdif_io, device_identifier
        )
        available_bytes = min(
            int(total_bytes),
            uncached_available_bytes + cached_staging_bytes,
        )
        if available_bytes <= 0:
            raise MemoryError(
                "no GPU memory is available for VDIF upload staging; "
                "split the recording or reduce the ingest chunk size"
            )
        (
            upload_worker_count,
            upload_chunk_bytes,
            upload_staging_bytes,
            per_worker_device_bytes,
        ) = _payload_upload_staging_plan(
            job,
            file_plan_groups,
            band_data,
            available_bytes,
            io_backend=vdif_io,
            requested_workers=upload_workers,
        )
        upload_footprints = [
            vdif_upload_staging_bytes(
                plans,
                target_chunk_bytes=chunk_bytes,
                io_backend=vdif_io,
            )
            for plans, chunk_bytes in zip(
                file_plan_groups, upload_chunk_bytes, strict=True
            )
        ]
        reusable_footprints = [
            reusable_vdif_upload_staging_bytes(
                plans,
                target_chunk_bytes=chunk_bytes,
                io_backend=vdif_io,
            )
            for plans, chunk_bytes in zip(
                file_plan_groups, upload_chunk_bytes, strict=True
            )
        ]
        staging_is_cached = bool(cached_staging_bytes) and (
            _reusable_vdif_upload_staging_covers(
                vdif_io,
                device_identifier,
                reusable_footprints,
                upload_worker_count,
            )
        )
        if cached_staging_bytes and not staging_is_cached:
            if vdif_io == "cufile":
                clear_cached_cufile_workspaces(device_identifier)
            else:
                clear_cached_vdif_upload_workspaces(device_identifier)
            free_bytes, total_bytes = xp.cuda.runtime.memGetInfo()
            uncached_available_bytes = _available_gpu_working_bytes(
                xp, memory_pool
            )
            (
                upload_worker_count,
                upload_chunk_bytes,
                upload_staging_bytes,
                per_worker_device_bytes,
            ) = _payload_upload_staging_plan(
                job,
                file_plan_groups,
                band_data,
                uncached_available_bytes,
                io_backend=vdif_io,
                requested_workers=upload_workers,
            )
            upload_footprints = [
                vdif_upload_staging_bytes(
                    plans,
                    target_chunk_bytes=chunk_bytes,
                    io_backend=vdif_io,
                )
                for plans, chunk_bytes in zip(
                    file_plan_groups, upload_chunk_bytes, strict=True
                )
            ]
            reusable_footprints = [
                reusable_vdif_upload_staging_bytes(
                    plans,
                    target_chunk_bytes=chunk_bytes,
                    io_backend=vdif_io,
                )
                for plans, chunk_bytes in zip(
                    file_plan_groups, upload_chunk_bytes, strict=True
                )
            ]
            staging_is_cached = False
        if staging_is_cached:
            transient_footprints = [
                footprint - reusable
                for footprint, reusable in zip(
                    upload_footprints, reusable_footprints, strict=True
                )
            ]
            additional_staging_bytes = sum(
                sorted(transient_footprints, reverse=True)[
                    :upload_worker_count
                ]
            )
        else:
            additional_staging_bytes = upload_staging_bytes
        planning_bytes = uncached_available_bytes - additional_staging_bytes
        if planning_bytes <= 0:
            raise MemoryError(
                "VDIF upload staging exceeds available device memory; "
                "reduce the ingest chunk size"
            )
        per_worker_device_bytes = max(upload_footprints, default=1)
        group_limit = _gpu_output_group_limit(
            len(timing_groups), bytes_per_group, planning_bytes
        )
        if ingest_manifest is not None:
            ingest_manifest["xmac_packs_per_launch"] = xmac_packs_per_launch()
            if vdif_io == "cufile":
                high_priority = cufile_high_priority(vdif_io)
                effective_priority = 0
                if high_priority:
                    priority_stream = cufile_strip_stream(xp)
                    effective_priority = priority_stream.priority
                    del priority_stream
                ingest_manifest["cufile_strip_priority"] = {
                    "high_priority_requested": high_priority,
                    "effective_priority": effective_priority,
                }
            ingest_manifest.setdefault("vdif_upload", {}).update(
                {
                    "effective_workers": upload_worker_count,
                    "staging_slots": vdif_upload_staging_slots(vdif_io),
                    "target_chunk_bytes": upload_chunk_bytes,
                    "staging_bytes": additional_staging_bytes,
                }
            )

        cufile_frontier_requested = _cufile_frontier_enabled(vdif_io)
        frontier_requested = len(file_plan_groups) > upload_worker_count and (
            vdif_io == "posix" or cufile_frontier_requested
        )
        frontier_upload = None
        frontier_diagnostics: dict[str, Any] = {}
        if frontier_requested:
            preparation_range = (
                "xbeam:vdif-cufile-plan-prepare"
                if vdif_io == "cufile"
                else "xbeam:vdif-posix-plan-prepare"
            )
            with nvtx_range(preparation_range):
                frontier_upload = _mapped_frontier_upload_tasks(
                    file_plan_groups,
                    payload_outputs,
                    payload_validity_outputs,
                    payload_progresses,
                    upload_chunk_bytes,
                    len(timing_groups),
                    trust_slices=True,
                    diagnostics=frontier_diagnostics,
                    xp=xp,
                    planner=current_policy().cufile_frontier_planner,
                )
        if frontier_upload is not None:
            if vdif_io == "cufile":
                # Keep exact-size cached registrations, not oversized entries
                # whose actual footprint would exceed the new reservation.
                required_raw_bytes = max(
                    _cufile_upload_state_requirements(
                        task.plans,
                        target_chunk_bytes=task.target_chunk_bytes,
                        full_mappings=True,
                    )[0]
                    for task in frontier_upload[0]
                )
                cached_frontier_bytes = _retain_exact_cufile_workspaces(
                    device_identifier,
                    required_raw_bytes,
                    vdif_upload_staging_slots("cufile"),
                    upload_worker_count,
                )
                free_bytes, total_bytes = xp.cuda.runtime.memGetInfo()
                uncached_available_bytes = _available_gpu_working_bytes(
                    xp, memory_pool
                )
                (
                    upload_worker_count,
                    required_raw_bytes,
                    required_mapping_bytes,
                    frontier_worker_device_bytes,
                    frontier_worker_pinned_bytes,
                ) = _frontier_staging_plan(
                    frontier_upload[0],
                    upload_worker_count,
                    uncached_available_bytes + cached_frontier_bytes,
                )
                # A memory-limited worker count may leave extra cache entries.
                cached_frontier_bytes = _retain_exact_cufile_workspaces(
                    device_identifier,
                    required_raw_bytes,
                    vdif_upload_staging_slots("cufile"),
                    upload_worker_count,
                )
                free_bytes, total_bytes = xp.cuda.runtime.memGetInfo()
                uncached_available_bytes = _available_gpu_working_bytes(
                    xp, memory_pool
                )
                additional_staging_bytes = (
                    upload_worker_count * frontier_worker_device_bytes
                    - cached_frontier_bytes
                )
                planning_bytes = (
                    uncached_available_bytes - additional_staging_bytes
                )
                if ingest_manifest is not None:
                    ingest_manifest["vdif_upload"].update(
                        {
                            "effective_workers": upload_worker_count,
                            "staging_bytes": additional_staging_bytes,
                            "retained_staging_bytes": (
                                upload_worker_count
                                * frontier_worker_device_bytes
                            ),
                            "pinned_mapping_bytes": (
                                upload_worker_count
                                * frontier_worker_pinned_bytes
                            ),
                        }
                    )
            group_limit = 1
        if ingest_replay is not None and (
            vdif_io != "cufile" or frontier_upload is None
        ):
            raise ValueError(
                "ingest replay requires mapped cuFile frontier tasks; "
                "enable the frontier scheduler and use fewer workers "
                "than files"
            )
        if ingest_manifest is not None:
            upload_manifest = ingest_manifest.setdefault("vdif_upload", {})
            upload_manifest["frontier_plan"] = frontier_diagnostics
            upload_manifest["frontier_scheduling"] = (
                frontier_upload is not None
            )
            upload_manifest["frontier_tasks"] = (
                0 if frontier_upload is None else len(frontier_upload[0])
            )
            upload_manifest["cufile_frontier_requested"] = (
                cufile_frontier_requested
            )
            upload_manifest["frontier_reason"] = (
                "enabled"
                if frontier_upload is not None
                else "disabled"
                if vdif_io == "cufile" and not cufile_frontier_requested
                else "enough-workers-for-source-groups"
                if not frontier_requested
                else "ineligible-mappings"
            )
            upload_manifest["cufile_frontier_effective"] = (
                cufile_frontier_requested and frontier_upload is not None
            )

        payload_futures: dict[int, Future[list[Any]]] = {}
        producer_futures: list[Future[Any]] = []
        frontier_task_queue: _FrontierUploadQueue | None = None
        frontier_abort: Event | None = None
        cufile_submission_executor: ThreadPoolExecutor | None = None
        cufile_statistics = (
            CuFileReadStatistics(capture_reads=ingest_replay is not None)
            if vdif_io == "cufile"
            and (host_timing_enabled() or ingest_replay is not None)
            else None
        )
        executor: ThreadPoolExecutor | None = None
        completed = False
        output_pipeline = None
        try:
            if file_plan_groups:
                executor = ThreadPoolExecutor(
                    max_workers=upload_worker_count,
                    thread_name_prefix="cuphoton-xbeam-vdif-upload",
                )
                if frontier_upload is not None:
                    frontier_tasks, payload_futures = frontier_upload
                    worker_count = min(
                        upload_worker_count, len(frontier_tasks)
                    )
                    frontier_task_queue = _FrontierUploadQueue(frontier_tasks)
                    if vdif_io == "cufile":
                        cufile_submission_executor = ThreadPoolExecutor(
                            max_workers=1,
                            thread_name_prefix="cuphoton-xbeam-cuda-submit",
                        )
                        for _worker_index in range(worker_count):
                            producer_futures.append(
                                submit_with_policy(
                                    executor,
                                    _run_cufile_frontier_upload_worker,
                                    frontier_task_queue,
                                    xp,
                                    device_identifier,
                                    payload_outputs_ready,
                                    frontier_worker_device_bytes,
                                    reservation_token,
                                    cufile_statistics,
                                    required_raw_bytes,
                                    required_mapping_bytes,
                                    cufile_submission_executor,
                                    read_only=(
                                        ingest_replay is not None
                                        and ingest_replay["mode"] == "read"
                                    ),
                                )
                            )
                    else:
                        frontier_abort = Event()
                        for _worker_index in range(worker_count):
                            producer_futures.append(
                                submit_with_policy(
                                    executor,
                                    _run_frontier_upload_worker,
                                    frontier_task_queue,
                                    xp,
                                    device_identifier,
                                    payload_outputs_ready,
                                    per_worker_device_bytes,
                                    reservation_token,
                                    frontier_abort,
                                )
                            )
                else:
                    for plans, chunk_bytes in zip(
                        file_plan_groups, upload_chunk_bytes, strict=True
                    ):
                        outputs = [
                            payload_outputs[id(plan)] for plan in plans
                        ]
                        validity_outputs = [
                            payload_validity_outputs[id(plan)]
                            for plan in plans
                        ]
                        progresses = [
                            payload_progresses[id(plan)] for plan in plans
                        ]
                        future = submit_with_policy(
                            executor,
                            _upload_vdif_plan_group,
                            plans,
                            outputs,
                            validity_outputs,
                            xp,
                            device_identifier,
                            payload_outputs_ready,
                            progresses,
                            chunk_bytes,
                            vdif_io,
                            per_worker_device_bytes,
                            reservation_token,
                            cufile_statistics=cufile_statistics,
                        )
                        for plan in plans:
                            payload_futures[id(plan)] = future
            if ingest_replay is not None:
                # The same producer path and cleanup run, but no station,
                # visibility, or output work is submitted in diagnostic mode.
                for future in producer_futures:
                    future.result()
                completed = True
                return
            largest_group = max(
                (len(group[2]) for group in timing_groups), default=1
            )
            if subintegration_tiling == "off":
                tile_limit = largest_group
            else:
                tile_limit = _automatic_subintegration_tile_limit(
                    job,
                    band_data,
                    timing_groups,
                    max(1, planning_bytes - bytes_per_group),
                    max_tile_subintegrations,
                )
            pipeline_manifest = {
                "requested": _output_pipeline_requested(),
                "effective": False,
                "reason": "disabled",
                "device_bytes": 0,
                "pinned_bytes": 0,
                "layout": "tiled"
                if tile_limit < largest_group
                else "untiled",
                "accumulator_bytes": 0,
                "scratch_bytes": 0,
            }
            pipeline_tiled = tile_limit < largest_group
            pipeline_arena_count = 4 if pipeline_tiled else 2
            if ingest_manifest is not None:
                ingest_manifest["output_pipeline"] = pipeline_manifest
            if pipeline_manifest["requested"]:
                if frontier_upload is None:
                    pipeline_manifest["reason"] = "requires-mapped-frontier"
                elif len(timing_groups) <= 1:
                    pipeline_manifest["reason"] = "single-integration"
                elif (
                    planning_bytes <= pipeline_arena_count * bytes_per_group
                    or _automatic_subintegration_tile_limit(
                        job,
                        band_data,
                        timing_groups,
                        planning_bytes
                        - pipeline_arena_count * bytes_per_group,
                        max_tile_subintegrations,
                    )
                    < tile_limit
                ):
                    pipeline_manifest["reason"] = "insufficient-device-budget"
                else:
                    try:
                        output_pipeline = OutputPipeline(
                            xp,
                            lambda: _new_gpu_output_arena(
                                xp, records_per_group, spectra_per_group
                            ),
                            tiled=pipeline_tiled,
                        )
                    except Exception as error:
                        # cudaErrorMemoryAllocation may come from pinned
                        # allocation rather than CuPy's device memory pool.
                        if (
                            not isinstance(error, MemoryError)
                            and getattr(error, "status", None) != 2
                        ):
                            raise
                        pipeline_manifest["reason"] = "allocation-failed"
                    else:
                        # The execution path below accounts for one arena
                        # (untiled) or two (tiled); reserve the other slot.
                        planning_bytes -= bytes_per_group * (
                            2 if pipeline_tiled else 1
                        )
                        pipeline_manifest.update(
                            effective=True,
                            reason="enabled",
                            device_bytes=pipeline_arena_count
                            * bytes_per_group,
                            pinned_bytes=2 * bytes_per_group,
                            accumulator_bytes=2 * bytes_per_group,
                            scratch_bytes=(
                                2 * bytes_per_group if pipeline_tiled else 0
                            ),
                        )
            tiling_manifest.update(
                {
                    "mode": subintegration_tiling,
                    "status": "complete",
                    "requested_max_subintegrations": (
                        max_tile_subintegrations
                    ),
                    "automatic_limit": tile_limit,
                    "effective": tile_limit < largest_group,
                    "logical_integrations": len(timing_groups),
                    "tiles": [],
                    "oom_retries": 0,
                }
            )
            if tile_limit < largest_group:
                for timing_group in timing_groups:
                    accumulator = (
                        output_pipeline.acquire()
                        if output_pipeline is not None
                        else None
                    )
                    recovery_arguments = (
                        {}
                        if recovery_plans is None
                        else {"recovery_plans": recovery_plans}
                    )
                    pending_gpu_records, gpu_output_arena, sizes, retries = (
                        _run_subintegration_tiled_gpu(
                            job,
                            xp,
                            timing_group,
                            band_data,
                            frequencies,
                            datastreams,
                            stations,
                            requested_auto_bands,
                            records_per_group,
                            spectra_per_group,
                            batch_blocks,
                            max(1, planning_bytes - bytes_per_group * 2),
                            autocorrelation_mode,
                            payload_futures,
                            min(tile_limit, len(timing_group[2])),
                            accumulator=accumulator,
                            visibility_layout=visibility_layout,
                            scratch=(
                                output_pipeline.scratch_for(accumulator)
                                if output_pipeline is not None
                                else None
                            ),
                            **recovery_arguments,
                        )
                    )
                    tiling_manifest["tiles"].append(
                        {
                            "integration_index": timing_group[0],
                            "subintegrations": sizes,
                        }
                    )
                    tiling_manifest["oom_retries"] += retries
                    if output_pipeline is None:
                        yield from _flush_gpu_records(
                            job, pending_gpu_records, gpu_output_arena
                        )
                    else:
                        output_pipeline.enqueue(
                            gpu_output_arena, pending_gpu_records
                        )
                        if len(output_pipeline.queue) == 2:
                            yield from _flush_pipeline_records(
                                job, output_pipeline
                            )
                    del pending_gpu_records, gpu_output_arena
                if output_pipeline is not None:
                    while output_pipeline.queue:
                        yield from _flush_pipeline_records(
                            job, output_pipeline
                        )
                completed = True
                return

            cursor = 0
            while cursor < len(timing_groups):
                group_count = min(group_limit, len(timing_groups) - cursor)
                while True:
                    try:
                        gpu_output_arena = (
                            output_pipeline.acquire()
                            if output_pipeline is not None
                            else _new_gpu_output_arena(
                                xp,
                                records_per_group * group_count,
                                spectra_per_group * group_count,
                            )
                        )
                        break
                    except xp.cuda.memory.OutOfMemoryError:
                        if group_count == 1:
                            raise
                        group_count = max(1, group_count // 2)
                        group_limit = min(group_limit, group_count)
                    memory_pool.free_all_blocks()

                recovery_arguments = (
                    {}
                    if recovery_plans is None
                    else {"recovery_plans": recovery_plans}
                )
                pending_gpu_records = _run_chunked_gpu(
                    job,
                    xp,
                    timing_groups[cursor : cursor + group_count],
                    band_data,
                    frequencies,
                    datastreams,
                    stations,
                    requested_auto_bands,
                    gpu_output_arena,
                    batch_blocks,
                    max(
                        1,
                        planning_bytes - bytes_per_group * group_count,
                    ),
                    autocorrelation_mode,
                    payload_futures,
                    visibility_layout=visibility_layout,
                    **recovery_arguments,
                )
                if output_pipeline is None:
                    yield from _flush_gpu_records(
                        job, pending_gpu_records, gpu_output_arena
                    )
                else:
                    output_pipeline.enqueue(
                        gpu_output_arena, pending_gpu_records
                    )
                    if len(output_pipeline.queue) == 2:
                        yield from _flush_pipeline_records(
                            job, output_pipeline
                        )
                cursor += group_count
                del pending_gpu_records, gpu_output_arena
            if output_pipeline is not None:
                while output_pipeline.queue:
                    yield from _flush_pipeline_records(job, output_pipeline)
            completed = True
        finally:
            pipeline_error = None
            if output_pipeline is not None:
                try:
                    output_pipeline.close(best_effort=not completed)
                except BaseException as error:
                    pipeline_error = error
            if executor is not None:
                if not completed and frontier_task_queue is not None:
                    cancellation = RuntimeError(
                        "correlation stopped before VDIF upload completed"
                    )
                    _fail_frontier_tasks(
                        frontier_task_queue.abort(cancellation), cancellation
                    )
                if not completed and frontier_abort is not None:
                    # POSIX frontier workers hold static task lists; stop them
                    # between tasks rather than draining the whole recording.
                    frontier_abort.set()
                executor.shutdown(wait=True, cancel_futures=not completed)
            if cufile_submission_executor is not None:
                cufile_submission_executor.shutdown(
                    wait=True,
                    cancel_futures=not completed,
                )
            if executor is not None and completed:
                futures = [
                    *dict.fromkeys(payload_futures.values()),
                    *producer_futures,
                ]
                for future in futures:
                    future.result()
            if pipeline_error is not None:
                raise pipeline_error
            if cufile_statistics is not None:
                statistics = cufile_statistics.snapshot()
                if ingest_replay is not None:
                    ingest_replay.update(
                        {
                            "statistics": statistics,
                            "read_descriptors": (
                                cufile_statistics.read_descriptors
                            ),
                            "finished_ns": time.perf_counter_ns(),
                            "upload": (ingest_manifest or {}).get(
                                "vdif_upload", {}
                            ),
                        }
                    )
                print(
                    "cuphoton-xbeam GDS ingest: "
                    f"logical_bytes={statistics.logical_bytes} "
                    f"requested_bytes={statistics.requested_bytes} "
                    f"calls={statistics.calls} "
                    f"span={statistics.span_seconds:.6f}s "
                    f"active_read={statistics.active_read_seconds:.6f}s "
                    f"concurrency={statistics.mean_read_concurrency:.3f} "
                    f"bandwidth={statistics.logical_gb_per_second:.3f}GB/s",
                    file=sys.stderr,
                    flush=True,
                )
        return
    yield from _run_reference(
        job,
        xp,
        timing_groups,
        band_data,
        frequencies,
        datastreams,
        stations,
        requested_auto_bands,
        batch_blocks,
        autocorrelation_mode,
        recovery_plans,
    )


@dataclass(frozen=True, slots=True)
class CorrelationRun:
    """An eagerly validated correlation ready to produce records."""

    job: Job
    backend: str
    profile: str
    batch_blocks: int | None
    vdif_io: str
    source: RecordingSource
    timing_groups: tuple[tuple[int, Scan, Sequence[int]], ...]
    xp: Any
    autocorrelation_mode: AutocorrelationMode = AutocorrelationMode.AVERAGE
    subintegration_tiling: str = "auto"
    max_tile_subintegrations: int | None = None
    upload_workers: int | None = None
    estimated_read_bytes: int | None = None
    ingest_manifest: dict[str, Any] | None = None
    gpu_warmup: Future[None] | None = None
    execution_policy: ExecutionPolicy | None = None
    planned_sample_coverage: bool = True
    recovery_plans: (
        Mapping[RecoveryPlanKey, GlobalFFTEligibilityPlan] | None
    ) = None

    @property
    def vdif_upload_workers(self) -> int | None:
        """Compatibility accessor for the former radio-specific keyword."""
        return self.upload_workers

    def records(self) -> Iterator[VisibilityRecord]:
        overlap_warmup = (
            type(self.source) is VdifFileRecordingSource
            and getattr(self.xp, "__name__", None) == "cupy"
        )
        if not overlap_warmup:
            _wait_gpu_warmup(self.gpu_warmup)

        def execute(reservation_token: object | None = None):
            tiling_manifest = (
                self.ingest_manifest.setdefault("subintegration_tiling", {})
                if self.ingest_manifest is not None
                else {}
            )
            yield from _execute_correlation(
                self.job,
                xp=self.xp,
                source=self.source,
                timing_groups=list(self.timing_groups),
                batch_blocks=self.batch_blocks,
                vdif_io=self.vdif_io,
                autocorrelation_mode=self.autocorrelation_mode,
                subintegration_tiling=self.subintegration_tiling,
                max_tile_subintegrations=self.max_tile_subintegrations,
                tiling_manifest=tiling_manifest,
                upload_workers=self.upload_workers,
                ingest_manifest=self.ingest_manifest,
                reservation_token=reservation_token,
                gpu_warmup=self.gpu_warmup if overlap_warmup else None,
                planned_sample_coverage=self.planned_sample_coverage,
                recovery_plans=self.recovery_plans,
            )

        if getattr(self.xp, "__name__", None) != "cupy":
            return policy_records(
                self.execution_policy
                or resolve_execution_policy(self.vdif_io),
                execute,
            )
        device_identifier = int(self.xp.cuda.Device().id)

        def execute_reserved():
            with reserve_vdif_ingest(device_identifier) as token:
                with _gpu_memory_headroom(self.xp):
                    yield from execute(token)

        return policy_records(
            self.execution_policy or resolve_execution_policy(self.vdif_io),
            execute_reserved,
        )


def _plan_correlation_vdif_ingest(
    job: Job,
    timing_groups: Sequence[tuple[int, Scan, Sequence[int]]],
    *,
    window: CorrelationWindow | None,
    max_read_bytes: int,
    assume_ordered: bool,
    index_dir: str | None,
    vdif_io: str,
    planning_cache: VdifPlanningCache | None,
    recovery_plans: Mapping[RecoveryPlanKey, GlobalFFTEligibilityPlan]
    | None = None,
) -> VdifIngestPlan:
    with nvtx_range("xbeam:required-sample-intervals"):
        ordinary_intervals = _required_sample_intervals(job, timing_groups)
        source_intervals = _extend_replay_intervals(
            job, timing_groups, ordinary_intervals, recovery_plans
        )
    with nvtx_range("xbeam:vdif-range-plan"):
        source_plan = plan_vdif_ingest(
            job,
            intervals=source_intervals,
            max_read_bytes=max_read_bytes,
            assume_ordered=assume_ordered,
            index_dir=index_dir,
            defer_ordered_headers=False,
            planning_cache=planning_cache,
        )
        if recovery_plans is None or source_intervals == ordinary_intervals:
            return source_plan
        if any(
            (plan.source_shifted_fft_count or plan.has_source_segments)
            and (
                not plan.has_source_frame_presentation
                or bool(plan.source_physical_frame_intervals)
            )
            for plan in recovery_plans.values()
        ):
            # Direct replay offsets address the widened job-relative sample
            # grid at execution time.  Replacing this window with the
            # ordinary presentation window would retain the wider file read
            # while discarding the addressable prefix/suffix it was meant to
            # provide. Timestamp-presentation plans instead keep the compact
            # ordinary window below and place source frames into it. A
            # physical-order presentation still uses execution offsets and
            # therefore retains the widened window.
            return source_plan
        ordinary_plan = plan_vdif_ingest(
            job,
            intervals=ordinary_intervals,
            max_read_bytes=max_read_bytes,
            assume_ordered=assume_ordered,
            index_dir=index_dir,
            defer_ordered_headers=False,
            planning_cache=planning_cache,
        )
        manifest = dict(source_plan.manifest)
        manifest["presentation_window"] = ordinary_plan.manifest
        return replace(
            source_plan,
            thread_windows=ordinary_plan.thread_windows,
            source_thread_windows=source_plan.thread_windows,
            estimated_materialized_bytes=(
                ordinary_plan.estimated_materialized_bytes
            ),
            estimated_decoded_bytes=ordinary_plan.estimated_decoded_bytes,
            manifest=manifest,
        )


def _extend_replay_intervals(
    job: Job,
    timing_groups: Sequence[tuple[int, Scan, Sequence[int]]],
    intervals: dict[tuple[int, int], SampleInterval],
    plans: Mapping[RecoveryPlanKey, GlobalFFTEligibilityPlan] | None,
) -> dict[tuple[int, int], SampleInterval]:
    """Include every explicitly selected source sample in bounded ingest."""

    if not plans:
        return intervals
    subintegrations_by_scan: dict[str, list[int]] = {}
    for _integration, scan, subintegrations in timing_groups:
        subintegrations_by_scan.setdefault(scan.identifier, []).extend(
            int(index) for index in subintegrations
        )
    datastreams = {item.identifier: item for item in job.datastreams}
    frequencies = {item.identifier: item for item in job.frequencies}
    offsets_by_band: dict[tuple[int, int], list[int]] = {}
    for (scan_id, datastream_id, band_id), plan in plans.items():
        selected = subintegrations_by_scan.get(scan_id)
        if not selected:
            continue
        datastream = datastreams[datastream_id]
        band = next(
            item
            for item in datastream.recorded_bands
            if item.identifier == band_id
        )
        frequency = frequencies[band.frequency_id]
        factor = 2 if datastream.sampling is SamplingKind.REAL else 1
        fft_samples = frequency.channel_count * factor
        sample_rate = (
            datastream.sample_rate_hz
            if datastream.sample_rate_hz is not None
            else frequency.bandwidth_hz * factor
        )
        blocks_per_subintegration = round(
            float(job.subintegration_duration_s) * sample_rate / fft_samples
        )
        start_fft = min(selected) * blocks_per_subintegration
        stop_fft = (max(selected) + 1) * blocks_per_subintegration
        minimum, maximum = plan.source_offset_bounds_for(start_fft, stop_fft)
        offsets_by_band.setdefault((datastream_id, band_id), []).extend(
            (minimum, maximum)
        )
    result = dict(intervals)
    for key, interval in intervals.items():
        offsets = offsets_by_band.get(key, ())
        if offsets:
            result[key] = SampleInterval(
                interval.first_sample + min(offsets),
                interval.stop_sample + max(offsets),
            )
    return result


def _build_file_recovery_plans(
    calibration_identifier: str,
    job: Job,
    timing_groups: Sequence[tuple[int, Scan, Sequence[int]]],
    ingest_plan: VdifIngestPlan,
    *,
    index_dir: str | None,
) -> Mapping[RecoveryPlanKey, GlobalFFTEligibilityPlan]:
    from cuphoton.xbeam.buffered_reader import (
        build_returned_window_replay_plans,
    )

    return build_returned_window_replay_plans(
        job,
        timing_groups,
        ingest_plan,
        index_dir=index_dir,
        alignment_bytes=8192,
        word_bytes=4,
        control_word_bits=30,
        timestamp_quantum_nanoseconds=1,
        replay_identifier=calibration_identifier,
        maximum_requests_per_tile=256,
    )


def prepare_vdif_ingest_plan(
    job: Job,
    *,
    profile: str,
    window: CorrelationWindow,
    max_read_bytes: int,
    assume_ordered: bool,
    index_dir: str | None,
    vdif_io: str,
    planning_cache: VdifPlanningCache | None = None,
) -> VdifIngestPlan:
    """Plan one GPU correlation window without creating a CUDA context."""

    if max_read_bytes <= 0:
        raise ConfigurationError(
            "correlate", "max_read_bytes", "must be positive"
        )
    if vdif_io not in {"posix", "cufile"}:
        raise ConfigurationError(
            "correlate",
            "vdif_io",
            "must be 'posix' or 'cufile'",
        )
    with nvtx_range("xbeam:job-validation"):
        validate_job(job)
    selected_profile = get_profile(profile)
    global_timing_groups = _timing_groups(job, selected_profile)
    timing_groups = select_complete_timing_groups(
        job,
        selected_profile,
        global_timing_groups,
        window,
    )
    recovery_plans = None
    if selected_profile.missing_timestamp_recovery is not None:
        ordinary_ingest = _plan_correlation_vdif_ingest(
            job,
            timing_groups,
            window=window,
            max_read_bytes=max_read_bytes,
            assume_ordered=assume_ordered,
            index_dir=index_dir,
            vdif_io=vdif_io,
            planning_cache=planning_cache,
        )
        recovery_plans = _build_file_recovery_plans(
            selected_profile.missing_timestamp_recovery,
            job,
            timing_groups,
            ordinary_ingest,
            index_dir=index_dir,
        )
        if not recovery_plans or any(
            not plan.has_complete_source_replay
            for plan in recovery_plans.values()
        ):
            raise ConfigurationError(
                "correlate",
                "profile",
                "file-backed missing-timestamp recovery requires a "
                "complete exact source replay plan",
            )
    ingest_plan = _plan_correlation_vdif_ingest(
        job,
        timing_groups,
        window=window,
        max_read_bytes=max_read_bytes,
        assume_ordered=assume_ordered,
        index_dir=index_dir,
        vdif_io=vdif_io,
        planning_cache=planning_cache,
        recovery_plans=recovery_plans,
    )
    return _extend_physical_presentation_ranges(
        job,
        ingest_plan,
        recovery_plans,
        max_read_bytes=max_read_bytes,
    )


def prepare_correlation(
    job: Job,
    *,
    backend: str = "numpy",
    profile: str = "analytic-v1",
    batch_blocks: int | None = None,
    recording_source: RecordingSource | None = None,
    window: CorrelationWindow | None = None,
    max_read_bytes: int = 1 << 30,
    assume_ordered: bool = False,
    index_dir: str | None = None,
    vdif_io: str = "posix",
    autocorrelation_mode: str | AutocorrelationMode = (
        AutocorrelationMode.AVERAGE
    ),
    subintegration_tiling: str = "auto",
    max_tile_subintegrations: int | None = None,
    planning_cache: VdifPlanningCache | None = None,
    start_gpu_warmup: bool = True,
    upload_workers: int | str | None = None,
    vdif_upload_workers: int | str | None = None,
    recovery_plans: Mapping[RecoveryPlanKey, GlobalFFTEligibilityPlan]
    | None = None,
    recovery_plan_window: CorrelationWindow | None = None,
) -> CorrelationRun:
    """Validate a correlation and resolve its bounded input plan eagerly."""

    if max_read_bytes <= 0:
        raise ConfigurationError(
            "correlate", "max_read_bytes", "must be positive"
        )
    if vdif_io not in {"posix", "cufile"}:
        raise ConfigurationError(
            "correlate",
            "vdif_io",
            "must be 'posix' or 'cufile'",
        )
    upload_workers = resolve_upload_workers(
        upload_workers, alias=vdif_upload_workers
    )
    try:
        autocorrelation_mode = AutocorrelationMode(autocorrelation_mode)
    except ValueError as exc:
        choices = ", ".join(repr(item.value) for item in AutocorrelationMode)
        raise ConfigurationError(
            "correlate",
            "autocorrelation_mode",
            f"must be one of {choices}",
        ) from exc
    if subintegration_tiling not in {"auto", "off"}:
        raise ConfigurationError(
            "correlate",
            "subintegration_tiling",
            "must be 'auto' or 'off'",
        )
    if max_tile_subintegrations is not None:
        if (
            not isinstance(max_tile_subintegrations, int)
            or isinstance(max_tile_subintegrations, bool)
            or max_tile_subintegrations <= 0
        ):
            raise ConfigurationError(
                "correlate",
                "max_tile_subintegrations",
                "must be a positive integer",
            )
        if subintegration_tiling == "off":
            raise ConfigurationError(
                "correlate",
                "max_tile_subintegrations",
                "cannot be set when subintegration tiling is off",
            )
    if window is not None and recording_source is not None:
        raise ConfigurationError(
            "correlate",
            "recording_source",
            "windowed runs require the bounded file-backed recording source",
        )
    with nvtx_range("xbeam:job-validation"):
        validate_job(job)
    selected_profile = get_profile(profile)
    global_timing_groups = _timing_groups(job, selected_profile)
    timing_groups = global_timing_groups
    if window is not None:
        timing_groups = select_complete_timing_groups(
            job, selected_profile, timing_groups, window
        )
    recovery_timing_groups = timing_groups
    if recovery_plan_window is not None:
        if recovery_plans is not None:
            raise ConfigurationError(
                "correlate",
                "recovery_plan_window",
                "cannot be combined with explicit recovery plans",
            )
        recovery_timing_groups = select_complete_timing_groups(
            job,
            selected_profile,
            global_timing_groups,
            recovery_plan_window,
        )
        recovery_groups = {
            (scan.identifier, integration)
            for integration, scan, _indices in recovery_timing_groups
        }
        if any(
            (scan.identifier, integration) not in recovery_groups
            for integration, scan, _indices in timing_groups
        ):
            raise ConfigurationError(
                "correlate",
                "recovery_plan_window",
                "must contain every selected execution integration",
            )
    if (
        selected_profile.missing_timestamp_recovery is not None
        and recording_source is None
        and recovery_plans is None
    ):
        with nvtx_range("xbeam:recovery-sidecar-plan"):
            ordinary_intervals = _required_sample_intervals(
                job, timing_groups
            )
            ordinary_ingest = plan_vdif_ingest(
                job,
                intervals=ordinary_intervals,
                max_read_bytes=max_read_bytes,
                assume_ordered=assume_ordered,
                index_dir=index_dir,
                planning_cache=planning_cache,
            )
            recovery_plans = _build_file_recovery_plans(
                selected_profile.missing_timestamp_recovery,
                job,
                recovery_timing_groups,
                ordinary_ingest,
                index_dir=index_dir,
            )
    if (
        selected_profile.missing_timestamp_recovery is not None
        and recording_source is None
        and (
            not recovery_plans
            or any(
                not plan.has_complete_source_replay
                for plan in recovery_plans.values()
            )
        )
    ):
        raise ConfigurationError(
            "correlate",
            "profile",
            "file-backed missing-timestamp recovery requires a complete "
            "exact source replay plan",
        )
    if recovery_plans is not None:
        recovery_plans = MappingProxyType(dict(recovery_plans))
    if selected_profile.missing_timestamp_recovery is None:
        if recovery_plans is not None:
            raise ConfigurationError(
                "correlate",
                "profile",
                "FFT recovery plans require an explicit missing-timestamp "
                "recovery profile",
            )
    elif recovery_plans is None:
        raise ConfigurationError(
            "correlate",
            "profile",
            "missing-timestamp recovery is not calibrated for this input; "
            "an exact FFT fingerprint artifact is required",
        )
    recovery_manifest: dict[str, Any] | None = None
    if selected_profile.missing_timestamp_recovery is not None:
        assert recovery_plans is not None
        expected_recovery_keys = {
            (scan.identifier, datastream.identifier, band.identifier)
            for _integration, scan, _subintegrations in timing_groups
            for datastream in job.datastreams
            for band in datastream.recorded_bands
        }
        observed_recovery_keys = set(recovery_plans)
        if observed_recovery_keys != expected_recovery_keys:
            missing = sorted(expected_recovery_keys - observed_recovery_keys)
            extra = sorted(observed_recovery_keys - expected_recovery_keys)
            raise ConfigurationError(
                "correlate",
                "profile",
                "exact recovery plans do not match selected scan bands; "
                f"missing={missing}, extra={extra}",
            )
        frequencies = {item.identifier: item for item in job.frequencies}
        for _integration, scan, subintegrations in timing_groups:
            for datastream in job.datastreams:
                for band in datastream.recorded_bands:
                    frequency = frequencies[band.frequency_id]
                    factor = (
                        2 if datastream.sampling is SamplingKind.REAL else 1
                    )
                    blocks = round(
                        float(job.subintegration_duration_s)
                        * _sample_rate(datastream, frequency)
                        / (frequency.channel_count * factor)
                    )
                    key = (
                        scan.identifier,
                        datastream.identifier,
                        band.identifier,
                    )
                    try:
                        recovery_plans[key].require_scope(
                            min(subintegrations) * blocks,
                            (max(subintegrations) + 1) * blocks,
                        )
                    except ValueError as exc:
                        raise ConfigurationError(
                            "correlate",
                            "recovery_plans",
                            f"plan {key!r} does not cover execution: {exc}",
                        ) from exc
        identifiers = {
            plan.calibration_identifier for plan in recovery_plans.values()
        }
        if identifiers != {selected_profile.missing_timestamp_recovery}:
            raise ConfigurationError(
                "correlate",
                "profile",
                "recovery calibration identifier does not match the "
                f"profile: {sorted(str(value) for value in identifiers)}",
            )
        recovery_manifest = {
            "calibration_identifier": (
                selected_profile.missing_timestamp_recovery
            ),
            "plans": [
                {
                    "scan_id": key[0],
                    "datastream_id": key[1],
                    "band_id": key[2],
                    "fft_count": plan.fft_count,
                    "ineligible_fft_count": plan.ineligible_fft_count,
                    "source_shifted_fft_count": (
                        plan.source_shifted_fft_count
                    ),
                    "source_masked_fft_count": (plan.source_masked_fft_count),
                    "source_invalid_sample_count": (
                        plan.source_invalid_sample_count
                    ),
                }
                for key, plan in sorted(recovery_plans.items())
            ],
        }
    execution_policy = resolve_execution_policy(
        vdif_io, upload_workers=upload_workers
    )
    xp = get_backend(backend)
    if max_tile_subintegrations is not None and xp.__name__ != "cupy":
        raise ConfigurationError(
            "correlate",
            "max_tile_subintegrations",
            "requires the CuPy backend",
        )
    if vdif_io != "posix" and xp.__name__ != "cupy":
        raise ConfigurationError(
            "correlate",
            "vdif_io",
            "cuFile ingest requires the CuPy backend",
        )
    gpu_warmup = (
        _start_gpu_warmup(job, xp, packed_input=recording_source is None)
        if xp.__name__ == "cupy" and start_gpu_warmup
        else None
    )
    estimated_read_bytes: int | None = None
    effective_vdif_io = vdif_io
    source: RecordingSource
    if recording_source is None:
        ingest_plan = _plan_correlation_vdif_ingest(
            job,
            timing_groups,
            window=window,
            max_read_bytes=max_read_bytes,
            assume_ordered=assume_ordered,
            index_dir=index_dir,
            vdif_io=vdif_io,
            planning_cache=planning_cache,
            recovery_plans=recovery_plans,
        )
        ingest_plan = _extend_physical_presentation_ranges(
            job,
            ingest_plan,
            recovery_plans,
            max_read_bytes=max_read_bytes,
        )
        if (
            xp.__name__ == "numpy"
            and ingest_plan.estimated_decoded_bytes is not None
            and ingest_plan.estimated_decoded_bytes > max_read_bytes
        ):
            raise ConfigurationError(
                "correlate",
                "max_read_bytes",
                "decoded NumPy samples and validity require "
                f"{ingest_plan.estimated_decoded_bytes} bytes; bounded host "
                f"limit is {max_read_bytes}",
            )
        source = VdifFileRecordingSource(
            job,
            selected_profile,
            source_ranges=ingest_plan.source_ranges,
            thread_windows=ingest_plan.thread_windows,
            source_thread_windows=ingest_plan.source_thread_windows,
            sample_ranges=(
                None
                if ingest_plan.payload_intervals is None
                else {
                    key: (value.first_sample, value.stop_sample)
                    for key, value in ingest_plan.payload_intervals.items()
                }
            ),
            max_materialized_bytes=max_read_bytes,
            defer_ordered_headers=ingest_plan.defer_ordered_headers,
            binary_sidecars=(
                ingest_plan.manifest.get("mode") == "binary-sidecar"
            ),
            binary_index_dir=index_dir,
            frame_presentation_plans=(
                None
                if recovery_plans is None
                else _frame_presentation_plans(recovery_plans)
            ),
            physical_frame_presentation_plans=(
                None
                if recovery_plans is None
                else _physical_frame_presentation_plans(recovery_plans)
            ),
            physical_frame_override_plans=(
                None
                if recovery_plans is None
                else _physical_frame_override_plans(recovery_plans)
            ),
        )
        estimated_read_bytes = ingest_plan.selected_source_bytes
        ingest_manifest = {
            **ingest_plan.manifest,
            "max_read_bytes": max_read_bytes,
        }
        ingest_manifest["vdif_io_requested"] = vdif_io
        ingest_manifest["vdif_io_effective"] = effective_vdif_io
        if selected_profile.missing_timestamp_recovery is not None:
            incompatible = [
                item["path"]
                for item in ingest_plan.manifest.get("sources", [])
                if not item.get("index", {}).get("has_frame_states", False)
            ]
            if incompatible:
                raise ConfigurationError(
                    "correlate",
                    "profile",
                    "missing-timestamp recovery requires schema-v4 "
                    "VDIF "
                    "sidecars; rebuild with --force: "
                    + ", ".join(incompatible),
                )
    else:
        source = recording_source
        ingest_manifest = {
            "schema_version": 1,
            "mode": "custom-recording-source",
            "bounded": False,
            "window_selected": window is not None,
            "max_read_bytes": None,
        }
    ingest_manifest["execution_policy"] = execution_policy.to_manifest()
    if recovery_manifest is not None:
        ingest_manifest["missing_timestamp_recovery"] = recovery_manifest
    ingest_manifest["output_pipeline"] = {
        "requested": execution_policy.output_pipeline,
        "effective": False,
        "reason": "disabled"
        if not execution_policy.output_pipeline
        else "pending"
        if xp.__name__ == "cupy"
        else "requires-cupy",
    }
    ingest_manifest["subintegration_tiling"] = {
        "mode": subintegration_tiling,
        "requested_max_subintegrations": max_tile_subintegrations,
        "effective": False,
        "status": "pending" if xp.__name__ == "cupy" else "unsupported",
    }
    ingest_manifest["vdif_upload"] = {
        "requested_workers": (
            "auto" if upload_workers is None else upload_workers
        ),
        "effective_workers": None,
        "frontier_scheduling": False,
        "frontier_reason": "pending"
        if xp.__name__ == "cupy"
        else "requires-cupy",
        "cufile_frontier_requested": execution_policy.cufile_frontier
        and vdif_io == "cufile",
        "cufile_frontier_effective": False,
    }
    ingest_manifest["timing_groups"] = [
        {
            "scan_id": scan.identifier,
            "integration_index": integration_index,
            "subintegration_count": len(subintegrations),
        }
        for integration_index, scan, subintegrations in timing_groups
    ]
    return CorrelationRun(
        job=job,
        backend=backend,
        profile=profile,
        batch_blocks=batch_blocks,
        vdif_io=effective_vdif_io,
        autocorrelation_mode=autocorrelation_mode,
        subintegration_tiling=subintegration_tiling,
        max_tile_subintegrations=max_tile_subintegrations,
        upload_workers=upload_workers,
        source=source,
        timing_groups=tuple(timing_groups),
        xp=xp,
        estimated_read_bytes=estimated_read_bytes,
        ingest_manifest=ingest_manifest,
        gpu_warmup=gpu_warmup,
        execution_policy=execution_policy,
        planned_sample_coverage=recording_source is None,
        recovery_plans=recovery_plans,
    )


def correlate(
    job: Job,
    *,
    backend: str = "numpy",
    profile: str = "analytic-v1",
    batch_blocks: int | None = None,
    recording_source: RecordingSource | None = None,
    window: CorrelationWindow | None = None,
    max_read_bytes: int = 1 << 30,
    assume_ordered: bool = False,
    index_dir: str | None = None,
    vdif_io: str = "posix",
    autocorrelation_mode: str | AutocorrelationMode = (
        AutocorrelationMode.AVERAGE
    ),
    subintegration_tiling: str = "auto",
    max_tile_subintegrations: int | None = None,
    upload_workers: int | str | None = None,
    vdif_upload_workers: int | str | None = None,
    recovery_plans: Mapping[RecoveryPlanKey, GlobalFFTEligibilityPlan]
    | None = None,
) -> Iterator[VisibilityRecord]:
    """Prepare and correlate a normalized job, yielding semantic records."""

    return prepare_correlation(
        job,
        backend=backend,
        profile=profile,
        batch_blocks=batch_blocks,
        recording_source=recording_source,
        window=window,
        max_read_bytes=max_read_bytes,
        assume_ordered=assume_ordered,
        index_dir=index_dir,
        vdif_io=vdif_io,
        autocorrelation_mode=autocorrelation_mode,
        subintegration_tiling=subintegration_tiling,
        max_tile_subintegrations=max_tile_subintegrations,
        upload_workers=upload_workers,
        vdif_upload_workers=vdif_upload_workers,
        recovery_plans=recovery_plans,
    ).records()


__all__ = [
    "CorrelationRun",
    "correlate",
    "prepare_correlation",
]
