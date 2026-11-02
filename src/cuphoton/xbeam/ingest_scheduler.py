# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Ingest scheduler implementation."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal
from threading import Condition, Event, Lock
from typing import Any

import numpy as np

from cuphoton.xbeam.adapters.radio.models import Job
from cuphoton.xbeam.adapters.vdif.recording import (
    VdifPayloadFilePlan,
    VdifPayloadPlan,
)
from cuphoton.xbeam.runtime.batches import WorkQueue
from cuphoton.xbeam.runtime.policy import current_policy
from cuphoton.xbeam.runtime.uploads import bounded_upload_workers

from .frontier_planning import gpu_frontier_bounds
from .profiling import nvtx_range
from .runtime_tuning import (
    vdif_upload_staging_slots,
)
from .runtime_types import _BandSamples
from .upload_resources import (
    _TRUSTED_MAPPED_VDIF_PLANS,
    CuFileReadStatistics,
    _close_vdif_cufile_upload_state,
    _close_vdif_posix_upload_state,
    _cufile_retained_staging_bytes,
    _cufile_upload_state_requirements,
    _VdifCuFileUploadState,
    _VdifPosixUploadState,
    automatic_vdif_upload_chunk_bytes,
    cached_cufile_staging_bytes,
    cached_cufile_staging_covers,
    cached_vdif_upload_staging_bytes,
    cached_vdif_upload_staging_covers,
    load_vdif_payload_plans,
    vdif_upload_slot_limit,
    vdif_upload_staging_bytes,
)

_DEFAULT_VDIF_UPLOAD_WORKERS = 4


_PAYLOAD_PROGRESS_WATCHDOG_SECONDS = 0.001


def _upload_vdif_plan_group(
    plans: list[VdifPayloadPlan],
    outputs: list[Any],
    validity_outputs: list[Any],
    xp: Any,
    device_identifier: int,
    ready_event: Any,
    progresses: list[Any],
    target_chunk_bytes: int,
    io_backend: str,
    available_device_bytes: int,
    reservation_token: object | None,
    posix_worker_state: _VdifPosixUploadState | None = None,
    cufile_worker_state: _VdifCuFileUploadState | None = None,
    cufile_statistics: CuFileReadStatistics | None = None,
    validated_mapped_token: object | None = None,
) -> list[Any]:
    """Fill preallocated payload buffers on a background host thread."""

    try:
        with xp.cuda.Device(device_identifier):
            with nvtx_range("xbeam:vdif-upload"):
                result = load_vdif_payload_plans(
                    plans,
                    xp=xp,
                    outputs=outputs,
                    validity_outputs=validity_outputs,
                    ready_event=ready_event,
                    progresses=progresses,
                    target_chunk_bytes=target_chunk_bytes,
                    io_backend=io_backend,
                    available_device_bytes=available_device_bytes,
                    reservation_token=reservation_token,
                    posix_worker_state=posix_worker_state,
                    cufile_worker_state=cufile_worker_state,
                    cufile_statistics=cufile_statistics,
                    _validated_mapped_token=validated_mapped_token,
                )
    except BaseException as exc:
        for progress in progresses:
            progress.finish(exc)
        raise
    if (
        cufile_worker_state is not None
        and cufile_worker_state.defer_task_completion
    ):
        submitted = cufile_worker_state.task_completion
        assert submitted is not None
        finalized: Future[None] = Future()

        def finish_submitted(completion: Future[None]) -> None:
            error: BaseException | None
            try:
                completion.result()
            except BaseException as exc:
                error = exc
            else:
                error = None
            try:
                for progress in progresses:
                    progress.finish(error)
            except BaseException as exc:
                error = error or exc
            if error is None:
                finalized.set_result(None)
            else:
                finalized.set_exception(error)

        submitted.add_done_callback(finish_submitted)
        cufile_worker_state.task_completion = finalized
    else:
        for progress in progresses:
            progress.finish(None)
    return result


class _FrontierPlanCompletion:
    """Complete one logical upload after all of its frontier pieces."""

    def __init__(
        self,
        progress: _PayloadProgress,
        piece_count: int,
        output: Any,
    ) -> None:
        if piece_count <= 0:
            raise ValueError(
                "frontier upload must contain at least one piece"
            )
        self._progress = progress
        self._remaining = piece_count
        self._output = output
        self._lock = Lock()
        self._terminal = False
        self.future: Future[list[Any]] = Future()

    def finish(self, error: BaseException | None) -> None:
        with self._lock:
            if self._terminal:
                return
            if error is not None:
                self._terminal = True
                self._progress.finish(error)
                self.future.set_exception(error)
                return
            self._remaining -= 1
            if self._remaining:
                return
            self._terminal = True
            self._progress.finish(None)
            self.future.set_result([self._output])


class _FrontierPayloadProgress:
    """Translate a sliced plan's destinations into its parent plan."""

    def __init__(
        self,
        progress: _PayloadProgress,
        completion: _FrontierPlanCompletion,
        first_slot: int,
    ) -> None:
        self._progress = progress
        self._completion = completion
        self._first_slot = first_slot

    def publish(self, destinations: np.ndarray, event: Any) -> None:
        selected = np.asarray(destinations)
        selected = selected[selected >= 0].astype(np.int64, copy=False)
        self._progress.publish_validated(
            selected + self._first_slot,
            event,
        )

    def finish(self, error: BaseException | None) -> None:
        self._completion.finish(error)


@dataclass(frozen=True, slots=True)
class _FrontierUploadTask:
    frontier_index: int
    plans: list[VdifPayloadPlan]
    outputs: list[Any]
    validity_outputs: list[Any]
    progresses: list[_FrontierPayloadProgress]
    target_chunk_bytes: int
    validated_mapped_token: object | None = None


def _frontier_staging_plan(
    tasks: Sequence[_FrontierUploadTask],
    requested_workers: int,
    available_bytes: int,
) -> tuple[int, int, int, int, int]:
    """Budget identical workers, including independently maximal capacities.

    Available memory includes exactly sized reusable raw workspaces. The
    caller subtracts retained bytes from the new allocation reservation.
    """

    requirements = [
        _cufile_upload_state_requirements(
            task.plans,
            target_chunk_bytes=task.target_chunk_bytes,
            full_mappings=True,
        )
        for task in tasks
    ]
    raw = max((item[0] for item in requirements), default=0)
    mapping = max((item[1] for item in requirements), default=0)
    device, pinned = _cufile_retained_staging_bytes(
        raw, mapping, vdif_upload_staging_slots("cufile")
    )
    # Match the normal staging policy: at most ten percent of available
    # device memory goes to upload staging, leaving room for compute.
    workers = min(
        len(tasks), requested_workers, available_bytes // 10 // device
    )
    if workers < 1:
        raise MemoryError(
            "cuFile frontier staging cannot fit one worker within the device "
            "staging budget; reduce the ingest window or slot size"
        )
    return workers, raw, mapping, device, pinned


class _FrontierUploadQueue(WorkQueue[_FrontierUploadTask]):
    """Apply frame-frontier admission ordering to generic task dispatch."""

    def __init__(self, tasks: Sequence[_FrontierUploadTask]) -> None:
        indices = [task.frontier_index for task in tasks]
        if indices != sorted(indices):
            raise ValueError("upload tasks must be ordered by frontier")
        super().__init__(tasks)


def _fail_frontier_tasks(
    tasks: Iterable[_FrontierUploadTask], error: BaseException
) -> None:
    for task in tasks:
        for progress in task.progresses:
            progress.finish(error)


def _run_frontier_upload_worker(
    tasks: Sequence[_FrontierUploadTask] | _FrontierUploadQueue,
    xp: Any,
    device_identifier: int,
    ready_event: Any,
    available_device_bytes: int,
    reservation_token: object | None,
    abort: Event | None = None,
) -> None:
    """Run one worker's frontier tasks through fixed double buffers.

    ``abort`` is set once the consumer has stopped; the worker then stops
    before its next task instead of uploading the rest of the recording, and
    the tasks that never started fail with the cancellation error.
    """

    queue = (
        tasks
        if isinstance(tasks, _FrontierUploadQueue)
        else _FrontierUploadQueue(tasks)
    )
    worker_state = _VdifPosixUploadState(device_identifier)
    try:
        with xp.cuda.Device(device_identifier):
            while True:
                if abort is not None and abort.is_set():
                    raise RuntimeError(
                        "correlation stopped before VDIF upload completed"
                    )
                task = queue.take()
                if task is None:
                    break
                try:
                    _upload_vdif_plan_group(
                        task.plans,
                        task.outputs,
                        task.validity_outputs,
                        xp,
                        device_identifier,
                        ready_event,
                        task.progresses,
                        task.target_chunk_bytes,
                        "posix",
                        available_device_bytes,
                        reservation_token,
                        posix_worker_state=worker_state,
                        validated_mapped_token=task.validated_mapped_token,
                    )
                except BaseException as exc:
                    _fail_frontier_tasks(queue.finish(exc), exc)
                    raise
                queue.finish(None)
    except BaseException as exc:
        _fail_frontier_tasks(queue.abort(exc), exc)
        raise
    finally:
        _close_vdif_posix_upload_state(worker_state)


def _run_cufile_frontier_upload_worker(
    task_queue: _FrontierUploadQueue,
    xp: Any,
    device_identifier: int,
    ready_event: Any,
    available_device_bytes: int,
    reservation_token: object | None,
    statistics: CuFileReadStatistics | None,
    required_raw_bytes: int,
    required_mapping_bytes: int,
    strip_submitter: ThreadPoolExecutor,
    read_only: bool = False,
) -> None:
    """Dynamically run cuFile frontiers through persistent staging slots."""

    worker_state = _VdifCuFileUploadState(
        device_identifier,
        required_raw_bytes=required_raw_bytes,
        required_mapping_bytes=required_mapping_bytes,
        strip_submitter=strip_submitter,
        defer_task_completion=True,
        read_only=read_only,
    )
    failed = False
    pending: deque[Future[None]] = deque()

    def finish_task(completion: Future[None]) -> None:
        error = completion.exception()
        cancelled = task_queue.finish(error)
        if error is not None:
            _fail_frontier_tasks(cancelled, error)

    try:
        with xp.cuda.Device(device_identifier):
            while True:
                if len(pending) == 2:
                    with nvtx_range("xbeam:vdif-cufile-task-credit-wait"):
                        pending.popleft().result()
                with nvtx_range("xbeam:vdif-cufile-frontier-queue"):
                    task = task_queue.take()
                if task is None:
                    break
                try:
                    with nvtx_range("xbeam:vdif-cufile-frontier-task"):
                        _upload_vdif_plan_group(
                            task.plans,
                            task.outputs,
                            task.validity_outputs,
                            xp,
                            device_identifier,
                            ready_event,
                            task.progresses,
                            task.target_chunk_bytes,
                            "cufile",
                            available_device_bytes,
                            reservation_token,
                            cufile_worker_state=worker_state,
                            cufile_statistics=statistics,
                            validated_mapped_token=(
                                task.validated_mapped_token
                            ),
                        )
                except BaseException as error:
                    failed = True
                    _fail_frontier_tasks(task_queue.finish(error), error)
                    raise
                completion = worker_state.task_completion
                assert completion is not None
                completion.add_done_callback(finish_task)
                pending.append(completion)
            with nvtx_range("xbeam:vdif-cufile-final-submission-drain"):
                while pending:
                    pending.popleft().result()
    except BaseException as error:
        failed = True
        _fail_frontier_tasks(task_queue.abort(error), error)
        raise
    finally:
        # An earlier task can fail while later submissions still reference
        # this worker's raw/mapping buffers. Drain those before closing it.
        for completion in pending:
            try:
                completion.result()
            except BaseException:
                failed = True
        _close_vdif_cufile_upload_state(
            worker_state,
            reusable=not failed,
            best_effort=failed,
        )


def _mapped_frontier_upload_tasks(
    file_plan_groups: list[list[VdifPayloadPlan]],
    payload_outputs: dict[int, Any],
    payload_validity_outputs: dict[int, Any],
    payload_progresses: dict[int, _PayloadProgress],
    target_chunk_bytes: list[int],
    frontier_count: int,
    *,
    trust_slices: bool = False,
    diagnostics: dict[str, Any] | None = None,
    xp: Any = None,
    planner: str = "auto",
) -> tuple[list[_FrontierUploadTask], dict[int, Future[list[Any]]]] | None:
    """Split indexed plans into bounded, destination-disjoint uploads."""

    if diagnostics is None:
        diagnostics = {}
    diagnostics.update(
        mapping_order="ordered", read_amplification=None, fallback_detail=None
    )
    diagnostics.update(
        planner_requested=planner,
        planner_effective="cpu",
        gpu_planned_mappings=0,
    )
    if planner not in {"auto", "cpu", "gpu"}:
        raise ValueError("frontier planner must be auto, cpu, or gpu")
    gpu_bounds: dict[tuple[int, int], np.ndarray] = {}
    unordered_files: set[tuple[int, int]] = set()

    if frontier_count <= 1 or not file_plan_groups:
        return None
    if len(target_chunk_bytes) != len(file_plan_groups):
        raise ValueError("one chunk size is required per upload plan group")
    selected_plan_frames: dict[
        tuple[int, int], tuple[np.ndarray, np.ndarray]
    ] = {}
    for plans in file_plan_groups:
        if not plans or any(plan.streaming is not None for plan in plans):
            return None
        frame_size = int(plans[0].frame_size_bytes)
        payload_bytes = int(plans[0].payload_bytes)
        if trust_slices and (
            frame_size <= 32
            or frame_size % 8
            or payload_bytes != frame_size - 32
        ):
            raise ValueError(
                "VDIF payload plans require aligned standard headers"
            )
        slot_counts = {int(plan.slot_count) for plan in plans}
        if len(slot_counts) != 1:
            return None
        reference_files = plans[0].files
        if any(len(plan.files) != len(reference_files) for plan in plans):
            return None
        for plan in plans:
            slot_count = int(plan.slot_count)
            if trust_slices and not 0 < slot_count <= np.iinfo(np.int32).max:
                raise ValueError(
                    "VDIF payload plan slot count is out of range"
                )
            if trust_slices and (
                int(plan.frame_size_bytes) != frame_size
                or int(plan.payload_bytes) != payload_bytes
                or int(plan.byte_count) != slot_count * payload_bytes
            ):
                raise ValueError("VDIF payload plan geometry is inconsistent")
            used_destinations = np.zeros(slot_count, dtype=np.bool_)
            for file_index, (file_plan, reference) in enumerate(
                zip(plan.files, reference_files, strict=True)
            ):
                if (
                    file_plan.destination_slots is None
                    or file_plan.path != reference.path
                    or file_plan.first_frame != reference.first_frame
                    or file_plan.frame_count != reference.frame_count
                ):
                    return None
                mapping = np.asarray(file_plan.destination_slots)
                source_mapping = file_plan.source_frame_indices
                expected_shape = (
                    (int(file_plan.frame_count),)
                    if source_mapping is None
                    else np.asarray(source_mapping).shape
                )
                if trust_slices and (
                    mapping.dtype.kind != "i"
                    or mapping.ndim != 1
                    or mapping.shape != expected_shape
                ):
                    raise TypeError(
                        "VDIF destination slots must be a one-dimensional "
                        "signed integer mapping"
                    )
                if source_mapping is not None:
                    source_mapping = np.asarray(source_mapping)
                    if trust_slices and (
                        source_mapping.dtype.kind != "i"
                        or source_mapping.ndim != 1
                        or np.any(source_mapping < 0)
                        or np.any(
                            source_mapping >= int(file_plan.frame_count)
                        )
                    ):
                        raise ValueError(
                            "VDIF source frame index is out of range"
                        )
                if trust_slices and np.any(
                    (mapping < -1) | (mapping >= slot_count)
                ):
                    raise ValueError("VDIF destination slot is out of range")
                selected_entries = mapping >= 0
                selected_frames = (
                    np.flatnonzero(selected_entries)
                    if source_mapping is None
                    else source_mapping[selected_entries]
                )
                selected = mapping[selected_entries]
                unordered = bool(
                    np.any(selected[1:] < selected[:-1])
                    or np.any(selected_frames[1:] < selected_frames[:-1])
                )
                if unordered:
                    diagnostics["mapping_order"] = "unordered"
                    if not trust_slices:
                        return None
                    unordered_files.add((id(plan), file_index))
                if not trust_slices:
                    if np.any(selected[1:] == selected[:-1]) or np.any(
                        used_destinations[selected]
                    ):
                        raise ValueError("duplicate VDIF destination slot")
                    used_destinations[selected] = True
                if trust_slices:
                    device_bounds = None
                    if unordered:
                        frontier_total = min(
                            frontier_count,
                            min(
                                int(group[0].slot_count)
                                for group in file_plan_groups
                                if group
                            ),
                        )
                        if frontier_total > 1:
                            stops = (
                                np.arange(
                                    1, frontier_total + 1, dtype=np.int64
                                )
                                * slot_count
                                // frontier_total
                            )
                            device_bounds = gpu_frontier_bounds(
                                selected_frames,
                                selected,
                                stops,
                                xp,
                                planner,
                                diagnostics,
                            )
                        if device_bounds is not None:
                            gpu_bounds[(id(plan), file_index)] = device_bounds
                    duplicate = (
                        False
                        if device_bounds is not None
                        else (
                            np.unique(selected).size != selected.size
                            if unordered
                            else np.any(selected[1:] == selected[:-1])
                        )
                    )
                    if duplicate or np.any(used_destinations[selected]):
                        raise ValueError("duplicate VDIF destination slot")
                    used_destinations[selected] = True
                    selected_plan_frames[(id(plan), file_index)] = (
                        selected_frames,
                        selected,
                    )

    effective_frontiers = min(
        frontier_count,
        min(int(plans[0].slot_count) for plans in file_plan_groups if plans),
    )
    if effective_frontiers <= 1:
        return None

    # Plan all physical envelopes before allocating slices or completions.
    # Only unordered files need a reduction by destination frontier; ordered
    # files retain searchsorted and never sort their physical mappings.
    planned_bounds: dict[int, list[list[tuple[int, int] | None]]] = {}
    worst_amplification = 1.0
    excessive = False
    for plans in file_plan_groups:
        slots = int(plans[0].slot_count)
        stops = (
            np.arange(1, effective_frontiers + 1, dtype=np.int64)
            * slots
            // effective_frontiers
        )
        bounds: list[list[tuple[int, int] | None]] = [
            [] for _ in range(effective_frontiers)
        ]
        for file_index, reference in enumerate(plans[0].files):
            starts = np.full(
                effective_frontiers,
                int(reference.frame_count),
                dtype=np.int64,
            )
            ends = np.zeros(effective_frontiers, dtype=np.int64)
            unsplit_start, unsplit_stop = int(reference.frame_count), 0
            file_unordered = False
            for plan in plans:
                key = (id(plan), file_index)
                if trust_slices:
                    physical, destinations = selected_plan_frames[key]
                else:
                    mapping = np.asarray(
                        plan.files[file_index].destination_slots
                    )
                    source_mapping = plan.files[
                        file_index
                    ].source_frame_indices
                    physical = (
                        np.flatnonzero(mapping >= 0)
                        if source_mapping is None
                        else np.asarray(source_mapping)[mapping >= 0]
                    )
                    destinations = mapping[mapping >= 0]
                if not physical.size:
                    continue
                unsplit_start = min(unsplit_start, int(physical.min()))
                unsplit_stop = max(unsplit_stop, int(physical.max()) + 1)
                if key in unordered_files:
                    file_unordered = True
                    if key in gpu_bounds:
                        device_bounds = gpu_bounds[key]
                        nonempty = device_bounds[:, 0] >= 0
                        starts[nonempty] = np.minimum(
                            starts[nonempty], device_bounds[nonempty, 0]
                        )
                        ends[nonempty] = np.maximum(
                            ends[nonempty], device_bounds[nonempty, 1]
                        )
                    else:
                        buckets = np.searchsorted(
                            stops, destinations, side="right"
                        )
                        np.minimum.at(starts, buckets, physical)
                        np.maximum.at(ends, buckets, physical + 1)
                else:
                    cuts = np.searchsorted(
                        destinations, np.asarray(np.r_[0, stops])
                    )
                    for frontier in range(effective_frontiers):
                        lo, hi = cuts[frontier : frontier + 2]
                        if lo < hi:
                            starts[frontier] = min(
                                starts[frontier], physical[lo]
                            )
                            ends[frontier] = max(
                                ends[frontier], physical[hi - 1] + 1
                            )
            total = 0
            for frontier in range(effective_frontiers):
                lo, hi = int(starts[frontier]), int(ends[frontier])
                bounds[frontier].append((lo, hi) if lo < hi else None)
                total += max(0, hi - lo)
            if unsplit_start < unsplit_stop:
                denominator = unsplit_stop - unsplit_start
                worst_amplification = max(
                    worst_amplification, total / denominator
                )
                # Integer comparison makes the inclusive 5% boundary exact.
                if file_unordered and total * 100 > denominator * 105:
                    excessive = True
        planned_bounds[id(plans)] = bounds
    gpu_count = diagnostics["gpu_planned_mappings"]
    if gpu_count:
        diagnostics["planner_effective"] = (
            "gpu" if gpu_count == len(unordered_files) else "mixed"
        )
    diagnostics["read_amplification"] = worst_amplification
    if excessive:
        diagnostics["fallback_detail"] = "excessive-read-amplification"
        return None
    # Release full-size temporary indices before copying sliced mappings.
    selected_plan_frames.clear()

    completions: dict[int, _FrontierPlanCompletion] = {}
    futures: dict[int, Future[list[Any]]] = {}
    for plans in file_plan_groups:
        for plan in plans:
            completion = _FrontierPlanCompletion(
                payload_progresses[id(plan)],
                effective_frontiers,
                payload_outputs[id(plan)],
            )
            completions[id(plan)] = completion
            futures[id(plan)] = completion.future

    tasks: list[_FrontierUploadTask] = []
    for frontier in range(effective_frontiers):
        for plans, chunk_bytes in zip(
            file_plan_groups, target_chunk_bytes, strict=True
        ):
            slot_count = int(plans[0].slot_count)
            first_slot = slot_count * frontier // effective_frontiers
            stop_slot = slot_count * (frontier + 1) // effective_frontiers
            file_bounds = planned_bounds[id(plans)][frontier]

            sliced_plans: list[VdifPayloadPlan] = []
            sliced_outputs: list[Any] = []
            sliced_validity: list[Any] = []
            sliced_progresses: list[_FrontierPayloadProgress] = []
            for plan in plans:
                sliced_files: list[VdifPayloadFilePlan] = []
                for file_plan, file_bound in zip(
                    plan.files, file_bounds, strict=True
                ):
                    if file_bound is None:
                        continue
                    first_frame, stop_frame = file_bound
                    source_mapping = file_plan.source_frame_indices
                    if source_mapping is None:
                        mapping = np.asarray(file_plan.destination_slots)[
                            first_frame:stop_frame
                        ].astype(np.int32, copy=True)
                        inside = (mapping >= first_slot) & (
                            mapping < stop_slot
                        )
                        mapping[inside] -= first_slot
                        mapping[~inside] = -1
                        sliced_sources = None
                    else:
                        sources = np.asarray(source_mapping)
                        destinations = np.asarray(file_plan.destination_slots)
                        inside = (
                            (destinations >= first_slot)
                            & (destinations < stop_slot)
                            & (sources >= first_frame)
                            & (sources < stop_frame)
                        )
                        mapping = (destinations[inside] - first_slot).astype(
                            np.int32, copy=False
                        )
                        sliced_sources = (
                            sources[inside] - first_frame
                        ).astype(np.int64, copy=False)
                    sliced_files.append(
                        VdifPayloadFilePlan(
                            path=file_plan.path,
                            frame_count=stop_frame - first_frame,
                            destination_slots=mapping,
                            first_frame=(
                                int(file_plan.first_frame) + first_frame
                            ),
                            expected_size_bytes=file_plan.expected_size_bytes,
                            expected_mtime_ns=file_plan.expected_mtime_ns,
                            expected_device=file_plan.expected_device,
                            expected_inode=file_plan.expected_inode,
                            source_frame_indices=sliced_sources,
                        )
                    )
                sliced_plans.append(
                    VdifPayloadPlan(
                        frame_size_bytes=int(plan.frame_size_bytes),
                        payload_bytes=int(plan.payload_bytes),
                        slot_count=stop_slot - first_slot,
                        files=tuple(sliced_files),
                    )
                )
                payload_start = first_slot * int(plan.payload_bytes)
                payload_stop = stop_slot * int(plan.payload_bytes)
                sliced_outputs.append(
                    payload_outputs[id(plan)][payload_start:payload_stop]
                )
                sliced_validity.append(
                    payload_validity_outputs[id(plan)][first_slot:stop_slot]
                )
                sliced_progresses.append(
                    _FrontierPayloadProgress(
                        payload_progresses[id(plan)],
                        completions[id(plan)],
                        first_slot,
                    )
                )
            tasks.append(
                _FrontierUploadTask(
                    frontier_index=frontier,
                    plans=sliced_plans,
                    outputs=sliced_outputs,
                    validity_outputs=sliced_validity,
                    progresses=sliced_progresses,
                    target_chunk_bytes=chunk_bytes,
                    validated_mapped_token=(
                        _TRUSTED_MAPPED_VDIF_PLANS if trust_slices else None
                    ),
                )
            )
    return tasks, futures


class _PayloadProgressNotifier:
    """Generation-counted wakeup shared by all payload producers."""

    def __init__(self, condition: Condition | None = None) -> None:
        self.condition = Condition() if condition is None else condition
        self.generation = 0

    def notify_locked(self) -> None:
        self.generation += 1
        self.condition.notify_all()


class _PayloadProgress:
    """Publish asynchronous readiness for destination VDIF frame slots."""

    def __init__(
        self,
        slot_count: int,
        *,
        dynamic_validity: bool = False,
        static_validity: np.ndarray | None = None,
        condition: Condition | None = None,
        notifier: _PayloadProgressNotifier | None = None,
    ) -> None:
        if condition is not None and notifier is not None:
            raise ValueError("condition and notifier are mutually exclusive")
        self._notifier = (
            _PayloadProgressNotifier(condition)
            if notifier is None
            else notifier
        )
        self._condition = self._notifier.condition
        self._tickets = np.zeros(slot_count, dtype=np.int32)
        self._events: dict[int, Any] = {}
        self._remaining: dict[int, int] = {}
        self._next_ticket = 1
        self._covered_stop = 0
        self._dynamic_validity = dynamic_validity
        self._static_validity = (
            None
            if static_validity is None
            else np.asarray(static_validity, dtype=np.bool_)
        )
        if (
            self._static_validity is not None
            and self._static_validity.shape != (slot_count,)
        ):
            raise ValueError("static VDIF validity must match upload slots")
        self._finished = False
        self._error: BaseException | None = None

    def samples_published(
        self,
        first_sample: int,
        stop_sample: int,
        samples_per_frame: int,
        validity: np.ndarray,
    ) -> bool:
        """Return whether dependencies for a sample interval are published.

        Publication is deliberately weaker than CUDA event completion.  The
        consumer can enqueue an event dependency without a host
        synchronization, while choosing a producer that has already made
        useful progress instead of blocking on fixed dictionary order.
        """

        first_frame = max(0, first_sample // samples_per_frame)
        stop_frame = min(
            self._tickets.size,
            -(-stop_sample // samples_per_frame),
        )
        with self._condition:
            if self._error is not None:
                raise self._error
            if first_frame >= stop_frame:
                return True
            requested = slice(first_frame, stop_frame)
            if self._dynamic_validity:
                ready = not np.any(self._tickets[requested] == 0)
            else:
                valid = (
                    np.flatnonzero(validity[first_frame:stop_frame])
                    + first_frame
                )
                ready = not valid.size or not np.any(
                    self._tickets[valid] == 0
                )
            if not ready and self._finished:
                raise RuntimeError(
                    "VDIF upload completed without required valid frames"
                )
            return ready

    def publish(self, destinations: np.ndarray, event: Any) -> None:
        selected = np.asarray(destinations, dtype=np.int64)
        selected = np.unique(selected[selected >= 0])
        self.publish_validated(selected, event)

    def publish_validated(self, destinations: np.ndarray, event: Any) -> None:
        """Publish engine-validated unique, in-range destinations."""

        selected = np.asarray(destinations, dtype=np.int64)
        if self._static_validity is not None:
            selected = selected[self._static_validity[selected]]
        if not selected.size:
            return
        with self._condition:
            ticket = self._next_ticket
            self._next_ticket += 1
            self._events[ticket] = event
            self._remaining[ticket] = int(selected.size)
            self._tickets[selected] = ticket
            self._notifier.notify_locked()

    def publish_through(
        self,
        stop: int,
        event: Any,
        validity: np.ndarray,
    ) -> None:
        """Publish ordered streaming coverage, including missing slots."""

        selected_stop = min(self._tickets.size, max(0, int(stop)))
        with self._condition:
            start = self._covered_stop
            if selected_stop <= start:
                return
            selected_validity = np.asarray(validity[start:selected_stop])
            if selected_validity.shape != (selected_stop - start,):
                raise ValueError(
                    "VDIF streaming validity must match upload progress"
                )
            covered = np.arange(start, selected_stop, dtype=np.int64)
            valid = covered[selected_validity.astype(np.bool_, copy=False)]
            self._tickets[start:selected_stop] = -1
            if valid.size:
                ticket = self._next_ticket
                self._next_ticket += 1
                self._events[ticket] = event
                self._remaining[ticket] = int(valid.size)
                self._tickets[valid] = ticket
            self._covered_stop = selected_stop
            self._notifier.notify_locked()

    def finish(self, error: BaseException | None) -> None:
        with self._condition:
            self._finished = True
            self._error = error
            self._notifier.notify_locked()

    def wait_samples(
        self,
        first_sample: int,
        stop_sample: int,
        samples_per_frame: int,
        validity: np.ndarray,
        xp: Any,
    ) -> None:
        first_frame = max(0, first_sample // samples_per_frame)
        stop_frame = min(
            self._tickets.size,
            -(-stop_sample // samples_per_frame),
        )
        if first_frame >= stop_frame:
            with self._condition:
                if self._error is not None:
                    raise self._error
            return
        with self._condition:
            if self._dynamic_validity:
                requested = slice(first_frame, stop_frame)
                while (
                    np.any(self._tickets[requested] == 0)
                    and not self._finished
                ):
                    self._condition.wait()
            if self._error is not None:
                raise self._error
            valid = (
                np.flatnonzero(validity[first_frame:stop_frame]) + first_frame
            )
            if not valid.size:
                return
            if not self._dynamic_validity:
                while (
                    np.any(self._tickets[valid] == 0) and not self._finished
                ):
                    self._condition.wait()
                if self._error is not None:
                    raise self._error
            tickets = self._tickets[valid]
            if np.any(tickets == 0):
                raise RuntimeError(
                    "VDIF upload completed without required valid frames"
                )
            pending_tickets = np.unique(tickets[tickets > 0])
            events = [self._events[int(ticket)] for ticket in pending_tickets]
        stream = xp.cuda.get_current_stream()
        for event in events:
            stream.wait_event(event)
        with self._condition:
            for ticket in pending_tickets:
                selected = valid[self._tickets[valid] == ticket]
                self._tickets[selected] = -1
                remaining = self._remaining[int(ticket)] - int(selected.size)
                if remaining == 0:
                    self._remaining.pop(int(ticket))
                    self._events.pop(int(ticket), None)
                else:
                    self._remaining[int(ticket)] = remaining


def _payload_group_chunk_bytes(
    job: Job,
    plans: list[VdifPayloadPlan],
    band_data: dict[tuple[int, int], _BandSamples],
    *,
    available_device_bytes: int | None = None,
    upload_worker_count: int = 1,
    io_backend: str = "posix",
) -> int:
    """Return an automatic whole-integration transfer staging size.

    The GPU loader stages each raw frame plus one int32 destination index for
    every plan sharing that read.  Include those indices here so its byte-to-
    frame conversion produces exactly the metadata-derived frame count rather
    than a slightly smaller, accidentally unaligned chunk. Large-memory GPUs
    batch more integrations to amortize pinned-buffer/event overhead, while a
    conservative staging budget keeps the same code safe on smaller devices.
    """

    if upload_worker_count <= 0:
        raise ValueError("upload_worker_count must be positive")

    plan_ids = {id(plan) for plan in plans}
    candidate_frames = []
    for data in band_data.values():
        if data.payload_plan is None or id(data.payload_plan) not in plan_ids:
            continue
        frames = int(
            (
                job.integration_duration_s
                * Decimal(str(data.sample_rate_hz))
                / Decimal(data.packed_samples_per_frame)
            ).to_integral_value(rounding=ROUND_CEILING)
        )
        candidate_frames.append(max(1, frames))
    if not candidate_frames:
        raise RuntimeError("VDIF upload group has no associated band data")
    if len({int(value) for value in candidate_frames}) != 1:
        raise RuntimeError(
            "VDIF plans sharing one raw read disagree on integration frame "
            "count"
        )
    frame_size = int(plans[0].frame_size_bytes)
    transfer_bytes_per_frame = (
        frame_size + len(plans) * np.dtype(np.int32).itemsize
    )
    integration_bytes = int(candidate_frames[0]) * transfer_bytes_per_frame
    if available_device_bytes is None:
        return integration_bytes
    if available_device_bytes <= 0:
        raise ValueError("available_device_bytes must be positive")

    selected_bytes = automatic_vdif_upload_chunk_bytes(
        integration_bytes,
        available_device_bytes,
        upload_worker_count,
        direct_storage=io_backend == "cufile",
        staging_slots=vdif_upload_staging_slots(io_backend),
        slot_limit_bytes=vdif_upload_slot_limit(io_backend),
    )
    chunk_frames = selected_bytes // transfer_bytes_per_frame
    if chunk_frames == 0:
        raise MemoryError(
            "available GPU memory cannot stage one complete VDIF frame"
        )
    return chunk_frames * transfer_bytes_per_frame


def _payload_upload_staging_plan(
    job: Job,
    plan_groups: list[list[VdifPayloadPlan]],
    band_data: dict[tuple[int, int], _BandSamples],
    available_device_bytes: int,
    io_backend: str = "posix",
    requested_workers: int | None = None,
) -> tuple[int, list[int], int, int]:
    """Budget concurrent upload workers from one aggregate VRAM allowance."""

    if not plan_groups:
        return 0, [], 0, available_device_bytes
    automatic_workers = (
        current_policy().cufile_upload_workers
        if io_backend == "cufile"
        else _DEFAULT_VDIF_UPLOAD_WORKERS
    )
    worker_count = bounded_upload_workers(
        requested_workers,
        automatic=automatic_workers,
        available=len(plan_groups),
    )
    chunks = [
        _payload_group_chunk_bytes(
            job,
            plans,
            band_data,
            available_device_bytes=available_device_bytes,
            upload_worker_count=worker_count,
            io_backend=io_backend,
        )
        for plans in plan_groups
    ]
    footprints = [
        vdif_upload_staging_bytes(
            plans,
            target_chunk_bytes=chunk_bytes,
            io_backend=io_backend,
        )
        for plans, chunk_bytes in zip(plan_groups, chunks, strict=True)
    ]
    concurrent_staging_bytes = sum(
        sorted(footprints, reverse=True)[:worker_count]
    )
    return (
        worker_count,
        chunks,
        concurrent_staging_bytes,
        available_device_bytes // worker_count,
    )


def _reusable_vdif_upload_staging_bytes(
    vdif_io: str, device_identifier: int
) -> int:
    """Return retained staging that the selected ingest backend can reuse."""

    if vdif_io == "posix":
        return cached_vdif_upload_staging_bytes(device_identifier)
    if vdif_io == "cufile":
        return cached_cufile_staging_bytes(device_identifier)
    return 0


def _reusable_vdif_upload_staging_covers(
    vdif_io: str,
    device_identifier: int,
    required_device_bytes: Sequence[int],
    worker_count: int,
) -> bool:
    """Return whether retained backend staging covers an active upload set."""

    if vdif_io == "posix":
        return cached_vdif_upload_staging_covers(
            device_identifier, required_device_bytes, worker_count
        )
    if vdif_io == "cufile":
        return cached_cufile_staging_covers(
            device_identifier, required_device_bytes, worker_count
        )
    return False


def _same_payload_file_plan(
    left: Any,
    right: Any,
) -> bool:
    return (
        left.path == right.path
        and left.frame_count == right.frame_count
        and left.first_frame == right.first_frame
        and left.expected_size_bytes == right.expected_size_bytes
        and left.expected_mtime_ns == right.expected_mtime_ns
        and left.expected_device == right.expected_device
        and left.expected_inode == right.expected_inode
        and (
            left.destination_slots is not None
            and right.destination_slots is not None
            and np.array_equal(
                left.destination_slots, right.destination_slots
            )
        )
        and (
            (left.source_frame_indices is None)
            == (right.source_frame_indices is None)
        )
        and (
            left.source_frame_indices is None
            or np.array_equal(
                left.source_frame_indices, right.source_frame_indices
            )
        )
    )


def _same_payload_plan(left: VdifPayloadPlan, right: VdifPayloadPlan) -> bool:
    if left.streaming is not None or right.streaming is not None:
        return left is right
    return (
        left.frame_size_bytes == right.frame_size_bytes
        and left.payload_bytes == right.payload_bytes
        and left.slot_count == right.slot_count
        and len(left.files) == len(right.files)
        and all(
            _same_payload_file_plan(left_file, right_file)
            for left_file, right_file in zip(
                left.files, right.files, strict=True
            )
        )
    )


def _deduplicate_payload_plan_groups(
    groups: list[list[VdifPayloadPlan]],
) -> tuple[list[list[VdifPayloadPlan]], dict[int, VdifPayloadPlan]]:
    """Canonicalize identical raw reads without hashing large index arrays."""

    unique: list[list[VdifPayloadPlan]] = []
    aliases: dict[int, VdifPayloadPlan] = {}
    for group in groups:
        canonical_group = next(
            (
                candidate
                for candidate in unique
                if len(candidate) == len(group)
                and all(
                    _same_payload_plan(left, right)
                    for left, right in zip(group, candidate, strict=True)
                )
            ),
            None,
        )
        if canonical_group is None:
            canonical_group = group
            unique.append(canonical_group)
        for alias, canonical in zip(group, canonical_group, strict=True):
            aliases[id(alias)] = canonical
    return unique, aliases
