# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Upload resources implementation."""

from __future__ import annotations

import atexit
import os
import time
from collections.abc import Iterator, Sequence
from concurrent.futures import Future
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from threading import Condition, Lock
from typing import Any, cast

import numpy as np

from cuphoton.xbeam.adapters.vdif.vdif import (
    DuplicatePolicy,
    VdifError,
    seconds_since_vdif_origin,
)
from cuphoton.xbeam.runtime.policy import (
    current_policy,
    submit_with_policy,
    with_execution_policy,
)

from .cufile_io import CuFileSession
from .kernel_compilation import (
    _cuda_launch_policy,
    _sparse_vdif_ingest_kernel,
    _vdif_ingest_kernel,
)
from .profiling import host_timing_enabled, nvtx_range, record_host_interval
from .runtime_tuning import cufile_strip_stream, vdif_upload_staging_slots

VDIF_TRANSFER_CHUNK_BYTES = 64 * 1024 * 1024
_PINNED_WORKSPACE_GRANULARITY = 4 * 1024 * 1024


# Keep registered addresses, read ranges, and footprint accounting aligned.
_CUFILE_IO_ALIGNMENT = 64 * 1024
_MAX_PINNED_UPLOAD_SLOT_BYTES = 64 * 1024 * 1024
_MAX_CUFILE_UPLOAD_SLOT_BYTES = 64 * 1024 * 1024
_MAX_CACHED_UPLOAD_WORKSPACE_BYTES = 1 * 1024 * 1024 * 1024
_MAX_CACHED_UPLOAD_WORKSPACES_PER_DEVICE = 4
_MAX_CACHED_CUFILE_WORKSPACE_BYTES = 2 * 1024 * 1024 * 1024
_MAX_CACHED_CUFILE_WORKSPACES_PER_DEVICE = 4
_VDIF_IO_BACKENDS = frozenset({"posix", "cufile"})
_CUFILE_SLOTS_ENV = "CUPHOTON_XBEAM_CUFILE_SLOTS"
_CUFILE_READ_TOKEN_FD = "_CUPHOTON_XBEAM_CUFILE_READ_TOKEN_FD"
_CUFILE_RETURN_TOKEN_FD = "_CUPHOTON_XBEAM_CUFILE_RETURN_TOKEN_FD"


# Identity capability passed only after validation of a full mapping group.
_TRUSTED_MAPPED_VDIF_PLANS = object()


@contextmanager
def _global_cufile_read_token() -> Iterator[None]:
    """Bound synchronous cuFile reads across local shard processes."""

    descriptors = current_policy().read_token_fds
    if descriptors is None:
        yield
        return
    read_fd, return_fd = descriptors
    with nvtx_range("xbeam:vdif-global-read-wait"):
        token = os.read(read_fd, 1)
    if len(token) != 1:
        raise RuntimeError(
            "global cuFile read token pipe closed unexpectedly"
        )
    try:
        yield
    finally:
        os.write(return_fd, token)


@dataclass(slots=True)
class _VdifUploadWorkspace:
    device_identifier: int
    capacity: int
    pinned: tuple[Any, Any]
    staging: tuple[Any, Any]
    copy_stream: Any

    @property
    def retained_bytes(self) -> int:
        # Two host and two device buffers.
        return 4 * self.capacity


_upload_workspace_lock = Lock()


_upload_workspaces: dict[int, list[_VdifUploadWorkspace]] = {}


_cached_upload_workspace_bytes = 0


@dataclass(slots=True)
class _VdifPosixUploadState:
    """One call's double-buffered POSIX upload state."""

    device_identifier: int
    workspace: _VdifUploadWorkspace | None = None
    strip_events: list[Any | None] | None = None
    chunk_index: int = 0


@dataclass(slots=True)
class _CuFileRegisteredWorkspace:
    device_identifier: int
    capacity: int
    allocations: tuple[Any, ...]
    buffers: tuple[Any, ...]
    session: CuFileSession

    @property
    def retained_bytes(self) -> int:
        return sum(int(allocation.nbytes) for allocation in self.allocations)


@dataclass(slots=True)
class _VdifCuFileUploadState:
    """Persistent registered resources for one cuFile upload worker."""

    device_identifier: int
    workspace: _CuFileRegisteredWorkspace | None = None
    pinned_mappings: tuple[Any, ...] | None = None
    device_mappings: tuple[Any, ...] | None = None
    mapping_capacity: int = 0
    stream: Any = None
    mapping_stream: Any = None
    slot_events: list[Any | None] | None = None
    chunk_index: int = 0
    mapping_copy_events: list[Any | None] | None = None
    mapping_index: int = 0
    strip_submitter: Any = None
    slot_submissions: list[Any | None] | None = None
    required_raw_bytes: int = 0
    required_mapping_bytes: int = 0
    defer_task_completion: bool = False
    task_completion: Future[None] | None = None
    read_only: bool = False


def _join_cufile_submissions(submissions: list[Any]) -> Future[None]:
    """Complete after every host submission, retaining the first failure.

    No callback waits: it may run on the only CUDA submission thread.
    GPU buffer lifetime is separately protected by the raw-slot events.
    """

    result: Future[None] = Future()
    remaining = len(submissions)
    lock = Lock()
    error: BaseException | None = None
    if not remaining:
        result.set_result(None)
        return result

    def completed(submission: Any) -> None:
        nonlocal remaining, error
        failure: BaseException | None
        try:
            submission.result()
        except BaseException as exc:
            failure = exc
        else:
            failure = None
        with lock:
            if error is None:
                error = failure
            remaining -= 1
            last = remaining == 0
        # User callbacks can take other locks; never invoke them under ours.
        if last:
            if error is None:
                result.set_result(None)
            else:
                result.set_exception(error)

    for submission in submissions:
        submission.add_done_callback(completed)
    return result


@dataclass(frozen=True, slots=True)
class CuFileReadStatisticsSnapshot:
    """Concurrent cuFile read accounting for one correlation process."""

    logical_bytes: int
    requested_bytes: int
    calls: int
    span_seconds: float
    active_read_seconds: float
    first_read_ns: int | None = None
    last_read_ns: int | None = None

    @property
    def logical_gb_per_second(self) -> float:
        return (
            self.logical_bytes / self.span_seconds / 1e9
            if self.span_seconds > 0.0
            else 0.0
        )

    @property
    def mean_read_concurrency(self) -> float:
        return (
            self.active_read_seconds / self.span_seconds
            if self.span_seconds > 0.0
            else 0.0
        )


class CuFileReadStatistics:
    """Thread-safe aggregate read-span and byte accounting."""

    def __init__(self, *, capture_reads: bool = False) -> None:
        self._lock = Lock()
        self._logical_bytes = 0
        self._requested_bytes = 0
        self._calls = 0
        self._active_ns = 0
        self._first_ns: int | None = None
        self._last_ns: int | None = None
        self.read_descriptors: list[tuple[str, int, int, int]] | None = (
            [] if capture_reads else None
        )

    def record(
        self,
        started_ns: int,
        stopped_ns: int,
        logical_bytes: int,
        requested_bytes: int,
        path: str = "",
        offset: int = 0,
    ) -> None:
        with self._lock:
            if self.read_descriptors is not None:
                self.read_descriptors.append(
                    (
                        str(path),
                        int(offset),
                        int(requested_bytes),
                        int(logical_bytes),
                    )
                )
            self._logical_bytes += logical_bytes
            self._requested_bytes += requested_bytes
            self._calls += 1
            self._active_ns += max(0, stopped_ns - started_ns)
            self._first_ns = (
                started_ns
                if self._first_ns is None
                else min(self._first_ns, started_ns)
            )
            self._last_ns = (
                stopped_ns
                if self._last_ns is None
                else max(self._last_ns, stopped_ns)
            )

    def snapshot(self) -> CuFileReadStatisticsSnapshot:
        with self._lock:
            span_ns = (
                0
                if self._first_ns is None or self._last_ns is None
                else self._last_ns - self._first_ns
            )
            return CuFileReadStatisticsSnapshot(
                logical_bytes=self._logical_bytes,
                requested_bytes=self._requested_bytes,
                calls=self._calls,
                span_seconds=span_ns / 1e9,
                active_read_seconds=self._active_ns / 1e9,
                first_read_ns=self._first_ns,
                last_read_ns=self._last_ns,
            )


_vdif_ingest_condition = Condition()


_vdif_ingest_reservations: dict[int, object] = {}


_unreserved_vdif_ingest_users: dict[int, int] = {}


_cufile_workspace_lock = Lock()


_cufile_workspaces: dict[int, list[_CuFileRegisteredWorkspace]] = {}


_cached_cufile_workspace_bytes = 0


def vdif_upload_slot_limit(io_backend: str) -> int:
    """Return the maximum bytes staged by one raw transfer slot."""

    if io_backend not in _VDIF_IO_BACKENDS:
        raise ValueError(f"unsupported VDIF I/O backend {io_backend!r}")
    return (
        _MAX_PINNED_UPLOAD_SLOT_BYTES
        if io_backend == "posix"
        else _MAX_CUFILE_UPLOAD_SLOT_BYTES
    )


def clear_cached_vdif_upload_workspaces(device_identifier: int) -> None:
    """Evict idle POSIX upload workspaces for one reserved device."""

    global _cached_upload_workspace_bytes
    with _upload_workspace_lock:
        workspaces = _upload_workspaces.pop(device_identifier, [])
        _cached_upload_workspace_bytes -= sum(
            workspace.retained_bytes for workspace in workspaces
        )


def _close_cufile_workspace(
    workspace: _CuFileRegisteredWorkspace, *, best_effort: bool = False
) -> None:
    try:
        with workspace.buffers[0].device:
            with nvtx_range("xbeam:cufile-workspace-session-close"):
                workspace.session.close()
    except BaseException:
        if not best_effort:
            raise


def _clear_cached_cufile_workspaces(
    device_identifier: int | None = None,
    *,
    best_effort: bool = False,
) -> None:
    """Close idle registered cuFile workspaces."""

    global _cached_cufile_workspace_bytes
    with _cufile_workspace_lock:
        if device_identifier is None:
            workspaces = [
                workspace
                for cached in _cufile_workspaces.values()
                for workspace in cached
            ]
            _cufile_workspaces.clear()
        else:
            workspaces = _cufile_workspaces.pop(device_identifier, [])
        _cached_cufile_workspace_bytes -= sum(
            workspace.retained_bytes for workspace in workspaces
        )
    first_error: BaseException | None = None
    for workspace in workspaces:
        try:
            _close_cufile_workspace(workspace)
        except BaseException as error:
            if first_error is None:
                first_error = error
    with nvtx_range("xbeam:cufile-workspace-reference-release"):
        # The loop variable retains the last workspace. A saved exception's
        # traceback can still defer its final release beyond this range.
        if workspaces:
            del workspace
        workspaces.clear()
    if first_error is not None and not best_effort:
        raise first_error


def _retain_exact_cufile_workspaces(
    device_identifier: int, capacity: int, slots: int, limit: int
) -> int:
    """Retain exact-size idle workspaces and return resident bytes."""

    global _cached_cufile_workspace_bytes
    kept: list[_CuFileRegisteredWorkspace] = []
    removed: list[_CuFileRegisteredWorkspace] = []
    with _cufile_workspace_lock:
        for workspace in _cufile_workspaces.get(device_identifier, []):
            if (
                workspace.capacity == capacity
                and len(workspace.buffers) == slots
                and len(kept) < limit
            ):
                kept.append(workspace)
            else:
                removed.append(workspace)
        _cufile_workspaces[device_identifier] = kept
        _cached_cufile_workspace_bytes -= sum(
            w.retained_bytes for w in removed
        )
    first_error = None
    for workspace in removed:
        try:
            _close_cufile_workspace(workspace, best_effort=False)
        except BaseException as error:
            if first_error is None:
                first_error = error
    if first_error is not None:
        raise first_error
    return sum(w.retained_bytes for w in kept)


def clear_cached_cufile_workspaces(device_identifier: int) -> None:
    """Evict idle registered cuFile workspaces for one GPU."""

    _clear_cached_cufile_workspaces(device_identifier)


atexit.register(_clear_cached_cufile_workspaces, best_effort=True)


@contextmanager
def reserve_vdif_ingest(device_identifier: int) -> Iterator[object]:
    """Reserve one GPU's uploader cache from planning through execution."""

    token = object()
    with _vdif_ingest_condition:
        while (
            device_identifier in _vdif_ingest_reservations
            or _unreserved_vdif_ingest_users.get(device_identifier, 0)
        ):
            _vdif_ingest_condition.wait()
        _vdif_ingest_reservations[device_identifier] = token
    try:
        yield token
    finally:
        with _vdif_ingest_condition:
            active = _vdif_ingest_reservations.get(device_identifier)
            if active is not token:
                raise RuntimeError("VDIF ingest reservation was lost")
            _vdif_ingest_reservations.pop(device_identifier)
            _vdif_ingest_condition.notify_all()


@contextmanager
def _access_vdif_ingest(
    device_identifier: int, token: object | None
) -> Iterator[None]:
    """Join an engine reservation or protect an independent loader call."""

    reserved = token is not None
    with _vdif_ingest_condition:
        if reserved:
            if _vdif_ingest_reservations.get(device_identifier) is not token:
                raise RuntimeError("invalid VDIF ingest reservation")
        else:
            while device_identifier in _vdif_ingest_reservations:
                _vdif_ingest_condition.wait()
            _unreserved_vdif_ingest_users[device_identifier] = (
                _unreserved_vdif_ingest_users.get(device_identifier, 0) + 1
            )
    try:
        yield
    finally:
        if not reserved:
            with _vdif_ingest_condition:
                users = _unreserved_vdif_ingest_users[device_identifier] - 1
                if users:
                    _unreserved_vdif_ingest_users[device_identifier] = users
                else:
                    _unreserved_vdif_ingest_users.pop(device_identifier)
                    _vdif_ingest_condition.notify_all()


@dataclass(slots=True)
class _StreamingMappingState:
    seen: np.ndarray
    last_slot: int | None = None


_ORDERED_DUPLICATE_POLICY_CODES = {
    DuplicatePolicy.FIRST: 0,
    DuplicatePolicy.FIRST_VALID: 1,
    DuplicatePolicy.LAST: 2,
    DuplicatePolicy.LAST_VALID: 3,
    DuplicatePolicy.REJECT: 4,
}


def cached_vdif_upload_staging_bytes(device_identifier: int) -> int:
    """Return device bytes retained by reusable upload workspaces."""

    with _upload_workspace_lock:
        return sum(
            2 * workspace.capacity
            for workspace in _upload_workspaces.get(device_identifier, ())
        )


def cached_vdif_upload_staging_covers(
    device_identifier: int,
    required_device_bytes: Sequence[int],
    worker_count: int,
) -> bool:
    """Return whether cached workspaces cover every possible active set."""

    if worker_count <= 0:
        return not required_device_bytes
    required = sorted(required_device_bytes, reverse=True)[:worker_count]
    with _upload_workspace_lock:
        available = sorted(
            2 * workspace.capacity
            for workspace in _upload_workspaces.get(device_identifier, ())
        )
    for size in required:
        candidate = next(
            (
                index
                for index, capacity in enumerate(available)
                if capacity >= size
            ),
            None,
        )
        if candidate is None:
            return False
        available.pop(candidate)
    return True


def cached_cufile_staging_bytes(device_identifier: int) -> int:
    """Return device bytes retained by registered cuFile workspaces."""

    with _cufile_workspace_lock:
        return sum(
            workspace.retained_bytes
            for workspace in _cufile_workspaces.get(device_identifier, ())
        )


def cached_cufile_staging_covers(
    device_identifier: int,
    required_device_bytes: Sequence[int],
    worker_count: int,
) -> bool:
    """Return whether cached cuFile buffers cover an active upload set."""

    if worker_count <= 0:
        return not required_device_bytes
    required = sorted(required_device_bytes, reverse=True)[:worker_count]
    slot_count = vdif_upload_staging_slots("cufile")
    with _cufile_workspace_lock:
        available = sorted(
            workspace.retained_bytes
            for workspace in _cufile_workspaces.get(device_identifier, ())
            if len(workspace.buffers) == slot_count
        )
    for size in required:
        candidate = next(
            (
                index
                for index, capacity in enumerate(available)
                if capacity >= size
            ),
            None,
        )
        if candidate is None:
            return False
        available.pop(candidate)
    return True


def automatic_vdif_upload_chunk_bytes(
    integration_bytes: int,
    available_device_bytes: int,
    upload_worker_count: int,
    *,
    direct_storage: bool = False,
    staging_slots: int = 2,
    slot_limit_bytes: int | None = None,
) -> int:
    """Choose an ingest chunk from job geometry and available VRAM."""

    if integration_bytes <= 0 or available_device_bytes <= 0:
        raise ValueError("VDIF integration and device sizes must be positive")
    if upload_worker_count <= 0:
        raise ValueError("upload_worker_count must be positive")
    if staging_slots < 2:
        raise ValueError("VDIF upload staging requires at least two slots")
    if slot_limit_bytes is not None and slot_limit_bytes <= 0:
        raise ValueError("VDIF upload slot limit must be positive")
    gib = 1 << 30
    if available_device_bytes >= 48 * gib:
        preferred_integrations = 8
    elif available_device_bytes >= 16 * gib:
        preferred_integrations = 4
    elif available_device_bytes >= 8 * gib:
        preferred_integrations = 2
    else:
        preferred_integrations = 1
    concurrent_budget = (available_device_bytes // 10) // (
        staging_slots * upload_worker_count
    )
    slot_limit = slot_limit_bytes
    if slot_limit is None:
        slot_limit = (
            _MAX_CUFILE_UPLOAD_SLOT_BYTES
            if direct_storage
            else _MAX_PINNED_UPLOAD_SLOT_BYTES
        )
    transfer_budget = min(slot_limit, concurrent_budget)
    if transfer_budget <= 0:
        raise MemoryError("available GPU memory cannot stage one VDIF frame")
    maximum_integrations = transfer_budget // integration_bytes
    if maximum_integrations == 0:
        return transfer_budget
    # cudaHostAlloc cost scales with the pages pinned, not just the number of
    # calls. Bound each double-buffer slot independently of VRAM and retain
    # power-of-two batches for stable transfer geometry.
    limit = min(
        preferred_integrations,
        maximum_integrations,
    )
    selected_integrations = next(
        (candidate for candidate in (8, 4, 2) if candidate <= limit), 1
    )
    return integration_bytes * selected_integrations


def _align_up(value: int, alignment: int) -> int:
    return -(-value // alignment) * alignment


def _aligned_device_buffer(
    xp: Any, size: int, alignment: int
) -> tuple[Any, Any]:
    """Return an allocation plus an aligned contiguous byte view."""

    allocation = xp.empty(size + alignment - 1, dtype=xp.uint8)
    offset = (-int(allocation.data.ptr)) % alignment
    return allocation, allocation[offset : offset + size]


def _acquire_cufile_workspace(
    xp: Any,
    device_identifier: int,
    required_bytes: int,
    slot_count: int,
) -> _CuFileRegisteredWorkspace:
    """Lease persistent buffers and their synchronous cuFile session."""

    global _cached_cufile_workspace_bytes
    with _cufile_workspace_lock:
        cached = _cufile_workspaces.get(device_identifier, [])
        candidate_index = min(
            (
                index
                for index, workspace in enumerate(cached)
                if workspace.capacity >= required_bytes
                and len(workspace.buffers) == slot_count
            ),
            key=lambda index: cached[index].capacity,
            default=None,
        )
        if candidate_index is not None:
            workspace = cached.pop(candidate_index)
            _cached_cufile_workspace_bytes -= workspace.retained_bytes
            return workspace

    with nvtx_range("xbeam:vdif-cufile-workspace-create"):
        with nvtx_range("xbeam:cufile-workspace-buffer-allocate"):
            allocations_and_buffers = [
                _aligned_device_buffer(
                    xp, required_bytes, _CUFILE_IO_ALIGNMENT
                )
                for _ in range(slot_count)
            ]
        allocations = tuple(item[0] for item in allocations_and_buffers)
        buffers = tuple(item[1] for item in allocations_and_buffers)
        with nvtx_range("xbeam:cufile-session-construct"):
            session = CuFileSession()
        try:
            session.__enter__()
            for buffer in buffers:
                session.register_buffer(
                    int(buffer.data.ptr), int(buffer.nbytes)
                )
        except BaseException:
            session.close()
            raise
    return _CuFileRegisteredWorkspace(
        device_identifier=device_identifier,
        capacity=required_bytes,
        allocations=allocations,
        buffers=buffers,
        session=session,
    )


def _release_cufile_workspace(
    workspace: _CuFileRegisteredWorkspace,
    *,
    reusable: bool,
    best_effort: bool = False,
) -> None:
    """Return a registered workspace to the bounded cache or close it."""

    global _cached_cufile_workspace_bytes
    retained = False
    if reusable:
        with _cufile_workspace_lock:
            cached = _cufile_workspaces.setdefault(
                workspace.device_identifier, []
            )
            if (
                len(cached) < _MAX_CACHED_CUFILE_WORKSPACES_PER_DEVICE
                and _cached_cufile_workspace_bytes + workspace.retained_bytes
                <= _MAX_CACHED_CUFILE_WORKSPACE_BYTES
            ):
                cached.append(workspace)
                _cached_cufile_workspace_bytes += workspace.retained_bytes
                retained = True
    if not retained:
        with nvtx_range("xbeam:vdif-cufile-workspace-close"):
            _close_cufile_workspace(workspace, best_effort=best_effort)


def _prepare_cufile_upload_state(
    state: _VdifCuFileUploadState,
    xp: Any,
    raw_bytes: int,
    mapping_bytes: int,
    ready_event: Any,
    slot_count: int,
    separate_mapping_stream: bool = False,
) -> _VdifCuFileUploadState:
    """Ensure one worker has reusable buffers for its largest task."""

    device_identifier = int(xp.cuda.Device().id)
    if state.device_identifier != device_identifier:
        raise ValueError("cuFile worker state belongs to a different device")

    if state.stream is None:
        state.stream = cufile_strip_stream(xp)
        state.slot_events = [None] * slot_count
    if separate_mapping_stream and state.mapping_stream is None:
        state.mapping_stream = cufile_strip_stream(xp)
    if (
        state.workspace is None
        or state.workspace.capacity < raw_bytes
        or len(state.workspace.buffers) != slot_count
    ):
        with nvtx_range("xbeam:cufile-workspace-resize-wait"):
            state.stream.synchronize()
            if state.mapping_stream is not None:
                state.mapping_stream.synchronize()
        previous_workspace = state.workspace
        state.workspace = None
        if previous_workspace is not None:
            _release_cufile_workspace(previous_workspace, reusable=True)
        state.workspace = _acquire_cufile_workspace(
            xp, device_identifier, raw_bytes, slot_count
        )
        state.slot_events = [None] * slot_count
        state.mapping_copy_events = [None] * slot_count
        state.slot_submissions = [None] * slot_count
    if (
        state.mapping_capacity < mapping_bytes
        or state.pinned_mappings is None
        or len(state.pinned_mappings) != slot_count
    ):
        with nvtx_range("xbeam:cufile-workspace-resize-wait"):
            state.stream.synchronize()
            if state.mapping_stream is not None:
                state.mapping_stream.synchronize()
        import cupyx

        state.pinned_mappings = tuple(
            cupyx.empty_pinned(mapping_bytes, dtype=np.uint8)
            for _ in range(slot_count)
        )
        state.device_mappings = tuple(
            xp.empty(mapping_bytes, dtype=xp.uint8) for _ in range(slot_count)
        )
        state.mapping_capacity = mapping_bytes
        state.slot_events = [None] * slot_count
        state.mapping_copy_events = [None] * slot_count
        state.slot_submissions = [None] * slot_count
    elif state.mapping_copy_events is None:
        state.mapping_copy_events = [None] * slot_count
    if state.slot_submissions is None:
        state.slot_submissions = [None] * slot_count
    # A reused session may receive a newly allocated/zeroed output. Its
    # initialization event is specific to this invocation, not the session.
    state.stream.wait_event(ready_event)
    return state


def _close_vdif_cufile_upload_state(
    state: _VdifCuFileUploadState,
    *,
    reusable: bool = True,
    best_effort: bool = False,
) -> None:
    """Drain one persistent cuFile worker and release its registration."""

    first_error: BaseException | None = None
    if reusable and state.workspace is not None:
        try:
            with nvtx_range("xbeam:vdif-cufile-source-final-stat"):
                state.workspace.session.finish_cached_reads()
        except BaseException as error:
            first_error = error
            reusable = False
    if state.stream is not None:
        try:
            with nvtx_range("xbeam:cufile-workspace-drain-wait"):
                state.stream.synchronize()
        except BaseException as error:
            if first_error is None:
                first_error = error
            reusable = False
    if state.mapping_stream is not None:
        try:
            with nvtx_range("xbeam:cufile-mapping-drain-wait"):
                state.mapping_stream.synchronize()
        except BaseException as error:
            if first_error is None:
                first_error = error
            reusable = False
    if state.workspace is not None:
        try:
            _release_cufile_workspace(
                state.workspace,
                reusable=reusable,
                best_effort=best_effort or first_error is not None,
            )
        except BaseException as error:
            if first_error is None:
                first_error = error
    state.workspace = None
    state.pinned_mappings = None
    state.device_mappings = None
    state.mapping_capacity = 0
    state.stream = None
    state.mapping_stream = None
    state.slot_events = None
    state.chunk_index = 0
    state.mapping_copy_events = None
    state.mapping_index = 0
    state.slot_submissions = None
    if first_error is not None and not best_effort:
        raise first_error


def _acquire_upload_workspace(
    xp: Any,
    device_identifier: int,
    required_bytes: int,
) -> _VdifUploadWorkspace:
    """Lease reusable pinned/device buffers and a nonblocking copy stream."""

    global _cached_upload_workspace_bytes
    with _upload_workspace_lock:
        candidates = _upload_workspaces.get(device_identifier, [])
        candidate_index = min(
            (
                index
                for index, item in enumerate(candidates)
                if item.capacity >= required_bytes
            ),
            key=lambda index: candidates[index].capacity,
            default=None,
        )
        if candidate_index is not None:
            workspace = candidates.pop(candidate_index)
            _cached_upload_workspace_bytes -= workspace.retained_bytes
            return workspace

    import cupyx

    capacity = _align_up(required_bytes, _PINNED_WORKSPACE_GRANULARITY)
    pinned = (
        cupyx.empty_pinned(capacity, dtype=np.uint8),
        cupyx.empty_pinned(capacity, dtype=np.uint8),
    )
    staging = (
        xp.empty(capacity, dtype=xp.uint8),
        xp.empty(capacity, dtype=xp.uint8),
    )
    return _VdifUploadWorkspace(
        device_identifier=device_identifier,
        capacity=capacity,
        pinned=pinned,
        staging=staging,
        copy_stream=xp.cuda.Stream(non_blocking=True),
    )


def _release_upload_workspace(workspace: _VdifUploadWorkspace) -> None:
    global _cached_upload_workspace_bytes
    with _upload_workspace_lock:
        candidates = _upload_workspaces.setdefault(
            workspace.device_identifier, []
        )
        if (
            _cached_upload_workspace_bytes + workspace.retained_bytes
            > _MAX_CACHED_UPLOAD_WORKSPACE_BYTES
            or len(candidates) >= _MAX_CACHED_UPLOAD_WORKSPACES_PER_DEVICE
        ):
            return
        candidates.append(workspace)
        _cached_upload_workspace_bytes += workspace.retained_bytes


def _drain_and_release_upload_workspace(
    workspace: _VdifUploadWorkspace,
    copy_stream: Any,
    strip_stream: Any,
) -> None:
    """Drain both upload streams and return the workspace exactly once."""

    first_error: BaseException | None = None
    try:
        copy_stream.synchronize()
    except BaseException as exc:
        first_error = exc
    try:
        if strip_stream is not copy_stream:
            strip_stream.synchronize()
    except BaseException as exc:
        if first_error is None:
            first_error = exc
    try:
        _release_upload_workspace(workspace)
    except BaseException as exc:
        if first_error is None:
            first_error = exc
    if first_error is not None:
        raise first_error


def _prepare_posix_upload_session(
    session: _VdifPosixUploadState,
    xp: Any,
    required_bytes: int,
    ready_event: Any,
) -> _VdifUploadWorkspace:
    """Ensure a persistent worker session can stage the next task."""

    device_identifier = int(xp.cuda.Device().id)
    if session.device_identifier != device_identifier:
        raise RuntimeError("POSIX upload session belongs to another device")
    workspace = session.workspace
    if workspace is not None and workspace.capacity < required_bytes:
        session.workspace = None
        session.strip_events = None
        session.chunk_index = 0
        _drain_and_release_upload_workspace(
            workspace,
            workspace.copy_stream,
            workspace.copy_stream,
        )
        workspace = None
    if workspace is None:
        workspace = _acquire_upload_workspace(
            xp,
            device_identifier,
            required_bytes,
        )
        session.workspace = workspace
        session.strip_events = [None, None]
    workspace.copy_stream.wait_event(ready_event)
    return workspace


def _close_vdif_posix_upload_state(
    session: _VdifPosixUploadState,
) -> None:
    """Drain and release one POSIX uploader."""

    workspace = session.workspace
    if workspace is None:
        return
    session.workspace = None
    session.strip_events = None
    _drain_and_release_upload_workspace(
        workspace,
        workspace.copy_stream,
        workspace.copy_stream,
    )


def _validate_vdif_source_stat(
    source_stat: Any,
    *,
    path_text: str,
    expected_size_bytes: int | None,
    expected_mtime_ns: int | None,
    expected_device: int | None,
    expected_inode: int | None,
) -> None:
    if (
        (
            expected_size_bytes is not None
            and source_stat.st_size != expected_size_bytes
        )
        or (
            expected_mtime_ns is not None
            and source_stat.st_mtime_ns != expected_mtime_ns
        )
        or (
            expected_device is not None
            and source_stat.st_dev != expected_device
        )
        or (
            expected_inode is not None
            and source_stat.st_ino != expected_inode
        )
    ):
        raise OSError(f"VDIF source changed during GPU ingest: {path_text}")


@contextmanager
def _lease_validated_cufile_source(
    session: CuFileSession,
    reference_file: tuple[Any, ...],
) -> Iterator[Any]:
    """Lease one source, with bounded first/final-stat bookkeeping."""

    path_text = reference_file[0]
    fingerprint = tuple(reference_file[3:7])

    def validate(registered_file: Any) -> None:
        descriptor_stat = os.fstat(registered_file.fd)
        path_stat = os.stat(path_text)
        if (descriptor_stat.st_dev, descriptor_stat.st_ino) != (
            path_stat.st_dev,
            path_stat.st_ino,
        ):
            raise OSError(
                f"VDIF source changed during GPU ingest: {path_text}"
            )
        for source_stat in (descriptor_stat, path_stat):
            _validate_vdif_source_stat(
                source_stat,
                path_text=path_text,
                expected_size_bytes=fingerprint[0],
                expected_mtime_ns=fingerprint[1],
                expected_device=fingerprint[2],
                expected_inode=fingerprint[3],
            )

    with nvtx_range("xbeam:vdif-cufile-handle-open"):
        lease = session.lease_file(
            path_text, fingerprint=fingerprint, validate=validate
        )
        registered_file = lease.__enter__()
    try:
        yield registered_file
    finally:
        lease.__exit__(None, None, None)


def _copy_cufile_mapping_to_device(
    xp: Any,
    device_identifier: int,
    stream: Any,
    device_mapping: Any,
    host_mapping: np.ndarray,
    mapping_bytes: int,
) -> Any:
    """Enqueue one complete frontier mapping and return its event."""

    with xp.cuda.Device(device_identifier), stream:
        with nvtx_range("xbeam:vdif-cufile-mapping-h2d"):
            device_mapping[:mapping_bytes].set(
                host_mapping[:mapping_bytes], stream=stream
            )
        completed = xp.cuda.Event(disable_timing=True)
        completed.record(stream)
    return completed


def _queue_cufile_submission(submitter: Any, function: Any, *args: Any):
    if not host_timing_enabled():
        return submit_with_policy(submitter, function, *args)
    queued_ns = time.perf_counter_ns()

    def submit():
        record_host_interval("xbeam:vdif-cufile-submit-queue", queued_ns)
        return function(*args)

    return submit_with_policy(submitter, submit)


def _submit_cufile_strip(
    xp: Any,
    device_identifier: int,
    stream: Any,
    kernel: Any,
    source: Any,
    destination_mappings: list[Any],
    outputs: list[Any],
    frame_count: int,
    frame_words: int,
    payload_words: int,
    threads: int,
    progresses: list[Any] | None,
    published_destinations: list[np.ndarray],
    source_mappings: list[Any | None] | None = None,
    sparse_kernel: Any | None = None,
    mapping_ready: Any | None = None,
) -> Any:
    """Submit strip/scatter work from the shared CUDA host submitter."""

    if source_mappings is None:
        source_mappings = [None] * len(destination_mappings)
    with xp.cuda.Device(device_identifier), stream:
        if mapping_ready is not None:
            wait_event = getattr(stream, "wait_event", None)
            if wait_event is None:
                # Compatibility fallback for stream wrappers that cannot
                # express a device-side dependency.
                mapping_ready.synchronize()
            else:
                wait_event(mapping_ready)
        with nvtx_range("xbeam:vdif-cufile-strip-submit"):
            for source_frames, destination_slots, output in zip(
                source_mappings, destination_mappings, outputs, strict=True
            ):
                mapping_count = (
                    frame_count
                    if source_frames is None
                    else int(destination_slots.size)
                )
                if not mapping_count:
                    continue
                word_count = mapping_count * payload_words
                if source_frames is None:
                    kernel(
                        ((word_count + threads - 1) // threads,),
                        (threads,),
                        (
                            source,
                            destination_slots,
                            np.int32(frame_count),
                            np.int32(frame_words),
                            np.int32(payload_words),
                            output,
                        ),
                    )
                else:
                    assert sparse_kernel is not None
                    if int(source_frames.size) != mapping_count:
                        raise ValueError(
                            "sparse VDIF source and destination mappings "
                            "must have equal lengths"
                        )
                    destination_count, remainder = divmod(
                        int(output.size), payload_words * 8
                    )
                    if remainder:
                        raise ValueError(
                            "sparse VDIF output has incomplete frame storage"
                        )
                    sparse_kernel(
                        ((word_count + threads - 1) // threads,),
                        (threads,),
                        (
                            source,
                            source_frames,
                            destination_slots,
                            np.int32(mapping_count),
                            np.int32(frame_count),
                            np.int32(destination_count),
                            np.int32(frame_words),
                            np.int32(payload_words),
                            output,
                        ),
                    )
        completed = xp.cuda.Event(disable_timing=True)
        completed.record(stream)
    if progresses is not None:
        for progress, destinations in zip(
            progresses, published_destinations, strict=True
        ):
            progress.publish(destinations, completed)
    return completed


def _streaming_vdif_mappings(
    plans: list[Any],
    raw: np.ndarray,
    frame_count: int,
    frame_size: int,
    states: list[_StreamingMappingState],
) -> tuple[list[np.ndarray], list[int]]:
    """Validate resident raw headers and derive per-thread destinations."""

    rows = np.asarray(raw[: frame_count * frame_size]).reshape(
        frame_count, frame_size
    )
    words = rows[:, :32].view("<u4")
    return _streaming_vdif_header_mappings(
        plans, words, frame_count, frame_size, states
    )


def _streaming_vdif_header_mappings(
    plans: list[Any],
    words: np.ndarray,
    frame_count: int,
    frame_size: int,
    states: list[_StreamingMappingState],
) -> tuple[list[np.ndarray], list[int]]:
    """Validate compact resident headers and derive thread destinations."""

    words = np.asarray(words, dtype="<u4")
    if words.shape != (frame_count, 8):
        raise ValueError(
            "VDIF header slab must contain eight words per frame"
        )
    word0, word1, word2, word3 = (words[:, index] for index in range(4))
    legacy = ((word0 >> np.uint32(30)) & np.uint32(1)).astype(bool)
    reserved = (word1 >> np.uint32(30)) != 0
    versions = (word2 >> np.uint32(29)) & np.uint32(0x7)
    frame_lengths = (word2 & np.uint32(0x00FF_FFFF)).astype(np.int64) * 8
    bits = ((word3 >> np.uint32(26)) & np.uint32(0x1F)).astype(np.int64) + 1
    log2_channels = ((word2 >> np.uint32(24)) & np.uint32(0x1F)).astype(
        np.int64
    )
    complex_data = (word3 >> np.uint32(31)).astype(bool)
    safe_log2 = np.minimum(log2_channels, 30)
    channels = np.left_shift(np.int64(1), safe_log2)
    complete_bits = channels * bits * np.where(complex_data, 2, 1)
    payload_bytes = frame_size - 32
    sample_invalid = (log2_channels > 30) | (
        (payload_bytes * 8) % complete_bits != 0
    )
    malformed = (
        legacy
        | reserved
        | (versions != 0)
        | (frame_lengths != frame_size)
        | (bits != 2)
        | sample_invalid
    )
    if malformed.any():
        first = int(np.flatnonzero(malformed)[0])
        raise VdifError(f"invalid VDIF header in streaming frame {first}")

    thread_ids = ((word3 >> np.uint32(16)) & np.uint32(0x3FF)).astype(
        np.int64
    )
    station_ids = (word3 & np.uint32(0xFFFF)).astype(np.int64)
    extended_versions = (words[:, 4] >> np.uint32(24)).astype(np.int64)
    invalid = (word0 >> np.uint32(31)).astype(bool)
    seconds = (word0 & np.uint32(0x3FFF_FFFF)).astype(np.int64)
    epochs = ((word1 >> np.uint32(24)) & np.uint32(0x3F)).astype(np.int64)
    frame_numbers = (word1 & np.uint32(0x00FF_FFFF)).astype(np.int64)
    mappings: list[np.ndarray] = []
    coverage_stops: list[int] = []
    for plan, state in zip(plans, states, strict=True):
        streaming = plan.streaming
        if streaming is None:
            raise RuntimeError("mixed mapped and streaming VDIF plans")
        template = streaming.template
        mapping = np.full(frame_count, -1, dtype=np.int32)
        selected = np.flatnonzero(thread_ids == streaming.thread_id)
        if not selected.size:
            mappings.append(mapping)
            coverage_stops.append(0)
            continue
        mismatch = (
            (log2_channels[selected] != template.log2_channels)
            | (complex_data[selected] != template.complex_data)
            | (station_ids[selected] != template.station_id)
            | (extended_versions[selected] != template.extended_data_version)
        )
        if mismatch.any():
            first = int(selected[np.flatnonzero(mismatch)[0]])
            raise VdifError(
                f"VDIF thread {streaming.thread_id} format changed in "
                f"streaming frame {first}"
            )
        fps = int(streaming.frames_per_second)
        outside = frame_numbers[selected] >= fps
        if outside.any():
            first = int(selected[np.flatnonzero(outside)[0]])
            raise VdifError(
                f"frame number {int(frame_numbers[first])} is outside "
                f"frames_per_second={fps}"
            )
        selected_epochs = epochs[selected]
        epoch_origins: dict[int, int] = {}
        template_epoch = int(template.reference_epoch)
        for epoch in np.unique(selected_epochs):
            epoch_value = int(epoch)
            epoch_cycle = int(
                epoch_value < template_epoch
                and template_epoch - epoch_value > 32
            )
            epoch_header = replace(
                template,
                reference_epoch=epoch_value,
                seconds_from_epoch=0,
                frame_number=0,
            )
            epoch_origins[epoch_value] = seconds_since_vdif_origin(
                epoch_header, epoch_cycle=epoch_cycle
            )
        absolute_seconds = seconds[selected].copy()
        for epoch_value, origin in epoch_origins.items():
            absolute_seconds[selected_epochs == epoch_value] += origin
        slots = absolute_seconds * fps + frame_numbers[selected]
        if np.any(slots[1:] < slots[:-1]):
            raise VdifError(
                "VDIF timestamps are not ordered as explicitly assumed"
            )
        if state.last_slot is not None and int(slots[0]) < state.last_slot:
            raise VdifError(
                "VDIF timestamps are not ordered as explicitly assumed"
            )
        state.last_slot = int(slots[-1])
        destinations = slots - int(streaming.start_slot)
        inside = (destinations >= 0) & (destinations < int(plan.slot_count))
        selected = selected[inside]
        destinations = destinations[inside]
        if not selected.size:
            mappings.append(mapping)
            coverage_stops.append(0)
            continue
        # Hold the final occupied slot until a later timestamp proves that no
        # duplicate can replace it across a chunk boundary.
        coverage_stops.append(int(destinations[-1]))

        starts = np.r_[True, destinations[1:] != destinations[:-1]]
        group_starts = np.flatnonzero(starts)
        group_ends = np.r_[group_starts[1:], destinations.size]
        duplicate_sizes = group_ends - group_starts
        policy = streaming.duplicate_policy
        if policy is DuplicatePolicy.REJECT:
            duplicate = duplicate_sizes > 1
            if duplicate.any() or np.any(state.seen[destinations]):
                raise VdifError("duplicate VDIF frame in streaming ingest")
            chosen = group_starts
        elif policy is DuplicatePolicy.FIRST:
            chosen = group_starts
        elif policy is DuplicatePolicy.LAST:
            chosen = group_ends - 1
        else:
            positions = np.arange(destinations.size, dtype=np.int64)
            selected_invalid = invalid[selected]
            if policy is DuplicatePolicy.LAST_VALID:
                valid_positions = np.where(~selected_invalid, positions, -1)
                selected_valid = np.maximum.reduceat(
                    valid_positions, group_starts
                )
                chosen = np.where(
                    selected_valid >= group_starts,
                    selected_valid,
                    group_ends - 1,
                )
            else:
                valid_positions = np.where(
                    ~selected_invalid, positions, destinations.size
                )
                selected_valid = np.minimum.reduceat(
                    valid_positions, group_starts
                )
                chosen = np.where(
                    selected_valid < group_ends,
                    selected_valid,
                    group_starts,
                )

        selected = selected[chosen]
        destinations = destinations[chosen].astype(np.int64, copy=False)
        selected_validity = ~invalid[selected]
        previous_seen = state.seen[destinations]
        previous_validity = streaming.frame_validity[destinations]
        if policy is DuplicatePolicy.FIRST:
            accept = ~previous_seen
        elif policy is DuplicatePolicy.LAST:
            accept = np.ones(destinations.size, dtype=np.bool_)
        elif policy is DuplicatePolicy.FIRST_VALID:
            accept = (~previous_seen) | (
                (~previous_validity) & selected_validity
            )
        elif policy is DuplicatePolicy.LAST_VALID:
            accept = selected_validity | (~previous_validity)
        else:
            accept = ~previous_seen
        selected = selected[accept]
        destinations = destinations[accept]
        selected_validity = selected_validity[accept]
        mapping[selected] = destinations.astype(np.int32, copy=False)
        state.seen[destinations] = True
        streaming.frame_validity[destinations] = selected_validity
        mappings.append(mapping)
    return mappings, coverage_stops


@contextmanager
def _cufile_worker_state_lease(state, xp):
    """Close call-owned resources after setup, work, and final source checks.

    Frontier workers pass persistent state and retain responsibility for its
    final drain. General loader calls own their state, including on failures
    during allocation or lazy kernel compilation.
    """
    owned = state is None
    if owned:
        state = _VdifCuFileUploadState(int(xp.cuda.Device().id))
    completed = False
    try:
        yield state
        completed = True
    finally:
        if owned:
            _close_vdif_cufile_upload_state(
                state,
                reusable=completed,
                best_effort=not completed,
            )


def _load_vdif_payload_plans_cufile(
    *,
    plans: list[Any],
    outputs: list[Any],
    reference_files: tuple[tuple[Any, ...], ...],
    validated_mappings: list[tuple[np.ndarray, ...]],
    frame_size: int,
    payload_bytes: int,
    chunk_frames: int,
    ready_event: Any,
    progresses: list[Any] | None,
    xp: Any,
    worker_state: _VdifCuFileUploadState | None = None,
    statistics: CuFileReadStatistics | None = None,
    trusted_mapped_plans: bool = False,
    validated_sources: list[tuple[np.ndarray | None, ...]] | None = None,
) -> list[Any]:
    """Read raw frames directly into registered GPU staging buffers."""

    if validated_sources is None:
        validated_sources = [
            tuple(None for _mapping in plan_mappings)
            for plan_mappings in validated_mappings
        ]
    alignment = _CUFILE_IO_ALIGNMENT
    max_raw_bytes = chunk_frames * frame_size
    registered_bytes = _align_up(
        _align_up(max_raw_bytes + 2 * (alignment - 1), alignment),
        _PINNED_WORKSPACE_GRANULARITY,
    )

    plan_count = len(plans)
    has_sparse_mappings = any(
        source is not None
        for plan_sources in validated_sources
        for source in plan_sources
    )
    mapping_frames = max(
        (int(item[2]) for item in reference_files), default=0
    )
    sparse_mapping_bytes = sum(
        max(
            (
                chunk_frames * np.dtype(np.int32).itemsize
                if plan_sources[file_index] is None
                else int(mapping.size) * 2 * np.dtype(np.int32).itemsize
                for file_index, mapping in enumerate(plan_mappings)
            ),
            default=0,
        )
        for plan_mappings, plan_sources in zip(
            validated_mappings, validated_sources, strict=True
        )
    )
    max_mapping_bytes = (
        plan_count * mapping_frames * np.dtype(np.int32).itemsize
        if trusted_mapped_plans and not has_sparse_mappings
        else sparse_mapping_bytes
    )
    with _cufile_worker_state_lease(worker_state, xp) as worker_state:
        slot_count = vdif_upload_staging_slots("cufile")
        _prepare_cufile_upload_state(
            worker_state,
            xp,
            max(registered_bytes, worker_state.required_raw_bytes),
            max(max_mapping_bytes, worker_state.required_mapping_bytes),
            ready_event,
            slot_count,
            has_sparse_mappings,
        )
        threads = _cuda_launch_policy(
            int(xp.cuda.Device().id), xp
        ).ingest_threads
        kernel = None if worker_state.read_only else _vdif_ingest_kernel()
        sparse_kernel = (
            None
            if worker_state.read_only or not has_sparse_mappings
            else _sparse_vdif_ingest_kernel()
        )
        workspace = worker_state.workspace
        pinned_mappings = worker_state.pinned_mappings
        device_mappings = worker_state.device_mappings
        stream = worker_state.stream
        mapping_stream = worker_state.mapping_stream
        slot_events = worker_state.slot_events
        mapping_copy_events = worker_state.mapping_copy_events
        slot_submissions = worker_state.slot_submissions
        assert workspace is not None
        assert pinned_mappings is not None
        assert device_mappings is not None
        assert stream is not None
        if has_sparse_mappings:
            assert mapping_stream is not None
        assert slot_events is not None
        assert mapping_copy_events is not None
        assert slot_submissions is not None
        payload_words = payload_bytes // 8

        raw_buffers = workspace.buffers
        device_identifier = int(xp.cuda.Device().id)
        task_submissions: list[Any] = []
        body_error: BaseException | None = None
        file_leases = ExitStack()
        try:
            session = workspace.session
            for file_index, (
                path_text,
                first_frame,
                file_frame_count,
                _expected_size_bytes,
                _expected_mtime_ns,
                _expected_device,
                _expected_inode,
            ) in enumerate(reference_files):
                persistent_mapping_slot: int | None = None
                if (
                    trusted_mapped_plans
                    and not has_sparse_mappings
                    and not worker_state.read_only
                ):
                    read_start = 0
                    read_stop = file_frame_count
                    persistent_mapping_slot = (
                        worker_state.mapping_index % slot_count
                    )
                    mapping_copy_event = mapping_copy_events[
                        persistent_mapping_slot
                    ]
                    if mapping_copy_event is not None:
                        with nvtx_range(
                            "xbeam:vdif-cufile-mapping-source-reuse-wait"
                        ):
                            if worker_state.strip_submitter is not None:
                                mapping_copy_event = (
                                    mapping_copy_event.result()
                                )
                            mapping_copy_event.synchronize()
                    host_mapping = pinned_mappings[persistent_mapping_slot]
                    with nvtx_range("xbeam:vdif-cufile-mapping-pack"):
                        for plan_index in range(plan_count):
                            map_start = plan_index * file_frame_count * 4
                            map_stop = map_start + file_frame_count * 4
                            host_mapping[map_start:map_stop].view(np.int32)[
                                :
                            ] = validated_mappings[plan_index][file_index]
                    mapping_bytes = plan_count * file_frame_count * 4
                    mapping_arguments = (
                        xp,
                        device_identifier,
                        stream,
                        device_mappings[persistent_mapping_slot],
                        host_mapping,
                        mapping_bytes,
                    )
                    if worker_state.strip_submitter is None:
                        mapping_copy_event = _copy_cufile_mapping_to_device(
                            *mapping_arguments
                        )
                    else:
                        mapping_copy_event = _queue_cufile_submission(
                            worker_state.strip_submitter,
                            _copy_cufile_mapping_to_device,
                            *mapping_arguments,
                        )
                        task_submissions.append(mapping_copy_event)
                    mapping_copy_events[persistent_mapping_slot] = (
                        mapping_copy_event
                    )
                    worker_state.mapping_index += 1
                elif trusted_mapped_plans and not has_sparse_mappings:
                    read_start, read_stop = 0, file_frame_count
                else:
                    selected_parts = [
                        (
                            np.flatnonzero(mapping[file_index] >= 0)
                            if validated_sources[plan_index][file_index]
                            is None
                            else cast(
                                np.ndarray,
                                validated_sources[plan_index][file_index],
                            )[mapping[file_index] >= 0]
                        )
                        for plan_index, mapping in enumerate(
                            validated_mappings
                        )
                    ]
                    selected_parts = [
                        part for part in selected_parts if part.size
                    ]
                    if not selected_parts:
                        continue
                    read_start = min(
                        int(part.min()) for part in selected_parts
                    )
                    read_stop = max(
                        int(part.max()) + 1 for part in selected_parts
                    )
                if trusted_mapped_plans and not has_sparse_mappings:
                    read_start, read_stop = 0, file_frame_count
                if read_stop > file_frame_count:
                    raise ValueError(
                        "VDIF destination mapping exceeds its "
                        "file frame range"
                    )
                file_leases.close()
                registered_file = file_leases.enter_context(
                    _lease_validated_cufile_source(
                        session, reference_files[file_index]
                    )
                )
                frame_cursor = read_start
                while frame_cursor < read_stop:
                    frame_stop = min(read_stop, frame_cursor + chunk_frames)
                    frame_count = frame_stop - frame_cursor
                    raw_bytes = frame_count * frame_size
                    slot = worker_state.chunk_index % slot_count
                    slot_submission = slot_submissions[slot]
                    if slot_submission is not None:
                        with nvtx_range("xbeam:vdif-cufile-slot-reuse-wait"):
                            with nvtx_range(
                                "xbeam:vdif-cufile-slot-submit-wait"
                            ):
                                slot_event = slot_submission.result()
                            with nvtx_range(
                                "xbeam:vdif-cufile-slot-gpu-wait"
                            ):
                                slot_event.synchronize()
                        slot_submissions[slot] = None
                    elif (slot_event := slot_events[slot]) is not None:
                        with nvtx_range("xbeam:vdif-cufile-slot-reuse-wait"):
                            with nvtx_range(
                                "xbeam:vdif-cufile-slot-gpu-wait"
                            ):
                                slot_event.synchronize()

                    file_offset = (first_frame + frame_cursor) * frame_size
                    aligned_file_offset = file_offset // alignment * alignment
                    prefix = file_offset - aligned_file_offset
                    request_bytes = _align_up(prefix + raw_bytes, alignment)
                    with _global_cufile_read_token():
                        with nvtx_range("xbeam:vdif-cufile-read"):
                            started_ns = time.perf_counter_ns()
                            received = session.read(
                                registered_file,
                                int(raw_buffers[slot].data.ptr),
                                request_bytes,
                                aligned_file_offset,
                            )
                            stopped_ns = time.perf_counter_ns()
                    if statistics is not None:
                        statistics.record(
                            started_ns,
                            stopped_ns,
                            raw_bytes,
                            request_bytes,
                            path_text,
                            aligned_file_offset,
                        )
                    required_bytes = prefix + raw_bytes
                    if received < required_bytes:
                        raise OSError(
                            "VDIF file changed during GPU ingest: expected "
                            f"at least {required_bytes} cuFile bytes, got "
                            f"{received}"
                        )

                    if worker_state.read_only:
                        frame_cursor = frame_stop
                        worker_state.chunk_index += 1
                        continue

                    if persistent_mapping_slot is None:
                        host_mapping = pinned_mappings[slot]
                        chunk_mappings = []
                        chunk_sources: list[np.ndarray | None] = []
                        for plan_index, mapping in enumerate(
                            validated_mappings
                        ):
                            source_frames = validated_sources[plan_index][
                                file_index
                            ]
                            if source_frames is None:
                                chunk_mappings.append(
                                    mapping[file_index][
                                        frame_cursor:frame_stop
                                    ]
                                )
                                chunk_sources.append(None)
                            else:
                                selected = (
                                    (source_frames >= frame_cursor)
                                    & (source_frames < frame_stop)
                                    & (mapping[file_index] >= 0)
                                )
                                chunk_mappings.append(
                                    mapping[file_index][selected]
                                )
                                chunk_sources.append(
                                    np.asarray(
                                        source_frames[selected]
                                        - frame_cursor,
                                        dtype=np.int32,
                                    )
                                )
                        with nvtx_range("xbeam:vdif-cufile-mapping-pack"):
                            mapping_layout = []
                            map_start = 0
                            for plan_index in range(plan_count):
                                destinations = chunk_mappings[plan_index]
                                map_stop = map_start + destinations.nbytes
                                host_mapping[map_start:map_stop].view(
                                    np.int32
                                )[:] = destinations
                                source_frames = chunk_sources[plan_index]
                                source_start = map_stop
                                if source_frames is not None:
                                    source_stop = (
                                        source_start + source_frames.nbytes
                                    )
                                    host_mapping[
                                        source_start:source_stop
                                    ].view(np.int32)[:] = source_frames
                                else:
                                    source_stop = source_start
                                mapping_layout.append(
                                    (
                                        map_start,
                                        map_stop,
                                        source_start,
                                        source_stop,
                                    )
                                )
                                map_start = source_stop
                        mapping_bytes = map_start
                    mapping_ready = None
                    if persistent_mapping_slot is None:
                        mapping_arguments = (
                            xp,
                            device_identifier,
                            mapping_stream if has_sparse_mappings else stream,
                            device_mappings[slot],
                            host_mapping,
                            mapping_bytes,
                        )
                        if has_sparse_mappings:
                            # The sparse map is produced on this worker
                            # thread, while the strip launch may run on the
                            # shared CUDA submitter. Record an explicit
                            # cross-stream edge; the strip-completion event
                            # then protects both the pinned and device mapping
                            # slots until reuse.
                            mapping_ready = _copy_cufile_mapping_to_device(
                                *mapping_arguments
                            )
                        elif worker_state.strip_submitter is None:
                            _copy_cufile_mapping_to_device(*mapping_arguments)
                        else:
                            task_submissions.append(
                                _queue_cufile_submission(
                                    worker_state.strip_submitter,
                                    _copy_cufile_mapping_to_device,
                                    *mapping_arguments,
                                )
                            )

                    destination_mappings: list[Any] = []
                    source_mappings: list[Any | None] = []
                    published_destinations: list[np.ndarray] = []
                    for plan_index in range(plan_count):
                        if persistent_mapping_slot is None:
                            mapping_slot = slot
                            (
                                map_start,
                                map_stop,
                                source_start,
                                source_stop,
                            ) = mapping_layout[plan_index]
                        else:
                            mapping_slot = persistent_mapping_slot
                            map_start = (
                                plan_index * file_frame_count + frame_cursor
                            ) * 4
                            map_stop = map_start + frame_count * 4
                            source_start = map_stop
                            source_stop = map_stop
                        destination_mappings.append(
                            device_mappings[mapping_slot][
                                map_start:map_stop
                            ].view(xp.int32)
                        )
                        source_mappings.append(
                            None
                            if (
                                persistent_mapping_slot is not None
                                or chunk_sources[plan_index] is None
                            )
                            else device_mappings[mapping_slot][
                                source_start:source_stop
                            ].view(xp.int32)
                        )
                        published_destinations.append(
                            validated_mappings[plan_index][file_index][
                                frame_cursor:frame_stop
                            ]
                            if persistent_mapping_slot is not None
                            else chunk_mappings[plan_index]
                        )
                    strip_arguments = (
                        xp,
                        device_identifier,
                        stream,
                        kernel,
                        raw_buffers[slot][prefix : prefix + raw_bytes],
                        destination_mappings,
                        outputs,
                        frame_count,
                        frame_size // 8,
                        payload_words,
                        threads,
                        progresses,
                        published_destinations,
                        source_mappings,
                        sparse_kernel,
                        mapping_ready,
                    )
                    if worker_state.strip_submitter is None:
                        completed = _submit_cufile_strip(*strip_arguments)
                        slot_events[slot] = completed
                    else:
                        submission = _queue_cufile_submission(
                            worker_state.strip_submitter,
                            _submit_cufile_strip,
                            *strip_arguments,
                        )
                        task_submissions.append(submission)
                        slot_submissions[slot] = submission
                        slot_events[slot] = None
                    frame_cursor = frame_stop
                    worker_state.chunk_index += 1
            if worker_state.defer_task_completion:
                worker_state.task_completion = _join_cufile_submissions(
                    task_submissions
                )
            elif task_submissions:
                with nvtx_range("xbeam:vdif-cufile-strip-drain-wait"):
                    for submission in task_submissions:
                        submission.result()
        except BaseException as error:
            body_error = error
            raise
        finally:
            file_leases.close()
            if body_error is not None:
                for submission in task_submissions:
                    try:
                        submission.result()
                    except BaseException:
                        pass
        return outputs


@with_execution_policy
def load_vdif_payload_plans(
    plans: list[Any],
    *,
    xp: Any,
    target_chunk_bytes: int = VDIF_TRANSFER_CHUNK_BYTES,
    outputs: list[Any] | None = None,
    validity_outputs: list[Any] | None = None,
    ready_event: Any | None = None,
    progresses: list[Any] | None = None,
    io_backend: str = "posix",
    available_device_bytes: int | None = None,
    reservation_token: object | None = None,
    posix_worker_state: _VdifPosixUploadState | None = None,
    cufile_worker_state: _VdifCuFileUploadState | None = None,
    cufile_statistics: CuFileReadStatistics | None = None,
    _validated_mapped_token: object | None = None,
) -> list[Any]:
    """Read VDIF while respecting any engine-wide device reservation.

    cuFile requires static destination mappings. Ordered streaming plans are
    supported by the POSIX uploader. Caller-owned worker state retains
    asynchronous work; wait for its completion before inspecting returned
    buffers on another stream. Calls that own their state drain before return.
    """

    if not plans or xp is None:
        return _load_vdif_payload_plans(
            plans,
            xp=xp,
            target_chunk_bytes=target_chunk_bytes,
            outputs=outputs,
            validity_outputs=validity_outputs,
            ready_event=ready_event,
            progresses=progresses,
            io_backend=io_backend,
            available_device_bytes=available_device_bytes,
            posix_worker_state=posix_worker_state,
            cufile_worker_state=cufile_worker_state,
            cufile_statistics=cufile_statistics,
            _validated_mapped_token=_validated_mapped_token,
        )
    device_identifier = int(xp.cuda.Device().id)
    with _access_vdif_ingest(device_identifier, reservation_token):
        return _load_vdif_payload_plans(
            plans,
            xp=xp,
            target_chunk_bytes=target_chunk_bytes,
            outputs=outputs,
            validity_outputs=validity_outputs,
            ready_event=ready_event,
            progresses=progresses,
            io_backend=io_backend,
            available_device_bytes=available_device_bytes,
            posix_worker_state=posix_worker_state,
            cufile_worker_state=cufile_worker_state,
            cufile_statistics=cufile_statistics,
            _validated_mapped_token=_validated_mapped_token,
        )


def _load_vdif_payload_plans(
    plans: list[Any],
    *,
    xp: Any,
    target_chunk_bytes: int = VDIF_TRANSFER_CHUNK_BYTES,
    outputs: list[Any] | None = None,
    validity_outputs: list[Any] | None = None,
    ready_event: Any | None = None,
    progresses: list[Any] | None = None,
    io_backend: str = "posix",
    available_device_bytes: int | None = None,
    posix_worker_state: _VdifPosixUploadState | None = None,
    cufile_worker_state: _VdifCuFileUploadState | None = None,
    cufile_statistics: CuFileReadStatistics | None = None,
    _validated_mapped_token: object | None = None,
) -> list[Any]:
    """Read raw VDIF files through pinned buffers and strip/scatter on GPU."""

    if not plans:
        return []
    if target_chunk_bytes <= 0:
        raise ValueError("target_chunk_bytes must be positive")
    if io_backend not in {"posix", "cufile"}:
        raise ValueError("VDIF I/O backend must be 'posix' or 'cufile'")
    first = plans[0]
    streaming = first.streaming is not None
    if any((plan.streaming is not None) != streaming for plan in plans):
        raise ValueError("cannot mix mapped and streaming VDIF plans")
    if streaming and io_backend == "cufile":
        raise ValueError(
            "cuFile ingestion requires mapped VDIF payload plans; "
            "use POSIX for streaming plans"
        )
    if (
        _validated_mapped_token is not None
        and _validated_mapped_token is not _TRUSTED_MAPPED_VDIF_PLANS
    ):
        raise ValueError("invalid mapped-plan validation token")
    trusted_mapped_plans = (
        _validated_mapped_token is _TRUSTED_MAPPED_VDIF_PLANS
    )
    if trusted_mapped_plans and streaming:
        raise ValueError("trusted mapped plans require mapped ingestion")
    frame_size = int(first.frame_size_bytes)
    payload_bytes = int(first.payload_bytes)
    if frame_size <= 32 or frame_size % 8 or payload_bytes != frame_size - 32:
        raise ValueError(
            "VDIF payload plans require aligned standard headers"
        )
    if frame_size // 8 > np.iinfo(np.int32).max:
        raise ValueError("VDIF frame geometry exceeds the GPU index range")
    reference_files = tuple(
        (
            str(item.path),
            int(item.first_frame),
            int(item.frame_count),
            item.expected_size_bytes,
            item.expected_mtime_ns,
            item.expected_device,
            item.expected_inode,
        )
        for item in first.files
    )
    if any(
        first_frame < 0 or frame_count < 0
        for (
            _path,
            first_frame,
            frame_count,
            _size,
            _mtime,
            _device,
            _inode,
        ) in reference_files
    ):
        raise ValueError("VDIF payload file frame counts must be nonnegative")
    validated_mappings: list[tuple[np.ndarray, ...]] = []
    validated_sources: list[tuple[np.ndarray | None, ...]] = []
    for plan in plans:
        files = tuple(
            (
                str(item.path),
                int(item.first_frame),
                int(item.frame_count),
                item.expected_size_bytes,
                item.expected_mtime_ns,
                item.expected_device,
                item.expected_inode,
            )
            for item in plan.files
        )
        if (
            int(plan.frame_size_bytes) != frame_size
            or int(plan.payload_bytes) != payload_bytes
            or files != reference_files
        ):
            raise ValueError(
                "bulk VDIF plans must describe the same raw files"
            )
        slot_count = int(plan.slot_count)
        if not 0 < slot_count <= np.iinfo(np.int32).max:
            raise ValueError("VDIF payload plan slot count is out of range")
        if int(plan.byte_count) != slot_count * payload_bytes:
            raise ValueError("VDIF payload plan byte count is inconsistent")
        plan_mappings = []
        plan_sources: list[np.ndarray | None] = []
        used_destinations = (
            None
            if trusted_mapped_plans
            else np.zeros(slot_count, dtype=np.bool_)
        )
        for file_plan, (
            _path,
            _first_frame,
            frame_count,
            _size,
            _mtime,
            _device,
            _inode,
        ) in zip(plan.files, reference_files, strict=True):
            if streaming:
                if (
                    file_plan.destination_slots is not None
                    or file_plan.source_frame_indices is not None
                ):
                    raise ValueError(
                        "streaming VDIF files must defer destination mapping"
                    )
                plan_mappings.append(np.empty(0, dtype=np.int32))
                plan_sources.append(None)
                continue
            mapping = np.asarray(file_plan.destination_slots)
            source_frames = file_plan.source_frame_indices
            expected_shape = (
                (frame_count,)
                if source_frames is None
                else np.asarray(source_frames).shape
            )
            if (
                mapping.dtype.kind != "i"
                or mapping.ndim != 1
                or mapping.shape != expected_shape
            ):
                raise TypeError(
                    "VDIF destination slots must be a one-dimensional "
                    "signed integer mapping"
                )
            validated_source = None
            if source_frames is not None:
                source_frames = np.asarray(source_frames)
                if source_frames.dtype.kind != "i" or source_frames.ndim != 1:
                    raise TypeError(
                        "VDIF source frame indices must be one-dimensional "
                        "signed integers"
                    )
                if np.any(
                    (source_frames < 0) | (source_frames >= frame_count)
                ):
                    raise ValueError(
                        "VDIF source frame index is out of range"
                    )
                validated_source = np.ascontiguousarray(
                    source_frames, dtype=np.int64
                )
            if not trusted_mapped_plans:
                if np.any((mapping < -1) | (mapping >= slot_count)):
                    raise ValueError("VDIF destination slot is out of range")
                destinations = mapping[mapping >= 0]
                unique_destinations = np.unique(destinations)
                assert used_destinations is not None
                if unique_destinations.size != destinations.size or np.any(
                    used_destinations[unique_destinations]
                ):
                    raise ValueError("duplicate VDIF destination slot")
                used_destinations[unique_destinations] = True
            if io_backend == "posix" and validated_source is not None:
                # Sort paired physical indices once, not once per transfer.
                keep = mapping >= 0
                validated_source = validated_source[keep]
                mapping = mapping[keep]
                if np.any(validated_source[1:] < validated_source[:-1]):
                    order = np.argsort(validated_source, kind="stable")
                    validated_source = validated_source[order]
                    mapping = mapping[order]
            plan_mappings.append(
                np.ascontiguousarray(mapping, dtype=np.int32)
            )
            plan_sources.append(validated_source)
        validated_mappings.append(tuple(plan_mappings))
        validated_sources.append(tuple(plan_sources))
    plan_count = len(plans)
    validity_ranges: list[tuple[int, int]] = []
    if streaming:
        if validity_outputs is None or len(validity_outputs) != plan_count:
            raise ValueError(
                "streaming VDIF plans require one device validity output"
            )
        for plan, validity in zip(plans, validity_outputs, strict=True):
            if (
                not isinstance(validity, xp.ndarray)
                or validity.dtype != xp.uint8
                or validity.ndim != 1
                or not validity.flags.c_contiguous
                or int(validity.size) != int(plan.slot_count)
            ):
                raise ValueError(
                    "VDIF validity outputs must be contiguous uint8 arrays "
                    "matching each streaming plan"
                )
            validity_ranges.append(
                (
                    int(validity.data.ptr),
                    int(validity.data.ptr) + int(validity.nbytes),
                )
            )
        for index, (start, stop) in enumerate(validity_ranges):
            if any(
                max(start, previous_start) < min(stop, previous_stop)
                for previous_start, previous_stop in validity_ranges[:index]
            ):
                raise ValueError(
                    "VDIF payload and validity buffers must not overlap"
                )
    if progresses is not None and len(progresses) != plan_count:
        raise ValueError("one upload-progress object is required per plan")
    chunk_frames = _vdif_upload_chunk_frames(
        plans, target_chunk_bytes=target_chunk_bytes
    )
    max_mapping_bytes = 0
    for plan_index, plan in enumerate(plans):
        max_mapping_bytes += max(
            (
                chunk_frames * 4
                if validated_sources[plan_index][file_index] is None
                else _sparse_chunk_mapping_bytes(
                    cast(
                        np.ndarray, validated_sources[plan_index][file_index]
                    ),
                    chunk_frames,
                )
                for file_index in range(len(plan.files))
            ),
            default=0,
        )
    max_transfer_bytes = _align_up(
        chunk_frames * frame_size + max_mapping_bytes,
        _PINNED_WORKSPACE_GRANULARITY,
    )
    effective_io_backend = _effective_vdif_io_backend(plans, io_backend)
    if posix_worker_state is not None and effective_io_backend != "posix":
        raise ValueError("POSIX worker state requires the POSIX backend")
    if cufile_worker_state is not None and effective_io_backend != "cufile":
        raise ValueError("cuFile worker state requires the cuFile backend")
    outputs_provided = outputs is not None
    if outputs is not None:
        if len(outputs) != plan_count:
            raise ValueError(
                "one output buffer is required per VDIF payload plan"
            )
        output_ranges: list[tuple[int, int]] = []
        for plan, output in zip(plans, outputs, strict=True):
            if (
                not isinstance(output, xp.ndarray)
                or output.dtype != xp.uint8
                or output.ndim != 1
                or not output.flags.c_contiguous
                or int(output.size) != int(plan.byte_count)
            ):
                raise ValueError(
                    "VDIF output buffers must be contiguous uint8 arrays "
                    "matching each plan"
                )
            start = int(output.data.ptr)
            stop = start + int(output.nbytes)
            if any(
                max(start, previous_start) < min(stop, previous_stop)
                for previous_start, previous_stop in output_ranges
            ) or any(
                max(start, validity_start) < min(stop, validity_stop)
                for validity_start, validity_stop in validity_ranges
            ):
                raise ValueError(
                    "VDIF payload and validity buffers must not overlap"
                )
            output_ranges.append((start, stop))
    if trusted_mapped_plans and effective_io_backend == "cufile":
        raw_capacity, mapping_capacity = _cufile_upload_state_requirements(
            plans,
            target_chunk_bytes=target_chunk_bytes,
            full_mappings=True,
        )
        required_device_bytes, _ = _cufile_retained_staging_bytes(
            raw_capacity,
            mapping_capacity,
            vdif_upload_staging_slots("cufile"),
        )
    else:
        required_device_bytes = vdif_upload_staging_bytes(
            plans,
            target_chunk_bytes=target_chunk_bytes,
            io_backend=effective_io_backend,
        )
    if not outputs_provided:
        required_device_bytes += sum(int(plan.byte_count) for plan in plans)
    if available_device_bytes is None:
        free_bytes, total_bytes = xp.cuda.runtime.memGetInfo()
        available_device_bytes = min(
            int(total_bytes),
            int(free_bytes) + int(xp.get_default_memory_pool().free_bytes()),
        )
    elif available_device_bytes <= 0:
        raise ValueError("available_device_bytes must be positive")
    if required_device_bytes > available_device_bytes:
        raise MemoryError(
            "VDIF payload plans exceed available device memory; split the "
            "recording or reduce the ingest chunk size"
        )

    if outputs is None:
        outputs = [
            xp.zeros(int(plan.byte_count), dtype=xp.uint8) for plan in plans
        ]
    current_stream = xp.cuda.get_current_stream()
    if streaming:
        if ready_event is not None:
            current_stream.wait_event(ready_event)
        assert validity_outputs is not None
        for plan in plans:
            assert plan.streaming is not None
            plan.streaming.frame_validity.fill(False)
        for validity in validity_outputs:
            validity.fill(0)
        ready_event = xp.cuda.Event(disable_timing=True)
        ready_event.record(current_stream)
    elif ready_event is None:
        ready_event = xp.cuda.Event()
        ready_event.record(current_stream)
    if effective_io_backend == "cufile":
        return _load_vdif_payload_plans_cufile(
            plans=plans,
            outputs=outputs,
            reference_files=reference_files,
            validated_mappings=validated_mappings,
            validated_sources=validated_sources,
            frame_size=frame_size,
            payload_bytes=payload_bytes,
            chunk_frames=chunk_frames,
            ready_event=ready_event,
            progresses=progresses,
            xp=xp,
            worker_state=cufile_worker_state,
            statistics=cufile_statistics,
            trusted_mapped_plans=trusted_mapped_plans,
        )

    device_identifier = int(xp.cuda.Device().id)
    owns_worker_state = posix_worker_state is None
    session = (
        _VdifPosixUploadState(device_identifier)
        if posix_worker_state is None
        else posix_worker_state
    )
    if session.device_identifier != device_identifier:
        raise ValueError("POSIX worker state belongs to a different device")
    body_error: BaseException | None = None
    try:
        with nvtx_range("xbeam:vdif-upload-workspace-acquire"):
            workspace = _prepare_posix_upload_session(
                session,
                xp,
                max_transfer_bytes,
                ready_event,
            )
        pinned = workspace.pinned
        staging = workspace.staging
        copy_stream = workspace.copy_stream
        strip_stream = copy_stream
        assert session.strip_events is not None
        strip_events = session.strip_events
        last_completed: Any | None = None
        chunk_index = session.chunk_index
        threads = _cuda_launch_policy(device_identifier, xp).ingest_threads
        kernel = _vdif_ingest_kernel(streaming)
        has_sparse_mappings = any(
            source is not None
            for plan_sources in validated_sources
            for source in plan_sources
        )
        sparse_kernel = (
            _sparse_vdif_ingest_kernel() if has_sparse_mappings else None
        )
        streaming_states = (
            [
                _StreamingMappingState(
                    np.zeros(int(plan.slot_count), dtype=np.bool_)
                )
                for plan in plans
            ]
            if streaming
            else []
        )
        incremental_progress = streaming and len(reference_files) == 1
        for file_index, (
            path_text,
            first_frame,
            file_frame_count,
            expected_size_bytes,
            expected_mtime_ns,
            expected_device,
            expected_inode,
        ) in enumerate(reference_files):
            selected_parts = (
                []
                if streaming
                else [
                    (
                        np.flatnonzero(mapping[file_index] >= 0)
                        if validated_sources[plan_index][file_index] is None
                        else cast(
                            np.ndarray,
                            validated_sources[plan_index][file_index],
                        )[mapping[file_index] >= 0]
                    )
                    for plan_index, mapping in enumerate(validated_mappings)
                ]
            )
            selected_parts = [part for part in selected_parts if part.size]
            if streaming:
                read_start = 0
                read_stop = file_frame_count
                for state in streaming_states:
                    state.last_slot = None
            else:
                if not selected_parts:
                    continue
                read_start = min(int(part.min()) for part in selected_parts)
                read_stop = max(
                    int(part.max()) + 1 for part in selected_parts
                )
            if read_stop > file_frame_count:
                raise ValueError(
                    "VDIF destination mapping exceeds its file frame range"
                )
            frame_cursor = read_start
            with open(path_text, "rb", buffering=0) as stream_file:
                source_stat = os.fstat(stream_file.fileno())
                if (
                    (
                        expected_size_bytes is not None
                        and source_stat.st_size != expected_size_bytes
                    )
                    or (
                        expected_mtime_ns is not None
                        and source_stat.st_mtime_ns != expected_mtime_ns
                    )
                    or (
                        expected_device is not None
                        and source_stat.st_dev != expected_device
                    )
                    or (
                        expected_inode is not None
                        and source_stat.st_ino != expected_inode
                    )
                ):
                    raise OSError(
                        "VDIF source changed after ingest planning: "
                        f"{path_text}"
                    )
                stream_file.seek((first_frame + read_start) * frame_size)
                while frame_cursor < read_stop:
                    frame_stop = min(read_stop, frame_cursor + chunk_frames)
                    frame_count = frame_stop - frame_cursor
                    raw_bytes = frame_count * frame_size
                    slot = chunk_index & 1
                    host_reuse_event = strip_events[slot]
                    if host_reuse_event is not None:
                        with nvtx_range("xbeam:vdif-upload-slot-reuse-wait"):
                            host_reuse_event.synchronize()
                    host_buffer = pinned[slot]
                    view = memoryview(host_buffer)[:raw_bytes]
                    with nvtx_range("xbeam:vdif-posix-read"):
                        received = stream_file.readinto(view)
                    if received != raw_bytes:
                        raise OSError(
                            "VDIF file changed during GPU ingest: "
                            f"expected {raw_bytes} bytes, got {received}"
                        )
                    del view
                    streaming_mapping = (
                        _streaming_vdif_mappings(
                            plans,
                            host_buffer,
                            frame_count,
                            frame_size,
                            streaming_states,
                        )
                        if streaming
                        else None
                    )
                    chunk_sources: list[np.ndarray | None]
                    if streaming_mapping is not None:
                        chunk_mappings, coverage_stops = streaming_mapping
                        chunk_sources = [None] * plan_count
                    else:
                        chunk_mappings = []
                        chunk_sources = []
                        for plan_index, plan_mapping in enumerate(
                            validated_mappings
                        ):
                            source_frames = validated_sources[plan_index][
                                file_index
                            ]
                            if source_frames is None:
                                chunk_mappings.append(
                                    plan_mapping[file_index][
                                        frame_cursor:frame_stop
                                    ]
                                )
                                chunk_sources.append(None)
                            else:
                                lo, hi = np.searchsorted(
                                    source_frames, (frame_cursor, frame_stop)
                                )
                                chunk_mappings.append(
                                    plan_mapping[file_index][lo:hi]
                                )
                                chunk_sources.append(
                                    np.asarray(
                                        source_frames[lo:hi] - frame_cursor,
                                        dtype=np.int32,
                                    )
                                )
                        coverage_stops = []
                    mapping_layout = []
                    map_start = raw_bytes
                    for plan_index in range(plan_count):
                        destinations = chunk_mappings[plan_index]
                        map_stop = map_start + destinations.nbytes
                        map_view = host_buffer[map_start:map_stop].view(
                            np.int32
                        )
                        map_view[:] = destinations
                        source_frames = chunk_sources[plan_index]
                        source_start = map_stop
                        if source_frames is not None:
                            source_stop = source_start + source_frames.nbytes
                            host_buffer[source_start:source_stop].view(
                                np.int32
                            )[:] = source_frames
                        else:
                            source_stop = source_start
                        mapping_layout.append(
                            (map_start, map_stop, source_start, source_stop)
                        )
                        map_start = source_stop
                    transfer_bytes = map_start
                    with (
                        copy_stream,
                        nvtx_range("xbeam:vdif-posix-h2d-submit"),
                    ):
                        # The host buffer is free after this copy, but the
                        # device staging slot is not free until its preceding
                        # strip/scatter kernels have stopped reading it.
                        staging[slot][:transfer_bytes].set(
                            host_buffer[:transfer_bytes], stream=copy_stream
                        )
                    with strip_stream:
                        payload_words = payload_bytes // 8
                        for plan_index, output in enumerate(outputs):
                            (
                                map_start,
                                map_stop,
                                source_start,
                                source_stop,
                            ) = mapping_layout[plan_index]
                            destination_slots = staging[slot][
                                map_start:map_stop
                            ].view(xp.int32)
                            source_frames = chunk_sources[plan_index]
                            mapping_count = int(destination_slots.size)
                            if not mapping_count:
                                continue
                            word_count = mapping_count * payload_words
                            if source_frames is None:
                                kernel(
                                    ((word_count + threads - 1) // threads,),
                                    (threads,),
                                    (
                                        staging[slot],
                                        destination_slots,
                                        np.int32(frame_count),
                                        np.int32(frame_size // 8),
                                        np.int32(payload_words),
                                        output,
                                        *(
                                            (
                                                cast(
                                                    Sequence[object],
                                                    validity_outputs,
                                                )[plan_index],
                                            )
                                            if streaming
                                            else ()
                                        ),
                                    ),
                                )
                            else:
                                assert sparse_kernel is not None
                                sparse_sources = staging[slot][
                                    source_start:source_stop
                                ].view(xp.int32)
                                if int(sparse_sources.size) != mapping_count:
                                    raise ValueError(
                                        "sparse VDIF source and destination "
                                        "mappings must have equal lengths"
                                    )
                                destination_count, remainder = divmod(
                                    int(output.size), payload_words * 8
                                )
                                if remainder:
                                    raise ValueError(
                                        "sparse VDIF output has incomplete "
                                        "frame storage"
                                    )
                                sparse_kernel(
                                    ((word_count + threads - 1) // threads,),
                                    (threads,),
                                    (
                                        staging[slot],
                                        sparse_sources,
                                        destination_slots,
                                        np.int32(mapping_count),
                                        np.int32(frame_count),
                                        np.int32(destination_count),
                                        np.int32(frame_size // 8),
                                        np.int32(payload_words),
                                        output,
                                    ),
                                )
                    completed = xp.cuda.Event(disable_timing=True)
                    completed.record(strip_stream)
                    last_completed = completed
                    strip_events[slot] = completed
                    if progresses is not None:
                        for plan_index, progress in enumerate(progresses):
                            if streaming:
                                if incremental_progress:
                                    progress.publish_through(
                                        coverage_stops[plan_index],
                                        completed,
                                        plans[
                                            plan_index
                                        ].streaming.frame_validity,
                                    )
                            else:
                                progress.publish(
                                    chunk_mappings[plan_index], completed
                                )
                    frame_cursor = frame_stop
                    chunk_index += 1
                    session.chunk_index = chunk_index
                source_stat = os.fstat(stream_file.fileno())
                if (
                    (
                        expected_size_bytes is not None
                        and source_stat.st_size != expected_size_bytes
                    )
                    or (
                        expected_mtime_ns is not None
                        and source_stat.st_mtime_ns != expected_mtime_ns
                    )
                    or (
                        expected_device is not None
                        and source_stat.st_dev != expected_device
                    )
                    or (
                        expected_inode is not None
                        and source_stat.st_ino != expected_inode
                    )
                ):
                    raise OSError(
                        f"VDIF source changed during GPU ingest: {path_text}"
                    )
        if streaming and progresses is not None:
            final_event = (
                ready_event if last_completed is None else last_completed
            )
            for plan, progress in zip(plans, progresses, strict=True):
                progress.publish_through(
                    int(plan.slot_count),
                    final_event,
                    plan.streaming.frame_validity,
                )
    except BaseException as exc:
        body_error = exc
        raise
    finally:
        try:
            if owns_worker_state:
                _close_vdif_posix_upload_state(session)
        except BaseException:
            if body_error is None:
                raise
    return outputs


def _vdif_upload_chunk_frames(
    plans: list[Any],
    *,
    target_chunk_bytes: int = VDIF_TRANSFER_CHUNK_BYTES,
) -> int:
    if not plans:
        return 0
    if target_chunk_bytes <= 0:
        raise ValueError("target_chunk_bytes must be positive")
    frame_size = int(plans[0].frame_size_bytes)
    if frame_size <= 32 or frame_size % 8:
        raise ValueError(
            "VDIF payload plans require aligned standard headers"
        )
    if any(int(plan.frame_size_bytes) != frame_size for plan in plans):
        raise ValueError("bulk VDIF plans must share frame geometry")
    transfer_bytes_per_frame = (
        frame_size + len(plans) * np.dtype(np.int32).itemsize
    )
    chunk_frames = max(1, target_chunk_bytes // transfer_bytes_per_frame)
    return min(chunk_frames, np.iinfo(np.int32).max)


def _sparse_chunk_mapping_bytes(
    sources: np.ndarray, chunk_frames: int
) -> int:
    """Bound a chunk, including physical frames reused at distinct slots."""

    if not sources.size:
        return 0
    _, counts = np.unique(sources, return_counts=True)
    return min(int(sources.size), chunk_frames * int(counts.max())) * 8


def _max_vdif_chunk_mapping_bytes(
    plans: list[Any], chunk_frames: int, *, bounded_sparse: bool = False
) -> int:
    """Bound dense or sparse mapping storage for one raw-frame chunk."""

    total = 0
    for plan in plans:
        total += max(
            (
                chunk_frames * np.dtype(np.int32).itemsize
                if getattr(file_plan, "source_frame_indices", None) is None
                else _sparse_chunk_mapping_bytes(
                    np.asarray(file_plan.source_frame_indices), chunk_frames
                )
                if bounded_sparse
                else np.asarray(file_plan.destination_slots).size
                * 2
                * np.dtype(np.int32).itemsize
                for file_plan in plan.files
            ),
            default=0,
        )
    return total


def _cufile_upload_state_requirements(
    plans: list[Any],
    *,
    target_chunk_bytes: int = VDIF_TRANSFER_CHUNK_BYTES,
    full_mappings: bool = False,
) -> tuple[int, int]:
    """Return registered raw and mapping capacities for one cuFile task."""

    chunk_frames = _vdif_upload_chunk_frames(
        plans, target_chunk_bytes=target_chunk_bytes
    )
    if not chunk_frames:
        return 0, 0
    alignment = _CUFILE_IO_ALIGNMENT
    frame_size = int(plans[0].frame_size_bytes)
    registered_bytes = _align_up(
        _align_up(chunk_frames * frame_size + 2 * (alignment - 1), alignment),
        _PINNED_WORKSPACE_GRANULARITY,
    )
    mapping_frames = (
        max(
            (
                int(file_plan.frame_count)
                for plan in plans
                for file_plan in plan.files
            ),
            default=0,
        )
        if full_mappings
        else chunk_frames
    )
    has_sparse = any(
        getattr(file_plan, "source_frame_indices", None) is not None
        for plan in plans
        for file_plan in plan.files
    )
    mapping_bytes = (
        len(plans) * mapping_frames * np.dtype(np.int32).itemsize
        if full_mappings and not has_sparse
        else _max_vdif_chunk_mapping_bytes(plans, chunk_frames)
    )
    return registered_bytes, mapping_bytes


def _effective_vdif_io_backend(plans: list[Any], io_backend: str) -> str:
    """Return the backend that the loader can actually execute."""

    return io_backend


def _cufile_retained_staging_bytes(
    raw_capacity: int, mapping_capacity: int, slots: int
) -> tuple[int, int]:
    """Device and pinned-host bytes retained by one preallocated worker."""

    return (
        slots * (raw_capacity + _CUFILE_IO_ALIGNMENT - 1 + mapping_capacity),
        slots * mapping_capacity,
    )


def vdif_upload_staging_bytes(
    plans: list[Any],
    *,
    target_chunk_bytes: int = VDIF_TRANSFER_CHUNK_BYTES,
    io_backend: str = "posix",
) -> int:
    """Return the complete peak device footprint of one VDIF uploader."""

    chunk_frames = _vdif_upload_chunk_frames(
        plans, target_chunk_bytes=target_chunk_bytes
    )
    if not chunk_frames:
        return 0
    effective_backend = _effective_vdif_io_backend(plans, io_backend)
    if effective_backend not in {"posix", "cufile"}:
        raise ValueError("VDIF I/O backend must be 'posix' or 'cufile'")
    frame_size = int(plans[0].frame_size_bytes)
    plan_count = len(plans)
    if effective_backend == "posix":
        return 2 * _align_up(
            chunk_frames * frame_size
            + _max_vdif_chunk_mapping_bytes(
                plans, chunk_frames, bounded_sparse=True
            ),
            _PINNED_WORKSPACE_GRANULARITY,
        )

    alignment = _CUFILE_IO_ALIGNMENT
    max_raw_bytes = chunk_frames * frame_size
    aligned_request_bytes = _align_up(
        max_raw_bytes + 2 * (alignment - 1), alignment
    )
    registered_bytes = _align_up(
        aligned_request_bytes,
        _PINNED_WORKSPACE_GRANULARITY,
    )
    staging_slots = vdif_upload_staging_slots(effective_backend)
    raw_allocation_bytes = staging_slots * (registered_bytes + alignment - 1)
    mapping_bytes = staging_slots * _max_vdif_chunk_mapping_bytes(
        plans, chunk_frames
    )
    ordered_bytes = 0
    if plans[0].streaming is not None:
        ordered_bytes = (
            sum(int(plan.slot_count) for plan in plans)
            * np.dtype(np.uint64).itemsize
            + plan_count * 64 * np.dtype(np.int64).itemsize
            + np.dtype(np.uint64).itemsize
            + 2 * np.dtype(np.uint64).itemsize
            + staging_slots * plan_count * np.dtype(np.uint64).itemsize
        )
    return raw_allocation_bytes + mapping_bytes + ordered_bytes


def reusable_vdif_upload_staging_bytes(
    plans: list[Any],
    *,
    target_chunk_bytes: int = VDIF_TRANSFER_CHUNK_BYTES,
    io_backend: str = "posix",
) -> int:
    """Return the portion of one footprint supplied by an idle workspace."""

    effective_backend = _effective_vdif_io_backend(plans, io_backend)
    if effective_backend == "posix":
        return vdif_upload_staging_bytes(
            plans,
            target_chunk_bytes=target_chunk_bytes,
            io_backend=effective_backend,
        )
    if effective_backend != "cufile":
        return 0
    chunk_frames = _vdif_upload_chunk_frames(
        plans, target_chunk_bytes=target_chunk_bytes
    )
    if not chunk_frames:
        return 0
    alignment = _CUFILE_IO_ALIGNMENT
    frame_size = int(plans[0].frame_size_bytes)
    max_raw_bytes = chunk_frames * frame_size
    aligned_request_bytes = _align_up(
        max_raw_bytes + 2 * (alignment - 1), alignment
    )
    registered_bytes = _align_up(
        aligned_request_bytes,
        _PINNED_WORKSPACE_GRANULARITY,
    )
    return vdif_upload_staging_slots(effective_backend) * (
        registered_bytes + alignment - 1
    )
