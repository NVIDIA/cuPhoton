# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Host submissions and GPU fences, independent of input formats."""

from __future__ import annotations

from concurrent.futures import Future
from threading import Lock
from typing import Any


def wait_for_host(event: Any, submitted: Any = None) -> None:
    """Drain host submission and I/O before overwriting host-owned buffers."""
    if submitted is not None:
        submitted.result()
    if event is not None:
        event.synchronize()


def record_completion(
    xp: Any, stream: Any, *, disable_timing: bool = True
) -> Any:
    event = xp.cuda.Event(disable_timing=disable_timing)
    event.record(stream)
    return event


def join_submissions(submissions: list[Any]) -> Future[None]:
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
