# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Validated invocation policy, shared with worker and iterator contexts.

Public low-level helpers resolve a policy when used outside a prepared run.
Prepared execution and its submissions never consult the environment again.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from contextvars import ContextVar, copy_context
from dataclasses import asdict, dataclass
from functools import wraps
from typing import Any

from .uploads import normalize_upload_workers

_PREFIX = "CUPHOTON_XBEAM_"
_active: ContextVar[ExecutionPolicy | None] = ContextVar(
    "xbeam_policy", default=None
)


def _boolean(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "on", "true", "yes"}:
        return True
    if normalized in {"", "0", "off", "false", "no"}:
        return False
    raise ValueError("must be a boolean value")


def _priority(value: str) -> bool:
    if value.strip() not in {"0", "1"}:
        raise ValueError("must be 0 or 1")
    return value.strip() == "1"


def _integer(value: str, low: int, high: int | None, message: str) -> int:
    try:
        result = int(value)
    except ValueError:
        raise ValueError(message) from None
    if result < low or (high is not None and result > high):
        raise ValueError(message)
    return result


def _planner(value: str) -> str:
    value = value.strip().lower()
    if value not in {"auto", "cpu", "gpu"}:
        raise ValueError("must be auto, cpu, or gpu")
    return value


def _setting(env, suffix, parser, default):
    name = _PREFIX + suffix
    if name not in env:
        return default, False
    try:
        return parser(env[name]), True
    except ValueError as error:
        raise ValueError(f"{name} {error}") from error


@dataclass(frozen=True, slots=True)
class ExecutionPolicy:
    cufile_frontier: bool = True
    cufile_frontier_planner: str = "auto"
    cufile_upload_workers: int = 4
    cufile_slots: int = 4
    cufile_high_priority: bool = True
    xmac_packs_per_launch: int = 64
    output_pipeline: bool = True
    global_gds_reads: int | None = None
    read_token_fds: tuple[int, int] | None = None

    def to_manifest(self):
        result = asdict(self)
        result.pop("read_token_fds")
        result["global_gds_read_scope"] = "local-launcher"
        result["global_gds_read_token_attached"] = (
            self.read_token_fds is not None
        )
        return result


def resolve_execution_policy(
    io_backend: str | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    upload_workers: int | str | None = None,
) -> ExecutionPolicy:
    env = os.environ if environ is None else environ
    if io_backend not in {None, "posix", "cufile"}:
        raise ValueError(f"unsupported VDIF I/O backend {io_backend!r}")
    defaults = ExecutionPolicy()
    definitions = {
        "cufile_frontier": ("CUFILE_FRONTIER", _boolean),
        "cufile_frontier_planner": ("CUFILE_FRONTIER_PLANNER", _planner),
        "cufile_slots": (
            "CUFILE_SLOTS",
            lambda v: _integer(
                v.strip() or str(defaults.cufile_slots),
                2,
                4,
                "must be 2, 3, or 4",
            ),
        ),
        "cufile_high_priority": ("CUFILE_HIGH_PRIORITY", _priority),
        "xmac_packs_per_launch": (
            "XMAC_PACKS_PER_LAUNCH",
            lambda v: _integer(v, 0, None, "must be a non-negative integer"),
        ),
        "output_pipeline": ("OUTPUT_PIPELINE", _boolean),
        "global_gds_reads": (
            "GLOBAL_GDS_READS",
            lambda v: _integer(
                v, 1, 64, "must be an integer from 1 through 64"
            ),
        ),
    }
    values: dict[str, Any] = {
        "cufile_upload_workers": normalize_upload_workers(upload_workers)
        or defaults.cufile_upload_workers
    }
    for field, (suffix, parser) in definitions.items():
        value, explicit = _setting(
            env, suffix, parser, getattr(defaults, field)
        )
        if (
            io_backend == "posix"
            and explicit
            and (
                field
                in {
                    "cufile_slots",
                    "global_gds_reads",
                    "cufile_frontier_planner",
                }
                or field in {"cufile_frontier", "cufile_high_priority"}
                and value
            )
        ):
            raise ValueError(
                f"{_PREFIX}{suffix} requires the cuFile backend "
                "(--vdif-io cufile)"
            )
        values[field] = value
    read_fd = env.get("_CUPHOTON_XBEAM_CUFILE_READ_TOKEN_FD")
    return_fd = env.get("_CUPHOTON_XBEAM_CUFILE_RETURN_TOKEN_FD")
    if (read_fd is None) != (return_fd is None):
        raise RuntimeError(
            "incomplete global cuFile read token configuration"
        )
    if read_fd is not None:
        assert return_fd is not None
        values["read_token_fds"] = (int(read_fd), int(return_fd))
    return ExecutionPolicy(**values)


def current_policy(io_backend: str | None = None) -> ExecutionPolicy:
    policy = _active.get()
    return (
        policy if policy is not None else resolve_execution_policy(io_backend)
    )


def submit_with_policy(executor, function, *args, **kwargs):
    """ThreadPoolExecutor does not propagate ContextVars itself."""
    context = copy_context()
    return executor.submit(context.run, function, *args, **kwargs)


def policy_records(policy, factory):
    """Keep policy active during next/close, never across a consumer yield."""
    context = copy_context()
    context.run(_active.set, policy)
    iterator = context.run(factory)
    try:
        while True:
            try:
                record = context.run(next, iterator)
            except StopIteration:
                return
            yield record
    finally:
        context.run(iterator.close)


def with_execution_policy(function):
    """Snapshot standalone upload policy before allocation."""

    @wraps(function)
    def call(*args, **kwargs):
        policy = current_policy(kwargs.get("io_backend", "posix"))
        token = _active.set(policy)
        try:
            return function(*args, **kwargs)
        finally:
            _active.reset(token)

    return call
