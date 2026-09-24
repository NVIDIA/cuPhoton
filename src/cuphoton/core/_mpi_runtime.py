# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Optional MPI startup and launcher contracts for component adapters."""

from __future__ import annotations

import re
import socket
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from .bulk import error_payload

_ALLOCATED_DEVICES = "CUPHOTON_ALLOCATED_CUDA_VISIBLE_DEVICES"
_UNSIGNED = re.compile(r"[0-9]{1,18}")


@dataclass(frozen=True)
class _RankContext:
    rank: int
    local_rank: int
    world_size: int
    host: str
    launcher: str
    launch_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _MPIAPI:
    MPI: Any
    comm: Any
    mpi4py_version: str | None
    library_version: str | None


def _mpi_failure_consensus(
    comm: Any,
    phase: str,
    error: Mapping[str, Any] | None,
    *,
    root_failure_handler: Callable[[Mapping[str, str]], None] | None = None,
) -> None:
    rank = int(comm.Get_rank())
    size = int(comm.Get_size())
    gathered = comm.gather(
        {"rank": rank, "error": dict(error) if error else None},
        root=0,
    )
    decision = None
    if rank == 0:
        failures: list[str] = []
        if not isinstance(gathered, Sequence) or len(gathered) != size:
            failures.append("invalid consensus result")
        else:
            for expected_rank, item in enumerate(gathered):
                if not isinstance(item, Mapping):
                    failures.append(
                        f"rank {expected_rank}: invalid consensus result"
                    )
                    continue
                if _integer(item.get("rank")) != expected_rank:
                    failures.append(
                        f"rank {expected_rank}: invalid rank identity"
                    )
                item_error = item.get("error")
                if item_error is None:
                    continue
                if not _valid_error(item_error):
                    failures.append(
                        f"rank {expected_rank}: invalid error payload"
                    )
                    continue
                failures.append(
                    f"rank {expected_rank}: {item_error['type']}: "
                    f"{item_error['message']}"
                )
        message = (
            f"{phase} failed: " + "; ".join(failures) if failures else None
        )
        if message is not None and root_failure_handler is not None:
            try:
                root_failure_handler(
                    {"type": "RuntimeError", "message": message}
                )
            except Exception as exc:
                persistence_error = error_payload(exc)
                message += (
                    "; failed to persist terminal setup evidence: "
                    f"{persistence_error['type']}: "
                    f"{persistence_error['message']}"
                )
        decision = {"error": message}
    decision = comm.bcast(decision, root=0)
    if not isinstance(decision, Mapping):
        raise RuntimeError(f"{phase} decision was not a mapping")
    if decision.get("error"):
        raise RuntimeError(str(decision["error"]))


def _valid_error(value: Any) -> bool:
    return isinstance(value, Mapping) and all(
        isinstance(value.get(field), str) and bool(value[field])
        for field in ("type", "message")
    )


def _mpi_context(
    mpi: Any, comm: Any, environ: Mapping[str, str]
) -> _RankContext:
    rank = _runtime_integer(comm.Get_rank(), "MPI rank")
    size = _runtime_integer(comm.Get_size(), "MPI world size")
    _validate_topology(rank, size)
    context = None
    failure = None
    try:
        try:
            shared = comm.Split_type(mpi.COMM_TYPE_SHARED, key=rank)
            try:
                local_rank = _runtime_integer(
                    shared.Get_rank(), "MPI local rank"
                )
            finally:
                shared.Free()
        except Exception as exc:
            raise RuntimeError("cannot determine MPI local rank") from exc
        launcher = _launcher_context(environ)
        if launcher is not None and launcher[1:] != (rank, size):
            raise RuntimeError("launcher topology disagrees with mpi4py")
        launcher_local_rank = _local_rank(
            environ, launcher[0] if launcher is not None else None
        )
        if (
            launcher_local_rank is not None
            and launcher_local_rank != local_rank
        ):
            raise RuntimeError(
                "launcher local rank disagrees with MPI shared communicator"
            )
        context = _RankContext(
            rank,
            local_rank,
            size,
            socket.gethostname(),
            launcher[0] if launcher is not None else "mpi",
        )
    except Exception as exc:
        failure = error_payload(exc)
    if failure is None and context is None:
        failure = error_payload(
            RuntimeError("rank context validation produced no context")
        )
    _mpi_failure_consensus(comm, "MPI rank-context validation", failure)
    if context is None:
        raise RuntimeError("MPI rank-context consensus returned no context")
    return context


def _launcher_context(
    environ: Mapping[str, str],
) -> tuple[str, int, int] | None:
    candidates = (
        ("openmpi", "OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE"),
        ("pmi", "PMI_RANK", "PMI_SIZE"),
        ("slurm", "SLURM_PROCID", "SLURM_NTASKS"),
    )
    for name, rank_key, size_key in candidates:
        present = rank_key in environ, size_key in environ
        if all(present):
            return (
                name,
                _environment_integer(environ, rank_key),
                _environment_integer(environ, size_key),
            )
        if any(present):
            raise RuntimeError(
                f"{name} launcher rank and size must appear together"
            )
    return None


def _local_rank(
    environ: Mapping[str, str], launcher: str | None = None
) -> int | None:
    keys_by_launcher = {
        "openmpi": ("OMPI_COMM_WORLD_LOCAL_RANK",),
        "pmi": ("PMI_LOCAL_RANK", "MPI_LOCALRANKID", "PMIX_LOCAL_RANK"),
        "slurm": (
            "SLURM_LOCALID",
            "PMIX_LOCAL_RANK",
            "MPI_LOCALRANKID",
            "PMI_LOCAL_RANK",
        ),
    }
    keys = keys_by_launcher.get(
        launcher,
        (
            "OMPI_COMM_WORLD_LOCAL_RANK",
            "PMIX_LOCAL_RANK",
            "MPI_LOCALRANKID",
            "PMI_LOCAL_RANK",
            "SLURM_LOCALID",
        ),
    )
    values = [
        (key, _environment_integer(environ, key))
        for key in keys
        if key in environ
    ]
    if launcher == "pmi" and not values and "SLURM_LOCALID" in environ:
        slurm_keys = ("SLURM_PROCID", "SLURM_NTASKS", "SLURM_LOCALID")
        topology_keys = ("PMI_RANK", "PMI_SIZE", *slurm_keys)
        if all(key in environ for key in topology_keys):
            pmi_topology = (
                _environment_integer(environ, "PMI_RANK"),
                _environment_integer(environ, "PMI_SIZE"),
            )
            slurm_topology = (
                _environment_integer(environ, "SLURM_PROCID"),
                _environment_integer(environ, "SLURM_NTASKS"),
            )
            if pmi_topology != slurm_topology:
                raise RuntimeError(
                    "PMI and Slurm launcher rank topology disagree"
                )
            values = [
                (
                    "SLURM_LOCALID",
                    _environment_integer(environ, "SLURM_LOCALID"),
                )
            ]
    if not values:
        return None
    if any(value != values[0][1] for _, value in values[1:]):
        family = launcher or "MPI"
        raise RuntimeError(f"{family} launcher local-rank variables disagree")
    return values[0][1]


def _prebound_visibility(environ: Mapping[str, str]) -> str:
    raw_visibility = environ.get("CUDA_VISIBLE_DEVICES")
    if raw_visibility is None:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES is unset; bind one device per MPI rank "
            "with cuphoton-openmpi-rank-exec or scheduler GPU binding"
        )
    tokens = _visible_tokens(raw_visibility)
    if len(tokens) != 1:
        raise RuntimeError(
            "MPI rank must see one CUDA device before mpi4py import; use "
            "cuphoton-openmpi-rank-exec or scheduler GPU binding"
        )
    allocated_raw = environ.get(_ALLOCATED_DEVICES)
    if allocated_raw is None:
        return tokens[0]
    allocated = _visible_tokens(allocated_raw)
    local_rank = _local_rank(environ, "openmpi")
    if local_rank is None or local_rank >= len(allocated):
        raise RuntimeError(
            f"{_ALLOCATED_DEVICES} requires a valid Open MPI local rank; "
            "unset it when launching without cuphoton-openmpi-rank-exec"
        )
    if allocated[local_rank] != tokens[0]:
        raise RuntimeError("pre-bound CUDA device does not match allocation")
    return tokens[0]


def _visible_tokens(raw: Any) -> tuple[str, ...]:
    if not isinstance(raw, str) or not raw.strip() or raw.strip() == "-1":
        raise RuntimeError("CUDA_VISIBLE_DEVICES exposes no devices")
    tokens = tuple(token.strip() for token in raw.split(","))
    if any(not token or token.startswith("-") for token in tokens) or len(
        tokens
    ) != len(set(tokens)):
        raise RuntimeError("CUDA_VISIBLE_DEVICES is invalid")
    return tokens


def _environment_integer(environ: Mapping[str, str], key: str) -> int:
    raw = environ[key]
    if not isinstance(raw, str) or not _UNSIGNED.fullmatch(raw):
        raise RuntimeError(f"launcher variable {key} must be unsigned")
    return int(raw)


def _runtime_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"{field} must be non-negative")
    return value


def _validate_topology(rank: int, size: int) -> None:
    if size <= 0 or rank >= size:
        raise RuntimeError("invalid launcher rank topology")


def _integer(value: Any) -> int | None:
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool)
        else None
    )


def _load_mpi_api() -> _MPIAPI:
    try:
        from mpi4py import MPI
    except (ImportError, OSError, RuntimeError) as exc:
        raise RuntimeError(
            "MPI collective aggregation requires mpi4py built against the "
            "allocation's MPI runtime"
        ) from exc
    try:
        library_version = _normalize_mpi_library_version(
            MPI.Get_library_version()
        )
    except Exception:  # pragma: no cover - implementation-specific
        library_version = None
    return _MPIAPI(
        MPI,
        MPI.COMM_WORLD,
        _distribution_version("mpi4py"),
        library_version,
    )


def _normalize_mpi_library_version(value: Any) -> str | None:
    if value is None:
        return None
    return str(value).rstrip("\x00 \t\r\n")


def _distribution_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None
