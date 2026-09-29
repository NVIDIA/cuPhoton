# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Common opt-in distributed execution options for component commands."""

from __future__ import annotations

from collections.abc import Mapping

from ..benchmark import BenchmarkOptions
from .command import CommandError
from .invariants import (
    FloatInvariant,
    NonNegativeIntegerInvariant,
    PositiveIntegerInvariant,
    SetInvariant,
    StringInvariant,
)


class ExecutorOptions:
    """Keep local command defaults and validate runtime-specific options."""

    executor: str | None = None
    max_workers: int | None = None
    worker_timeout_sec: float | None = None
    result_timeout_sec: float | None = None
    rank_setup_timeout_sec: float | None = None
    warmup_rounds: int | None = None
    measure_rounds: int | None = None
    workers_per_gpu: int | None = None
    mps_pipe_directory: str | None = None

    class ExecutorArg(SetInvariant):
        _arg = "--executor"
        _help = (
            "Execution runtime: local, dragon, or mpi. [default: %default]"
        )
        _set = {"local", "dragon", "mpi"}
        _default: str | None = "local"

    class MaxWorkersArg(PositiveIntegerInvariant):
        _arg = "--max-workers"
        _help = "Dragon-only maximum number of GPU workers."
        _default = None

    class WorkerTimeoutSecArg(FloatInvariant):
        _arg = "--worker-timeout-sec"
        _help = "Dragon or local GPU worker lifetime. [default: 3600]"
        _default = None
        _min = 0.001

    class WorkersPerGpuArg(PositiveIntegerInvariant):
        _arg = "--workers-per-gpu"
        _help = (
            "Maximum workers sharing each GPU with Dragon or a local "
            "pipeline executor. [default: 1]"
        )
        _default = None

    class MpsPipeDirectoryArg(StringInvariant):
        _arg = "--mps-pipe-directory"
        _help = (
            "Existing MPS v2 pipe directory for Dragon or process workers; "
            "require every worker to connect to that service."
        )
        _default = None

    class ResultTimeoutSecArg(FloatInvariant):
        _arg = "--result-timeout-sec"
        _help = "Dragon result and artifact grace period. [default: 60]"
        _default = None
        _min = 0.001

    class RankSetupTimeoutSecArg(FloatInvariant):
        _arg = "--rank-setup-timeout-sec"
        _help = "MPI shared-artifact visibility timeout. [default: 600]"
        _default = None
        _min = 0.001

    class WarmupRoundsArg(NonNegativeIntegerInvariant):
        _arg = "--warmup-rounds"
        _help = (
            "Opt into the benchmark round layout with this many warmup "
            "passes; outputs are retained. [default when benchmarking: 0]"
        )
        _default = None

    class MeasureRoundsArg(PositiveIntegerInvariant):
        _arg = "--measure-rounds"
        _help = (
            "Opt into the benchmark round layout with this many measured "
            "passes in persistent workers. [default when benchmarking: 1]"
        )
        _default = None

    def executor_options(self) -> dict:
        """Reject ignored flags before loading a numerical runtime."""

        dragon_only = {
            "max_workers": self.max_workers,
            "result_timeout_sec": self.result_timeout_sec,
        }
        sharing = {
            "workers_per_gpu": self.workers_per_gpu,
            "worker_timeout_sec": self.worker_timeout_sec,
        }
        mps = {"mps_pipe_directory": self.mps_pipe_directory}
        dragon = {**dragon_only, **sharing, **mps}
        mpi = {"rank_setup_timeout_sec": self.rank_setup_timeout_sec}
        rounds = {
            "warmup_rounds": self.warmup_rounds,
            "measure_rounds": self.measure_rounds,
        }
        invalid: Mapping[str, str | int | float | None]
        if self.executor == "local":
            invalid = {**dragon, **mpi, **rounds}
        elif self.executor == "dragon":
            invalid = mpi
        elif self.executor == "mpi":
            invalid = dragon
        elif self.executor == "processes":
            invalid = {**dragon_only, **mpi}
        elif self.executor == "threads":
            invalid = {**dragon_only, **mpi, **mps}
        else:
            raise CommandError(f"unsupported executor: {self.executor}")
        supplied = [
            "--" + name.replace("_", "-")
            for name, value in invalid.items()
            if value is not None
        ]
        if supplied:
            raise CommandError(
                f"{', '.join(supplied)} cannot be used with "
                f"--executor {self.executor}"
            )
        if self.executor == "local":
            return {}
        by_executor: dict[str, Mapping[str, str | int | float | None]] = {
            "dragon": dragon,
            "mpi": mpi,
            "processes": {**sharing, **mps},
            "threads": sharing,
        }
        runtime_options = by_executor[self.executor]
        options: dict[str, str | int | float | BenchmarkOptions | None] = {
            name: value
            for name, value in runtime_options.items()
            if value is not None
        }
        options["benchmark"] = (
            BenchmarkOptions(
                warmup_rounds=self.warmup_rounds or 0,
                measure_rounds=self.measure_rounds or 1,
            )
            if any(value is not None for value in rounds.values())
            else None
        )
        return options
