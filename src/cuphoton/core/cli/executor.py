# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Common opt-in distributed execution options for component commands."""

from __future__ import annotations

from ..benchmark import BenchmarkOptions
from .command import CommandError
from .invariants import (
    FloatInvariant,
    NonNegativeIntegerInvariant,
    PositiveIntegerInvariant,
    SetInvariant,
)


class ExecutorOptions:
    """Keep local command defaults and validate runtime-specific options."""

    executor = None
    max_workers = None
    worker_timeout_sec = None
    result_timeout_sec = None
    rank_setup_timeout_sec = None
    warmup_rounds = None
    measure_rounds = None

    class ExecutorArg(SetInvariant):
        _arg = "--executor"
        _help = "Execution runtime. [default: %default]"
        _set = {"local", "dragon", "mpi"}
        _default = "local"

    class MaxWorkersArg(PositiveIntegerInvariant):
        _arg = "--max-workers"
        _help = "Dragon-only maximum number of GPU workers."
        _default = None

    class WorkerTimeoutSecArg(FloatInvariant):
        _arg = "--worker-timeout-sec"
        _help = "Dragon worker lifetime across all rounds. [default: 3600]"
        _default = None
        _min = 0.001

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
        _help = "Persistent-worker warmup passes, with outputs retained."
        _default = None

    class MeasureRoundsArg(PositiveIntegerInvariant):
        _arg = "--measure-rounds"
        _help = "Measured passes in persistent workers. [default: 1]"
        _default = None

    def executor_options(self) -> dict:
        """Reject ignored flags before loading a numerical runtime."""

        dragon = {
            "max_workers": self.max_workers,
            "worker_timeout_sec": self.worker_timeout_sec,
            "result_timeout_sec": self.result_timeout_sec,
        }
        mpi = {"rank_setup_timeout_sec": self.rank_setup_timeout_sec}
        rounds = {
            "warmup_rounds": self.warmup_rounds,
            "measure_rounds": self.measure_rounds,
        }
        if self.executor == "local":
            invalid = {**dragon, **mpi, **rounds}
        elif self.executor == "dragon":
            invalid = mpi
        elif self.executor == "mpi":
            invalid = dragon
        else:
            raise CommandError("executor must be local, dragon, or mpi")
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
        options = {
            name: value
            for name, value in (
                dragon if self.executor == "dragon" else mpi
            ).items()
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
