# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from cuphoton.core.cli import CommandError
from cuphoton.core.cli.executor import ExecutorOptions


@pytest.mark.parametrize(
    ("executor", "option"),
    [
        ("local", "warmup_rounds"),
        ("local", "measure_rounds"),
        ("local", "max_workers"),
        ("local", "rank_setup_timeout_sec"),
        ("mpi", "max_workers"),
        ("mpi", "worker_timeout_sec"),
        ("mpi", "result_timeout_sec"),
        ("dragon", "rank_setup_timeout_sec"),
    ],
)
def test_rejects_flags_which_selected_executor_cannot_honor(executor, option):
    command = ExecutorOptions()
    command.executor = executor
    setattr(command, option, 1)
    with pytest.raises(CommandError, match=option.replace("_", "-")):
        command.executor_options()


def test_local_default_does_not_enable_distributed_execution():
    command = ExecutorOptions()
    command.executor = "local"
    assert command.executor_options() == {}


@pytest.mark.parametrize("executor", ["dragon", "mpi"])
def test_round_controls_are_opt_in(executor):
    command = ExecutorOptions()
    command.executor = executor
    assert command.executor_options() == {"benchmark": None}
    command.warmup_rounds = 0
    assert command.executor_options()["benchmark"].to_payload() == {
        "warmup_rounds": 0,
        "measure_rounds": 1,
    }
