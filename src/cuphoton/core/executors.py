# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Lazy selection of the distributed runtime for component workloads."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .execution import ExecutionResult, WorkloadSpec


def run_workload(
    *,
    executor: str,
    prepare_workload: Callable[[int], WorkloadSpec],
    output_root: Path,
    run_id: str | None = None,
    **options: Any,
) -> ExecutionResult | None:
    """Run one component's workload through the selected external launcher."""

    if executor == "dragon":
        from .dragon import run_dragon_work_items as run
    elif executor == "mpi":
        from .mpi import run_mpi_work_items as run
    else:
        raise ValueError("distributed executor must be dragon or mpi")
    return run(
        prepare_workload=prepare_workload,
        output_root=output_root,
        run_id=run_id,
        **options,
    )
