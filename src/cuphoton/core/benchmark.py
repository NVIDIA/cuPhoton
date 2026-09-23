# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Round identities and reports for repeated work in persistent executors.

Executors own synchronization and workload-specific artifact validation. This
module has no optional runtime dependencies and never discards a slow round.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from statistics import median
from typing import Any


@dataclass(frozen=True)
class BenchmarkRound:
    """A warmup or measured pass over the complete input manifest."""

    phase: str
    index: int

    def __post_init__(self) -> None:
        if self.phase not in {"warmup", "measure"}:
            raise ValueError("round phase must be warmup or measure")
        _validate_count(self.index, "round index", minimum=0)

    @property
    def round_id(self) -> str:
        return f"{self.phase}-{self.index:04d}"

    def to_payload(self) -> dict[str, Any]:
        return {"round_id": self.round_id, **asdict(self)}

    def run_id(self, parent_run_id: str) -> str:
        """Bind ordinary per-item artifacts to this invocation and round."""

        digest = hashlib.sha256(parent_run_id.encode("utf-8")).hexdigest()[
            :32
        ]
        return f"benchmark-{digest}-{self.round_id}"


@dataclass(frozen=True)
class BenchmarkOptions:
    """Opt-in repetitions; None keeps an executor's ordinary run."""

    warmup_rounds: int = 0
    measure_rounds: int = 1

    def __post_init__(self) -> None:
        _validate_count(self.warmup_rounds, "warmup_rounds", minimum=0)
        _validate_count(self.measure_rounds, "measure_rounds", minimum=1)

    def rounds(self) -> tuple[BenchmarkRound, ...]:
        return tuple(
            BenchmarkRound(phase, index)
            for phase, count in (
                ("warmup", self.warmup_rounds),
                ("measure", self.measure_rounds),
            )
            for index in range(count)
        )

    def to_payload(self) -> dict[str, int]:
        return asdict(self)


def build_benchmark_report(
    options: BenchmarkOptions,
    rounds: Sequence[Mapping[str, Any]],
    *,
    errors: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Keep raw evidence and summarize only a complete successful invocation.

    An adapter must audit scientific artifacts and worker provenance before
    marking a round successful. Durations use the coordinator's monotonic
    clock; worker durations use each worker's local clock, never subtracted
    timestamps.
    """

    receipts = [dict(receipt) for receipt in rounds]
    problems = [dict(error) for error in errors]
    expected = options.rounds()
    if len(receipts) != len(expected):
        problems.append(
            {"phase": "round_audit", "message": "incomplete round sequence"}
        )
    for index, receipt in enumerate(receipts):
        identity = (
            expected[index].to_payload() if index < len(expected) else {}
        )
        if not identity or any(
            type(receipt.get(key)) is not type(value)
            or receipt.get(key) != value
            for key, value in identity.items()
        ):
            problems.append(
                {"phase": "round_audit", "message": f"invalid round {index}"}
            )
        if receipt.get("status") != "success":
            problems.append(
                {"phase": "round_audit", "message": f"failed round {index}"}
            )
        for field in ("batch_wall_sec", "worker_wall_max_sec"):
            value = receipt.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                problems.append(
                    {
                        "phase": "round_audit",
                        "message": f"invalid {field} in round {index}",
                    }
                )
    measured = (
        [
            receipt["batch_wall_sec"]
            for receipt in receipts
            if receipt.get("phase") == "measure"
        ]
        if not problems
        else []
    )
    return {
        "schema": "cuphoton.core.benchmark/v1",
        "options": options.to_payload(),
        "status": "failed" if problems else "success",
        "rounds": receipts,
        "errors": problems,
        "measured_batch_wall_sec": (
            {
                "min": min(measured),
                "median": median(measured),
                "max": max(measured),
            }
            if measured
            else None
        ),
        "timing_definitions": {
            "batch_wall_sec": (
                "Coordinator time from before round release through receipt "
                "of all worker completions, including ordinary input reads, "
                "numerical work, output writes and record publication; "
                "excludes subsequent coordinator artifact audits."
            ),
            "worker_wall_max_sec": (
                "Maximum worker-local elapsed time for the round. Its "
                "difference from batch_wall_sec is a mixed scheduling, "
                "communication and publication remainder, not pure transport."
            ),
        },
    }


def _validate_count(value: int, name: str, *, minimum: int) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
