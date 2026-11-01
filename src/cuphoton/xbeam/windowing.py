# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Selection of complete integration windows for bounded correlation runs."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from decimal import Decimal

from cuphoton.xbeam.adapters.radio.models import Job, Scan

from .errors import ConfigurationError
from .profiles import CompatibilityProfile

TimingGroup = tuple[int, Scan, Sequence[int]]


@dataclass(frozen=True, slots=True)
class CorrelationWindow:
    """A contiguous range of complete integrations within one scan."""

    scan_id: str
    first_integration: int
    integration_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.scan_id, str):
            raise TypeError("scan_id must be a string")
        if not self.scan_id.strip():
            raise ValueError("scan_id must not be empty")
        if not isinstance(self.first_integration, int) or isinstance(
            self.first_integration, bool
        ):
            raise TypeError("first_integration must be an integer")
        if self.first_integration < 0:
            raise ValueError("first_integration must be nonnegative")
        if not isinstance(self.integration_count, int) or isinstance(
            self.integration_count, bool
        ):
            raise TypeError("integration_count must be an integer")
        if self.integration_count <= 0:
            raise ValueError("integration_count must be positive")


def _effective_execution_end_s(
    job: Job, profile: CompatibilityProfile
) -> Decimal:
    execution_end_s = job.execution_duration_s
    if profile.execution_end_on_integer_second:
        execution_end_s -= job.start_time.seconds_of_day % Decimal(1)
    return execution_end_s


def select_complete_timing_groups(
    job: Job,
    profile: CompatibilityProfile,
    timing_groups: Iterable[TimingGroup],
    window: CorrelationWindow,
) -> list[TimingGroup]:
    """Select an exact range of complete timing groups.

    ``timing_groups`` is the output of the engine's timing-group scheduler.
    Unlike the legacy unwindowed path, a bounded window may not include a
    trailing integration selected only by subintegration midpoint.  Every
    requested integration must exist and its full right-open interval must be
    covered by both the scan and the profile-adjusted execution interval.
    """

    scan = next(
        (item for item in job.scans if item.identifier == window.scan_id),
        None,
    )
    if scan is None:
        raise ConfigurationError(
            "correlate.window",
            "scan_id",
            f"unknown scan identifier {window.scan_id!r}",
        )

    requested_indices = range(
        window.first_integration,
        window.first_integration + window.integration_count,
    )
    groups_by_index: dict[int, TimingGroup] = {}
    for timing_group in timing_groups:
        integration_index, group_scan, _subintegrations = timing_group
        if group_scan.identifier != window.scan_id:
            continue
        if integration_index in groups_by_index:
            raise ConfigurationError(
                "correlate.window",
                "scan_id",
                f"scan {window.scan_id!r} has duplicate timing group "
                f"{integration_index}",
            )
        groups_by_index[integration_index] = timing_group

    integration_origin = max(Decimal(0), scan.start_offset_s)
    coverage_end_s = min(
        _effective_execution_end_s(job, profile),
        scan.start_offset_s + scan.duration_s,
    )

    selected: list[TimingGroup] = []
    for integration_index in requested_indices:
        group = groups_by_index.get(integration_index)
        if group is None:
            raise ConfigurationError(
                "correlate.window",
                "first_integration",
                f"scan {window.scan_id!r} has no timing group "
                f"{integration_index}",
            )
        integration_start_s = (
            integration_origin
            + Decimal(integration_index) * job.integration_duration_s
        )
        integration_end_s = integration_start_s + job.integration_duration_s
        if (
            integration_start_s < integration_origin
            or integration_end_s > coverage_end_s
        ):
            raise ConfigurationError(
                "correlate.window",
                "integration_count",
                f"integration {integration_index} in scan "
                f"{window.scan_id!r} is not wholly covered by the scan and "
                "effective execution interval",
            )
        selected.append(group)

    return selected


__all__ = [
    "CorrelationWindow",
    "TimingGroup",
    "select_complete_timing_groups",
]
