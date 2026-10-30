# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Invented measurements demonstrating deterministic monitoring."""

from pathlib import Path

from .dashboard import Dashboard
from .monitor import Monitor
from .records import Finding, Measurement, Sample


def quickstart(directory: str | Path) -> Monitor:
    """Create eight samples with one signal and one quality flag."""
    directory = Path(directory)
    monitor = Monitor(directory / "state", run_id="invented-measurements")
    times = [0.0, 0.7, 2.0, 2.5, 4.8, 6.0, 9.1, 10.0]
    for index, observed in enumerate(times):
        value = (
            None
            if index == 6
            else 1.8
            if index == 4
            else 1.0 + 0.03 * (index % 3)
        )
        findings: tuple[Finding, ...] = ()
        if index == 4:
            findings = (
                Finding("level_change", "Invented level exceeds 1.5 units."),
            )
        if index == 6:
            findings = (
                Finding(
                    "missing_input",
                    "No measurement was available.",
                    "data_quality",
                ),
            )
        monitor.publish(
            Sample(
                f"sample-{index}",
                index,
                {
                    "level": Measurement(
                        value, "arb", coverage=0.0 if value is None else 1.0
                    )
                },
                observed_start=observed,
                findings=findings,
            )
        )
    Dashboard(monitor, title="Invented measurement monitor").export(
        directory / "report.html"
    )
    return monitor
