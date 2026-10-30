# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import FrozenInstanceError
from decimal import Decimal
from types import SimpleNamespace

import pytest

from cuphoton.xbeam.adapters.radio.models import MJDTime, Scan
from cuphoton.xbeam.errors import ConfigurationError
from cuphoton.xbeam.profiles import ANALYTIC_V1, INTEROP_REAL_USB_V1
from cuphoton.xbeam.windowing import (
    CorrelationWindow,
    select_complete_timing_groups,
)


def _scan(
    identifier: str = "scan-0",
    *,
    start: str = "0",
    duration: str = "3",
) -> Scan:
    return Scan(
        identifier=identifier,
        start_offset_s=Decimal(start),
        duration_s=Decimal(duration),
        source_id="source",
        configuration_id="config",
        station_models=(),
    )


def _job(
    scan: Scan,
    *,
    execution: str = "3",
    integration: str = "1",
    start_seconds: str = "0",
):
    return SimpleNamespace(
        scans=(scan,),
        execution_duration_s=Decimal(execution),
        integration_duration_s=Decimal(integration),
        start_time=MJDTime(60000, Decimal(start_seconds)),
    )


def _groups(scan: Scan, *indices: int):
    return [(index, scan, [index]) for index in indices]


def test_correlation_window_is_immutable():
    window = CorrelationWindow("scan-0", 1, 2)

    with pytest.raises(FrozenInstanceError):
        window.integration_count = 3


@pytest.mark.parametrize("scan_id", ["", "  "])
def test_correlation_window_rejects_empty_scan_id(scan_id):
    with pytest.raises(ValueError, match="scan_id must not be empty"):
        CorrelationWindow(scan_id, 0, 1)


@pytest.mark.parametrize("first_integration", [-1, -100])
def test_correlation_window_rejects_negative_first_integration(
    first_integration,
):
    with pytest.raises(ValueError, match="must be nonnegative"):
        CorrelationWindow("scan-0", first_integration, 1)


@pytest.mark.parametrize("integration_count", [0, -1])
def test_correlation_window_rejects_nonpositive_count(integration_count):
    with pytest.raises(ValueError, match="must be positive"):
        CorrelationWindow("scan-0", 0, integration_count)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("scan_id", None),
        ("first_integration", 1.5),
        ("first_integration", True),
        ("integration_count", 1.5),
        ("integration_count", True),
    ],
)
def test_correlation_window_rejects_invalid_field_types(field, value):
    values = {
        "scan_id": "scan-0",
        "first_integration": 0,
        "integration_count": 1,
    }
    values[field] = value

    with pytest.raises(TypeError, match=field):
        CorrelationWindow(**values)


def test_selects_requested_complete_integrations():
    scan = _scan()
    groups = _groups(scan, 0, 1, 2)

    selected = select_complete_timing_groups(
        _job(scan),
        ANALYTIC_V1,
        groups,
        CorrelationWindow("scan-0", 1, 2),
    )

    assert selected == groups[1:]


def test_accepts_integration_ending_at_scan_and_execution_boundary():
    scan = _scan(duration="1")
    groups = _groups(scan, 0)

    selected = select_complete_timing_groups(
        _job(scan, execution="1"),
        ANALYTIC_V1,
        groups,
        CorrelationWindow("scan-0", 0, 1),
    )

    assert selected == groups


def test_rejects_unknown_scan():
    scan = _scan()

    with pytest.raises(ConfigurationError, match="unknown scan identifier"):
        select_complete_timing_groups(
            _job(scan),
            ANALYTIC_V1,
            _groups(scan, 0),
            CorrelationWindow("missing", 0, 1),
        )


def test_rejects_missing_timing_group():
    scan = _scan()

    with pytest.raises(ConfigurationError, match="no timing group 1"):
        select_complete_timing_groups(
            _job(scan),
            ANALYTIC_V1,
            _groups(scan, 0, 2),
            CorrelationWindow("scan-0", 0, 2),
        )


def test_rejects_trailing_integration_clipped_by_scan():
    scan = _scan(duration="1.5")

    with pytest.raises(ConfigurationError, match="not wholly covered"):
        select_complete_timing_groups(
            _job(scan),
            ANALYTIC_V1,
            _groups(scan, 0, 1),
            CorrelationWindow("scan-0", 1, 1),
        )


def test_profile_integer_second_execution_end_limits_window():
    scan = _scan(duration="1")
    job = _job(
        scan,
        execution="1",
        integration="0.5",
        start_seconds="0.5",
    )
    groups = _groups(scan, 0, 1)
    window = CorrelationWindow("scan-0", 1, 1)

    assert select_complete_timing_groups(
        job, ANALYTIC_V1, groups, window
    ) == [groups[1]]
    with pytest.raises(ConfigurationError, match="not wholly covered"):
        select_complete_timing_groups(
            job, INTEROP_REAL_USB_V1, groups, window
        )


def test_scan_start_before_execution_uses_zero_as_integration_origin():
    scan = _scan(start="-0.25", duration="1.25")
    groups = _groups(scan, 0)

    selected = select_complete_timing_groups(
        _job(scan, execution="1"),
        ANALYTIC_V1,
        groups,
        CorrelationWindow("scan-0", 0, 1),
    )

    assert selected == groups


def test_rejects_duplicate_group_index_for_selected_scan():
    scan = _scan()
    groups = _groups(scan, 0)

    with pytest.raises(ConfigurationError, match="duplicate timing group 0"):
        select_complete_timing_groups(
            _job(scan),
            ANALYTIC_V1,
            groups + groups,
            CorrelationWindow("scan-0", 0, 1),
        )
