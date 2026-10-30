# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace

import numpy as np
import pytest

from cuphoton.xbeam.adapters.radio.compute_plan import _block_starts
from cuphoton.xbeam.missing_timestamp_recovery import (
    FFTEligibilityCheckpoint,
    GlobalFFTEligibilityPlan,
    UncalibratedRecoveryError,
    build_global_fft_eligibility_plan,
    select_fft_authoritative_validity,
    select_fft_eligibility,
    select_fft_source_replay,
    select_fft_source_segments,
    select_fft_source_validity,
    select_fft_weight_exclusions,
    select_fft_weight_overrides,
)


@pytest.mark.parametrize("authoritative_global", [False, True])
def test_bulk_selectors_preserve_sparse_boundaries_and_duplicate_requests(
    authoritative_global,
):
    plan = GlobalFFTEligibilityPlan(
        missing_timestamps=(),
        eligible_ffts=(),
        checkpoint_stops=(100,),
        checkpoints=(FFTEligibilityCheckpoint(0, 0, 100, 100, 0),),
        calibration_identifier="selector-regression",
        timeline_fft_count=100,
        ineligible_intervals=((2, 5), (10, 12), (90, 100)),
        source_replay_intervals=((2, 8, -32), (10, 20, 16)),
        source_replay_events=((4, -64, 7), (11, -8, 31)),
        source_validity_events=((4, 3, 17), (11, 0, 0)),
        source_authoritative_validity_events=(4, 11),
        source_validity_is_authoritative=authoritative_global,
        source_weight_exclusion_events=((4, 9), (11, 32)),
        source_weight_override_events=((4, 0), (11, 23)),
        source_segment_events=((4, ((0, 7, -32), (7, 32, 0))),),
    )
    # Reverse order, repeated events, and both sides of right-open boundaries.
    indices = (99, 90, 89, 20, 19, 12, 11, 10, 8, 7, 5, 4, 4, 3, 2, 1, 0)
    arguments = dict(
        scan_origin_s=0.125,
        fft_seconds=0.000064,
        block_starts_s=tuple(0.125 + index * 0.000064 for index in indices),
    )
    assert select_fft_eligibility(plan, **arguments) == tuple(
        plan.is_eligible(index) for index in indices
    )
    assert select_fft_authoritative_validity(plan, **arguments) == tuple(
        plan.source_validity_is_authoritative_at(index) for index in indices
    )
    replay = tuple(plan.source_replay(index) for index in indices)
    assert select_fft_source_replay(plan, **arguments, fft_samples=32) == (
        tuple(item[0] for item in replay),
        tuple(item[1] for item in replay),
    )
    validity = tuple(
        plan.source_validity(index) or (0, 32) for index in indices
    )
    assert select_fft_source_validity(plan, **arguments, fft_samples=32) == (
        tuple(item[0] for item in validity),
        tuple(item[1] for item in validity),
    )
    assert select_fft_weight_exclusions(
        plan, **arguments, fft_samples=32
    ) == tuple(plan.source_weight_exclusion(index) for index in indices)
    assert select_fft_weight_overrides(
        plan, **arguments, fft_samples=32
    ) == tuple(plan.source_weight_override(index) for index in indices)
    assert select_fft_source_segments(
        plan, **arguments, fft_samples=32
    ) == tuple(plan.source_segments(index) for index in indices)


@pytest.mark.parametrize(
    "selector",
    [
        select_fft_source_replay,
        select_fft_source_validity,
        select_fft_weight_exclusions,
        select_fft_weight_overrides,
        select_fft_source_segments,
    ],
)
def test_bulk_selectors_reject_invalid_grid_even_without_matching_events(
    selector,
):
    plan = GlobalFFTEligibilityPlan(
        missing_timestamps=(),
        eligible_ffts=(),
        checkpoint_stops=(10,),
        checkpoints=(FFTEligibilityCheckpoint(0, 0, 10, 10, 0),),
        calibration_identifier="selector-grid",
        timeline_fft_count=10,
        source_replay_intervals=((0, 10, -1),),
    )
    with pytest.raises(ValueError, match="calibrated FFT timeline"):
        selector(
            plan,
            scan_origin_s=0.0,
            fft_seconds=1.0,
            block_starts_s=(10.0,),
            fft_samples=32,
        )


def test_sparse_source_offset_bounds_are_limited_to_selected_window():
    plan = GlobalFFTEligibilityPlan(
        missing_timestamps=(),
        eligible_ffts=(),
        checkpoint_stops=(200,),
        checkpoints=(FFTEligibilityCheckpoint(0, 0, 200, 200, 0),),
        calibration_identifier="test",
        timeline_fft_count=200,
        source_replay_intervals=((0, 100, -8), (100, 200, -8000)),
    )

    assert plan.source_offset_bounds == (-8000, -8)
    assert plan.source_offset_bounds_for(20, 80) == (-8, -8)
    assert plan.source_offset_bounds_for(120, 180) == (-8000, -8000)


def test_sparse_source_offset_bounds_include_unshifted_gaps():
    plan = GlobalFFTEligibilityPlan(
        missing_timestamps=(),
        eligible_ffts=(),
        checkpoint_stops=(200,),
        checkpoints=(FFTEligibilityCheckpoint(0, 0, 200, 200, 0),),
        calibration_identifier="test",
        timeline_fft_count=200,
        source_replay_intervals=((0, 80, -8), (120, 200, -8000)),
    )

    assert plan.source_offset_bounds_for(60, 140) == (-8000, 0)


def test_sparse_source_offset_bounds_ignore_fully_ineligible_gaps():
    plan = GlobalFFTEligibilityPlan(
        missing_timestamps=(),
        eligible_ffts=(),
        checkpoint_stops=(200,),
        checkpoints=(FFTEligibilityCheckpoint(0, 0, 200, 200, 0),),
        calibration_identifier="test",
        timeline_fft_count=200,
        ineligible_intervals=((80, 120),),
        source_replay_intervals=((0, 80, -8), (120, 200, -8000)),
    )

    assert plan.source_offset_bounds_for(60, 140) == (-8000, -8)


def test_sparse_source_offset_bounds_include_split_replay_origin():
    plan = GlobalFFTEligibilityPlan(
        missing_timestamps=(),
        eligible_ffts=(),
        checkpoint_stops=(10,),
        checkpoints=(FFTEligibilityCheckpoint(0, 0, 10, 10, 0),),
        calibration_identifier="test",
        timeline_fft_count=10,
        source_replay_intervals=((0, 10, -8),),
        source_replay_events=((5, -8, 4),),
    )

    assert plan.source_offset_bounds_for(4, 5) == (-8, -8)
    assert plan.source_offset_bounds_for(5, 6) == (-8, 0)


def test_physical_frame_presentation_allows_reused_out_of_order_sources():
    intervals = (
        (0, 20, 21, 100),
        (0, 10, 12, 101),
        (0, 11, 12, 103),
    )

    plan = GlobalFFTEligibilityPlan(
        missing_timestamps=(),
        eligible_ffts=(),
        checkpoint_stops=(1,),
        checkpoints=(FFTEligibilityCheckpoint(0, 0, 1, 1, 0),),
        calibration_identifier="test",
        timeline_fft_count=1,
        source_physical_frame_intervals=intervals,
    )

    assert plan.source_physical_frame_intervals == intervals


def test_physical_frame_presentation_requires_disjoint_destinations():
    with pytest.raises(ValueError, match="destination ordered"):
        GlobalFFTEligibilityPlan(
            missing_timestamps=(),
            eligible_ffts=(),
            checkpoint_stops=(1,),
            checkpoints=(FFTEligibilityCheckpoint(0, 0, 1, 1, 0),),
            calibration_identifier="test",
            timeline_fft_count=1,
            source_physical_frame_intervals=(
                (0, 20, 22, 100),
                (0, 10, 11, 101),
            ),
        )


def test_sparse_piecewise_source_segments_are_selected_exactly():
    segments = ((0, 7, 0), (11, 19, -32), (19, 32, -64))
    plan = GlobalFFTEligibilityPlan(
        missing_timestamps=(),
        eligible_ffts=(),
        checkpoint_stops=(10,),
        checkpoints=(FFTEligibilityCheckpoint(0, 0, 10, 10, 0),),
        calibration_identifier="test",
        timeline_fft_count=10,
        source_segment_events=((5, segments),),
    )

    assert plan.source_offset_bounds_for(5, 6) == (-64, 0)
    assert select_fft_source_segments(
        plan,
        scan_origin_s=0.0,
        fft_seconds=1.0,
        block_starts_s=(4.0, 5.0, 6.0),
        fft_samples=32,
    ) == (None, segments, None)


def _losses(plan):
    return tuple(item.ineligible_fft_count for item in plan.checkpoints)


def test_missing_events_without_exact_calibration_fail_closed():
    with pytest.raises(UncalibratedRecoveryError, match="exact calibration"):
        build_global_fft_eligibility_plan(
            fft_count=7,
            checkpoint_stops=(7,),
            missing_timestamps=(2,),
        )


def test_missing_free_plan_needs_no_recovery_calibration():
    plan = build_global_fft_eligibility_plan(
        fft_count=7,
        checkpoint_stops=(3, 7),
        missing_timestamps=(),
    )
    assert plan.eligible_ffts == (True,) * 7
    assert _losses(plan) == (0, 0)


def test_select_fft_eligibility_rejects_off_grid_and_out_of_range():
    plan = build_global_fft_eligibility_plan(
        fft_count=7,
        checkpoint_stops=(7,),
        missing_timestamps=(),
    )
    with pytest.raises(ValueError, match="calibrated FFT timeline"):
        select_fft_eligibility(
            plan,
            scan_origin_s=0.0,
            fft_seconds=0.25,
            block_starts_s=(0.1,),
        )
    with pytest.raises(ValueError, match="outside"):
        select_fft_eligibility(
            plan,
            scan_origin_s=0.0,
            fft_seconds=0.25,
            block_starts_s=(1.75,),
        )


@pytest.mark.parametrize(
    "origin,fft_samples", [(32768, 262144), (43200, 262144), (8192, 32768)]
)
def test_selectors_accept_engine_grid_at_late_scan_origins(
    origin, fft_samples
):
    rate = 4096000000
    fft_seconds = fft_samples / rate
    starts = _block_starts(
        SimpleNamespace(subintegration_duration_s=Decimal("0.0032")),
        rate,
        fft_samples,
        SimpleNamespace(start_offset_s=Decimal(origin)),
        range(100),
    )
    plan = GlobalFFTEligibilityPlan(
        (), (), (), (), None, timeline_fft_count=starts.size
    )
    assert all(
        select_fft_eligibility(
            plan,
            scan_origin_s=float(origin),
            fft_seconds=fft_seconds,
            block_starts_s=tuple(starts),
        )
    )
    with pytest.raises(ValueError, match="does not lie"):
        select_fft_eligibility(
            plan,
            scan_origin_s=float(origin),
            fft_seconds=fft_seconds,
            block_starts_s=(starts[0] + fft_seconds / 4,),
        )


def test_selector_rejects_clock_too_coarse_to_resolve_fft_grid():
    plan = GlobalFFTEligibilityPlan(
        (), (), (), (), None, timeline_fft_count=8
    )
    with pytest.raises(ValueError, match="clock precision"):
        select_fft_eligibility(
            plan,
            scan_origin_s=float(2**40),
            fft_seconds=64e-6,
            block_starts_s=(float(2**40),),
        )


def test_compiled_scope_rejects_unknown_prefix_without_restricting_identity():
    identity = GlobalFFTEligibilityPlan(
        (), (), (), (), None, timeline_fft_count=100
    )
    scoped = replace(identity, known_fft_scope=(50, 75))
    for index in (49, 75):
        assert identity.is_eligible(index)
        with pytest.raises(ValueError, match="known FFT scope"):
            scoped.is_eligible(index)
        with pytest.raises(ValueError, match="known FFT scope"):
            select_fft_source_replay(
                scoped,
                scan_origin_s=0.0,
                fft_seconds=1.0,
                block_starts_s=(float(index),),
                fft_samples=32,
            )
    np.testing.assert_equal(scoped.window(50, 75).eligible_ffts, (True,) * 25)
