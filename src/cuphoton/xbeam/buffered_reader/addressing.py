# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


"""Physical source identities and validated timeline state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import RawBufferedRequestState
from .timeline import PhysicalAllocationTimeline


@dataclass(frozen=True, slots=True)
class LatchedRawSourceSpan:
    """Consecutive FFTs reading consecutive raw physical samples."""

    first_fft: int
    stop_fft: int
    physical_sample_start: int


@dataclass(frozen=True, slots=True)
class UnresolvedRawSourceSpan:
    first_fft: int
    stop_fft: int
    reason: str


@dataclass(frozen=True, slots=True)
class LatchedRawSourceCompilation:
    spans: tuple[LatchedRawSourceSpan, ...]
    unresolved: tuple[UnresolvedRawSourceSpan, ...]

    @property
    def complete(self) -> bool:
        return not self.unresolved


@dataclass(frozen=True, slots=True)
class LatchedRawSourceState:
    """Validated physical history cached once before compiling batches."""

    timeline: PhysicalAllocationTimeline
    request_state: RawBufferedRequestState
    unsupported_reason: str | None
    raw_source_stop: int
    recorded_source_stop: int | None = None

    @classmethod
    def from_timeline(
        cls,
        timeline: PhysicalAllocationTimeline,
        request_state: RawBufferedRequestState,
    ) -> LatchedRawSourceState:
        if not isinstance(timeline, PhysicalAllocationTimeline):
            raise TypeError("timeline must be a PhysicalAllocationTimeline")
        if not isinstance(request_state, RawBufferedRequestState):
            raise TypeError("request state must be a RawBufferedRequestState")
        state = request_state
        if (
            timeline._allocation_frames != state.allocation_frames
            or timeline.anchors.shape != state.anchor_samples.shape
        ):
            raise ValueError("raw state and physical timeline must agree")
        # Python integer subtraction preserves arbitrary signed origins.
        if any(
            int(sample)
            != (int(anchor) - state.timestamp_origin_frame)
            * state.samples_per_frame
            for sample, anchor in zip(
                state.anchor_samples, timeline.anchors, strict=True
            )
        ):
            raise ValueError("raw state and physical timeline anchors differ")
        reason = None
        if np.any(timeline.residuals < 0):
            reason = "backward raw clock residual is unsupported"
        elif np.any(timeline.physical_gaps):
            reason = "noncontiguous raw input is unsupported"
        return cls(
            timeline,
            state,
            reason,
            int(timeline.physical_stops[-1]) * state.samples_per_frame,
        )
