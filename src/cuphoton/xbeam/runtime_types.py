# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Runtime types implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from cuphoton.xbeam.adapters.radio.models import (
    Datastream,
    Frequency,
    RecordedBand,
)
from cuphoton.xbeam.adapters.vdif.recording import VdifPayloadPlan


@dataclass(slots=True)
class _BandSamples:
    datastream: Datastream
    band: RecordedBand
    frequency: Frequency
    samples: np.ndarray
    validity: np.ndarray
    start_sample: int
    sample_rate_hz: float
    device_samples: Any = None
    device_validity: Any = None
    validity_stride: int = 1
    device_payload: Any = None
    source_sample_count: int | None = None
    packed_samples_per_frame: int = 0
    packed_payload_bytes: int = 0
    packed_channel_count: int = 0
    packed_channel_index: int = 0
    payload_plan: VdifPayloadPlan | None = None
    payload_frame_validity: np.ndarray | None = None
    payload_progress: Any = None
    # Planned ingest reserves execution samples and delay margins.
    planned_sample_coverage: bool = True


@dataclass(slots=True)
class _StationBlocks:
    spectrum: Any
    weight: Any
    baseband_hz: Any
    reported_weight: Any | None = None


@dataclass(frozen=True, slots=True)
class _BlockSchedule:
    block_starts: np.ndarray
    group_starts: np.ndarray
    group_lengths: np.ndarray


@dataclass(frozen=True, slots=True)
class _GpuGroupChunk:
    scan_identifier: str
    first_group: int
    stop_group: int
