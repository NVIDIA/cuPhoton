# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Compatibility policies for behavior not universal in a file format."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class HalfSampleTie(StrEnum):
    TO_EVEN = "to_even"
    AWAY_FROM_ZERO = "away_from_zero"


class PartialValidity(StrEnum):
    SIMULTANEOUS_VALID = "simultaneous_valid"
    ZERO_FILLED = "zero_filled"
    REJECT = "reject"


@dataclass(frozen=True, slots=True)
class CompatibilityProfile:
    name: str
    interoperability_frozen: bool
    half_sample_tie: HalfSampleTie
    duplicate_policy: str
    partial_validity: PartialValidity
    integration_left_closed: bool = True
    emit_trailing_partial: bool = True
    omit_zero_weight: bool = True
    omit_real_nyquist: bool = True
    execution_end_on_integer_second: bool = False
    validate_vdif_station_code: bool = True
    missing_timestamp_recovery: str | None = None


ANALYTIC_V1 = CompatibilityProfile(
    name="analytic-v1",
    interoperability_frozen=False,
    half_sample_tie=HalfSampleTie.TO_EVEN,
    duplicate_policy="first_valid",
    partial_validity=PartialValidity.ZERO_FILLED,
)

INTEROP_REAL_USB_V1 = CompatibilityProfile(
    name="interop-real-usb-v1",
    interoperability_frozen=True,
    half_sample_tie=HalfSampleTie.TO_EVEN,
    duplicate_policy="first_valid",
    partial_validity=PartialValidity.ZERO_FILLED,
    # The text interoperability profile's EXECUTE TIME ends at the integer
    # second obtained by adding it to the integral part of START SECONDS.
    execution_end_on_integer_second=True,
    # DiFX TELESCOPE NAME is an antenna label, not necessarily the two-byte
    # station identifier carried by VDIF frames.
    validate_vdif_station_code=False,
)

INTEROP_REAL_USB_BUFFERED_WINDOW_REPLAY_V1 = CompatibilityProfile(
    name="interop-real-usb-buffered-window-replay-v1",
    interoperability_frozen=True,
    half_sample_tie=HalfSampleTie.TO_EVEN,
    duplicate_policy="first_valid",
    partial_validity=PartialValidity.ZERO_FILLED,
    execution_end_on_integer_second=True,
    validate_vdif_station_code=False,
    missing_timestamp_recovery="buffered-window-replay-v1",
)

PROFILES = {
    p.name: p
    for p in (
        ANALYTIC_V1,
        INTEROP_REAL_USB_V1,
        INTEROP_REAL_USB_BUFFERED_WINDOW_REPLAY_V1,
    )
}


def get_profile(name: str) -> CompatibilityProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        from .errors import UnsupportedProfileError

        raise UnsupportedProfileError(
            f"compatibility profile {name!r} is not frozen or supported"
        ) from exc
