# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral signal-processing primitives for the correlator."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .core.products import (
    accumulate_products as accumulate_products,
)
from .core.products import (
    average_channels as average_channels,
)
from .core.products import (
    cross_power as cross_power,
)
from .core.products import (
    evaluate_polynomial as evaluate_polynomial,
)
from .core.products import (
    product_validity as product_validity,
)
from .core.products import (
    reduce_cycles as reduce_cycles,
)

TWO_PI = 2.0 * math.pi


def fringe_phasor(
    delay_s: Any,
    time_s: Any,
    reference_frequency_hz: float,
    lo_offset_hz: float,
    xp: Any,
) -> Any:
    cycles = -xp.float64(reference_frequency_hz) * xp.asarray(
        delay_s, dtype=xp.float64
    )
    cycles += xp.float64(lo_offset_hz) * xp.asarray(time_s, dtype=xp.float64)
    phase = xp.float64(TWO_PI) * reduce_cycles(cycles, xp)
    return xp.exp(xp.asarray(1j, dtype=xp.complex128) * phase)


def apply_fringe_rotation(
    samples: Any,
    delay_s: Any,
    time_s: Any,
    reference_frequency_hz: float,
    lo_offset_hz: float,
    xp: Any,
) -> Any:
    return xp.asarray(samples) * fringe_phasor(
        delay_s, time_s, reference_frequency_hz, lo_offset_hz, xp
    )


@dataclass(frozen=True, slots=True)
class Channelized:
    spectrum: Any
    sky_frequencies_hz: Any


def complex_lsb_channels(
    transformed: Any, sample_rate_hz: float, xp: Any
) -> tuple[Any, Any]:
    """Map the last FFT axis to complex LSB channels and frequency labels."""
    transformed = xp.asarray(transformed)
    if transformed.ndim == 0 or transformed.shape[-1] == 0:
        raise ValueError(
            "complex LSB channelization needs at least one FFT bin"
        )
    count = transformed.shape[-1]
    if count == 1:
        channels = transformed
    else:
        dtype = transformed.dtype
        if dtype.kind == "b":
            dtype = xp.conjugate(xp.empty((0,), dtype=dtype)).dtype
        channels = xp.empty(transformed.shape, dtype=dtype)
        channels[..., 0] = transformed[..., 0]
        xp.conjugate(transformed[..., :0:-1], out=channels[..., 1:])

    frequencies = xp.arange(-count, 0, dtype=xp.float64)
    frequencies *= xp.float64(sample_rate_hz / count)
    return channels, frequencies


def channelize(
    samples: Any,
    *,
    sample_rate_hz: float,
    sky_reference_hz: float,
    sideband: str,
    sampling_kind: str,
    xp: Any,
) -> Channelized:
    """Transform blocks and sort returned bins by increasing sky frequency."""

    data = xp.asarray(samples)
    if data.ndim == 1:
        data = data[xp.newaxis, :]
    direction = 1.0 if sideband.upper() in {"U", "USB"} else -1.0
    if sideband.upper() not in {"U", "USB", "L", "LSB"}:
        raise ValueError(f"unsupported sideband {sideband!r}")
    if sampling_kind.lower() == "real":
        transformed = xp.fft.rfft(data, axis=-1)
        if data.shape[-1] % 2 == 0:
            transformed = transformed[..., :-1]
        baseband = xp.arange(transformed.shape[-1], dtype=xp.float64)
        baseband *= xp.float64(sample_rate_hz / data.shape[-1])
    elif sampling_kind.lower() == "complex" and direction > 0:
        transformed = xp.fft.fft(data, axis=-1)
        baseband = xp.arange(data.shape[-1], dtype=xp.float64)
        baseband *= xp.float64(sample_rate_hz / data.shape[-1])
    elif sampling_kind.lower() == "complex":
        transformed = xp.fft.fft(data, axis=-1)
        spectrum, baseband = complex_lsb_channels(
            transformed, sample_rate_hz, xp
        )
        sky = xp.float64(sky_reference_hz) + baseband
        return Channelized(spectrum, sky)
    else:
        raise ValueError(f"unsupported sampling kind {sampling_kind!r}")
    sky = xp.float64(sky_reference_hz) + xp.float64(direction) * baseband
    order = xp.argsort(sky)
    return Channelized(transformed[..., order], sky[order])


def apply_fractional_delay(
    spectrum: Any,
    baseband_frequencies_hz: Any,
    residual_delay_s: Any,
    xp: Any,
) -> Any:
    frequencies = xp.asarray(baseband_frequencies_hz, dtype=xp.float64)
    delay = xp.asarray(residual_delay_s, dtype=xp.float64)
    if delay.ndim:
        delay = delay[..., xp.newaxis]
    phase = xp.float64(TWO_PI) * reduce_cycles(delay * frequencies, xp)
    correction = xp.exp(xp.asarray(1j, dtype=xp.complex128) * phase)
    return xp.asarray(spectrum) * correction


def normalize_visibility(
    accumulator: Any, *, fft_length: int, block_weight: float, xp: Any
) -> Any:
    if block_weight <= 0:
        raise ValueError("cannot normalize a zero-weight visibility")
    # The forward FFT is unnormalized.  Dividing power by N and by the
    # contributing block weight preserves input voltage-squared units.
    scale = xp.float64(fft_length) * xp.float64(block_weight)
    return xp.asarray(accumulator) / scale


def integration_index(
    midpoint_s: float, start_s: float, duration_s: float
) -> int:
    if duration_s <= 0:
        raise ValueError("integration duration must be positive")
    return math.floor((midpoint_s - start_s) / duration_s)


def mjd_seconds(day: int, seconds_of_day: float) -> tuple[int, float]:
    """Normalize MJD seconds across either midnight direction."""

    day_delta = math.floor(seconds_of_day / 86400.0)
    day += day_delta
    seconds_of_day -= day_delta * 86400.0
    if seconds_of_day >= 86400.0:  # protect against one-ulp overflow
        day += 1
        seconds_of_day = 0.0
    return day, seconds_of_day


def group_midpoints(
    midpoints_s: Iterable[float], start_s: float, duration_s: float
) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = {}
    for block_index, midpoint in enumerate(midpoints_s):
        index = integration_index(midpoint, start_s, duration_s)
        groups.setdefault(index, []).append(block_index)
    return groups
