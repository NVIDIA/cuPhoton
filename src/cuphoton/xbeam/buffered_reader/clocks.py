# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Buffered reader clocks: input-derived compatibility planning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._checks import _integer, _readonly

_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1


@dataclass(frozen=True, slots=True)
class LeadingRawHeadContract:
    """Explicit clock quantum; supported reader geometry checked by caller."""

    sample_rate: int
    timestamp_quantum_nanoseconds: int = 1

    def __post_init__(self) -> None:
        for name in ("sample_rate", "timestamp_quantum_nanoseconds"):
            object.__setattr__(
                self, name, _integer(getattr(self, name), name, minimum=1)
            )


@dataclass(frozen=True, slots=True)
class LeadingRawHeadInputs:
    """Actual request output clock, endpoint delay, and first-centre delay.

    Each field is a bounded one-dimensional request vector. Output samples
    use the raw state timestamp origin. Seconds are independently evaluated
    model values, never reconstructed from rounded desired source starts.
    """

    request_output_samples: np.ndarray
    endpoint_delays_seconds: np.ndarray
    first_centre_delays_seconds: np.ndarray

    def __post_init__(self) -> None:
        shape = None
        for name, dtype in (
            ("request_output_samples", np.int64),
            ("endpoint_delays_seconds", np.float64),
            ("first_centre_delays_seconds", np.float64),
        ):
            array = getattr(self, name)
            if (
                not isinstance(array, np.ndarray)
                or array.ndim != 1
                or array.dtype != np.dtype(dtype)
            ):
                raise TypeError(f"{name} must be a {np.dtype(dtype)} vector")
            if shape is None:
                shape = array.shape
            elif array.shape != shape:
                raise ValueError("head input vectors must have equal shape")
            if dtype == np.float64 and not np.all(np.isfinite(array)):
                raise ValueError("head model delays must be finite")
            object.__setattr__(self, name, _readonly(array))


@dataclass(frozen=True, slots=True)
class LeadingRawHeadCoordinates:
    head_sample_starts: np.ndarray
    source_sample_starts: np.ndarray
    nearest_samples: np.ndarray
    missing: np.ndarray


@dataclass(frozen=True, slots=True)
class RawWindowLowerBounds:
    """Signed read offsets and negative-window rejection only.

    Passing this lower bound does not establish the positive window end,
    payload validity, publication ownership or reported exposure.
    """

    nearest_samples: np.ndarray
    missing: np.ndarray


@dataclass(frozen=True, slots=True)
class RawWindowBounds:
    """Clock bounds against an explicitly supplied returned sample extent.

    The extent belongs to the aligned window head, not the physical file or
    an allocation's spare capacity. These bounds do not establish payload
    validity, source ownership, publication, or reported exposure.
    """

    nearest_samples: np.ndarray
    before_window: np.ndarray
    after_window: np.ndarray
    missing: np.ndarray


@dataclass(frozen=True, slots=True)
class RawWindowPayloadSpans:
    """Available half-open sample spans relative to each FFT's start.

    Samples outside these spans are unavailable from the declared decoder
    window. This does not decide whether the reader admits the FFT or how
    its exposure is reported.
    """

    first_samples: np.ndarray
    stop_samples: np.ndarray


def _checked_int64(value: int, name: str) -> int:
    if not _INT64_MIN <= value <= _INT64_MAX:
        raise OverflowError(name + " exceeds int64")
    return value


def _nearest_window_sample(
    output: int, head: int, delay_samples: float
) -> int:
    # Preserve the supplied float64 model product as an exact binary ratio.
    # Keep sample origins out of floating point; implement signed truncation,
    # since Python // alone rounds negative values toward minus infinity.
    numerator, denominator = delay_samples.as_integer_ratio()
    shifted = ((output - head) * 2 + 1) * denominator - 2 * numerator
    divisor = 2 * denominator
    return shifted // divisor if shifted >= 0 else -((-shifted) // divisor)


def _nearest_window_samples(
    output: np.ndarray, head: np.ndarray, delays: np.ndarray
) -> np.ndarray:
    """Vectorize signed rounding, keeping exceptional integer spans exact."""
    whole = np.floor(delays).astype(np.int64)
    # Keep both timestamp origins in integer arithmetic. Signed subtraction
    # wraps only on exceptional rows, detected before using their results.
    difference = output - head
    overflow = ((output ^ head) & (output ^ difference)) < 0
    nearest = difference - whole
    overflow |= ((difference ^ whole) & (difference ^ nearest)) < 0
    safe = ~overflow
    # For nonintegral float64 delays, floor(delay) + 1/2 is representable.
    # Integral delays need the explicit equality test at large magnitudes.
    # Fractional remainders need no subtraction (which could round tiny
    # negative delays to a remainder of one).
    midpoint = whole.astype(np.float64) + 0.5
    integral = delays == whole
    positive_fraction = integral | (delays < midpoint)
    negative_fraction = ~integral & (delays > midpoint)
    nearest[safe & (nearest < 0) & positive_fraction] += 1
    nearest[safe & (nearest > 0) & negative_fraction] -= 1
    for index in np.flatnonzero(overflow):
        nearest[index] = _checked_int64(
            _nearest_window_sample(
                int(output[index]), int(head[index]), float(delays[index])
            ),
            "window sample offset",
        )
    return nearest


def compile_raw_window_lower_bounds(
    output_samples: np.ndarray,
    aligned_head_samples: np.ndarray,
    centre_delays_seconds: np.ndarray,
    *,
    sample_rate: int,
) -> RawWindowLowerBounds:
    """Check actual resumed FFTs against their returned aligned window heads.

    Unlike frame-head selection, this accepts an explicit cursor origin.
    The model must be evaluated at each actual FFT's discrete sample centre,
    including FFTs after an initial cursor skip. Compute
    trunc(output - head - delay*rate + 1/2), rejecting offsets below -1.
    Endpoint delays and storage replay translations cannot substitute for
    these logical clock operands. No execution profile is activated here.
    """
    rate = _integer(sample_rate, "sample rate", minimum=1)
    shape = None
    for array, dtype, name in (
        (output_samples, np.int64, "output samples"),
        (aligned_head_samples, np.int64, "aligned head samples"),
        (centre_delays_seconds, np.float64, "centre delays"),
    ):
        if (
            not isinstance(array, np.ndarray)
            or array.ndim != 1
            or array.dtype != np.dtype(dtype)
        ):
            raise TypeError(f"{name} must be a {np.dtype(dtype)} vector")
        if shape is None:
            shape = array.shape
        elif array.shape != shape:
            raise ValueError("window clock vectors must have equal shape")
    if not np.all(np.isfinite(centre_delays_seconds)):
        raise ValueError("window model delays must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        delay_samples = centre_delays_seconds * rate
    if not np.all(np.isfinite(delay_samples)):
        raise OverflowError("window delay products must remain finite")
    if np.any(np.abs(delay_samples) >= 2**53):
        raise OverflowError("window delay exceeds exact sample-phase range")
    nearest = _nearest_window_samples(
        output_samples, aligned_head_samples, delay_samples
    )
    return RawWindowLowerBounds(_readonly(nearest), _readonly(nearest < -1))


def compile_raw_window_bounds(
    output_samples: np.ndarray,
    aligned_head_samples: np.ndarray,
    centre_delays_seconds: np.ndarray,
    returned_extent_samples: np.ndarray,
    *,
    sample_rate: int,
    fft_samples: int,
) -> RawWindowBounds:
    """Check a complete FFT against its actual aligned returned window.

    Use the signed clock offset from ``compile_raw_window_lower_bounds``.
    A positive bound passes when ``nearest + fft_samples <= returned_extent``.
    Compare by subtraction to avoid overflowing at large sample offsets.
    Callers must supply the returned extent in samples for every FFT; neither
    framed byte counts nor extra allocated bytes are converted here. No
    execution profile is activated by this pure coordinate operation.
    """
    length = _checked_int64(
        _integer(fft_samples, "FFT samples", minimum=1), "FFT samples"
    )
    lower = compile_raw_window_lower_bounds(
        output_samples,
        aligned_head_samples,
        centre_delays_seconds,
        sample_rate=sample_rate,
    )
    if (
        not isinstance(returned_extent_samples, np.ndarray)
        or returned_extent_samples.ndim != 1
        or returned_extent_samples.dtype != np.dtype(np.int64)
    ):
        raise TypeError("returned extent samples must be an int64 vector")
    if returned_extent_samples.shape != lower.nearest_samples.shape:
        raise ValueError("returned extent shape must match window clocks")
    if np.any(returned_extent_samples < 0):
        raise ValueError("returned extent samples must be nonnegative")
    after = lower.nearest_samples > returned_extent_samples - length
    return RawWindowBounds(
        lower.nearest_samples,
        lower.missing,
        _readonly(after),
        _readonly(lower.missing | after),
    )


def compile_raw_window_payload_spans(
    nearest_samples: np.ndarray,
    decoded_extent_samples: np.ndarray,
    *,
    fft_samples: int,
) -> RawWindowPayloadSpans:
    """Intersect each FFT with an explicit decoded window, without overflow.

    The window is [0, decoded_extent) relative to its aligned head. Returned
    spans use FFT-relative coordinates in [0, fft_samples]. An empty span
    contains no samples; no complete-FFT discard or weight rule is inferred.
    """
    length = _checked_int64(
        _integer(fft_samples, "FFT samples", minimum=1), "FFT samples"
    )
    shape = None
    for array, name in (
        (nearest_samples, "nearest samples"),
        (decoded_extent_samples, "decoded extent samples"),
    ):
        if (
            not isinstance(array, np.ndarray)
            or array.ndim != 1
            or array.dtype != np.dtype(np.int64)
        ):
            raise TypeError(f"{name} must be an int64 vector")
        if shape is None:
            shape = array.shape
        elif array.shape != shape:
            raise ValueError("decoded extents must match sample offsets")
    if np.any(decoded_extent_samples < 0):
        raise ValueError("decoded extent samples must be nonnegative")
    assert shape is not None
    first = np.zeros(shape, dtype=np.int64)
    first[nearest_samples <= -length] = length
    left = (nearest_samples < 0) & (nearest_samples > -length)
    first[left] = -nearest_samples[left]
    stop = np.full(shape, length, dtype=np.int64)
    stop[nearest_samples >= decoded_extent_samples] = 0
    right = (nearest_samples < decoded_extent_samples) & (
        nearest_samples > decoded_extent_samples - length
    )
    stop[right] = decoded_extent_samples[right] - nearest_samples[right]
    stop = np.maximum(stop, first)
    return RawWindowPayloadSpans(_readonly(first), _readonly(stop))


def compile_leading_raw_head_coordinates(
    inputs: LeadingRawHeadInputs,
    contract: LeadingRawHeadContract,
    *,
    samples_per_frame: int,
) -> LeadingRawHeadCoordinates:
    """Compile bounded head addresses without converting origins to float.

    Endpoint and centre products are float64 model-delay operations. Their
    origins and frame division remain exact Python integers; add-half and
    signed truncation preserve low origin bits even beyond2**53. Raw ownership
    and absence/eligibility interpretation are separate caller decisions.
    """
    if not isinstance(inputs, LeadingRawHeadInputs):
        raise TypeError("head inputs must be LeadingRawHeadInputs")
    if not isinstance(contract, LeadingRawHeadContract):
        raise TypeError("head contract must be LeadingRawHeadContract")
    samples = _integer(samples_per_frame, "samples per frame", minimum=1)
    quantum = contract.timestamp_quantum_nanoseconds
    with np.errstate(over="ignore", invalid="ignore"):
        endpoint_units = inputs.endpoint_delays_seconds * (
            1_000_000_000 / quantum
        )
        centre_samples = (
            inputs.first_centre_delays_seconds * contract.sample_rate
        )
    if not np.all(np.isfinite(endpoint_units)) or not np.all(
        np.isfinite(centre_samples)
    ):
        raise OverflowError("head model delay products must remain finite")
    # A float delay sample product beyond this bound cannot retain a sample
    # phase. Large integer OUTPUT origins remain fully supported.
    if np.any(np.abs(centre_samples) >= 2**53):
        raise OverflowError("centre delay exceeds exact sample-phase range")
    heads, sources, offsets = [], [], []
    for output, endpoint, centre in zip(
        inputs.request_output_samples,
        endpoint_units,
        centre_samples,
        strict=True,
    ):
        output = int(output)
        delay_ns = int(float(endpoint)) * quantum
        _checked_int64(delay_ns, "truncated endpoint delay nanoseconds")
        head = (
            (output * 1_000_000_000 - delay_ns * contract.sample_rate)
            // (1_000_000_000 * samples)
        ) * samples
        _checked_int64(head, "request frame head")
        nearest = _nearest_window_sample(output, head, float(centre))
        _checked_int64(nearest, "request leading sample offset")
        source = head + max(nearest, 0)
        _checked_int64(source, "request leading source sample")
        heads.append(head)
        sources.append(source)
        offsets.append(nearest)
    head_array = np.asarray(heads, dtype=np.int64)
    source_array = np.asarray(sources, dtype=np.int64)
    offset_array = np.asarray(offsets, dtype=np.int64)
    return LeadingRawHeadCoordinates(
        _readonly(head_array),
        _readonly(source_array),
        _readonly(offset_array),
        _readonly(offset_array < -1),
    )


def compile_leading_raw_timing_offsets(
    inputs: LeadingRawHeadInputs,
    contract: LeadingRawHeadContract,
    coarse_samples: np.ndarray,
    *,
    samples_per_frame: int,
) -> np.ndarray:
    """Return request-head clock translations, independently of storage.

    The ordinary source clock is output minus coarse delay. A published head
    clock differing by delta requires residual delay (coarse-delta)/rate minus
    centre delay. Physical replay addresses, allocation ownership and holes
    do not appear in that identity. The caller selects actual request heads;
    body rows retain their ordinary residual and rejected heads stay rejected.

    This does not select a production profile or alter exposure accounting.
    Caller-supplied coarse delays must match the ordinary first-centre model.
    Output origins remain integer samples, including those beyond 2**53.
    """
    if not isinstance(inputs, LeadingRawHeadInputs):
        raise TypeError("head inputs must be LeadingRawHeadInputs")
    if not isinstance(contract, LeadingRawHeadContract):
        raise TypeError("head contract must be LeadingRawHeadContract")
    if (
        not isinstance(coarse_samples, np.ndarray)
        or coarse_samples.dtype != np.dtype(np.int64)
        or coarse_samples.shape != inputs.request_output_samples.shape
    ):
        raise TypeError("coarse samples must be a matching int64 vector")
    coordinates = compile_leading_raw_head_coordinates(
        inputs, contract, samples_per_frame=samples_per_frame
    )
    rounded = np.rint(
        inputs.first_centre_delays_seconds * contract.sample_rate
    ).astype(np.int64)
    if np.any(rounded != coarse_samples):
        raise ValueError("coarse samples differ from first-centre model")
    values = [
        _checked_int64(
            int(source) - (int(output) - int(coarse)),
            "head timing offset",
        )
        for output, coarse, source in zip(
            inputs.request_output_samples,
            coarse_samples,
            coordinates.source_sample_starts,
            strict=True,
        )
    ]
    return _readonly(np.asarray(values, dtype=np.int64))
