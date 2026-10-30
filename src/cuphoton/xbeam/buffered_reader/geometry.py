# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Buffered reader geometry: input-derived compatibility planning."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from fractions import Fraction
from math import isfinite
from numbers import Integral

import numpy as np

from cuphoton.xbeam.adapters.radio.models import (
    BufferedReaderGeometry,
    DelayCoordinateModel,
)

from ..errors import ConfigurationError
from ._checks import _integer
from .timeline import PhysicalAllocationTimeline


def reference_model_rate_allowance(
    selected_reference_models: Sequence[DelayCoordinateModel],
    *,
    reference_epoch_seconds: Decimal,
) -> float:
    """Extract float64 reference allowance from one explicit model group.

    The caller selects one reference group from one scan. No scan, epoch,
    telescope clock or interval-extremum selection is inferred. Only a
    strictly positive stored geometric constant contributes the absolute
    first derivative. Higher coefficients do not contribute. Independent
    linear/quadratic controls support this operand. Use
    ``declared_model_rate_allowance`` to combine explicitly supplied epochs
    from one scan. No decimal spelling normalization occurs.
    """
    if (
        not isinstance(reference_epoch_seconds, Decimal)
        or not reference_epoch_seconds.is_finite()
    ):
        raise ValueError("reference_epoch_seconds must be a finite Decimal")
    if not selected_reference_models:
        raise ValueError("selected reference model group must not be empty")
    stations = set()
    allowance = 0.0
    for model in selected_reference_models:
        if not isinstance(model, DelayCoordinateModel):
            raise TypeError("reference models must be DelayCoordinateModel")
        if model.epoch_offset_s != reference_epoch_seconds:
            raise ValueError("reference model group contains another epoch")
        if model.station_id in stations:
            raise ValueError("reference group contains duplicate station")
        stations.add(model.station_id)
        coefficients = model.delay_coefficients_s
        if not coefficients or any(
            type(c) not in (float, np.float64) or not isfinite(c)
            for c in coefficients
        ):
            raise ValueError(
                "model delay coefficients must be finite float64"
            )
        if coefficients[0] > 0 and len(coefficients) > 1:
            allowance = max(allowance, abs(float(coefficients[1])))
    return allowance


def declared_model_rate_allowance(
    reference_models: Sequence[DelayCoordinateModel],
) -> float:
    """Combine all declared reference epochs of one explicitly supplied scan.

    Future epochs outside execution still contribute their stored reference
    coefficients. Four independent two-epoch controls distinguish this from
    first-group or active-group selection. Telescope compensation, evaluation
    times and higher derivatives do not enter the sizing operand. Selection
    across multiple scans remains the caller's responsibility; this function
    does not infer global job policy or returned decoder-window contents.
    """
    if not reference_models:
        raise ValueError("declared reference models must not be empty")
    groups: dict[Decimal, list[DelayCoordinateModel]] = {}
    for model in reference_models:
        if not isinstance(model, DelayCoordinateModel):
            raise TypeError("reference models must be DelayCoordinateModel")
        epoch = model.epoch_offset_s
        if not isinstance(epoch, Decimal) or not epoch.is_finite():
            raise ValueError("reference model epoch must be a finite Decimal")
        groups.setdefault(epoch, []).append(model)
    return max(
        reference_model_rate_allowance(models, reference_epoch_seconds=epoch)
        for epoch, models in groups.items()
    )


@dataclass(frozen=True, slots=True)
class BufferedNormalFullExtent:
    """Explicit normal-full request contract, distinct from storage alone.

    Framed admission counts header bytes at the supplied packing ratio;
    decoded coverage counts payload samples. Admission can be rational and
    is never implicitly rounded. No extra-buffer, partial-read, initial-read
    or EOF rule is inferred. Construction asserts the caller's normal-full
    returned-request contract; allocation geometry alone does not prove it.
    All byte and sample extents must fit positive signed int64 coordinates.
    """

    frame_bytes: int
    payload_bytes: int
    request_frames: int
    read_buffer_ratio: int
    samples_per_byte: Fraction | Decimal | int
    samples_per_frame: int = field(init=False)
    request_framed_bytes: int = field(init=False)
    request_payload_bytes: int = field(init=False)
    framed_admission_samples: Fraction = field(init=False)
    decoded_payload_samples: int = field(init=False)
    allocation_framed_bytes: int = field(init=False)

    def __post_init__(self) -> None:
        maximum = 2**63 - 1

        def checked(value: int | Fraction, name: str):
            if value <= 0 or value > maximum:
                raise OverflowError(f"{name} exceeds positive int64 extent")
            return value

        for name in (
            "frame_bytes",
            "payload_bytes",
            "request_frames",
            "read_buffer_ratio",
        ):
            value = _integer(getattr(self, name), name, minimum=1)
            checked(value, name)
            object.__setattr__(self, name, value)
        if self.payload_bytes > self.frame_bytes:
            raise ValueError("payload_bytes exceeds framed size")
        packing = self.samples_per_byte
        if isinstance(packing, bool) or not isinstance(
            packing, (Fraction, Decimal, Integral)
        ):
            raise TypeError("samples_per_byte must be exact rational")
        if isinstance(packing, Decimal) and not packing.is_finite():
            raise ValueError("samples_per_byte must be finite")
        packing = Fraction(packing)
        if packing <= 0:
            raise ValueError("samples_per_byte must be positive")
        object.__setattr__(self, "samples_per_byte", packing)
        frame_samples = self.payload_bytes * packing
        if frame_samples.denominator != 1:
            raise ValueError("frame payload must contain integral samples")
        values: dict[str, int | Fraction] = {
            "samples_per_frame": int(frame_samples),
            "request_framed_bytes": self.request_frames * self.frame_bytes,
            "request_payload_bytes": self.request_frames * self.payload_bytes,
            "framed_admission_samples": (
                self.request_frames * self.frame_bytes * packing
            ),
            "decoded_payload_samples": self.request_frames
            * int(frame_samples),
            "allocation_framed_bytes": (
                self.request_frames
                * self.read_buffer_ratio
                * self.frame_bytes
            ),
        }
        checked(
            self.request_frames * self.read_buffer_ratio, "allocation_frames"
        )
        for name, extent_value in values.items():
            object.__setattr__(self, name, checked(extent_value, name))


@dataclass(frozen=True)
class BufferedWindowGeometry:
    frames_per_fft: int
    ffts_per_request: int
    buffered_ffts: int
    ring_segments: int
    allocation_frames: int

    @classmethod
    def from_reader(
        cls,
        reader: BufferedReaderGeometry,
        *,
        samples_per_frame: int,
        fft_samples: int,
        sample_rate: int,
        subintegration_s: Decimal,
        reference_rate_allowance: float = 0.0,
    ) -> BufferedWindowGeometry:
        """Size framed storage with an explicit finite float64 allowance.

        Compute increment = float64(nominal frames) * allowance, then
        expanded = float64(nominal frames) + increment as separate rounded
        operations. Floor expanded, add two guards, then multiply by the
        integer read-buffer ratio. No fused or sum-before-product expression
        is equivalent at all frame thresholds. No decimal normalization or
        exact-rational reinterpretation is performed. Counts entering float64
        must be below 2**53; the guarded expanded count must also remain below
        this limit. Zero allowance preserves ordinary supported geometry.
        Allocation storage does not establish decoder returned extent or the
        independently allocated extra buffer.
        """

        def reject(reason: str) -> None:
            raise ConfigurationError("correlate", "buffered_reader", reason)

        for name in ("buffer_factor", "data_segments", "buffered_ffts"):
            value = getattr(reader, name)
            if type(value) is not int or value <= 0:
                reject(f"{name} must be a positive integer")
        for name, value in (
            ("samples_per_frame", samples_per_frame),
            ("fft_samples", fft_samples),
            ("sample_rate", sample_rate),
        ):
            if type(value) is not int or value <= 0:
                reject(f"{name} must be a positive integer")
        if not subintegration_s.is_finite() or subintegration_s <= 0:
            reject("subintegration must be finite and positive")
        if fft_samples % samples_per_frame:
            reject("FFT duration is not frame aligned")
        ffts = subintegration_s * sample_rate / fft_samples
        if ffts != ffts.to_integral_value():
            reject("request duration is not FFT aligned")
        if reader.buffer_factor % reader.data_segments:
            reject("buffer factor must be divisible by segment count")
        if type(reference_rate_allowance) not in (float, np.float64):
            reject("reference rate allowance must be float64")
        allowance = float(reference_rate_allowance)
        if not isfinite(allowance) or allowance < 0:
            reject("reference rate allowance must be finite and nonnegative")
        frames_per_fft = fft_samples // samples_per_frame
        ffts_per_request = int(ffts)
        nominal_frames = ffts_per_request * frames_per_fft
        if nominal_frames >= 2**53:
            reject("nominal frame count must be below 2**53")
        # Separate binary64 operations; do not fuse or reassociate them.
        increment = float(nominal_frames) * allowance
        expanded = float(nominal_frames) + increment
        if not isfinite(expanded) or expanded >= 2**53 - 2:
            reject("guarded frame count must be finite and below 2**53")
        request_frames = int(expanded) + 2
        return cls(
            frames_per_fft,
            ffts_per_request,
            reader.buffered_ffts,
            reader.data_segments,
            reader.buffer_factor // reader.data_segments * request_frames,
        )

    def framed_allocation_bytes(self, frame_bytes: int) -> int:
        """Framed storage size; not payload or returned decoder extent."""
        return self.allocation_frames * _integer(
            frame_bytes, "frame_bytes", minimum=1
        )

    def normal_full_extent(
        self,
        reader: BufferedReaderGeometry,
        *,
        frame_bytes: int,
        payload_bytes: int,
        samples_per_byte: Fraction | Decimal | int,
    ) -> BufferedNormalFullExtent:
        """Convert storage under an explicit normal-full return assertion.

        The reader ratio is an input, not inferred from header/payload sizes.
        This does not bind a raw read count or allocated extra buffer to the
        decoder extent. Caller must establish the normal-full request case.
        """
        if not isinstance(reader, BufferedReaderGeometry):
            raise TypeError("reader must be BufferedReaderGeometry")
        for name in ("buffer_factor", "data_segments", "buffered_ffts"):
            _integer(getattr(reader, name), name, minimum=1)
        if reader.data_segments != self.ring_segments:
            raise ValueError("reader segment count differs from geometry")
        if reader.buffered_ffts != self.buffered_ffts:
            raise ValueError("reader FFT buffering differs from geometry")
        ratio, remainder = divmod(reader.buffer_factor, reader.data_segments)
        if remainder:
            raise ValueError(
                "buffer factor must be divisible by segment count"
            )
        allocation = _integer(
            self.allocation_frames, "allocation_frames", minimum=1
        )
        frames, remainder = divmod(allocation, ratio)
        if remainder:
            raise ValueError("allocation is not a whole framed request")
        return BufferedNormalFullExtent(
            frame_bytes, payload_bytes, frames, ratio, samples_per_byte
        )


def frame_edge_underflows(
    fft_indices: np.ndarray,
    delay_seconds: np.ndarray,
    output_samples: np.ndarray,
    *,
    sample_rate: float,
    samples_per_frame: int,
    minimum_samples: float,
    maximum_samples: float,
) -> tuple[int, ...]:
    """Select frame-edge phases covered by an explicitly specified rule.

    Reducing the integer output coordinate before subtracting delay keeps
    fractional sample phases independent of large job/scan time offsets.
    The accepted phase interval belongs to the reader contract, not to a
    station or scan. This helper does not infer that interval from data.
    """

    indices = np.asarray(fft_indices, dtype=np.int64)
    source_phase = (
        np.remainder(output_samples, samples_per_frame).astype(np.float64)
        - delay_seconds * sample_rate
    )
    wrapped = np.remainder(source_phase, samples_per_frame)
    relative = np.where(wrapped == 0.0, 0.0, wrapped - samples_per_frame)
    selected = (minimum_samples <= relative) & (relative <= maximum_samples)
    return tuple(indices[selected].tolist())


def sample_offset_integer(value: float) -> int:
    """Add half a sample, then truncate toward zero."""

    return int(np.trunc(value + 0.5))


def divide_toward_zero(numerator: int, denominator: int) -> int:
    """Divide integers with truncation toward zero."""

    if denominator <= 0:
        raise ValueError("denominator must be positive")
    quotient = abs(numerator) // denominator
    return -quotient if numerator < 0 else quotient


@dataclass(frozen=True, slots=True)
class BufferedRequestGeometry:
    """Explicit byte/sample contract for one latched reader request.

    FFTs need not cover an integer number of source frames. Alignment and
    word size are reader properties supplied independently of stream names.
    Samples per byte is an exact per-lane ratio, including multichannel
    payloads that store one lane sample across multiple bytes.
    """

    samples_per_byte: Fraction | int
    fft_samples: int
    ffts_per_request: int
    alignment_bytes: int
    word_bytes: int

    def __post_init__(self) -> None:
        ratio = self.samples_per_byte
        if isinstance(ratio, Fraction):
            if ratio <= 0:
                raise ValueError("samples_per_byte must be positive")
        else:
            ratio = Fraction(_integer(ratio, "samples_per_byte", minimum=1))
        object.__setattr__(self, "samples_per_byte", ratio)
        for name in (
            "fft_samples",
            "ffts_per_request",
            "alignment_bytes",
            "word_bytes",
        ):
            object.__setattr__(
                self, name, _integer(getattr(self, name), name, minimum=1)
            )
        if (self.fft_samples * ratio.denominator) % ratio.numerator:
            raise ValueError("FFT samples must cover complete payload bytes")
        if self.alignment_bytes % self.word_bytes:
            raise ValueError("request alignment must cover complete words")
        if (self.alignment_bytes * ratio).denominator != 1:
            raise ValueError("request alignment must cover complete samples")

    @property
    def payload_bytes_per_fft(self) -> int:
        ratio = self.samples_per_byte
        return self.fft_samples * ratio.denominator // ratio.numerator


@dataclass(frozen=True, slots=True)
class BufferedRequestCursor:
    origin_bytes: int
    skipped_ffts: int
    decoder_delta_samples: int


def buffered_request_cursor(
    local_samples: int,
    delay_drift_seconds: float,
    geometry: BufferedRequestGeometry,
    *,
    sample_rate: float,
) -> BufferedRequestCursor | None:
    """Latch byte alignment and decoder displacement for one request.

    This operates on request-relative integer coordinates. The decoder
    displacement preserves samples dropped by byte/word alignment; it must
    not be confused with an eligibility mask or a source-validity decision.
    A request wholly preceding its returned window has no cursor.
    """
    if not isinstance(local_samples, Integral) or isinstance(
        local_samples, bool
    ):
        raise TypeError("local_samples must be an integer")
    local_samples = int(local_samples)
    if not isfinite(delay_drift_seconds):
        raise ValueError("delay drift must be finite")
    if not isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("sample rate must be finite and positive")
    if not isinstance(geometry, BufferedRequestGeometry):
        raise TypeError("geometry must be a BufferedRequestGeometry")
    ratio = geometry.samples_per_byte
    origin = divide_toward_zero(
        local_samples * ratio.denominator, ratio.numerator
    )
    origin -= origin % geometry.word_bytes
    skipped = 0
    while origin < 0 and skipped < geometry.ffts_per_request:
        origin += geometry.payload_bytes_per_fft
        skipped += 1
    if origin < 0 or skipped == geometry.ffts_per_request:
        return None
    drift = 0
    if skipped:
        drift = sample_offset_integer(
            skipped
            * delay_drift_seconds
            * sample_rate
            / geometry.ffts_per_request
        )
        correction = divide_toward_zero(
            drift * ratio.denominator, ratio.numerator
        )
        if origin - correction < 0:
            skipped += 1
            origin += geometry.payload_bytes_per_fft
            if skipped == geometry.ffts_per_request:
                return None
            drift = sample_offset_integer(
                skipped
                * delay_drift_seconds
                * sample_rate
                / geometry.ffts_per_request
            )
            origin -= divide_toward_zero(
                drift * ratio.denominator, ratio.numerator
            )
    origin -= origin % geometry.word_bytes
    origin -= origin % geometry.alignment_bytes
    return BufferedRequestCursor(
        origin,
        skipped,
        local_samples
        + skipped * geometry.fft_samples
        - drift
        - origin * ratio.numerator // ratio.denominator,
    )


@dataclass(frozen=True, slots=True)
class BufferedRequestCursors:
    """Batch request cursors; only rows marked valid may be consumed."""

    origin_bytes: np.ndarray
    skipped_ffts: np.ndarray
    decoder_delta_samples: np.ndarray
    valid: np.ndarray


def _cursor_multiply(
    values: np.ndarray, factor: int, name: str
) -> np.ndarray:
    limits = np.iinfo(np.int64)
    if values.size and (
        int(values.min()) < -(-int(limits.min) // factor)
        or int(values.max()) > int(limits.max) // factor
    ):
        raise OverflowError(f"request cursor int64 overflow in {name}")
    return values * factor


def _cursor_add(
    left: np.ndarray, right: np.ndarray | int, name: str
) -> np.ndarray:
    result = left + right
    if np.any(((left ^ result) & (right ^ result)) < 0):
        raise OverflowError(f"request cursor int64 overflow in {name}")
    return result


def _cursor_subtract(
    left: np.ndarray, right: np.ndarray | int, name: str
) -> np.ndarray:
    result = left - right
    if np.any(((left ^ right) & (left ^ result)) < 0):
        raise OverflowError(f"request cursor int64 overflow in {name}")
    return result


def _cursor_divide(values: np.ndarray, denominator: int) -> np.ndarray:
    quotient = values // denominator
    return quotient + ((values < 0) & (values % denominator != 0))


def _cursor_drift(
    skipped: np.ndarray,
    delay: np.ndarray,
    *,
    sample_rate: float,
    ffts_per_request: int,
) -> np.ndarray:
    # Preserve the scalar operation order and its add-half/truncate rule.
    with np.errstate(over="ignore", invalid="ignore"):
        rounded = np.trunc(
            skipped.astype(delay.dtype)
            * delay
            * sample_rate
            / ffts_per_request
            + 0.5
        )
    if np.any(
        ~np.isfinite(rounded) | (rounded < -(2**63)) | (rounded >= 2**63)
    ):
        raise OverflowError("request cursor int64 overflow in delay drift")
    return rounded.astype(np.int64)


def buffered_request_cursors(
    local_samples: np.ndarray,
    delay_drift_seconds: np.ndarray,
    geometry: BufferedRequestGeometry,
    *,
    sample_rate: float,
) -> BufferedRequestCursors:
    """Vectorized exact equivalent of ``buffered_request_cursor``.

    Origins use int64 arithmetic exclusively; drift uses floating time
    arithmetic with the scalar rounding rule. Integer intermediates and
    drift casts fail on overflow rather than wrap. Invalid rows are zeroed
    and must be ignored. Empty batches retain their one-dimensional shapes.
    """
    if not isinstance(geometry, BufferedRequestGeometry):
        raise TypeError("geometry must be a BufferedRequestGeometry")
    if not isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("sample rate must be finite and positive")
    if (
        not isinstance(local_samples, np.ndarray)
        or local_samples.ndim != 1
        or local_samples.dtype != np.dtype(np.int64)
    ):
        raise TypeError("local_samples must be a one-dimensional int64 array")
    if (
        not isinstance(delay_drift_seconds, np.ndarray)
        or delay_drift_seconds.ndim != 1
        or delay_drift_seconds.dtype.kind != "f"
    ):
        raise TypeError("delay drift must be a one-dimensional float array")
    if delay_drift_seconds.shape != local_samples.shape:
        raise ValueError("local sample and delay drift shapes must agree")
    if np.any(~np.isfinite(delay_drift_seconds)):
        raise ValueError("delay drift must be finite")
    ratio = geometry.samples_per_byte
    payload = geometry.payload_bytes_per_fft
    request = geometry.ffts_per_request
    maximum = int(np.iinfo(np.int64).max)
    if any(
        value > maximum
        for value in (
            ratio.numerator,
            ratio.denominator,
            payload,
            request,
            geometry.fft_samples,
            geometry.word_bytes,
            geometry.alignment_bytes,
            request * payload,
        )
    ):
        raise OverflowError("request geometry exceeds int64 arithmetic")
    delays = delay_drift_seconds
    origin = _cursor_divide(
        _cursor_multiply(local_samples, ratio.denominator, "sample to byte"),
        ratio.numerator,
    )
    origin = _cursor_subtract(
        origin, origin % geometry.word_bytes, "initial word alignment"
    )
    valid = origin > -(request * payload)
    origin = np.where(valid, origin, 0)
    skipped = np.where(origin < 0, -(origin // payload), 0)
    valid &= skipped < request
    origin[~valid] = 0
    skipped[~valid] = 0
    origin = _cursor_add(
        origin,
        _cursor_multiply(skipped, payload, "initial FFT skip"),
        "initial FFT skip",
    )
    # Scalar requests with no initial skip never apply delay drift.
    drift = np.zeros(local_samples.shape, dtype=np.int64)
    needs_drift = valid & (skipped > 0)
    drift[needs_drift] = _cursor_drift(
        skipped[needs_drift],
        delays[needs_drift],
        sample_rate=sample_rate,
        ffts_per_request=request,
    )
    correction = _cursor_divide(
        _cursor_multiply(drift, ratio.denominator, "drift to byte"),
        ratio.numerator,
    )
    extra = needs_drift & (
        _cursor_subtract(origin, correction, "drift underflow check") < 0
    )
    skipped = _cursor_add(skipped, extra.astype(np.int64), "extra FFT skip")
    valid &= skipped < request
    extra &= valid
    origin[extra] = _cursor_add(origin[extra], payload, "extra FFT skip")
    drift[extra] = _cursor_drift(
        skipped[extra],
        delays[extra],
        sample_rate=sample_rate,
        ffts_per_request=request,
    )
    origin[extra] = _cursor_subtract(
        origin[extra],
        _cursor_divide(
            _cursor_multiply(
                drift[extra], ratio.denominator, "corrected drift to byte"
            ),
            ratio.numerator,
        ),
        "corrected byte origin",
    )
    origin[~valid] = 0
    skipped[~valid] = 0
    drift[~valid] = 0
    origin = _cursor_subtract(
        origin, origin % geometry.word_bytes, "final word alignment"
    )
    origin = _cursor_subtract(
        origin, origin % geometry.alignment_bytes, "final request alignment"
    )
    aligned_samples = (
        _cursor_multiply(origin, ratio.numerator, "byte to sample")
        // ratio.denominator
    )
    delta = _cursor_subtract(
        _cursor_subtract(
            _cursor_add(
                np.where(valid, local_samples, 0),
                _cursor_multiply(
                    skipped, geometry.fft_samples, "sample skip"
                ),
                "skipped sample origin",
            ),
            drift,
            "decoder drift",
        ),
        aligned_samples,
        "decoder displacement",
    )
    return BufferedRequestCursors(origin, skipped, delta, valid)


def request_control_mask(
    *,
    origin: int,
    skipped: int,
    bit_count: int,
    extent_frames: int,
    frame_bytes: int = 8224,
    payload_bytes_per_fft: int = 65536,
    control_word_bits: int = 30,
) -> np.ndarray:
    """Build the framed request mask for a validated control-word contract.

    Defaults describe the verified fixed reader. Geometry is validated by
    the planner once, rather than inside every request's control-mask loop.
    """

    _integer(origin, "reader request origin")
    _integer(skipped, "reader request skipped FFT count")
    _integer(bit_count, "reader request bit count", minimum=1)
    _integer(extent_frames, "reader allocation extent")
    if skipped > bit_count:
        raise ValueError("reader request skip count exceeds its bit count")
    bits = np.zeros(bit_count, dtype=np.bool_)
    if skipped < bit_count:
        indices = np.arange(skipped, bit_count, dtype=np.int64)
        bits[skipped:] = (
            origin + indices * payload_bytes_per_fft
            < extent_frames * frame_bytes
        )
    # The request-wide slow-path gate examines only the first complete
    # control word.  A set bit beyond it cannot keep the request alive.
    if not np.any(bits[: min(control_word_bits, bit_count)]):
        bits[:] = False
    return bits


class RawResidualOwner(StrEnum):
    """Explicit event ownership when filler separates trustworthy clocks."""

    PHYSICAL_FRAME = "preceding-physical-frame"
    CLOCK_FRAME = "preceding-clock-frame"


@dataclass(frozen=True, slots=True)
class RawBufferedRequestState:
    """Reusable raw allocation anchors and source-derived event slots."""

    anchor_samples: np.ndarray
    event_allocations: frozenset[int]
    event_flags: np.ndarray
    allocation_frames: int
    samples_per_frame: int
    timestamp_origin_frame: int

    @classmethod
    def from_timeline(
        cls,
        timeline: PhysicalAllocationTimeline,
        *,
        samples_per_frame: int,
        timestamp_origin_frame: int = 0,
        residual_owner: RawResidualOwner = RawResidualOwner.PHYSICAL_FRAME,
    ) -> RawBufferedRequestState:
        """Express physical-read anchors in one exact sample coordinate.

        The timestamp origin may be signed and is subtracted before sample
        multiplication. A positive residual belongs to the read containing
        the preceding physical frame by default: a transition exactly at a
        read's end belongs to that completed read. The explicit
        CLOCK_FRAME contract uses the last trustworthy frame before
        a filler hole instead. These contracts coincide without physical
        holes. Independent constant-clock controls establish the latter for
        adjacent filler/jump histories; callers must establish which contract
        describes their reader. Zero residuals create no event. No output
        coordinate or recording identity chooses this contract. The file
        replay adapter explicitly selects CLOCK_FRAME.
        """
        if not isinstance(timeline, PhysicalAllocationTimeline):
            raise TypeError("timeline must be a PhysicalAllocationTimeline")
        if not isinstance(residual_owner, RawResidualOwner):
            raise TypeError("residual owner must be RawResidualOwner")
        samples_per_frame = _integer(
            samples_per_frame, "samples per frame", minimum=1
        )
        if not isinstance(timestamp_origin_frame, Integral) or isinstance(
            timestamp_origin_frame, bool
        ):
            raise TypeError("timestamp origin frame must be an integer")
        origin = int(timestamp_origin_frame)
        limits = np.iinfo(np.int64)
        if samples_per_frame > int(limits.max):
            raise OverflowError("samples per frame exceeds int64 arithmetic")
        if int(limits.min) <= origin <= int(limits.max):
            relative = _cursor_subtract(
                timeline.anchors, origin, "relative raw allocation anchor"
            )
            anchors = _cursor_multiply(
                relative, samples_per_frame, "raw allocation sample anchor"
            )
        else:
            # An arbitrary origin can lie outside int64 even when every
            # relative coordinate fits. Subtract with exact Python integers.
            anchors = np.fromiter(
                (
                    (int(anchor) - origin) * samples_per_frame
                    for anchor in timeline.anchors
                ),
                dtype=np.int64,
                count=timeline.anchors.size,
            )
        if np.any(anchors[1:] < anchors[:-1]):
            raise ValueError("raw allocation sample anchors must be ordered")
        capacity = int(timeline._allocation_frames)
        preceding = (
            timeline.physical_stops[:-1]
            if residual_owner is RawResidualOwner.CLOCK_FRAME
            else timeline.physical_starts[1:]
        )
        allocations = (preceding[timeline.residuals > 0] - 1) // capacity
        events = frozenset(map(int, allocations))
        flags = np.zeros(anchors.shape, dtype=np.bool_)
        if allocations.size:
            flags[allocations] = True
        anchors.setflags(write=False)
        flags.setflags(write=False)
        return cls(
            anchors, events, flags, capacity, samples_per_frame, origin
        )


@dataclass(frozen=True, slots=True)
class BufferedRequestDiscardBatch:
    """Exact cursor/control state and compact discards for one bounded batch.

    Invalid allocation/cursor rows have zero eligibility and zero cursor
    fields. They contribute no exclusion intervals: initial-source and EOF
    decisions remain with the caller. Padded columns after each explicit
    request width also have zero eligibility and contribute no discards.
    """

    selected_allocations: np.ndarray
    cursors: BufferedRequestCursors
    valid_requests: np.ndarray
    event_requests: np.ndarray
    eligible: np.ndarray
    ineligible_intervals: tuple[tuple[int, int], ...]


def _int64_vector(value: object, name: str) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.ndim != 1
        or value.dtype != np.dtype(np.int64)
    ):
        raise TypeError(f"{name} must be a one-dimensional int64 array")
    return value


def compile_latched_raw_window_heads(
    state: RawBufferedRequestState,
    batch: BufferedRequestDiscardBatch,
    geometry: BufferedRequestGeometry,
) -> np.ndarray:
    """Derive aligned heads for explicitly known retained raw requests.

    The selected allocation's clock remains latched through internal
    timestamp jumps. Add only the aligned byte cursor in exact packing
    units. Decoder displacement and initial FFT skips belong to subsequent
    decoding, not this head. Neither current frame timestamps nor storage
    translations replace the retained allocation clock.

    This arithmetic does not establish that an allocation
    was returned, its extent, successor transfers, filler or EOF behavior.
    Callers must supply known requests; unresolved rows fail explicitly.
    No source mapping, admission, validity or reported exposure is changed.
    """
    if not isinstance(state, RawBufferedRequestState):
        raise TypeError("state must be RawBufferedRequestState")
    if not isinstance(batch, BufferedRequestDiscardBatch):
        raise TypeError("batch must be BufferedRequestDiscardBatch")
    if not isinstance(geometry, BufferedRequestGeometry):
        raise TypeError("geometry must be BufferedRequestGeometry")
    anchors = _int64_vector(state.anchor_samples, "raw sample anchors")
    selected = _int64_vector(
        batch.selected_allocations, "selected allocations"
    )
    origins = _int64_vector(
        batch.cursors.origin_bytes, "aligned byte cursors"
    )
    if origins.shape != selected.shape:
        raise ValueError("aligned cursor and allocation shapes must agree")
    for valid in (batch.valid_requests, batch.cursors.valid):
        if (
            not isinstance(valid, np.ndarray)
            or valid.dtype != np.dtype(np.bool_)
            or valid.shape != selected.shape
        ):
            raise TypeError("request validity must be a matching bool vector")
        if not np.all(valid):
            raise ValueError("returned aligned head is unresolved")
    if np.any((selected < 0) | (selected >= anchors.size)):
        raise ValueError("returned allocation lies outside raw history")
    if np.any(origins < 0):
        raise ValueError("aligned byte cursors must be nonnegative")
    ratio = geometry.samples_per_byte
    if np.any(origins % ratio.denominator):
        raise ValueError("aligned cursor does not name a whole sample")
    offsets = _cursor_multiply(
        origins // ratio.denominator,
        ratio.numerator,
        "aligned raw cursor samples",
    )
    heads = _cursor_add(
        anchors[selected], offsets, "latched returned raw window head"
    )
    return np.frombuffer(heads.tobytes(), dtype=np.int64)


def compile_buffered_request_discards(
    state: RawBufferedRequestState,
    request_fft_starts: np.ndarray,
    desired_sample_starts: np.ndarray,
    endpoint_drifts: np.ndarray,
    ordinary_source_starts: np.ndarray,
    geometry: BufferedRequestGeometry,
    *,
    sample_rate: float,
    frame_bytes: int,
    control_word_bits: int,
    request_fft_counts: np.ndarray | None = None,
) -> BufferedRequestDiscardBatch:
    """Compile exact framed masks and next-anchor reentry without expansion.

    Desired and ordinary source starts share ``state.anchor_samples``'s
    timestamp-origin-relative sample coordinate. Each ordinary-source row
    has one entry per FFT, in request order. Request widths may clip padded
    final rows; they do not establish EOF behavior. Only event allocations
    apply the framed-byte/control-word gate. A successor anchor restores
    its per-FFT suffix after that gate, then skipped leading FFTs are cleared.
    Returned masks can directly replace the corresponding physical planner
    chunk logic; source spans and exposure remain separate.
    """
    if not isinstance(state, RawBufferedRequestState):
        raise TypeError("state must be a RawBufferedRequestState")
    if not isinstance(geometry, BufferedRequestGeometry):
        raise TypeError("geometry must be a BufferedRequestGeometry")
    frame_bytes = _integer(frame_bytes, "frame bytes", minimum=1)
    control_word_bits = _integer(
        control_word_bits, "control word bits", minimum=1
    )
    maximum = int(np.iinfo(np.int64).max)
    if (
        frame_bytes > maximum
        or control_word_bits > maximum
        or state.allocation_frames * frame_bytes > maximum
    ):
        raise OverflowError(
            "framed control geometry exceeds int64 arithmetic"
        )
    starts = _int64_vector(request_fft_starts, "request FFT starts")
    desired = _int64_vector(desired_sample_starts, "desired sample starts")
    if desired.shape != starts.shape:
        raise ValueError("request and desired sample shapes must agree")
    if np.any(starts < 0):
        raise ValueError("request FFT starts must be nonnegative")
    if (
        not isinstance(ordinary_source_starts, np.ndarray)
        or ordinary_source_starts.ndim != 2
        or ordinary_source_starts.dtype != np.dtype(np.int64)
    ):
        raise TypeError("ordinary source starts must be an int64 matrix")
    rows, width = ordinary_source_starts.shape
    if rows != starts.size:
        raise ValueError("ordinary source rows must match the request count")
    if width > geometry.ffts_per_request or (rows and width == 0):
        raise ValueError(
            "ordinary source width must fit one nonempty request"
        )
    if request_fft_counts is None:
        counts = np.full(starts.shape, width, dtype=np.int64)
    else:
        counts = _int64_vector(request_fft_counts, "request FFT counts")
        if counts.shape != starts.shape:
            raise ValueError("request FFT count shape must match requests")
        if np.any((counts < 1) | (counts > width)):
            raise ValueError(
                "request FFT counts must lie within matrix width"
            )
    _cursor_add(starts, np.maximum(counts - 1, 0), "global FFT request stop")
    allocations = (
        np.searchsorted(state.anchor_samples, desired, side="right") - 1
    )
    allocated = (allocations >= 0) & (allocations < state.anchor_samples.size)
    local = np.zeros(desired.shape, dtype=np.int64)
    local[allocated] = _cursor_subtract(
        desired[allocated],
        state.anchor_samples[allocations[allocated]],
        "raw request local origin",
    )
    batch = buffered_request_cursors(
        local, endpoint_drifts, geometry, sample_rate=sample_rate
    )
    valid = allocated & batch.valid
    cursors = BufferedRequestCursors(
        np.where(valid, batch.origin_bytes, 0),
        np.where(valid, batch.skipped_ffts, 0),
        np.where(valid, batch.decoder_delta_samples, 0),
        valid,
    )
    event_rows = np.zeros(starts.shape, dtype=np.bool_)
    event_rows[valid] = state.event_flags[allocations[valid]]
    positions = np.arange(width, dtype=np.int64)
    active = positions[None, :] < counts[:, None]
    eligible = active & valid[:, None]
    if np.any(event_rows):
        byte_offsets = _cursor_multiply(
            positions,
            geometry.payload_bytes_per_fft,
            "control FFT byte offset",
        )
        framed = (
            _cursor_add(
                cursors.origin_bytes[event_rows, None],
                byte_offsets[None, :],
                "framed request byte position",
            )
            < state.allocation_frames * frame_bytes
        )
        framed &= active[event_rows]
        framed &= positions[None, :] >= cursors.skipped_ffts[event_rows, None]
        framed &= np.any(framed[:, : min(width, control_word_bits)], axis=1)[
            :, None
        ]
        event_allocations = allocations[event_rows]
        successor = event_allocations + 1 < state.anchor_samples.size
        if np.any(successor):
            next_anchors = state.anchor_samples[
                event_allocations[successor] + 1, None
            ]
            framed[successor] |= (
                ordinary_source_starts[event_rows][successor] >= next_anchors
            )
        eligible[event_rows] = framed & active[event_rows]
    eligible &= positions[None, :] >= cursors.skipped_ffts[:, None]
    # Invalid requests are left for the caller's initial/EOF handling. Merge
    # only the exact known-request exclusions, including overlapping batches.
    excluded = active & valid[:, None] & ~eligible
    request_rows, columns = np.nonzero(excluded)
    indices = np.unique(
        _cursor_add(
            starts[request_rows],
            columns.astype(np.int64),
            "global FFT exclusion",
        )
    )
    if indices.size:
        edges = np.flatnonzero(np.diff(indices) != 1)
        run_starts = np.concatenate((indices[:1], indices[edges + 1]))
        run_stops = np.concatenate((indices[edges], indices[-1:]))
        intervals = tuple(
            (int(start), int(stop) + 1)
            for start, stop in zip(run_starts, run_stops, strict=True)
        )
    else:
        intervals = ()
    return BufferedRequestDiscardBatch(
        allocations, cursors, valid, event_rows, eligible, intervals
    )
