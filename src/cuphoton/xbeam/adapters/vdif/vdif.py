# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""VDIF frame ingestion.

This module implements the standard 32-byte (non-legacy) VDIF header and the
two-bit offset-binary sample representation described by the public VDIF 1.0
specification.  It deliberately contains no correlator-specific file layout
or job-model assumptions.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from functools import lru_cache
from os import PathLike
from pathlib import Path
from struct import Struct
from typing import Any, BinaryIO

import numpy as np

HEADER_BYTES = 32
LEGACY_HEADER_BYTES = 16
_LEGACY_HEADER_STRUCT = Struct("<4I")
_HEADER_STRUCT = Struct("<8I")
TWO_BIT_HIGH_LEVEL = 3.3359
_TWO_BIT_LEVELS = np.asarray(
    [-TWO_BIT_HIGH_LEVEL, -1.0, 1.0, TWO_BIT_HIGH_LEVEL], dtype=np.float32
)


class VdifError(ValueError):
    """Base class for malformed or unsupported VDIF data."""


class UnsupportedVdifError(VdifError):
    """A well-formed VDIF feature is not supported by this package."""


class TruncatedVdifFrameError(VdifError):
    """A stream ended in the middle of a VDIF header or frame."""


class DuplicateVdifFrameError(VdifError):
    """More than one frame occupied a slot under the reject policy."""


@dataclass(frozen=True, slots=True)
class VdifHeader:
    """Decoded fields from a standard 32-byte VDIF header."""

    invalid: bool
    legacy: bool
    seconds_from_epoch: int
    reference_epoch: int
    frame_number: int
    version: int
    log2_channels: int
    frame_length_bytes: int
    complex_data: bool
    bits_per_sample: int
    thread_id: int
    station_id: int
    extended_data_version: int
    extended_words: tuple[int, int, int, int]

    @property
    def channel_count(self) -> int:
        return 1 << self.log2_channels

    @property
    def payload_bytes(self) -> int:
        return self.frame_length_bytes - HEADER_BYTES

    @property
    def components_per_sample(self) -> int:
        return 2 if self.complex_data else 1

    @property
    def bits_per_complete_sample(self) -> int:
        return (
            self.channel_count
            * self.components_per_sample
            * self.bits_per_sample
        )

    @property
    def samples_per_frame(self) -> int:
        bits = self.payload_bytes * 8
        complete_sample_bits = self.bits_per_complete_sample
        if complete_sample_bits <= 0 or bits % complete_sample_bits:
            raise VdifError(
                "VDIF payload does not contain an integer number of complete "
                "samples"
            )
        return bits // complete_sample_bits

    @property
    def station_code(self) -> str | None:
        """Return the conventional two-character station code, if textual."""

        first = (self.station_id >> 8) & 0xFF
        second = self.station_id & 0xFF
        if first < ord("0"):
            return None
        try:
            return bytes((first, second)).decode("ascii")
        except UnicodeDecodeError:
            return None

    @classmethod
    def from_bytes(
        cls,
        data: bytes | bytearray | memoryview,
        *,
        strict: bool = True,
        require_two_bit: bool = True,
    ) -> VdifHeader:
        """Parse and validate a standard VDIF header.

        ``require_two_bit`` defaults to true because two-bit voltage input is
        the package's declared ingestion contract.  Set it false only for
        format inspection; sample decoding still rejects other widths.
        """

        view = memoryview(data)
        if len(view) < LEGACY_HEADER_BYTES:
            raise TruncatedVdifFrameError(
                f"VDIF header needs at least {LEGACY_HEADER_BYTES} bytes; "
                f"got {len(view)}"
            )
        word0, word1, word2, word3 = _LEGACY_HEADER_STRUCT.unpack_from(view)

        invalid = bool(word0 >> 31)
        legacy = bool((word0 >> 30) & 1)
        if legacy:
            raise UnsupportedVdifError(
                "legacy 16-byte VDIF headers are unsupported"
            )
        if len(view) < HEADER_BYTES:
            raise TruncatedVdifFrameError(
                f"standard VDIF header needs {HEADER_BYTES} bytes; "
                f"got {len(view)}"
            )

        words = _HEADER_STRUCT.unpack_from(view)
        extended_words = words[4:8]
        version = (word2 >> 29) & 0x7
        bits_per_sample = ((word3 >> 26) & 0x1F) + 1
        frame_length_bytes = (word2 & 0x00FF_FFFF) * 8
        header = cls(
            invalid=invalid,
            legacy=False,
            seconds_from_epoch=word0 & 0x3FFF_FFFF,
            reference_epoch=(word1 >> 24) & 0x3F,
            frame_number=word1 & 0x00FF_FFFF,
            version=version,
            log2_channels=(word2 >> 24) & 0x1F,
            frame_length_bytes=frame_length_bytes,
            complex_data=bool(word3 >> 31),
            bits_per_sample=bits_per_sample,
            thread_id=(word3 >> 16) & 0x3FF,
            station_id=word3 & 0xFFFF,
            extended_data_version=(extended_words[0] >> 24) & 0xFF,
            extended_words=extended_words,
        )

        if strict and (word1 >> 30):
            raise VdifError("VDIF word 1 reserved bits 31-30 must be zero")
        if version != 0:
            raise UnsupportedVdifError(
                f"unsupported VDIF header version {version}"
            )
        if frame_length_bytes < HEADER_BYTES:
            raise VdifError(
                f"VDIF frame length {frame_length_bytes} is smaller than "
                "its header"
            )
        if frame_length_bytes % 8:
            raise VdifError("VDIF frame length must be a multiple of 8 bytes")
        if require_two_bit and bits_per_sample != 2:
            raise UnsupportedVdifError(
                f"unsupported VDIF quantization: {bits_per_sample} "
                "bits/sample; expected 2"
            )
        # Accessing this property also verifies complete-sample divisibility.
        _ = header.samples_per_frame
        return header


@dataclass(frozen=True, slots=True)
class VdifFrame:
    """One complete VDIF frame and its byte offset in the source stream."""

    header: VdifHeader
    payload: bytes
    offset: int = 0

    @classmethod
    def from_bytes(
        cls,
        data: bytes | bytearray | memoryview,
        *,
        offset: int = 0,
        strict: bool = True,
    ) -> VdifFrame:
        header = VdifHeader.from_bytes(data, strict=strict)
        if len(data) < header.frame_length_bytes:
            raise TruncatedVdifFrameError(
                f"VDIF frame at byte {offset} declares "
                f"{header.frame_length_bytes} bytes "
                f"but only {len(data)} are available"
            )
        if strict and len(data) != header.frame_length_bytes:
            raise VdifError(
                f"VDIF frame buffer has {len(data)} bytes; header declares "
                f"{header.frame_length_bytes}"
            )
        return cls(
            header=header,
            payload=bytes(
                memoryview(data)[HEADER_BYTES : header.frame_length_bytes]
            ),
            offset=offset,
        )


@dataclass(frozen=True, slots=True)
class DecodedVdifFrame:
    """Decoded time samples and their per-sample validity."""

    header: VdifHeader
    samples: Any
    validity: Any


def decode_2bit(
    payload: bytes | bytearray | memoryview | np.ndarray,
    *,
    complex_data: bool = False,
    channels: int = 1,
    valid: bool = True,
    xp: Any = np,
) -> tuple[Any, Any]:
    """Decode a two-bit VDIF payload into ``[time, channel]`` samples.

    Codes are unpacked earliest-sample first from bits 1:0, 3:2, 5:4,
    and 7:6 of each byte.  Complex payloads pair adjacent I and Q codes,
    with I in the lower-order position.  ``xp`` may be NumPy or CuPy.
    """

    if channels <= 0 or channels & (channels - 1):
        raise VdifError("VDIF channel count must be a positive power of two")
    raw = np.frombuffer(payload, dtype=np.uint8)
    scalar_components = raw.size * 4
    components_per_time = channels * (2 if complex_data else 1)
    if scalar_components % components_per_time:
        raise VdifError(
            "VDIF payload does not contain an integer number of complete "
            "samples"
        )

    device_raw = xp.asarray(raw)
    shifts = xp.asarray((0, 2, 4, 6), dtype=xp.uint8)
    codes = ((device_raw[:, None] >> shifts[None, :]) & 0x3).reshape(-1)
    levels = xp.asarray(_TWO_BIT_LEVELS)
    values = levels[codes]
    sample_count = scalar_components // components_per_time

    if complex_data:
        components = values.reshape(sample_count, channels, 2)
        samples = (components[..., 0] + 1j * components[..., 1]).astype(
            xp.complex64, copy=False
        )
    else:
        samples = values.reshape(sample_count, channels).astype(
            xp.float32, copy=False
        )

    validity = xp.full(sample_count, bool(valid), dtype=xp.bool_)
    if not valid:
        samples = xp.zeros_like(samples)
    return samples, validity


def decode_frame(frame: VdifFrame, *, xp: Any = np) -> DecodedVdifFrame:
    """Decode a frame, zeroing samples when its invalid bit is set."""

    header = frame.header
    if header.bits_per_sample != 2:
        raise UnsupportedVdifError(
            f"unsupported VDIF quantization: {header.bits_per_sample} "
            "bits/sample; expected 2"
        )
    samples, validity = decode_2bit(
        frame.payload,
        complex_data=header.complex_data,
        channels=header.channel_count,
        valid=not header.invalid,
        xp=xp,
    )
    return DecodedVdifFrame(header=header, samples=samples, validity=validity)


def _iter_stream(stream: BinaryIO, *, strict: bool) -> Iterator[VdifFrame]:
    offset = 0
    while True:
        header_bytes = stream.read(HEADER_BYTES)
        if not header_bytes:
            return
        if len(header_bytes) != HEADER_BYTES:
            raise TruncatedVdifFrameError(
                f"truncated VDIF header at byte {offset}: got "
                f"{len(header_bytes)} bytes"
            )
        header = VdifHeader.from_bytes(header_bytes, strict=strict)
        remainder_size = header.frame_length_bytes - HEADER_BYTES
        payload = stream.read(remainder_size)
        if len(payload) != remainder_size:
            raise TruncatedVdifFrameError(
                f"truncated VDIF frame at byte {offset}: expected "
                f"{remainder_size} payload "
                f"bytes, got {len(payload)}"
            )
        yield VdifFrame(header=header, payload=payload, offset=offset)
        offset += header.frame_length_bytes


def iter_vdif_frames(
    source: str | PathLike[str] | BinaryIO,
    *,
    strict: bool = True,
) -> Iterator[VdifFrame]:
    """Iterate complete frames from a path or open binary stream."""

    if isinstance(source, (str, PathLike)):
        with Path(source).open("rb") as stream:
            yield from _iter_stream(stream, strict=strict)
    else:
        yield from _iter_stream(source, strict=strict)


# Leap seconds whose insertion occurred after the VDIF origin and before an
# epoch boundary.  The listed datetime is the first UTC second after the leap.
_LEAP_TRANSITIONS = (
    datetime(2006, 1, 1, tzinfo=UTC),
    datetime(2009, 1, 1, tzinfo=UTC),
    datetime(2012, 7, 1, tzinfo=UTC),
    datetime(2015, 7, 1, tzinfo=UTC),
    datetime(2017, 1, 1, tzinfo=UTC),
)
_VDIF_ORIGIN = datetime(2000, 1, 1, tzinfo=UTC)


@lru_cache(maxsize=128)
def reference_epoch_datetime(
    reference_epoch: int, *, epoch_cycle: int = 0
) -> datetime:
    """Return the UTC start of a VDIF six-month reference epoch.

    The six-bit header field wraps every 32 years.  ``epoch_cycle`` selects
    which 32-year cycle is intended; zero covers 2000 through 2031.
    """

    if not 0 <= reference_epoch < 64:
        raise VdifError("VDIF reference epoch must be in [0, 63]")
    if epoch_cycle < 0:
        raise VdifError("VDIF epoch cycle must be nonnegative")
    half_year = reference_epoch + 64 * epoch_cycle
    year = 2000 + half_year // 2
    month = 1 if half_year % 2 == 0 else 7
    return datetime(year, month, 1, tzinfo=UTC)


@lru_cache(maxsize=128)
def _epoch_origin_seconds(reference_epoch: int, epoch_cycle: int) -> int:
    epoch = reference_epoch_datetime(reference_epoch, epoch_cycle=epoch_cycle)
    nominal = int((epoch - _VDIF_ORIGIN).total_seconds())
    leap_seconds = sum(
        transition <= epoch for transition in _LEAP_TRANSITIONS
    )
    return nominal + leap_seconds


def mjd_seconds_to_vdif_seconds(value: Decimal) -> Decimal:
    """Convert a normalized UTC MJD instant to the leap-aware VDIF line."""
    nominal = value - Decimal(51544 * 86400)
    leaps = sum(
        nominal >= int((transition - _VDIF_ORIGIN).total_seconds())
        for transition in _LEAP_TRANSITIONS
    )
    return nominal + leaps


def vdif_seconds_to_mjd_seconds(value: Decimal) -> Decimal:
    """Convert a VDIF instant to UTC MJD seconds outside leap seconds."""
    leaps = 0
    for count, transition in enumerate(_LEAP_TRANSITIONS, 1):
        boundary = int((transition - _VDIF_ORIGIN).total_seconds()) + count
        if boundary - 1 <= value < boundary:
            raise VdifError("a leap-second instant has no normalized MJDTime")
        if value >= boundary:
            leaps = count
    return value - leaps + Decimal(51544 * 86400)


def seconds_since_vdif_origin(
    header: VdifHeader,
    *,
    epoch_cycle: int = 0,
) -> int:
    """Return the frame's integral second on a leap-aware VDIF time line."""

    return (
        _epoch_origin_seconds(header.reference_epoch, epoch_cycle)
        + header.seconds_from_epoch
    )


def absolute_frame_slot(
    header: VdifHeader,
    frames_per_second: int,
    *,
    epoch_cycle: int = 0,
) -> int:
    """Return an absolute integer frame slot on the VDIF time line."""

    if frames_per_second <= 0:
        raise VdifError("frames_per_second must be positive")
    if header.frame_number >= frames_per_second:
        raise VdifError(
            f"frame number {header.frame_number} is outside "
            "frames_per_second="
            f"{frames_per_second}"
        )
    return (
        seconds_since_vdif_origin(header, epoch_cycle=epoch_cycle)
        * frames_per_second
        + header.frame_number
    )


class DuplicatePolicy(StrEnum):
    """Deterministic policies for frames sharing a thread/time slot."""

    FIRST = "first"
    FIRST_VALID = "first-valid"
    LAST = "last"
    LAST_VALID = "last-valid"
    REJECT = "reject"


def _select_duplicate(
    current: VdifFrame,
    candidate: VdifFrame,
    policy: DuplicatePolicy,
) -> VdifFrame:
    if policy is DuplicatePolicy.REJECT:
        raise DuplicateVdifFrameError(
            f"duplicate VDIF frame for thread {candidate.header.thread_id}, "
            f"epoch {candidate.header.reference_epoch}, second "
            f"{candidate.header.seconds_from_epoch}, frame "
            f"{candidate.header.frame_number}"
        )
    if policy is DuplicatePolicy.FIRST:
        return current
    if policy is DuplicatePolicy.LAST:
        return candidate
    if policy is DuplicatePolicy.FIRST_VALID:
        return (
            candidate
            if current.header.invalid and not candidate.header.invalid
            else current
        )
    if policy is DuplicatePolicy.LAST_VALID:
        return (
            candidate
            if not candidate.header.invalid or current.header.invalid
            else current
        )
    raise AssertionError(f"unhandled duplicate policy {policy}")


def _stream_signature(
    header: VdifHeader,
) -> tuple[int, int, bool, int, int, int]:
    return (
        header.frame_length_bytes,
        header.channel_count,
        header.complex_data,
        header.bits_per_sample,
        header.station_id,
        header.samples_per_frame,
    )


@dataclass(frozen=True, slots=True)
class ThreadTimeline:
    """VDIF-thread slots where missing entries contain ``None``."""

    thread_id: int
    start_slot: int
    frames_per_second: int
    frames: tuple[VdifFrame | None, ...]
    template_header: VdifHeader

    @property
    def stop_slot(self) -> int:
        return self.start_slot + len(self.frames)

    @property
    def frame_validity(self) -> np.ndarray:
        return np.fromiter(
            (
                frame is not None and not frame.header.invalid
                for frame in self.frames
            ),
            dtype=np.bool_,
            count=len(self.frames),
        )

    def samples(self, *, xp: Any = np) -> tuple[Any, Any]:
        """Return zero-filled samples and per-sample validity."""

        header = self.template_header
        count = header.samples_per_frame
        shape = (count, header.channel_count)
        sample_dtype = xp.complex64 if header.complex_data else xp.float32
        sample_chunks: list[Any] = []
        validity_chunks: list[Any] = []
        for frame in self.frames:
            if frame is None:
                sample_chunks.append(xp.zeros(shape, dtype=sample_dtype))
                validity_chunks.append(xp.zeros(count, dtype=xp.bool_))
            else:
                decoded = decode_frame(frame, xp=xp)
                sample_chunks.append(decoded.samples)
                validity_chunks.append(decoded.validity)
        if not sample_chunks:
            return (
                xp.empty((0, header.channel_count), dtype=sample_dtype),
                xp.empty((0,), dtype=xp.bool_),
            )
        return xp.concatenate(sample_chunks, axis=0), xp.concatenate(
            validity_chunks, axis=0
        )


def build_thread_timelines(
    frames: Iterable[VdifFrame],
    *,
    frames_per_second: int,
    duplicate_policy: DuplicatePolicy | str = DuplicatePolicy.FIRST_VALID,
    start_slot: int | None = None,
    stop_slot: int | None = None,
    epoch_cycle: int = 0,
) -> Mapping[int, ThreadTimeline]:
    """Index reordered/multiplexed frames and zero-fill gaps per thread.

    All returned threads cover the same half-open slot range.  Invalid frames
    retain their slot but decode to zero with zero validity.  The default
    duplicate rule is explicit and may be replaced by a compatibility profile.
    """

    try:
        policy = DuplicatePolicy(duplicate_policy)
    except ValueError as exc:
        raise VdifError(
            f"unknown VDIF duplicate policy {duplicate_policy!r}"
        ) from exc
    if frames_per_second <= 0:
        raise VdifError("frames_per_second must be positive")

    slots_by_thread: dict[int, dict[int, VdifFrame]] = {}
    templates: dict[int, VdifHeader] = {}
    signatures: dict[int, tuple[int, int, bool, int, int, int]] = {}
    observed_slots: list[int] = []

    for frame in frames:
        header = frame.header
        slot = absolute_frame_slot(
            header, frames_per_second, epoch_cycle=epoch_cycle
        )
        thread = header.thread_id
        signature = _stream_signature(header)
        if thread in signatures and signatures[thread] != signature:
            raise VdifError(
                f"inconsistent VDIF frame format in thread {thread} at "
                f"byte {frame.offset}"
            )
        signatures.setdefault(thread, signature)
        templates.setdefault(thread, header)
        thread_slots = slots_by_thread.setdefault(thread, {})
        if slot in thread_slots:
            thread_slots[slot] = _select_duplicate(
                thread_slots[slot], frame, policy
            )
        else:
            thread_slots[slot] = frame
        observed_slots.append(slot)

    if not observed_slots:
        return {}
    first = min(observed_slots) if start_slot is None else start_slot
    last = max(observed_slots) + 1 if stop_slot is None else stop_slot
    if first < 0 or last < first:
        raise VdifError("timeline requires 0 <= start_slot <= stop_slot")

    result: dict[int, ThreadTimeline] = {}
    for thread in sorted(slots_by_thread):
        indexed = slots_by_thread[thread]
        ordered = tuple(indexed.get(slot) for slot in range(first, last))
        result[thread] = ThreadTimeline(
            thread_id=thread,
            start_slot=first,
            frames_per_second=frames_per_second,
            frames=ordered,
            template_header=templates[thread],
        )
    return result


__all__ = [
    "HEADER_BYTES",
    "TWO_BIT_HIGH_LEVEL",
    "DecodedVdifFrame",
    "DuplicatePolicy",
    "DuplicateVdifFrameError",
    "ThreadTimeline",
    "TruncatedVdifFrameError",
    "UnsupportedVdifError",
    "VdifError",
    "VdifFrame",
    "VdifHeader",
    "absolute_frame_slot",
    "build_thread_timelines",
    "decode_2bit",
    "decode_frame",
    "iter_vdif_frames",
    "reference_epoch_datetime",
    "seconds_since_vdif_origin",
]
