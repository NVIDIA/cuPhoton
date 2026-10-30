# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Codec for the observable SWIN visibility record stream."""

from __future__ import annotations

import struct
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import BinaryIO, SupportsFloat, SupportsInt, cast

import numpy as np

from cuphoton.xbeam.integrity import HashingWriter

SYNC_WORD = 0xFF00FF00
FORMAT_VERSION = 1
HEADER = struct.Struct("<IIiIdiii2sIdddd")


class SwinFormatError(ValueError):
    """Malformed or unsupported SWIN data."""


def encode_baseline(antenna_a: int, antenna_b: int) -> int:
    """Encode zero-based antenna indices into the observable baseline word."""

    if not 0 <= antenna_a <= 254 or not 0 <= antenna_b <= 254:
        raise ValueError("SWIN baseline antenna indices must be in [0, 254]")
    return (antenna_a + 1) * 256 + antenna_b + 1


def decode_baseline(code: int) -> tuple[int, int]:
    first, second = divmod(int(code), 256)
    if not 1 <= first <= 255 or not 1 <= second <= 255:
        raise SwinFormatError(f"invalid SWIN baseline code {code}")
    return first - 1, second - 1


@dataclass(frozen=True, slots=True)
class SwinRecord:
    baseline: int
    mjd: int
    seconds: float
    configuration_id: int
    source_id: int
    frequency_id: int
    polarization: str
    pulsar_bin: int
    weight: float
    u_m: float
    v_m: float
    w_m: float
    spectrum: np.ndarray

    @property
    def antenna_pair(self) -> tuple[int, int]:
        return decode_baseline(self.baseline)


def _spectrum(value: object) -> np.ndarray:
    array = np.asarray(value, dtype=np.dtype("<c8"))
    if array.ndim != 1:
        raise ValueError("SWIN spectrum must be one-dimensional")
    return np.ascontiguousarray(array)


def _polarization_bytes(value: str) -> bytes:
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError as error:
        raise ValueError("SWIN polarization must be ASCII") from error
    if len(encoded) != 2:
        raise ValueError(
            "SWIN polarization must contain exactly two ASCII characters"
        )
    return encoded


def write_record(
    stream: BinaryIO | HashingWriter, record: SwinRecord
) -> None:
    spectrum = _spectrum(record.spectrum)
    decode_baseline(record.baseline)
    header = HEADER.pack(
        SYNC_WORD,
        FORMAT_VERSION,
        int(record.baseline),
        int(record.mjd),
        float(record.seconds),
        int(record.configuration_id),
        int(record.source_id),
        int(record.frequency_id),
        _polarization_bytes(record.polarization),
        int(record.pulsar_bin),
        float(record.weight),
        float(record.u_m),
        float(record.v_m),
        float(record.w_m),
    )
    stream.write(header)
    stream.write(spectrum.tobytes(order="C"))


def write_swin(
    path: str | PathLike[str], records: Iterable[SwinRecord]
) -> int:
    """Write records and return their count."""

    count = 0
    with Path(path).open("xb") as stream:
        for record in records:
            write_record(stream, record)
            count += 1
    return count


def from_visibility(
    record: object,
    antenna_indices: Mapping[str | int, int],
    *,
    configuration_indices: Mapping[str | int, int] | None = None,
    source_indices: Mapping[str | int, int] | None = None,
) -> SwinRecord:
    """Convert a visibility using explicit SWIN index maps.

    Configuration and source IDs must be numeric when their map is omitted.
    """

    def field(name: str) -> object:
        try:
            return getattr(record, name)
        except AttributeError as error:
            raise ValueError(
                f"visibility record is missing field {name!r}"
            ) from error

    def index(name: str, indices: Mapping[str | int, int] | None) -> int:
        value = field(name)
        if indices is None:
            return int(cast(SupportsInt | str, value))
        try:
            return indices[cast(str | int, value)]
        except KeyError as error:
            raise ValueError(f"no SWIN {name} index for {value!r}") from error

    antenna_a = cast(str | int, field("antenna_a"))
    antenna_b = cast(str | int, field("antenna_b"))
    try:
        baseline = encode_baseline(
            antenna_indices[antenna_a], antenna_indices[antenna_b]
        )
    except KeyError as error:
        raise ValueError(
            f"no SWIN antenna index for {error.args[0]!r}"
        ) from error
    return SwinRecord(
        baseline=baseline,
        mjd=int(cast(SupportsInt | str, field("mjd"))),
        seconds=float(cast(SupportsFloat | str, field("seconds_of_day"))),
        configuration_id=index("configuration_id", configuration_indices),
        source_id=index("source_id", source_indices),
        frequency_id=int(
            cast(SupportsInt | str, field("output_frequency_id"))
        ),
        polarization=str(field("polarization_a"))
        + str(field("polarization_b")),
        pulsar_bin=0,
        weight=float(cast(SupportsFloat | str, field("weight"))),
        u_m=float(cast(SupportsFloat | str, field("u_m"))),
        v_m=float(cast(SupportsFloat | str, field("v_m"))),
        w_m=float(cast(SupportsFloat | str, field("w_m"))),
        spectrum=_spectrum(field("spectrum")),
    )


class SwinWriter:
    """Visibility sink that emits observable SWIN records."""

    def __init__(
        self,
        path: str | PathLike[str],
        antenna_indices: Mapping[str | int, int],
        *,
        configuration_indices: Mapping[str | int, int] | None = None,
        source_indices: Mapping[str | int, int] | None = None,
    ) -> None:
        self.path = Path(path)
        self.antenna_indices = dict(antenna_indices)
        self.configuration_indices = (
            None
            if configuration_indices is None
            else dict(configuration_indices)
        )
        self.source_indices = (
            None if source_indices is None else dict(source_indices)
        )
        self._stream: HashingWriter | None = HashingWriter(
            self.path.open("xb")
        )
        self._digest = self._stream
        self._channels: dict[str, int] = {}
        self.count = 0

    @property
    def integrity(self):
        return {
            "version": 1,
            "record_count": self.count,
            "channels_by_frequency": self._channels,
            "files": {self.path.name: self._digest.summary()},
        }

    def write(self, record: object) -> None:
        if self._stream is None:
            raise RuntimeError("SWIN writer is closed")
        converted = from_visibility(
            record,
            self.antenna_indices,
            configuration_indices=self.configuration_indices,
            source_indices=self.source_indices,
        )
        key = str(converted.frequency_id)
        count = int(converted.spectrum.size)
        if self._channels.setdefault(key, count) != count:
            raise ValueError("SWIN frequency changes channel count")
        write_record(self._stream, converted)
        self.count += 1

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def __enter__(self) -> SwinWriter:
        return self

    def __exit__(
        self, exc_type: object, exc: object, traceback: object
    ) -> None:
        self.close()


def read_record(stream: BinaryIO, channel_count: int) -> SwinRecord | None:
    if channel_count <= 0:
        raise ValueError("channel_count must be positive")
    header = stream.read(HEADER.size)
    if not header:
        return None
    if len(header) != HEADER.size:
        raise SwinFormatError("truncated SWIN record header")
    (
        sync,
        version,
        baseline,
        mjd,
        seconds,
        configuration_id,
        source_id,
        frequency_id,
        polarization,
        pulsar_bin,
        weight,
        u_m,
        v_m,
        w_m,
    ) = HEADER.unpack(header)
    if sync != SYNC_WORD:
        raise SwinFormatError(f"invalid SWIN sync word 0x{sync:08x}")
    if version != FORMAT_VERSION:
        raise SwinFormatError(f"unsupported SWIN format version {version}")
    decode_baseline(baseline)
    try:
        polarization_text = polarization.decode("ascii")
    except UnicodeDecodeError as error:
        raise SwinFormatError("non-ASCII SWIN polarization") from error
    payload_size = channel_count * np.dtype("<c8").itemsize
    payload = stream.read(payload_size)
    if len(payload) != payload_size:
        raise SwinFormatError("truncated SWIN visibility spectrum")
    spectrum = np.frombuffer(payload, dtype="<c8").copy()
    return SwinRecord(
        baseline,
        mjd,
        seconds,
        configuration_id,
        source_id,
        frequency_id,
        polarization_text,
        pulsar_bin,
        weight,
        u_m,
        v_m,
        w_m,
        spectrum,
    )


def iter_swin(
    source: str | PathLike[str] | BinaryIO, channel_count: int
) -> Iterator[SwinRecord]:
    """Iterate a record stream with a caller-supplied channel count."""

    if hasattr(source, "read"):
        while (
            record := read_record(cast(BinaryIO, source), channel_count)
        ) is not None:
            yield record
        return
    with Path(source).open("rb") as stream:
        while (record := read_record(stream, channel_count)) is not None:
            yield record


def read_swin(
    path: str | PathLike[str], channel_count: int
) -> list[SwinRecord]:
    return list(iter_swin(path, channel_count))


__all__ = [
    "FORMAT_VERSION",
    "HEADER",
    "SYNC_WORD",
    "SwinFormatError",
    "SwinRecord",
    "SwinWriter",
    "decode_baseline",
    "encode_baseline",
    "from_visibility",
    "iter_swin",
    "read_record",
    "read_swin",
    "write_record",
    "write_swin",
]
