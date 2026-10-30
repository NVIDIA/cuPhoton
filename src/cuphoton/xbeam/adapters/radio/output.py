# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Portable native ``xbeam-v1`` visibility output."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

from cuphoton.xbeam.integrity import HashingWriter

FORMAT_NAME = "xbeam-v1"
METADATA_FILENAME = "metadata.jsonl"
SPECTRA_FILENAME = "spectra.c64"


class OutputFormatError(ValueError):
    """Malformed or unsupported native output."""


@dataclass(frozen=True, slots=True)
class OutputRecord:
    antenna_a: str | int
    antenna_b: str | int
    mjd: int
    seconds_of_day: Decimal
    source_id: str | int
    configuration_id: str | int
    output_frequency_id: str | int
    polarization_a: str
    polarization_b: str
    weight: float
    u_m: float
    v_m: float
    w_m: float
    spectrum: np.ndarray

    @property
    def seconds(self) -> Decimal:
        """Alias for consumers that call this field ``seconds``."""

        return self.seconds_of_day


_FIELDS = (
    "antenna_a",
    "antenna_b",
    "mjd",
    "seconds_of_day",
    "source_id",
    "configuration_id",
    "output_frequency_id",
    "polarization_a",
    "polarization_b",
    "weight",
    "u_m",
    "v_m",
    "w_m",
    "spectrum",
)


def _field(record: object, name: str) -> Any:
    aliases = {"seconds_of_day": "seconds"}
    if isinstance(record, Mapping):
        try:
            return record[name]
        except KeyError as error:
            alias = aliases.get(name)
            if alias is not None and alias in record:
                return record[alias]
            raise OutputFormatError(
                f"record is missing field {name!r}"
            ) from error
    try:
        return getattr(record, name)
    except AttributeError as error:
        alias = aliases.get(name)
        if alias is not None and hasattr(record, alias):
            return getattr(record, alias)
        raise OutputFormatError(
            f"record is missing field {name!r}"
        ) from error


def _array(value: object) -> np.ndarray:
    array = np.asarray(value, dtype=np.dtype("<c8"))
    if array.ndim != 1:
        raise OutputFormatError("record spectrum must be one-dimensional")
    return np.ascontiguousarray(array)


class XBeamV1Writer:
    """Streaming writer for JSONL metadata plus contiguous complex64 data."""

    def __init__(self, directory: str | PathLike[str]) -> None:
        self.directory = Path(directory)
        self._metadata: HashingWriter | None = None
        self._spectra: HashingWriter | None = None
        self._offset = 0
        self.count = 0

    def __enter__(self) -> XBeamV1Writer:
        self.directory.mkdir(parents=True, exist_ok=True)
        metadata_path = self.directory / METADATA_FILENAME
        spectra_path = self.directory / SPECTRA_FILENAME
        if metadata_path.exists() or spectra_path.exists():
            raise FileExistsError(
                f"xbeam-v1 output already exists in {self.directory}"
            )
        self._metadata = HashingWriter(
            metadata_path.open("x", encoding="utf-8", newline="\n")
        )
        try:
            self._spectra = HashingWriter(spectra_path.open("xb"))
        except BaseException:
            self._metadata.close()
            metadata_path.unlink()
            raise
        self._digests = {
            METADATA_FILENAME: self._metadata,
            SPECTRA_FILENAME: self._spectra,
        }
        return self

    @property
    def integrity(self):
        return {
            "version": 1,
            "record_count": self.count,
            "files": {
                name: writer.summary()
                for name, writer in self._digests.items()
            },
        }

    def write(self, record: object) -> None:
        if self._metadata is None or self._spectra is None:
            raise RuntimeError("xbeam-v1 writer is not open")
        values = {name: _field(record, name) for name in _FIELDS}
        spectrum = _array(values.pop("spectrum"))
        try:
            # Decimal text retains the exact normalized boundary time.
            values["mjd"] = int(values["mjd"])
            values["seconds_of_day"] = str(
                Decimal(str(values["seconds_of_day"]))
            )
            values["weight"] = float(values["weight"])
            values["u_m"] = float(values["u_m"])
            values["v_m"] = float(values["v_m"])
            values["w_m"] = float(values["w_m"])
            for name in (
                "antenna_a",
                "antenna_b",
                "source_id",
                "configuration_id",
                "output_frequency_id",
            ):
                item = values[name]
                if isinstance(item, np.generic):
                    values[name] = item.item()
        except (InvalidOperation, TypeError, ValueError) as error:
            raise OutputFormatError(
                f"record contains an invalid numeric field: {error}"
            ) from error
        metadata = {
            "format": FORMAT_NAME,
            **values,
            "spectrum_offset": self._offset,
            "channel_count": int(spectrum.size),
        }
        try:
            encoded = json.dumps(
                metadata, separators=(",", ":"), ensure_ascii=False
            )
        except (TypeError, ValueError) as error:
            raise OutputFormatError(
                f"record metadata is not JSON-serializable: {error}"
            ) from error
        self._metadata.write(encoded + "\n")
        payload = spectrum.tobytes(order="C")
        self._spectra.write(payload)
        self._offset += len(payload)
        self.count += 1

    def close(self) -> None:
        if self._metadata is not None:
            self._metadata.close()
            self._metadata = None
        if self._spectra is not None:
            self._spectra.close()
            self._spectra = None

    def __exit__(
        self, exc_type: object, exc: object, traceback: object
    ) -> None:
        self.close()


def write_xbeam_v1(
    directory: str | PathLike[str], records: Iterable[object]
) -> int:
    with XBeamV1Writer(directory) as writer:
        for record in records:
            writer.write(record)
        return writer.count


def iter_xbeam_v1(directory: str | PathLike[str]) -> Iterator[OutputRecord]:
    root = Path(directory)
    spectra_path = root / SPECTRA_FILENAME
    expected_offset = 0
    with (root / METADATA_FILENAME).open(
        "r", encoding="utf-8"
    ) as metadata_stream:
        with spectra_path.open("rb") as spectra_stream:
            for line_number, line in enumerate(metadata_stream, 1):
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as error:
                    raise OutputFormatError(
                        f"{METADATA_FILENAME}:{line_number}: invalid JSON"
                    ) from error
                if item.get("format") != FORMAT_NAME:
                    raise OutputFormatError(
                        f"{METADATA_FILENAME}:{line_number}: unsupported "
                        "format"
                    )
                offset = item.get("spectrum_offset")
                count = item.get("channel_count")
                if not isinstance(offset, int) or offset != expected_offset:
                    raise OutputFormatError(
                        f"{METADATA_FILENAME}:{line_number}: non-contiguous "
                        "spectrum offset"
                    )
                if not isinstance(count, int) or count < 0:
                    raise OutputFormatError(
                        f"{METADATA_FILENAME}:{line_number}: invalid channel "
                        "count"
                    )
                payload_size = count * np.dtype("<c8").itemsize
                payload = spectra_stream.read(payload_size)
                if len(payload) != payload_size:
                    raise OutputFormatError("truncated spectra.c64 payload")
                expected_offset += payload_size
                try:
                    yield OutputRecord(
                        antenna_a=item["antenna_a"],
                        antenna_b=item["antenna_b"],
                        mjd=int(item["mjd"]),
                        seconds_of_day=Decimal(
                            str(
                                item.get(
                                    "seconds_of_day", item.get("seconds")
                                )
                            )
                        ),
                        source_id=item["source_id"],
                        configuration_id=item["configuration_id"],
                        output_frequency_id=item["output_frequency_id"],
                        polarization_a=item["polarization_a"],
                        polarization_b=item["polarization_b"],
                        weight=float(item["weight"]),
                        u_m=float(item["u_m"]),
                        v_m=float(item["v_m"]),
                        w_m=float(item["w_m"]),
                        spectrum=np.frombuffer(payload, dtype="<c8").copy(),
                    )
                except (
                    KeyError,
                    TypeError,
                    ValueError,
                    InvalidOperation,
                ) as error:
                    raise OutputFormatError(
                        f"{METADATA_FILENAME}:{line_number}: invalid record "
                        "metadata"
                    ) from error
            if spectra_stream.read(1):
                raise OutputFormatError(
                    "spectra.c64 contains unreferenced trailing data"
                )


def read_xbeam_v1(directory: str | PathLike[str]) -> list[OutputRecord]:
    return list(iter_xbeam_v1(directory))


__all__ = [
    "FORMAT_NAME",
    "METADATA_FILENAME",
    "OutputFormatError",
    "OutputRecord",
    "SPECTRA_FILENAME",
    "XBeamV1Writer",
    "iter_xbeam_v1",
    "read_xbeam_v1",
    "write_xbeam_v1",
]
