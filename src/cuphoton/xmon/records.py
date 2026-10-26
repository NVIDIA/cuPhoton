# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Small, serializable scientific records; arrays belong in artifacts."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import PurePosixPath
from typing import Any


def encode(value: Any) -> str:
    """Canonical, strict JSON for storage and detached snapshots."""
    return json.dumps(
        value, allow_nan=False, sort_keys=True, separators=(",", ":")
    )


def _text(value: str, name: str, maximum: int = 256) -> None:
    if not isinstance(value, str) or len(value) > maximum:
        raise ValueError(
            f"{name} must be text of at most {maximum} characters"
        )


@dataclass(frozen=True)
class Measurement:
    """One named scalar and its validity, independent of finding severity."""

    value: float | int | str | bool | None
    unit: str = ""
    status: str = "valid"
    coverage: float | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.status not in {"valid", "missing", "invalid", "unavailable"}:
            raise ValueError("unknown measurement status")
        if self.value is not None and not isinstance(
            self.value, (float, int, str, bool)
        ):
            raise TypeError(
                "measurement values must be ordinary JSON scalars"
            )
        if isinstance(self.value, float) and not math.isfinite(self.value):
            object.__setattr__(self, "value", None)
            object.__setattr__(self, "status", "invalid")
        if self.value is None and self.status == "valid":
            object.__setattr__(self, "status", "missing")
        if (
            self.status in {"missing", "unavailable"}
            and self.value is not None
        ):
            raise ValueError("missing and unavailable values must be null")
        if self.coverage is not None and (
            not math.isfinite(self.coverage) or not 0 <= self.coverage <= 1
        ):
            raise ValueError("coverage must be a finite fraction in [0, 1]")
        _text(self.unit, "unit", 64)
        _text(self.note, "note", 1024)
        if isinstance(self.value, str):
            _text(self.value, "value", 1024)


@dataclass(frozen=True)
class Finding:
    """An application rule result, separate from model interpretation."""

    code: str
    message: str
    category: str = "scientific"
    severity: str = "warning"

    def __post_init__(self) -> None:
        _text(self.code, "finding code", 128)
        _text(self.message, "finding message", 2048)
        if not self.code:
            raise ValueError("finding code is required")
        if self.category not in {"scientific", "data_quality", "processing"}:
            raise ValueError("unknown finding category")
        if self.severity not in {"info", "warning", "error"}:
            raise ValueError("unknown finding severity")


@dataclass(frozen=True)
class Artifact:
    """A content-identified local reference; never fetched by the model."""

    name: str
    path: str
    sha256: str
    media_type: str = "application/octet-stream"

    def __post_init__(self) -> None:
        _text(self.name, "artifact name")
        _text(self.path, "artifact path", 1024)
        _text(self.media_type, "media type", 128)
        path = PurePosixPath(self.path)
        if (
            not self.path
            or path.is_absolute()
            or ".." in path.parts
            or "\\" in self.path
            or ":" in self.path
        ):
            raise ValueError(
                "artifact paths must be relative local references"
            )
        if not re.fullmatch(r"[a-f0-9]{64}", self.sha256):
            raise ValueError("artifact requires a lowercase SHA-256 digest")


@dataclass(frozen=True)
class Sample:
    """One exposure, window, iteration or step, with explicit time semantics.

    Sequence orders samples; it does not imply cadence or physical time.
    Observation values share an application-defined time unit and origin.
    Processing timestamps are added by Monitor, never inferred here.
    """

    id: str
    sequence: int
    measurements: dict[str, Measurement]
    revision: int = 1
    observed_start: float | None = None
    observed_end: float | None = None
    time_unit: str = "s"
    findings: tuple[Finding, ...] = ()
    artifacts: tuple[Artifact, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _text(self.id, "sample id", 128)
        _text(self.time_unit, "time unit", 64)
        if not self.id:
            raise ValueError("sample id is required")
        if type(self.sequence) is not int or type(self.revision) is not int:
            raise ValueError("sequence and revision must be integers")
        if self.revision < 1:
            raise ValueError("revision starts at one")
        for value in (self.observed_start, self.observed_end):
            if value is not None and not math.isfinite(value):
                raise ValueError("observation times must be finite or null")
        if self.observed_end is not None and (
            self.observed_start is None
            or self.observed_end < self.observed_start
        ):
            raise ValueError("observation end requires an ordered start")
        if len(self.measurements) > 128:
            raise ValueError("at most 128 measurements per sample")
        for name, measurement in self.measurements.items():
            _text(name, "measurement name", 128)
            if not name or not isinstance(measurement, Measurement):
                raise ValueError(
                    "measurements require names and Measurement values"
                )
        if len(encode(self.metadata).encode()) > 16384:
            raise ValueError(
                "metadata exceeds 16 KiB; use referenced artifacts"
            )
        if len(encode(asdict(self)).encode()) > 65536:
            raise ValueError(
                "sample exceeds 64 KiB; use referenced artifacts"
            )


def sample_from_dict(value: dict[str, Any]) -> Sample:
    """Validate and detach a saved sample."""
    value = json.loads(encode(value))
    value["measurements"] = {
        name: Measurement(**item)
        for name, item in value["measurements"].items()
    }
    value["findings"] = tuple(Finding(**x) for x in value.get("findings", ()))
    value["artifacts"] = tuple(
        Artifact(**x) for x in value.get("artifacts", ())
    )
    return Sample(**value)
