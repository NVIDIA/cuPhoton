# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Strict parser for colon-delimited interoperability job/model files.

The parser deliberately exposes a neutral representation.  Interpretation of
individual fields belongs to the job adapter, not to this syntax layer.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO


class JobTextError(ValueError):
    """A malformed key/value job file."""


@dataclass(frozen=True, slots=True)
class TextEntry:
    key: str
    value: str
    line_number: int


@dataclass(frozen=True, slots=True)
class JobText(Mapping[str, tuple[str, ...]]):
    """Ordered entries with a lossless one-to-many mapping view."""

    entries: tuple[TextEntry, ...]
    source: str | None = None

    def __iter__(self) -> Iterator[str]:
        seen: set[str] = set()
        for entry in self.entries:
            if entry.key not in seen:
                seen.add(entry.key)
                yield entry.key

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __getitem__(self, key: str) -> tuple[str, ...]:
        values = self.get_all(key)
        if not values:
            raise KeyError(key)
        return values

    def get_all(self, key: str) -> tuple[str, ...]:
        return tuple(
            entry.value for entry in self.entries if entry.key == key
        )

    def get_unique(self, key: str) -> str:
        values = self.get_all(key)
        if not values:
            raise KeyError(key)
        if len(values) != 1:
            raise JobTextError(
                f"{self.source or '<text>'}: field {key!r} occurs "
                f"{len(values)} times"
            )
        return values[0]

    def as_dict(self) -> dict[str, tuple[str, ...]]:
        return {key: self[key] for key in self}


def parse_job_text(
    source: str | Path | TextIO,
    *,
    comment_prefixes: Sequence[str] = ("#", "@"),
) -> JobText:
    """Parse a strict ``KEY: value`` file while preserving duplicates/order.

    Empty values are valid.  Blank lines and lines whose first non-whitespace
    character is a configured comment prefix are ignored.  Every other line
    must contain a non-empty key followed by a colon.
    """

    if not isinstance(source, (str, Path)):
        stream = source
        source_name = getattr(source, "name", None)
        return _parse_stream(stream, source_name, comment_prefixes)

    path = Path(source)
    with path.open("r", encoding="utf-8", newline=None) as stream:
        return _parse_stream(stream, str(path), comment_prefixes)


def _parse_stream(
    stream: TextIO, source_name: str | None, comment_prefixes: Sequence[str]
) -> JobText:
    entries: list[TextEntry] = []
    for line_number, raw_line in enumerate(stream, 1):
        line = raw_line.rstrip("\r\n")
        stripped = line.lstrip()
        if not stripped or any(
            stripped.startswith(prefix) for prefix in comment_prefixes
        ):
            continue
        if ":" not in line:
            raise JobTextError(
                f"{source_name or '<text>'}:{line_number}: expected "
                "'KEY: value'"
            )
        key, value = line.split(":", 1)
        key = key.strip()
        if not key:
            raise JobTextError(
                f"{source_name or '<text>'}:{line_number}: field name is "
                "empty"
            )
        entries.append(
            TextEntry(key=key, value=value.strip(), line_number=line_number)
        )
    return JobText(tuple(entries), source_name)


def parse_input_file(path: str | Path) -> JobText:
    return parse_job_text(path)


def parse_calc_file(path: str | Path) -> JobText:
    return parse_job_text(path)


def parse_im_file(path: str | Path) -> JobText:
    return parse_job_text(path)


__all__ = [
    "JobText",
    "JobTextError",
    "TextEntry",
    "parse_calc_file",
    "parse_im_file",
    "parse_input_file",
    "parse_job_text",
]
