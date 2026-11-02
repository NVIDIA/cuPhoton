# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Format-independent limits for concurrent input delivery."""

from ..errors import ConfigurationError

MAX_UPLOAD_WORKERS = 16


def normalize_upload_workers(value: int | str | None) -> int | None:
    """Return an explicit worker limit, or None for backend-selected auto."""
    if isinstance(value, str):
        text = value.strip().lower()
        if text == "auto":
            return None
        try:
            value = int(text)
        except ValueError:
            value = -1
    if value is not None and (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 1 <= value <= MAX_UPLOAD_WORKERS
    ):
        raise ConfigurationError(
            "correlate",
            "upload_workers",
            "must be 'auto' or an integer from 1 through 16",
        )
    return value


def resolve_upload_workers(
    value: int | str | None, *, alias: int | str | None = None
) -> int | None:
    """Resolve canonical and compatibility keywords, rejecting conflicts."""
    requested = normalize_upload_workers(value)
    previous = normalize_upload_workers(alias)
    if (
        requested is not None
        and previous is not None
        and requested != previous
    ):
        raise ConfigurationError(
            "correlate",
            "upload_workers",
            "conflicts with the compatibility keyword",
        )
    return requested if requested is not None else previous


def bounded_upload_workers(
    value: int | str | None, *, automatic: int, available: int
) -> int:
    """Bound active workers independently of queued-buffer capacity."""
    requested = normalize_upload_workers(value)
    return min(available, automatic if requested is None else requested)
