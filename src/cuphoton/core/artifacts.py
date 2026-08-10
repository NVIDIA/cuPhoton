# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Deterministic hashes for portable workflow artifacts and arrays."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

_CHUNK_BYTES = 1024 * 1024


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of one file without loading it at once."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(values: Any) -> str:
    """Hash an array's versioned dtype, shape, and C-order bytes."""

    array = np.asarray(values)
    if array.dtype.hasobject:
        raise TypeError("object arrays do not have a portable numeric hash")
    contiguous = np.ascontiguousarray(array)
    header = json.dumps(
        {
            "contract": "cuphoton.ndarray.sha256.v1",
            "dtype": contiguous.dtype.str,
            "shape": list(contiguous.shape),
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    payload = memoryview(contiguous).cast("B")
    for offset in range(0, len(payload), _CHUNK_BYTES):
        digest.update(payload[offset : offset + _CHUNK_BYTES])
    return digest.hexdigest()


__all__ = ["array_sha256", "file_sha256"]
