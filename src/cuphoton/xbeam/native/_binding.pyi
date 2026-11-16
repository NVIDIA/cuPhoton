# SPDX-License-Identifier: Apache-2.0
"""Typed execution entry points for the optional Cython extension."""

from os import PathLike
from typing import Any

def cuda_available() -> bool: ...
def correlate(
    job: str | PathLike[str],
    output: str | PathLike[str],
    index_dir: str | PathLike[str],
    scan: str,
    first_integration: int = ...,
    integration_count: int = ...,
    fft_tile: int = ...,
    device: int = ...,
    vdif_io: str = ...,
    *,
    profile: str,
    upload_workers: int = ...,
    prefetch: bool = ...,
) -> dict[str, Any]: ...
