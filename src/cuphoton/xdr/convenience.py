# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""High-level convenience wrappers for the xDataReader FITS GPU loader."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from .prefetch import batch_to_device_stream


def open_gpu(path, **fits_open_kwargs):
    """Astropy-compatible single-file entry point.

    xDataReader intentionally has no Astropy dependency, so it
    cannot return an Astropy ``HDUList`` with monkey-patched HDUs. Use
    :func:`batch_to_device` or :func:`batch_to_device_stream` for
    dependency-free FITS loading.
    """
    raise NotImplementedError(
        "open_gpu() requires Astropy HDU objects in the original "
        "integration. "
        "cuphoton.xdr does not depend on Astropy; use "
        "batch_to_device() or batch_to_device_stream() instead."
    )


def batch_to_device(
    paths: Sequence[str | Path],
    hdu_indices: Iterable[int] = (1,),
    *,
    out=None,
    stream=None,
    section=None,
    parallel: bool = True,
    prefetch_depth: int = 2,
    decode_batch_files: int = 1,
    batch_queue_depth: int = 2,
    native_read_threads: int | None = None,
    native_plan_threads: int | None = None,
    native_batcher: str | bool = "auto",
):
    """Load image HDUs from FITS files into stacked device arrays.

    Parameters
    ----------
    paths : sequence of paths
        The FITS files to load.
    hdu_indices : iterable of int
        HDU indices to load from each file. All files must have matching
        HDU shapes/dtypes at the same index.
    stream : cupy.cuda.Stream or None
        Stream to run the reads on. Defaults to the null stream.
    out : cupy.ndarray or sequence of cupy.ndarray, optional
        Preallocated output buffer(s). One buffer is required per HDU index,
        each shaped ``(N, H, W)`` or the section shape.
    section : tuple of slice or None
        Optional 2D ROI applied uniformly to the CompImageHDUs.
    parallel : bool
    When True, use the streaming batch reader. Set False for a depth-1 path
    that preserves the existing API without requiring Astropy HDU objects.
    prefetch_depth, decode_batch_files, batch_queue_depth,
    native_read_threads, native_plan_threads, native_batcher
        Passed through to `batch_to_device_stream` when ``parallel=True``.

    Returns
    -------
    tuple of cupy.ndarray, one per entry in ``hdu_indices``, each shaped
    ``(N, H, W)`` (or the ROI shape in H/W when ``section`` is given).
    """
    if not parallel:
        prefetch_depth = 1
        decode_batch_files = 1
        batch_queue_depth = 1
        native_batcher = False

    return batch_to_device_stream(
        paths,
        hdu_indices=hdu_indices,
        out=out,
        prefetch_depth=prefetch_depth,
        decode_batch_files=decode_batch_files,
        batch_queue_depth=batch_queue_depth,
        native_read_threads=native_read_threads,
        native_plan_threads=native_plan_threads,
        native_batcher=native_batcher,
        section=section,
        stream=stream,
    )
