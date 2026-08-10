# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Invariant-backed xDataReader commands."""

from __future__ import annotations

import os
from pathlib import Path

from cuphoton.core.cli import (
    BoolInvariant,
    CSVIntegerInvariant,
    IntegerInvariant,
    InvariantAwareCommand,
    SetInvariant,
    StringInvariant,
    VariablePositionalInvariant,
)

from .benchmark_fits import parse_hdu_indices, run_benchmark

_KNOWN_COMMAND_EXCEPTIONS = (
    FileNotFoundError,
    ImportError,
    RuntimeError,
    ValueError,
)


class NativeBatcherInvariant(SetInvariant):
    """Native batcher selection."""

    _set = {"auto", "on", "off"}


class MockStorageInvariant(SetInvariant):
    """Mock storage location."""

    _set = {"device", "host"}


class BenchmarkFitsCommand(InvariantAwareCommand):
    """Benchmark xdr GPU FITS loading.

    It uses xDataReader's native CFITSIO planning path instead of Astropy
    HDU objects. The benchmark operates on explicit HDU indices. By default it
    benchmarks HDU 1; use ``--hdu-indices 1,2,3`` for multi-extension FITS
    files.
    """

    _shortname_ = ""
    _log_level_ = False
    _summary_ = "Benchmark GPU FITS reading."

    fits_file: list[str] | None = None
    scan_dir: str | None = None
    max_files: int | None = None
    hdu_indices: list[int] | tuple[int, ...] = (1,)
    iterations = 5
    prefetch_depth = 2
    decode_batch_files = 1
    batch_queue_depth = 2
    native_read_threads = 4
    native_plan_threads = max(1, os.cpu_count() or 1)
    native_batcher = "auto"
    mock_storage: str | None = None
    skip_gds_read = False

    class FitsFileArg(VariablePositionalInvariant):
        _metavar = "fits_file"
        _help = "FITS files to benchmark."

    class ScanDirArg(StringInvariant):
        _arg = "--dir"
        _help = "Directory of FITS files to scan."
        _mandatory = False
        _default = None
        _metavar = "SCAN_DIR"
        _maxlen = 4096

    class MaxFilesArg(IntegerInvariant):
        _arg = "--max-files"
        _help = "Maximum files to take from --dir."
        _mandatory = False
        _default = None

    class HduIndicesArg(CSVIntegerInvariant):
        _arg = "--hdu-indices"
        _help = "Comma-separated HDU indices. Default: 1."
        _parser_error = "--hdu-indices must be comma-separated integers"
        _mandatory = False
        _default = "1"

    class IterationsArg(IntegerInvariant):
        _arg = "--iterations"
        _mandatory = False
        _default = 5

    class PrefetchDepthArg(IntegerInvariant):
        _arg = "--prefetch-depth"
        _mandatory = False
        _default = 2

    class DecodeBatchFilesArg(IntegerInvariant):
        _arg = "--decode-batch-files"
        _mandatory = False
        _default = 1

    class BatchQueueDepthArg(IntegerInvariant):
        _arg = "--batch-queue-depth"
        _mandatory = False
        _default = 2

    class NativeReadThreadsArg(IntegerInvariant):
        _arg = "--native-read-threads"
        _mandatory = False
        _default = 4

    class NativePlanThreadsArg(IntegerInvariant):
        _arg = "--native-plan-threads"
        _mandatory = False
        _default = max(1, os.cpu_count() or 1)

    class NativeBatcherArg(NativeBatcherInvariant):
        _arg = "--native-batcher"
        _mandatory = False
        _default = "auto"
        _metavar = "{auto,on,off}"

    class MockStorageArg(MockStorageInvariant):
        _arg = "--mock-storage"
        _mandatory = False
        _default = None
        _metavar = "{device,host}"

    class SkipGdsReadArg(BoolInvariant):
        _arg = "--skip-gds-read"
        _help = "Skip raw NativeBatchBuilder read timing."

    def run(self) -> None:
        try:
            run_benchmark(
                self.fits_file or (),
                scan_dir=(
                    Path(self.scan_dir).expanduser()
                    if self.scan_dir is not None
                    else None
                ),
                max_files=self.max_files,
                hdu_indices=list(parse_hdu_indices(self.hdu_indices)),
                iterations=self.iterations,
                prefetch_depth=self.prefetch_depth,
                decode_batch_files=self.decode_batch_files,
                batch_queue_depth=self.batch_queue_depth,
                native_read_threads=self.native_read_threads,
                native_plan_threads=self.native_plan_threads,
                native_batcher=self.native_batcher,
                mock_storage_kind=self.mock_storage,
                skip_gds_read=self.skip_gds_read,
                out=self._out,
            )
        except _KNOWN_COMMAND_EXCEPTIONS as exc:
            raise SystemExit(str(exc)) from exc


__all__ = ["BenchmarkFitsCommand"]
