# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""JSON configuration adapter for the series CLI."""

import json
from pathlib import Path

from ..core.series import CorrelationSpec, correlate_series
from .hdf5 import Hdf5Source, write_series_hdf5


def run_series_config(
    config_path, output_path, backend="numpy", *, upload_workers=None
):
    config_path = Path(config_path)
    config = json.loads(config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError("series configuration must be a JSON object")
    required = {"input", "datasets", "correlation"}
    if required - set(config):
        raise ValueError(
            f"missing series keys: {sorted(required - set(config))}"
        )
    if not isinstance(config["correlation"], dict):
        raise ValueError("correlation must be a JSON object")
    unknown = set(config) - {
        "input",
        "datasets",
        "validity",
        "correlation",
        "io_backend",
        "read_buffer_bytes",
        "io_queue_depth",
        "upload_workers",
    }
    if unknown:
        raise ValueError(
            f"unknown series configuration keys: {sorted(unknown)}"
        )
    if not isinstance(config["input"], str):
        raise ValueError("input must be a path string")
    if not isinstance(config["datasets"], dict):
        raise ValueError("datasets must be a JSON object")
    validity = config.get("validity")
    if validity is not None and not isinstance(validity, dict):
        raise ValueError("validity must be a JSON object or null")
    source_path = Path(config["input"])
    if not source_path.is_absolute():
        source_path = config_path.parent / source_path
    spec = CorrelationSpec(**config["correlation"], backend=backend)
    with Hdf5Source(
        source_path,
        config["datasets"],
        validity_paths=validity,
        io_backend=config.get("io_backend", "posix"),
        read_buffer_bytes=config.get("read_buffer_bytes", 64 * 1024**2),
        io_queue_depth=config.get("io_queue_depth", 2),
        upload_workers=(
            config.get("upload_workers", "auto")
            if upload_workers is None
            else upload_workers
        ),
    ) as source:
        write_series_hdf5(output_path, correlate_series(source, spec))
