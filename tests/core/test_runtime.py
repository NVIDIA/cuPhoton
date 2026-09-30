# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from types import SimpleNamespace

from cuphoton import __version__
from cuphoton.core import runtime
from cuphoton.core.runtime import runtime_metadata


def test_cpu_runtime_metadata_is_self_describing() -> None:
    metadata = runtime_metadata(backend="cpu", dtype="float64")

    assert metadata["package_version"] == __version__
    assert metadata["backend"] == "cpu"
    assert metadata["device"] == "cpu"
    assert metadata["dtype"] == "float64"
    assert metadata["python_version"]
    assert metadata["numpy_version"]


def test_numba_mlir_runtime_keeps_cupy_device_and_compiler_version(
    monkeypatch,
) -> None:
    cuda_runtime = SimpleNamespace(
        getDevice=lambda: 0,
        getDeviceProperties=lambda index: {
            "name": b"test GPU",
            "major": 12,
            "minor": 0,
        },
        driverGetVersion=lambda: 13000,
        runtimeGetVersion=lambda: 13000,
    )
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(
            __version__="14.0.0", cuda=SimpleNamespace(runtime=cuda_runtime)
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_package_version",
        lambda distribution: {
            "numpy": "2.3.0",
            "numba-cuda-mlir": "0.5.4",
        }.get(distribution),
    )

    metadata = runtime_metadata(backend="numba-cuda-mlir", dtype="float64")

    assert metadata["backend"] == "numba-cuda-mlir"
    assert metadata["device"] == "cuda"
    assert metadata["device_name"] == "test GPU"
    assert metadata["cupy_version"] == "14.0.0"
    assert metadata["numba_cuda_mlir_version"] == "0.5.4"
    assert metadata["compute_capability"] == "12.0"
