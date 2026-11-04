# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from cuphoton import __version__
from cuphoton.core.runtime import runtime_metadata


def test_cpu_runtime_metadata_is_self_describing() -> None:
    metadata = runtime_metadata(backend="cpu", dtype="float64")

    assert metadata["package_version"] == __version__
    assert metadata["backend"] == "cpu"
    assert metadata["device"] == "cpu"
    assert metadata["dtype"] == "float64"
    assert metadata["python_version"]
    assert metadata["numpy_version"]
