# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from cuphoton.core.artifacts import array_sha256, file_sha256


def test_file_sha256_streams_exact_bytes(tmp_path) -> None:
    path = tmp_path / "artifact.bin"
    payload = b"cuphoton-artifact\0" * 100_000
    path.write_bytes(payload)

    assert file_sha256(path) == hashlib.sha256(payload).hexdigest()


def test_array_sha256_binds_dtype_shape_and_values() -> None:
    values = np.arange(12, dtype=np.float32).reshape(3, 4)

    assert array_sha256(values) == array_sha256(values.copy())
    assert array_sha256(values[:, ::-1]) == array_sha256(
        np.ascontiguousarray(values[:, ::-1])
    )
    assert array_sha256(values) != array_sha256(values.astype(np.float64))
    assert array_sha256(values) != array_sha256(values.reshape(2, 6))
    changed = values.copy()
    changed[0, 0] = 1
    assert array_sha256(values) != array_sha256(changed)


def test_array_sha256_rejects_object_arrays() -> None:
    with pytest.raises(TypeError, match="object arrays"):
        array_sha256(np.asarray([object()], dtype=object))
