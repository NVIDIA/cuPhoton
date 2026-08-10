# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import os
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

import pytest


def test_public_import_does_not_import_legate():
    code = """
import sys
import cuphoton.xdr as module
assert callable(module.load_hdf5)
assert "legate" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_public_hdf5_api_keeps_small_surface_case():
    from cuphoton import xdr

    assert tuple(inspect.signature(xdr.load_hdf5).parameters) == (
        "path",
        "dataset_name",
    )


def test_load_hdf5_delegates_to_legate(monkeypatch, tmp_path: Path):
    from cuphoton.xdr import hdf5

    calls = []
    expected = object()

    def fake_from_file(path, dataset_name):
        calls.append((path, dataset_name))
        return expected

    monkeypatch.setattr(
        hdf5, "_load_legate_from_file", lambda: fake_from_file
    )
    path = tmp_path / "images.h5"

    assert hdf5.load_hdf5(path, "/images/science") is expected
    assert calls == [(path, "/images/science")]


@pytest.mark.parametrize("dataset_name", ["", "   "])
def test_load_hdf5_rejects_empty_dataset_name(dataset_name):
    from cuphoton.xdr import load_hdf5

    with pytest.raises(ValueError, match="must not be empty"):
        load_hdf5("images.h5", dataset_name)


def test_load_hdf5_rejects_non_string_dataset_name():
    from cuphoton.xdr import load_hdf5

    with pytest.raises(TypeError, match="must be a string"):
        load_hdf5("images.h5", 1)


def test_missing_legate_error_is_actionable(monkeypatch):
    from cuphoton.xdr import hdf5

    def missing_backend():
        raise hdf5.LegateHdf5Unavailable("install the hdf5 extra")

    monkeypatch.setattr(hdf5, "_load_legate_from_file", missing_backend)

    with pytest.raises(hdf5.LegateHdf5Unavailable, match="hdf5 extra"):
        hdf5.load_hdf5("images.h5", "images")


def test_legate_hdf5_available_reports_backend_state(monkeypatch):
    from cuphoton.xdr import hdf5

    monkeypatch.setattr(hdf5, "_load_legate_from_file", lambda: object())
    assert hdf5.legate_hdf5_available()

    def missing_backend():
        raise hdf5.LegateHdf5Unavailable("missing")

    monkeypatch.setattr(hdf5, "_load_legate_from_file", missing_backend)
    assert not hdf5.legate_hdf5_available()


@pytest.mark.skipif(
    find_spec("legate") is None, reason="Legate not installed"
)
def test_real_legate_cpu_roundtrip(tmp_path: Path):
    import h5py
    import numpy as np

    path = tmp_path / "real-legate.h5"
    expected = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("/images/science", data=expected)

    code = """
import sys
import numpy as np
from legate.core import get_legate_runtime
from cuphoton.xdr import load_hdf5

array = load_hdf5(sys.argv[1], "/images/science")
get_legate_runtime().issue_execution_fence(block=True)
actual = np.asarray(array.get_physical_array())
expected = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
np.testing.assert_array_equal(actual, expected)
"""
    env = os.environ.copy()
    env["LEGATE_CONFIG"] = "--cpus 2 --gpus 0 --io-use-vfd-gds false"
    result = subprocess.run(
        [sys.executable, "-c", code, str(path)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
