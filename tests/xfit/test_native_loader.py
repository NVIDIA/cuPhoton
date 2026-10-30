# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""CPU-only coverage for native xFit runtime library loading."""

from __future__ import annotations

import builtins
import sys
from types import ModuleType, SimpleNamespace

import pytest

from cuphoton.xfit import _native


def _runtime(monkeypatch, load):
    pathfinder = ModuleType("cuda.pathfinder")
    pathfinder.load_nvidia_dynamic_lib = load
    monkeypatch.setitem(sys.modules, "cuda.pathfinder", pathfinder)
    monkeypatch.setattr(
        _native.importlib.util, "find_spec", lambda _: object()
    )


def test_missing_extension_does_not_discover_runtime(monkeypatch):
    monkeypatch.setattr(_native.importlib.util, "find_spec", lambda _: None)
    _runtime_import = builtins.__import__

    def reject_pathfinder(name, *args, **kwargs):
        if name == "cuda.pathfinder":
            pytest.fail("missing extension must not load CUDA libraries")
        return _runtime_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_pathfinder)
    with pytest.raises(RuntimeError, match=r"install cuphoton\[io\]"):
        _native.load_extension()


def test_runtime_libraries_load_before_extension(monkeypatch):
    events = []

    def load(name):
        events.append(name)
        return SimpleNamespace(abs_path=f"/cuda/lib64/lib{name}.so.13")

    _runtime(monkeypatch, load)
    extension = ModuleType("cuphoton.xfit._native_ext")

    def import_extension(name):
        events.append(name)
        return extension

    monkeypatch.setattr(_native.importlib, "import_module", import_extension)
    assert _native.load_extension() is extension
    assert events == ["cudart", "cublas", "cuphoton.xfit._native_ext"]


@pytest.mark.parametrize("missing", ["cudart", "cublas"])
def test_missing_runtime_reports_dependency_without_loading_extension(
    monkeypatch, missing
):
    error = RuntimeError(f"{missing} not found")

    def load(name):
        if name == missing:
            raise error
        return SimpleNamespace(abs_path=f"/cuda/lib64/lib{name}.so.13")

    _runtime(monkeypatch, load)
    monkeypatch.setattr(
        _native.importlib,
        "import_module",
        lambda _: pytest.fail("extension imported before its dependencies"),
    )
    with pytest.raises(RuntimeError, match=f"{missing} not found") as caught:
        _native.load_extension()
    assert "CUDA 13" in str(caught.value)
    assert caught.value.__cause__ is error


def test_extension_linker_error_is_preserved(monkeypatch):
    _runtime(
        monkeypatch,
        lambda name: SimpleNamespace(
            abs_path=f"/cuda/lib64/lib{name}.so.13.0.96"
        ),
    )
    error = ImportError("libcublas.so.13: cannot open shared object file")

    def import_extension(name):
        raise error

    monkeypatch.setattr(_native.importlib, "import_module", import_extension)
    with pytest.raises(RuntimeError, match="libcublas.so.13") as caught:
        _native.load_extension()
    assert caught.value.__cause__ is error


@pytest.mark.parametrize("wrong_major", ["cudart", "cublas"])
def test_cuda_12_library_is_rejected_before_extension_import(
    monkeypatch, wrong_major
):
    events = []

    def load(name):
        events.append(name)
        version = "12.8.90" if name == wrong_major else "13.0.96"
        return SimpleNamespace(abs_path=f"/system/lib{name}.so.{version}")

    _runtime(monkeypatch, load)
    monkeypatch.setattr(
        _native.importlib,
        "import_module",
        lambda _: pytest.fail("extension imported with CUDA 12 libraries"),
    )
    with pytest.raises(RuntimeError, match=f"/system/lib{wrong_major}.so.12"):
        _native.load_extension()
    assert events == (
        ["cudart"] if wrong_major == "cudart" else ["cudart", "cublas"]
    )
