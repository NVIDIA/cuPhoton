# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest

from cuphoton.xdr import nvcomp_batch, setup_package


def _cuda_tree(root, *, library="libcudart.so.13"):
    (root / "include" / "crt").mkdir(parents=True)
    (root / "include" / "crt" / "host_config.h").touch()
    (root / "include" / "cuda_runtime.h").touch()
    (root / "lib").mkdir()
    (root / "lib" / library).touch()
    return root


@pytest.fixture
def cuda_wheel(monkeypatch, tmp_path):
    # CUDA 13 wheels share a namespace directory without __init__.py files.
    root = _cuda_tree(tmp_path / "site-packages" / "nvidia" / "cu13")
    (root / "include" / "cufile.h").touch()
    namespace = ModuleType("nvidia")
    namespace.__path__ = [str(root.parent)]
    monkeypatch.setitem(sys.modules, "nvidia", namespace)
    for name in ("nvidia.cu13", "nvidia.cuda_runtime", "nvidia.cufile"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    return root


def test_cuda_wheel_supplies_runtime_and_build_headers(
    monkeypatch, tmp_path, cuda_wheel
):
    system_toolkit = _cuda_tree(tmp_path / "system-cuda")
    monkeypatch.setattr(
        setup_package.shutil,
        "which",
        lambda name: str(system_toolkit / "bin" / "nvcc"),
    )
    loaded = []
    monkeypatch.setattr(nvcomp_batch, "_load_shared_library", loaded.append)

    nvcomp_batch._preload_cudart()
    headers, libraries = setup_package._find_cuda_toolkit()

    assert loaded == [cuda_wheel / "lib" / "libcudart.so.13"]
    assert headers == cuda_wheel / "include"
    assert libraries == cuda_wheel / "lib"
    assert not (libraries / "libcudart.so").exists()
    # A system toolkit can lack cuFile even when its wheel is installed.
    assert (
        setup_package._find_cufile_include(system_toolkit / "include")
        == cuda_wheel / "include"
    )


@pytest.mark.parametrize("variable", ["CUDA_HOME", "CUDA_PATH"])
def test_explicit_cuda_toolkit_precedes_wheel(
    monkeypatch, tmp_path, cuda_wheel, variable
):
    toolkit = _cuda_tree(tmp_path / "selected-cuda")
    monkeypatch.setenv(variable, str(toolkit))
    loaded = []
    monkeypatch.setattr(nvcomp_batch, "_load_shared_library", loaded.append)

    nvcomp_batch._preload_cudart()

    assert loaded == [toolkit / "lib" / "libcudart.so.13"]
    assert setup_package._find_cuda_toolkit() == (
        toolkit / "include",
        toolkit / "lib",
    )


@pytest.mark.parametrize("available", [True, False])
def test_runtime_fallback_never_loads_unversioned_cuda(
    monkeypatch, tmp_path, available
):
    toolkit = _cuda_tree(tmp_path / "old-cuda", library="libcudart.so")
    monkeypatch.setattr(
        nvcomp_batch, "_candidate_cuda_homes", lambda: iter([toolkit])
    )
    loaded = []

    def load_library(path):
        loaded.append(path)
        if not available:
            raise OSError("CUDA 13 runtime is unavailable")

    monkeypatch.setattr(nvcomp_batch, "_load_shared_library", load_library)

    if available:
        nvcomp_batch._preload_cudart()
    else:
        with pytest.raises(
            ImportError, match="CUDA 13 runtime is unavailable"
        ):
            nvcomp_batch._preload_cudart()

    assert loaded == ["libcudart.so.13"]


def test_runtime_falls_back_after_unloadable_library(monkeypatch, tmp_path):
    toolkit = _cuda_tree(tmp_path / "broken-cuda")
    monkeypatch.setattr(
        nvcomp_batch, "_candidate_cuda_homes", lambda: iter([toolkit])
    )
    loaded = []

    def load_library(path):
        loaded.append(path)
        if isinstance(path, Path):
            raise OSError("invalid ELF header")

    monkeypatch.setattr(nvcomp_batch, "_load_shared_library", load_library)

    nvcomp_batch._preload_cudart()

    assert loaded == [
        toolkit / "lib" / "libcudart.so.13",
        "libcudart.so.13",
    ]


def test_build_rejects_toolkit_without_cuda13_runtime(monkeypatch, tmp_path):
    toolkit = _cuda_tree(tmp_path / "old-cuda", library="libcudart.so")
    monkeypatch.setattr(
        setup_package, "_cuda_home_candidates", lambda: iter([toolkit])
    )

    with pytest.raises(RuntimeError, match="libcudart.so.13"):
        setup_package._find_cuda_toolkit()


def test_cuda_package_lookup_handles_absent_nvidia_namespace(monkeypatch):
    monkeypatch.setitem(sys.modules, "nvidia", None)
    monkeypatch.delitem(sys.modules, "nvidia.cu13", raising=False)

    assert nvcomp_batch._package_dir("nvidia.cu13") is None
    with pytest.raises(RuntimeError, match="not importable: nvidia.cu13"):
        setup_package._package_dir("nvidia.cu13")


def test_build_skips_runtime_wheel_without_crt(monkeypatch, tmp_path):
    runtime = _cuda_tree(tmp_path / "runtime-only")
    (runtime / "include" / "crt" / "host_config.h").unlink()
    toolkit = _cuda_tree(tmp_path / "complete-sdk")
    monkeypatch.setattr(
        setup_package,
        "_cuda_home_candidates",
        lambda: iter([runtime, toolkit]),
    )

    assert setup_package._find_cuda_toolkit() == (
        toolkit / "include",
        toolkit / "lib",
    )
