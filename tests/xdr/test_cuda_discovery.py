# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import platform
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

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
    monkeypatch.delenv("CUPHOTON_XDR_NATIVE_PREFIX", raising=False)
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


@pytest.fixture
def conda_native_prefix(monkeypatch, tmp_path):
    prefix = tmp_path / "conda"
    (prefix / "conda-meta").mkdir(parents=True)
    (prefix / "lib").mkdir()
    (prefix / "include").mkdir()
    for name in (
        "libcudart.so.13",
        "libnvcomp.so.5",
        "librapids_logger.so",
        "libkvikio.so",
        "libcfitsio.so",
    ):
        (prefix / "lib" / name).touch()
    for name in ("fitsio.h", "cufile.h"):
        (prefix / "include" / name).touch()
    monkeypatch.setattr(sys, "prefix", str(prefix))
    monkeypatch.setattr(nvcomp_batch, "_package_dir", lambda _: None)
    monkeypatch.setitem(
        sys.modules,
        "pybind11",
        SimpleNamespace(get_include=lambda: str(prefix / "include")),
    )
    for name in (
        "CUDA_HOME",
        "CUDA_PATH",
        "CUPHOTON_XDR_NATIVE_PREFIX",
        "CUPHOTON_XDR_CFITSIO_ROOT",
        "CUPHOTON_XDR_NVCOMP_LIB_DIR",
        "CUPHOTON_XDR_KVIKIO_LIB_DIR",
        "CUPHOTON_XDR_RAPIDS_LOGGER_LIB_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    return prefix


@pytest.mark.parametrize(
    ("architecture", "target_name"),
    [("x86_64", "x86_64-linux"), ("aarch64", "sbsa-linux")],
)
def test_conda_build_uses_prefix_headers_and_libraries(
    monkeypatch, conda_native_prefix, architecture, target_name
):
    monkeypatch.setattr(platform, "machine", lambda: architecture)
    prefix = conda_native_prefix
    target = _cuda_tree(prefix / "targets" / target_name)
    monkeypatch.setenv("CUPHOTON_XDR_NATIVE_PREFIX", str(prefix))

    def unexpected_wheel_lookup(name):
        pytest.fail(f"Conda build looked for a wheel package: {name}")

    monkeypatch.setattr(
        setup_package, "_package_dir", unexpected_wheel_lookup
    )

    (extension,) = setup_package.get_extensions()

    assert str(prefix / "include") in extension.include_dirs
    assert str(target / "include") in extension.include_dirs
    assert str(prefix / "lib") in extension.library_dirs
    assert str(target / "lib") in extension.library_dirs
    assert "-lcfitsio" in extension.extra_link_args


def test_explicit_native_prefix_cannot_fall_back_to_cuda_wheel(
    monkeypatch, tmp_path, cuda_wheel
):
    monkeypatch.setenv("CUPHOTON_XDR_NATIVE_PREFIX", str(tmp_path / "empty"))

    with pytest.raises(RuntimeError, match="libcudart.so.13"):
        setup_package._find_cuda_toolkit()


@pytest.mark.parametrize("target_layout", [False, True])
@pytest.mark.parametrize(
    ("architecture", "target_name"),
    [("x86_64", "x86_64-linux"), ("aarch64", "sbsa-linux")],
)
def test_conda_runtime_uses_python_prefix_not_shell_prefix(
    monkeypatch,
    tmp_path,
    conda_native_prefix,
    target_layout,
    architecture,
    target_name,
):
    monkeypatch.setattr(platform, "machine", lambda: architecture)
    monkeypatch.setenv("CONDA_PREFIX", str(tmp_path / "unrelated"))
    loaded = []
    monkeypatch.setattr(nvcomp_batch, "_load_shared_library", loaded.append)
    runtime = conda_native_prefix / "lib" / "libcudart.so.13"
    if target_layout:
        target_runtime = (
            conda_native_prefix
            / "targets"
            / target_name
            / "lib"
            / runtime.name
        )
        target_runtime.parent.mkdir(parents=True)
        runtime.rename(target_runtime)
        runtime = target_runtime

    nvcomp_batch._preload_cudart()
    nvcomp_batch._preload_gpu_package_libraries()

    assert loaded == [runtime] + [
        conda_native_prefix / "lib" / name
        for name in (
            "libnvcomp.so.5",
            "librapids_logger.so",
            "libkvikio.so",
        )
    ]


def test_conda_runtime_honors_explicit_library_override(
    monkeypatch, tmp_path, conda_native_prefix
):
    override = tmp_path / "selected"
    override.mkdir()
    (override / "libnvcomp.so.5").touch()
    monkeypatch.setenv("CUPHOTON_XDR_NVCOMP_LIB_DIR", str(override))
    loaded = []
    monkeypatch.setattr(nvcomp_batch, "_load_shared_library", loaded.append)

    nvcomp_batch._preload_gpu_package_libraries()

    assert loaded[0] == override / "libnvcomp.so.5"


def test_conda_missing_library_does_not_search_nested_environments(
    monkeypatch, conda_native_prefix
):
    prefix = conda_native_prefix
    library = prefix / "lib" / "libnvcomp.so.5"
    nested_library = prefix / "envs" / "unrelated" / "lib" / library.name
    nested_library.parent.mkdir(parents=True)
    library.rename(nested_library)
    monkeypatch.setenv("CUPHOTON_XDR_NATIVE_PREFIX", str(prefix))

    with pytest.raises(ImportError, match="Could not find nvCOMP library"):
        nvcomp_batch._preload_gpu_package_libraries()
    with pytest.raises(RuntimeError, match="Could not find nvCOMP library"):
        setup_package._find_gpu_package_paths()


def test_plain_python_prefix_is_not_treated_as_conda(conda_native_prefix):
    (conda_native_prefix / "conda-meta").rmdir()

    with pytest.raises(ImportError, match="Python packages required"):
        nvcomp_batch._preload_gpu_package_libraries()


@pytest.mark.parametrize(
    "missing_package",
    [None, "nvidia.libnvcomp", "rapids_logger", "libkvikio"],
)
def test_gpu_wheels_remain_usable_inside_conda(
    monkeypatch, tmp_path, conda_native_prefix, missing_package
):
    package_dirs = {}
    expected = []
    for module, library in (
        ("nvidia.cu13", "libcudart.so.13"),
        ("nvidia.libnvcomp", "libnvcomp.so.5"),
        ("rapids_logger", "librapids_logger.so"),
        ("libkvikio", "libkvikio.so"),
    ):
        if module == missing_package:
            expected.append(conda_native_prefix / "lib" / library)
            continue
        package = tmp_path / "site-packages" / module
        (package / "lib").mkdir(parents=True)
        path = package / "lib" / library
        path.touch()
        package_dirs[module] = package
        expected.append(path)
    monkeypatch.setattr(
        nvcomp_batch, "_package_dir", lambda name: package_dirs.get(name)
    )
    loaded = []
    monkeypatch.setattr(nvcomp_batch, "_load_shared_library", loaded.append)

    nvcomp_batch._preload_cudart()
    nvcomp_batch._preload_gpu_package_libraries()

    assert loaded == expected
