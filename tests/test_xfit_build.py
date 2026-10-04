# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""CPU-only checks for optional native xFit source builds."""

from __future__ import annotations

import importlib.util
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import setuptools

_ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="native xFit builds require Linux"
)


def _helpers():
    spec = importlib.util.spec_from_file_location(
        "_test_xfit_setup", _ROOT / "src/cuphoton/xfit/setup_package.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _toolkit(tmp_path, version=13000, library_dir="lib64"):
    root = tmp_path / "cuda"
    for name in ("include", "bin", library_dir):
        (root / name).mkdir(parents=True)
    (root / "include/cuda_runtime_api.h").write_text(
        f"#define CUDART_VERSION {version}\n"
    )
    (root / "include/cublas_api.h").write_text(
        "#define CUBLAS_VER_MAJOR 13\n"
    )
    for name in (
        "include/cuda_runtime.h",
        "include/cublas_v2.h",
        f"{library_dir}/libcudart.so.13",
        f"{library_dir}/libcublas.so.13",
        f"{library_dir}/libcublasLt.so.13",
    ):
        (root / name).touch()
    nvcc = root / "bin/nvcc"
    nvcc.write_text("#!/bin/sh\nexit 1\n")
    nvcc.chmod(0o755)
    return root


def _cccl(tmp_path):
    root = tmp_path / "cccl"
    for name in (
        "cub/cub/block/block_reduce.cuh",
        "cub/cub/version.cuh",
        "libcudacxx/cuda/std/type_traits",
        "libcudacxx/cuda/std/__cccl/version.h",
        "thrust/thrust/version.h",
    ):
        header = root / name
        header.parent.mkdir(parents=True, exist_ok=True)
        header.touch()
    return root


def test_default_build_does_not_discover_cuda(monkeypatch):
    captured = {}
    monkeypatch.delenv("CUPHOTON_XDR_BUILD_EXT", raising=False)
    monkeypatch.delenv("CUPHOTON_XFIT_BUILD_EXT", raising=False)
    monkeypatch.setenv("CUDA_HOME", "/not/a/toolkit")
    monkeypatch.setenv("CUPHOTON_XFIT_CCCL_ROOT", "/not/cccl")
    monkeypatch.setattr(setuptools, "setup", lambda **kw: captured.update(kw))

    def reject_helper(*args, **kwargs):
        raise AssertionError(
            "disabled extensions must not load build helpers"
        )

    monkeypatch.setattr(
        importlib.util, "spec_from_file_location", reject_helper
    )
    runpy.run_path(str(_ROOT / "setup.py"))
    assert captured["ext_modules"] == []


def test_explicit_build_requires_cuda_13(monkeypatch, tmp_path):
    helper = _helpers()
    monkeypatch.delenv("CUDA_HOME", raising=False)
    with pytest.raises(RuntimeError, match="Set CUDA_HOME"):
        helper.get_extensions()
    root = _toolkit(tmp_path, version=12090)
    monkeypatch.setenv("CUDA_HOME", str(root))
    with pytest.raises(RuntimeError, match="CUDA 13"):
        helper.get_extensions()


@pytest.mark.parametrize("library_dir", ["lib64", "lib"])
def test_native_build_is_enabled_independently(
    monkeypatch, tmp_path, library_dir
):
    root = _toolkit(tmp_path, library_dir=library_dir)
    cccl = _cccl(tmp_path)
    captured = {}
    monkeypatch.setenv("CUDA_HOME", str(root))
    monkeypatch.setenv("CUPHOTON_XFIT_CCCL_ROOT", str(cccl))
    monkeypatch.setenv("CUPHOTON_XDR_BUILD_EXT", "0")
    monkeypatch.setenv("CUPHOTON_XFIT_BUILD_EXT", "1")
    monkeypatch.setenv("CUPHOTON_XFIT_CUDA_ARCHS", "100;120")
    monkeypatch.setattr(setuptools, "setup", lambda **kw: captured.update(kw))
    runpy.run_path(str(_ROOT / "setup.py"))
    (extension,) = captured["ext_modules"]
    assert extension.name == "cuphoton.xfit._native_ext"
    assert extension.architectures == (100, 120)
    assert extension.py_limited_api is False
    assert extension.library_dirs == [str(root / library_dir)]
    assert extension.libraries == [":libcudart.so.13", ":libcublas.so.13"]
    assert extension.include_dirs == [
        str(cccl / "cub"),
        str(cccl / "libcudacxx"),
        str(cccl / "thrust"),
        str(root / "include"),
    ]
    assert {
        "src/cuphoton/xfit/src/native_norm.cu",
        "src/cuphoton/xfit/src/native_jacobian.h",
        "src/cuphoton/xfit/src/native_jacobian.cu",
        "src/cuphoton/xfit/src/native_jacobian_lean.cu",
    } <= set(extension.depends)
    assert "build_ext" in captured["cmdclass"]


def test_native_and_xdr_builds_can_be_enabled_together(monkeypatch):
    captured = {}
    xdr_extension = setuptools.Extension("cuphoton.xdr._native_ext", [])
    xfit_extension = setuptools.Extension("cuphoton.xfit._native_ext", [])
    build_command = object()
    helpers = {
        "_cuphoton_xdr_setup_package": SimpleNamespace(
            get_extensions=lambda: [xdr_extension]
        ),
        "_cuphoton_xfit_setup_package": SimpleNamespace(
            get_extensions=lambda: [xfit_extension],
            CUDABuildExt=build_command,
        ),
    }
    loader = SimpleNamespace(
        create_module=lambda spec: helpers[spec.name],
        exec_module=lambda module: None,
    )
    monkeypatch.setenv("CUPHOTON_XDR_BUILD_EXT", "1")
    monkeypatch.setenv("CUPHOTON_XFIT_BUILD_EXT", "1")
    monkeypatch.setattr(setuptools, "setup", lambda **kw: captured.update(kw))
    monkeypatch.setattr(
        importlib.util,
        "spec_from_file_location",
        lambda name, path: importlib.machinery.ModuleSpec(name, loader),
    )

    runpy.run_path(str(_ROOT / "setup.py"))

    assert captured["ext_modules"] == [xdr_extension, xfit_extension]
    assert captured["cmdclass"] == {"build_ext": build_command}


def test_native_build_requires_versioned_runtime(monkeypatch, tmp_path):
    root = _toolkit(tmp_path)
    (root / "lib64/libcudart.so.13").unlink()
    (root / "lib64/libcudart.so").touch()
    monkeypatch.setenv("CUDA_HOME", str(root))
    with pytest.raises(RuntimeError, match=r"libcudart\.so\.13"):
        _helpers().get_extensions()


def test_native_build_requires_cublas_13(monkeypatch, tmp_path):
    root = _toolkit(tmp_path)
    monkeypatch.setenv("CUDA_HOME", str(root))
    (root / "include/cublas_api.h").write_text(
        "#define CUBLAS_VER_MAJOR 12\n"
    )
    with pytest.raises(RuntimeError, match="cuBLAS 13"):
        _helpers().get_extensions()


def test_native_build_requires_explicit_complete_cccl(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_HOME", str(_toolkit(tmp_path)))
    monkeypatch.delenv("CUPHOTON_XFIT_CCCL_ROOT", raising=False)
    with pytest.raises(RuntimeError, match="Set CUPHOTON_XFIT_CCCL_ROOT"):
        _helpers().get_extensions()
    cccl = _cccl(tmp_path)
    (cccl / "cub/cub/block/block_reduce.cuh").unlink()
    monkeypatch.setenv("CUPHOTON_XFIT_CCCL_ROOT", str(cccl))
    with pytest.raises(RuntimeError, match="block_reduce.cuh"):
        _helpers().get_extensions()


@pytest.mark.parametrize("architectures", [None, "89", "100;120"])
def test_cuda_targets_math_flags_and_relink_dependencies(
    monkeypatch, tmp_path, architectures
):
    monkeypatch.setenv("CUDA_HOME", str(_toolkit(tmp_path)))
    monkeypatch.setenv("CUPHOTON_XFIT_CCCL_ROOT", str(_cccl(tmp_path)))
    if architectures is None:
        monkeypatch.delenv("CUPHOTON_XFIT_CUDA_ARCHS", raising=False)
    else:
        monkeypatch.setenv("CUPHOTON_XFIT_CUDA_ARCHS", architectures)
    helper = _helpers()
    (extension,) = helper.get_extensions()
    previous_objects, previous_depends = (
        extension.extra_objects,
        extension.depends,
    )
    command = helper.CUDABuildExt(setuptools.Distribution())
    command.build_temp = str(tmp_path / "build")
    compiled = {}

    def compile_object(arguments):
        source = Path(arguments[arguments.index("-c") + 1]).name
        assert source not in compiled
        compiled[source] = arguments
        Path(arguments[arguments.index("-o") + 1]).touch()

    def fail_link(self, linked):
        assert set(compiled) == {
            "native.cu",
            "native_norm.cu",
            "native_jacobian.cu",
            "native_jacobian_lean.cu",
        }
        assert {Path(path).name for path in linked.extra_objects} == {
            "native.o",
            "native_norm.o",
            "native_jacobian.o",
            "native_jacobian_lean.o",
        }
        assert all(path in linked.depends for path in linked.extra_objects)
        portable_targets = {
            None: {"75"},
            "89": {"89"},
            "100;120": {"100", "120"},
        }[architectures]
        for source, arguments in compiled.items():
            specialized = source.startswith("native_jacobian")
            targets = {"100"} if specialized else portable_targets
            ptx_target = max(targets, key=int)
            assert {
                flag
                for flag in arguments
                if flag.startswith("--generate-code=")
            } == {
                *(
                    f"--generate-code=arch=compute_{target},code=sm_{target}"
                    for target in targets
                ),
                f"--generate-code=arch=compute_{ptx_target},"
                f"code=compute_{ptx_target}",
            }
            assert f"--fmad={str(source == 'native.cu').lower()}" in arguments
            assert f"--ftz={str(not specialized).lower()}" in arguments
            assert ("-DCUB_DISABLE_BF16_SUPPORT" in arguments) == (
                source == "native_norm.cu"
            )
        raise RuntimeError("link failed")

    monkeypatch.setattr(command, "spawn", compile_object)
    monkeypatch.setattr(helper.build_ext, "build_extension", fail_link)
    with pytest.raises(RuntimeError, match="link failed"):
        command.build_extension(extension)
    assert extension.extra_objects is previous_objects
    assert extension.depends is previous_depends


@pytest.mark.parametrize(
    "architectures", ["", "70", "120a", "120 --fast-math"]
)
def test_invalid_architectures_fail_before_compilation(
    monkeypatch, tmp_path, architectures
):
    monkeypatch.setenv("CUDA_HOME", str(_toolkit(tmp_path)))
    monkeypatch.setenv("CUPHOTON_XFIT_CCCL_ROOT", str(_cccl(tmp_path)))
    monkeypatch.setenv("CUPHOTON_XFIT_CUDA_ARCHS", architectures)
    with pytest.raises(RuntimeError, match="CUPHOTON_XFIT_CUDA_ARCHS"):
        _helpers().get_extensions()
