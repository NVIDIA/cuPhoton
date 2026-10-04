# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build the optional xFit extension with an explicitly selected CUDA SDK."""

from __future__ import annotations

import os
import platform
import re
from pathlib import Path

from setuptools import Extension
from setuptools.command.build_ext import build_ext

_SOURCE = Path("src/cuphoton/xfit/src")
# Explicit arithmetic boundaries preserve the array path's rounding while
# CUDA math functions retain fused operations internally.
_MAIN_FMAD = True
_MAIN_FTZ = True
_NORM_FMAD = False


def _architectures() -> tuple[int, ...]:
    value = os.environ.get("CUPHOTON_XFIT_CUDA_ARCHS", "75")
    parts = re.split(r"[,;\s]+", value.strip())
    if not parts or any(
        not re.fullmatch(r"[0-9]{2,3}", part) or int(part) < 75
        for part in parts
    ):
        raise RuntimeError(
            "CUPHOTON_XFIT_CUDA_ARCHS must contain CUDA architecture "
            "numbers >= 75 separated by commas, semicolons, or spaces"
        )
    return tuple(sorted({int(part) for part in parts}))


class CUDAExtension(Extension):
    """An ordinary Python extension with separately compiled CUDA files."""

    def __init__(
        self, cuda_home: Path, include: Path, libraries: Path, cccl: Path
    ):
        super().__init__(
            name="cuphoton.xfit._native_ext",
            sources=[str(_SOURCE / "native_ext.cpp")],
            depends=[
                str(_SOURCE / "native_api.h"),
                str(_SOURCE / "native.cu"),
                str(_SOURCE / "native_norm.cu"),
                str(_SOURCE / "native_jacobian.h"),
                str(_SOURCE / "native_jacobian.cu"),
                str(_SOURCE / "native_jacobian_lean.cu"),
            ],
            language="c++",
            include_dirs=[
                # Selected CCCL precedes any headers bundled with the SDK.
                *(
                    str(cccl / name)
                    for name in ("cub", "libcudacxx", "thrust")
                ),
                str(include),
            ],
            library_dirs=[str(libraries)],
            # CUDA wheels ship SONAME files without unversioned symlinks.
            libraries=[":libcudart.so.13", ":libcublas.so.13"],
            extra_compile_args=[
                "-O3",
                "-std=c++17",
                "-fvisibility=hidden",
                "-pthread",
            ],
            extra_link_args=["-pthread"],
        )
        self.nvcc = str(cuda_home / "bin" / "nvcc")
        self.architectures = _architectures()


class CUDABuildExt(build_ext):
    """Leave other extensions on setuptools' normal compiler path."""

    def build_extension(self, extension):
        if not isinstance(extension, CUDAExtension):
            super().build_extension(extension)
            return
        object_directory = Path(self.build_temp) / "xfit"
        object_directory.mkdir(parents=True, exist_ok=True)
        command = [
            extension.nvcc,
            "-O3",
            "-std=c++17",
            "-Xcompiler=-fPIC,-fvisibility=hidden,-pthread",
        ]
        for include in extension.include_dirs:
            command.extend(("-I", include))
        objects = []
        for name, fmad, ftz, architectures in (
            ("native", _MAIN_FMAD, _MAIN_FTZ, extension.architectures),
            ("native_norm", _NORM_FMAD, True, extension.architectures),
            # Match native transcendental contraction; explicit RN intrinsics
            # preserve the Jacobian's individual arithmetic boundaries.
            ("native_jacobian", True, False, (100,)),
            ("native_jacobian_lean", True, False, (100,)),
        ):
            architecture_flags = [
                f"--generate-code=arch=compute_{architecture},"
                f"code=sm_{architecture}"
                for architecture in architectures
            ]
            # Keep PTX for each unit's highest target, including the SM 75
            # default used by the portable native and norm kernels.
            architecture = architectures[-1]
            architecture_flags.append(
                f"--generate-code=arch=compute_{architecture},"
                f"code=compute_{architecture}"
            )
            object_path = object_directory / f"{name}.o"
            compile_command = [
                *command,
                *architecture_flags,
                "-c",
                str(_SOURCE / f"{name}.cu"),
                "-o",
                str(object_path),
                f"--fmad={str(fmad).lower()}",
                f"--ftz={str(ftz).lower()}",
            ]
            if name == "native_norm":
                compile_command.append("-DCUB_DISABLE_BF16_SUPPORT")
            self.spawn(compile_command)
            objects.append(str(object_path))
        previous = extension.extra_objects
        previous_depends = extension.depends
        extension.extra_objects = [*(previous or ()), *objects]
        # setuptools checks sources/depends before considering extra_objects.
        # Relink all freshly compiled CUDA objects even when only SDK or
        # architecture settings changed since the previous extension build.
        extension.depends = [*(previous_depends or ()), *objects]
        try:
            super().build_extension(extension)
        finally:
            extension.extra_objects = previous
            extension.depends = previous_depends


def get_extensions():
    """Validate the SDK without loading CUDA or initializing a device."""
    if platform.system() != "Linux":
        raise RuntimeError("The native xFit extension supports Linux only")
    value = os.environ.get("CUDA_HOME")
    if not value:
        raise RuntimeError(
            "Set CUDA_HOME to a CUDA 13 toolkit to build native xFit"
        )
    cuda_home = Path(value).resolve()
    include = cuda_home / "include"
    version_header = include / "cuda_runtime_api.h"
    if (
        not version_header.is_file()
        or not (include / "cuda_runtime.h").is_file()
    ):
        raise RuntimeError(f"CUDA_HOME={cuda_home} lacks CUDA headers")
    match = re.search(
        r"^\s*#\s*define\s+CUDART_VERSION\s+(\d+)",
        version_header.read_text(),
        re.MULTILINE,
    )
    if match is None or int(match.group(1)) // 1000 != 13:
        raise RuntimeError("Native xFit requires a CUDA 13 toolkit")
    if not os.access(cuda_home / "bin" / "nvcc", os.X_OK):
        raise RuntimeError(f"CUDA_HOME={cuda_home} lacks executable bin/nvcc")
    cublas_header = include / "cublas_api.h"
    if not (include / "cublas_v2.h").is_file() or not cublas_header.is_file():
        raise RuntimeError(f"CUDA_HOME={cuda_home} lacks cuBLAS headers")
    cublas_version = re.search(
        r"^\s*#\s*define\s+CUBLAS_VER_MAJOR\s+(\d+)",
        cublas_header.read_text(),
        re.MULTILINE,
    )
    if cublas_version is None or int(cublas_version.group(1)) != 13:
        raise RuntimeError("Native xFit requires cuBLAS 13")
    library_names = (
        "libcudart.so.13",
        "libcublas.so.13",
        "libcublasLt.so.13",
    )
    libraries = next(
        (
            cuda_home / name
            for name in ("lib64", "lib")
            if all(
                (cuda_home / name / library).is_file()
                for library in library_names
            )
        ),
        None,
    )
    if libraries is None:
        raise RuntimeError(
            f"CUDA_HOME={cuda_home} lacks CUDA 13 runtime libraries "
            f"({', '.join(library_names)})"
        )
    cccl_value = os.environ.get("CUPHOTON_XFIT_CCCL_ROOT")
    if not cccl_value:
        raise RuntimeError(
            "Set CUPHOTON_XFIT_CCCL_ROOT to the selected CuPy "
            "wheel's _cccl header directory"
        )
    cccl = Path(cccl_value).expanduser().resolve()
    for header in (
        "cub/cub/block/block_reduce.cuh",
        "cub/cub/version.cuh",
        "libcudacxx/cuda/std/type_traits",
        "libcudacxx/cuda/std/__cccl/version.h",
        "thrust/thrust/version.h",
    ):
        if not (cccl / header).is_file():
            raise RuntimeError(
                f"CUPHOTON_XFIT_CCCL_ROOT={cccl} lacks {header}"
            )
    return [CUDAExtension(cuda_home, include, libraries, cccl)]
