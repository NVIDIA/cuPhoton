# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

from setuptools import Extension

ROOT = Path(__file__).parent
SRC_DIR = ROOT / "src"
_NVCOMP_LIB_ENV = "CUPHOTON_XDR_NVCOMP_LIB_DIR"
_KVIKIO_LIB_ENV = "CUPHOTON_XDR_KVIKIO_LIB_DIR"
_CFITSIO_ROOT_ENV = "CUPHOTON_XDR_CFITSIO_ROOT"


def _unique(paths):
    out = []
    seen = set()
    for path in paths:
        text = str(path)
        if text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _require_dir(path: Path, description: str) -> Path:
    if not path.is_dir():
        raise RuntimeError(f"Missing {description}: {path}")
    return path


def _package_dir(module_name: str) -> Path:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        raise RuntimeError(
            f"Required Python package is not importable: {module_name}"
        )
    locations = spec.submodule_search_locations
    if locations:
        return Path(next(iter(locations)))
    if spec.origin:
        return Path(spec.origin).parent
    raise RuntimeError(
        f"Cannot determine install path for package: {module_name}"
    )


def _library_in_dir(lib_dir: Path, names: tuple[str, ...]) -> bool:
    return any((lib_dir / name).is_file() for name in names)


def _find_library_dir(
    package_base: Path,
    names: tuple[str, ...],
    *,
    env_var: str,
    description: str,
) -> Path:
    env_value = os.environ.get(env_var)
    if env_value:
        lib_dir = Path(env_value)
        if _library_in_dir(lib_dir, names):
            return lib_dir
        raise RuntimeError(
            f"{env_var}={lib_dir} does not contain {description}; "
            f"expected one of: {', '.join(names)}"
        )

    for child in ("lib64", "lib"):
        lib_dir = package_base / child
        if _library_in_dir(lib_dir, names):
            return lib_dir

    for name in names:
        matches = sorted(package_base.rglob(name))
        if matches:
            return matches[0].parent

    raise RuntimeError(
        f"Could not find {description} under {package_base}. "
        "Searched lib64, lib, and recursive matches for: "
        f"{', '.join(names)}. "
        f"Set {env_var} to the directory containing the library."
    )


def _cuda_home_candidates():
    seen = set()
    for value in (
        os.environ.get("CUDA_HOME"),
        os.environ.get("CUDA_PATH"),
    ):
        if value:
            path = Path(value)
            if path not in seen:
                seen.add(path)
                yield path
    default = Path("/usr/local/cuda")
    if default not in seen:
        seen.add(default)
        yield default
    nvcc = shutil.which("nvcc")
    if nvcc:
        path = Path(nvcc).resolve().parents[1]
        if path not in seen:
            yield path


def _find_cuda_toolkit():
    for cuda_home in _cuda_home_candidates():
        include_dir = cuda_home / "include"
        if not (include_dir / "cuda_runtime.h").is_file():
            continue

        for lib_name in ("lib64", "lib"):
            lib_dir = cuda_home / lib_name
            if any(
                (lib_dir / soname).is_file()
                for soname in (
                    "libcudart.so",
                    "libcudart.so.13",
                )
            ):
                return include_dir, lib_dir

    raise RuntimeError(
        "CUDA toolkit with cuda_runtime.h and libcudart is required to build "
        "cuphoton.xdr._nvcomp_batch_ext. Set CUDA_HOME to the "
        "toolkit root."
    )


def _find_gpu_package_paths():
    import pybind11

    pybind11_include = Path(pybind11.get_include())
    kvikio_base = _package_dir("libkvikio")
    nvcomp_base = _package_dir("nvidia.libnvcomp")
    kvikio_include = kvikio_base / "include"
    kvikio_lib = _find_library_dir(
        kvikio_base,
        ("libkvikio.so",),
        env_var=_KVIKIO_LIB_ENV,
        description="KvikIO library",
    )
    nvcomp_include = nvcomp_base / "include"
    nvcomp_lib = _find_library_dir(
        nvcomp_base,
        ("libnvcomp.so.5", "libnvcomp.so"),
        env_var=_NVCOMP_LIB_ENV,
        description="nvCOMP library",
    )

    return {
        "pybind11_include": _require_dir(
            pybind11_include, "pybind11 include directory"
        ),
        "kvikio_include": _require_dir(
            kvikio_include, "libkvikio include directory"
        ),
        "kvikio_lib": _require_dir(kvikio_lib, "libkvikio library directory"),
        "nvcomp_include": _require_dir(
            nvcomp_include, "nvCOMP include directory"
        ),
        "nvcomp_lib": _require_dir(nvcomp_lib, "nvCOMP library directory"),
    }


def _find_cufile_include(cuda_include: Path) -> Path:
    if (cuda_include / "cufile.h").is_file():
        return cuda_include

    try:
        candidate = _package_dir("nvidia.cufile") / "include"
        if (candidate / "cufile.h").is_file():
            return candidate
    except RuntimeError:
        pass

    raise RuntimeError(
        "cuFile header cufile.h is required to build "
        "cuphoton.xdr._nvcomp_batch_ext. Install a CUDA toolkit "
        "that includes cuFile headers or set CUDA_HOME."
    )


def _split_pkg_config_flags(flags: str, prefix: str) -> list[str]:
    out = []
    for token in flags.split():
        if token.startswith(prefix) and len(token) > len(prefix):
            out.append(token[len(prefix) :])
    return out


def _find_cfitsio_paths():
    env_value = os.environ.get(_CFITSIO_ROOT_ENV)
    if env_value:
        root = Path(env_value)
        include_candidates = [root / "include", root]
        lib_candidates = [root / "lib64", root / "lib", root]
        include_dir = next(
            (
                path
                for path in include_candidates
                if (path / "fitsio.h").is_file()
            ),
            None,
        )
        lib_dir = next(
            (
                path
                for path in lib_candidates
                if any(
                    (path / name).is_file()
                    for name in ("libcfitsio.so", "libcfitsio.a")
                )
            ),
            None,
        )
        if include_dir is None or lib_dir is None:
            raise RuntimeError(
                f"{_CFITSIO_ROOT_ENV}={root} must contain fitsio.h and "
                "libcfitsio"
            )
        return {
            "include_dirs": [str(include_dir)],
            "library_dirs": [str(lib_dir)],
            "extra_link_args": [f"-Wl,-rpath,{lib_dir}", "-lcfitsio"],
        }

    pkg_config = shutil.which("pkg-config")
    if pkg_config:
        exists = subprocess.run(
            [pkg_config, "--exists", "cfitsio"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if exists.returncode == 0:
            cflags = subprocess.check_output(
                [pkg_config, "--cflags", "cfitsio"], text=True
            )
            libs = subprocess.check_output(
                [pkg_config, "--libs", "cfitsio"], text=True
            )
            library_dirs = _split_pkg_config_flags(libs, "-L")
            link_args = [
                token for token in libs.split() if not token.startswith("-L")
            ]
            for lib_dir in library_dirs:
                link_args.append(f"-Wl,-rpath,{lib_dir}")
            return {
                "include_dirs": _split_pkg_config_flags(cflags, "-I"),
                "library_dirs": library_dirs,
                "extra_link_args": link_args,
            }

    raise RuntimeError(
        "CFITSIO is required to build "
        "cuphoton.xdr._nvcomp_batch_ext. "
        "Install the cfitsio development package with pkg-config metadata, "
        "or set "
        f"{_CFITSIO_ROOT_ENV} to a CFITSIO installation prefix."
    )


def get_extensions():
    cuda_include, cuda_lib = _find_cuda_toolkit()
    cufile_include = _find_cufile_include(cuda_include)
    package_paths = _find_gpu_package_paths()
    cfitsio_paths = _find_cfitsio_paths()

    include_dirs = _unique(
        [
            package_paths["pybind11_include"],
            package_paths["nvcomp_include"],
            package_paths["kvikio_include"],
            cufile_include,
            cuda_include,
            *cfitsio_paths["include_dirs"],
        ]
    )
    library_dirs = _unique(
        [
            package_paths["nvcomp_lib"],
            package_paths["kvikio_lib"],
            cuda_lib,
            *cfitsio_paths["library_dirs"],
        ]
    )

    return [
        Extension(
            name="cuphoton.xdr._nvcomp_batch_ext",
            sources=[
                os.path.relpath(SRC_DIR / "nvcomp_batch_ext.cpp"),
                os.path.relpath(SRC_DIR / "memory_manager.cpp"),
                os.path.relpath(SRC_DIR / "io.cpp"),
            ],
            language="c++",
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            define_macros=[
                ("KVIKIO_CUFILE_FOUND", "1"),
                ("KVIKIO_CUFILE_VERSION_API_FOUND", "1"),
            ],
            extra_compile_args=[
                "-O3",
                "-Wall",
                "-std=c++17",
                "-fvisibility=hidden",
                "-pthread",
            ],
            extra_link_args=[
                "-pthread",
                "-l:libnvcomp.so.5",
                "-l:libkvikio.so",
                "-lcudart",
                *cfitsio_paths["extra_link_args"],
            ],
        )
    ]
