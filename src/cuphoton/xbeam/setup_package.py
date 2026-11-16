# SPDX-License-Identifier: Apache-2.0
"""Build helpers for the optional native execution extension."""

import os
import subprocess
from pathlib import Path

from setuptools import Extension


def get_extensions():
    from importlib import import_module

    cythonize = import_module("Cython.Build").cythonize

    source = Path("src/cuphoton/xbeam/native")
    libraries = ["crypto"]
    library_dirs = []
    macros = []
    runtime_dirs = []
    cuda_enabled = os.environ.get("CUPHOTON_XBEAM_NATIVE_CUDA", "0") == "1"
    mode_file = Path("build/xbeam-native-binding-mode").resolve()
    mode = "cuda\n" if cuda_enabled else "cpu\n"
    mode_file.parent.mkdir(parents=True, exist_ok=True)
    if not mode_file.exists() or mode_file.read_text() != mode:
        mode_file.write_text(mode)
    depends = [
        str(mode_file),
        *(str(path) for path in sorted((source / "include").rglob("*.hpp"))),
        str(source / "third_party/nlohmann/json.hpp"),
    ]
    if cuda_enabled:
        cuda = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
        build = Path("build/xbeam-native-cuda").resolve()
        subprocess.run(
            [
                "cmake",
                "-S",
                str(source.resolve()),
                "-B",
                str(build),
                "-DCMAKE_BUILD_TYPE=Release",
                "-DXBEAM_ENABLE_CUDA=ON",
                "-DXBEAM_ENABLE_CUFILE=ON",
                f"-DCUDAToolkit_ROOT={cuda}",
                f"-DCMAKE_CUDA_COMPILER={cuda / 'bin/nvcc'}",
                "-DCMAKE_CUDA_ARCHITECTURES="
                + os.environ.get(
                    "CUPHOTON_XBEAM_CUDA_ARCHITECTURES", "native"
                ),
            ],
            check=True,
        )
        subprocess.run(
            [
                "cmake",
                "--build",
                str(build),
                "--target",
                "xbeam_execution",
                "-j4",
            ],
            check=True,
        )
        libraries += [
            "xbeam_execution",
            "xbeam_cuda",
            "cufft",
            "cudart",
            "cufile",
        ]
        library_dirs += [str(build), str(cuda / "lib64")]
        runtime_dirs += [str(cuda / "lib64")]
        macros += [("XBEAM_BIND_CUDA", "1")]
        depends += [
            str(build / "libxbeam_execution.a"),
            str(build / "libxbeam_cuda.a"),
        ]
    return cythonize(
        [
            Extension(
                "cuphoton.xbeam.native._binding",
                [
                    str(source / "_binding.pyx"),
                    str(source / "src/vdif.cpp"),
                    str(source / "src/swin.cpp"),
                    str(source / "src/requests.cpp"),
                    str(source / "src/job_text.cpp"),
                    str(source / "src/job.cpp"),
                    str(source / "src/index.cpp"),
                    str(source / "src/timeline.cpp"),
                    str(source / "src/replay.cpp"),
                    str(source / "src/window.cpp"),
                    str(source / "src/recording.cpp"),
                    str(source / "src/execution_bridge.cpp"),
                ],
                include_dirs=[
                    str(source / "include"),
                    str(source / "third_party"),
                ],
                depends=depends,
                language="c++",
                libraries=libraries,
                library_dirs=library_dirs,
                runtime_library_dirs=runtime_dirs,
                define_macros=macros,
                extra_compile_args=[
                    "-std=c++17",
                    "-O3",
                    "-Wall",
                    "-Wextra",
                    "-ffp-contract=off",
                ],
            )
        ],
        compiler_directives={"language_level": 3},
        build_dir="build/xbeam-cython",
    )
