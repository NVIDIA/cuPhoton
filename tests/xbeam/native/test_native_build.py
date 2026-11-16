# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Incremental builds must use current native code and execution mode."""

import ctypes
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from setuptools import Distribution
from setuptools.command.build_ext import build_ext

from cuphoton.xbeam import setup_package


@pytest.fixture
def native_build(tmp_path, monkeypatch, native_executable):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(
        sys.modules,
        "Cython.Build",
        SimpleNamespace(cythonize=lambda e, **k: e),
    )
    # Exercise setuptools with small real C++ inputs; CUDA compilation is
    # independently covered by the native execution build.
    monkeypatch.setattr(
        setup_package, "subprocess", SimpleNamespace(run=lambda *a, **k: None)
    )
    source = Path("src/cuphoton/xbeam/native")
    header = source / "include/probe.hpp"
    header.parent.mkdir(parents=True)
    header.write_text("#define PROBE_VALUE 10\n")
    (header.parent / "xbeam").mkdir()
    (header.parent / "xbeam/radio.hpp").touch()
    third_party = source / "third_party/nlohmann/json.hpp"
    third_party.parent.mkdir(parents=True)
    third_party.touch()
    archives = Path("build/xbeam-native-cuda")
    archives.mkdir(parents=True)
    (archives / "libxbeam_cuda.a").touch()
    (archives / "libxbeam_execution.a").touch()

    def extension(*, cuda):
        monkeypatch.setenv("CUPHOTON_XBEAM_NATIVE_CUDA", "1" if cuda else "0")
        result = setup_package.get_extensions()[0]
        result.name = "native_build_probe"
        result.sources = ["probe.cpp"]
        result.libraries = []
        result.library_dirs = []
        result.runtime_library_dirs = []
        return result

    def build(extension, name):
        command = build_ext(Distribution({"ext_modules": [extension]}))
        command.ensure_finalized()
        command.run()
        output = Path(command.get_ext_fullpath(extension.name))
        # Each dlopen needs a distinct path to avoid reusing a prior handle.
        loaded = tmp_path / (name + ".so")
        shutil.copyfile(output, loaded)
        return ctypes.CDLL(str(loaded)).probe()

    return extension, build, header, archives


def test_native_binding_recompiles_when_execution_mode_changes(native_build):
    extension, build, _, _ = native_build
    Path("probe.cpp").write_text(
        'extern "C" int probe() {\n'
        "#ifdef XBEAM_BIND_CUDA\n"
        "return 1;\n"
        "#else\n"
        "return 0;\n"
        "#endif\n"
        "}\n"
    )
    for name, cuda in (("cpu", False), ("cuda", True), ("cpu_again", False)):
        assert build(extension(cuda=cuda), name) == int(cuda)


def test_native_binding_recompiles_changed_headers(native_build):
    extension, build, header, _ = native_build
    Path("probe.cpp").write_text(
        '#include "probe.hpp"\n'
        'extern "C" int probe() { return PROBE_VALUE; }\n'
    )
    assert build(extension(cuda=False), "header_before") == 10
    header.write_text("#define PROBE_VALUE 20\n")
    assert build(extension(cuda=False), "header_after") == 20


def test_native_binding_relinks_changed_cuda_archives(native_build):
    extension, build, _, archives = native_build
    Path("probe.cpp").write_text(
        'extern "C" int archive_value();\n'
        'extern "C" int probe() { return archive_value(); }\n'
    )

    def rebuild_archive(value):
        Path("archive.cpp").write_text(
            f'extern "C" int archive_value() {{ return {value}; }}\n'
        )
        for arguments in (
            ["c++", "-fPIC", "-c", "archive.cpp", "-o", "archive.o"],
            [
                "ar",
                "rcs",
                str(archives / "libxbeam_execution.a"),
                "archive.o",
            ],
        ):
            subprocess.run(arguments, check=True)

    (archives / "libxbeam_execution.a").unlink()
    rebuild_archive(10)
    first = extension(cuda=True)
    first.libraries = ["xbeam_execution"]
    first.library_dirs = [str(archives)]
    assert build(first, "archive_before") == 10
    rebuild_archive(20)
    second = extension(cuda=True)
    second.libraries = ["xbeam_execution"]
    second.library_dirs = [str(archives)]
    assert build(second, "archive_after") == 20
