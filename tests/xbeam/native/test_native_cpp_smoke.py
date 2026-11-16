# SPDX-License-Identifier: Apache-2.0
"""Exercise native C++ contracts without a CUDA runtime."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "name", ["header", "request", "work_queue", "run_report", "radio_options"]
)
def test_native_cpp_contract(name, native_executable, tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("a C++17 compiler is required")
    root = Path(__file__).resolve().parents[3]
    options_source = root / "src/cuphoton/xbeam/native/cuda/radio_options.cpp"
    sources = [str(Path(__file__).with_name(name + "_smoke.cpp"))]
    if name == "radio_options":
        sources.append(str(options_source))
    output = tmp_path / name
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-pthread",
            "-ffp-contract=off",
            "-I" + str(root / "src/cuphoton/xbeam/native/include"),
            "-I" + str(root / "src/cuphoton/xbeam/native/cuda"),
            *sources,
            str(native_executable.parent / "libxbeam_radio.a"),
            "-lcrypto",
            "-o",
            str(output),
        ],
        check=True,
    )
    subprocess.run([str(output)], check=True, timeout=30, cwd=tmp_path)
