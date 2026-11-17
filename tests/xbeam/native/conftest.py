# SPDX-License-Identifier: Apache-2.0
"""The compiled backend is optional for pure Python installations."""

from pathlib import Path

import pytest


@pytest.fixture
def native_executable():
    executable = (
        Path(__file__).resolve().parents[3]
        / "build/xbeam-native/xbeam-native"
    )
    if not executable.is_file():
        pytest.skip("build the optional xbeam-native executable with CMake")
    return executable
