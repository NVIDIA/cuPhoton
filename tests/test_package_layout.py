# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import subprocess
import sys
from importlib import metadata, resources
from pathlib import Path

import pytest

import cuphoton

ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = (
    "core",
    "xdr",
    "xfit",
    "xpois",
    "xscan",
    "xrep",
    "xray",
)


def test_xfit_contains_no_npy_or_pickle_enabled_loads() -> None:
    roots = (ROOT / "src" / "cuphoton" / "xfit", ROOT / "tests" / "xfit")

    assert [path for root in roots for path in root.rglob("*.npy")] == []
    sources = [path for root in roots for path in root.rglob("*.py")]
    assert all(
        "allow_pickle=True" not in path.read_text(encoding="utf-8")
        for path in sources
    )


def test_all_components_use_cuphoton_namespace() -> None:
    for component in COMPONENTS:
        module = importlib.import_module(f"cuphoton.{component}")
        assert module.__name__ == f"cuphoton.{component}"


def test_distribution_has_single_top_level_package() -> None:
    top_level = metadata.distribution("cuphoton").read_text("top_level.txt")
    assert top_level is not None
    assert top_level.splitlines() == ["cuphoton"]


def test_distribution_and_package_share_version() -> None:
    assert metadata.version("cuphoton") == cuphoton.__version__


def test_distribution_metadata_declares_supported_profiles() -> None:
    distribution = metadata.distribution("cuphoton")
    assert set(distribution.metadata["Requires-Python"].split(",")) == {
        "<3.15",
        ">=3.11",
    }
    assert "Programming Language :: Python :: 3.14" in (
        distribution.metadata.get_all("Classifier") or ()
    )
    assert set(distribution.metadata.get_all("Provides-Extra") or ()) == {
        "cutile",
        "dev",
        "gpu",
        "hdf5",
        "torch",
        "viz",
    }
    requirements = distribution.metadata.get_all("Requires-Dist") or ()
    cutile_requirements = [
        requirement
        for requirement in requirements
        if 'extra == "cutile"' in requirement
    ]
    assert len(cutile_requirements) == 2
    assert all(
        'python_version < "3.14"' in requirement
        for requirement in cutile_requirements
    )


def test_distribution_exposes_only_the_umbrella_console_script() -> None:
    scripts = {
        entry.name: entry.value
        for entry in metadata.distribution("cuphoton").entry_points
        if entry.group == "console_scripts"
    }
    assert scripts == {"cuphoton": "cuphoton.core.cli:main"}


def test_packaged_assets_are_available() -> None:
    assert (
        resources.files("cuphoton.xpois")
        .joinpath("assets/cuphoton-logo.png")
        .is_file()
    )
    assert (
        resources.files("cuphoton.xscan")
        .joinpath("assets/cuphoton-logo.png")
        .is_file()
    )
    assert (
        resources.files("cuphoton.xray")
        .joinpath("templates/report.html")
        .is_file()
    )


@pytest.mark.parametrize(
    "argv",
    (
        ["--version"],
        ["xdr", "--help"],
        ["xfit", "--help"],
        ["xpois", "--help"],
        ["xscan", "--help"],
        ["xrep", "--help"],
        ["xray", "--help"],
    ),
)
def test_cli_help_and_version_do_not_import_optional_gpu_packages(
    argv: list[str],
) -> None:
    code = f"""
import importlib.abc
import sys

BLOCKED = ("cupy", "kvikio", "legate", "numba", "torch")

class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(
            fullname == prefix or fullname.startswith(prefix + ".")
            for prefix in BLOCKED
        ):
            raise ModuleNotFoundError(
                "blocked optional package for base-install test",
                name=fullname,
            )
        return None

sys.meta_path.insert(0, BlockOptional())
from cuphoton.core.cli import main
raise SystemExit(main({argv!r}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    if argv == ["--version"]:
        assert result.stdout == f"{cuphoton.__version__}\n"
    else:
        assert result.stdout
