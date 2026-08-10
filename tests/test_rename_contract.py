# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import re
import tomllib
from pathlib import Path

import pytest

from cuphoton.core.cli import (
    COMPONENT_REGISTRY,
    COMPONENTS,
    ApplicationContext,
    ComponentSpec,
)

ROOT = Path(__file__).resolve().parents[1]
GROUPS = (
    "xdr",
    "xfit",
    "xpois",
    "xscan",
    "xrep",
    "xray",
)
PUBLIC_NAMESPACES = (
    "cuphoton.core",
    "cuphoton.xdr",
    "cuphoton.xfit",
    "cuphoton.xpois",
    "cuphoton.xscan",
    "cuphoton.xrep",
    "cuphoton.xray",
)


def _project_metadata() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def test_distribution_exposes_only_the_cuphoton_console_script():
    metadata = _project_metadata()

    assert metadata["project"]["dynamic"] == ["version"]
    assert metadata["project"]["scripts"] == {
        "cuphoton": "cuphoton.core.cli:main"
    }


@pytest.mark.parametrize("namespace", PUBLIC_NAMESPACES)
def test_public_namespaces_import(namespace):
    module = importlib.import_module(namespace)

    assert module.__name__ == namespace


@pytest.mark.parametrize("group", GROUPS)
def test_all_public_group_packages_import(group):
    module = importlib.import_module(f"cuphoton.{group}")

    assert module.__name__ == f"cuphoton.{group}"


@pytest.mark.parametrize("group", GROUPS)
def test_component_packages_have_no_module_entry_point(group):
    package_dir = ROOT / "src" / "cuphoton" / group

    assert package_dir.is_dir()
    assert not (package_dir / "__main__.py").exists()


def test_packaged_asset_source_files_exist():
    for group in ("xpois", "xscan"):
        assets = ROOT / "src" / "cuphoton" / group / "assets"
        assert (assets / "cuphoton-logo.png").is_file()
        assert (assets / "cuphoton-logo-128.png").is_file()


def test_renamed_environment_and_schema_identifiers_are_present():
    expected = {
        ROOT
        / "src"
        / "cuphoton"
        / "xdr"
        / "src"
        / "build.sh": "CUPHOTON_XDR_BUILD_EXT",
        ROOT
        / "src"
        / "cuphoton"
        / "xscan"
        / "raw_compare_review.py": "CUPHOTON_XSCAN_REVIEW_DIR",
        ROOT
        / "src"
        / "cuphoton"
        / "xray"
        / "detector_artifacts.py": '"kind": "xray-detector-artifacts"',
        ROOT
        / "src"
        / "cuphoton"
        / "xray"
        / "workflow_viz.py": '"kind": "xray-workflow-viz-bundle"',
    }

    for path, identifier in expected.items():
        assert identifier in path.read_text(encoding="utf-8")

    package_data = _project_metadata()["tool"]["setuptools"]["package-data"]
    assert "cuphoton.xscan" in package_data
    assert "cuphoton.xray" in package_data
    assert COMPONENT_REGISTRY["xray"].log_env == "CUPHOTON_XRAY_LOG_FILE"


def test_core_owns_the_fixed_component_registry():
    assert tuple(component.group for component in COMPONENTS) == GROUPS
    assert set(COMPONENT_REGISTRY) == set(GROUPS)
    assert COMPONENT_REGISTRY["xdr"].import_name == "cuphoton.xdr"
    assert COMPONENT_REGISTRY["xdr"].log_env == "CUPHOTON_XDR_LOG_FILE"
    assert all(
        isinstance(component, ComponentSpec) for component in COMPONENTS
    )


def test_components_have_no_cli_shims_or_local_argparse():
    failures = []
    for group in GROUPS:
        package_dir = ROOT / "src" / "cuphoton" / group
        assert not (package_dir / "_app.py").exists()
        assert not (package_dir / "cli.py").exists()
        for path in package_dir.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            pattern = r"(^|\n)\s*(?:import argparse|from argparse import)"
            if re.search(pattern, text):
                failures.append(path.relative_to(ROOT).as_posix())
    assert failures == []

    bridge = ROOT / "src" / "cuphoton" / "xray" / "_argparse_cli.py"
    assert not bridge.exists()

    core_argparse = []
    for path in (ROOT / "src" / "cuphoton").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        pattern = r"(^|\n)\s*(?:import argparse|from argparse import)"
        if re.search(pattern, text):
            core_argparse.append(path.relative_to(ROOT).as_posix())
    assert core_argparse == ["src/cuphoton/core/cli/app.py"]


def test_xdg_path_lookup_is_group_scoped_and_side_effect_free(
    monkeypatch,
    tmp_path,
):
    config_home = tmp_path / "config"
    state_home = tmp_path / "state"
    data_home = tmp_path / "data"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    monkeypatch.setenv("XDG_STATE_HOME", str(state_home))
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home))

    for group in GROUPS:
        context = ApplicationContext.for_component(group)
        assert (
            context.config_dir == (config_home / "cuphoton" / group).resolve()
        )
        assert (
            context.state_dir == (state_home / "cuphoton" / group).resolve()
        )
        assert context.data_dir == (data_home / "cuphoton" / group).resolve()
        assert (
            context.runs_dir
            == (state_home / "cuphoton" / group / "runs").resolve()
        )
        assert (
            context.logs_dir
            == (state_home / "cuphoton" / group / "logs").resolve()
        )

    assert not config_home.exists()
    assert not state_home.exists()
    assert not data_home.exists()
