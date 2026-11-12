# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.metadata
import io
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest
from cuphoton.core.cli import COMPONENTS, ComponentSpec
from cuphoton.core.cli import app as product_cli

from cuphoton_dev.cli import main


def test_root_help_and_invocation_require_explicit_installation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(name: str) -> None:
        assert name == "cuphoton-dev"
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(
        product_cli.importlib.metadata, "distribution", missing
    )
    out, err = io.StringIO(), io.StringIO()
    # Developer source is importable in this test environment, but metadata
    # still controls the opt-in route and help advertisement.
    assert product_cli.main(["--help"], out=out, err=err) == 0
    assert "  dev  " not in out.getvalue()
    out.seek(0)
    out.truncate()
    assert product_cli.main(["dev", "--help"], out=out, err=err) == 2
    assert out.getvalue() == ""
    assert "uv sync --project dev --locked" in err.getvalue()


def test_installed_root_help_is_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        product_cli.importlib.metadata, "distribution", lambda _: object()
    )

    def no_import(name: str) -> None:
        pytest.fail(f"root help imported {name}")

    monkeypatch.setattr(product_cli.importlib, "import_module", no_import)
    out, err = io.StringIO(), io.StringIO()
    assert product_cli.main(["--help"], out=out, err=err) == 0
    assert "  dev  " in out.getvalue()
    assert err.getvalue() == ""
    assert "dev" not in {component.group for component in COMPONENTS}


def test_product_delegation_preserves_streams_and_program_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        product_cli.importlib.metadata, "distribution", lambda _: object()
    )
    out, err = io.StringIO(), io.StringIO()
    assert (
        product_cli.main(
            ["help", "dev", "docs-publish"],
            out=out,
            err=err,
            program_name="custom",
        )
        == 0
    )
    assert "custom dev docs-publish" in out.getvalue()
    assert "--no-wait" in out.getvalue()
    assert err.getvalue() == ""


def test_external_namespace_exception_is_scoped() -> None:
    assert (
        ComponentSpec("dev", "cuphoton_dev", "Developer tools.").group
        == "dev"
    )
    with pytest.raises(ValueError, match="namespace"):
        ComponentSpec("other", "cuphoton_dev", "Other tools.")
    with pytest.raises(ValueError, match="namespace"):
        ComponentSpec("dev", "arbitrary_plugin", "Other tools.")


@pytest.fixture
def publisher(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    module = types.ModuleType("cuphoton_dev.publisher")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


@pytest.mark.parametrize(
    ("arguments", "function", "expected"),
    [
        (
            ["docs-build", "--ref", "v0.2.0", "--output", "book"],
            "docs_build",
            {"ref": "v0.2.0", "output": Path("book"), "repo": Path(".")},
        ),
        (
            ["docs-status"],
            "docs_status",
            {"repository": "NVIDIA/cuPhoton"},
        ),
        (
            ["docs-publish", "--ref", "main", "--no-wait"],
            "docs_publish",
            {
                "ref": "main",
                "missing": False,
                "wait": False,
                "repository": "NVIDIA/cuPhoton",
            },
        ),
        (
            ["docs-publish", "--missing"],
            "docs_publish",
            {
                "ref": None,
                "missing": True,
                "wait": True,
                "repository": "NVIDIA/cuPhoton",
            },
        ),
    ],
)
def test_commands_share_publisher_and_return_json(
    publisher: types.ModuleType,
    arguments: list[str],
    function: str,
    expected: dict[str, object],
) -> None:
    def invoke(**kwargs: object) -> dict[str, object]:
        assert kwargs == expected
        return {"url": "https://example.org/run/123", "run_id": 123}

    setattr(publisher, function, invoke)
    out, err = io.StringIO(), io.StringIO()
    assert main([*arguments, "--json"], out=out, err=err) == 0
    assert json.loads(out.getvalue())["run_id"] == 123
    assert err.getvalue() == ""


def test_check_accepts_positional_directory(
    publisher: types.ModuleType,
) -> None:
    def check(directory: Path) -> dict[str, object]:
        assert directory == Path("site")
        return {"valid": True}

    publisher.docs_check = check
    out, err = io.StringIO(), io.StringIO()
    assert main(["docs-check", "site", "--json"], out=out, err=err) == 0
    assert json.loads(out.getvalue()) == {"valid": True}
    assert err.getvalue() == ""


@pytest.mark.parametrize(
    "arguments",
    [
        ["docs-publish"],
        ["docs-publish", "--missing", "--ref", "main"],
        ["docs-build", "--ref", "main"],
        ["docs-check"],
    ],
)
def test_invalid_options_fail_without_calling_publisher(
    publisher: types.ModuleType,
    arguments: list[str],
) -> None:
    def forbidden(**kwargs: object) -> None:
        pytest.fail("invalid arguments reached publication")

    publisher.docs_publish = forbidden
    out, err = io.StringIO(), io.StringIO()
    assert main(arguments, out=out, err=err) != 0
    assert out.getvalue() == ""
    assert err.getvalue()


def test_workflow_failure_propagates(publisher: types.ModuleType) -> None:
    def publish(**kwargs: object) -> None:
        raise RuntimeError("Publication failed: https://example.org/run/123")

    publisher.docs_publish = publish
    out, err = io.StringIO(), io.StringIO()
    assert main(["docs-publish", "--ref", "main"], out=out, err=err) == 1
    assert out.getvalue() == ""
    assert "https://example.org/run/123" in err.getvalue()


def test_module_entrypoint() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "cuphoton_dev", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "docs-build" in result.stdout
    assert "docs-publish" in result.stdout
    assert result.stderr == ""
