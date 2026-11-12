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


@pytest.mark.parametrize(
    "conclusion", ["failure", "cancelled", "timed_out", None]
)
def test_tracked_publication_failure_keeps_json_and_nonzero_status(
    monkeypatch: pytest.MonkeyPatch, conclusion: str | None
) -> None:
    from cuphoton_dev import github

    request_id = ""

    def api(path: str, *, data=None, **kwargs):
        nonlocal request_id
        if data:
            request_id = data["inputs"]["request_id"]
            return {"workflow_run_id": 123}
        assert path.endswith("/123")
        return {
            "status": "completed" if conclusion else "in_progress",
            "conclusion": conclusion,
            "head_sha": "a" * 40,
            "html_url": "https://example.org/run/123",
        }

    monkeypatch.setattr(github, "api", api)
    if conclusion is None:
        clock = iter([0, 7201])
        monkeypatch.setattr(
            github.time, "monotonic", lambda: next(clock, 7201)
        )
    out, err = io.StringIO(), io.StringIO()
    assert (
        product_cli.main(
            ["dev", "docs-publish", "--ref", "main", "--json"],
            out=out,
            err=err,
        )
        == 1
    )
    result = json.loads(out.getvalue())
    assert result["request_id"] == request_id
    assert result["run_id"] == 123
    assert result["url"] == "https://example.org/run/123"
    assert result["controller_sha"] == "a" * 40
    assert result["conclusion"] == conclusion
    assert result["status"] == ("completed" if conclusion else "in_progress")
    assert result["error"] in err.getvalue()
    assert result["url"] in result["error"]


def test_dispatch_discovery_timeout_reports_recovery_request_as_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cuphoton_dev import github

    request_id = ""

    def api(path: str, *, data=None, **kwargs):
        nonlocal request_id
        if data:
            request_id = data["inputs"]["request_id"]
            return None
        assert "workflow_dispatch" in path
        return {"workflow_runs": []}

    clock = iter([0, 181])
    monkeypatch.setattr(github, "api", api)
    monkeypatch.setattr(github.time, "monotonic", lambda: next(clock, 181))
    out, err = io.StringIO(), io.StringIO()
    assert (
        product_cli.main(
            ["dev", "docs-publish", "--ref", "main", "--json"],
            out=out,
            err=err,
        )
        == 1
    )
    result = json.loads(out.getvalue())
    assert request_id and result["request_id"] == request_id
    assert result["status"] == "dispatch_accepted"
    assert result["run_id"] is None
    assert result["url"] is None
    assert request_id in result["error"]
    assert result["error"] in err.getvalue()


@pytest.mark.parametrize("known_run", [False, True])
def test_api_failure_after_dispatch_retains_last_known_json_state(
    monkeypatch: pytest.MonkeyPatch, known_run: bool
) -> None:
    from cuphoton_dev import github

    request_id = ""
    polls = 0

    def api(path: str, *, data=None, **kwargs):
        nonlocal request_id, polls
        if data:
            request_id = data["inputs"]["request_id"]
            return {"workflow_run_id": 123} if known_run else None
        polls += 1
        if known_run and polls == 1:
            return {
                "status": "in_progress",
                "conclusion": None,
                "head_sha": "a" * 40,
                "html_url": "https://example.org/run/123",
            }
        raise RuntimeError("temporary API outage")

    monkeypatch.setattr(github, "api", api)
    monkeypatch.setattr(github.time, "sleep", lambda _: None)
    out, err = io.StringIO(), io.StringIO()
    assert (
        product_cli.main(
            ["dev", "docs-publish", "--ref", "main", "--json"],
            out=out,
            err=err,
        )
        == 1
    )
    result = json.loads(out.getvalue())
    assert request_id and result["request_id"] == request_id
    assert result["status"] == (
        "in_progress" if known_run else "dispatch_accepted"
    )
    assert result["run_id"] == (123 if known_run else None)
    assert result["url"] == (
        "https://example.org/run/123" if known_run else None
    )
    assert result["controller_sha"] == ("a" * 40 if known_run else None)
    assert "temporary API outage" in result["error"]
    assert result["error"] in err.getvalue()


def test_unconfirmed_dispatch_preserves_request_without_redispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cuphoton_dev import github

    submitted = []

    def api(path: str, *, data=None, **kwargs):
        assert path.endswith("/dispatches") and data is not None
        submitted.append(data["inputs"]["request_id"])
        # The server may create the run before the client loses its response.
        raise OSError("connection closed after request was sent")

    monkeypatch.setattr(github, "api", api)
    out, err = io.StringIO(), io.StringIO()
    assert (
        product_cli.main(
            ["dev", "docs-publish", "--ref", "main", "--json"],
            out=out,
            err=err,
        )
        == 1
    )
    result = json.loads(out.getvalue())
    assert len(submitted) == 1
    assert result["request_id"] == submitted[0]
    assert result["status"] == "dispatch_unknown"
    assert result["run_id"] is None
    assert result["url"] is None
    assert result["conclusion"] is None
    assert result["controller_sha"] is None
    assert submitted[0] in result["error"]
    assert "Inspect that request before redispatching" in result["error"]
    assert result["error"] in err.getvalue()


@pytest.mark.parametrize(
    ("http_status", "expected"),
    [
        (403, "dispatch_rejected"),
        (404, "dispatch_rejected"),
        (422, "dispatch_rejected"),
        (408, "dispatch_unknown"),
        (429, "dispatch_unknown"),
        (500, "dispatch_unknown"),
    ],
)
def test_dispatch_http_rejection_is_distinct_from_uncertain_submission(
    monkeypatch: pytest.MonkeyPatch, http_status: int, expected: str
) -> None:
    from cuphoton_dev import github

    submitted = []

    def gh(args, **kwargs):
        submitted.append(json.loads(kwargs["input"])["inputs"]["request_id"])
        body = '{"message":"Request failed"}'
        stdout = (
            f"HTTP/2.0 {http_status} Request failed\n"
            f"Content-Type: application/json\n\n{body}"
            if "--include" in args
            else body
        )
        return subprocess.CompletedProcess(
            args, 1, stdout, "gh: Request failed"
        )

    monkeypatch.setattr(github.subprocess, "run", gh)
    out, err = io.StringIO(), io.StringIO()
    assert (
        product_cli.main(
            ["dev", "docs-publish", "--ref", "main", "--json"],
            out=out,
            err=err,
        )
        == 1
    )
    result = json.loads(out.getvalue())
    assert len(submitted) == 1
    assert result["request_id"] == submitted[0]
    assert result["status"] == expected
    assert result["run_id"] is None
    assert result["error"] in err.getvalue()
    assert ("Inspect that request" in result["error"]) == (
        expected == "dispatch_unknown"
    )


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
