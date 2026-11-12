# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Publication eligibility, isolated builds, and workflow correlation."""

import subprocess
from pathlib import Path

import pytest

from cuphoton_dev import build, github, publisher


def release(tag="v0.2.0", **changes):
    return {
        "tag_name": tag,
        "draft": False,
        "prerelease": False,
        "published_at": "2026-11-11T00:00:00Z",
        "html_url": "https://example.test/release",
        **changes,
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"draft": True},
        {"prerelease": True},
        {"published_at": None},
        {"tag_name": "v0.2.0rc0"},
        {"tag_name": "v01.2.0"},
        {"tag_name": "v0.2.3\u0663"},
    ],
)
def test_ineligible(changes):
    assert not github.eligible(release(**changes))


def test_promotion():
    assert not github.eligible(release(prerelease=True))
    assert github.eligible(release(prerelease=False))


def test_build_environment_does_not_use_controller(monkeypatch, tmp_path):
    for key in (
        "PYTHONPATH",
        "PYTHONHOME",
        "VIRTUAL_ENV",
        "UV_PROJECT_ENVIRONMENT",
        "UV_PROJECT",
        "UV_WORKING_DIR",
        "JUPYTER_PATH",
        "QUARTO_PROFILE",
        "GH_TOKEN",
        "GITHUB_TOKEN",
        "SETUPTOOLS_SCM_PRETEND_VERSION",
    ):
        monkeypatch.setenv(key, "/controller")
    env = build.render_environment(tmp_path)
    assert env["QUARTO_PYTHON"] == str(tmp_path / ".venv/bin/python")
    assert env["UV_PROJECT_ENVIRONMENT"] == str(tmp_path / ".venv")
    assert env["CUPHOTON_REPO"] == str(tmp_path)
    assert "PYTHONPATH" not in env
    assert "VIRTUAL_ENV" not in env
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert "GH_TOKEN" not in env
    assert "GITHUB_TOKEN" not in env
    assert "SETUPTOOLS_SCM_PRETEND_VERSION" not in env
    assert env["PATH"].startswith(str(tmp_path / ".venv/bin") + ":")
    assert env["JUPYTER_PATH"] == str(tmp_path / ".venv/share/jupyter")


def test_dispatch_exact_run_id(monkeypatch):
    calls = []

    def api(path, *, data=None, **kwargs):
        calls.append((path, data))
        if data:
            assert data["ref"] == "main"
            assert data["inputs"]["tag"] == "main"
            return {"workflow_run_id": 123}
        assert path.endswith("/123")
        return {
            "status": "in_progress",
            "conclusion": None,
            "head_sha": "abc",
            "html_url": "https://example.test/runs/123",
        }

    monkeypatch.setattr(github, "api", api)
    result = github.docs_publish(ref="main", wait=False)
    assert result["run_id"] == 123
    assert result["conclusion"] is None
    assert len(calls) == 2


def test_legacy_dispatch_correlates_unique_request(monkeypatch):
    request_id = None

    def api(path, *, data=None, **kwargs):
        nonlocal request_id
        if data:
            request_id = data["inputs"]["request_id"]
            return None
        if "workflow_dispatch" in path:
            return {
                "workflow_runs": [
                    {"id": 999, "display_title": "unrelated"},
                    {"id": 123, "display_title": f"docs main / {request_id}"},
                ]
            }
        assert path.endswith("/123")
        return {
            "status": "completed",
            "conclusion": "success",
            "head_sha": "abc",
            "html_url": "https://example.test/runs/123",
        }

    monkeypatch.setattr(github, "api", api)
    assert github.docs_publish(ref="main")["run_id"] == 123


def test_failed_run_propagates_url(monkeypatch):
    monkeypatch.setattr(
        github,
        "api",
        lambda *a, **k: {
            "status": "completed",
            "conclusion": "failure",
            "head_sha": "abc",
            "html_url": "https://example.test/runs/123",
        },
    )
    with pytest.raises(
        github.PublicationError,
        match="failure: https://example.test/runs/123",
    ) as error:
        github.wait_for_run(
            github.REPOSITORY, 123, wait=True, request_id="test"
        )
    assert error.value.result["request_id"] == "test"
    assert error.value.result["run_id"] == 123
    assert error.value.result["conclusion"] == "failure"


def test_draft_cannot_dispatch(monkeypatch):
    monkeypatch.setattr(github, "releases", lambda *a: [release(draft=True)])
    monkeypatch.setattr(
        github, "api", lambda *a, **k: pytest.fail("must not dispatch")
    )
    with pytest.raises(ValueError, match="published final"):
        github.docs_publish(ref="v0.2.0")


def test_status_reports_unavailable_and_draft(monkeypatch):
    monkeypatch.setattr(
        github,
        "published_catalog",
        lambda: {
            "stable": "v0.2.0",
            "versions": [{"version": "main"}, {"version": "v0.2.0"}],
        },
    )
    monkeypatch.setattr(
        github,
        "releases",
        lambda *a: [
            release("v0.1.3"),
            release(),
            release("v0.2.1", draft=True),
            release("v0.3.0"),
        ],
    )
    monkeypatch.setattr(github, "has_guide", lambda tag, *a: tag != "v0.1.3")
    result = github.docs_status()
    assert [r["state"] for r in result["releases"]] == [
        "unavailable",
        "published",
        "ineligible",
        "missing",
    ]


def test_release_pagination(monkeypatch):
    pages = iter([[release()] * 100, [release("v0.1.3")]])
    monkeypatch.setattr(github, "api", lambda *a: next(pages))
    assert len(github.releases()) == 101


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [(200, '{"value":1}', {"value": 1}), (204, "", None), (404, "{}", None)],
)
def test_api_reads_status_separately_from_json(
    monkeypatch, status, body, expected
):
    def gh(args, **kwargs):
        return subprocess.CompletedProcess(
            args,
            1 if status == 404 else 0,
            f"HTTP/2.0 {status}\nContent-Type: application/json\n\n{body}",
            "gh: Not Found" if status == 404 else "",
        )

    monkeypatch.setattr(github.subprocess, "run", gh)
    assert github.api("repos/NVIDIA/cuPhoton", missing_ok=True) == expected


def test_prepare_rejects_tag_workflow(monkeypatch, tmp_path):
    monkeypatch.setenv("GITHUB_REPOSITORY", github.REPOSITORY)
    monkeypatch.setenv("GITHUB_REF", "refs/tags/v0.2.0")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "release")
    with pytest.raises(ValueError, match="canonical main"):
        publisher.prepare(tag="v0.2.0", output=tmp_path / "site")


def test_build_failure_preserves_existing_output(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    (output / "index.html").write_text("previous")
    with pytest.raises(ValueError, match="already exists"):
        build.docs_build(ref="main", output=output, repo=Path.cwd())
    assert (output / "index.html").read_text() == "previous"


def test_build_progress_supports_core_captured_streams():
    import io
    import sys
    from contextlib import redirect_stderr

    progress = io.StringIO()
    with redirect_stderr(progress):
        build.run(
            [sys.executable, "-c", "print('render progress')"], capture=False
        )
    assert progress.getvalue() == "render progress\n"
