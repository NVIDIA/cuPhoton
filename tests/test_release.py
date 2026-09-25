# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Exercise release identity and artifact promotion without publishing."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "wheels"
SPEC = importlib.util.spec_from_file_location(
    "release", SCRIPTS / "release.py"
)
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


@pytest.mark.parametrize(
    "version", ["0.1.3", "1.2.30", "0.1.3rc0", "0.1.3rc12"]
)
def test_canonical_release_tags(version):
    assert release.release_version(f"v{version}") == version


@pytest.mark.parametrize(
    "tag",
    [
        "0.1.3",
        "v0.1",
        "v00.1.3",
        "v0.01.3",
        "v0.1.03",
        "v0.1.3rc01",
        "v0.1.3RC1",
        "v0.1.3.dev1",
        "v0.1.3+local",
        "v0.1.3\n",
        "v0.1.3; echo bad",
        "--help",
    ],
)
def test_noncanonical_release_tags_are_rejected(tag):
    with pytest.raises(ValueError, match="Expected vX.Y.Z"):
        release.release_version(tag)


@pytest.fixture
def source(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def git(*args):
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.PIPE
        ).strip()

    git("init", "--initial-branch=main")
    git("config", "user.name", "Release Tests")
    git("config", "user.email", "tests@nvidia.com")
    git("config", "commit.gpgsign", "false")
    git("config", "tag.gpgsign", "false")
    git("commit", "--allow-empty", "-m", "Initial source")
    sha = git("rev-parse", "HEAD")
    git("update-ref", "refs/remotes/origin/main", sha)
    return git, sha


def test_tags_resolve_to_commits_and_detect_a_moved_tag(source):
    git, sha = source
    git("tag", "-a", "v0.1.3rc0", "-m", "Release candidate")
    git("tag", "v0.1.3")
    assert release.resolve("v0.1.3rc0", sha) == ("0.1.3rc0", sha)
    assert release.resolve("v0.1.3", sha) == ("0.1.3", sha)
    git("commit", "--allow-empty", "-m", "Later source")
    git("update-ref", "refs/remotes/origin/main", "HEAD")
    # A reviewed ancestor remains valid after the release branch advances.
    assert release.resolve("v0.1.3", sha) == ("0.1.3", sha)
    git("tag", "--force", "v0.1.3")
    with pytest.raises(ValueError, match="Release tag moved"):
        release.resolve("v0.1.3", sha)


def test_tag_requires_a_release_branch_ancestor(source):
    git, _ = source
    git("commit", "--allow-empty", "-m", "Unreviewed source")
    sha = git("rev-parse", "HEAD")
    git("tag", "v0.1.4")
    with pytest.raises(ValueError, match="must belong to main or 0.1.x"):
        release.resolve("v0.1.4")
    git("update-ref", "refs/remotes/origin/0.1.x", sha)
    assert release.resolve("v0.1.4") == ("0.1.4", sha)


@pytest.fixture
def build(tmp_path, monkeypatch):
    dist = tmp_path / "dist"
    dist.mkdir()
    for filename in ("cuphoton-0.1.3.tar.gz", "cuphoton-0.1.3-example.whl"):
        (dist / filename).write_bytes(filename.encode())
    source_sha, workflow_sha = "a" * 40, "b" * 40
    manifest = {
        "version": "0.1.3",
        "tag": "v0.1.3",
        "source_sha": source_sha,
        "repository": "NVIDIA/cuPhoton",
        "run_id": 123,
        "run_attempt": 1,
        "workflow_sha": workflow_sha,
        "files": release.hashes(dist),
    }
    run = {
        "id": 123,
        "head_repository": {"full_name": "NVIDIA/cuPhoton"},
        "path": ".github/workflows/publish.yml",
        "head_sha": workflow_sha,
        "event": "workflow_dispatch",
        "head_branch": "main",
        "run_attempt": 2,
        "status": "in_progress",
        "conclusion": None,
    }
    jobs = [
        {
            "name": "release wheels / distributions",
            "status": "completed",
            "conclusion": "success",
        }
    ]
    monkeypatch.setenv("GITHUB_REPOSITORY", "NVIDIA/cuPhoton")
    return SimpleNamespace(
        dist=dist,
        manifest=manifest,
        run=run,
        jobs=jobs,
        sha=source_sha,
        manifest_path=tmp_path / "provenance.json",
    )


def validate(build):
    release.validate_provenance(
        build.manifest,
        build.run,
        build.jobs,
        tag="v0.1.3",
        sha=build.sha,
        repository="NVIDIA/cuPhoton",
    )


def test_manual_promotion_uses_original_build_attempt(build, monkeypatch):
    # The workflow runs on main while building an older tagged source SHA.
    # Later publication can still be pending or failed in the parent run.
    endpoints = []

    def gh(endpoint):
        endpoints.append(endpoint)
        if endpoint.endswith("/jobs?per_page=100"):
            return {"total_count": 1, "jobs": build.jobs}
        return build.run

    monkeypatch.setattr(release, "gh", gh)
    build.manifest_path.write_text(json.dumps(build.manifest))
    release.verify(
        build.dist, build.manifest_path, "v0.1.3", build.sha, "123"
    )
    assert endpoints == [
        "repos/NVIDIA/cuPhoton/actions/runs/123",
        "repos/NVIDIA/cuPhoton/actions/runs/123/attempts/1/jobs?per_page=100",
    ]
    build.run.update(status="completed", conclusion="failure")
    validate(build)


@pytest.mark.parametrize(
    "section,key,value,error",
    [
        ("manifest", "version", "0.1.3rc0", "Wrong version"),
        ("manifest", "tag", "v0.1.3rc0", "Wrong source tag"),
        ("manifest", "source_sha", "b" * 40, "Wrong source commit"),
        ("manifest", "workflow_sha", "a" * 40, "Wrong workflow commit"),
        (
            "manifest",
            "repository",
            "elsewhere/project",
            "Wrong build repository",
        ),
        ("manifest", "run_id", 456, "Wrong build run"),
        ("run", "path", ".github/workflows/ci.yml", "release workflow"),
        ("run", "event", "pull_request", "manual release from main"),
        ("run", "head_branch", "topic", "manual release from main"),
        (
            "run",
            "head_repository",
            {"full_name": "fork/cuPhoton"},
            "another repository",
        ),
    ],
)
def test_provenance_mismatch_is_rejected(build, section, key, value, error):
    getattr(build, section)[key] = value
    with pytest.raises(ValueError, match=error):
        validate(build)


def test_push_requires_the_exact_tag_and_commit(build):
    build.run.update(event="push", head_branch="v0.1.3", head_sha=build.sha)
    build.manifest["workflow_sha"] = build.sha
    validate(build)
    build.run["head_branch"] = "v0.1.3rc0"
    with pytest.raises(ValueError, match="another tag"):
        validate(build)
    build.run["head_branch"] = "v0.1.3"
    build.run["head_sha"] = build.manifest["workflow_sha"] = "c" * 40
    with pytest.raises(ValueError, match="another commit"):
        validate(build)


@pytest.mark.parametrize(
    "status,conclusion",
    [
        ("in_progress", None),
        ("queued", None),
        ("completed", "failure"),
        ("completed", "cancelled"),
        ("completed", "skipped"),
    ],
)
def test_incomplete_or_failed_build_cannot_be_promoted(
    build, status, conclusion
):
    build.jobs[0].update(status=status, conclusion=conclusion)
    with pytest.raises(
        ValueError, match="Distribution validation did not succeed"
    ):
        validate(build)


def test_another_successful_job_is_not_distribution_validation(build):
    build.jobs[0]["name"] = "stage"
    with pytest.raises(
        ValueError, match="Distribution validation did not succeed"
    ):
        validate(build)


def test_changed_artifact_bytes_cannot_be_promoted(build, monkeypatch):
    monkeypatch.setattr(
        release,
        "gh",
        lambda endpoint: (
            {"total_count": 1, "jobs": build.jobs}
            if "/attempts/" in endpoint
            else build.run
        ),
    )
    build.manifest_path.write_text(json.dumps(build.manifest))
    next(build.dist.glob("*.whl")).write_bytes(b"changed after qualification")
    with pytest.raises(ValueError, match="Artifact hashes differ"):
        release.verify(
            build.dist, build.manifest_path, "v0.1.3", build.sha, "123"
        )


def test_provenance_sidecar_must_stay_outside_distributions(build):
    (build.dist / "provenance.json").write_text("{}")
    with pytest.raises(ValueError, match="unexpected files"):
        release.hashes(build.dist)


@pytest.mark.parametrize("published_count", [0, 1, 2])
def test_retry_removes_only_byte_identical_published_files(
    build, published_count
):
    files = list(build.manifest["files"].items())
    published = [
        {"filename": name, "digests": {"sha256": digest}}
        for name, digest in files[:published_count]
    ]
    assert (
        release.remove_published(build.dist, published) == 2 - published_count
    )
    assert {path.name for path in build.dist.iterdir()} == {
        name for name, _ in files[published_count:]
    }


@pytest.mark.parametrize("unknown_filename", [False, True])
def test_retry_hash_collision_keeps_every_local_file(build, unknown_filename):
    before = release.hashes(build.dist)
    (first, digest), (second, _) = before.items()
    published = [
        {"filename": first, "digests": {"sha256": digest}},
        {
            "filename": "unknown.whl" if unknown_filename else second,
            "digests": {"sha256": "0" * 64},
        },
    ]
    with pytest.raises(ValueError, match="different files"):
        release.remove_published(build.dist, published)
    assert release.hashes(build.dist) == before


@pytest.mark.parametrize(
    "target,host", [("pypi", "pypi.org"), ("testpypi", "test.pypi.org")]
)
def test_new_index_version_keeps_all_files(build, monkeypatch, target, host):
    def missing(url, *, timeout):
        assert url == f"https://{host}/pypi/cuphoton/0.1.3/json"
        assert timeout > 0
        raise HTTPError(url, 404, "Not found", {}, None)

    monkeypatch.setattr(release, "urlopen", missing)
    assert release.pending(build.dist, "0.1.3", target) == 2


def test_index_failure_is_not_treated_as_an_unpublished_version(
    build, monkeypatch
):
    def failed(url, *, timeout):
        raise HTTPError(url, 503, "Unavailable", {}, None)

    monkeypatch.setattr(release, "urlopen", failed)
    with pytest.raises(HTTPError):
        release.pending(build.dist, "0.1.3", "pypi")
    assert release.hashes(build.dist) == build.manifest["files"]


def test_record_keeps_tagged_source_and_workflow_identity_separate(
    source, tmp_path, monkeypatch
):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    from check_distributions import LICENSES, SOURCES

    _, sha = source
    dist = tmp_path / "dist"
    dist.mkdir()
    sdist = dist / "cuphoton-0.1.3.tar.gz"
    metadata = (
        "Name: cuphoton\nVersion: 0.1.3\nLicense-Expression: Apache-2.0\n"
        "Requires-Python: >=3.12,<3.15\n"
        "Provides-Extra: io\nProvides-Extra: gpu\n"
    ).encode()
    with tarfile.open(sdist, "w:gz") as archive:
        for name in sorted(SOURCES | LICENSES | {"PKG-INFO"}):
            content = metadata if name == "PKG-INFO" else b"source\n"
            member = tarfile.TarInfo(f"cuphoton-0.1.3/{name}")
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    for key, value in {
        "RELEASE_VERSION": "0.1.3",
        "GITHUB_REPOSITORY": "NVIDIA/cuPhoton",
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "2",
        "GITHUB_WORKFLOW_SHA": "b" * 40,
    }.items():
        monkeypatch.setenv(key, value)
    manifest = release.record(dist)
    assert manifest["source_sha"] == sha
    assert manifest["workflow_sha"] == "b" * 40
    assert manifest["tag"] == "v0.1.3"
    assert manifest["run_attempt"] == 2
    assert manifest["files"] == {
        sdist.name: hashlib.sha256(sdist.read_bytes()).hexdigest()
    }
    monkeypatch.setenv("RELEASE_VERSION", "0.1.3rc0")
    with pytest.raises(ValueError, match="Build version mismatch"):
        release.record(dist)
