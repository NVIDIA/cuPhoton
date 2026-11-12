# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Exercise publication transactions against real local Git repositories."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from cuphoton_dev import publisher
from cuphoton_dev.transforms import file_hashes, postprocess

from .test_publisher import release
from .test_transforms import rendered_guide


def git(path, *args):
    return subprocess.check_output(
        ["git", "-C", str(path), *args], text=True, stderr=subprocess.PIPE
    ).strip()


@pytest.fixture
def publication(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    git(source, "init", "-q", "--initial-branch=main")
    git(source, "config", "user.name", "Test")
    git(source, "config", "user.email", "test@example.com")
    git(source, "config", "commit.gpgsign", "false")
    (source / "source.txt").write_text("first source\n")
    git(source, "add", ".")
    git(source, "commit", "-qm", "first source")
    sha = git(source, "rev-parse", "HEAD")
    git(source, "tag", "v0.2.0")
    remote = tmp_path / "remote.git"
    remote.mkdir()
    git(remote, "init", "-q", "--bare")

    def clone(destination):
        heads = git(remote, "for-each-ref", "refs/heads/gh-pages")
        if heads:
            subprocess.run(
                [
                    "git",
                    "clone",
                    "-q",
                    "--branch",
                    "gh-pages",
                    str(remote),
                    str(destination),
                ],
                check=True,
            )
        else:
            destination.mkdir()
            git(destination, "init", "-q", "--initial-branch=gh-pages")
            git(destination, "remote", "add", "origin", str(remote))

    builds = []

    def build(*, ref, output, repo):
        builds.append(ref)
        rendered_guide(output)
        return postprocess(
            output,
            version=ref if ref.startswith("v") else "main",
            source_sha=publisher.resolve(repo, ref),
            controller_sha=publisher.resolve(source, "HEAD"),
            renderer="1.9.36",
            lock_sha256="c" * 64,
        )

    monkeypatch.chdir(source)
    monkeypatch.setenv("GITHUB_REPOSITORY", "NVIDIA/cuPhoton")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setenv("GITHUB_SHA", sha)
    monkeypatch.setattr(publisher, "clone_archive", clone)
    monkeypatch.setattr(publisher, "docs_build", build)
    monkeypatch.setattr(publisher, "releases", lambda: [release()])
    monkeypatch.setattr(publisher, "has_guide", lambda tag: True)
    monkeypatch.setattr(
        publisher,
        "remote_tag_commit",
        lambda tag: git(source, "rev-parse", f"refs/tags/{tag}^{{commit}}"),
    )
    return SimpleNamespace(
        source=source,
        remote=remote,
        sha=sha,
        builds=builds,
        root=tmp_path,
        clone=clone,
    )


def test_prepare_retries_retained_main_after_deployment_failure(
    publication, monkeypatch
):
    first = publication.root / "first"
    result = publisher.prepare(tag="main", output=first)
    assert result["catalog"]["preferred"] == "main"
    hashes = file_hashes(first)
    assert len(publication.builds) == 1
    # Pages could fail after retention. Recovery must work even if rendering
    # is temporarily unavailable and must preserve the complete artifact.
    monkeypatch.setattr(
        publisher, "docs_build", lambda **kwargs: pytest.fail("must reuse")
    )
    retry = publication.root / "retry"
    recovered = publisher.prepare(tag="main", output=retry)
    assert recovered["archive_sha"] == result["archive_sha"]
    assert file_hashes(retry) == hashes
    assert not (retry / ".git").exists()


def test_prepare_duplicate_release_and_next_main_preserve_history(
    publication, monkeypatch
):
    first = publication.root / "first"
    original = publisher.prepare(tag="v0.2.0", output=first)
    release_hashes = file_hashes(first / "v0.2.0")
    assert len(publication.builds) == 2  # Bootstrap main and the release.
    duplicate = publisher.prepare(
        tag="v0.2.0", output=publication.root / "duplicate"
    )
    assert duplicate["archive_sha"] == original["archive_sha"]
    assert len(publication.builds) == 2
    (publication.source / "source.txt").write_text("next source\n")
    git(publication.source, "commit", "-qam", "next source")
    sha = git(publication.source, "rev-parse", "HEAD")
    monkeypatch.setenv("GITHUB_SHA", sha)
    updated = publication.root / "updated"
    publisher.prepare(tag="main", output=updated)
    assert len(publication.builds) == 3
    assert file_hashes(updated / "v0.2.0") == release_hashes
    assert publisher.docs_check(updated)["stable"] == "v0.2.0"


def test_prepare_rejects_moved_tag_without_pushing(publication, monkeypatch):
    first = publication.root / "first"
    original = publisher.prepare(tag="v0.2.0", output=first)
    (publication.source / "source.txt").write_text("different source\n")
    git(publication.source, "commit", "-qam", "different source")
    sha = git(publication.source, "rev-parse", "HEAD")
    monkeypatch.setenv("GITHUB_SHA", sha)
    git(publication.source, "tag", "-f", "v0.2.0")
    output = publication.root / "rejected"
    with pytest.raises(ValueError, match="conflicts with the current tag"):
        publisher.prepare(tag="v0.2.0", output=output)
    assert not output.exists()
    assert (
        git(publication.remote, "rev-parse", "gh-pages")
        == original["archive_sha"]
    )


@pytest.mark.parametrize("failure", ["build", "eligibility", "push"])
def test_prepare_failure_does_not_change_retained_site(
    publication, monkeypatch, failure
):
    original = publisher.prepare(
        tag="main", output=publication.root / "first"
    )
    if failure == "build":

        def broken_build(**kwargs):
            raise RuntimeError("renderer failed")

        monkeypatch.setattr(publisher, "docs_build", broken_build)
    elif failure == "eligibility":
        records = iter([[release()], [release(draft=True)]])
        monkeypatch.setattr(publisher, "releases", lambda: next(records))
    else:
        real_git = publisher.git

        def broken_push(repo, *args):
            if "push" in args:
                raise RuntimeError("push failed")
            return real_git(repo, *args)

        monkeypatch.setattr(publisher, "git", broken_push)
    output = publication.root / "failed"
    with pytest.raises(RuntimeError):
        publisher.prepare(tag="v0.2.0", output=output)
    assert not output.exists()
    assert (
        git(publication.remote, "rev-parse", "gh-pages")
        == original["archive_sha"]
    )


@pytest.mark.parametrize("annotated", [False, True])
def test_remote_tag_commit_prefers_peeled_annotated_target(
    tmp_path, monkeypatch, annotated
):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.name", "Test")
    git(tmp_path, "config", "user.email", "test@example.com")
    git(tmp_path, "config", "commit.gpgsign", "false")
    (tmp_path / "source.txt").write_text("release source\n")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "release source")
    sha = git(tmp_path, "rev-parse", "HEAD")
    if annotated:
        git(tmp_path, "tag", "-a", "v0.2.0", "-m", "release")
        assert git(tmp_path, "rev-parse", "v0.2.0") != sha
    else:
        git(tmp_path, "tag", "v0.2.0")
    actual_run = publisher.run
    calls = []

    def local_remote(args):
        calls.append(args)
        assert args[3] == "https://github.com/NVIDIA/cuPhoton.git"
        return actual_run([*args[:3], str(tmp_path), *args[4:]])

    monkeypatch.setattr(publisher, "run", local_remote)
    assert publisher.remote_tag_commit("v0.2.0") == sha
    assert calls[0][-2:] == ["refs/tags/v0.2.0", "refs/tags/v0.2.0^{}"]
    with pytest.raises(ValueError, match="missing from the remote"):
        publisher.remote_tag_commit("v0.2.1")


@pytest.mark.parametrize("remote_change", ["moved", "missing"])
def test_remote_tag_change_during_render_cannot_publish(
    publication, monkeypatch, remote_change
):
    initial = publisher.prepare(tag="main", output=publication.root / "first")
    real_build = publisher.docs_build
    remote_sha = publication.sha

    def build_then_change_tag(**kwargs):
        nonlocal remote_sha
        result = real_build(**kwargs)
        remote_sha = "f" * 40 if remote_change == "moved" else None
        return result

    def current_remote_tag(tag):
        if remote_sha is None:
            raise ValueError(f"Release tag is missing from the remote: {tag}")
        return remote_sha

    monkeypatch.setattr(publisher, "docs_build", build_then_change_tag)
    monkeypatch.setattr(publisher, "remote_tag_commit", current_remote_tag)
    output = publication.root / "rejected-remote"
    with pytest.raises(ValueError, match="moved|missing"):
        publisher.prepare(tag="v0.2.0", output=output)
    assert not output.exists()
    assert (
        git(publication.remote, "rev-parse", "gh-pages")
        == initial["archive_sha"]
    )
    assert git(publication.source, "rev-parse", "v0.2.0") == publication.sha
