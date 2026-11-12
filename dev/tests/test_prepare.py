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
