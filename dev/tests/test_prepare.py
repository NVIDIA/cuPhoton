# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Exercise publication transactions against real local Git repositories."""

from __future__ import annotations

import hashlib
import json
import shutil
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
    (source / "uv.lock").write_text("# exact lock bytes\n", encoding="utf-8")
    workflow = source / ".github/workflows/docs.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(
        "url: https://github.com/quarto-dev/quarto-cli/releases/download/"
        "v1.9.36/quarto-1.9.36-linux-amd64.deb\n",
        encoding="utf-8",
    )
    git(source, "add", ".")
    git(source, "commit", "-qm", "first source")
    sha = git(source, "rev-parse", "HEAD")
    git(source, "tag", "v0.2.0")
    git(source, "update-ref", "refs/remotes/origin/main", sha)
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
            version=ref.removeprefix("refs/tags/")
            if ref.startswith("refs/tags/")
            else "main",
            source_sha=publisher.resolve(repo, ref),
            controller_sha=publisher.resolve(source, "HEAD"),
            renderer="1.9.36",
            lock_sha256=hashlib.sha256(
                subprocess.check_output(
                    ["git", "-C", str(repo), "show", f"{ref}:uv.lock"]
                )
            ).hexdigest(),
        )

    monkeypatch.chdir(source)
    monkeypatch.setenv("GITHUB_REPOSITORY", "NVIDIA/cuPhoton")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setenv("GITHUB_SHA", sha)
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
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


def publish(*, tag, output, request_id=""):
    bundle = output.with_name(output.name + "-bundle")
    publisher.render_bundle(tag=tag, output=bundle, request_id=request_id)
    return publisher.prepare(
        tag=tag, snapshots=bundle, output=output, request_id=request_id
    )


def test_prepare_retries_retained_main_after_deployment_failure(
    publication, monkeypatch
):
    first = publication.root / "first"
    result = publish(tag="main", output=first)
    assert result["catalog"]["preferred"] == "main"
    hashes = file_hashes(first)
    assert len(publication.builds) == 1
    # Pages could fail after retention. Recovery must work even if rendering
    # is temporarily unavailable and must preserve the complete artifact.
    monkeypatch.setattr(
        publisher, "docs_build", lambda **kwargs: pytest.fail("must reuse")
    )
    retry = publication.root / "retry"
    recovered = publish(tag="main", output=retry)
    assert recovered["archive_sha"] == result["archive_sha"]
    assert file_hashes(retry) == hashes
    assert not (retry / ".git").exists()


def test_prepare_duplicate_release_and_next_main_preserve_history(
    publication, monkeypatch
):
    first = publication.root / "first"
    original = publish(tag="v0.2.0", output=first)
    release_hashes = file_hashes(first / "v0.2.0")
    assert len(publication.builds) == 2  # Bootstrap main and the release.
    duplicate = publish(tag="v0.2.0", output=publication.root / "duplicate")
    assert duplicate["archive_sha"] == original["archive_sha"]
    assert len(publication.builds) == 2
    (publication.source / "source.txt").write_text("next source\n")
    git(publication.source, "commit", "-qam", "next source")
    sha = git(publication.source, "rev-parse", "HEAD")
    monkeypatch.setenv("GITHUB_SHA", sha)
    git(publication.source, "update-ref", "refs/remotes/origin/main", sha)
    updated = publication.root / "updated"
    publish(tag="main", output=updated)
    assert len(publication.builds) == 3
    assert file_hashes(updated / "v0.2.0") == release_hashes
    assert publisher.docs_check(updated)["stable"] == "v0.2.0"


def test_prepare_rejects_moved_tag_without_pushing(publication, monkeypatch):
    first = publication.root / "first"
    original = publish(tag="v0.2.0", output=first)
    (publication.source / "source.txt").write_text("different source\n")
    git(publication.source, "commit", "-qam", "different source")
    sha = git(publication.source, "rev-parse", "HEAD")
    monkeypatch.setenv("GITHUB_SHA", sha)
    git(publication.source, "tag", "-f", "v0.2.0")
    git(publication.source, "update-ref", "refs/remotes/origin/main", sha)
    output = publication.root / "rejected"
    with pytest.raises(ValueError, match="conflicts with the current tag"):
        publish(tag="v0.2.0", output=output)
    assert not output.exists()
    assert (
        git(publication.remote, "rev-parse", "gh-pages")
        == original["archive_sha"]
    )


@pytest.mark.parametrize("failure", ["build", "eligibility", "push"])
def test_prepare_failure_does_not_change_retained_site(
    publication, monkeypatch, failure
):
    original = publish(tag="main", output=publication.root / "first")
    if failure == "build":

        def broken_build(**kwargs):
            raise RuntimeError("renderer failed")

        monkeypatch.setattr(publisher, "docs_build", broken_build)
    elif failure == "eligibility":
        records = iter([[release()], [release()], [release(draft=True)]])
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
        publish(tag="v0.2.0", output=output)
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
    initial = publish(tag="main", output=publication.root / "first")
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
        publish(tag="v0.2.0", output=output)
    assert not output.exists()
    assert (
        git(publication.remote, "rev-parse", "gh-pages")
        == initial["archive_sha"]
    )
    assert git(publication.source, "rev-parse", "v0.2.0") == publication.sha


def test_prepare_missing_skips_releases_without_a_guide(
    publication, monkeypatch
):
    git(publication.source, "tag", "v0.1.3")
    monkeypatch.setattr(
        publisher, "releases", lambda: [release(), release("v0.1.3")]
    )
    monkeypatch.setattr(publisher, "has_guide", lambda tag: tag == "v0.2.0")
    output = publication.root / "backfill"
    result = publish(tag="missing", output=output)
    assert publication.builds == [publication.sha, "refs/tags/v0.2.0"]
    assert {entry["version"] for entry in result["catalog"]["versions"]} == {
        "main",
        "v0.2.0",
    }
    assert result["catalog"]["stable"] == "v0.2.0"
    assert not (output / "v0.1.3").exists()
    publisher.docs_check(output)


def test_prepare_empty_backfill_refreshes_withdrawn_release(
    publication, monkeypatch
):
    first = publication.root / "first"
    initial = publish(tag="v0.2.0", output=first)
    assert initial["catalog"]["stable"] == "v0.2.0"
    retained_hashes = {
        version: file_hashes(first / version)
        for version in ("main", "v0.2.0")
    }
    release_receipt = (first / "v0.2.0/build.json").read_bytes()
    monkeypatch.setattr(
        publisher, "releases", lambda: [release(prerelease=True)]
    )
    monkeypatch.setattr(
        publisher, "docs_build", lambda **kwargs: pytest.fail("must retain")
    )
    refreshed = publication.root / "refreshed"
    result = publish(tag="missing", output=refreshed)
    assert result["catalog"]["stable"] is None
    assert result["catalog"]["preferred"] == "main"
    entries = {e["version"]: e for e in result["catalog"]["versions"]}
    assert not entries["v0.2.0"]["released"]
    assert {
        v: file_hashes(refreshed / v) for v in retained_hashes
    } == retained_hashes
    assert (refreshed / "v0.2.0/build.json").read_bytes() == release_receipt
    assert result["archive_sha"] != initial["archive_sha"]
    assert publisher.docs_check(refreshed)["stable"] is None


def test_writer_never_renders_and_retries_earlier_attempt(
    publication, monkeypatch
):
    bundle = publication.root / "bundle"
    publisher.render_bundle(tag="v0.2.0", output=bundle, request_id="request")
    assert not git(publication.remote, "for-each-ref", "refs/heads/gh-pages")
    monkeypatch.setattr(
        publisher,
        "docs_build",
        lambda **kwargs: pytest.fail("writer cannot render"),
    )
    first = publisher.prepare(
        tag="v0.2.0",
        snapshots=bundle,
        output=publication.root / "first",
        request_id="request",
    )
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    recovered = publisher.prepare(
        tag="v0.2.0",
        snapshots=bundle,
        output=publication.root / "recovered",
        request_id="request",
    )
    assert recovered["archive_sha"] == first["archive_sha"]
    assert file_hashes(publication.root / "first") == file_hashes(
        publication.root / "recovered"
    )
    duplicate = publisher.render_bundle(
        tag="v0.2.0", output=publication.root / "duplicate"
    )
    assert duplicate["snapshots"] == {}
    assert {p.name for p in (publication.root / "duplicate").iterdir()} == {
        "manifest.json"
    }


@pytest.mark.parametrize(
    "key,value",
    [
        ("repository", "other/cuPhoton"),
        ("controller_sha", "f" * 40),
        ("run_id", "999"),
        ("run_attempt", 0),
        ("run_attempt", 2),
        ("run_attempt", True),
        ("tag", "v0.2.0"),
        ("request_id", "other"),
    ],
)
def test_writer_rejects_bundle_from_another_publication(
    publication, key, value
):
    bundle = publication.root / "bundle"
    publisher.render_bundle(tag="main", output=bundle)
    path = bundle / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest[key] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unexpected " + key):
        publisher.prepare(
            tag="main", snapshots=bundle, output=publication.root / "site"
        )
    assert not git(publication.remote, "for-each-ref", "refs/heads/gh-pages")


@pytest.mark.parametrize(
    "key,value",
    [
        ("source_sha", "f" * 40),
        ("controller_sha", "f" * 40),
        ("lock_sha256", "f" * 64),
        ("renderer", "1.0.0"),
    ],
)
def test_writer_rejects_self_consistent_but_wrong_receipt(
    publication, key, value
):
    bundle = publication.root / "bundle"
    publisher.render_bundle(tag="main", output=bundle)
    path = bundle / "main/build.json"
    receipt = json.loads(path.read_text())
    receipt[key] = value
    path.write_text(json.dumps(receipt))
    # Receipt fields are outside the file hashes, but must still match the
    # independently resolved target rather than trusting the producer.
    publisher.docs_check(bundle / "main")
    with pytest.raises(ValueError, match="unexpected " + key):
        publisher.prepare(
            tag="main", snapshots=bundle, output=publication.root / "site"
        )
    assert not git(publication.remote, "for-each-ref", "refs/heads/gh-pages")


@pytest.mark.parametrize(
    "failure", ["missing", "inventory", "hash", "source"]
)
def test_writer_refuses_incomplete_or_changed_bundle(publication, failure):
    bundle = publication.root / "bundle"
    publisher.render_bundle(tag="main", output=bundle)
    if failure in {"missing", "source"}:
        path = bundle / "manifest.json"
        manifest = json.loads(path.read_text())
        if failure == "missing":
            shutil.rmtree(bundle / "main")
            manifest["snapshots"] = {}
        else:
            manifest["snapshots"]["main"] = "f" * 40
        path.write_text(json.dumps(manifest))
    elif failure == "inventory":
        (bundle / "extra.txt").write_text("unexpected")
    else:
        (bundle / "main/index.html").write_text("changed")
    with pytest.raises(
        ValueError,
        match="missing required|inventory|hashes|unexpected source",
    ):
        publisher.prepare(
            tag="main", snapshots=bundle, output=publication.root / "site"
        )
    assert not git(publication.remote, "for-each-ref", "refs/heads/gh-pages")


@pytest.mark.parametrize("phase", ["render", "prepare"])
def test_publication_rejects_release_outside_supported_branches(
    publication, phase
):
    bundle = publication.root / "bundle"
    if phase == "prepare":
        publisher.render_bundle(tag="v0.2.0", output=bundle)
    # A real unrelated commit exists, but none of the supported origin refs
    # contains the tagged source any longer.
    tree = git(publication.source, "rev-parse", "HEAD^{tree}")
    unrelated = subprocess.check_output(
        ["git", "-C", str(publication.source), "commit-tree", tree],
        input="unrelated root\n",
        text=True,
    ).strip()
    git(
        publication.source,
        "update-ref",
        "refs/remotes/origin/main",
        unrelated,
    )
    with pytest.raises(
        ValueError, match="must belong to main, 0.1.x or 0.2.x"
    ):
        if phase == "render":
            publisher.render_bundle(tag="v0.2.0", output=bundle)
        else:
            publisher.prepare(
                tag="v0.2.0",
                snapshots=bundle,
                output=publication.root / "site",
            )
    assert not git(publication.remote, "for-each-ref", "refs/heads/gh-pages")


@pytest.mark.parametrize("branch", ["0.1.x", "0.2.x"])
def test_publication_accepts_supported_release_branch(publication, branch):
    git(publication.source, "update-ref", "-d", "refs/remotes/origin/main")
    git(
        publication.source,
        "update-ref",
        f"refs/remotes/origin/{branch}",
        publication.sha,
    )
    result = publish(tag="v0.2.0", output=publication.root / "site")
    assert result["catalog"]["stable"] == "v0.2.0"


@pytest.mark.parametrize("phase", ["render", "prepare"])
def test_both_phases_require_canonical_main(publication, monkeypatch, phase):
    monkeypatch.setenv("GITHUB_REF", "refs/tags/v0.2.0")
    with pytest.raises(ValueError, match="canonical main"):
        if phase == "render":
            publisher.render_bundle(
                tag="main", output=publication.root / "bundle"
            )
        else:
            publisher.prepare(
                tag="main",
                snapshots=publication.root / "bundle",
                output=publication.root / "site",
            )
    assert not publication.builds
    assert not git(publication.remote, "for-each-ref", "refs/heads/gh-pages")


def test_similarly_named_branch_does_not_authorize_release(publication):
    git(publication.source, "update-ref", "-d", "refs/remotes/origin/main")
    git(
        publication.source,
        "update-ref",
        "refs/remotes/origin/main/topic",
        publication.sha,
    )
    with pytest.raises(
        ValueError, match="must belong to main, 0.1.x or 0.2.x"
    ):
        publisher.render_bundle(
            tag="v0.2.0", output=publication.root / "bundle"
        )


def test_retry_ignores_obsolete_bootstrap_main(publication, monkeypatch):
    old_controller = publication.sha
    old_bundle = publication.root / "old-bundle"
    publisher.render_bundle(tag="v0.2.0", output=old_bundle)
    (publication.source / "source.txt").write_text("new main\n")
    git(publication.source, "commit", "-qam", "new main")
    new_controller = git(publication.source, "rev-parse", "HEAD")
    git(
        publication.source,
        "update-ref",
        "refs/remotes/origin/main",
        new_controller,
    )
    monkeypatch.setenv("GITHUB_SHA", new_controller)
    current = publication.root / "current"
    publish(tag="main", output=current)
    hashes = file_hashes(current / "main")
    git(publication.source, "checkout", "--detach", old_controller)
    monkeypatch.setenv("GITHUB_SHA", old_controller)
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    site = publication.root / "retry"
    publisher.prepare(tag="v0.2.0", snapshots=old_bundle, output=site)
    assert file_hashes(site / "main") == hashes
    assert publisher.docs_check(site)["stable"] == "v0.2.0"


def test_writer_checks_release_lock_from_tag_not_controller(
    publication, monkeypatch
):
    tag_lock = (publication.source / "uv.lock").read_bytes()
    (publication.source / "uv.lock").write_text("# updated controller lock\n")
    git(publication.source, "commit", "-qam", "update lock")
    controller = git(publication.source, "rev-parse", "HEAD")
    git(
        publication.source,
        "update-ref",
        "refs/remotes/origin/main",
        controller,
    )
    monkeypatch.setenv("GITHUB_SHA", controller)
    site = publication.root / "site"
    publish(tag="v0.2.0", output=site)
    receipt = publisher.docs_check(site / "v0.2.0")
    main = publisher.docs_check(site / "main")
    assert receipt["lock_sha256"] == hashlib.sha256(tag_lock).hexdigest()
    assert receipt["lock_sha256"] != main["lock_sha256"]
    assert receipt["source_sha"] == publication.sha
    assert receipt["controller_sha"] == controller
