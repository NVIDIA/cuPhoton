# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""One implementation for local commands and the main-only Pages publisher."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from cuphoton_dev.build import (
    FINAL_TAG,
    docs_build,
    git,
    renderer_version,
    resolve,
    run,
)
from cuphoton_dev.github import (
    REPOSITORY,
    docs_publish,
    docs_status,
    eligible,
    has_guide,
    releases,
)

__all__ = ["docs_build", "docs_check", "docs_publish", "docs_status"]


def docs_check(directory: Path) -> dict[str, Any]:
    from cuphoton_dev.archive import check_archive
    from cuphoton_dev.transforms import check_snapshot

    if (directory / "versions.json").exists():
        return check_archive(directory)
    return check_snapshot(directory)


def is_ancestor(repo: Path, older: str, newer: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", older, newer],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(f"Cannot establish main ancestry: {result.stderr}")
    return result.returncode == 0


def remote_tag_commit(tag: str) -> str:
    """Read the canonical tag, peeling annotated tags to their target."""
    if not FINAL_TAG.fullmatch(tag):
        raise ValueError(f"Expected a full release tag: {tag}")
    ref = f"refs/tags/{tag}"
    rows = run(
        [
            "git",
            "ls-remote",
            "--tags",
            f"https://github.com/{REPOSITORY}.git",
            ref,
            ref + "^{}",
        ]
    )
    refs = {
        name: sha for sha, name in (row.split() for row in rows.splitlines())
    }
    if ref not in refs:
        raise ValueError(f"Release tag is missing from the remote: {tag}")
    return refs.get(ref + "^{}", refs[ref])


def clone_archive(destination: Path) -> None:
    remote = f"https://github.com/{REPOSITORY}.git"
    existing = run(
        ["git", "ls-remote", "--heads", remote, "refs/heads/gh-pages"]
    )
    if existing:
        run(
            [
                "git",
                "clone",
                "--quiet",
                "--single-branch",
                "--branch",
                "gh-pages",
                remote,
                str(destination),
            ]
        )
    else:
        destination.mkdir()
        git(destination, "init", "--quiet", "--initial-branch=gh-pages")
        git(destination, "remote", "add", "origin", remote)


def _context(tag: str) -> tuple[Path, str, str, str, int]:
    if (
        os.environ.get("GITHUB_REPOSITORY") != REPOSITORY
        or os.environ.get("GITHUB_REF") != "refs/heads/main"
        or os.environ.get("GITHUB_EVENT_NAME")
        not in {"push", "workflow_dispatch"}
    ):
        raise ValueError(
            "Archive publication requires the canonical main workflow; "
            "use docs-publish"
        )
    repo = Path.cwd().resolve()
    controller = resolve(repo, "HEAD")
    if controller != os.environ.get("GITHUB_SHA"):
        raise ValueError(
            "Controller checkout must match the workflow main SHA"
        )
    tag = tag or "main"
    if tag not in {"main", "missing"} and not FINAL_TAG.fullmatch(tag):
        raise ValueError(
            "Expected main, missing, or a full release tag vX.Y.Z"
        )
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    attempt = int(os.environ.get("GITHUB_RUN_ATTEMPT", "0"))
    if not run_id.isascii() or not run_id.isdecimal() or attempt < 1:
        raise ValueError("Publication requires a workflow run and attempt")
    return repo, controller, tag, run_id, attempt


def _release_source(repo: Path, tag: str) -> str:
    source = resolve(repo, f"refs/tags/{tag}")
    if remote_tag_commit(tag) != source:
        raise ValueError(f"Release tag moved during publication: {tag}")
    supported = {
        "refs/remotes/origin/main",
        "refs/remotes/origin/0.1.x",
        "refs/remotes/origin/0.2.x",
    }
    branches = supported.intersection(
        git(
            repo, "for-each-ref", "--format=%(refname)", *sorted(supported)
        ).splitlines()
    )
    if not any(is_ancestor(repo, source, branch) for branch in branches):
        raise ValueError("Release commit must belong to main, 0.1.x or 0.2.x")
    return source


def _targets(
    repo: Path, controller: str, tag: str, retained: Path
) -> tuple[dict[str, str], set[str]]:
    from cuphoton_dev.archive import check_archive

    if (retained / "versions.json").exists():
        check_archive(retained)
    eligible_tags = {r["tag_name"] for r in releases() if eligible(r)}
    if tag not in {"main", "missing"} and tag not in eligible_tags:
        raise ValueError(f"{tag} is not a published final GitHub Release")
    targets = (
        sorted(t for t in eligible_tags if has_guide(t))
        if tag == "missing"
        else [tag]
    )
    if not (retained / "main/build.json").exists() and "main" not in targets:
        targets.insert(0, "main")
    sources = {
        target: controller
        if target == "main"
        else _release_source(repo, target)
        for target in targets
    }
    return sources, eligible_tags


def _reuse(prior: Path, target: str, source: str, controller: str) -> bool:
    from cuphoton_dev.transforms import check_snapshot

    if not prior.exists():
        return False
    receipt = check_snapshot(prior)
    if target != "main" and receipt["source_sha"] != source:
        raise ValueError(f"Retained {target} conflicts with the current tag")
    return target != "main" or (
        receipt["source_sha"] == source
        and receipt["controller_sha"] == controller
    )


def _check_render(
    path: Path, repo: Path, target: str, source: str, controller: str
) -> None:
    from cuphoton_dev.renderers import RENDERERS
    from cuphoton_dev.transforms import SITE_URL, check_snapshot

    receipt = check_snapshot(path)
    workflow = git(repo, "show", f"{source}:.github/workflows/docs.yml")
    renderer = renderer_version(workflow)
    if renderer not in RENDERERS:
        raise ValueError(f"Unsupported Quarto {renderer}")
    lock = subprocess.check_output(
        ["git", "-C", str(repo), "show", f"{source}:uv.lock"]
    )
    expected = {
        "version": target,
        "source_sha": source,
        "controller_sha": controller,
        "lock_sha256": hashlib.sha256(lock).hexdigest(),
        "renderer": renderer,
        "site_url": SITE_URL,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"Rendered {target} has unexpected {key}")


def render_bundle(
    *, tag: str, output: Path, request_id: str = ""
) -> dict[str, Any]:
    """Render missing snapshots in a read-only workflow job.

    Retained snapshots are never copied into the bundle. The writer re-reads
    them independently, so a retry after retention does not rebuild history.
    """
    repo, controller, tag, run_id, attempt = _context(tag)
    output = output.resolve()
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    with tempfile.TemporaryDirectory(prefix="cuphoton-render-") as temporary:
        work = Path(temporary)
        retained = work / "archive"
        clone_archive(retained)
        sources, _ = _targets(repo, controller, tag, retained)
        bundle = work / "bundle"
        bundle.mkdir()
        rendered = {}
        for target, source in sources.items():
            if _reuse(retained / target, target, source, controller):
                continue
            docs_build(
                ref=controller if target == "main" else f"refs/tags/{target}",
                output=bundle / target,
                repo=repo,
            )
            _check_render(bundle / target, repo, target, source, controller)
            rendered[target] = source
        manifest = {
            "schema": 1,
            "repository": REPOSITORY,
            "controller_sha": controller,
            "run_id": run_id,
            "run_attempt": attempt,
            "tag": tag,
            "request_id": request_id,
            "snapshots": rendered,
        }
        (bundle / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(bundle, output)
    return {"output": str(output), **manifest}


def _bundle_sources(
    bundle: Path,
    *,
    controller: str,
    tag: str,
    run_id: str,
    attempt: int,
    request_id: str,
) -> dict[str, str]:
    if bundle.is_symlink() or any(p.is_symlink() for p in bundle.iterdir()):
        raise ValueError("Snapshot bundle contains a symbolic link")
    manifest = json.loads(
        (bundle / "manifest.json").read_text(encoding="utf-8")
    )
    expected = {
        "schema": 1,
        "repository": REPOSITORY,
        "controller_sha": controller,
        "run_id": run_id,
        "tag": tag,
        "request_id": request_id,
    }
    if not isinstance(manifest, dict):
        raise ValueError("Snapshot bundle manifest must be an object")
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Snapshot bundle has unexpected {key}")
    producer_attempt = manifest.get("run_attempt")
    if (
        type(producer_attempt) is not int
        or not 1 <= producer_attempt <= attempt
    ):
        raise ValueError("Snapshot bundle has unexpected run_attempt")
    sources = manifest.get("snapshots")
    if not isinstance(sources, dict) or not all(
        isinstance(target, str)
        and (target == "main" or FINAL_TAG.fullmatch(target))
        and isinstance(source, str)
        for target, source in sources.items()
    ):
        raise ValueError("Snapshot bundle has invalid sources")
    if {p.name for p in bundle.iterdir()} != {"manifest.json", *sources}:
        raise ValueError(
            "Snapshot bundle inventory differs from its manifest"
        )
    return sources


def prepare(
    *, tag: str, snapshots: Path, output: Path, request_id: str = ""
) -> dict[str, Any]:
    """Validate transferred output and retain the site without rendering.

    The caller verifies the immutable Actions artifact identity and digest.
    This trusted-main phase independently checks source policy and receipts.
    """
    from cuphoton_dev.archive import assemble, check_archive

    repo, controller, tag, run_id, attempt = _context(tag)
    output = output.resolve()
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    supplied = _bundle_sources(
        snapshots,
        controller=controller,
        tag=tag,
        run_id=run_id,
        attempt=attempt,
        request_id=request_id,
    )
    with tempfile.TemporaryDirectory(
        prefix="cuphoton-publication-"
    ) as temporary:
        work = Path(temporary)
        retained = work / "archive"
        clone_archive(retained)
        sources, eligible_tags = _targets(repo, controller, tag, retained)
        for target, source in supplied.items():
            # A downstream retry can find main already bootstrapped, possibly
            # by a newer publication. Validate this unused proposal but do
            # not replace the currently retained main with it.
            expected_source = sources.get(target)
            if target == "main" and expected_source is None and tag != "main":
                expected_source = controller
            if expected_source != source:
                raise ValueError(
                    f"Snapshot bundle has unexpected source: {target}"
                )
            _check_render(
                snapshots / target, repo, target, source, controller
            )
        for target, source in sources.items():
            prior = retained / target
            if _reuse(prior, target, source, controller):
                snapshot = prior
            elif target in supplied:
                snapshot = snapshots / target
            else:
                raise ValueError(
                    f"Snapshot bundle is missing required {target}"
                )
            assemble(
                retained,
                snapshot,
                eligible_releases=eligible_tags,
                is_ancestor=lambda older, newer: is_ancestor(
                    repo, older, newer
                ),
            )
        if not sources:
            assemble(
                retained,
                retained / "main",
                eligible_releases=eligible_tags,
                is_ancestor=lambda older, newer: is_ancestor(
                    repo, older, newer
                ),
            )
        if eligible_tags != {
            r["tag_name"] for r in releases() if eligible(r)
        }:
            raise RuntimeError(
                "Release eligibility changed during publication; retry"
            )
        catalog = check_archive(retained)
        git(retained, "add", "--all")
        changed = bool(git(retained, "status", "--porcelain"))
        if changed:
            git(
                retained,
                "-c",
                "user.name=github-actions[bot]",
                "-c",
                "user.email=41898282+github-actions[bot]@users.noreply.github.com",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "-m",
                "Retain validated documentation and refresh version routes",
            )
        # Tags can move independently of Release eligibility. Re-read them
        # after rendering and before pushing or deploying retained output.
        for release_tag, source_sha in sources.items():
            if release_tag == "main":
                continue
            if remote_tag_commit(release_tag) != source_sha:
                raise ValueError(
                    f"Release tag moved during publication: {release_tag}"
                )
        if changed:
            # Scope credentials to this push. No credential is written
            # to repository files, command arguments, logs or rendered output.
            git(
                retained,
                "-c",
                "credential.helper=",
                "-c",
                "credential.helper=!gh auth git-credential",
                "push",
                "origin",
                "HEAD:refs/heads/gh-pages",
            )
        archive_sha = resolve(retained, "HEAD")
        shutil.copytree(
            retained, output, ignore=shutil.ignore_patterns(".git")
        )
    return {
        "request_id": request_id,
        "controller_sha": controller,
        "archive_sha": archive_sha,
        "output": str(output),
        "catalog": catalog,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name in ("render", "prepare"):
        command = subcommands.add_parser(name)
        command.add_argument("--tag", default="main")
        command.add_argument("--request-id", default="")
        command.add_argument("--output", required=True, type=Path)
        if name == "prepare":
            command.add_argument("--snapshots", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "render":
        result = render_bundle(
            tag=args.tag, request_id=args.request_id, output=args.output
        )
    else:
        result = prepare(
            tag=args.tag,
            request_id=args.request_id,
            snapshots=args.snapshots,
            output=args.output,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
