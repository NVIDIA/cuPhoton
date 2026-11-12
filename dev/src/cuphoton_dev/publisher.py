# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""One implementation for local commands and the main-only Pages publisher."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from cuphoton_dev.build import FINAL_TAG, docs_build, git, resolve, run
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
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(f"Cannot establish main ancestry: {result.stderr}")
    return result.returncode == 0


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


def prepare(
    *, tag: str, output: Path, request_id: str = ""
) -> dict[str, Any]:
    """Assemble and retain a full site inside the workflow's publication lock.

    Deployment uses native Pages actions after this succeeds. A failed deploy
    can retry this operation, reusing matching release snapshots verbatim.
    """
    from cuphoton_dev.archive import assemble, check_archive
    from cuphoton_dev.transforms import check_snapshot

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
    output = output.resolve()
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    tag = tag or "main"
    if tag not in {"main", "missing"} and not FINAL_TAG.fullmatch(tag):
        raise ValueError(
            "Expected main, missing, or a full release tag vX.Y.Z"
        )
    release_records = releases()
    eligible_tags = {r["tag_name"] for r in release_records if eligible(r)}
    if tag not in {"main", "missing"} and tag not in eligible_tags:
        raise ValueError(f"{tag} is not a published final GitHub Release")
    if tag == "missing":
        targets = sorted(t for t in eligible_tags if has_guide(t))
    else:
        targets = [tag]
    with tempfile.TemporaryDirectory(
        prefix="cuphoton-publication-"
    ) as temporary:
        work = Path(temporary)
        retained = work / "archive"
        clone_archive(retained)
        if (retained / "versions.json").exists():
            check_archive(retained)
        if (
            not (retained / "main/build.json").exists()
            and "main" not in targets
        ):
            targets.insert(0, "main")
        for target in targets:
            source_ref = (
                controller if target == "main" else f"refs/tags/{target}"
            )
            source_sha = resolve(repo, source_ref)
            prior = retained / target
            receipt = check_snapshot(prior) if prior.exists() else None
            if (
                target != "main"
                and receipt is not None
                and receipt["source_sha"] != source_sha
            ):
                raise ValueError(
                    f"Retained {target} conflicts with the current tag"
                )
            reuse = receipt is not None and (
                target != "main"
                or (
                    receipt["source_sha"] == source_sha
                    and receipt["controller_sha"] == controller
                )
            )
            if reuse:
                # Retained main also permits deployment recovery without
                # rendering again, but a new controller must rebuild it.
                snapshot = prior
            else:
                snapshot = work / f"build-{target}"
                docs_build(
                    ref=controller if target == "main" else target,
                    output=snapshot,
                    repo=repo,
                )
            assemble(
                retained,
                snapshot,
                eligible_releases=eligible_tags,
                is_ancestor=lambda older, newer: is_ancestor(
                    repo, older, newer
                ),
            )
        if not targets:
            # Empty reconciliation refreshes generated aliases/catalog.
            assemble(
                retained,
                retained / "main",
                eligible_releases=eligible_tags,
                is_ancestor=lambda older, newer: is_ancestor(
                    repo, older, newer
                ),
            )
        # Release state can change during a render. Do not publish a decision
        # made before a release was withdrawn or promoted; retry instead.
        if eligible_tags != {
            r["tag_name"] for r in releases() if eligible(r)
        }:
            raise RuntimeError(
                "Release eligibility changed during publication; retry"
            )
        catalog = check_archive(retained)
        git(retained, "add", "--all")
        if git(retained, "status", "--porcelain"):
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
    command = subcommands.add_parser("prepare")
    command.add_argument("--tag", default="main")
    command.add_argument("--request-id", default="")
    command.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = prepare(
        tag=args.tag, request_id=args.request_id, output=args.output
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
