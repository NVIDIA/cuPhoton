#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Resolve release tags and verify unchanged artifacts before publication."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

NUMBER = r"(?:0|[1-9][0-9]*)"
TAG = re.compile(rf"v({NUMBER}\.{NUMBER}\.{NUMBER}(?:rc{NUMBER})?)")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def release_version(tag):
    match = TAG.fullmatch(tag)
    require(match is not None, f"Expected vX.Y.Z or vX.Y.ZrcN: {tag!r}")
    return match[1]


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def resolve(tag, expected_sha=""):
    version = release_version(tag)
    sha = git("rev-parse", f"refs/tags/{tag}^{{commit}}")
    require(not expected_sha or sha == expected_sha, "Release tag moved")
    branches = ("refs/remotes/origin/main", "refs/remotes/origin/0.1.x")
    require(
        any(
            subprocess.run(
                ["git", "merge-base", "--is-ancestor", sha, branch],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            == 0
            for branch in branches
        ),
        "Release commit must belong to main or 0.1.x",
    )
    return version, sha


def hashes(directory):
    paths = sorted(directory.iterdir())
    require(bool(paths), "No distributions found")
    require(
        all(
            path.is_file()
            and not path.is_symlink()
            and path.name.endswith((".whl", ".tar.gz"))
            for path in paths
        ),
        "Distribution directory contains unexpected files",
    )
    result = {}
    for path in paths:
        with path.open("rb") as stream:
            result[path.name] = hashlib.file_digest(
                stream, "sha256"
            ).hexdigest()
    return result


def record(directory):
    from check_distributions import check_sdist

    (sdist,) = directory.glob("*.tar.gz")
    version = check_sdist(sdist)
    requested = os.environ.get("RELEASE_VERSION", "")
    require(not requested or version == requested, "Build version mismatch")
    return {
        "version": version,
        "tag": f"v{requested}" if requested else None,
        "source_sha": git("rev-parse", "HEAD"),
        "repository": os.environ["GITHUB_REPOSITORY"],
        "run_id": int(os.environ["GITHUB_RUN_ID"]),
        "run_attempt": int(os.environ["GITHUB_RUN_ATTEMPT"]),
        "workflow_sha": os.environ["GITHUB_WORKFLOW_SHA"],
        "files": hashes(directory),
    }


def validate_provenance(manifest, run, jobs, *, tag, sha, repository):
    require(manifest["version"] == release_version(tag), "Wrong version")
    require(manifest["tag"] == tag, "Wrong source tag")
    require(manifest["source_sha"] == sha, "Wrong source commit")
    require(manifest["repository"] == repository, "Wrong build repository")
    require(manifest["run_id"] == run["id"], "Wrong build run")
    require(
        run["head_repository"]["full_name"] == repository,
        "Build run belongs to another repository",
    )
    require(
        run["path"] == ".github/workflows/publish.yml",
        "Expected a release workflow build",
    )
    require(
        manifest["workflow_sha"] == run["head_sha"],
        "Wrong workflow commit",
    )
    if run["event"] == "push":
        require(run["head_branch"] == tag, "Run was for another tag")
        require(run["head_sha"] == sha, "Tag run built another commit")
    else:
        require(
            run["event"] == "workflow_dispatch"
            and run["head_branch"] == "main",
            "Expected tag push or manual release from main",
        )
    require(
        any(
            job["name"] == "release wheels / distributions"
            and job["status"] == "completed"
            and job["conclusion"] == "success"
            for job in jobs
        ),
        "Distribution validation did not succeed in the build attempt",
    )


def gh(endpoint):
    return json.loads(subprocess.check_output(["gh", "api", endpoint]))


def verify(directory, manifest_path, tag, sha, run_id):
    manifest = json.loads(manifest_path.read_text())
    repository = os.environ["GITHUB_REPOSITORY"]
    require(str(manifest["run_id"]) == run_id, "Wrong artifact run")
    require(
        isinstance(manifest["run_attempt"], int)
        and manifest["run_attempt"] > 0,
        "Invalid build attempt",
    )
    endpoint = f"repos/{repository}/actions/runs/{run_id}"
    run = gh(endpoint)
    jobs = gh(
        f"{endpoint}/attempts/{manifest['run_attempt']}/jobs?per_page=100"
    )
    require(jobs["total_count"] <= 100, "Unexpected release job count")
    validate_provenance(
        manifest, run, jobs["jobs"], tag=tag, sha=sha, repository=repository
    )
    require(manifest["files"] == hashes(directory), "Artifact hashes differ")


def remove_published(directory, published):
    """Retry partial uploads only when existing filenames match bytes."""
    local = hashes(directory)
    remote = {
        item["filename"]: item["digests"]["sha256"] for item in published
    }
    require(
        all(
            name in local and local[name] == sha
            for name, sha in remote.items()
        ),
        "Index already contains different files for this version",
    )
    for name in remote:
        (directory / name).unlink()
    return len(local) - len(remote)


def pending(directory, version, target):
    release_version(f"v{version}")
    host = "test.pypi.org" if target == "testpypi" else "pypi.org"
    try:
        with urlopen(
            f"https://{host}/pypi/cuphoton/{version}/json", timeout=30
        ) as response:
            published = json.load(response)["urls"]
    except HTTPError as error:
        if error.code != 404:
            raise
        published = []
    return remove_published(directory, published)


def output(**values):
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as stream:
        for key, value in values.items():
            stream.write(f"{key}={value}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    select = commands.add_parser("resolve")
    select.add_argument("tag")
    select.add_argument("--expected-sha", default="")
    save = commands.add_parser("record")
    save.add_argument("directory", type=Path)
    save.add_argument("manifest", type=Path)
    check = commands.add_parser("verify")
    check.add_argument("directory", type=Path)
    check.add_argument("manifest", type=Path)
    check.add_argument("--tag", required=True)
    check.add_argument("--sha", required=True)
    check.add_argument("--run-id", required=True)
    upload = commands.add_parser("pending")
    upload.add_argument("directory", type=Path)
    upload.add_argument("--version", required=True)
    upload.add_argument(
        "--target", choices=("pypi", "testpypi"), required=True
    )
    args = parser.parse_args()
    if args.command == "resolve":
        version, sha = resolve(args.tag, args.expected_sha)
        output(version=version, sha=sha, tag=args.tag)
    elif args.command == "record":
        args.manifest.write_text(
            json.dumps(record(args.directory), indent=2) + "\n"
        )
    elif args.command == "verify":
        verify(args.directory, args.manifest, args.tag, args.sha, args.run_id)
    else:
        output(pending=pending(args.directory, args.version, args.target))


if __name__ == "__main__":
    main()
