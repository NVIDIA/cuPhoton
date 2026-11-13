# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Download and verify the immutable snapshot artifact for a main workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from cuphoton_dev.github import REPOSITORY, api


def _extract(archive: Path, destination: Path) -> None:
    seen: set[str] = set()
    with zipfile.ZipFile(archive) as package:
        for member in package.infolist():
            name = member.filename.removesuffix("/")
            path = PurePosixPath(name)
            kind = stat.S_IFMT(member.external_attr >> 16)
            if (
                not path.parts
                or member.filename != member.orig_filename
                or path.is_absolute()
                or path.as_posix() != name
                or ".." in path.parts
                or any(part.lower() == ".git" for part in path.parts)
                or "\\" in name
                or ":" in name
                or name in seen
                or kind not in {0, stat.S_IFREG, stat.S_IFDIR}
                or (kind == stat.S_IFDIR and not member.is_dir())
                or (kind == stat.S_IFREG and member.is_dir())
            ):
                raise ValueError(f"Unsafe snapshot artifact entry: {name!r}")
            seen.add(name)
            target = destination.joinpath(*path.parts)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with package.open(member) as source, target.open("xb") as out:
                    shutil.copyfileobj(source, out)


def download(
    *, artifact_id: int, digest: str, output: Path
) -> dict[str, Any]:
    """Accept only verified bytes from this canonical main workflow run."""
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise ValueError(f"Output already exists: {output}")
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "")
    sha = os.environ.get("GITHUB_SHA", "")
    digest = digest.lower()
    if (
        os.environ.get("GITHUB_REPOSITORY") != REPOSITORY
        or not re.fullmatch(r"[1-9][0-9]*", run_id)
        or not re.fullmatch(r"[1-9][0-9]*", attempt)
        or not re.fullmatch(r"[0-9a-f]{40}", sha)
        or not re.fullmatch(r"[0-9a-f]{64}", digest)
        or artifact_id < 1
    ):
        raise ValueError(
            "Expected canonical workflow identity and artifact digest"
        )
    base = f"repos/{REPOSITORY}/actions"
    run = api(f"{base}/runs/{run_id}")
    if (
        run.get("id") != int(run_id)
        or run.get("path") != ".github/workflows/docs.yml"
        or run.get("event") not in {"push", "workflow_dispatch"}
        or run.get("head_branch") != "main"
        or run.get("head_sha") != sha
        or run.get("repository", {}).get("full_name") != REPOSITORY
        or run.get("head_repository", {}).get("full_name") != REPOSITORY
    ):
        raise ValueError(
            "Artifact download requires the canonical main docs workflow"
        )
    artifact = api(f"{base}/artifacts/{artifact_id}")
    producer = artifact.get("workflow_run") or {}
    name = re.fullmatch(
        rf"docs-snapshots-{run_id}-([1-9][0-9]*)", artifact.get("name", "")
    )
    if (
        artifact.get("id") != artifact_id
        or artifact.get("expired") is not False
        or artifact.get("digest") != f"sha256:{digest}"
        or producer.get("id") != int(run_id)
        or producer.get("head_sha") != sha
        or name is None
        or int(name[1]) > int(attempt)
    ):
        raise ValueError(
            "Snapshot artifact identity or digest does not match this run"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".docs-artifact-", dir=output.parent
    ) as temporary:
        work = Path(temporary)
        archive = work / "artifact.zip"
        with archive.open("wb") as stream:
            result = subprocess.run(
                [
                    "gh",
                    "api",
                    "-H",
                    "X-GitHub-Api-Version: 2026-03-10",
                    f"{base}/artifacts/{artifact_id}/zip",
                ],
                stdout=stream,
                stderr=subprocess.PIPE,
            )
        if result.returncode:
            raise RuntimeError(
                "Artifact download failed: "
                + result.stderr.decode("utf-8", errors="replace").strip()
            )
        with archive.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != digest:
            raise ValueError("Downloaded snapshot artifact SHA256 mismatch")
        extracted = work / "snapshots"
        extracted.mkdir()
        _extract(archive, extracted)
        extracted.rename(output)
    return {
        "artifact_id": artifact_id,
        "digest": digest,
        "output": str(output),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("download")
    command.add_argument("--artifact-id", required=True, type=int)
    command.add_argument("--digest", required=True)
    command.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            download(
                artifact_id=args.artifact_id,
                digest=args.digest,
                output=args.output,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
