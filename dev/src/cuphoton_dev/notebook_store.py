# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Content-addressed notebook assets for the cuPhoton documentation."""

from __future__ import annotations

import hashlib
import re
import subprocess
import tempfile
import time
from pathlib import Path

from cuphoton_dev.github import REPOSITORY

CHAPTERS = ("model-sky", "model-scene", "model-alerts")

# Read older assets after a new documentation asset release is added.
ASSET_RELEASES = ("docs-notebook-assets",)


def digest_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _asset_name(digest: str) -> str:
    if not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise ValueError("Expected a SHA-256 notebook asset digest")
    return f"sha256-{digest}"


def _download(digest: str, destination: Path, *, fresh: bool = False) -> bool:
    name = _asset_name(digest)
    for tag in reversed(ASSET_RELEASES):
        url = (
            f"https://github.com/{REPOSITORY}/releases/download/{tag}/{name}"
        )
        # The release CDN may cache the 404 from the pre-upload lookup.
        if fresh:
            url += f"?verify={time.time_ns()}"
        result = subprocess.run(
            [
                "curl",
                "--silent",
                "--show-error",
                "--fail",
                "--location",
                "--proto",
                "=https",
                "--proto-redir",
                "=https",
                "--retry",
                "3",
                "--connect-timeout",
                "20",
                "--max-time",
                "300",
                "--write-out",
                "%{http_code}",
                "--output",
                str(destination),
                url,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 22 and result.stdout == "404":
            continue
        if result.returncode:
            raise RuntimeError(
                f"Notebook asset download failed: {result.stderr.strip()}"
            )
        if digest_file(destination) != digest:
            raise ValueError(f"Notebook asset SHA-256 mismatch: {digest}")
        return True
    return False


def get_blob(digest: str, destination: Path) -> None:
    """Download and authenticate an asset before exposing it to callers."""
    with tempfile.TemporaryDirectory(dir=destination.parent) as scratch:
        incoming = Path(scratch) / "download"
        if not _download(digest, incoming) and not _download(
            digest, incoming, fresh=True
        ):
            raise FileNotFoundError(f"Missing documentation asset: {digest}")
        incoming.replace(destination)


def put_blob(path: Path, digest: str) -> None:
    """Publish an immutable-by-name asset. Existing bytes must match."""
    name = _asset_name(digest)
    if path.is_symlink() or not path.is_file() or digest_file(path) != digest:
        raise ValueError(f"Invalid content for notebook asset: {digest}")
    with tempfile.TemporaryDirectory() as scratch:
        downloaded = Path(scratch) / "download"
        if _download(digest, downloaded):
            return
        # gh uses the local filename as the remote asset name.
        asset = Path(scratch) / name
        asset.symlink_to(path.resolve())
        result = subprocess.run(
            [
                "gh",
                "release",
                "upload",
                ASSET_RELEASES[-1],
                str(asset),
                "--repo",
                REPOSITORY,
            ],
            text=True,
            capture_output=True,
        )
        # Concurrent uploads may race. Authenticate the saved bytes.
        if not _download(digest, downloaded, fresh=True):
            raise RuntimeError(
                f"Notebook asset upload failed: {result.stderr.strip()}"
            )
