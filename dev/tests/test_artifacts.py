# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import io
import os
import stat
import sys
import warnings
import zipfile
from types import SimpleNamespace

import pytest

from cuphoton_dev import artifacts


def zipped(entries):
    output = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(output, "w") as package:
            for entry, content in entries:
                package.writestr(entry, content)
    return output.getvalue()


@pytest.fixture
def transfer(tmp_path, monkeypatch):
    sha = "a" * 40
    run_id = 123
    for key, value in {
        "GITHUB_REPOSITORY": "NVIDIA/cuPhoton",
        "GITHUB_RUN_ID": str(run_id),
        "GITHUB_RUN_ATTEMPT": "2",
        "GITHUB_SHA": sha,
    }.items():
        monkeypatch.setenv(key, value)
    run = {
        "id": run_id,
        "path": ".github/workflows/docs.yml",
        "event": "workflow_dispatch",
        "head_branch": "main",
        "head_sha": sha,
        "repository": {"full_name": "NVIDIA/cuPhoton"},
        "head_repository": {"full_name": "NVIDIA/cuPhoton"},
    }
    artifact = {
        "id": 456,
        "name": "docs-snapshots-123-1",
        "expired": False,
        "workflow_run": {"id": run_id, "head_sha": sha},
    }
    payload = tmp_path / "response.zip"

    def set_payload(content):
        payload.write_bytes(content)
        digest = hashlib.sha256(content).hexdigest()
        artifact["digest"] = "sha256:" + digest
        return digest

    digest = set_payload(
        zipped(
            [
                ("manifest.json", b"{}"),
                ("main/", b""),
                ("main/index.html", b"verified snapshot"),
            ]
        )
    )
    calls = []

    def api(path):
        calls.append(path)
        if path == "repos/NVIDIA/cuPhoton/actions/runs/123":
            return run
        assert path == "repos/NVIDIA/cuPhoton/actions/artifacts/456"
        return artifact

    monkeypatch.setattr(artifacts, "api", api)
    tools = tmp_path / "tools"
    tools.mkdir()
    downloaded = tmp_path / "downloaded"
    gh = tools / "gh"
    gh.write_text(
        f"#!{sys.executable}\nimport sys\nfrom pathlib import Path\n"
        "assert sys.argv[-1] == "
        "'repos/NVIDIA/cuPhoton/actions/artifacts/456/zip'\n"
        f"Path({str(downloaded)!r}).touch()\n"
        f"sys.stdout.buffer.write(Path({str(payload)!r}).read_bytes())\n",
        encoding="utf-8",
    )
    gh.chmod(0o755)
    monkeypatch.setenv("PATH", str(tools) + os.pathsep + os.environ["PATH"])
    return SimpleNamespace(
        run=run,
        artifact=artifact,
        payload=payload,
        digest=digest,
        output=tmp_path / "snapshot",
        downloaded=downloaded,
        set_payload=set_payload,
        calls=calls,
    )


def test_prior_producer_attempt_downloads_verified_bytes(
    transfer, monkeypatch, capsys
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "artifacts",
            "download",
            "--artifact-id",
            "456",
            "--digest",
            transfer.digest,
            "--output",
            str(transfer.output),
        ],
    )
    assert artifacts.main() == 0
    assert (
        transfer.output / "main/index.html"
    ).read_bytes() == b"verified snapshot"
    assert transfer.downloaded.exists()
    assert transfer.digest in capsys.readouterr().out
    assert len(transfer.calls) == 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("run", 789),
        ("head", "b" * 40),
        ("digest", "sha256:" + "0" * 64),
        ("expired", True),
        ("attempt", "docs-snapshots-123-3"),
        ("name", "docs-snapshots-789-1"),
    ],
)
def test_rejects_other_or_unverified_artifact_before_download(
    transfer, field, value
):
    if field == "run":
        transfer.artifact["workflow_run"]["id"] = value
    elif field == "head":
        transfer.artifact["workflow_run"]["head_sha"] = value
    else:
        transfer.artifact[
            "name" if field in {"attempt", "name"} else field
        ] = value
    with pytest.raises(ValueError, match="identity or digest"):
        artifacts.download(
            artifact_id=456, digest=transfer.digest, output=transfer.output
        )
    assert not transfer.downloaded.exists()
    assert not transfer.output.exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("path", ".github/workflows/docs-preview.yml"),
        ("event", "pull_request"),
        ("head_branch", "topic"),
        ("head_sha", "b" * 40),
        ("repository", {"full_name": "fork/cuPhoton"}),
    ],
)
def test_rejects_nonpublisher_run(transfer, field, value):
    transfer.run[field] = value
    with pytest.raises(ValueError, match="canonical main docs workflow"):
        artifacts.download(
            artifact_id=456, digest=transfer.digest, output=transfer.output
        )
    assert not transfer.downloaded.exists()
    assert not transfer.output.exists()


def test_rejects_changed_download_bytes_without_exposing_output(transfer):
    transfer.payload.write_bytes(
        zipped([("main/index.html", b"substituted")])
    )
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        artifacts.download(
            artifact_id=456, digest=transfer.digest, output=transfer.output
        )
    assert transfer.downloaded.exists()
    assert not transfer.output.exists()
    assert not list(transfer.output.parent.glob(".docs-artifact-*"))


@pytest.mark.parametrize(
    "unsafe", ["traversal", "absolute", "symlink", "duplicate", "git"]
)
def test_rejects_unsafe_zip_with_matching_digest(transfer, unsafe):
    if unsafe == "symlink":
        entry = zipfile.ZipInfo("link")
        entry.create_system = 3
        entry.external_attr = (stat.S_IFLNK | 0o777) << 16
        entries = [(entry, b"../escaped")]
    elif unsafe == "duplicate":
        entries = [
            ("main/index.html", b"first"),
            ("main/index.html", b"second"),
        ]
    elif unsafe == "git":
        entries = [
            ("main/.git/config", b"[core]\nrepositoryformatversion=0\n")
        ]
    else:
        name = (
            "../escaped"
            if unsafe == "traversal"
            else str(transfer.output.parent / "escaped")
        )
        entries = [(name, b"escaped")]
    # A preceding good entry exercises cleanup after extraction has begun.
    digest = transfer.set_payload(
        zipped([("manifest.json", b"{}"), *entries])
    )
    with pytest.raises(ValueError, match="Unsafe snapshot artifact entry"):
        artifacts.download(
            artifact_id=456, digest=digest, output=transfer.output
        )
    assert not transfer.output.exists()
    assert not (transfer.output.parent / "escaped").exists()
    assert not list(transfer.output.parent.glob(".docs-artifact-*"))
