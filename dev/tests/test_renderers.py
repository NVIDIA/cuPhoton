# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import io
import tarfile

import pytest

from cuphoton_dev import renderers


def renderer_archive(version="1.9.36", *, unsafe=None):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as package:
        if unsafe:
            member = tarfile.TarInfo(
                "../escaped" if unsafe == "path" else "escape"
            )
            if unsafe == "symlink":
                member.type = tarfile.SYMTYPE
                member.linkname = "../escaped"
            package.addfile(member)
        binary = (
            '#!/bin/sh\nprintf "%s\\n" '
            f'"${{QUARTO_FORCE_VERSION:-{version}}}"\n'
        ).encode()
        member = tarfile.TarInfo("quarto-1.9.36/bin/quarto")
        member.size = len(binary)
        member.mode = 0o755
        package.addfile(member, io.BytesIO(binary))
    return buffer.getvalue()


def serve_renderer(monkeypatch, payload, *, checksum_matches=True):
    monkeypatch.setattr(renderers.platform, "system", lambda: "Linux")
    monkeypatch.setattr(renderers.platform, "machine", lambda: "x86_64")
    expected = (
        hashlib.sha256(payload).hexdigest() if checksum_matches else "0" * 64
    )
    monkeypatch.setattr(
        renderers, "RENDERERS", {"1.9.36": {"amd64": expected}}
    )
    urls = []

    def download(url, **kwargs):
        urls.append(url)
        return io.BytesIO(payload)

    monkeypatch.setattr(renderers, "urlopen", download)
    return urls


def test_matching_installed_renderer_does_not_use_network(
    tmp_path, monkeypatch
):
    binary = tmp_path / "quarto"
    binary.write_text(
        '#!/bin/sh\nprintf "%s\\n" "${QUARTO_FORCE_VERSION:-1.9.36}"\n'
    )
    binary.chmod(0o755)
    monkeypatch.setenv("QUARTO_FORCE_VERSION", "2.0.0")
    monkeypatch.setattr(
        renderers, "urlopen", lambda *a, **k: pytest.fail("must not download")
    )
    scratch = tmp_path / "unused"
    env = {"PATH": str(tmp_path)}
    assert renderers.renderer_for("1.9.36", scratch, env=env) == binary
    assert not scratch.exists()


def test_unknown_renderer_requires_controller_update(tmp_path, monkeypatch):
    monkeypatch.setattr(
        renderers.shutil,
        "which",
        lambda *a, **k: pytest.fail("unsupported pin"),
    )
    with pytest.raises(ValueError, match="Unsupported Quarto 99.0.0"):
        renderers.renderer_for("99.0.0", tmp_path, env={})


@pytest.mark.parametrize("checksum_matches", [True, False])
def test_mismatch_downloads_and_verifies_target(
    tmp_path, monkeypatch, checksum_matches
):
    installed = tmp_path / "quarto"
    installed.write_text(
        '#!/bin/sh\nprintf "%s\\n" "${QUARTO_FORCE_VERSION:-2.0.0}"\n'
    )
    installed.chmod(0o755)
    # Ambient configuration must not spoof either the host or extracted probe.
    monkeypatch.setenv("QUARTO_FORCE_VERSION", "1.9.36")
    urls = serve_renderer(
        monkeypatch, renderer_archive(), checksum_matches=checksum_matches
    )
    env = {"PATH": str(tmp_path)}
    scratch = tmp_path / "render"
    if checksum_matches:
        executable = renderers.renderer_for("1.9.36", scratch, env=env)
        assert executable.is_relative_to(scratch)
        assert renderers._version(executable, env=env) == "1.9.36"
        assert renderers._version(installed, env=env) == "2.0.0"
    else:
        with pytest.raises(ValueError, match="checksum mismatch"):
            renderers.renderer_for("1.9.36", scratch, env=env)
        assert not (scratch / "quarto-1.9.36-amd64/quarto-1.9.36").exists()
    assert urls == [
        "https://github.com/quarto-dev/quarto-cli/releases/download/"
        "v1.9.36/quarto-1.9.36-linux-amd64.tar.gz"
    ]


@pytest.mark.parametrize("unsafe", ["path", "symlink"])
def test_verified_archive_cannot_escape_scratch(
    tmp_path, monkeypatch, unsafe
):
    serve_renderer(monkeypatch, renderer_archive(unsafe=unsafe))
    scratch = tmp_path / "render"
    with pytest.raises(tarfile.FilterError):
        renderers.renderer_for("1.9.36", scratch, env={"PATH": ""})
    assert not (scratch / "escaped").exists()
    assert not (scratch / "quarto-1.9.36-amd64/escape").is_symlink()


def test_verified_archive_must_report_pinned_version(tmp_path, monkeypatch):
    serve_renderer(monkeypatch, renderer_archive("2.0.0"))
    # A trusted digest alone does not prove the executable matches the pin.
    monkeypatch.setenv("QUARTO_FORCE_VERSION", "1.9.36")
    with pytest.raises(
        ValueError, match="Extracted Quarto does not report version 1.9.36"
    ):
        renderers.renderer_for(
            "1.9.36", tmp_path / "render", env={"PATH": ""}
        )
