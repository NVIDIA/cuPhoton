# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import io
import tarfile

import pytest

from cuphoton_dev import renderers


def test_matching_installed_renderer_does_not_use_network(
    tmp_path, monkeypatch
):
    binary = tmp_path / "quarto"
    binary.write_text("#!/bin/sh\nprintf '1.9.36\\n'\n")
    binary.chmod(0o755)
    monkeypatch.setattr(renderers.shutil, "which", lambda name: str(binary))
    monkeypatch.setattr(
        renderers, "urlopen", lambda *a, **k: pytest.fail("must not download")
    )
    scratch = tmp_path / "unused"
    assert renderers.renderer_for("1.9.36", scratch) == binary
    assert not scratch.exists()


def test_unknown_renderer_requires_controller_update(tmp_path, monkeypatch):
    monkeypatch.setattr(
        renderers.shutil, "which", lambda name: pytest.fail("unsupported pin")
    )
    with pytest.raises(ValueError, match="Unsupported Quarto 99.0.0"):
        renderers.renderer_for("99.0.0", tmp_path)


@pytest.mark.parametrize("checksum_matches", [True, False])
def test_mismatch_downloads_and_verifies_target(
    tmp_path, monkeypatch, checksum_matches
):
    installed = tmp_path / "quarto-newer"
    installed.write_text("#!/bin/sh\nprintf '2.0.0\\n'\n")
    installed.chmod(0o755)
    monkeypatch.setattr(
        renderers.shutil, "which", lambda name: str(installed)
    )
    monkeypatch.setattr(renderers.platform, "system", lambda: "Linux")
    monkeypatch.setattr(renderers.platform, "machine", lambda: "x86_64")
    buffer = io.BytesIO()
    binary = b"#!/bin/sh\nprintf '1.9.36\\n'\n"
    with tarfile.open(fileobj=buffer, mode="w:gz") as package:
        member = tarfile.TarInfo("quarto-1.9.36/bin/quarto")
        member.size = len(binary)
        member.mode = 0o755
        package.addfile(member, io.BytesIO(binary))
    payload = buffer.getvalue()
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
    scratch = tmp_path / "render"
    if checksum_matches:
        executable = renderers.renderer_for("1.9.36", scratch)
        assert executable.is_relative_to(scratch)
        assert renderers._version(executable) == "1.9.36"
        assert installed.read_text() == "#!/bin/sh\nprintf '2.0.0\\n'\n"
    else:
        with pytest.raises(ValueError, match="checksum mismatch"):
            renderers.renderer_for("1.9.36", scratch)
        assert not (scratch / "quarto-1.9.36-amd64/quarto-1.9.36").exists()
    assert urls == [
        "https://github.com/quarto-dev/quarto-cli/releases/download/"
        "v1.9.36/quarto-1.9.36-linux-amd64.tar.gz"
    ]
