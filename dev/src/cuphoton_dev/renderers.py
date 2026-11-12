# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Select a target renderer without changing the host installation."""

from __future__ import annotations

import hashlib
import platform
import shutil
import subprocess
import tarfile
from pathlib import Path
from urllib.request import urlopen

# Official quarto-dev/quarto-cli release asset digests, also published in
# quarto-<version>-checksums.txt. Retain old entries when upgrading main.
RENDERERS = {
    "1.9.36": {
        "amd64": (
            "39d2e4546274be39a27cceeeed21a60b693c4c477a8c8cb43fb34326442f3910"
        ),
        "arm64": (
            "6f4b2fee6be469ae038a23fe3196010f29d7716129eed2891524c1f88ad82f2f"
        ),
    },
}


def _version(executable: Path, *, env: dict[str, str]) -> str | None:
    try:
        result = subprocess.run(
            [str(executable), "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def renderer_for(version: str, scratch: Path, *, env: dict[str, str]) -> Path:
    """Reuse a matching renderer or extract a verified one beneath scratch.

    The caller owns scratch and its lifetime. No global installation or
    credentials are needed. Unknown versions require a reviewed table update.
    """
    if version not in RENDERERS:
        raise ValueError(
            f"Unsupported Quarto {version}; add its verified release hashes "
            "to the main controller's renderer table"
        )
    installed = shutil.which("quarto", path=env.get("PATH", ""))
    if installed and _version(Path(installed), env=env) == version:
        return Path(installed).resolve()
    machine = platform.machine().lower()
    architecture = {
        "x86_64": "amd64",
        "amd64": "amd64",
        "aarch64": "arm64",
        "arm64": "arm64",
    }.get(machine)
    if platform.system() != "Linux" or architecture is None:
        raise ValueError(
            "Automatic Quarto provisioning supports Linux amd64 and arm64; "
            f"install the supported Quarto {version} on this host"
        )
    destination = scratch / f"quarto-{version}-{architecture}"
    destination.mkdir(parents=True, exist_ok=False)
    archive = destination / "renderer.tar.gz"
    asset = f"quarto-{version}-linux-{architecture}.tar.gz"
    url = (
        "https://github.com/quarto-dev/quarto-cli/releases/download/"
        f"v{version}/{asset}"
    )
    with urlopen(url, timeout=60) as response, archive.open("wb") as stream:
        shutil.copyfileobj(response, stream)
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != RENDERERS[version][architecture]:
        raise ValueError(f"Quarto {version} download checksum mismatch")
    with tarfile.open(archive, "r:gz") as package:
        package.extractall(destination, filter="data")
    executable = destination / f"quarto-{version}" / "bin/quarto"
    if _version(executable, env=env) != version:
        raise ValueError(
            f"Extracted Quarto does not report version {version}"
        )
    return executable.resolve()
