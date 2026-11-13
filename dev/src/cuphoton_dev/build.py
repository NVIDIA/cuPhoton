# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Render a complete, immutable source checkout in its own environment."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from cuphoton_dev.renderers import renderer_for

FINAL_TAG = re.compile(
    r"v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\Z"
)


def run(
    args: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    capture: bool = True,
) -> str:
    """Keep progress on stderr so command stdout can be machine-readable."""
    if not capture:
        with subprocess.Popen(
            args,
            cwd=cwd,
            env=env,
            text=True,
            encoding="utf-8",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ) as process:
            assert process.stdout is not None
            for line in process.stdout:
                sys.stderr.write(line)
            code = process.wait()
        if code:
            raise RuntimeError(f"{args[0]} failed ({code}); see build output")
        return ""
    result = subprocess.run(
        args,
        cwd=cwd,
        env=env,
        text=True,
        encoding="utf-8",
        capture_output=True,
    )
    if result.returncode:
        raise RuntimeError(
            f"{args[0]} failed ({result.returncode}): "
            f"{result.stderr or 'see build output'}"
        )
    return (result.stdout or "").strip()


def git(repo: Path, *args: str) -> str:
    return run(["git", "-C", str(repo), *args])


def resolve(repo: Path, ref: str) -> str:
    return git(
        repo, "rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}"
    )


def renderer_version(workflow: str) -> str:
    """Read the target renderer pin, ignoring full-line comments."""
    active = "\n".join(
        line
        for line in workflow.splitlines()
        if not line.lstrip().startswith("#")
    )
    pin = re.search(r"releases/download/v([0-9.]+)/quarto", active)
    if not pin:
        raise ValueError("Target has no recognized pinned Quarto renderer")
    return pin[1]


def render_environment(checkout: Path) -> dict[str, str]:
    """Isolate Python, uv and Quarto from the controller's configuration."""
    env = dict(os.environ)
    for key in (
        "PYTHONPATH",
        "PYTHONHOME",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "UV_PROJECT_ENVIRONMENT",
        "UV_PROJECT",
        "UV_WORKING_DIR",
        "JUPYTER_PATH",
        "JUPYTER_CONFIG_DIR",
        "JUPYTER_DATA_DIR",
        "GH_TOKEN",
        "GITHUB_TOKEN",
        "ACTIONS_ID_TOKEN_REQUEST_TOKEN",
        "ACTIONS_ID_TOKEN_REQUEST_URL",
    ):
        env.pop(key, None)
    for key in tuple(env):
        if key.startswith(("SETUPTOOLS_SCM_", "QUARTO_")):
            env.pop(key)
    env.update(
        UV_PROJECT_ENVIRONMENT=str(checkout / ".venv"),
        CUPHOTON_XDR_BUILD_EXT="0",
        CUPHOTON_IMAGING_BUILD_EXT="0",
        CUPHOTON_XFIT_BUILD_EXT="0",
        CUPHOTON_XREP_TORCH_DEVICE="cpu",
        CUDA_VISIBLE_DEVICES="",
        MPLBACKEND="Agg",
        CUPHOTON_REPO=str(checkout),
        QUARTO_PYTHON=str(checkout / ".venv/bin/python"),
        JUPYTER_CONFIG_DIR=str(checkout / ".jupyter"),
        IPYTHONDIR=str(checkout / ".ipython"),
        JUPYTER_DATA_DIR=str(checkout / ".jupyter/data"),
        JUPYTER_PATH=str(checkout / ".venv/share/jupyter"),
        JUPYTER_PREFER_ENV_PATH="1",
        PATH=str(checkout / ".venv/bin") + os.pathsep + env.get("PATH", ""),
    )
    return env


def docs_build(
    *, ref: str, output: Path, repo: Path | None = None
) -> dict[str, Any]:
    """Build in a fresh checkout without editing the controller."""
    from cuphoton_dev.transforms import postprocess

    repo = (repo or Path.cwd()).resolve()
    output = output.resolve()
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    tag = (
        ref.removeprefix("refs/tags/")
        if ref.startswith("refs/tags/")
        else ref.removeprefix("tags/")
    )
    version = "main"
    if FINAL_TAG.fullmatch(tag):
        try:
            source_sha = resolve(repo, f"refs/tags/{tag}")
        except RuntimeError:
            source_sha = resolve(repo, ref)
        else:
            version = tag
    else:
        source_sha = resolve(repo, ref)
    controller_repo = Path(__file__).resolve().parents[3]
    controller_sha = resolve(controller_repo, "HEAD")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cuphoton-docs-") as temporary:
        checkout = Path(temporary) / "source"
        run(
            [
                "git",
                "clone",
                "--quiet",
                "--no-local",
                "--no-checkout",
                str(repo),
                str(checkout),
            ]
        )
        git(checkout, "checkout", "--quiet", "--detach", source_sha)
        guide = checkout / "docs/guide"
        if not (guide / "_quarto.yml").is_file():
            raise ValueError(
                f"Documentation is unavailable at {ref} ({source_sha})"
            )
        workflow = (checkout / ".github/workflows/docs.yml").read_text(
            encoding="utf-8"
        )
        renderer = renderer_version(workflow)
        env = render_environment(checkout)
        quarto = renderer_for(renderer, Path(temporary), env=env)
        # Allow a site-specific environment provisioning wrapper.
        uv = os.environ.get("CUPHOTON_DOCS_UV", "uv")
        run(
            [
                uv,
                "sync",
                "--locked",
                "--python",
                "3.12",
                "--group",
                "docs",
                "--extra",
                "ximage",
                "--extra",
                "catalog",
            ],
            cwd=checkout,
            env=env,
            capture=False,
        )
        run(
            [
                str(checkout / ".venv/bin/python"),
                "-m",
                "ipykernel",
                "install",
                "--prefix",
                str(checkout / ".venv"),
                "--name",
                "python3",
            ],
            cwd=checkout,
            env=env,
            capture=False,
        )
        run(
            [
                str(quarto),
                "render",
                "docs/guide",
                "--execute",
                "--no-cache",
            ],
            cwd=checkout,
            env=env,
            capture=False,
        )
        git(checkout, "diff", "--exit-code", "HEAD")
        rendered = guide / "_book"
        receipt = postprocess(
            rendered,
            version=version,
            source_sha=source_sha,
            controller_sha=controller_sha,
            renderer=renderer,
            lock_sha256=hashlib.sha256(
                (checkout / "uv.lock").read_bytes()
            ).hexdigest(),
        )
        # Output appears only after rendering and all snapshot checks pass.
        with tempfile.TemporaryDirectory(
            prefix=".docs-output-", dir=output.parent
        ) as stage:
            staged = Path(stage) / "snapshot"
            shutil.copytree(rendered, staged)
            staged.rename(output)
    return {"output": str(output), **receipt}
