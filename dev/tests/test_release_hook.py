# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

HOOK = (
    Path(__file__).resolve().parents[2] / ".github/workflows/release-docs.yml"
)


@pytest.mark.parametrize(
    ("tag", "dispatch"),
    [
        ("v0.2.0", True),
        ("v1.20.301", True),
        ("v0.2.0rc0", False),
        ("v0.2.0-rc.1", False),
        ("v01.2.0", False),
        ("0.2.x", False),
        ("v0.2.0; touch injected", False),
    ],
)
def test_hook_dispatches_only_final_tags_on_main(tmp_path, tag, dispatch):
    workflow = yaml.safe_load(HOOK.read_text())
    script = workflow["jobs"]["forward"]["steps"][0]["run"]
    # Execute the actual hook with a recording CLI, without network access.
    gh = tmp_path / "gh"
    gh.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, pathlib, sys\n"
        "pathlib.Path(os.environ['CALL_FILE']).write_text("
        "json.dumps(sys.argv[1:]))\n"
    )
    gh.chmod(0o755)
    call = tmp_path / "call.json"
    env = {
        **os.environ,
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        "CALL_FILE": str(call),
        "GITHUB_REPOSITORY": "NVIDIA/cuPhoton",
        "RELEASE_TAG": tag,
    }
    subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", script], env=env, check=True
    )
    assert call.exists() == dispatch
    if dispatch:
        assert json.loads(call.read_text()) == [
            "workflow",
            "run",
            "docs.yml",
            "--repo",
            "NVIDIA/cuPhoton",
            "--ref",
            "main",
            "--field",
            f"tag={tag}",
        ]
    assert not (tmp_path / "injected").exists()
