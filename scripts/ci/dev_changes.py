# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Classify developer-tool dependencies without installing developer tools."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

DEPENDENCIES = (
    "dev/",
    "src/cuphoton/core/",
    "tests/core/",
    "docs/guide/",
    "examples/",
    ".github/workflows/",
    "scripts/ci/",
    "scripts/conda/",
    "scripts/wheels/",
    "conda/",
)
METADATA = {
    "pyproject.toml",
    "uv.lock",
    "setup.py",
    "setup.cfg",
    "MANIFEST.in",
    "Makefile",
    ".gitattributes",
    ".gitignore",
    ".pre-commit-config.yaml",
    ".coveragerc",
    "pytest.ini",
    "tox.ini",
    "mypy.ini",
    "ruff.toml",
    ".ruff.toml",
    "tests/conftest.py",
}


def affects_dev(path: str) -> bool:
    """Unknown paths run the tests; only established unrelated paths skip."""
    if path in METADATA or path.startswith(DEPENDENCIES):
        return True
    if path.startswith("src/cuphoton/"):
        # Package entrypoints can change the lazy developer CLI route.
        return "/" not in path.removeprefix("src/cuphoton/")
    if path.startswith("tests/"):
        return "/" not in path.removeprefix("tests/")
    return not (
        path.startswith(("docs/", "licenses/", ".github/ISSUE_TEMPLATE/"))
        or path
        in {
            "README.md",
            "README-pypi.md",
            "CONTRIBUTING.md",
            "CODE_OF_CONDUCT.md",
            "SECURITY.md",
            "LICENSE",
            "THIRD_PARTY_NOTICES.md",
            ".github/PULL_REQUEST_TEMPLATE.md",
        }
    )


def command(*args: str) -> str:
    return subprocess.check_output(args, text=True, stderr=subprocess.PIPE)


def ensure_commit(sha: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Comparison needs a complete commit SHA")
    try:
        command("git", "cat-file", "-e", f"{sha}^{{commit}}")
    except subprocess.CalledProcessError:
        command("git", "fetch", "--no-tags", "origin", sha)


def comparison(
    event: dict[str, Any], event_name: str, ref: str, repository: str
) -> tuple[str, str, bool]:
    """Return base, head and whether to compare from their merge base."""
    if event_name == "pull_request":
        pr = event["pull_request"]
        return pr["base"]["sha"], pr["head"]["sha"], True
    if event_name != "push":
        raise ValueError(f"Unsupported event: {event_name}")
    copied = re.fullmatch(r"refs/heads/pull-request/(\d+)", ref)
    if copied:
        if repository != "NVIDIA/cuPhoton":
            raise ValueError("Unrecognized repository for a vetted PR copy")
        pr = json.loads(
            command("gh", "api", f"repos/{repository}/pulls/{copied[1]}")
        )
        if pr["head"]["sha"] != event["after"]:
            raise ValueError("Vetted copy no longer matches the PR head")
        return pr["base"]["sha"], event["after"], True
    if event["before"] == "0" * 40 or event["after"] == "0" * 40:
        raise ValueError("Branch creation or deletion has no complete range")
    return event["before"], event["after"], False


def changed_paths(base: str, head: str, merge_base: bool) -> list[str]:
    ensure_commit(base)
    ensure_commit(head)
    if merge_base:
        base = command("git", "merge-base", base, head).strip()
    # Disabling rename detection lists both old and new names. A dependency
    # moved out of its directory still invalidates developer tests.
    output = command(
        "git", "diff", "--name-only", "--no-renames", "-z", base, head
    )
    return [path for path in output.split("\0") if path]


def classify(
    event: dict[str, Any], event_name: str, ref: str, repository: str
) -> dict[str, Any]:
    try:
        base, head, merge_base = comparison(
            event, event_name, ref, repository
        )
        paths = changed_paths(base, head, merge_base)
    except (
        KeyError,
        ValueError,
        OSError,
        subprocess.CalledProcessError,
    ) as exc:
        return {
            "dev": True,
            "reason": f"Uncertain comparison: {type(exc).__name__}",
        }
    relevant = [path for path in paths if affects_dev(path)]
    return {
        "dev": bool(relevant),
        "base": base,
        "head": head,
        "merge_base": merge_base,
        "paths": relevant,
        "reason": "Dependency changed"
        if relevant
        else "No dependency changed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--event", type=Path, default=os.getenv("GITHUB_EVENT_PATH")
    )
    args = parser.parse_args()
    event = json.loads(args.event.read_text())
    result = classify(
        event,
        os.environ["GITHUB_EVENT_NAME"],
        os.environ["GITHUB_REF"],
        os.environ["GITHUB_REPOSITORY"],
    )
    print(json.dumps(result, sort_keys=True))
    if output := os.getenv("GITHUB_OUTPUT"):
        with Path(output).open("a") as stream:
            stream.write(f"dev={str(result['dev']).lower()}\n")


if __name__ == "__main__":
    main()
