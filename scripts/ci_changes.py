# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Select reduced CI for confirmed documentation-only pull requests."""

import json
import os
import re
import subprocess
from pathlib import Path, PurePosixPath


def is_documentation(path: str) -> bool:
    if path in {
        "README.md",
        "CHANGELOG.md",
        "CITATION.md",
        "CODE_OF_CONDUCT.md",
        "CONTRIBUTING.md",
        "GOVERNANCE.md",
        "MAINTAINERS.md",
        "SECURITY.md",
        "SUPPORT.md",
        "docs/guide/_quarto.yml",
        "docs/guide/_environment",
        "docs/guide/.gitignore",
    }:
        return True
    return path.startswith("docs/") and PurePosixPath(path).suffix in {
        ".md",
        ".qmd",
        ".scss",
        ".css",
        ".png",
        ".jpg",
        ".jpeg",
        ".svg",
        ".webp",
        ".gif",
        ".ico",
        ".bib",
    }


def changed_paths(base: str, head: str) -> list[str]:
    # Disable rename detection so moving code into docs still counts as code.
    output = subprocess.check_output(
        [
            "git",
            "diff",
            "--name-only",
            "--no-renames",
            "-z",
            f"{base}...{head}",
            "--",
        ],
        stderr=subprocess.PIPE,
        timeout=30,
    )
    return [os.fsdecode(path) for path in output.split(b"\0") if path]


def docs_only(
    event_name: str, ref: str, sha: str, repository: str, event: dict
) -> bool:
    try:
        if event_name == "pull_request":
            pr = event["pull_request"]
        elif event_name == "push" and (
            match := re.fullmatch(r"refs/heads/pull-request/([0-9]+)", ref)
        ):
            pr = json.loads(
                subprocess.check_output(
                    ["gh", "api", f"repos/{repository}/pulls/{match[1]}"],
                    text=True,
                    stderr=subprocess.PIPE,
                    timeout=30,
                )
            )
            if pr["head"]["sha"] != sha:
                return False
        else:
            # Maintained branches and manual qualification keep full CI.
            return False

        base, head = pr["base"]["sha"], pr["head"]["sha"]
        if not all(
            isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value)
            for value in (base, head)
        ):
            return False
        # Compare the whole PR, including code changed before a docs update.
        paths = changed_paths(base, head)
        return bool(paths) and all(is_documentation(path) for path in paths)
    except (
        KeyError,
        TypeError,
        ValueError,
        OSError,
        subprocess.SubprocessError,
    ):
        # Missing metadata or history must never suppress qualification.
        print("Could not classify the complete PR diff; using full CI.")
        return False


def main() -> None:
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    reduced = docs_only(
        os.environ["GITHUB_EVENT_NAME"],
        os.environ["GITHUB_REF"],
        os.environ["GITHUB_SHA"],
        os.environ["GITHUB_REPOSITORY"],
        event,
    )
    profile = "documentation only" if reduced else "full"
    print(f"CI profile: {profile}")
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
        output.write(f"docs-only={str(reduced).lower()}\n")
    with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as summary:
        summary.write(f"CI profile: **{profile}**.\n")


if __name__ == "__main__":
    main()
