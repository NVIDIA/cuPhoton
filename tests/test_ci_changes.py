# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Exercise CI selection against complete pull-request histories."""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import ci_changes


class CIChangesTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.git("init", "--initial-branch=main")
        self.git("config", "user.name", "CI Tests")
        self.git("config", "user.email", "tests@example.org")
        self.git("config", "commit.gpgsign", "false")
        self.git("config", "core.hooksPath", "/dev/null")
        self.write("README.md", "Initial documentation\n")
        self.write("src/example.py", "VALUE = 1\n")
        self.base = self.commit()

    def git(self, *args):
        return subprocess.check_output(
            ["git", *args], cwd=self.root, stderr=subprocess.PIPE, text=True
        ).strip()

    def write(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def commit(self):
        self.git("add", "--all")
        self.git("commit", "-m", "Test change")
        return self.git("rev-parse", "HEAD")

    def metadata(self, head, base=None):
        return {
            "base": {"sha": base or self.base},
            "head": {"sha": head},
        }

    def select(self, head, *, event_name="pull_request", ref="", event=None):
        if event is None:
            event = {"pull_request": self.metadata(head)}
        with contextlib.chdir(self.root):
            return ci_changes.docs_only(
                event_name, ref, head, "NVIDIA/cuPhoton", event
            )

    def test_documentation_and_guide_assets_use_reduced_ci(self):
        for name in (
            "README.md",
            "docs/guide/first-difference.qmd",
            "docs/guide/_quarto.yml",
            "docs/guide/assets/logo.png",
            "docs/a name\nwith whitespace.md",
        ):
            self.write(name, "Updated documentation\n")
        self.assertTrue(self.select(self.commit()))

    def test_unknown_and_package_inputs_require_full_ci(self):
        for name in (
            "src/readme.md",
            "docs/build.py",
            "docs/config.yml",
            "docs/requirements.txt",
            "docs/guide/_extensions/filter.lua",
            "pyproject.toml",
            "setup.py",
            "MANIFEST.in",
            "Makefile",
            "LICENSE",
            "THIRD_PARTY_NOTICES.md",
            "CITATION.cff",
            "uv.lock",
            ".github/workflows/ci.yml",
            "examples/example.ipynb",
        ):
            with self.subTest(path=name):
                self.assertFalse(ci_changes.is_documentation(name))
        self.write("README.md", "Updated documentation\n")
        self.write("new-build-config", "Unknown input\n")
        self.assertFalse(self.select(self.commit()))

    def test_docs_followup_does_not_hide_earlier_code_change(self):
        self.write("src/example.py", "VALUE = 2\n")
        self.commit()
        self.write("README.md", "Updated documentation\n")
        self.assertFalse(self.select(self.commit()))

    def test_renaming_source_into_docs_requires_full_ci(self):
        self.git("mv", "src/example.py", "CONTRIBUTING.md")
        self.assertFalse(self.select(self.commit()))

    def test_deleted_documentation_uses_reduced_ci(self):
        (self.root / "README.md").unlink()
        self.assertTrue(self.select(self.commit()))

    def test_empty_or_unavailable_diff_requires_full_ci(self):
        self.assertFalse(self.select(self.base))
        self.assertFalse(self.select("0" * 40))

    def test_maintained_pushes_and_manual_runs_keep_full_ci(self):
        self.write("README.md", "Updated documentation\n")
        head = self.commit()
        for ref in ("main", "0.1.x", "0.2.x"):
            with self.subTest(ref=ref):
                self.assertFalse(
                    self.select(
                        head, event_name="push", ref=f"refs/heads/{ref}"
                    )
                )
        self.assertFalse(self.select(head, event_name="workflow_dispatch"))

    def test_vetted_copy_uses_verified_pr_base_and_head(self):
        self.write("src/example.py", "VALUE = 2\n")
        release_base = self.commit()
        self.write("README.md", "Updated documentation\n")
        head = self.commit()
        check_output = subprocess.check_output

        def metadata_or_git(args, **kwargs):
            if args[0] == "gh":
                self.assertEqual(args[2], "repos/NVIDIA/cuPhoton/pulls/12")
                return json.dumps(self.metadata(head, release_base))
            return check_output(args, **kwargs)

        with mock.patch.object(
            ci_changes.subprocess, "check_output", side_effect=metadata_or_git
        ):
            self.assertTrue(
                self.select(
                    head, event_name="push", ref="refs/heads/pull-request/12"
                )
            )

    def test_stale_vetted_copy_requires_full_ci(self):
        self.write("src/example.py", "VALUE = 2\n")
        copied_head = self.commit()
        self.git("checkout", "-b", "replacement", self.base)
        self.write("README.md", "Updated documentation\n")
        current_head = self.commit()
        check_output = subprocess.check_output

        def metadata_or_git(args, **kwargs):
            if args[0] == "gh":
                return json.dumps(self.metadata(current_head))
            return check_output(args, **kwargs)

        with mock.patch.object(
            ci_changes.subprocess, "check_output", side_effect=metadata_or_git
        ):
            self.assertFalse(
                self.select(
                    copied_head,
                    event_name="push",
                    ref="refs/heads/pull-request/12",
                )
            )

    def test_unavailable_pr_metadata_requires_full_ci(self):
        self.write("README.md", "Updated documentation\n")
        head = self.commit()
        responses = (
            "{}",
            "invalid JSON",
            subprocess.CalledProcessError(1, ["gh", "api"]),
        )
        for response in responses:
            with self.subTest(response=response):
                kwargs = (
                    {"side_effect": response}
                    if isinstance(response, Exception)
                    else {"return_value": response}
                )
                with mock.patch.object(
                    ci_changes.subprocess, "check_output", **kwargs
                ):
                    self.assertFalse(
                        self.select(
                            head,
                            event_name="push",
                            ref="refs/heads/pull-request/12",
                        )
                    )

    def test_main_writes_selected_profile(self):
        for name, expected in (
            ("README.md", "true"),
            ("src/example.py", "false"),
        ):
            with self.subTest(path=name):
                self.write(name, "Updated content\n")
                head = self.commit()
                event_path = self.root / ".git" / "event.json"
                event_path.write_text(
                    json.dumps({"pull_request": self.metadata(head)})
                )
                output = self.root / ".git" / "output"
                output.write_text("")
                env = {
                    "GITHUB_EVENT_PATH": str(event_path),
                    "GITHUB_EVENT_NAME": "pull_request",
                    "GITHUB_REF": "refs/pull/12/merge",
                    "GITHUB_SHA": head,
                    "GITHUB_REPOSITORY": "NVIDIA/cuPhoton",
                    "GITHUB_OUTPUT": str(output),
                    "GITHUB_STEP_SUMMARY": str(
                        self.root / ".git" / "summary"
                    ),
                }
                with (
                    mock.patch.dict(os.environ, env),
                    contextlib.chdir(self.root),
                ):
                    ci_changes.main()
                self.assertEqual(
                    output.read_text(), f"docs-only={expected}\n"
                )


if __name__ == "__main__":
    unittest.main()
