# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Developer commands using the product's command and invariant framework."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO

from cuphoton.core.cli import (
    BoolInvariant,
    ComponentSpec,
    InvariantAwareCommand,
    PathValueInvariant,
    StringInvariant,
    run_component,
)

DEVELOPER_COMPONENT = ComponentSpec(
    "dev",
    "cuphoton_dev",
    "Repository development and documentation tools.",
    commands_module="cuphoton_dev.cli",
    parser_style="argparse",
    supports_version_command=False,
)


class _DocsCommand(InvariantAwareCommand):
    _log_level_ = False
    _quiet_ = False
    _verbose_ = False
    _shortname_ = None
    json = False

    class JsonArg(BoolInvariant):
        _arg = "--json"
        _help = "Write a machine-readable JSON result."

    def emit(self, result: Mapping[str, Any]) -> None:
        if self.json:
            self._out(json.dumps(result, sort_keys=True))
            return
        for key, value in result.items():
            if isinstance(value, (dict, list)):
                value = json.dumps(value, sort_keys=True)
            self._out(f"{key}: {value}")


class DocsBuildCommand(_DocsCommand):
    _description_ = "Render and validate documentation from an exact source."
    ref: str
    output_dir: Path
    repo = Path(".")

    class RefArg(StringInvariant):
        _arg = "--ref"
        _required = True
        _help = "Main revision or full release tag to render."

    class OutputDirArg(PathValueInvariant):
        _arg = "--output"
        _required = True
        _help = "Destination for the validated snapshot."

    class RepoArg(PathValueInvariant):
        _arg = "--repo"
        _help = "Source Git checkout (default: current directory)."

    def run(self) -> None:
        from .publisher import docs_build

        self.emit(
            docs_build(ref=self.ref, output=self.output_dir, repo=self.repo)
        )


class DocsCheckCommand(_DocsCommand):
    _description_ = "Validate a rendered snapshot or complete site archive."
    directory: Path

    class DirectoryArg(PathValueInvariant):
        _required = True
        _metavar = "DIR"

    def run(self) -> None:
        from .publisher import docs_check

        self.emit(docs_check(self.directory))


class _RemoteDocsCommand(_DocsCommand):
    repository = "NVIDIA/cuPhoton"

    class RepositoryArg(StringInvariant):
        _arg = "--repository"
        _help = "GitHub repository (default: NVIDIA/cuPhoton)."


class DocsStatusCommand(_RemoteDocsCommand):
    _description_ = "List published, missing, and unavailable documentation."

    def run(self) -> None:
        from .publisher import docs_status

        self.emit(docs_status(repository=self.repository))


class DocsPublishCommand(_RemoteDocsCommand):
    _description_ = "Dispatch publication on main and track its workflow run."
    ref: str | None = None
    missing = False
    no_wait = False

    class RefArg(StringInvariant):
        _arg = "--ref"
        _help = "Publish main or an eligible full release tag."

    class MissingArg(BoolInvariant):
        _arg = "--missing"
        _help = "Backfill all eligible releases without documentation."

    class NoWaitArg(BoolInvariant):
        _arg = "--no-wait"
        _help = (
            "Return the tracked workflow run without waiting for completion."
        )

    def run(self) -> None:
        from .publisher import docs_publish

        if (self.ref is None) == (not self.missing):
            raise ValueError("supply exactly one of --ref or --missing")
        self.emit(
            docs_publish(
                ref=self.ref,
                missing=self.missing,
                wait=not self.no_wait,
                repository=self.repository,
            )
        )


def main(
    argv: Sequence[str] | None = None,
    *,
    out: TextIO | None = None,
    err: TextIO | None = None,
    environ: Mapping[str, str] | None = None,
    program_name: str = "cuphoton dev",
) -> int:
    """Run the explicitly installed developer group."""

    return run_component(
        DEVELOPER_COMPONENT,
        argv,
        out=out,
        err=err,
        environ=environ,
        program_name=program_name,
    )
