# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared CLI commands and explicit inference configuration."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from cuphoton.core.cli import (
    BoolInvariant,
    CommandError,
    IntegerInvariant,
    InvariantAwareCommand,
    PathValueInvariant,
    SequenceInvariant,
    StringInvariant,
)

from .dashboard import Dashboard, serve
from .inference import InferenceConfig, Investigator
from .monitor import Monitor
from .quickstart import quickstart
from .session import MonitorSession
from .site import normalize_prefix, serve_site


class _XMonitorCommand(InvariantAwareCommand):
    _log_level_ = False
    _quiet_ = False
    _verbose_ = False
    _shortname_ = None

    def run(self) -> None:
        try:
            self.execute()
        except (ValueError, OSError, KeyError, TypeError) as exc:
            raise CommandError(str(exc)) from exc

    def execute(self) -> None:
        raise NotImplementedError


class QuickstartCommand(_XMonitorCommand):
    _description_ = "Generate invented measurements and an offline dashboard."
    output_dir: Path

    class OutputDirArg(PathValueInvariant):
        _arg = "--output-dir"
        _help = "Fresh run output directory."
        _required = True

    def execute(self) -> None:
        quickstart(self.output_dir)


class _StateCommand(_XMonitorCommand):
    state: Path

    class StateArg(PathValueInvariant):
        _arg = "--state"
        _help = "Existing xMonitor state directory."
        _required = True


class ExportCommand(_StateCommand):
    _description_ = "Export saved measurements and reports as offline HTML."
    output_file: Path

    class OutputFileArg(PathValueInvariant):
        _arg = "--output-file"
        _help = "Destination HTML file."
        _required = True

    def execute(self) -> None:
        Dashboard(Monitor.load(self.state, recover=False)).export(
            self.output_file
        )


class ServeCommand(_XMonitorCommand):
    _description_ = "Serve a saved monitor or installed application site."
    state: Path | None = None
    site: Path | None = None
    prefix: str | None = None
    host = "127.0.0.1"
    port = 8765
    inference_config: Path | None = None
    auto_investigate = False
    access_token_env: str | None = None

    class StateArg(PathValueInvariant):
        _arg = "--state"
        _help = "Existing xMonitor state directory; use --state or --site."

    class SiteArg(PathValueInvariant):
        _arg = "--site"
        _help = "Application directory or its trusted xmonitor-site.toml."

    class PrefixArg(StringInvariant):
        _arg = "--prefix"
        _help = "URL mount path; defaults to the site's prefix or /."

    class InferenceConfigArg(PathValueInvariant):
        _arg = "--inference-config"
        _help = "Optional endpoint/backend TOML for browser investigation."

    class AutoInvestigateArg(BoolInvariant):
        _arg = "--auto-investigate"
        _help = "Automatically investigate new findings in this session."
        _default = False

    class AccessTokenEnvArg(StringInvariant):
        _arg = "--access-token-env"
        _help = "Environment reference for remote investigation access."

    class HostArg(StringInvariant):
        _arg = "--host"
        _help = "Listen address; shared deployments supply access control."
        _default = "127.0.0.1"

    class PortArg(IntegerInvariant):
        _arg = "--port"
        _help = "Listen port."
        _default = 8765
        _min = 0
        _max = 65535

    def execute(self) -> None:
        if (self.state is None) == (self.site is None):
            raise ValueError("supply exactly one of --state or --site")
        if self.site is not None:
            if (
                self.inference_config
                or self.auto_investigate
                or self.access_token_env
            ):
                raise ValueError(
                    "site applications configure their own investigation; "
                    "inference flags require --state"
                )
            serve_site(
                self.site, host=self.host, port=self.port, prefix=self.prefix
            )
            return
        assert self.state is not None
        prefix = normalize_prefix(self.prefix or "/")
        config = (
            InferenceConfig.from_toml(self.inference_config)
            if self.inference_config
            else None
        )
        monitor = Monitor.load(self.state, recover=config is not None)
        session = (
            MonitorSession(
                monitor,
                config=config,
                auto_investigate=self.auto_investigate,
                access_token_env=self.access_token_env,
            )
            if config
            else None
        )
        if self.auto_investigate and config is None:
            raise ValueError(
                "automatic investigation requires inference configuration"
            )
        serve(
            Dashboard(monitor),
            host=self.host,
            port=self.port,
            session=session,
            prefix=prefix,
        )


class AskCommand(_StateCommand):
    _description_ = (
        "Investigate selected scalar evidence with an explicit endpoint."
    )
    config: Path
    question: str
    sample: list[str] = []

    class ConfigArg(PathValueInvariant):
        _arg = "--config"
        _help = "TOML file containing an explicit [inference] table."
        _required = True

    class QuestionArg(StringInvariant):
        _arg = "--question"
        _help = "Question about the selected evidence."
        _required = True

    class SampleArg(SequenceInvariant):
        _arg = "--sample"
        _help = "Sample ID to include; repeat for each selected sample."
        _required = True
        _item_type = str

    def execute(self) -> None:
        report = asyncio.run(
            Investigator(InferenceConfig.from_toml(self.config)).investigate(
                Monitor.load(self.state),
                self.sample,
                question=self.question,
            )
        )
        print(json.dumps(report, indent=2, allow_nan=False))
        if report["status"] != "completed":
            raise CommandError(
                "investigation did not complete; see saved report status"
            )
