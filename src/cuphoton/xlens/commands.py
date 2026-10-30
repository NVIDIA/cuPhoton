# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared CLI commands and explicit inference configuration."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory

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
from .inference import InferenceConfig, InferenceError, Investigator
from .monitor import Monitor
from .nooa_backend import NooaError
from .quickstart import quickstart
from .records import encode
from .session import MonitorSession
from .site import normalize_prefix, serve_site


class _XLensCommand(InvariantAwareCommand):
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


class QuickstartCommand(_XLensCommand):
    _description_ = "Generate invented measurements and an offline dashboard."
    output_dir: Path

    class OutputDirArg(PathValueInvariant):
        _arg = "--output-dir"
        _help = "Fresh run output directory."
        _required = True

    def execute(self) -> None:
        quickstart(self.output_dir)


class _StateCommand(_XLensCommand):
    state: Path

    class StateArg(PathValueInvariant):
        _arg = "--state"
        _help = "Existing xLens state directory."
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


class ServeCommand(_XLensCommand):
    _description_ = "Serve a saved monitor or installed application site."
    state: Path | None = None
    site: Path | None = None
    prefix: str | None = None
    host = "127.0.0.1"
    port = 8765
    inference_config: Path | None = None
    auto_investigate = False
    access_token_env: str | None = None
    allowed_host: list[str] = []

    class StateArg(PathValueInvariant):
        _arg = "--state"
        _help = "Existing xLens state directory; use --state or --site."

    class SiteArg(PathValueInvariant):
        _arg = "--site"
        _help = "Application directory or its trusted xlens-site.toml."

    class PrefixArg(StringInvariant):
        _arg = "--prefix"
        _help = "URL mount path; defaults to the site's prefix or /."

    class InferenceConfigArg(PathValueInvariant):
        _arg = "--inference-config"
        _help = "Optional endpoint/backend TOML for browser investigation."

    class AutoInvestigateArg(BoolInvariant):
        _arg = "--auto-investigate"
        _help = (
            "Investigate current warning/error sample revisions without an "
            "automatic report, including saved samples."
        )
        _default = False

    class AccessTokenEnvArg(StringInvariant):
        _arg = "--access-token-env"
        _help = "Environment reference for remote investigation access."

    class AllowedHostArg(SequenceInvariant):
        _arg = "--allowed-host"
        _help = (
            "Additional trusted Host authority (name[:port]); repeat for "
            "proxy or LAN names."
        )
        _item_type = str

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
                or self.allowed_host
            ):
                raise ValueError(
                    "site applications configure their own investigation; "
                    "inference and allowed-host flags require --state"
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
            allowed_hosts=self.allowed_host,
        )


class AskCommand(_StateCommand):
    _description_ = (
        "Print an investigation of saved evidence without changing the run."
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
        investigator = Investigator(InferenceConfig.from_toml(self.config))
        try:
            investigator.check_available()
        except (InferenceError, NooaError) as exc:
            raise CommandError(str(exc)) from exc
        snapshot = Monitor.load(self.state, recover=False).snapshot()
        snapshot.pop("storage_error")
        # The live directory keeps its sole writer. Reports from this command
        # exist only in the isolated copy and the JSON printed below.
        with TemporaryDirectory(prefix="cuphoton-xlens-ask-") as directory:
            Path(directory, "state.json").write_text(encode(snapshot))
            report = asyncio.run(
                investigator.investigate(
                    Monitor.load(directory),
                    self.sample,
                    question=self.question,
                )
            )
        print(json.dumps(report, indent=2, allow_nan=False))
        if report["status"] != "completed":
            raise CommandError(
                "investigation did not complete; see printed report status"
            )
