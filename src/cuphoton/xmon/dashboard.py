# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Installed, offline-capable dashboard with application-owned HTML views."""

from __future__ import annotations

import html
import json
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from typing import Any

from .monitor import Monitor
from .records import Sample, encode, sample_from_dict


class Dashboard:
    """Read-only inspection of a monitor and optional trusted domain views.

    ``render_sample`` is trusted application code, like its analyzer.
    It returns HTML with bounded inline images/SVG. Never use model
    output or untrusted input as HTML. Assets, fonts and scripts from external
    origins are not needed. The built-in shell bundles no third-party code.
    """

    def __init__(
        self,
        monitor: Monitor,
        *,
        title: str = "cuPhoton xMon",
        render_sample: Callable[[Sample], str] | None = None,
    ) -> None:
        self.monitor = monitor
        self.title = title
        self.render_sample = render_sample

    def render(self) -> str:
        snapshot = self.monitor.snapshot()
        records = sorted(
            snapshot["samples"].values(),
            key=lambda item: (
                item["sample"]["sequence"],
                item["sample"]["id"],
            ),
        )
        # Keep the UI bounded. The complete saved run remains available in
        # state.json; inspection displays the most recent 256 samples.
        records = records[-256:]
        panels = []
        for index, record in enumerate(records):
            sample = sample_from_dict(record["sample"])
            fragment = ""
            if self.render_sample:
                try:
                    fragment = self.render_sample(sample)
                    if len(fragment.encode()) > 2 * 1024 * 1024:
                        raise ValueError("domain view exceeds 2 MiB")
                except Exception:
                    fragment = (
                        '<p class="error">Domain view unavailable; '
                        "inspect application artifacts.</p>"
                    )
            panels.append(
                f'<section class="domain" data-index="{index}" hidden>'
                f"{fragment}</section>"
            )
        display: dict[str, Any] = {
            "run_id": snapshot["run_id"],
            "collector": snapshot["collector"],
            "storage_error": snapshot["storage_error"],
            "records": records,
            "history_count": len(snapshot["history"]),
            "total_samples": len(snapshot["samples"]),
            "errors": snapshot["errors"],
            "reports": snapshot["reports"],
        }
        resources = files("cuphoton.xmon").joinpath("assets")
        template = resources.joinpath("dashboard.html").read_text()
        data = (
            encode(display)
            .replace("<", "\\u003c")
            .replace(">", "\\u003e")
            .replace("&", "\\u0026")
        )
        return (
            template.replace("__TITLE__", html.escape(self.title))
            .replace(
                "__STYLE__", resources.joinpath("dashboard.css").read_text()
            )
            .replace("__PANELS__", "".join(panels))
            .replace("__DATA__", data)
            .replace(
                "__SCRIPT__", resources.joinpath("dashboard.js").read_text()
            )
        )

    def export(self, path: str | Path) -> Path:
        """Write a self-contained HTML snapshot, including domain views."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.render())
        return path


def make_server(
    dashboard: Dashboard,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> ThreadingHTTPServer:
    """Construct a read-only server; shared access control is external."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            try:
                dashboard.monitor.refresh()
            except (OSError, ValueError, KeyError, TypeError):
                self.send_error(503, "Saved monitor state is unavailable")
                return
            if self.path == "/":
                body = dashboard.render().encode()
                content_type = "text/html; charset=utf-8"
            elif self.path == "/api/state":
                body = encode(dashboard.monitor.snapshot()).encode()
                content_type = "application/json"
            elif self.path == "/health":
                state = dashboard.monitor.snapshot()
                body = json.dumps(
                    {
                        "collector": state["collector"],
                        "storage_error": state["storage_error"],
                    }
                ).encode()
                content_type = "application/json"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            # Scientific identifiers and HTTP input need not enter logs.
            pass

    return ThreadingHTTPServer((host, port), Handler)


def serve(
    dashboard: Dashboard, *, host: str = "127.0.0.1", port: int = 8765
) -> None:
    """Serve until interrupted; inference configuration stays separate."""
    with make_server(dashboard, host=host, port=port) as server:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
