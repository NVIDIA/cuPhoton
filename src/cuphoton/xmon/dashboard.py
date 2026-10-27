# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Installed, offline-capable dashboard with application-owned HTML views."""

from __future__ import annotations

import html
import ipaddress
import json
import threading
from collections import OrderedDict
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from .monitor import Monitor
from .records import Sample, encode, sample_from_dict
from .session import MonitorSession


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
        self._views: OrderedDict[str, str] = OrderedDict()
        self._view_lock = threading.Lock()

    def display(self, *, ui: dict[str, Any] | None = None) -> dict[str, Any]:
        """Bounded display data shared by live polling and offline exports."""
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
        return {
            "run_id": snapshot["run_id"],
            "collector": snapshot["collector"],
            "storage_error": snapshot["storage_error"],
            "records": records,
            "history_count": len(snapshot["history"]),
            "total_samples": len(snapshot["samples"]),
            "errors": snapshot["errors"],
            "reports": snapshot["reports"],
            "ui": ui
            or {
                "live": False,
                "can_ask": False,
                "reason": "Saved snapshot; questions need a live monitor.",
            },
        }

    def domain_view(self, sample: Sample) -> str:
        if not self.render_sample:
            return ""
        try:
            digest = Monitor._digest(sample)
            with self._view_lock:
                if digest in self._views:
                    self._views.move_to_end(digest)
                    return self._views[digest]
                fragment = self.render_sample(sample)
                if len(fragment.encode()) > 2 * 1024 * 1024:
                    raise ValueError("domain view exceeds 2 MiB")
                self._views[digest] = fragment
                if len(self._views) > 16:
                    self._views.popitem(last=False)
                return fragment
        except Exception:
            return (
                '<p class="error">Domain view unavailable; '
                "inspect application artifacts.</p>"
            )

    def render(self, *, ui: dict[str, Any] | None = None) -> str:
        display = self.display(ui=ui)
        panels = []
        for index, record in enumerate(display["records"]):
            sample = sample_from_dict(record["sample"])
            fragment = self.domain_view(sample)
            panels.append(
                f'<section class="domain" data-index="{index}" hidden>'
                f"{fragment}</section>"
            )
        resources = files("cuphoton.xmon").joinpath("assets")
        template = resources.joinpath("dashboard.html").read_text()
        if display["ui"].get("live"):
            template = template.replace(
                "connect-src 'none'", "connect-src 'self'"
            )
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
    session: MonitorSession | None = None,
) -> ThreadingHTTPServer:
    """Construct a viewer with optional investigation actions."""
    if session is not None and session.monitor is not dashboard.monitor:
        raise ValueError("dashboard and session must share one monitor")
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = host == "localhost"
    if (
        session
        and session.config
        and not loopback
        and not session.authentication_required
    ):
        raise ValueError(
            "non-loopback investigation requires access_token_env"
        )

    def ui() -> dict[str, Any]:
        return (
            session.ui()
            if session
            else {
                "live": True,
                "can_ask": False,
                "reason": "Start a session with inference configuration "
                "to ask questions.",
                "refresh_seconds": 1,
            }
        )

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            try:
                if session is None:
                    dashboard.monitor.refresh()
            except (OSError, ValueError, KeyError, TypeError):
                self.send_error(503, "Saved monitor state is unavailable")
                return
            parsed = urlsplit(self.path)
            if parsed.path == "/":
                body = dashboard.render(ui=ui()).encode()
                content_type = "text/html; charset=utf-8"
            elif parsed.path == "/api/state":
                body = encode(dashboard.monitor.snapshot()).encode()
                content_type = "application/json"
            elif parsed.path == "/api/dashboard":
                display = dashboard.display(ui=ui())
                identifier = parse_qs(parsed.query).get("sample_id", [""])[0]
                record = next(
                    (
                        item
                        for item in display["records"]
                        if item["sample"]["id"] == identifier
                    ),
                    None,
                )
                if record:
                    sample = sample_from_dict(record["sample"])
                    display["domain_view"] = {
                        "sample_id": sample.id,
                        "revision": sample.revision,
                        "digest": record["digest"],
                        "html": dashboard.domain_view(sample),
                    }
                body = encode(display).encode()
                content_type = "application/json"
            elif parsed.path == "/health":
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
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Frame-Options", "DENY")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload: dict[str, Any]) -> None:
            body = encode(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            if self.path not in {"/api/ask", "/api/cancel"}:
                self._json(404, {"error": "Unknown monitor action."})
                return
            if session is None or not session.can_ask:
                self._json(
                    503,
                    {
                        "error": session.reason
                        if session
                        else "Investigation is not configured."
                    },
                )
                return
            origin = self.headers.get("Origin")
            if origin and origin not in {
                "http://" + self.headers.get("Host", ""),
                "https://" + self.headers.get("Host", ""),
            }:
                self._json(
                    403,
                    {"error": "Open the monitor page to investigate."},
                )
                return
            if self.headers.get("Sec-Fetch-Site") in {
                "cross-site",
                "same-site",
            }:
                self._json(
                    403,
                    {"error": "Cross-origin investigation is unavailable."},
                )
                return
            if not session.authenticate(self.headers.get("X-XMon-Token", "")):
                self._json(
                    401,
                    {"error": "Enter a valid investigation access token."},
                )
                return
            if (
                self.headers.get("Content-Type", "").split(";", 1)[0]
                != "application/json"
            ):
                self._json(415, {"error": "Monitor actions require JSON."})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 1 <= length <= 16384:
                    raise ValueError(
                        "request must contain at most 16384 bytes"
                    )
                request = json.loads(self.rfile.read(length))
                if not isinstance(request, dict):
                    raise ValueError("request must be a JSON object")
                if self.path == "/api/ask":
                    if set(request) - {"question", "scope", "sample_ids"}:
                        raise ValueError("unsupported question fields")
                    identifier = session.submit(
                        request.get("question", ""),
                        request.get("sample_ids", []),
                        scope=request.get("scope", "selected"),
                    )
                    self._json(202, {"report_id": identifier})
                else:
                    report_id = request.get("report_id")
                    if set(request) != {"report_id"} or not isinstance(
                        report_id, str
                    ):
                        raise ValueError("supply a report ID to cancel")
                    self._json(200, {"cancelled": session.cancel(report_id)})
            except (ValueError, TypeError, KeyError) as exc:
                self._json(400, {"error": str(exc)})

        def log_message(self, format: str, *args: Any) -> None:
            # Scientific identifiers and HTTP input need not enter logs.
            pass

    return ThreadingHTTPServer((host, port), Handler)


def serve(
    dashboard: Dashboard,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    session: MonitorSession | None = None,
) -> None:
    """Serve until interrupted, owning the optional live session lifecycle."""
    with make_server(
        dashboard, host=host, port=port, session=session
    ) as server:
        try:
            if session:
                session.start()
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            if session:
                session.close()
