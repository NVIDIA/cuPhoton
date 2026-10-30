# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Installed, offline-capable dashboard with application-owned HTML views."""

from __future__ import annotations

import base64
import hashlib
import html
import ipaddress
import json
import re
import threading
from collections import OrderedDict
from collections.abc import Callable, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from .monitor import Monitor
from .records import Sample, encode, sample_from_dict
from .session import MonitorSession
from .site import normalize_prefix


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
        title: str = "cuPhoton xMonitor",
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

    def display_sample(
        self, sample_id: str, *, ui: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Display data plus one selected domain view for live polling."""
        display = self.display(ui=ui)
        record = next(
            (
                item
                for item in display["records"]
                if item["sample"]["id"] == sample_id
            ),
            None,
        )
        if record:
            sample = sample_from_dict(record["sample"])
            display["domain_view"] = {
                "sample_id": sample.id,
                "revision": sample.revision,
                "digest": record["digest"],
                "html": self.domain_view(sample),
            }
        return display

    def render(self, *, ui: dict[str, Any] | None = None) -> str:
        display = self.display(ui=ui)
        panels = []
        # Live viewers request other domain views as samples are selected.
        # Offline exports retain every visible sample for disconnected use.
        records = display["records"]
        if display["ui"].get("live"):
            records = records[:1]
        for index, record in enumerate(records):
            sample = sample_from_dict(record["sample"])
            fragment = self.domain_view(sample)
            panels.append(
                f'<section class="domain" data-index="{index}" hidden>'
                f"{fragment}</section>"
            )
        resources = files("cuphoton.xmonitor").joinpath("assets")
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
        script = resources.joinpath("dashboard.js").read_text()
        script_hash = base64.b64encode(
            hashlib.sha256(script.encode()).digest()
        ).decode()
        values = {
            "__TITLE__": html.escape(self.title),
            "__STYLE__": resources.joinpath("dashboard.css").read_text(),
            "__PANELS__": "".join(panels),
            "__DATA__": data,
            "__SCRIPT__": script,
            "__SCRIPT_HASH__": script_hash,
        }
        return re.sub(
            "|".join(values), lambda match: values[match.group(0)], template
        )

    def export(self, path: str | Path) -> Path:
        """Write a self-contained HTML snapshot, including domain views."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.render())
        return path


def _authority(value: str) -> str:
    """Validate an explicitly trusted HTTP Host authority."""
    parsed = urlsplit("//" + value)
    if (
        not value
        or any(character.isspace() for character in value)
        or parsed.username is not None
        or parsed.password is not None
        or not parsed.hostname
        or parsed.path
        or parsed.query
        or parsed.fragment
        or value.endswith(":")
    ):
        raise ValueError(
            "allowed hosts require a hostname or [IPv6], "
            "optionally with a port"
        )
    # Accessing port also rejects malformed and out-of-range ports.
    port = parsed.port
    hostname = parsed.hostname.lower()
    if ":" in hostname:
        hostname = f"[{ipaddress.IPv6Address(hostname)}]"
    elif not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?", hostname):
        raise ValueError("invalid allowed hostname")
    return hostname if port is None else f"{hostname}:{port}"


def make_server(
    dashboard: Dashboard,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    session: MonitorSession | None = None,
    prefix: str = "/",
    allowed_hosts: Sequence[str] = (),
) -> ThreadingHTTPServer:
    """Construct a viewer with optional investigation actions."""
    prefix = normalize_prefix(prefix)
    trusted_hosts = {_authority(value) for value in allowed_hosts}
    _authority(
        f"[{host}]" if ":" in host and not host.startswith("[") else host
    )
    trusted_origins: set[str] = set()
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
        def _trusted_host(self) -> bool:
            values = self.headers.get_all("Host", [])
            try:
                valid = (
                    len(values) == 1
                    and _authority(values[0]) in trusted_hosts
                )
            except ValueError:
                valid = False
            if not valid:
                self.send_error(421, "Unrecognized monitor host")
            return valid

        def do_GET(self) -> None:
            if not self._trusted_host():
                return
            parsed = urlsplit(self.path)
            if prefix != "/" and parsed.path == prefix.rstrip("/"):
                self.send_response(308)
                location = prefix
                if parsed.query:
                    location += "?" + parsed.query
                self.send_header("Location", location)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if not parsed.path.startswith(prefix):
                self.send_error(404)
                return
            path = parsed.path[len(prefix) :]
            try:
                if session is None:
                    dashboard.monitor.refresh()
            except (OSError, ValueError, KeyError, TypeError):
                self.send_error(503, "Saved monitor state is unavailable")
                return
            if path == "":
                body = dashboard.render(ui=ui()).encode()
                content_type = "text/html; charset=utf-8"
            elif path == "api/state":
                body = encode(dashboard.monitor.snapshot()).encode()
                content_type = "application/json"
            elif path == "api/dashboard":
                identifier = parse_qs(parsed.query).get("sample_id", [""])[0]
                display = dashboard.display_sample(identifier, ui=ui())
                body = encode(display).encode()
                content_type = "application/json"
            elif path == "health":
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
            if not self._trusted_host():
                return
            if self.path not in {prefix + "api/ask", prefix + "api/cancel"}:
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
            if origin and origin not in trusted_origins:
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
            if not session.authenticate(
                self.headers.get("X-XMonitor-Token", "")
            ):
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
                if self.path == prefix + "api/ask":
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

    server = ThreadingHTTPServer((host, port), Handler)
    addresses = {"127.0.0.1", "localhost", "[::1]"} if loopback else {host}
    for address in addresses:
        if ":" in address and not address.startswith("["):
            address = f"[{address}]"
        trusted_hosts.add(_authority(f"{address}:{server.server_port}"))
        if server.server_port in {80, 443}:
            trusted_hosts.add(_authority(address))
    trusted_origins.update(
        f"{scheme}://{authority}"
        for authority in trusted_hosts
        for scheme in ("http", "https")
    )
    return server


def serve(
    dashboard: Dashboard,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    session: MonitorSession | None = None,
    prefix: str = "/",
    allowed_hosts: Sequence[str] = (),
) -> None:
    """Serve until interrupted, owning the optional live session lifecycle."""
    with make_server(
        dashboard,
        host=host,
        port=port,
        session=session,
        prefix=prefix,
        allowed_hosts=allowed_hosts,
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
