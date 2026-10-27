# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import http.client
import io
import json
import sys
import threading
import types
import urllib.error
import urllib.request

import pytest

from cuphoton.core.cli import main
from cuphoton.xmon import Dashboard, Measurement, Monitor, Sample
from cuphoton.xmon.dashboard import make_server
from cuphoton.xmon.site import normalize_prefix, serve_site


def test_serve_cli_retains_saved_state_viewer(tmp_path, monkeypatch):
    monitor = Monitor(tmp_path / "state", run_id="saved-viewer")
    calls = []
    monkeypatch.setattr(
        "cuphoton.xmon.commands.serve",
        lambda dashboard, **kwargs: calls.append((dashboard, kwargs)),
    )
    assert (
        main(
            [
                "xmon",
                "serve",
                "--state",
                str(monitor.directory),
                "--prefix",
                "/review",
            ]
        )
        == 0
    )
    assert calls[0][0].monitor.snapshot()["run_id"] == "saved-viewer"
    assert calls[0][1] == {
        "host": "127.0.0.1",
        "port": 8765,
        "session": None,
        "prefix": "/review/",
    }


def test_site_cli_loads_installed_application_and_preserves_configuration(
    tmp_path, monkeypatch
):
    calls = []
    module = types.ModuleType("example_site")
    module.serve = lambda **kwargs: calls.append(kwargs)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    manifest = tmp_path / "xmon-site.toml"
    manifest.write_text(
        '[site]\napplication = "example_site:serve"\n'
        'prefix = "/observatory"\n[collection]\ntitle = "Measurements"\n'
    )
    args = ["xmon", "serve", "--site", str(tmp_path)]
    assert main(args + ["--host", "0.0.0.0", "--port", "8910"]) == 0
    assert calls[0] == {
        "root": tmp_path,
        "config": {
            "site": {
                "application": "example_site:serve",
                "prefix": "/observatory",
            },
            "collection": {"title": "Measurements"},
        },
        "host": "0.0.0.0",
        "port": 8910,
        "prefix": "/observatory/",
    }
    serve_site(manifest, prefix="/review/")
    assert calls[-1]["prefix"] == "/review/"
    for extra in [
        ["--state", str(tmp_path)],
        ["--auto-investigate"],
        ["--inference-config", "inference.toml"],
        ["--access-token-env", "ACCESS_TOKEN"],
    ]:
        assert main(args + extra, err=io.StringIO()) != 0
    assert len(calls) == 2
    assert main(["xmon", "serve"], err=io.StringIO()) != 0


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "requires a \\[site\\] table"),
        ('[site]\napplication = "missing"', "module:callable"),
        ('[site]\napplication = "example_site:"', "module:callable"),
        ('[site]\napplication = "not_installed_xmon_site:serve"', "install"),
    ],
)
def test_site_reports_actionable_configuration_errors(
    tmp_path, text, message
):
    (tmp_path / "xmon-site.toml").write_text(text)
    with pytest.raises(ValueError, match=message):
        serve_site(tmp_path)


def test_site_reports_the_cause_of_application_import_failure(
    tmp_path, monkeypatch
):
    (tmp_path / "xmon-site.toml").write_text(
        '[site]\napplication = "example_site:serve"\n'
    )

    def unavailable(module):
        raise ModuleNotFoundError("No module named 'application_dependency'")

    monkeypatch.setattr(
        "cuphoton.xmon.site.importlib.import_module", unavailable
    )
    with pytest.raises(ValueError, match="application_dependency"):
        serve_site(tmp_path)


@pytest.mark.parametrize("prefix", ["relative", "//site", "/a/../b", "/a?b"])
def test_site_prefix_requires_a_plain_absolute_mount(prefix):
    with pytest.raises(ValueError):
        normalize_prefix(prefix)


def test_mounted_dashboard_refresh_and_lazy_domain_views(tmp_path):
    source = Monitor(tmp_path / "state", run_id="mounted")
    for sequence in range(3):
        source.publish(
            Sample(
                f"sample-{sequence}",
                sequence,
                {"level": Measurement(float(sequence), "count")},
            )
        )
    rendered = []

    def view(sample):
        rendered.append(sample.id)
        return f"<p>domain-view-{sample.sequence}</p>"

    dashboard = Dashboard(
        Monitor.load(source.directory, recover=False), render_sample=view
    )
    server = make_server(dashboard, port=0, prefix="/review/run")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        connection = http.client.HTTPConnection(
            "127.0.0.1", server.server_port
        )
        connection.request("GET", "/review/run?view=saved")
        redirect = connection.getresponse()
        assert redirect.status == 308
        assert redirect.getheader("Location") == "/review/run/?view=saved"
        assert redirect.getheader("Cache-Control") == "no-store"
        redirect.read()
        connection.close()
        with urllib.request.urlopen(base + "/review/run") as response:
            assert response.url.endswith("/review/run/")
            initial = response.read().decode()
            assert "domain-view-0" in initial
            assert "domain-view-1" not in initial
            assert "connect-src 'self'" in initial
        assert rendered == ["sample-0"]
        source.publish(
            Sample("sample-3", 3, {"level": Measurement(3.0, "count")})
        )
        with urllib.request.urlopen(
            base + "/review/run/api/dashboard?sample_id=sample-3"
        ) as response:
            data = json.load(response)
        assert data["total_samples"] == 4
        assert data["domain_view"]["sample_id"] == "sample-3"
        assert data["domain_view"]["html"] == "<p>domain-view-3</p>"
        assert rendered == ["sample-0", "sample-3"]
        for path in ["/", "/api/dashboard", "/review/runner/"]:
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(base + path)
            assert error.value.code == 404
        exported = dashboard.export(tmp_path / "offline.html").read_text()
        assert all(f"domain-view-{index}" in exported for index in range(4))
        assert '"live":false' in exported
        assert "connect-src 'none'" in exported
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
