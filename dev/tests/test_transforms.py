# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from cuphoton_dev.transforms import SITE_URL, check_snapshot, postprocess


def rendered_guide(path: Path, *, extra_page: bool = False) -> Path:
    """Small Quarto-shaped fixture preserving v0.2.0's split syntax spans."""
    path.mkdir(parents=True)
    pages = {
        "index.html": (
            "<p>The main image calculations remain unchanged.</p>"
            '<a href="install.html#clone-the-repository">Install</a>'
        ),
        "install.html": (
            '<section id="clone-the-repository"><pre><code><span id="cb1-1">'
            '<a href="#cb1-1"></a><span class="fu">git</span> clone '
            "https://github.com/NVIDIA/cuPhoton.git</span>\n"
            '<span id="cb1-2">cd cuPhoton</span></code></pre>'
            "<p>This clones the development branch, "
            "which this guide describes. "
            "To use a published version instead, pick its tag from "
            '<a href="https://github.com/NVIDIA/cuPhoton/releases">'
            "Releases</a>, "
            "add <code>--branch TAG</code> to the clone command, and use the "
            "documentation in that checkout.</p></section>"
        ),
        "reference.html": (
            "<p>The links point to the repository’s "
            "<code>main</code> branch, "
            "which can be newer than this guide.</p>"
            '<a href="https://github.com/NVIDIA/cuPhoton/blob/main/docs/cli.md#example">API</a>'
            '<a href="https://github.com/another/project/blob/main/README.md">Other</a>'
            '<a href="https://raw.githubusercontent.com/NVIDIA/cuPhoton/main/README.md">Raw</a>'
        ),
    }
    if extra_page:
        pages["nested/example.html"] = (
            '<a href="../index.html">Home</a><p id="new-topic">New</p>'
        )
    for name, body in pages.items():
        file = path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(
            "<!doctype html><html><head><title>cuPhoton</title></head>"
            "<body><main>" + body + "</main></body></html>"
        )
    (path / "search.json").write_text(
        json.dumps(
            [
                {
                    "href": name,
                    "objectID": name,
                    "text": BeautifulSoup(body, "html.parser").get_text(),
                }
                for name, body in pages.items()
            ]
        )
    )
    (path / "sitemap.xml").write_text(
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        + "".join(f"<url><loc>{SITE_URL}{name}</loc></url>" for name in pages)
        + "</urlset>"
    )
    return path


def snapshot(
    path: Path, version: str = "main", sha: str = "a" * 40, **kwargs
) -> Path:
    rendered_guide(path, **kwargs)
    postprocess(
        path,
        version=version,
        source_sha=sha,
        controller_sha="b" * 40,
        renderer="1.9.36",
        lock_sha256="c" * 64,
    )
    return path


def test_release_conversion_preserves_unrelated_main_and_markup(tmp_path):
    path = snapshot(tmp_path / "render", "v0.2.0", extra_page=True)
    install = BeautifulSoup(
        (path / "install.html").read_text(), "html.parser"
    )
    assert (
        "git clone --branch v0.2.0 --single-branch https://github.com/NVIDIA/cuPhoton.git"
        in install.code.get_text()
    )
    assert install.find(id="cb1-1") is not None
    assert "This clones release v0.2.0" in install.get_text()
    reference = (path / "reference.html").read_text()
    assert "/cuPhoton/blob/v0.2.0/" in reference
    assert "/cuPhoton/v0.2.0/README.md" in reference
    assert "/another/project/blob/main/" in reference
    assert (
        "The main image calculations remain unchanged"
        in (path / "index.html").read_text()
    )
    assert "development branch" not in (path / "search.json").read_text()
    receipt = check_snapshot(path)
    assert receipt["lock_sha256"] == "c" * 64
    assert "clone-the-repository" in receipt["pages"]["install.html"]
    nested = BeautifulSoup(
        (path / "nested/example.html").read_text(), "html.parser"
    )
    assert (
        nested.select_one("script[data-catalog]")["data-catalog"]
        == "../../versions.json"
    )
    assert (
        nested.select_one('link[rel="canonical"]')["href"]
        == SITE_URL + "v0.2.0/nested/example.html"
    )


def test_main_keeps_authored_install_and_source_links(tmp_path):
    path = snapshot(tmp_path / "main")
    assert "development branch" in (path / "install.html").read_text()
    assert "/cuPhoton/blob/main/" in (path / "reference.html").read_text()
    assert "main (development)" in (path / "index.html").read_text()


@pytest.mark.parametrize("mutation", ["wording", "clone", "search", "link"])
def test_release_conversion_fails_on_changed_contract(tmp_path, mutation):
    path = rendered_guide(tmp_path / "render")
    file = path / ("search.json" if mutation == "search" else "install.html")
    text = file.read_text()
    if mutation == "wording":
        text = text.replace(
            "This clones the development branch", "This checks out main"
        )
    elif mutation == "clone":
        text = text.replace("git</span> clone", "git</span> <em>clone</em>")
    elif mutation == "search":
        text = text.replace(
            "This clones the development branch",
            "This clones the development branch (updated)",
        )
    else:
        text = text.replace(
            "</main>", '<a href="missing.html">Missing</a></main>'
        )
    file.write_text(text)
    with pytest.raises(ValueError):
        postprocess(
            path,
            version="v0.2.0",
            source_sha="a",
            controller_sha="b",
            renderer="test",
        )


def test_checks_detect_tampering_and_external_asset_escape(tmp_path):
    path = snapshot(tmp_path / "render")
    (path / "index.html").write_text("tampered")
    with pytest.raises(ValueError, match="hashes"):
        check_snapshot(path)
    path2 = rendered_guide(tmp_path / "other")
    (path2 / "index.html").write_text(
        (path2 / "index.html")
        .read_text()
        .replace("</main>", '<img src="../main/image.png"></main>')
    )
    with pytest.raises(ValueError, match="Missing local target"):
        postprocess(
            path2,
            version="v0.2.0",
            source_sha="a",
            controller_sha="b",
            renderer="test",
        )


def test_source_contract_tracks_guide_inputs(tmp_path):
    """Keep authored instructions synchronized with conversion fixtures."""
    guide = Path(__file__).parents[2] / "docs" / "guide"
    install = (guide / "install.qmd").read_text()
    reference = (guide / "reference.qmd").read_text()
    fixture = rendered_guide(tmp_path / "fixture")
    paragraph = (
        next(
            line
            for line in install.splitlines()
            if line.startswith("This clones")
        )
        .replace(
            "[Releases](https://github.com/NVIDIA/cuPhoton/releases)",
            "Releases",
        )
        .replace("`", "")
    )
    fixture_html = BeautifulSoup(
        (fixture / "install.html").read_text(), "html.parser"
    )
    assert fixture_html.p.get_text() == paragraph
    assert "git clone https://github.com/NVIDIA/cuPhoton.git" in install
    assert (
        "This clones the development branch, which this guide describes."
        in install
    )
    assert (
        "The links point to the repository's `main` branch, "
        "which can be newer than this guide." in reference
    )


@pytest.mark.parametrize("link", ["symbolic", "hard"])
def test_snapshot_rejects_links_that_pages_cannot_upload(tmp_path, link):
    path = rendered_guide(tmp_path / "render")
    target = path / "linked.html"
    if link == "symbolic":
        target.symlink_to("index.html")
    else:
        os.link(path / "index.html", target)
    with pytest.raises(ValueError, match="link"):
        postprocess(
            path,
            version="main",
            source_sha="a",
            controller_sha="b",
            renderer="test",
        )


@pytest.mark.parametrize("location", ["html", "search"])
def test_snapshot_rejects_missing_bookmarks(tmp_path, location):
    path = rendered_guide(tmp_path / "render")
    file = path / ("index.html" if location == "html" else "search.json")
    text = file.read_text()
    if location == "html":
        text = text.replace("#clone-the-repository", "#missing")
    else:
        records = json.loads(text)
        records[0]["href"] = "install.html#missing"
        text = json.dumps(records)
    file.write_text(text)
    with pytest.raises(ValueError, match="bookmark"):
        postprocess(
            path,
            version="main",
            source_sha="a",
            controller_sha="b",
            renderer="test",
        )
