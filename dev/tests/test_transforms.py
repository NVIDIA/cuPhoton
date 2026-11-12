# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest
from bs4 import BeautifulSoup

from cuphoton_dev.transforms import (
    SITE_URL,
    check_snapshot,
    file_hashes,
    postprocess,
)


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
    with pytest.raises(ValueError, match="escapes selected version"):
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


def test_directory_links_resolve_index_pages_and_bookmarks(tmp_path):
    path = rendered_guide(tmp_path / "render", extra_page=True)
    (path / "nested/example.html").rename(path / "nested/index.html")
    for name in ("sitemap.xml", "search.json"):
        file = path / name
        file.write_text(
            file.read_text(encoding="utf-8").replace(
                "nested/example.html", "nested/index.html"
            ),
            encoding="utf-8",
        )
    root_links = (
        "./",
        "nested/",
        "nested/./",
        "nested/../",
        ".",
        "/cuPhoton/nested/",
        SITE_URL + "nested/",
    )
    index = path / "index.html"
    index.write_text(
        index.read_text(encoding="utf-8").replace(
            "</main>",
            "".join(f'<a href="{link}">Directory</a>' for link in root_links)
            + "</main>",
        ),
        encoding="utf-8",
    )
    nested = path / "nested/index.html"
    nested.write_text(
        nested.read_text(encoding="utf-8").replace(
            "</main>",
            '<a href="../">Root</a><a href="..">Root dot segment</a>'
            '<a href="./#new-topic">Same directory bookmark</a></main>',
        ),
        encoding="utf-8",
    )
    search = path / "search.json"
    records = json.loads(search.read_text(encoding="utf-8"))
    records[-1]["href"] = "nested/?search=1#new-topic"
    search.write_text(json.dumps(records), encoding="utf-8")
    postprocess(
        path,
        version="v0.2.0",
        source_sha="a",
        controller_sha="b",
        renderer="test",
    )
    assert "nested/index.html" in check_snapshot(path)["pages"]


@pytest.mark.parametrize(
    "href",
    [
        "../",
        "/cuPhoton/main/../",
        SITE_URL + "main/../",
    ],
)
def test_directory_links_cannot_escape_snapshot(tmp_path, href):
    path = rendered_guide(tmp_path / "render")
    index = path / "index.html"
    index.write_text(
        index.read_text(encoding="utf-8").replace(
            "</main>", f'<a href="{href}">Outside</a></main>'
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="escapes selected version"):
        postprocess(
            path,
            version="main",
            source_sha="a",
            controller_sha="b",
            renderer="test",
        )


def test_utf8_snapshot_and_archive_under_ascii_locale(tmp_path):
    path = rendered_guide(tmp_path / "render", extra_page=True)
    (path / "nested/example.html").rename(path / "nested/café.html")
    for name in ("sitemap.xml", "search.json"):
        file = path / name
        file.write_text(
            file.read_text(encoding="utf-8").replace(
                "nested/example.html", "nested/café.html"
            ),
            encoding="utf-8",
        )
    index = path / "index.html"
    index.write_text(
        index.read_text(encoding="utf-8").replace(
            "</main>", '<p id="café">μ and café</p></main>'
        ),
        encoding="utf-8",
    )
    script = r"""
import locale
import sys
from pathlib import Path
from cuphoton_dev.archive import assemble, check_archive
from cuphoton_dev.transforms import postprocess, check_snapshot
assert not sys.flags.utf8_mode
locale.setlocale(locale.LC_CTYPE, "C")
assert locale.getencoding().lower() in {"ascii", "ansi_x3.4-1968"}
source, archive = map(Path, sys.argv[1:])
postprocess(source, version="v0.2.0", source_sha="a", controller_sha="b",
            renderer="test")
assert "caf\u00e9" in check_snapshot(source)["pages"]["index.html"]
text = (source / "index.html").read_text(encoding="utf-8")
assert "\u03bc and caf\u00e9" in text
assemble(archive, source, eligible_releases={"v0.2.0"})
assemble(archive, source, eligible_releases={"v0.2.0"})
assert check_archive(archive)["stable"] == "v0.2.0"
redirect = archive / "nested/caf\u00e9.html"
assert "nested/caf\u00e9.html" in redirect.read_text(encoding="utf-8")
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8=0",
            "-c",
            script,
            str(path),
            str(tmp_path / "archive"),
        ],
        env={
            **os.environ,
            # Keep UTF-8 filesystem paths, then switch text IO to ASCII above.
            "LC_ALL": "C.UTF-8",
            "PYTHONUTF8": "0",
            "PYTHONCOERCECLOCALE": "0",
        },
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("schema", "Unsupported documentation receipt schema"),
        ("version", "Not a documentation version"),
        ("required-output", "missing required guide output"),
        ("page-inventory", "HTML inventory is incomplete"),
        ("canonical", "Wrong canonical URL"),
        ("anchor-inventory", "Wrong anchor inventory"),
        ("search-target", "Search target is outside snapshot"),
        ("sitemap", "Sitemap does not describe"),
    ],
)
def test_snapshot_rejects_inconsistent_metadata(
    tmp_path, corruption, message
):
    path = snapshot(tmp_path / "snapshot", "v0.2.0")
    receipt_path = path / "build.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if corruption == "schema":
        receipt["schema"] = 2
    elif corruption == "version":
        receipt["version"] = "v0.2.3\u0663"
    elif corruption == "required-output":
        (path / "search.json").unlink()
    elif corruption == "page-inventory":
        receipt["pages"].pop("install.html")
    elif corruption == "canonical":
        index = path / "index.html"
        soup = BeautifulSoup(index.read_text(encoding="utf-8"), "html.parser")
        soup.select_one('link[rel="canonical"]')["href"] = SITE_URL + "main/"
        index.write_text(str(soup), encoding="utf-8")
    elif corruption == "anchor-inventory":
        receipt["pages"]["install.html"].append("unrendered-anchor")
    elif corruption == "search-target":
        search = path / "search.json"
        records = json.loads(search.read_text(encoding="utf-8"))
        records[0]["href"] = "not-a-chapter.html"
        search.write_text(json.dumps(records), encoding="utf-8")
    else:
        sitemap = path / "sitemap.xml"
        tree = ET.parse(sitemap)
        tree.getroot().remove(tree.getroot()[0])
        tree.write(sitemap, encoding="utf-8", xml_declaration=True)
    # Exercise semantic checks even when the receipt matches the file bytes.
    receipt["files"] = file_hashes(path)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        check_snapshot(path)
