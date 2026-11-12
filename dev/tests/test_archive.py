# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import json
from xml.etree import ElementTree as ET

import pytest

from cuphoton_dev.archive import assemble, check_archive, redirect_html
from cuphoton_dev.transforms import SITE_URL, SITEMAP_NAMESPACE, file_hashes

from .test_transforms import snapshot


def test_archive_preserves_releases_and_semantic_stable(tmp_path):
    archive = tmp_path / "site"
    main = snapshot(tmp_path / "main", extra_page=True)
    initial = assemble(archive, main, eligible_releases=set())
    assert initial["preferred"] == "main"
    assert initial["stable"] is None
    newest = snapshot(tmp_path / "new", "v0.10.0")
    catalog = assemble(archive, newest, eligible_releases={"v0.10.0"})
    hashes = file_hashes(archive / "v0.10.0")
    older = snapshot(tmp_path / "old", "v0.2.1")
    catalog = assemble(
        archive, older, eligible_releases={"v0.10.0", "v0.2.1"}
    )
    assert catalog["stable"] == "v0.10.0"
    assert file_hashes(archive / "v0.10.0") == hashes
    assert (
        catalog["aliases"]["nested/example.html"] == "../v0.10.0/index.html"
    )
    updated = snapshot(tmp_path / "next-main", sha="d" * 40)
    assemble(
        archive,
        updated,
        eligible_releases={"v0.10.0", "v0.2.1"},
        is_ancestor=lambda old, new: old == "a" * 40 and new == "d" * 40,
    )
    assert file_hashes(archive / "v0.10.0") == hashes
    assert check_archive(archive)["stable"] == "v0.10.0"


def test_duplicate_release_keeps_original_bytes_and_moved_tag_fails(tmp_path):
    archive = tmp_path / "site"
    first = snapshot(tmp_path / "first", "v0.2.0")
    assemble(archive, first, eligible_releases={"v0.2.0"})
    original = file_hashes(archive)
    duplicate = snapshot(tmp_path / "duplicate", "v0.2.0", extra_page=True)
    assemble(archive, duplicate, eligible_releases={"v0.2.0"})
    assert file_hashes(archive) == original
    moved = snapshot(tmp_path / "moved", "v0.2.0", sha="f" * 40)
    with pytest.raises(ValueError, match="Tag moved"):
        assemble(archive, moved, eligible_releases={"v0.2.0"})
    assert file_hashes(archive) == original


def test_failed_or_stale_update_leaves_archive_intact(tmp_path):
    archive = tmp_path / "site"
    main = snapshot(tmp_path / "main")
    assemble(archive, main, eligible_releases=set())
    original = file_hashes(archive)
    stale = snapshot(tmp_path / "stale", sha="c" * 40)
    with pytest.raises(ValueError, match="Stale"):
        assemble(
            archive,
            stale,
            eligible_releases=set(),
            is_ancestor=lambda old, new: False,
        )
    assert file_hashes(archive) == original
    with pytest.raises(ValueError, match="Stale"):
        assemble(archive, stale, eligible_releases=set())
    (stale / "index.html").write_text("bad render")
    with pytest.raises(ValueError, match="hashes"):
        assemble(archive, stale, eligible_releases=set())
    assert file_hashes(archive) == original
    # A retry of the valid state is idempotent and still deployable.
    assemble(archive, main, eligible_releases=set())
    assert file_hashes(archive) == original


def test_draft_and_prerelease_are_not_publishable(tmp_path):
    archive = tmp_path / "site"
    release = snapshot(tmp_path / "release", "v0.2.0")
    with pytest.raises(ValueError, match="not eligible"):
        assemble(archive, release, eligible_releases=set())
    assert not archive.exists()
    with pytest.raises(ValueError, match="Not a formal version"):
        assemble(archive, release, eligible_releases={"v0.2.0rc1"})


def test_catalog_and_alias_tampering_detected(tmp_path):
    archive = tmp_path / "site"
    main = snapshot(tmp_path / "main")
    assemble(archive, main, eligible_releases=set())
    (archive / "install.html").write_text("wrong redirect")
    with pytest.raises(ValueError, match="redirect"):
        check_archive(archive)
    assemble(archive, main, eligible_releases=set())
    catalog = json.loads((archive / "versions.json").read_text())
    catalog["versions"][0]["pages"].pop("install.html")
    (archive / "versions.json").write_text(json.dumps(catalog))
    with pytest.raises(ValueError, match="Catalog"):
        check_archive(archive)


def test_redirect_preserves_query_and_bookmark():
    text = redirect_html("v0.2.0/install.html")
    assert "target.search = location.search" in text
    assert "target.hash = location.hash" in text
    assert '<a href="v0.2.0/install.html">' in text
    assert "http-equiv" in text


def test_removed_main_chapter_keeps_legacy_redirect(tmp_path):
    archive = tmp_path / "site"
    first = snapshot(tmp_path / "first", extra_page=True)
    assemble(archive, first, eligible_releases=set())
    second = snapshot(tmp_path / "second", sha="e" * 40)
    catalog = assemble(
        archive,
        second,
        eligible_releases=set(),
        is_ancestor=lambda old, new: True,
    )
    assert catalog["aliases"]["nested/example.html"] == "../main/index.html"
    assert check_archive(archive) == catalog


def test_root_sitemap_retains_snapshots_when_release_loses_eligibility(
    tmp_path,
):
    archive = tmp_path / "site"
    main = snapshot(tmp_path / "main")
    release = snapshot(tmp_path / "release", "v0.2.0")
    assemble(archive, main, eligible_releases=set())
    namespace = {"s": SITEMAP_NAMESPACE}
    index = ET.parse(archive / "sitemap.xml").getroot()
    assert index.tag == f"{{{SITEMAP_NAMESPACE}}}sitemapindex"
    assert [
        loc.text for loc in index.findall("s:sitemap/s:loc", namespace)
    ] == [SITE_URL + "main/sitemap.xml"]
    assemble(archive, release, eligible_releases={"v0.2.0"})
    release_hashes = file_hashes(archive / "v0.2.0")
    catalog = assemble(archive, main, eligible_releases=set())
    assert catalog["stable"] is None
    assert catalog["preferred"] == "main"
    index = ET.parse(archive / "sitemap.xml").getroot()
    assert {
        loc.text for loc in index.findall("s:sitemap/s:loc", namespace)
    } == {
        SITE_URL + "main/sitemap.xml",
        SITE_URL + "v0.2.0/sitemap.xml",
    }
    assert file_hashes(archive / "v0.2.0") == release_hashes
    assert check_archive(archive) == catalog


@pytest.mark.parametrize(
    "corruption", ["missing", "namespace", "target", "entry", "duplicate"]
)
def test_archive_rejects_invalid_root_sitemap(tmp_path, corruption):
    archive = tmp_path / "site"
    main = snapshot(tmp_path / "main")
    assemble(archive, main, eligible_releases=set())
    sitemap = archive / "sitemap.xml"
    if corruption == "missing":
        sitemap.unlink()
    else:
        tree = ET.parse(sitemap)
        index = tree.getroot()
        if corruption == "namespace":
            index.tag = "{https://example.invalid/namespace}sitemapindex"
        elif corruption == "target":
            index[0][0].text = SITE_URL + "v0.0.1/sitemap.xml"
        elif corruption == "entry":
            index.remove(index[0])
        else:
            index.append(index[0])
        tree.write(sitemap, encoding="utf-8", xml_declaration=True)
    with pytest.raises(ValueError, match="sitemap"):
        check_archive(archive)
