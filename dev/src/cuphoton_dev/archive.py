# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Retain historical snapshots when assembling a Pages archive."""

from __future__ import annotations

import html
import json
import posixpath
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

from cuphoton_dev.transforms import RELEASE, check_snapshot


def _json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _release_key(version: str) -> tuple[int, ...]:
    match = RELEASE.fullmatch(version)
    if not match:
        raise ValueError(f"Not a formal version: {version}")
    return tuple(int(n) for n in match.groups())


def _snapshots(archive: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for path in sorted(archive.iterdir()):
        if path.name == "main" or RELEASE.fullmatch(path.name):
            if path.is_symlink() or not path.is_dir():
                raise ValueError(f"Invalid snapshot directory: {path}")
            receipt = check_snapshot(path)
            if receipt["version"] != path.name:
                raise ValueError(
                    f"Snapshot directory and receipt disagree: {path}"
                )
            result[path.name] = receipt
    return result


def redirect_html(target: str) -> str:
    """Redirect with a no-JS fallback and query/fragment preservation."""
    escaped = html.escape(target, quote=True)
    # The target is a generated relative path.
    script_target = json.dumps(target).replace("<", "\\u003c")
    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>cuPhoton documentation</title></head><body>"
        f'<p>Continue to <a href="{escaped}">cuPhoton documentation</a>.</p>'
        "<script>const target = new URL("
        + script_target
        + ", location.href); target.search = location.search; "
        "target.hash = location.hash; location.replace(target.href);</script>"
        f'<noscript><meta http-equiv="refresh" content="0;url={escaped}">'
        "</noscript>"
        "</body></html>\n"
    )


def _catalog(
    receipts: dict[str, dict[str, Any]],
    eligible_releases: set[str],
    legacy_pages: set[str] | None = None,
) -> dict[str, Any]:
    releases = sorted(
        (v for v in receipts if v != "main"), key=_release_key, reverse=True
    )
    stable = next((v for v in releases if v in eligible_releases), None)
    preferred = stable or ("main" if "main" in receipts else None)
    if preferred is None:
        raise ValueError("Archive has no main or eligible release snapshot")
    versions = (["main"] if "main" in receipts else []) + releases
    entries = [
        {
            "version": version,
            "path": version + "/",
            "source_sha": receipts[version]["source_sha"],
            "released": version in eligible_releases,
            "pages": receipts[version]["pages"],
        }
        for version in versions
    ]
    pages = sorted(
        {p for receipt in receipts.values() for p in receipt["pages"]}
        | (legacy_pages or set())
    )
    aliases: dict[str, str] = {}
    for page in pages:
        page_path = PurePosixPath(page)
        if (
            page_path.is_absolute()
            or ".." in page_path.parts
            or page_path.as_posix() != page
            or page_path.suffix != ".html"
        ):
            raise ValueError(f"Invalid legacy chapter path: {page}")
        # Never replace a snapshot/catalog directory with a legacy alias.
        first = page.split("/", 1)[0]
        if first in {
            "main",
            "stable",
            "versions.html",
            "versions.json",
        } or RELEASE.fullmatch(first):
            raise ValueError(
                f"Chapter path collides with archive layout: {page}"
            )
        target_page = (
            page if page in receipts[preferred]["pages"] else "index.html"
        )
        target = preferred + "/" + target_page
        for alias in (page, "stable/" + page):
            aliases[alias] = posixpath.relpath(
                target, posixpath.dirname(alias) or "."
            )
    return {
        "schema": 1,
        "stable": stable,
        "preferred": preferred,
        "versions": entries,
        "legacy_pages": pages,
        "aliases": aliases,
    }


def _generate(archive: Path, catalog: dict[str, Any]) -> None:
    _json(archive / "versions.json", catalog)
    for alias, target in catalog["aliases"].items():
        file = archive / alias
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(redirect_html(target), encoding="utf-8")
    items = "".join(
        f'<li><a href="{entry["path"]}index.html">'
        f"{html.escape(entry['version'])}"
        f"{' (stable)' if entry['version'] == catalog['stable'] else ''}"
        "</a></li>"
        for entry in catalog["versions"]
    )
    (archive / "versions.html").write_text(
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>cuPhoton documentation versions</title></head><body>"
        "<main><h1>cuPhoton documentation versions</h1><ul>"
        + items
        + "</ul></main></body></html>\n",
        encoding="utf-8",
    )
    (archive / ".nojekyll").touch()


def assemble(
    archive: Path,
    snapshot: Path,
    *,
    eligible_releases: set[str],
    is_ancestor: Callable[[str, str], bool] | None = None,
) -> dict[str, Any]:
    """Validate and stage a complete update, preserving immutable releases.

    The caller holds the workflow concurrency lock from reading the archive
    through deployment, and commits this generated tree only after success.
    An ancestry callback is mandatory when main changes to another source SHA.
    """
    archive, snapshot = Path(archive), Path(snapshot)
    incoming = check_snapshot(snapshot)
    version = incoming["version"]
    for release in eligible_releases:
        _release_key(release)
    if version != "main" and version not in eligible_releases:
        raise ValueError(
            f"Release is not eligible for publication: {version}"
        )
    archive.mkdir(parents=True, exist_ok=True)
    receipts = _snapshots(archive)
    legacy_pages: set[str] = set()
    if (archive / "versions.json").is_file():
        previous_catalog = json.loads(
            (archive / "versions.json").read_text(encoding="utf-8")
        )
        legacy_pages.update(previous_catalog["legacy_pages"])
    previous = receipts.get(version)
    reuse = False
    if previous:
        if incoming["site_url"] != previous["site_url"]:
            raise ValueError("Snapshot site URL changed")
        if version != "main":
            if incoming["source_sha"] != previous["source_sha"]:
                raise ValueError(
                    f"Tag moved since its retained snapshot: {version}"
                )
            reuse = True
        elif incoming["source_sha"] != previous["source_sha"]:
            if is_ancestor is None or not is_ancestor(
                previous["source_sha"], incoming["source_sha"]
            ):
                raise ValueError(
                    "Stale or unrelated main snapshot would replace "
                    "published main"
                )
    if receipts and incoming["site_url"] not in {
        r["site_url"] for r in receipts.values()
    }:
        raise ValueError("Snapshots disagree about the site URL")
    with tempfile.TemporaryDirectory(
        prefix="cuphoton-archive-", dir=archive.parent
    ) as scratch:
        staged = Path(scratch) / "site"
        shutil.copytree(
            archive, staged, ignore=shutil.ignore_patterns(".git")
        )
        if not reuse:
            destination = staged / version
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(snapshot, destination)
        retained = _snapshots(staged)
        catalog = _catalog(retained, eligible_releases, legacy_pages)
        _generate(staged, catalog)
        check_archive(staged)
        # All validation has completed before the first mutation. Preserve Git
        # metadata and copy ordinary files, which native Pages accepts.
        if not reuse:
            destination = archive / version
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(staged / version, destination)
        _generate(archive, catalog)
    return catalog


def check_archive(path: Path) -> dict[str, Any]:
    """Validate snapshots, catalog, stable choice and legacy aliases."""
    path = Path(path)
    for entry in path.rglob("*"):
        if ".git" in entry.relative_to(path).parts:
            continue
        if entry.is_symlink() or (
            entry.is_file() and entry.stat().st_nlink != 1
        ):
            raise ValueError(f"Archive contains a link: {entry}")
    catalog: dict[str, Any] = json.loads(
        (path / "versions.json").read_text(encoding="utf-8")
    )
    if catalog.get("schema") != 1:
        raise ValueError("Unsupported documentation catalog schema")
    receipts = _snapshots(path)
    eligible = {
        entry["version"] for entry in catalog["versions"] if entry["released"]
    }
    expected = _catalog(receipts, eligible, set(catalog["legacy_pages"]))
    if catalog != expected:
        raise ValueError("Catalog does not match retained snapshots")
    for alias, target in catalog["aliases"].items():
        redirect = path / alias
        if not redirect.is_file() or redirect.read_text(
            encoding="utf-8"
        ) != redirect_html(target):
            raise ValueError(f"Invalid legacy redirect: {alias}")
        resolved = (redirect.parent / target).resolve()
        if (
            not resolved.is_relative_to(path.resolve())
            or not resolved.is_file()
        ):
            raise ValueError(f"Legacy redirect has no target: {alias}")
    if (
        not (path / "versions.html").is_file()
        or not (path / ".nojekyll").is_file()
    ):
        raise ValueError(
            "Archive is missing its fallback catalog or Pages marker"
        )
    return catalog
