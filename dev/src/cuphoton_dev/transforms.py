# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Convert rendered guide sources into self-contained snapshots."""

from __future__ import annotations

import hashlib
import json
import posixpath
import re
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit, urlunsplit
from xml.etree import ElementTree as ET

from bs4 import BeautifulSoup, NavigableString

SITE_URL = "https://nvidia.github.io/cuPhoton/"
RELEASE = re.compile(r"v(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)\Z")
RECEIPT = "build.json"
_INSTALL = (
    "This clones the development branch, which this guide describes. "
    "To use a published version instead, pick its tag from Releases, "
    "add --branch TAG to the clone command, and use the documentation "
    "in that checkout."
)
_REFERENCE = re.compile(
    r"The links point to the repository[’']s main branch, "
    r"which can be newer than this guide\."
)
_CLONE = "git clone https://github.com/NVIDIA/cuPhoton.git"
_REQUIRED = {
    "index.html",
    "install.html",
    "reference.html",
    "search.json",
    "sitemap.xml",
}


def validate_version(version: str) -> None:
    if version != "main" and not RELEASE.fullmatch(version):
        raise ValueError(f"Not a documentation version: {version!r}")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def file_hashes(path: Path) -> dict[str, str]:
    """Hash regular files only; Pages artifacts cannot contain links."""
    hashes = {}
    for file in sorted(path.rglob("*")):
        if file.is_symlink():
            raise ValueError(f"Snapshot contains a symbolic link: {file}")
        if file.is_file():
            if file.stat().st_nlink != 1:
                raise ValueError(f"Snapshot contains a hard link: {file}")
            relative = file.relative_to(path).as_posix()
            if relative != RECEIPT:
                hashes[relative] = hashlib.sha256(
                    file.read_bytes()
                ).hexdigest()
    return hashes


def _owned_url(value: str, version: str, site_url: str) -> str:
    url = urlsplit(value)
    if url.netloc.lower() in {"github.com", "raw.githubusercontent.com"}:
        parts = url.path.split("/")
        if len(parts) >= 4 and [p.lower() for p in parts[1:3]] == [
            "nvidia",
            "cuphoton",
        ]:
            ref_index = 3 if url.netloc.lower().startswith("raw.") else 4
            if (
                len(parts) > ref_index
                and parts[ref_index] == "main"
                and (
                    ref_index == 3
                    or parts[3] in {"blob", "tree", "raw", "edit"}
                )
            ):
                parts[ref_index] = version
                return urlunsplit(url._replace(path="/".join(parts)))
    base = urlsplit(site_url)
    if url.netloc == base.netloc and url.path.startswith(base.path):
        tail = url.path[len(base.path) :]
        if not tail.startswith(version + "/"):
            return urlunsplit(
                url._replace(path=base.path + version + "/" + tail)
            )
    # Quarto may use root-relative navigation and asset references.
    if not url.netloc and url.path.startswith(base.path) and base.path != "/":
        tail = url.path[len(base.path) :]
        if not tail.startswith(version + "/"):
            return urlunsplit(
                url._replace(path=base.path + version + "/" + tail)
            )
    return value


def _release_text(value: str, version: str) -> str:
    value = value.replace(
        _INSTALL,
        f"This clones release {version}, which this guide describes.",
    )
    value = _REFERENCE.sub(
        f"The links point to the repository's {version} tag, "
        "matching this guide.",
        value,
    )
    return value.replace(
        _CLONE,
        f"git clone --branch {version} --single-branch "
        "https://github.com/NVIDIA/cuPhoton.git",
    )


def _transform_json(value: Any, version: str, site_url: str) -> Any:
    if isinstance(value, str):
        value = _owned_url(value, version, site_url)
        return _release_text(value, version) if version != "main" else value
    if isinstance(value, list):
        return [_transform_json(v, version, site_url) for v in value]
    if isinstance(value, dict):
        return {
            k: _transform_json(v, version, site_url) for k, v in value.items()
        }
    return value


def _assert_release_text(text: str, filename: str) -> None:
    normalized = " ".join(text.split())
    stale = (
        "This clones the development branch",
        "--branch TAG",
        "which can be newer than this guide",
        _CLONE,
    )
    if any(term in normalized for term in stale) or re.search(
        r"https://(?:github\.com/NVIDIA/cuPhoton/(?:blob|tree|raw|edit)|"
        r"raw\.githubusercontent\.com/NVIDIA/cuPhoton)/main(?:/|[\"\s])",
        text,
        re.IGNORECASE,
    ):
        raise ValueError(
            f"Unconverted release instructions or links: {filename}"
        )


def postprocess(
    output: Path,
    *,
    version: str,
    source_sha: str,
    controller_sha: str,
    renderer: str,
    site_url: str = SITE_URL,
    lock_sha256: str | None = None,
) -> dict[str, Any]:
    """Transform an isolated Quarto output, then record its validated bytes.

    Call only on freshly rendered output. Release transformation contracts
    deliberately fail when known source instructions change, requiring the
    controller and its fixtures to be reviewed together.
    """
    output = Path(output)
    validate_version(version)
    site_url = site_url.rstrip("/") + "/"
    if (output / RECEIPT).exists():
        raise ValueError(
            "Output is already postprocessed; render a fresh tree"
        )
    missing = _REQUIRED - set(file_hashes(output))
    if missing:
        raise ValueError(f"Incomplete rendered guide: {sorted(missing)}")
    pages: dict[str, list[str]] = {}
    converted = {"clone": 0, "install": 0, "reference": 0}
    for file in sorted(output.rglob("*.html")):
        relative = file.relative_to(output).as_posix()
        soup = BeautifulSoup(file.read_text(encoding="utf-8"), "html.parser")
        if soup.head is None or soup.body is None:
            raise ValueError(f"Missing HTML head/body: {relative}")
        for node in soup.find_all(True):
            for attr in ("href", "src", "content", "data-url"):
                value = node.get(attr)
                if isinstance(value, str):
                    node[attr] = _owned_url(value, version, site_url)
        if version != "main":
            for paragraph in soup.find_all("p"):
                text = " ".join(paragraph.get_text().split())
                if _INSTALL in text:
                    paragraph.clear()
                    paragraph.append(_release_text(text, version))
                    converted["install"] += 1
                elif _REFERENCE.search(text):
                    paragraph.clear()
                    paragraph.append(_release_text(text, version))
                    converted["reference"] += 1
            for code in soup.find_all("code"):
                if _CLONE in code.get_text():
                    # Preserve Quarto's line IDs and syntax highlighting.
                    for text in list(code.find_all(string=True)):
                        if (
                            " clone https://github.com/"
                            "NVIDIA/cuPhoton.git" in text
                        ):
                            text.replace_with(
                                str(text).replace(
                                    " clone https://github.com/"
                                    "NVIDIA/cuPhoton.git",
                                    f" clone --branch {version} "
                                    "--single-branch "
                                    "https://github.com/NVIDIA/cuPhoton.git",
                                )
                            )
                            converted["clone"] += 1
                        elif _CLONE in text:
                            text.replace_with(
                                _release_text(str(text), version)
                            )
                            converted["clone"] += 1
            _assert_release_text(str(soup), relative)
        for old in soup.select('link[rel="canonical"]'):
            old.decompose()
        canonical = soup.new_tag(
            "link", rel="canonical", href=site_url + version + "/" + relative
        )
        soup.head.append(canonical)
        for name, value in (
            ("cuphoton-version", version),
            ("cuphoton-source", source_sha),
        ):
            soup.head.append(
                soup.new_tag("meta", attrs={"name": name, "content": value})
            )
        label = "main (development)" if version == "main" else version
        if soup.title:
            soup.title.append(NavigableString(" · " + label))
        depth = len(Path(relative).parts) - 1
        prefix = "../" * depth
        shared = "../" * (depth + 1)
        soup.head.append(
            soup.new_tag(
                "link",
                rel="stylesheet",
                href=prefix + "_version/selector.css",
            )
        )
        script = soup.new_tag(
            "script", src=prefix + "_version/selector.js", defer=""
        )
        script["data-catalog"] = shared + "versions.json"
        script["data-page"] = relative
        script["data-version"] = version
        soup.head.append(script)
        banner = soup.new_tag(
            "nav",
            attrs={
                "class": "cuphoton-version",
                "aria-label": "Documentation version",
            },
        )
        label_tag = soup.new_tag(
            "label", attrs={"for": "cuphoton-version-select"}
        )
        label_tag.append("Documentation: ")
        banner.append(label_tag)
        select = soup.new_tag(
            "select", id="cuphoton-version-select", disabled=""
        )
        option = soup.new_tag("option", value=version)
        option.string = label
        select.append(option)
        banner.append(select)
        fallback = soup.new_tag("a", href=shared + "versions.html")
        fallback.string = "Versions"
        banner.append(fallback)
        notice = soup.new_tag("span", id="cuphoton-version-notice")
        notice.string = (
            "Development documentation."
            if version == "main"
            else f"Documentation for release {version}."
        )
        banner.append(notice)
        # Inside main content avoids obscuring the fixed Quarto navigation.
        container = soup.find("main") or soup.body
        container.insert(0, banner)
        pages[relative] = sorted(
            {str(tag["id"]) for tag in soup.find_all(id=True)}
        )
        file.write_text(str(soup), encoding="utf-8")
    if version != "main" and converted != {
        "clone": 1,
        "install": 1,
        "reference": 1,
    }:
        raise ValueError(f"Release conversion contracts changed: {converted}")
    search_path = output / "search.json"
    search = _transform_json(
        json.loads(search_path.read_text(encoding="utf-8")), version, site_url
    )
    _write_json(search_path, search)
    if version != "main":
        _assert_release_text(
            search_path.read_text(encoding="utf-8"), "search.json"
        )
    sitemap_path = output / "sitemap.xml"
    tree = ET.parse(sitemap_path)
    for location in tree.iter():
        if location.tag.rsplit("}", 1)[-1] == "loc" and location.text:
            location.text = _owned_url(location.text, version, site_url)
    tree.write(sitemap_path, encoding="utf-8", xml_declaration=True)
    asset_dir = output / "_version"
    asset_dir.mkdir(exist_ok=True)
    for asset in (Path(__file__).parent / "assets").iterdir():
        shutil.copyfile(asset, asset_dir / asset.name)
    receipt = {
        "schema": 1,
        "version": version,
        "source_sha": source_sha,
        "controller_sha": controller_sha,
        "renderer": renderer,
        "lock_sha256": lock_sha256,
        "site_url": site_url,
        "pages": pages,
        "files": file_hashes(output),
    }
    _write_json(output / RECEIPT, receipt)
    return check_snapshot(output)


def _local_target(
    value: str, relative: str, receipt: dict[str, Any]
) -> str | None:
    url = urlsplit(value)
    base = urlsplit(receipt["site_url"])
    prefix = base.path + receipt["version"] + "/"
    if url.scheme in {"mailto", "tel", "data", "javascript"}:
        return None
    if url.netloc:
        if url.netloc != base.netloc or not url.path.startswith(base.path):
            return None
        if not url.path.startswith(prefix):
            raise ValueError(f"Link escapes selected version: {value}")
        target = unquote(url.path[len(prefix) :])
    elif url.path.startswith("/"):
        if not url.path.startswith(prefix):
            raise ValueError(f"Link escapes selected version: {value}")
        target = unquote(url.path[len(prefix) :])
    elif not url.path:
        return relative
    else:
        target = posixpath.join(
            posixpath.dirname(relative), unquote(url.path)
        )
    # Keep directory URL semantics before normpath removes the trailing slash.
    directory = (
        not target
        or target.endswith(("/", "/.", "/.."))
        or target in {".", ".."}
    )
    target = posixpath.normpath(target)
    if target == ".." or target.startswith(("../", "/")):
        raise ValueError(f"Link escapes selected version: {value}")
    if directory:
        return posixpath.normpath(posixpath.join(target, "index.html"))
    return target


def check_snapshot(path: Path) -> dict[str, Any]:
    """Check receipt integrity and the browser-facing version boundary."""
    path = Path(path)
    receipt: dict[str, Any] = json.loads(
        (path / RECEIPT).read_text(encoding="utf-8")
    )
    if receipt.get("schema") != 1:
        raise ValueError("Unsupported documentation receipt schema")
    validate_version(receipt["version"])
    actual = file_hashes(path)
    if actual != receipt["files"]:
        raise ValueError(f"Snapshot hashes differ from receipt: {path}")
    if _REQUIRED - actual.keys():
        raise ValueError("Snapshot is missing required guide output")
    inventory = receipt["pages"]
    if set(inventory) != {f for f in actual if f.endswith(".html")}:
        raise ValueError("Snapshot HTML inventory is incomplete")
    for relative, anchors in inventory.items():
        text = (path / relative).read_text(encoding="utf-8")
        soup = BeautifulSoup(text, "html.parser")
        canonical = soup.select('link[rel="canonical"]')
        expected = receipt["site_url"] + receipt["version"] + "/" + relative
        if len(canonical) != 1 or canonical[0].get("href") != expected:
            raise ValueError(f"Wrong canonical URL: {relative}")
        if sorted({str(t["id"]) for t in soup.find_all(id=True)}) != anchors:
            raise ValueError(f"Wrong anchor inventory: {relative}")
        if receipt["version"] != "main":
            _assert_release_text(text, relative)
            _assert_release_text(soup.get_text(" "), relative)
        for node in soup.find_all(True):
            for attr in ("href", "src"):
                value = node.get(attr)
                if not isinstance(value, str):
                    continue
                # The one deliberate escape is the shared fallback catalog.
                if (
                    node.name == "a"
                    and value
                    == "../" * len(Path(relative).parts) + "versions.html"
                ):
                    continue
                target = _local_target(value, relative, receipt)
                if target is not None and target not in actual:
                    raise ValueError(
                        f"Missing local target in {relative}: {value}"
                    )
                fragment = unquote(urlsplit(value).fragment)
                if (
                    target in inventory
                    and fragment
                    and fragment not in inventory[target]
                ):
                    raise ValueError(
                        f"Missing local bookmark in {relative}: {value}"
                    )
    search_text = (path / "search.json").read_text(encoding="utf-8")
    if receipt["version"] != "main":
        _assert_release_text(search_text, "search.json")
    for record in json.loads(search_text):
        target = _local_target(record["href"], "index.html", receipt)
        if target not in inventory:
            raise ValueError(
                f"Search target is outside snapshot: {record['href']}"
            )
        fragment = unquote(urlsplit(record["href"]).fragment)
        if fragment and fragment not in inventory[target]:
            raise ValueError(f"Search bookmark is missing: {record['href']}")
    locations = {
        element.text
        for element in ET.parse(path / "sitemap.xml").iter()
        if element.tag.rsplit("}", 1)[-1] == "loc"
    }
    expected_locations = {
        receipt["site_url"] + receipt["version"] + "/" + relative
        for relative in inventory
    }
    if locations != expected_locations:
        raise ValueError(
            "Sitemap does not describe the snapshot's HTML pages"
        )
    return receipt
