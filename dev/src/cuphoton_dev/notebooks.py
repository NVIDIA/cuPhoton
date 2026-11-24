# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Stage GPU reference outputs as native chapters of the CPU guide."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

from cuphoton_dev.build import git
from cuphoton_dev.notebook_store import (
    CHAPTERS,
    digest_file,
    get_blob,
    put_blob,
)

COMPONENTS = {
    "model-sky": ("xrep", "xpois", "xfit"),
    "model-scene": ("ximage", "xscan"),
    "model-alerts": ("xscan", "xpois", "xfit", "xrep"),
}


# Catch residual machine paths and credential markers without printing them.
_PRIVATE = re.compile(
    rb"/(?:home|Users|mnt)/|/tmp/tmp[\w.-]{6,}|file:///"
    rb"|-----BEGIN [A-Z ]*PRIVATE KEY-----"
    rb"|\bgh[pousr]_[a-zA-Z0-9]{25,}|\bgithub_pat_[a-zA-Z0-9_]{30,}"
)


def _check_reference(
    checkout: Path, name: str, revision: str, notebook: Path, receipt: Path
) -> dict[str, Any]:
    if name not in CHAPTERS or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Invalid reference notebook name or source revision")
    for file in (notebook, receipt):
        if _PRIVATE.search(file.read_bytes()):
            raise ValueError(
                f"Private path or credential marker in {file.name}"
            )
    record = json.loads(receipt.read_text(encoding="utf-8"))
    if record["source_revision"] != revision or record[
        "public_notebook_sha256"
    ] != digest_file(notebook):
        raise ValueError(f"{name}: execution receipt and notebook disagree")
    if name == "model-scene":
        passed = (
            record.get("completed") is True
            and record.get("profile") == "reference"
        )
    else:
        original = record["output_sha256"][f"{name}.executed.ipynb"]
        if original != record["public_notebook_sha256"]:
            raise ValueError(f"{name}: original execution hash mismatch")
        passed = record.get("status") == "passed"
        if name == "model-alerts":
            passed = passed and record.get("profile") == "full-reference"
    if not passed:
        raise ValueError(
            f"{name}: successful full reference execution required"
        )

    # The reference can span documentation-only commits. Require identical
    # scientific sources, dependency declarations and captured example inputs.
    example = Path("examples") / name
    hashes = record.get("source_sha256", record.get("source_hashes", {}))
    canonical_name = f"{name}.ipynb"
    if canonical_name not in hashes:
        raise ValueError(f"{name}: source notebook missing from receipt")
    inputs = [
        "uv.lock",
        "setup.py",
        "pyproject.toml",
        "src/cuphoton/core",
        ":(top,glob)src/cuphoton/*.py",
        *(f"src/cuphoton/{part}" for part in COMPONENTS[name]),
    ]
    for file, expected in hashes.items():
        relative = Path(file)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != file
        ):
            raise ValueError(f"Invalid reference input: {file}")
        # Some receipts also cover instructions and tests.
        if (
            file in {"README.md", "PUBLISHING.md"}
            or relative.parts[0] == "tests"
        ):
            continue
        if digest_file(checkout / example / file) != expected:
            raise ValueError(f"{name}: reference input changed: {file}")
    if git(checkout, "diff", "--name-only", revision, "HEAD", "--", *inputs):
        raise ValueError(
            f"{name}: cuPhoton execution inputs changed since {revision}"
        )

    source: dict[str, Any] = json.loads(
        (checkout / example / canonical_name).read_text()
    )
    executed = json.loads(notebook.read_text())
    source_code = [c for c in source["cells"] if c["cell_type"] == "code"]
    executed_code = [c for c in executed["cells"] if c["cell_type"] == "code"]
    for authored, result in zip(source_code, executed_code, strict=True):
        expected_code = "".join(authored["source"])
        if name == "model-sky":
            for flag in ("RUN_BASELINE_EXERCISE", "RUN_BLEND_EXTENSION"):
                expected_code = expected_code.replace(
                    f"{flag} = False", f"{flag} = True"
                )
        if (
            "".join(result["source"]) != expected_code
            or result["execution_count"] is None
            or any(o["output_type"] == "error" for o in result["outputs"])
        ):
            raise ValueError(
                f"{name}: cell differs from its reference profile"
            )
        authored.update(
            source=result["source"],
            outputs=result["outputs"],
            execution_count=result["execution_count"],
        )
    return source


def _stage_chapter(
    checkout: Path, guide: Path, name: str, entry: dict[str, str]
) -> None:
    example = checkout / "examples" / name
    downloads = guide / "source"
    downloads.mkdir(exist_ok=True)
    notebook = downloads / f"{name}.executed.ipynb"
    receipt = downloads / f"{name}-execution.json"
    get_blob(entry["notebook_sha256"], notebook)
    get_blob(entry["receipt_sha256"], receipt)
    source = _check_reference(
        checkout, name, entry["source_revision"], notebook, receipt
    )

    assets = guide / "assets" / name
    shutil.copytree(
        example / "assets", assets, ignore=shutil.ignore_patterns("*.html")
    )
    includes = guide.parent / f"{name}-includes"
    includes.mkdir()
    first = source["cells"][0]
    _, header, body = "".join(first["source"]).split("---\n", 2)
    metadata = yaml.safe_load(header)
    html = metadata.setdefault("format", {}).setdefault("html", {})
    # Preserve the interactive plots when readers save a chapter offline.
    html.update({"embed-resources": True, "html-math-method": "mathml"})
    if name == "model-scene":
        html["css"] = f"assets/{name}/scene.css"
        (includes / "include.html").write_text(
            f"<script>\n{(assets / 'scene.js').read_text()}\n</script>\n"
        )
        html["include-after-body"] = f"../{includes.name}/include.html"
    else:
        html["css"] = f"assets/{name}/guide.css"
        for option, file in (
            ("include-in-header", "guide-head.html"),
            ("include-after-body", "guide-include.html"),
        ):
            shutil.copyfile(example / "assets" / file, includes / file)
            html[option] = f"../{includes.name}/{file}"
    if name != "model-scene":
        metadata["number-sections"] = False
    metadata["execute"] = {"enabled": False, "warning": True}
    metadata["code-links"] = [
        {
            "text": "Notebook source",
            "href": f"https://github.com/NVIDIA/cuPhoton/blob/main/examples/{name}/{name}.ipynb",
        },
        {
            "text": "Download executed notebook",
            "href": f"source/{notebook.name}",
        },
        {"text": "Execution receipt", "href": f"source/{receipt.name}"},
    ]
    revision = entry["source_revision"]
    body = (
        f"GPU reference run from [cuPhoton {revision[:8]}]"
        f"(https://github.com/NVIDIA/cuPhoton/commit/{revision}).\n\n"
        + body.lstrip()
    )
    first["source"] = (
        "---\n" + yaml.safe_dump(metadata, sort_keys=False) + "---\n\n" + body
    )
    for cell in source["cells"]:
        if cell["cell_type"] == "markdown":
            markdown = "".join(cell["source"]).replace(
                'src="assets/story-', f'src="assets/{name}/story-'
            )
            for peer in CHAPTERS:
                # Replace peer review links with links into this guide.
                markdown = re.sub(
                    rf"https://github\.com/NVIDIA/cuPhoton/blob/codex/[^)]+?/examples/{peer}/{peer}\.ipynb",
                    f"{peer}.html",
                    markdown,
                )
            # Quarto reads the on-disk Jupyter array of source lines.
            cell["source"] = markdown.splitlines(keepends=True)
    (guide / f"{name}.ipynb").write_text(json.dumps(source), encoding="utf-8")


def stage_notebooks(checkout: Path, guide: Path) -> None:
    """Render from a copy so the selected source stays byte-identical."""
    original = checkout / "docs/guide"
    entries = json.loads((original / "notebooks.json").read_text())
    if set(entries) != set(CHAPTERS):
        raise ValueError("Expected all three notebook reference records")
    shutil.copytree(original, guide)
    # The staged project is outside docs/guide. CPU chapters still load
    # their tracked example configurations from this exact checkout.
    (guide / "_environment").write_text(f"CUPHOTON_REPO={checkout}\n")
    for name in CHAPTERS:
        _stage_chapter(checkout, guide, name, entries[name])
    config_file = guide / "_quarto.yml"
    config = yaml.safe_load(config_file.read_text())
    config["book"]["chapters"].append(
        {
            "part": "Science notebooks",
            "chapters": [f"{n}.ipynb" for n in CHAPTERS],
        }
    )
    config["project"]["resources"].append("source/**")
    config["science-notebooks"] = True
    config_file.write_text(yaml.safe_dump(config, sort_keys=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    record = parser.add_subparsers(dest="command", required=True).add_parser(
        "record"
    )
    record.add_argument("--name", choices=CHAPTERS, required=True)
    record.add_argument("--revision", required=True)
    record.add_argument("--notebook", type=Path, required=True)
    record.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    checkout = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True
        ).strip()
    )
    _check_reference(
        checkout, args.name, args.revision, args.notebook, args.receipt
    )
    entry = {"source_revision": args.revision}
    for kind in ("notebook", "receipt"):
        path = getattr(args, kind)
        digest = digest_file(path)
        put_blob(path, digest)
        entry[f"{kind}_sha256"] = digest
    print(json.dumps({args.name: entry}, indent=2))


if __name__ == "__main__":
    main()
