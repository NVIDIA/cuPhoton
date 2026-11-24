# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Prepare a reviewed reference notebook and public receipt for Pages."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
from pathlib import Path

from cuphoton_dev.build import git, resolve
from cuphoton_dev.notebook_store import CHAPTERS, digest_file
from cuphoton_dev.notebooks import _check_reference


def _check_scene_outputs(raw: Path, public: Path) -> None:
    """The Scene preview rewrites Markdown and local paths in text streams."""
    before = json.loads(raw.read_text("utf-8"))["cells"]
    after = json.loads(public.read_text("utf-8"))["cells"]
    for source, exported in zip(before, after, strict=True):
        if (source["cell_type"], source.get("id")) != (
            exported["cell_type"],
            exported.get("id"),
        ):
            raise ValueError(
                "Scene preview changed the notebook cell sequence"
            )
        if source["cell_type"] != "code":
            continue
        if source["execution_count"] != exported["execution_count"]:
            raise ValueError("Scene preview changed execution counts")
        for a, b in zip(source["outputs"], exported["outputs"], strict=True):
            if a == b:
                continue
            if a["output_type"] != "stream" or b["output_type"] != "stream":
                raise ValueError("Scene preview changed a scientific output")
            if a["name"] != b["name"]:
                raise ValueError("Scene preview changed the output stream")
            # Only the preview's documented path substitutions can differ.
            text = "".join(a["text"]).replace(str(raw.parent), ".")
            text = re.sub(
                r"""/[^\s"']*/site-packages/""",
                "<environment>/site-packages/",
                text,
            )
            # A source checkout may have moved since the GPU run.
            text = re.sub(
                r"""/[^\s"']+(?=/(?:examples/model-scene|src/cuphoton)/)""",
                "<source-checkout>",
                text,
            )
            if text != "".join(b["text"]):
                raise ValueError("Scene preview changed a warning or result")


def prepare(
    *,
    checkout: Path,
    name: str,
    revision: str,
    reference: Path,
    output: Path,
    scene_downloads: Path | None = None,
) -> None:
    if name not in CHAPTERS or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Expected notebook name and full execution revision")
    if (name == "model-scene") != (scene_downloads is not None):
        raise ValueError(
            "Model Scene requires its preview's --scene-downloads"
        )
    if output.exists():
        raise ValueError(f"Export directory already exists: {output}")
    reference = reference.resolve()
    raw = reference / f"{name}.executed.ipynb"
    raw_receipt = reference / "execution.json"
    record = json.loads(raw_receipt.read_text("utf-8"))
    hashes = record[
        "artifact_hashes" if name == "model-scene" else "output_sha256"
    ]
    if hashes.get(raw.name) != digest_file(raw):
        raise ValueError(
            "Executed notebook differs from its original receipt"
        )

    notebook = raw
    if scene_downloads is not None:
        notebook = scene_downloads / raw.name
        receipt = json.loads(
            (scene_downloads / f"{name}-execution.json").read_text("utf-8")
        )
        if (
            receipt["original_receipt_sha256"] != digest_file(raw_receipt)
            or receipt["public_notebook_sha256"] != digest_file(notebook)
            or not record["source_tree_clean"]
            or record["source_revision"] != revision
            or any(
                record[k] != receipt[k]
                for k in (
                    "source_hashes",
                    "profile",
                    "completed",
                    "protocol_sha256",
                    "artifact_hashes",
                    "device",
                    "packages",
                )
            )
        ):
            raise ValueError("Scene public download and GPU receipt disagree")
        _check_scene_outputs(raw, notebook)
    else:
        receipt = dict(record)
        if name == "model-sky":
            # Older Sky runners record the installed SCM version. Resolve its
            # abbreviated Git hash (or an exact release tag) to a full commit.
            package = record["environment"]["packages"]["cuphoton"]
            scm = re.search(r"(?:\+|\.)g([0-9a-f]{7,40})(?:\.|$)", package)
            if not scm and not re.fullmatch(
                r"[0-9]+\.[0-9]+\.[0-9]+", package
            ):
                raise ValueError("Model Sky receipt needs a source version")
            ref = scm[1] if scm else f"refs/tags/v{package}"
            if resolve(checkout, ref) != revision:
                raise ValueError("Model Sky cuPhoton revision differs")
            receipt["original_receipt_sha256"] = digest_file(raw_receipt)
        elif record["cuphoton_revision"] != revision:
            raise ValueError("Model Alerts cuPhoton revision differs")
        receipt["source_revision"] = revision
        receipt["public_notebook_sha256"] = digest_file(notebook)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        staged = Path(temp) / "public"
        staged.mkdir()
        public_notebook = staged / raw.name
        public_receipt = staged / f"{name}-execution.json"
        shutil.copyfile(notebook, public_notebook)
        public_receipt.write_text(
            json.dumps(receipt, indent=2, allow_nan=False) + "\n", "utf-8"
        )
        _check_reference(
            checkout, name, revision, public_notebook, public_receipt
        )
        staged.rename(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", choices=CHAPTERS, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--scene-downloads", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(
        checkout=Path(git(Path.cwd(), "rev-parse", "--show-toplevel")),
        name=args.name,
        revision=args.revision,
        reference=args.reference,
        scene_downloads=args.scene_downloads,
        output=args.output,
    )


if __name__ == "__main__":
    main()
