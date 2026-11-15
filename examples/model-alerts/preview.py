# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render a local guide preview with a completed Model Alerts reference.

The source notebook stays output-free. This stages its executed counterpart
in a copy of the guide, leaving the checkout and publication workflows alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote
from zipfile import ZIP_DEFLATED, ZipFile

import nbformat
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--site-url", default="https://nvidia.github.io/cuPhoton/"
    )
    parser.add_argument(
        "--source-ref",
        default="main",
        help="GitHub branch, tag or commit for the visible notebook link",
    )
    parser.add_argument(
        "--allow-smoke",
        action="store_true",
        help="Render an explicitly labeled API-only smoke preview",
    )
    options = parser.parse_args()
    output = options.output.resolve()
    if output == HERE or output.is_relative_to(HERE):
        parser.error("--output must be outside the example source directory")
    if output.exists() and any(output.iterdir()):
        parser.error("--output must be absent or empty")

    executed = options.executed.resolve()
    receipt = json.loads(executed.with_name("execution.json").read_text())
    if receipt["status"] != "passed":
        raise ValueError("The reference execution has not passed")
    if receipt["profile"] != "full-reference" and not options.allow_smoke:
        raise ValueError(
            "scientific preview requires full-reference; "
            "use --allow-smoke only for API review"
        )
    expected_notebook = receipt["output_sha256"][executed.name]
    if hashlib.sha256(executed.read_bytes()).hexdigest() != expected_notebook:
        raise ValueError("Executed notebook does not match its receipt")
    for name, expected in receipt["source_sha256"].items():
        actual = hashlib.sha256((HERE / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Reference source no longer matches: {name}")
    notebook = nbformat.read(executed, as_version=4)
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    if len(code) != 14 or any(cell.execution_count is None for cell in code):
        raise ValueError("A completed 14-cell reference is required")
    if any(o.output_type == "error" for c in code for o in c.outputs):
        raise ValueError("Reference contains an execution error")
    if (
        sum(
            "application/vnd.bokehjs_exec.v0+json" in o.get("data", {})
            for c in code
            for o in c.outputs
        )
        != 3
    ):
        raise ValueError("Reference must contain three interactive documents")

    guide = output / "guide"
    shutil.copytree(
        REPO / "docs/guide",
        guide,
        ignore=shutil.ignore_patterns(
            "_book", "_freeze", ".quarto", "__pycache__"
        ),
    )
    shutil.copytree(
        HERE / "assets",
        guide / "assets/model-alerts",
        ignore=shutil.ignore_patterns("*.html"),
    )
    includes = output / "includes"
    includes.mkdir()
    for name in ("guide-head.html", "guide-include.html"):
        shutil.copy2(HERE / "assets" / name, includes / name)
    (guide / "_environment").write_text(f"CUPHOTON_REPO={REPO}\n")
    config_path = guide / "_quarto.yml"
    config = yaml.safe_load(config_path.read_text())
    config["book"]["site-url"] = options.site_url
    config["book"]["chapters"].insert(
        2, {"part": "Examples", "chapters": ["model-alerts.ipynb"]}
    )
    config["project"].setdefault("resources", []).append("source/**")
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    index = guide / "index.qmd"
    index.write_text(
        index.read_text().replace(
            "## About this guide",
            "## Model Alerts\n\n"
            "[Which changes deserve attention?](model-alerts.ipynb): "
            "generate independent fields, train fresh classifiers, "
            "and inspect "
            "a modeled human review queue. The notebook includes executed "
            "cell outputs, interactive figures, terminology definitions "
            "and expandable scientific explanations.\n\n"
            "## About this guide",
        )
    )

    for cell in notebook.cells:
        if cell.cell_type == "markdown":
            cell.source = cell.source.replace(
                "assets/story-", "assets/model-alerts/story-"
            )
    first = notebook.cells[0]
    match = re.match(r"\A---\n(.*?)\n---\n", first.source, flags=re.S)
    metadata = yaml.safe_load(match[1]) if match else {}
    body = first.source[match.end() :] if match else first.source
    body = re.sub(
        r"\[View the source notebook in NVIDIA/cuPhoton\]"
        r"\([^)]*examples/model-alerts/model-alerts\.ipynb\)\.?\n*",
        "",
        body,
    )
    if receipt["profile"] != "full-reference":
        body = (
            "::: {.callout-warning}\nAPI smoke preview only. "
            "These small runs do not qualify scientific performance.\n:::\n\n"
            + body
        )
    metadata.update(
        title="Model Alerts", subtitle="Which changes deserve attention?"
    )
    metadata["execute"] = {"enabled": False, "warning": True}
    metadata["number-sections"] = False
    metadata["format"] = {
        "html": {
            "css": "assets/model-alerts/guide.css",
            "include-in-header": "../includes/guide-head.html",
            "include-after-body": "../includes/guide-include.html",
            "embed-resources": True,
            "html-math-method": "mathml",
        }
    }
    metadata["code-links"] = [
        {
            "text": "Download source notebook",
            "icon": "file-code",
            "href": "source/model-alerts/model-alerts.ipynb",
        },
        {
            "text": "Download example sources",
            "icon": "download",
            "href": "source/model-alerts.zip",
        },
        {
            "text": "cuPhoton on GitHub",
            "icon": "github",
            "href": "https://github.com/NVIDIA/cuPhoton",
        },
    ]
    source_link = (
        "[View the source Jupyter notebook in NVIDIA/cuPhoton]"
        "(https://github.com/NVIDIA/cuPhoton/blob/"
        f"{quote(options.source_ref, safe='')}/"
        "examples/model-alerts/model-alerts.ipynb).\n\n"
    )
    first.source = (
        "---\n"
        + yaml.safe_dump(metadata, sort_keys=False)
        + "---\n\n"
        + source_link
        + body.lstrip()
    )
    nbformat.write(notebook, guide / "model-alerts.ipynb")

    sources = output / "example-sources/model-alerts"
    shutil.copytree(
        HERE,
        sources,
        ignore=shutil.ignore_patterns(
            "__pycache__",
            ".pytest_cache",
            ".ruff_cache",
            "data",
            "results",
            "*.executed.ipynb",
            "model-alerts-output",
            "model-alerts-guide-output",
        ),
    )
    shutil.copy2(REPO / "LICENSE", sources / "LICENSE")
    # HTML includes are build inputs, not standalone documentation pages.
    shutil.copytree(
        sources,
        guide / "source/model-alerts",
        ignore=shutil.ignore_patterns("*.html"),
    )
    with ZipFile(
        guide / "source/model-alerts.zip", "w", ZIP_DEFLATED
    ) as archive:
        for path in sorted(sources.rglob("*")):
            if path.is_file():
                archive.write(
                    path,
                    "model-alerts/" + path.relative_to(sources).as_posix(),
                )

    env = os.environ.copy()
    env.update(
        QUARTO_PYTHON=sys.executable,
        CUPHOTON_REPO=str(REPO),
        CUPHOTON_XREP_TORCH_DEVICE="cpu",
        CUDA_VISIBLE_DEVICES="",
        MPLBACKEND="Agg",
        PATH=str(Path(sys.executable).parent) + os.pathsep + env["PATH"],
    )
    # QMD CPU examples execute; the prepared IPYNB retains its saved outputs.
    subprocess.run(
        ["quarto", "render", str(guide), "--no-cache"], env=env, check=True
    )
    # Reuse current developer transforms/checks without changing publication.
    sys.path.insert(0, str(REPO / "dev/src"))
    from cuphoton_dev.publisher import docs_check
    from cuphoton_dev.transforms import postprocess

    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    postprocess(
        guide / "_book",
        version="main",
        source_sha=revision,
        controller_sha=revision,
        site_url=options.site_url,
        renderer="quarto "
        + subprocess.check_output(["quarto", "--version"], text=True).strip(),
    )
    checked = docs_check(guide / "_book")
    (output / "docs-check.json").write_text(
        json.dumps(checked, indent=2) + "\n"
    )
    print(guide / "_book/model-alerts.html")


if __name__ == "__main__":
    main()
