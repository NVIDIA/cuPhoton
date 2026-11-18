# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Export an executed notebook and CPU guide as a local static preview."""

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


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-ref", default="main")
    parser.add_argument("--controller-python", default=sys.executable)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("Use an absent destination")
    output = args.output.resolve()
    if output.is_relative_to(REPO):
        parser.error("Build outside the source checkout")
    executed = args.executed.resolve()
    receipt = json.loads(executed.with_name("execution.json").read_text())
    if not receipt.get("completed"):
        raise ValueError("A completed execution receipt is required")
    if digest(executed) != receipt["artifact_hashes"][executed.name]:
        raise ValueError("Executed notebook hash mismatch")
    for name, expected in receipt["source_hashes"].items():
        if digest(HERE / name) != expected:
            raise ValueError(f"Changed source since execution: {name}")
    notebook = nbformat.read(executed, as_version=4)
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    if any(cell.execution_count is None for cell in code):
        raise ValueError("Incomplete notebook execution")
    if any(o.output_type == "error" for cell in code for o in cell.outputs):
        raise ValueError("Notebook contains execution errors")
    guide = output / "guide"
    shutil.copytree(
        REPO / "docs/guide",
        guide,
        ignore=shutil.ignore_patterns(
            "_book", "_freeze", ".quarto", "__pycache__"
        ),
    )
    # Includes are build inputs outside the guide resource tree.
    includes = output / "includes"
    includes.mkdir()
    (includes / "scene.js.html").write_text(
        "<script>\n"
        + (HERE / "assets/scene.js").read_text()
        + "\n</script>\n"
    )
    (guide / "assets/model-scene").mkdir()
    shutil.copy2(
        HERE / "assets/scene.css", guide / "assets/model-scene/scene.css"
    )
    config_path = guide / "_quarto.yml"
    config = yaml.safe_load(config_path.read_text())
    config["book"]["chapters"].insert(
        2, {"part": "Examples", "chapters": ["model-scene.ipynb"]}
    )
    config["project"].setdefault("resources", []).append("source/**")
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    first = notebook.cells[0]
    match = re.match(r"\A---\n(.*?)\n---\n", first.source, re.S)
    metadata = yaml.safe_load(match[1])
    metadata["execute"] = {"enabled": False, "warning": True}
    metadata["format"]["html"].update(
        {
            "css": "assets/model-scene/scene.css",
            "include-after-body": "../includes/scene.js.html",
        }
    )
    first.source = (
        "---\n"
        + yaml.safe_dump(metadata, sort_keys=False)
        + "---\n"
        + first.source[match.end() :]
    )
    source_url = (
        "https://github.com/NVIDIA/cuPhoton/blob/"
        + quote(args.source_ref, safe="")
        + "/examples/model-scene/model-scene.ipynb"
    )
    for cell in notebook.cells:
        if cell.cell_type == "markdown":
            cell.source = cell.source.replace(
                "https://github.com/NVIDIA/cuPhoton/blob/main/examples/model-scene/model-scene.ipynb",
                source_url,
            )
        for item in cell.get("outputs", []):
            if item.output_type == "stream":
                text = item.text
                text = text.replace(str(executed.parent), ".")
                text = text.replace(str(REPO), "<source-checkout>")
                # Warnings retain their message and source-relative filename.
                text = re.sub(
                    r"/[^\s\"\']*/site-packages/",
                    "<environment>/site-packages/",
                    text,
                )
                cell.outputs[cell.outputs.index(item)].text = text
    downloads = guide / "source"
    downloads.mkdir(exist_ok=True)
    public_notebook = downloads / "model-scene.executed.ipynb"
    nbformat.write(notebook, public_notebook)
    public_receipt = {
        "profile": receipt["profile"],
        "completed": True,
        "source_revision": receipt["source_revision"],
        "source_hashes": receipt["source_hashes"],
        "protocol_sha256": receipt["protocol_sha256"],
        "packages": receipt["packages"],
        "architecture": receipt["architecture"],
        "gpu": receipt.get("gpu"),
        "device": receipt["device"],
        "science_seeds": receipt["science_seeds"],
        "calibration_seeds": receipt["calibration_seeds"],
        "source_import": receipt["source_import"],
        "source_tree_clean": receipt["source_tree_clean"],
        "artifact_hashes": receipt["artifact_hashes"],
        "runtime_cuphoton_version": receipt.get("runtime_cuphoton_version"),
        "original_receipt_sha256": digest(
            executed.with_name("execution.json")
        ),
        "public_notebook_sha256": digest(public_notebook),
        "notebook_sanitization": "local paths replaced in text streams",
    }
    (downloads / "model-scene-execution.json").write_text(
        json.dumps(public_receipt, indent=2) + "\n"
    )
    first.source += (
        "\n\n[Download the executed notebook]"
        "(source/model-scene.executed.ipynb) · "
        "[Public execution receipt]"
        "(source/model-scene-execution.json)\n"
    )
    nbformat.write(notebook, guide / "model-scene.ipynb")
    sources = guide / "source/model-scene"
    for name in receipt["source_hashes"]:
        target = sources / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(HERE / name, target)
    shutil.copy2(REPO / "LICENSE", sources / "LICENSE")
    with ZipFile(
        guide / "source/model-scene.zip", "w", ZIP_DEFLATED
    ) as archive:
        for path in sources.rglob("*"):
            if path.is_file():
                archive.write(
                    path,
                    "model-scene/" + path.relative_to(sources).as_posix(),
                )
    env = os.environ.copy()
    env.update(
        QUARTO_PYTHON=sys.executable,
        CUPHOTON_REPO=str(REPO),
        CUPHOTON_XREP_TORCH_DEVICE="cpu",
        CUDA_VISIBLE_DEVICES="",
        MPLBACKEND="Agg",
    )
    # CPU chapters execute; the IPYNB preserves saved GPU outputs.
    subprocess.run(
        ["quarto", "render", str(guide), "--no-cache"], env=env, check=True
    )
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    controller = """
import sys
from pathlib import Path
from cuphoton_dev.transforms import postprocess, check_snapshot
root = Path(sys.argv[1])
postprocess(root, version="main", source_sha=sys.argv[2],
            controller_sha=sys.argv[2], renderer=sys.argv[3],
            lock_sha256=sys.argv[4])
check_snapshot(root)
"""
    env["PYTHONPATH"] = (
        str(REPO / "dev/src") + os.pathsep + env.get("PYTHONPATH", "")
    )
    subprocess.run(
        [
            args.controller_python,
            "-c",
            controller,
            str(guide / "_book"),
            revision,
            subprocess.check_output(
                ["quarto", "--version"], text=True
            ).strip(),
            digest(REPO / "uv.lock"),
        ],
        env=env,
        check=True,
    )
    print(guide / "_book/model-scene.html")


if __name__ == "__main__":
    main()
