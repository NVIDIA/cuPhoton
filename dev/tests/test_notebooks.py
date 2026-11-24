# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from cuphoton_dev import notebooks
from cuphoton_dev.notebook_store import digest_file


def write_json(path, data):
    path.write_text(json.dumps(data))
    return digest_file(path)


def test_mixed_book_uses_the_reviewed_outputs(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    original = checkout / "docs/guide"
    original.mkdir(parents=True)
    (original / "_quarto.yml").write_text(
        'book: {chapters: [index.qmd]}\nproject: {resources: ["assets/**"]}'
    )
    blobs = {}
    entries = {}
    revision = "a" * 40
    for name in notebooks.CHAPTERS:
        example = checkout / "examples" / name
        assets = example / "assets"
        assets.mkdir(parents=True)
        (assets / "scene.js").write_text("// scene")
        for item in ("guide-head.html", "guide-include.html"):
            (assets / item).write_text("<!-- include -->")
        flag = "RUN_BLEND_EXTENSION = False"
        source = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "source": "---\ntitle: Science\n---\nText",
                },
                {
                    "cell_type": "code",
                    "source": flag,
                    "outputs": [],
                    "execution_count": None,
                },
            ],
        }
        source_digest = write_json(example / f"{name}.ipynb", source)
        cell = source["cells"][1]
        cell["execution_count"] = 1
        cell["outputs"] = [{"output_type": "stream", "text": "Reference"}]
        if name == "model-sky":
            cell["source"] = flag.replace("False", "True")
        notebook = tmp_path / f"{name}-input.json"
        nb_digest = write_json(notebook, source)
        receipt = tmp_path / f"{name}-receipt.json"
        rc_digest = write_json(
            receipt,
            {
                "source_revision": revision,
                "public_notebook_sha256": nb_digest,
                "output_sha256": {f"{name}.executed.ipynb": nb_digest},
                "source_hashes": {f"{name}.ipynb": source_digest},
                "status": "passed",
                "completed": True,
                "profile": "reference"
                if name == "model-scene"
                else "full-reference",
            },
        )
        entries[name] = dict(
            source_revision=revision,
            notebook_sha256=nb_digest,
            receipt_sha256=rc_digest,
        )
        blobs[nb_digest] = notebook.read_bytes()
        blobs[rc_digest] = receipt.read_bytes()
    (original / "notebooks.json").write_text(json.dumps(entries))
    monkeypatch.setattr(notebooks, "git", lambda *args: "")
    monkeypatch.setattr(
        notebooks, "get_blob", lambda d, p: p.write_bytes(blobs[d])
    )

    guide = tmp_path / "guide"
    notebooks.stage_notebooks(checkout, guide)
    config = yaml.safe_load((guide / "_quarto.yml").read_text())
    assert config["book"]["chapters"][0] == "index.qmd"
    assert config["book"]["chapters"][1]["chapters"] == [
        f"{n}.ipynb" for n in notebooks.CHAPTERS
    ]
    for name in notebooks.CHAPTERS:
        cells = json.loads((guide / f"{name}.ipynb").read_text())["cells"]
        assert cells[1]["outputs"][0]["text"] == "Reference"
        lines = cells[0]["source"]
        assert isinstance(lines, list) and lines[0] == "---\n"
        assert f"source/{name}.executed.ipynb" in "".join(lines)
        assert (
            json.loads(
                (checkout / "examples" / name / f"{name}.ipynb").read_text()
            )["cells"][1]["outputs"]
            == []
        )
    assert not list((guide / "assets").rglob("*.html"))

    sky = entries["model-sky"]
    notebook = guide / "source/model-sky.executed.ipynb"
    receipt = guide / "source/model-sky-execution.json"
    # Even a re-hashed external notebook cannot substitute a different cell.
    data = json.loads(notebook.read_text())
    data["cells"][1]["source"] = "unreviewed()"
    new_digest = write_json(notebook, data)
    proof = json.loads(receipt.read_text())
    proof["public_notebook_sha256"] = new_digest
    write_json(receipt, proof)
    with pytest.raises(ValueError, match="original execution hash"):
        notebooks._check_reference(
            checkout, "model-sky", sky["source_revision"], notebook, receipt
        )
    proof["output_sha256"]["model-sky.executed.ipynb"] = new_digest
    write_json(receipt, proof)
    with pytest.raises(ValueError, match="reference profile"):
        notebooks._check_reference(
            checkout, "model-sky", sky["source_revision"], notebook, receipt
        )
    monkeypatch.setattr(notebooks, "git", lambda *args: "changed.py")
    with pytest.raises(ValueError, match="execution inputs changed"):
        notebooks._check_reference(
            checkout, "model-sky", sky["source_revision"], notebook, receipt
        )


def test_blob_download_discards_corrupted_response(tmp_path, monkeypatch):
    import subprocess

    from cuphoton_dev import notebook_store

    digest = hashlib.sha256(b"reviewed").hexdigest()

    def corrupt(args, **kwargs):
        Path(args[args.index("--output") + 1]).write_bytes(b"changed")
        return subprocess.CompletedProcess(args, 0, "200", "")

    monkeypatch.setattr(subprocess, "run", corrupt)
    destination = tmp_path / "output"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        notebook_store.get_blob(digest, destination)
    assert not destination.exists()


def test_fresh_read_recovers_a_cached_miss(tmp_path, monkeypatch):
    from cuphoton_dev import notebook_store

    payload = b"qualified"
    digest = hashlib.sha256(payload).hexdigest()
    attempts = []

    def download(key, path, *, fresh=False):
        attempts.append((key, fresh))
        if fresh:
            path.write_bytes(payload)
        return fresh

    monkeypatch.setattr(notebook_store, "_download", download)
    output = tmp_path / "reference"
    notebook_store.get_blob(digest, output)
    assert output.read_bytes() == payload
    assert attempts == [(digest, False), (digest, True)]
