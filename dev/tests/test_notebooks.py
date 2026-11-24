# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import zipfile
from pathlib import Path

import pytest
import yaml

from cuphoton_dev import notebooks
from cuphoton_dev.build import git
from cuphoton_dev.notebook_store import digest_file


def write_json(path, data):
    path.write_text(json.dumps(data))
    return digest_file(path)


@pytest.fixture
def reference_book(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    original = checkout / "docs/guide"
    original.mkdir(parents=True)
    (original / "_quarto.yml").write_text(
        'book: {chapters: [index.qmd]}\nproject: {resources: ["assets/**"]}'
    )
    blobs = {}
    entries = {}
    (checkout / "LICENSE").write_text("example license")
    component = checkout / "src/cuphoton/xfit/photometry.py"
    component.parent.mkdir(parents=True)
    component.write_text("version = 1")
    for name in notebooks.COMPONENTS:
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
        if name == "model-scene":
            for peer in ("model-sky", "model-alerts"):
                source["cells"][0]["source"] += (
                    f"\n[{peer}](https://github.com/NVIDIA/cuPhoton/blob/"
                    f"codex/review/examples/{peer}/{peer}.ipynb)"
                )
        write_json(example / f"{name}.ipynb", source)
        cell = source["cells"][1]
        cell["execution_count"] = 1
        cell["outputs"] = [{"output_type": "stream", "text": "Reference"}]
        if name == "model-sky":
            cell["source"] = flag.replace("False", "True")
        notebook = tmp_path / f"{name}-input.json"
        write_json(notebook, source)
    git(checkout, "init", "-q")
    git(checkout, "add", ".")
    git(
        checkout,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-qm",
        "recorded study",
    )
    revision = git(checkout, "rev-parse", "HEAD")
    for name in notebooks.COMPONENTS:
        notebook = tmp_path / f"{name}-input.json"
        nb_digest = digest_file(notebook)
        source_digest = digest_file(
            checkout / "examples" / name / f"{name}.ipynb"
        )
        receipt = tmp_path / f"{name}-receipt.json"
        rc_digest = write_json(
            receipt,
            {
                "source_revision": revision,
                "source_tree_clean": True,
                "cuphoton_revision": revision,
                "public_notebook_sha256": nb_digest,
                "output_sha256": {f"{name}.executed.ipynb": nb_digest},
                "source_hashes": {f"{name}.ipynb": source_digest},
                "environment": {
                    "packages": {"cuphoton": f"0.3.0.dev1+g{revision[:12]}"}
                },
                "status": "passed",
                "completed": True,
                "device": "cuda",
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
    monkeypatch.setattr(
        notebooks, "get_blob", lambda d, p: p.write_bytes(blobs[d])
    )

    return checkout, revision, entries, blobs


def test_mixed_book_uses_the_reviewed_outputs(
    tmp_path, monkeypatch, reference_book
):
    checkout, revision, entries, _ = reference_book
    monkeypatch.chdir(tmp_path)
    guide = Path("preview/guide")
    notebooks.stage_notebooks(checkout, guide, release=True)
    config = yaml.safe_load((guide / "_quarto.yml").read_text())
    assert config["book"]["chapters"][0] == "index.qmd"
    assert config["book"]["chapters"][1]["chapters"] == [
        f"{n}.ipynb" for n in notebooks.COMPONENTS
    ]
    for name in notebooks.COMPONENTS:
        cells = json.loads((guide / f"{name}.ipynb").read_text())["cells"]
        assert cells[1]["outputs"][0]["text"] == "Reference"
        lines = cells[0]["source"]
        assert isinstance(lines, list) and lines[0] == "---\n"
        assert f"source/{name}.executed.ipynb" in "".join(lines)
        assert f"/blob/{revision}/" in "".join(lines)
        assert "callout-warning" not in "".join(lines)
        assert (
            json.loads(
                (checkout / "examples" / name / f"{name}.ipynb").read_text()
            )["cells"][1]["outputs"]
            == []
        )
    assert not list((guide / "assets").rglob("*.html"))

    scene_notebook = guide / "source/model-scene.executed.ipynb"
    scene_receipt = guide / "source/model-scene-execution.json"
    proof = json.loads(scene_receipt.read_text())
    proof["device"] = "cpu"
    write_json(scene_receipt, proof)
    with pytest.raises(ValueError, match="GPU reference execution required"):
        notebooks._check_reference(
            checkout, "model-scene", revision, scene_notebook, scene_receipt
        )

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


def test_development_guide_preserves_the_qualified_study(
    tmp_path, reference_book
):
    checkout, revision, entries, _ = reference_book
    sky = checkout / "examples/model-sky"
    source = sky / "model-sky.ipynb"
    original = source.read_bytes()
    source.write_text(
        source.read_text()
        .replace("Text", "New conclusions")
        .replace("RUN_BLEND_EXTENSION = False", "unreviewed()")
    )
    (sky / "assets/guide-head.html").write_text("new script")
    component = "src/cuphoton/xfit/photometry.py"
    (checkout / component).write_text("version = 2")
    guide = tmp_path / "preview/guide"
    notebooks.stage_notebooks(checkout, guide)
    for name in entries:
        chapter = json.loads((guide / f"{name}.ipynb").read_text())
        prose = "".join(chapter["cells"][0]["source"])
        assert f"/blob/{revision}/" in prose
        assert f"/commit/{revision}" in prose
        assert ("callout-warning" in prose) == (name != "model-scene")
        assert "New conclusions" not in prose
        assert chapter["cells"][1]["source"] != "unreviewed()"
        assert chapter["cells"][1]["outputs"][0]["text"] == "Reference"
    assert (
        guide.parent / "model-sky-includes/guide-head.html"
    ).read_text() == "<!-- include -->"
    with zipfile.ZipFile(guide / "source/model-sky.zip") as archive:
        assert archive.read("examples/model-sky/model-sky.ipynb") == original

    with pytest.raises(
        ValueError, match="model-sky: execution inputs changed"
    ) as err:
        notebooks.stage_notebooks(
            checkout, tmp_path / "release/guide", release=True
        )
    assert "examples/model-sky/model-sky.ipynb" in str(err.value)
    assert component in str(err.value)


def test_development_guide_checks_the_pinned_source(tmp_path, reference_book):
    checkout, _, entries, blobs = reference_book
    name = "model-alerts"
    entry = entries[name]
    source = checkout / "examples" / name / f"{name}.ipynb"
    source.write_text(source.read_text().replace("Text", "Unqualified prose"))
    proof = json.loads(blobs[entry["receipt_sha256"]])
    proof["source_hashes"][source.name] = digest_file(source)
    receipt = tmp_path / "changed.json"
    digest = write_json(receipt, proof)
    blobs[digest] = receipt.read_bytes()
    entry["receipt_sha256"] = digest
    write_json(checkout / "docs/guide/notebooks.json", entries)
    with pytest.raises(ValueError, match="pinned reference input mismatch"):
        notebooks.stage_notebooks(checkout, tmp_path / "guide")


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


@pytest.mark.parametrize("chapters", [("model-scene", "model-sky"), ()])
def test_release_stages_its_own_chapters(tmp_path, reference_book, chapters):
    checkout, _, entries, _ = reference_book
    manifest = checkout / "docs/guide/notebooks.json"
    write_json(manifest, {name: entries[name] for name in chapters})
    guide = tmp_path / "release/guide"
    notebooks.stage_notebooks(checkout, guide, release=True)
    config = yaml.safe_load((guide / "_quarto.yml").read_text())
    assert set(p.stem for p in guide.glob("*.ipynb")) == set(chapters)
    if chapters:
        assert config["book"]["chapters"][-1]["chapters"] == [
            f"{name}.ipynb" for name in chapters
        ]
        scene = json.loads((guide / "model-scene.ipynb").read_text())
        prose = "".join(scene["cells"][0]["source"])
        assert "[model-sky](model-sky.html)" in prose
        assert "model-alerts" in prose
        assert (
            "model-alerts.html" not in prose and "/codex/review" not in prose
        )
    else:
        assert config["book"]["chapters"] == ["index.qmd"]
        assert not config.get("science-notebooks")
