# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import json
import sys

import pytest

from cuphoton_dev import notebook_export, notebooks
from cuphoton_dev.build import git
from cuphoton_dev.notebook_store import digest_file


def reference_run(tmp_path, name, sky_version="development"):
    checkout = tmp_path / "checkout"
    example = checkout / "examples" / name
    example.mkdir(parents=True)
    (checkout / "LICENSE").write_text("license\n")
    runner = example / "run.py"
    runner.write_text("fit()\n")
    source = example / f"{name}.ipynb"
    cell = {
        "id": "fit",
        "cell_type": "code",
        "source": "fit()",
        "execution_count": None,
        "outputs": [],
    }
    source.write_text(json.dumps({"cells": [cell]}))
    git(checkout, "init", "-q")
    git(checkout, "config", "user.name", "Test")
    git(checkout, "config", "user.email", "test@example.com")
    git(checkout, "config", "commit.gpgsign", "false")
    git(checkout, "add", ".")
    git(checkout, "commit", "-qm", "Reference source")
    revision = git(checkout, "rev-parse", "HEAD")
    inputs = {path.name: digest_file(path) for path in (source, runner)}

    reference = tmp_path / "reference"
    reference.mkdir()
    raw = reference / f"{name}.executed.ipynb"
    stream = {"output_type": "stream", "name": "stderr", "text": "warning"}
    cell.update(execution_count=1, outputs=[stream])
    scene = name == "model-scene"
    if scene:
        stream["text"] = f"{reference}/input: warning"
    raw.write_text(json.dumps({"cells": [cell]}))
    if scene:
        record = {
            "source_revision": revision,
            "source_tree_clean": True,
            "profile": "reference",
            "device": "cuda",
            "completed": True,
            "source_hashes": inputs,
            "artifact_hashes": {raw.name: digest_file(raw)},
            "packages": {},
            "protocol_sha256": "c" * 64,
        }
    else:
        record = {
            "status": "passed",
            "source_sha256": inputs,
            "output_sha256": {raw.name: digest_file(raw)},
        }
        if name == "model-sky":
            if sky_version == "development":
                sky_version = f"0.3.0.dev1+g{revision[:12]}"
            else:
                git(checkout, "tag", "--no-sign", f"v{sky_version}")
            record["environment"] = {"packages": {"cuphoton": sky_version}}
        else:
            record.update(
                profile="full-reference", cuphoton_revision=revision
            )
    receipt = reference / "execution.json"
    receipt.write_text(json.dumps(record))
    downloads = None
    if scene:
        downloads = tmp_path / "downloads"
        downloads.mkdir()
        stream["text"] = "./input: warning"
        public = downloads / raw.name
        public.write_text(json.dumps({"cells": [cell]}))
        projected = dict(
            record,
            original_receipt_sha256=digest_file(receipt),
            public_notebook_sha256=digest_file(public),
        )
        (downloads / f"{name}-execution.json").write_text(
            json.dumps(projected)
        )
    return {
        "checkout": checkout,
        "name": name,
        "revision": revision,
        "reference": reference,
        "scene_downloads": downloads,
    }


@pytest.mark.parametrize(
    ("name", "sky_version"),
    [
        ("model-sky", "development"),
        ("model-sky", "0.2.2"),
        ("model-sky", "0.2.2rc1"),
        ("model-alerts", None),
        ("model-scene", None),
    ],
)
def test_prepare_public_reference(tmp_path, name, sky_version):
    args = reference_run(tmp_path, name, sky_version)
    output = tmp_path / "public"
    notebook_export.prepare(**args, output=output)
    published = output / f"{name}.executed.ipynb"
    metadata = json.loads((output / f"{name}-execution.json").read_text())
    assert metadata["source_revision"] == args["revision"]
    assert metadata["public_notebook_sha256"] == digest_file(published)
    stream = json.loads(published.read_text())["cells"][0]["outputs"][0]
    assert stream == {
        "output_type": "stream",
        "name": "stderr",
        "text": "./input: warning" if name == "model-scene" else "warning",
    }

    if name == "model-scene":
        # Even a self-consistently hashed preview cannot change a result.
        downloads = args["scene_downloads"]
        public = downloads / published.name
        book = json.loads(public.read_text())
        book["cells"][0]["outputs"][0]["text"] = "./input: a different result"
        public.write_text(json.dumps(book))
        receipt = downloads / f"{name}-execution.json"
        projected = json.loads(receipt.read_text())
        projected["public_notebook_sha256"] = digest_file(public)
        receipt.write_text(json.dumps(projected))
        error = "changed a warning"
    else:
        receipt = args["reference"] / "execution.json"
        record = json.loads(receipt.read_text())
        record["unexpected"] = "/home/example/local-file"
        receipt.write_text(json.dumps(record))
        error = "Private path"
    rejected = tmp_path / "rejected"
    with pytest.raises(ValueError, match=error):
        notebook_export.prepare(**args, output=rejected)
    assert not rejected.exists()


@pytest.mark.parametrize(
    ("name", "change", "error"),
    [
        ("model-scene", "public dirty tree", "clean source tree"),
        ("model-alerts", "cuPhoton revision", "cuPhoton revision differs"),
        ("model-sky", "resolved revision", "cuPhoton revision differs"),
        ("model-sky", "dirty version", "clean source version"),
        ("model-alerts", "recorded example changed", "inputs? changed"),
    ],
)
def test_rejects_invalid_reference(
    tmp_path, monkeypatch, name, change, error
):
    args = reference_run(tmp_path, name)
    checkout = args["checkout"]
    output = tmp_path / "public"
    notebook_export.prepare(**args, output=output)
    published = output / f"{name}.executed.ipynb"
    receipt = output / f"{name}-execution.json"
    metadata = json.loads(receipt.read_text())
    upstream = args["reference"] / "execution.json"
    if change == "public dirty tree":
        upstream = args["scene_downloads"] / f"{name}-execution.json"
    record = json.loads(upstream.read_text())
    if change == "public dirty tree":
        metadata["source_tree_clean"] = record["source_tree_clean"] = False
    elif change == "cuPhoton revision":
        metadata["cuphoton_revision"] = record["cuphoton_revision"] = "b" * 40
    elif change == "resolved revision":
        git(checkout, "commit", "--allow-empty", "-qm", "Different revision")
        other = git(checkout, "rev-parse", "--short=12", "HEAD")
        for obj in (record, metadata):
            obj["environment"]["packages"]["cuphoton"] = (
                f"0.3.0.dev2+g{other}"
            )
    elif change == "dirty version":
        for obj in (record, metadata):
            obj["environment"]["packages"]["cuphoton"] += ".d20261123"
    else:
        (checkout / "examples" / name / "run.py").write_text(
            "modified_fit()\n"
        )
    upstream.write_text(json.dumps(record))
    receipt.write_text(json.dumps(metadata))
    monkeypatch.chdir(checkout)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "notebooks",
            "record",
            "--name",
            name,
            "--revision",
            args["revision"],
            "--notebook",
            str(published),
            "--receipt",
            str(receipt),
        ],
    )
    monkeypatch.setattr(
        notebooks, "put_blob", lambda *_: pytest.fail("Unexpected upload")
    )
    with pytest.raises(ValueError, match=error):
        notebooks.main()

    if change in {"public dirty tree", "cuPhoton revision"}:
        # Even if a new manifest pins the corrupted receipt, main staging
        # must authenticate the execution independently of input freshness.
        guide = checkout / "docs/guide"
        guide.mkdir(parents=True)
        blobs = {
            digest_file(file): file.read_bytes()
            for file in (published, receipt)
        }
        monkeypatch.setattr(
            notebooks, "get_blob", lambda sha, p: p.write_bytes(blobs[sha])
        )
        (guide / "notebooks.json").write_text(
            json.dumps(
                {
                    name: {
                        "source_revision": args["revision"],
                        "notebook_sha256": digest_file(published),
                        "receipt_sha256": digest_file(receipt),
                    }
                }
            )
        )
        with pytest.raises(ValueError, match=error):
            notebooks.stage_notebooks(checkout, tmp_path / "staged")
    rejected = tmp_path / "rejected"
    with pytest.raises(ValueError, match=error):
        notebook_export.prepare(**args, output=rejected)
    assert not rejected.exists()


def test_scene_export_rejects_dirty_raw_execution(tmp_path):
    args = reference_run(tmp_path, "model-scene")
    raw = args["reference"] / "execution.json"
    record = json.loads(raw.read_text())
    record["source_tree_clean"] = False
    raw.write_text(json.dumps(record))
    receipt = args["scene_downloads"] / "model-scene-execution.json"
    preview = json.loads(receipt.read_text())
    preview["original_receipt_sha256"] = digest_file(raw)
    receipt.write_text(json.dumps(preview))
    with pytest.raises(
        ValueError, match="Scene public download and GPU receipt disagree"
    ):
        notebook_export.prepare(**args, output=tmp_path / "public")
    assert not (tmp_path / "public").exists()
