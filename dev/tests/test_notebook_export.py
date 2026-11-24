# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from cuphoton_dev import notebook_export, notebooks
from cuphoton_dev.notebook_store import digest_file


@pytest.mark.parametrize("name", ["model-sky", "model-alerts", "model-scene"])
def test_prepare_public_reference(tmp_path, monkeypatch, name):
    revision = "a" * 40
    checkout = tmp_path / "checkout"
    example = checkout / "examples" / name
    example.mkdir(parents=True)
    source = example / f"{name}.ipynb"
    cell = {
        "id": "fit",
        "cell_type": "code",
        "source": "fit()",
        "execution_count": None,
        "outputs": [],
    }
    source.write_text(json.dumps({"cells": [cell]}))
    inputs = {source.name: digest_file(source)}
    reference = tmp_path / "reference"
    reference.mkdir()
    raw = reference / f"{name}.executed.ipynb"
    stream = {"output_type": "stream", "name": "stderr", "text": "warning"}
    cell.update(execution_count=1, outputs=[stream])
    scene = name == "model-scene"
    if scene:
        stream["text"] = f"{reference}/input: warning"
    raw.write_text(json.dumps({"cells": [cell]}))
    digest = digest_file(raw)
    if scene:
        record = {
            "source_revision": revision,
            "source_tree_clean": True,
            "profile": "reference",
            "device": "cuda",
            "completed": True,
            "source_hashes": inputs,
            "artifact_hashes": {raw.name: digest},
            "packages": {},
            "protocol_sha256": "c" * 64,
        }
    else:
        record = {
            "status": "passed",
            "source_sha256": inputs,
            "output_sha256": {raw.name: digest},
        }
        if name == "model-sky":
            record["environment"] = {
                "packages": {"cuphoton": "0.3.0.dev1+gaaaaaaaa"}
            }
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
    monkeypatch.setattr(notebooks, "git", lambda *args: "")
    monkeypatch.setattr(notebook_export, "resolve", lambda *args: revision)
    kwargs = {
        "checkout": checkout,
        "name": name,
        "revision": revision,
        "reference": reference,
        "scene_downloads": downloads,
    }
    output = tmp_path / "public"
    notebook_export.prepare(**kwargs, output=output)
    published = output / raw.name
    metadata = json.loads((output / f"{name}-execution.json").read_text())
    assert metadata["source_revision"] == revision
    assert metadata["public_notebook_sha256"] == digest_file(published)
    assert json.loads(published.read_text())["cells"][0]["outputs"] == [
        stream
    ]

    if scene:
        # Even a self-consistently hashed preview cannot change a result.
        stream["text"] = "./input: a different result"
        public.write_text(json.dumps({"cells": [cell]}))
        projected["public_notebook_sha256"] = digest_file(public)
        (downloads / f"{name}-execution.json").write_text(
            json.dumps(projected)
        )
        error = "changed a warning"
    else:
        record["unexpected"] = "/home/example/local-file"
        receipt.write_text(json.dumps(record))
        error = "Private path"
    rejected = tmp_path / "rejected"
    with pytest.raises(ValueError, match=error):
        notebook_export.prepare(**kwargs, output=rejected)
    assert not rejected.exists()
