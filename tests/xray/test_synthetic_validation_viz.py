# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import sys
from html.parser import HTMLParser

import pytest

from cuphoton.xray.synthetic_validation import (
    validation_sweep,
    write_validation_figure,
)


class _ResourceParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.resource_urls = []

    def handle_starttag(self, tag, attrs):
        resource_attr = {"script": "src", "link": "href"}.get(tag)
        if resource_attr is not None:
            url = dict(attrs).get(resource_attr)
            if url is not None:
                self.resource_urls.append(url)


@pytest.fixture
def sweep():
    return validation_sweep(snr_db=(30.0,), trials=2)


def test_validation_figure_embeds_resources_for_offline_use(tmp_path, sweep):
    pytest.importorskip("bokeh")
    path = tmp_path / "validation.html"

    assert write_validation_figure([sweep], path, title="Validation figure")

    html = path.read_text(encoding="utf-8")
    resources = _ResourceParser()
    resources.feed(html)
    assert resources.resource_urls == []
    assert "/* BEGIN bokeh.min.js */" in html
    assert "<title>Validation figure</title>" in html
    assert "angular_frequency std (points) vs bound (dashed)" in html
    assert "decay std (points) vs bound (dashed)" in html
    assert "loss rate" in html


def test_validation_figure_preserves_bokeh_output_state(
    tmp_path, sweep, monkeypatch
):
    pytest.importorskip("bokeh")
    from bokeh.io import state

    current = state.State()
    monkeypatch.setattr(state, "_STATE", current)
    current.output_file(tmp_path / "existing.html", title="Existing output")
    existing_file = current.file
    existing_document = current.document

    assert write_validation_figure(
        [sweep], tmp_path / "validation.html", title="Validation figure"
    )

    assert current.file is existing_file
    assert current.document is existing_document
    assert current.document.roots == []


def test_validation_figure_returns_false_without_bokeh(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "bokeh", None)
    monkeypatch.setitem(sys.modules, "bokeh.embed", None)
    monkeypatch.setitem(sys.modules, "bokeh.layouts", None)
    path = tmp_path / "validation.html"

    assert not write_validation_figure([], path, title="Validation figure")
    assert not path.exists()
