# SPDX-License-Identifier: Apache-2.0
import importlib.util
import io
import os
from pathlib import Path

import pytest

from cuphoton.xbeam.job_text import parse_job_text


@pytest.fixture(scope="module")
def binding():
    directory = os.environ.get("XBEAM_NATIVE_BINDING_DIR")
    if not directory:
        pytest.skip("set XBEAM_NATIVE_BINDING_DIR")
    files = list(Path(directory).rglob("_binding*.so"))
    assert len(files) == 1
    spec = importlib.util.spec_from_file_location("_binding", files[0])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "text",
    [
        "",
        "# comments\n@ comments\n",
        "FOO: bar\n",
        "A: 1\nA: 2\n",
        " EMPTY : \n\tCOLON: a:b:c\r\n",
        "TELESCOPE NAME 0: AA\n",
        "PATH: /path/with space/input.vdif\n",
        "PATH: /data/日本/input.vdif\n",
    ],
)
def test_text_syntax(binding, text):
    reference = parse_job_text(io.StringIO(text))
    assert binding.parse_text(text) == [
        (entry.key, entry.value, entry.line_number)
        for entry in reference.entries
    ]


@pytest.mark.parametrize("text", ["no colon", ": value", " \n\t : value"])
def test_text_errors(binding, text):
    with pytest.raises(ValueError):
        parse_job_text(io.StringIO(text))
    with pytest.raises(RuntimeError):
        binding.parse_text(text)
