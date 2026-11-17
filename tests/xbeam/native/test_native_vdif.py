# SPDX-License-Identifier: Apache-2.0
"""Differential validation of the optional native header/inspection layer."""

import importlib.util
import json
import os
import random
import struct
import subprocess
from pathlib import Path

import pytest

from cuphoton.xbeam.adapters.vdif.vdif import VdifError, VdifHeader

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def binding():
    directory = os.environ.get("XBEAM_NATIVE_BINDING_DIR")
    if not directory:
        pytest.skip("build the binding and set XBEAM_NATIVE_BINDING_DIR")
    candidates = list(Path(directory).rglob("_binding*.so"))
    assert len(candidates) == 1
    spec = importlib.util.spec_from_file_location("_binding", candidates[0])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def header(seed):
    rng = random.Random(seed)
    complex_data = rng.randrange(2)
    channels = rng.randrange(6)
    words = [
        rng.randrange(1 << 30) | (rng.randrange(2) << 31),
        rng.randrange(1 << 24) | (rng.randrange(64) << 24),
        (channels << 24) | (8224 // 8),
        (complex_data << 31)
        | (1 << 26)
        | (rng.randrange(1024) << 16)
        | rng.randrange(65536),
        *[rng.randrange(1 << 32) for _ in range(4)],
    ]
    return struct.pack("<8I", *words)


@pytest.mark.parametrize("seed", range(256))
def test_native_header_matches_python(binding, seed):
    data = header(seed)
    reference = VdifHeader.from_bytes(data)
    actual = binding.parse_header(data)
    for key, value in actual.items():
        assert value == getattr(reference, key), key


@pytest.mark.parametrize("length", range(32))
def test_truncated_header(binding, length):
    with pytest.raises(RuntimeError):
        binding.parse_header(header(1)[:length])


@pytest.mark.parametrize(
    "word,mask", [(0, 1 << 30), (1, 1 << 31), (2, 1 << 29), (3, 1 << 26)]
)
def test_malformed_header(binding, word, mask):
    words = list(struct.unpack("<8I", header(0)))
    words[word] ^= mask
    data = struct.pack("<8I", *words)
    with pytest.raises(VdifError):
        VdifHeader.from_bytes(data)
    with pytest.raises(RuntimeError):
        binding.parse_header(data)


def test_inspection_and_standalone(binding, tmp_path):
    path = tmp_path / "recording.vdif"
    path.write_bytes(b"".join(header(i) + bytes(8192) for i in range(12)))
    expected = binding.inspect_recording(path)
    assert expected["frames"] == 12
    assert expected["source_bytes"] == 12 * 8224
    assert sum(t["frames"] for t in expected["threads"]) == 12
    executable = ROOT / "build/xbeam-native/xbeam-native"
    actual = json.loads(
        subprocess.check_output([str(executable), "inspect-vdif", str(path)])
    )
    assert actual == expected
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(RuntimeError, match="truncated VDIF frame"):
        binding.inspect_recording(path)
    result = subprocess.run(
        [str(executable), "inspect-vdif", str(path)], capture_output=True
    )
    assert result.returncode == 1
    assert b"truncated VDIF frame" in result.stderr
