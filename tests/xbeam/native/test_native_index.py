# SPDX-License-Identifier: Apache-2.0
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from cuphoton.xbeam.adapters.vdif.vdif_sidecar import SIDECAR_HEADER_BYTES
from cuphoton.xbeam.adapters.vdif.vdif_sidecar_builder import (
    build_vdif_binary_index,
)
from tests.xbeam.test_vdif_sidecar import _frame


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


def make_index(tmp_path):
    source = tmp_path / "test.vdif"
    source.write_bytes(b"".join(_frame(i) for i in [5, 1, 3, 2, 0, 4]))
    summary = build_vdif_binary_index(source, backend="posix")
    return source, summary


def test_index_entries_match_python(binding, tmp_path):
    source, summary = make_index(tmp_path)
    reader = binding.NativeIndex(source, summary.index_path)
    metadata = reader.metadata()
    assert metadata["entry_count"] == 6
    entries = np.memmap(
        summary.index_path,
        mode="r",
        dtype="<u8",
        offset=SIDECAR_HEADER_BYTES,
        shape=(6, 3),
    )
    expected = [
        (
            int(row[0]),
            int(row[1] & np.uint64((1 << 53) - 1)),
            int(row[1] >> np.uint64(54)),
            int(row[2]),
        )
        for row in entries
    ]
    thread = expected[0][2]
    assert reader.query(thread, 0, 2**64 - 1) == expected
    assert (
        reader.query(thread, expected[1][0], expected[4][0]) == expected[1:4]
    )
    assert reader.query(1023, 0, 2**64 - 1) == []


def test_index_replacement_rejected(binding, tmp_path):
    source, summary = make_index(tmp_path)
    reader = binding.NativeIndex(source, summary.index_path)
    replacement = tmp_path / "replace.vdif"
    replacement.write_bytes(source.read_bytes())
    replacement.replace(source)
    with pytest.raises(RuntimeError, match="replaced"):
        reader.query(0, 0, 2**64 - 1)


def test_index_source_mutation_rejected(binding, tmp_path):
    source, summary = make_index(tmp_path)
    with source.open("r+b") as out:
        out.seek(33)
        out.write(b"\x7f")
    with pytest.raises(RuntimeError, match="stale"):
        binding.NativeIndex(source, summary.index_path)


@pytest.mark.parametrize("corruption", ["duplicate", "unsorted", "state"])
def test_corrupt_entries_rejected(binding, tmp_path, corruption):
    source, summary = make_index(tmp_path)
    rows = np.memmap(
        summary.index_path,
        mode="r+",
        dtype="<u8",
        offset=SIDECAR_HEADER_BYTES,
        shape=(6, 3),
    )
    if corruption == "duplicate":
        rows[0, 1] = rows[1, 1]
    elif corruption == "unsorted":
        rows[0, 0] = rows[-1, 0] + np.uint64(1)
    else:
        rows[0, 2] = np.uint64(1 << 30)
    rows.flush()
    del rows
    with pytest.raises(RuntimeError, match="sidecar"):
        binding.NativeIndex(source, summary.index_path)


def test_native_physical_validity_retains_invalid_clock(binding, tmp_path):
    from cuphoton.xbeam.adapters.vdif.vdif_sidecar import (
        _summary_physical_timestamp_runs,
    )
    from tests.xbeam.test_vdif_sidecar import _recorded_filler

    source = tmp_path / "physical.vdif"
    source.write_bytes(
        _frame(0) + _frame(1, invalid=True) + _recorded_filler() + _frame(3)
    )
    summary = build_vdif_binary_index(source, backend="posix")
    reader = binding.NativeIndex(source, summary.index_path)
    assert reader.frame_validity(0, 4) == [1, 0, 0, 1]
    assert reader.physical_runs() == [
        tuple(map(int, row))
        for row in _summary_physical_timestamp_runs(summary)
    ]
    with pytest.raises(RuntimeError, match="exceeds"):
        reader.frame_validity(3, 2)
