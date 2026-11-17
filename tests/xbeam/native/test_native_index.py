# SPDX-License-Identifier: Apache-2.0
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from cuphoton.xbeam.adapters.vdif.vdif_sidecar import SIDECAR_HEADER_BYTES
from cuphoton.xbeam.adapters.vdif.vdif_sidecar_builder import (
    build_vdif_binary_index,
)
from tests.xbeam.test_vdif_sidecar import _frame, _replace_with_v2


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


def test_index_queries_across_read_windows(binding, tmp_path):
    source = tmp_path / "ordered.vdif"
    count = 6000
    source.write_bytes(b"".join(_frame(i) for i in reversed(range(count))))
    summary = build_vdif_binary_index(source, backend="posix")
    reader = binding.NativeIndex(source, summary.index_path)
    rows = reader.query(0, 0, 2**64 - 1)
    assert len(rows) == count
    assert [row[1] for row in rows] == list(reversed(range(count)))
    assert [row[2:] for row in rows] == [(0, 0)] * count
    assert reader.query(0, rows[2500][0], rows[3500][0]) == rows[2500:3500]
    assert len(reader.physical_runs()) == count


@pytest.mark.parametrize("phase", ["admission", "query"])
def test_index_truncation_raises_without_terminating(
    native_executable, tmp_path, phase
):
    source, summary = make_index(tmp_path)
    probe = tmp_path / "truncate.cpp"
    probe.write_text(
        "// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION "
        "& AFFILIATES. All rights reserved.\n"
        r"""// SPDX-License-Identifier: Apache-2.0
#include "xbeam/radio.hpp"
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

static const char* index_path;
static bool during_admission = false;
static bool truncated = false;

extern "C" ssize_t __real_pread(int, void*, size_t, off_t);
extern "C" ssize_t __wrap_pread(
    int fd, void* data, size_t size, off_t offset) {
    const auto result = __real_pread(fd, data, size, offset);
    // Truncate after the header read so admission has already recorded the
    // original extent, but has not accessed the index entries yet.
    struct stat opened{}, named{};
    if (during_admission && !truncated && offset == 16 && result > 0
        && !fstat(fd, &opened) && !stat(index_path, &named)
        && opened.st_dev == named.st_dev && opened.st_ino == named.st_ino) {
        truncated = truncate(index_path, 65536) == 0;
    }
    return result;
}

int main(int argc, char** argv) {
    if (argc != 4) return 2;
    index_path = argv[2];
    during_admission = std::string(argv[3]) == "admission";
    try {
        xbeam::IndexReader reader(argv[1], index_path);
        if (!during_admission) {
            truncated = truncate(index_path, 65536) == 0;
            reader.query(0, 0, UINT64_MAX);
        }
    } catch (const xbeam::RadioError& error) {
        std::cout << error.what() << '\n';
        return truncated ? 0 : 3;
    }
    return 1;
}
"""
    )
    native = Path(__file__).resolve().parents[3] / "src/cuphoton/xbeam/native"
    executable = tmp_path / "truncate-index"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-I" + str(native / "include"),
            str(probe),
            str(native_executable.parent / "libxbeam_radio.a"),
            "-Wl,--wrap=pread",
            "-lcrypto",
            "-o",
            str(executable),
        ],
        check=True,
    )
    result = subprocess.run(
        [str(executable), str(source), str(summary.index_path), phase],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "truncated" in result.stdout or "changed" in result.stdout


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


def test_execution_selects_v4_after_older_valid_sidecar(
    native_executable, tmp_path
):
    source, summary = make_index(tmp_path)
    current_bytes = summary.index_path.read_bytes()
    _replace_with_v2(summary)
    directory = tmp_path / "candidates"
    directory.mkdir()
    old = directory / (source.name + ".000-old.vdix")
    current = directory / (source.name + ".zzz-current.vdix")
    old.write_bytes(summary.index_path.read_bytes())
    current.write_bytes(current_bytes)
    root = Path(__file__).resolve().parents[3]
    native = root / "src/cuphoton/xbeam/native"
    executable = tmp_path / "index-selection"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-I" + str(native / "include"),
            "-I" + str(native / "cuda"),
            "-I" + str(native / "third_party"),
            str(Path(__file__).with_name("index_selection_smoke.cpp")),
            str(native / "cuda/radio_model.cpp"),
            str(native_executable.parent / "libxbeam_radio.a"),
            "-lcrypto",
            "-o",
            str(executable),
        ],
        check=True,
    )
    arguments = [str(executable), str(source), str(directory)]
    selected = json.loads(subprocess.check_output(arguments))
    assert selected["schema_version"] == 4
    current.unlink()
    refused = subprocess.run(arguments, capture_output=True, text=True)
    assert refused.returncode == 1
    assert "schema-v4" in refused.stderr
