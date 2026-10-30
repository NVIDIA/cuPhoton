# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Both sidecar backends enforce the same strict frame contract."""

from __future__ import annotations

from struct import pack

import numpy as np
import pytest

from cuphoton.xbeam.adapters.vdif.vdif_index import (
    VdifIndexError,
    VdifTimestampKey,
)
from cuphoton.xbeam.adapters.vdif.vdif_sidecar import read_vdif_binary_frames
from cuphoton.xbeam.adapters.vdif.vdif_sidecar_builder import (
    build_vdif_binary_index,
)


@pytest.fixture(
    params=["posix", pytest.param("cufile", marks=pytest.mark.gpu)]
)
def backend(request):
    if request.param == "cufile":
        cupy = pytest.importorskip("cupy")
        try:
            count = cupy.cuda.runtime.getDeviceCount()
        except cupy.cuda.runtime.CUDARuntimeError:
            pytest.skip("CUDA runtime unavailable")
        if count < 1:
            pytest.skip("no CUDA device")
    return request.param


def _frame(number, *, invalid=False, changed_word=None):
    words = [17 | (int(invalid) << 31), (40 << 24) | number, 5, 1 << 26]
    words.extend([0] * 4)
    if changed_word is not None:
        index, value = changed_word
        words[index] = value
    return pack("<8I", *words) + bytes(8)


@pytest.mark.parametrize(
    "changed_word,reason",
    [
        ((0, 17 | (1 << 30)), "legacy"),
        ((1, (40 << 24) | (1 << 30) | 0xFFFFFF), "reserved"),
        ((2, 5 | (1 << 29)), "version"),
        ((2, 6), "frame length"),
        ((3, 2 << 26), "samples|sample geometry"),
        ((2, 5 | (31 << 24)), "samples|sample geometry"),
    ],
)
@pytest.mark.parametrize("position", [0, 2])
def test_malformed_frame_aborts_before_publishing(
    tmp_path, backend, changed_word, reason, position
):
    source = tmp_path / "malformed.vdif"
    source.write_bytes(
        b"".join(
            _frame(i, changed_word=changed_word if i == position else None)
            for i in range(3)
        )
    )

    with pytest.raises(
        VdifIndexError, match=rf"frame {position}: .*({reason})"
    ):
        build_vdif_binary_index(
            source, backend=backend, frame_size_bytes=40, chunk_bytes=40
        )

    assert list(tmp_path.iterdir()) == [source]


@pytest.mark.parametrize("changed_word", [(3, (1 << 26) | 1), (4, 1 << 24)])
@pytest.mark.parametrize("invalid", [False, True])
@pytest.mark.parametrize("chunk_bytes", [40, 120])
def test_conflicting_leading_format_never_masks_valid_continuation(
    tmp_path, backend, changed_word, invalid, chunk_bytes
):
    source = tmp_path / "conflicting-format.vdif"
    source.write_bytes(
        _frame(0, invalid=invalid, changed_word=changed_word)
        + _frame(1)
        + _frame(2)
    )

    with pytest.raises(VdifIndexError, match="changes (format|within)"):
        build_vdif_binary_index(
            source, backend=backend, chunk_bytes=chunk_bytes
        )

    assert list(tmp_path.iterdir()) == [source]


@pytest.mark.parametrize("all_invalid", [False, True])
def test_consistent_invalid_frames_keep_their_timed_validity(
    tmp_path, backend, all_invalid
):
    source = tmp_path / "invalid.vdif"
    source.write_bytes(
        _frame(0, invalid=True)
        + _frame(1, invalid=True)
        + _frame(2, invalid=all_invalid)
    )

    summary = build_vdif_binary_index(source, backend=backend, chunk_bytes=40)
    _, threads, frames, invalid = read_vdif_binary_frames(
        source, VdifTimestampKey(40, 17, 0), VdifTimestampKey(40, 17, 3)
    )

    np.testing.assert_array_equal(threads, [0, 0, 0])
    np.testing.assert_array_equal(frames, [0, 1, 2])
    np.testing.assert_array_equal(invalid, [True, True, all_invalid])
    assert summary.source_ordered
    assert summary.rejected_header_frame_count == 0
    assert summary.rejected_format_frame_count == 0


def test_incorrect_selected_frame_size_is_reported(tmp_path, backend):
    source = tmp_path / "wrong-stride.vdif"
    source.write_bytes(b"".join(_frame(i) for i in range(4)))

    with pytest.raises(VdifIndexError, match="frame 0: frame length"):
        build_vdif_binary_index(source, backend=backend, frame_size_bytes=80)

    assert list(tmp_path.iterdir()) == [source]


def test_malformed_forced_rebuild_preserves_existing_index(tmp_path, backend):
    source = tmp_path / "recording.vdif"
    source.write_bytes(_frame(0) + _frame(1))
    summary = build_vdif_binary_index(source, backend=backend)
    original = summary.index_path.read_bytes()
    source.write_bytes(_frame(0) + _frame(1, changed_word=(2, 6)))

    with pytest.raises(VdifIndexError, match="frame 1: frame length"):
        build_vdif_binary_index(source, backend=backend, force=True)

    assert summary.index_path.read_bytes() == original
    assert not list(tmp_path.glob(".*"))
