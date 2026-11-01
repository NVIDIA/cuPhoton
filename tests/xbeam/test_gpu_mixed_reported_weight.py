# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Mixed replay validity must define every reported exposure row."""

from types import SimpleNamespace

import numpy as np
import pytest

from cuphoton.xbeam.adapters.radio import gpu_execution as execution


class PoisonEmptyArrayModule:
    __name__ = "numpy"

    def __init__(self):
        self.empty_calls = 0

    def __getattr__(self, name):
        return getattr(np, name)

    def empty(self, shape, dtype):
        self.empty_calls += 1
        return np.full(shape, np.nan, dtype=dtype)


@pytest.mark.parametrize(
    "reported_modes", [(), (False,), (True,), (False, True)]
)
@pytest.mark.parametrize("permutation", [(0, 1, 2, 3, 4), (3, 0, 4, 2, 1)])
def test_mixed_reported_weights_define_every_row(
    monkeypatch, reported_modes, permutation
):
    _assert_assembled(monkeypatch, reported_modes, permutation)


@pytest.mark.parametrize("authoritative_mode", [False, True])
@pytest.mark.parametrize("has_reported", [False, True])
def test_uniform_validity_keeps_single_piece_path(
    monkeypatch, authoritative_mode, has_reported
):
    modes = (authoritative_mode,) if has_reported else ()
    xp = _assert_assembled(
        monkeypatch, modes, (3, 0, 4, 2, 1), authoritative_mode
    )
    assert xp.empty_calls == 0


def _assert_assembled(
    monkeypatch, reported_modes, permutation, uniform_mode=None
):
    starts = np.asarray(permutation, dtype=np.float64)
    row_authoritative = np.array([False, True, False, True, True])
    if uniform_mode is not None:
        row_authoritative[:] = uniform_mode
    authoritative = row_authoritative[np.asarray(permutation)]
    count = starts.size
    monkeypatch.setattr(execution, "_fft_length", lambda data: 8)
    monkeypatch.setattr(
        execution,
        "_ordered_model_runs",
        lambda *args: [(object(), slice(0, count))],
    )
    monkeypatch.setattr(
        execution,
        "select_fft_eligibility",
        lambda *args, **kw: [True] * count,
    )
    monkeypatch.setattr(
        execution,
        "select_fft_source_replay",
        lambda *args, **kw: ([0] * count, [0] * count),
    )
    monkeypatch.setattr(
        execution,
        "select_fft_source_segments",
        lambda *args, **kw: (None,) * count,
    )
    monkeypatch.setattr(
        execution,
        "select_fft_weight_exclusions",
        lambda *args, **kw: [0] * count,
    )
    monkeypatch.setattr(
        execution,
        "select_fft_authoritative_validity",
        lambda *args, **kw: authoritative,
    )
    calls = []

    def station_blocks(
        job, data, model, selected_starts, xp, *args, **kwargs
    ):
        mode = bool(args[-1])
        calls.append(mode)
        expected_modes = row_authoritative[selected_starts.astype(int)]
        assert np.all(expected_modes == mode)
        weight = (selected_starts + 1) / 10
        spectrum = np.column_stack((selected_starts, selected_starts + 10))
        reported = weight + 0.25 if mode in reported_modes else None
        return execution._StationBlocks(
            spectrum, weight, np.array([0.0, 1.0]), reported
        )

    monkeypatch.setattr(execution, "_station_blocks", station_blocks)
    xp = PoisonEmptyArrayModule()
    data = SimpleNamespace(
        sample_rate_hz=8.0,
        datastream=SimpleNamespace(station_id=0),
        frequency=SimpleNamespace(channel_count=2),
    )
    plan = SimpleNamespace(
        source_timing_events=(),
        source_masked_fft_count=0,
        source_weight_credit_intervals=(),
        source_reported_weight_credit_intervals=(),
    )
    result = execution._station_blocks_for_scan(
        SimpleNamespace(fringe_approximation_order=1),
        data,
        SimpleNamespace(start_offset_s=0.0),
        starts,
        xp,
        plan,
    )
    expected_weight = (starts + 1) / 10
    np.testing.assert_array_equal(result.weight, expected_weight)
    np.testing.assert_array_equal(
        result.spectrum, np.column_stack((starts, starts + 10))
    )
    applicable = np.array(
        [bool(mode) in reported_modes for mode in authoritative]
    )
    if np.any(applicable):
        expected_reported = expected_weight + np.where(applicable, 0.25, 0.0)
        assert np.all(np.isfinite(result.reported_weight))
        np.testing.assert_array_equal(
            result.reported_weight, expected_reported
        )
    else:
        assert result.reported_weight is None
    assert calls == (
        [False, True] if uniform_mode is None else [uniform_mode]
    )
    return xp
