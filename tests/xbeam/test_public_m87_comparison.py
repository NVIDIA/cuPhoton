# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline checks for the optional public-data comparison example."""

import itertools
import runpy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest


def comparison_example():
    return runpy.run_path(
        str(
            Path(__file__).resolve().parents[2]
            / "examples/xbeam/compare_public_m87_closures.py"
        )
    )


@pytest.fixture
def observation():
    eh = pytest.importorskip("ehtim")
    stations = ("A", "B", "C", "D")
    array = np.zeros(4, dtype=eh.const_def.DTARR)
    array["site"] = stations
    array["sefdr"] = array["sefdl"] = 1.0
    rows = []
    for time in (1.0, 2.0):
        for i, (a, b) in enumerate(itertools.combinations(stations, 2)):
            if time == 2.0 and (a, b) == ("A", "B"):
                continue
            row = np.zeros((), dtype=eh.const_def.DTPOL_STOKES)
            row["time"], row["tint"] = time, 10.0
            # Obsdata normalizes these reversed rows to forward orientation
            # before they reach the record adapter.
            row["t1"], row["t2"] = b, a
            row["vis"] = (1 + i * 0.3) * np.exp(1j * (i * i + 1) * 0.2)
            row["sigma"] = 0.01
            rows.append(row)
    return eh.obsdata.Obsdata(
        0.0,
        0.0,
        230e9,
        2e9,
        np.array(rows),
        array,
        source="synthetic",
        mjd=58000,
    )


def test_comparison_matches_ehtim_with_missing_baseline(observation):
    result = comparison_example()["compare_observation"](observation)
    assert result["closure_phases"] == 6
    assert result["log_closure_amplitudes"] == 4
    assert result["max_phase_error_rad"] < 1e-12
    assert result["max_log_amplitude_error"] < 1e-12


def test_comparison_detects_conjugation_error(observation, monkeypatch):
    compare = comparison_example()["compare_observation"]
    compute = compare.__globals__["compute_closures"]

    def conjugated(*args, **kwargs):
        return [
            replace(record, values=-record.values)
            if record.kind == "phase"
            else record
            for record in compute(*args, **kwargs)
        ]

    monkeypatch.setitem(compare.__globals__, "compute_closures", conjugated)
    with pytest.raises(AssertionError, match="closure mismatch"):
        compare(observation)


def test_comparison_rejects_duplicate_xbeam_closure(observation, monkeypatch):
    compare = comparison_example()["compare_observation"]
    compute = compare.__globals__["compute_closures"]

    def duplicated(*args, **kwargs):
        records = compute(*args, **kwargs)
        valid = next(record for record in records if record.valid[0])
        return [replace(valid, values=valid.values + 0.5), *records]

    monkeypatch.setitem(compare.__globals__, "compute_closures", duplicated)
    with pytest.raises(AssertionError, match="duplicate closure in xBeam"):
        compare(observation)


def test_comparison_rejects_unpinned_input(tmp_path):
    source = tmp_path / "changed.uvfits"
    source.write_bytes(b"not the released data")
    with pytest.raises(ValueError, match="pinned release"):
        comparison_example()["verify_data"](source)
