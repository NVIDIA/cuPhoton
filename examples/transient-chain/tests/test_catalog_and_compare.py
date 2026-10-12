# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import compare_catalogs
import numpy as np
import pandas as pd
import pytest
from transient_chain import catalog

NAMES = (
    "amplitude",
    "sigma_x",
    "sigma_y",
    "theta",
    "x_pos",
    "y_pos",
    "x_neg",
    "y_neg",
)


def _full_table(n, rng):
    t = {
        "x": rng.integers(30, 900, n),
        "y": rng.integers(30, 900, n),
        "peak_abs_sigma": rng.uniform(5, 50, n),
        "pixel_count": rng.integers(3, 40, n),
        "peak_value": rng.normal(0, 100, n),
        "sign": np.sign(rng.normal(size=n)).astype(int),
        "noise_match_factor": np.ones(n, np.float32),
    }
    for name in NAMES:
        t[name] = rng.normal(size=n)
        t[f"{name}_err"] = rng.uniform(size=n)
    t.update(
        converged=np.ones(n, bool),
        chi_square=rng.uniform(size=n),
        reduced_chi_square=rng.uniform(size=n),
        uncertainty_valid=np.ones(n, bool),
        evaluations=rng.integers(1, 200, n),
        valid_pixel_count=np.full(n, 2601),
    )
    t["ra"] = rng.uniform(size=n)
    t["dec"] = rng.uniform(size=n)
    return t


def test_empty_and_full_items_share_one_schema_and_merge_without_promotion():
    rng = np.random.default_rng(0)
    ident = {
        "exposure_id": "1",
        "ccd_id": "R22_S00",
        "band": "r",
        "visit": 1,
        "detector": "R22_S00",
    }
    full = catalog.finish_table(
        _full_table(5, rng),
        5,
        ident,
        "header-wcs-approx",
        "1",
        "R22_S00",
        NAMES,
    )
    empty = catalog.finish_table(
        catalog.empty_table(NAMES),
        0,
        ident,
        "header-wcs-approx",
        "1",
        "R22_S00",
        NAMES,
    )
    assert list(full.columns) == list(empty.columns)
    assert (
        full["x"].dtype == np.int64
        and full["decision"].dtype == bool
        and np.isnan(full["probability"]).all()
    )
    merged = pd.concat([empty, full], ignore_index=True)
    assert (
        merged["x"].dtype == np.int64
        and merged["pixel_count"].dtype == np.int64
        and merged["decision"].dtype == bool
    )
    assert merged["candidate_id"].iloc[0] == "cu-1-R22_S00-00000"
    assert set(catalog.schema(NAMES)) <= set(
        full.columns
    )  # survey extras (visit, detector) come on top


def test_compare_counts_each_reference_once():
    ref = pd.DataFrame(
        {
            "detector": ["D"] * 10,
            "visit": [7] * 10,
            "x": np.arange(10) * 20.0,
            "y": np.zeros(10),
        }
    )
    ours_x = np.concatenate(
        [np.arange(8) * 20.0, np.arange(4) * 20.0 + 0.5]
    )  # 12 rows on 8 reference positions
    ours = pd.DataFrame(
        {
            "ccd_id": ["D"] * 12,
            "exposure_id": ["7"] * 12,
            "x": ours_x,
            "y": np.zeros(12),
        }
    )
    row = compare_catalogs.compare(ours, ref, 1.5)[0]
    assert row["matched"] == 8 and row["duplicate_matches_dropped"] == 4
    assert row["matched_fraction_reference"] == pytest.approx(0.8) and row[
        "matched_fraction_ours"
    ] == pytest.approx(8 / 12)
    assert row["positions_identical"] == 8


def test_compare_tolerates_blank_reference_ids_and_all_nan_columns():
    ref = pd.DataFrame(
        {
            "detector": ["D", "D"],
            "visit": [7, None],
            "x": [1.0, 5.0],
            "y": [1.0, 5.0],
            "peak_abs_sigma": [np.nan, np.nan],
        }
    )
    ours = pd.DataFrame(
        {
            "ccd_id": ["D"],
            "exposure_id": ["7"],
            "x": [1.0],
            "y": [1.0],
            "peak_abs_sigma": [6.0],
        }
    )
    row = compare_catalogs.compare(ours, ref, 1.5)[0]
    assert row["matched"] == 1 and row["peak_abs_sigma_max_rel_diff"] is None
    import json

    json.dumps(row, allow_nan=False)
