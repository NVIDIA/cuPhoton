# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import json
import sys

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
        quality_flags=rng.integers(0, 128, n),
    )
    for name in catalog.UNCERTAINTY_COLUMNS:
        if name not in (
            "quality_flags",
            "psf_flux",
            "psf_flux_err_fit",
            "psf_flux_err",
        ):
            t[name] = rng.normal(size=n)
    for name, (dtype, _, _) in catalog.BUDGET_COLUMNS.items():
        if name.startswith("psf_flux"):
            continue
        t[name] = {
            "bool": np.ones(n, bool),
            "object": np.array(["survey"] * n, dtype=object),
        }.get(dtype, rng.uniform(size=n))
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
        and full["quality_flags"].dtype == np.int64
        and np.isnan(full["psf_flux"]).all()
        and np.isnan(full["psf_flux_err"]).all()
    )
    for q in ("center_x", "center_y", "positive_x", "positive_y", "psf_flux"):
        assert {f"{q}_err_fit", f"{q}_err"} <= set(full.columns)
        assert {f"{q}_err_{t}" for t in catalog.BUDGET_TERMS} <= set(
            full.columns
        )
    assert np.isnan(full["psf_flux_err_kfit"]).all()
    assert full["uq_valid"].dtype == bool and empty["uq_valid"].dtype == bool
    assert full["sky_err_arcsec"].dtype == np.float64
    assert str(full["sky_err_source"].dtype) in ("object", "str")
    # the source columns are appended at the end of the uq block
    budget = list(catalog.BUDGET_COLUMNS)
    assert budget[-6:] == [
        "source_x",
        "source_y",
        "source_x_err_fit",
        "source_y_err_fit",
        "source_x_err",
        "source_y_err",
    ]
    assert full["source_x_err"].dtype == np.float64
    assert list(full.columns).index("source_y_err") + 1 == list(
        full.columns
    ).index("logit")
    assert full["sky_err_source"].iloc[0] == "survey"
    # the columns #37 defined keep their order; the budget's follow them
    names = list(full.columns)
    assert names.index("psf_flux_err") < names.index("center_x_err_corr")
    assert names.index("sky_err_source") < names.index("logit")
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


def test_compare_reports_reference_exposures_we_lost_entirely():
    ref = pd.DataFrame(
        {
            "detector": ["D", "D", "E"],
            "visit": [7, 7, 8],
            "x": [1.0, 5.0, 3.0],
            "y": [1.0, 5.0, 3.0],
        }
    )
    ours = pd.DataFrame(
        {"ccd_id": ["D"], "exposure_id": ["7"], "x": [1.0], "y": [1.0]}
    )
    rows = {
        (r["ccd_id"], r["exposure_id"]): r
        for r in compare_catalogs.compare(ours, ref, 1.5)
    }
    assert set(rows) == {("D", "7"), ("E", "8")}
    lost = rows[("E", "8")]
    assert lost["n_ours"] == 0 and lost["n_reference"] == 1
    assert lost["matched"] == 0 and lost["matched_fraction_reference"] == 0.0


@pytest.mark.parametrize(
    "name", ["x", "candidate_id", "amplitude", "astrometry"]
)
def test_catalog_rejects_identity_extras_that_overwrite_measurements(name):
    with pytest.raises(ValueError, match="conflicts with the catalog schema"):
        catalog.finish_table(
            _full_table(1, np.random.default_rng(0)),
            1,
            {"exposure_id": "1", "ccd_id": "D", "band": "r", name: "extra"},
            "header-wcs",
            "1",
            "D",
            NAMES,
        )


def test_compare_reports_exposures_absent_from_reference():
    ours = pd.DataFrame(
        {"ccd_id": ["D"], "exposure_id": ["7"], "x": [1.0], "y": [1.0]}
    )
    ref = ours.iloc[:0]
    row = compare_catalogs.compare(ours, ref, 1.5)[0]
    assert row["matched"] == 0
    assert row["matched_fraction_ours"] == 0.0
    assert row["n_reference"] == 0


def test_compare_cli_preserves_zero_prefixed_csv_identity(
    tmp_path, monkeypatch, capsys
):
    ours = tmp_path / "ours.parquet"
    pd.DataFrame(
        {"ccd_id": ["001"], "exposure_id": ["0007"], "x": [1.0], "y": [2.0]}
    ).to_parquet(ours)
    reference = tmp_path / "reference.csv"
    reference.write_text("detector,visit,x,y\n001,0007,1.0,2.0\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_catalogs",
            "--ours",
            str(ours),
            "--reference",
            str(reference),
        ],
    )
    compare_catalogs.main()
    rows = json.loads(capsys.readouterr().out)["per_visit"]
    assert len(rows) == 1
    assert rows[0]["ccd_id"] == "001" and rows[0]["exposure_id"] == "0007"
    assert rows[0]["matched"] == 1
