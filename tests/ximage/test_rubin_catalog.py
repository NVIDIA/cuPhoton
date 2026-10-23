# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv

import pytest

from cuphoton.ximage.data.rubin_catalog import (
    load_visit_file,
    parse_visit_text,
    validate_rubin_catalog_export,
)


@pytest.mark.parametrize("observed_visits", [(101, 102), (101,)])
def test_validate_local_catalog_reports_product_coverage(
    tmp_path, observed_visits
) -> None:
    write_csv(
        tmp_path / "object.csv",
        [
            "objectId",
            "coord_ra",
            "coord_dec",
            "r_psfFlux",
            "r_psfFluxErr",
            "r_cModelFlux",
            "r_cModelFluxErr",
            "r_extendedness",
        ],
        [
            {
                "objectId": "1",
                "coord_ra": "10.0",
                "coord_dec": "-5.0",
                "r_psfFlux": "100.0",
                "r_psfFluxErr": "2.0",
                "r_cModelFlux": "101.0",
                "r_cModelFluxErr": "3.0",
                "r_extendedness": "0.0",
            }
        ],
    )
    write_csv(
        tmp_path / "source.csv",
        [
            "sourceId",
            "visit",
            "band",
            "ra",
            "dec",
            "psfFlux",
            "psfFluxErr",
            "ap12Flux",
            "ap25Flux",
            "extendedness",
            "sky",
        ],
        [
            {
                "sourceId": str(index),
                "visit": str(visit),
                "band": "r",
                "ra": "10.0",
                "dec": "-5.0",
                "psfFlux": "100.0",
                "psfFluxErr": "2.0",
                "ap12Flux": "90.0",
                "ap25Flux": "95.0",
                "extendedness": "0.0",
                "sky": "1.0",
            }
            for index, visit in enumerate(observed_visits, start=1)
        ],
    )

    (tmp_path / "export_manifest.json").write_text("Unrelated non-JSON note")
    result = validate_rubin_catalog_export(
        input_dir=tmp_path,
        products=["object", "source"],
        visits=[101, 102],
        band="r",
    )

    assert result["artifact_kind"] == "rubin_catalog_validation"
    assert "manifest_path" not in result
    expected_complete = observed_visits == (101, 102)
    assert result["ok"] is expected_complete
    assert result["product_results"]["object"]["row_count"] == 1
    source = result["product_results"]["source"]
    assert source["ok"] is expected_complete
    assert source["row_count"] == len(observed_visits)
    assert source["visits"] == list(observed_visits)
    assert source["missing_expected_visits"] == (
        [] if expected_complete else [102]
    )


def test_validate_local_catalog_reports_missing_columns(tmp_path) -> None:
    write_csv(
        tmp_path / "object.csv",
        ["objectId", "coord_ra", "coord_dec"],
        [{"objectId": "1", "coord_ra": "10.0", "coord_dec": "-5.0"}],
    )

    result = validate_rubin_catalog_export(
        input_dir=tmp_path,
        products=["object"],
        band="r",
    )

    assert result["ok"] is False
    assert (
        "r_psfFlux" in result["product_results"]["object"]["missing_columns"]
    )


def test_parse_visit_text_and_file(tmp_path) -> None:
    visits_file = tmp_path / "visits.txt"
    visits_file.write_text("# visits\n102\n101 trailing text\n")

    assert parse_visit_text("7, 9\n") == [7, 9]
    assert load_visit_file(visits_file) == [102, 101]


def write_csv(path, fieldnames, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
