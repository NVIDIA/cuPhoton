# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Catalog-only dependencies fail with their supported install profile."""

from __future__ import annotations

import sys

import pytest

from cuphoton.xscan import butler, lsstcomcam


@pytest.mark.parametrize(
    "operation", ["registry-read", "registry-write", "candidates"]
)
def test_catalog_operations_report_missing_pandas(
    tmp_path, monkeypatch, operation
):
    path = tmp_path / "catalog.csv"
    path.write_text("visit,band\n1,i\n")
    monkeypatch.setitem(sys.modules, "pandas", None)

    with pytest.raises(RuntimeError, match=r"pandas.*cuphoton\[catalog\]"):
        if operation == "registry-read":
            butler._load_registry_dataframe(path)
        elif operation == "registry-write":
            butler._write_registry_dataframe([{"visit": 1}], path)
        else:
            lsstcomcam.load_lsstcomcam_candidate_catalog(path)


def test_registry_query_reports_missing_numexpr(monkeypatch):
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"visit": [1, 2]})
    monkeypatch.setitem(sys.modules, "numexpr", None)

    with pytest.raises(RuntimeError, match=r"numexpr.*cuphoton\[catalog\]"):
        butler._safe_query(frame, "visit > 1", source="test registry")
