#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Compare a transient-chain catalog with a reference catalog of the same
detector-exposures.

Reference: a CSV with pixel positions and identity columns, or another
run's catalog.parquet.
Candidates are matched by pixel position within --radius, one reference row
per candidate and
each reference row counted once. Reference detector-exposures with no rows in
ours are reported with n_ours = 0. Reports counts, position agreement, and
relative differences
of the xFit parameters on converged matched rows.

Usage: compare_catalogs.py --ours catalog.parquet --reference reference.csv
[--out report.json] [--radius 1.5]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

FIT = (
    "amplitude",
    "sigma_x",
    "sigma_y",
    "theta",
    "x_pos",
    "y_pos",
    "x_neg",
    "y_neg",
    "chi_square",
)
ID_COLUMNS = (("ccd_id", "exposure_id"), ("detector", "visit"))


def _ids(df: pd.DataFrame) -> tuple[str, str]:
    for ccd, exp in ID_COLUMNS:
        if ccd in df and exp in df:
            return ccd, exp
    raise ValueError(
        "catalog needs (ccd_id, exposure_id) or (detector, visit) columns"
    )


def _id_str(v) -> str:
    """Identity values as strings: 7, 7.0 and "7" all compare equal; blanks
    become "".
    """
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def _finite(v: float) -> float | None:
    return None if v is None or not math.isfinite(v) else float(v)


def _max_rel_diff(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) == 0:
        return None
    d = np.abs(x - y) / np.maximum(np.abs(y), 1e-30)
    return _finite(float(np.nanmax(d))) if np.isfinite(d).any() else None


def match(ours: pd.DataFrame, ref: pd.DataFrame, radius: float):
    """Nearest reference row within radius for each of ours; keep only the
    closest of ours per reference row.
    """
    d, j = cKDTree(ref[["x", "y"]].to_numpy(float)).query(
        ours[["x", "y"]].to_numpy(float), distance_upper_bound=radius
    )
    m = np.isfinite(d)
    order = np.lexsort((d, j))  # by reference row, then distance
    seen = set()
    keep = np.zeros(len(ours), bool)
    duplicates = 0
    for i in order:
        if not m[i]:
            continue
        if j[i] in seen:
            duplicates += 1
            continue
        seen.add(j[i])
        keep[i] = True
    return keep, j, d, duplicates


def compare(
    ours: pd.DataFrame, ref: pd.DataFrame, radius: float
) -> list[dict]:
    o_ccd, o_exp = _ids(ours)
    r_ccd, r_exp = _ids(ref)
    ref = ref.copy()
    ref[r_exp] = ref[r_exp].map(_id_str)
    ref[r_ccd] = ref[r_ccd].map(_id_str)
    groups = {
        (_id_str(ccd), _id_str(exp)): o
        for (ccd, exp), o in ours.groupby([o_ccd, o_exp])
    }
    # reference detector-exposures where we kept nothing are reported too
    keys = set(groups) | {
        k for k in zip(ref[r_ccd], ref[r_exp], strict=True) if all(k)
    }
    rows = []
    for ccd, exp in sorted(keys):
        o = groups.get((ccd, exp), ours.iloc[:0])
        r = ref[(ref[r_ccd] == ccd) & (ref[r_exp] == exp)]
        row: dict = {
            "ccd_id": ccd,
            "exposure_id": exp,
            "n_ours": int(len(o)),
            "n_reference": int(len(r)),
        }
        if len(r) and not len(o):
            row["matched"] = 0
            row["matched_fraction_reference"] = 0.0
        if len(r) and len(o):
            keep, j, d, dup = match(o, r, radius)
            row["matched"] = int(keep.sum())
            row["duplicate_matches_dropped"] = int(dup)
            row["matched_fraction_ours"] = float(keep.mean())
            row["matched_fraction_reference"] = float(keep.sum() / len(r))
            row["positions_identical"] = int((d[keep] == 0).sum())
            om, rm = (
                o[keep].reset_index(drop=True),
                r.iloc[j[keep]].reset_index(drop=True),
            )
            for c in ("peak_abs_sigma", "pixel_count"):
                if c in om and c in rm:
                    row[f"{c}_max_rel_diff"] = _max_rel_diff(
                        om[c].to_numpy(float), rm[c].to_numpy(float)
                    )
            if "converged" in om and all(c in rm for c in FIT[:8]):
                conv = (
                    om["converged"].astype(bool).to_numpy()
                    & rm.get("converged", pd.Series(True, index=rm.index))
                    .astype(bool)
                    .to_numpy()
                )
                row["fit_max_rel_diff_converged"] = {
                    c: _max_rel_diff(
                        om[c].to_numpy(float)[conv],
                        rm[c].to_numpy(float)[conv],
                    )
                    for c in FIT
                    if c in om and c in rm
                }
                row["n_converged_compared"] = int(conv.sum())
            if "probability" in om and "score_mean" in rm:
                row["score_abs_diff_median"] = _finite(
                    float(
                        np.nanmedian(
                            np.abs(
                                om["probability"].to_numpy(float)
                                - rm["score_mean"].to_numpy(float)
                            )
                        )
                    )
                )
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--ours", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument(
        "--out",
        default=None,
        help="write the JSON report here as well as printing it",
    )
    ap.add_argument(
        "--radius",
        type=float,
        default=1.5,
        help="match radius in pixels (default %(default)s)",
    )
    a = ap.parse_args()
    ours = pd.read_parquet(a.ours)
    ref = (
        pd.read_csv(a.reference)
        if a.reference.endswith(".csv")
        else pd.read_parquet(a.reference)
    )
    report = {
        "ours": a.ours,
        "reference": a.reference,
        "radius_px": a.radius,
        "per_visit": compare(ours, ref, a.radius),
    }
    text = json.dumps(report, indent=1, allow_nan=False) + "\n"
    if a.out:
        Path(a.out).write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
