#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Repeat test of the transient-chain error bars across repeated exposures
of a detector, without a reference catalog.

A source that does not move lands at the same place in every exposure, so
the scatter of its linked detections divided by their error bars gives a
chi-square per degree of freedom near 1 when the bars are right. The test
links the chain's detections across the exposures of each detector in the
reference frame, drops duplicates within an exposure, and reports the
pooled chi-square per degree of freedom and the median per-group ratio for
two quantities (the dipole center from ``center_*``, the source position
from ``positive_*``), per class derived from ``quality_flags``, for the
full error bar ``<q>_err``, for ``<q>_err_fit`` alone, and with each of the
corr, kfit, kmatch and mis terms dropped in turn.

Positions go to the sky through the catalog's own ``ra``/``dec`` at the
integer candidate pixel: per exposure an affine pixel-to-tangent-plane map
is fitted to those anchors and applied to the sub-pixel position, after the
refine stage's affine from the item's ``summary.json`` (read when it sits
beside the catalog) has moved the position into the reference frame.

What the test cannot see: the reference's noise is the same in every
exposure of a detector and never scatters between them, so every arm keeps a
term the test does not measure; and a bias common to all exposures (a
reference position error, a systematic of the fit) passes unseen.

Usage: repeat_catalogs.py CATALOG [CATALOG ...] [--out report.json]
[--link-px 2.0] [--duplicate-px 1.0]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.stats import chi2 as chi2_dist

TERMS = ("corr", "kfit", "kmatch", "mis")
ARMS = (
    "full",
    "fit only",
    "without corr",
    "without kfit",
    "without kmatch",
    "without mis",
)
QUANTITIES = {
    "dipole center": ("center", ("resolved", "overlapping")),
    "source position": ("positive", ("resolved", "single")),
}
CLASSES = ("invalid", "failed", "overlapping", "single", "resolved")
BLIND_SPOTS = (
    "The reference's noise is shared by every exposure of a detector and "
    "does not scatter between them: each arm keeps a term the test cannot "
    "measure, so a chi-square below 1 by that share is still consistent.",
    "A bias common to all exposures (a reference position error, a "
    "systematic of the fit) moves every detection alike and passes unseen.",
)
MIN_ANCHORS = 3


# --- classes from the quality flags ----------------------------------------


def classify(df: pd.DataFrame) -> np.ndarray:
    """One class per row from ``quality_flags``, ``converged`` and
    ``uncertainty_valid``: ``invalid`` (no usable covariance), ``failed``
    (width out of range or no lobe at the candidate), ``overlapping``,
    ``single`` (exactly one unsupported lobe) or ``resolved``.
    """
    from cuphoton.xfit import DipoleQualityFlag as F

    flags = df["quality_flags"].to_numpy(np.int64)
    neg = (flags & int(F.NEGATIVE_LOBE_UNSUPPORTED)) != 0
    pos = (flags & int(F.POSITIVE_LOBE_UNSUPPORTED)) != 0
    kind = np.full(len(df), "resolved", dtype=object)
    kind[neg != pos] = "single"
    kind[(flags & int(F.LOBES_OVERLAP)) != 0] = "overlapping"
    kind[
        (flags & int(F.WIDTH_OUT_OF_RANGE | F.NO_LOBE_AT_CANDIDATE)) != 0
    ] = "failed"
    valid = df["converged"].to_numpy(bool) & df["uncertainty_valid"].to_numpy(
        bool
    )
    kind[~valid] = "invalid"
    return kind


# --- positions in the reference frame, on the sky --------------------------


def tangent_plane(ra, dec, ra0, dec0):
    """Offsets in arcsec on the tangent plane at ``(ra0, dec0)``."""
    return (
        (np.asarray(ra) - ra0) * np.cos(np.radians(dec0)) * 3600.0,
        (np.asarray(dec) - dec0) * 3600.0,
    )


def fit_anchor_affine(x, y, ex, ey):
    """Affine pixel-to-tangent-plane map ``(M, c)`` with ``[ex, ey] = M
    [x, y] + c`` from the integer-pixel anchors of one exposure; None with
    fewer than three anchors or a degenerate layout.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < MIN_ANCHORS:
        return None
    g = np.column_stack([x, y, np.ones(len(x))])
    if np.linalg.matrix_rank(g) < 3:
        return None
    coef, *_ = np.linalg.lstsq(g, np.column_stack([ex, ey]), rcond=None)
    return coef[:2].T, coef[2]


def read_affine(summary_path: Path | None):
    """The refine stage's ``(A, b)`` from an item summary, identity without
    one or when refinement did not fit.
    """
    if summary_path is None or not summary_path.exists():
        return np.eye(2), np.zeros(2)
    s = json.loads(summary_path.read_text())
    aff = (
        ((s.get("diagnostics") or {}).get("align") or {}).get("refine") or {}
    ).get("affine")
    if not aff:
        return np.eye(2), np.zeros(2)
    return np.asarray(aff["A"], float), np.asarray(aff["b"], float)


def sky_positions(
    df: pd.DataFrame, prefix: str, affine, ra0: float, dec0: float
):
    """Tangent-plane offsets (arcsec) of ``x + <prefix>_x, y + <prefix>_y``
    for one exposure, moved to the reference frame by ``affine`` and to the
    sky by the anchor fit, plus the arcsec per pixel along each pixel axis
    (None when the anchors cannot fix the map).
    """
    ex0, ey0 = tangent_plane(df["ra"].values, df["dec"].values, ra0, dec0)
    fit = fit_anchor_affine(df["x"].values, df["y"].values, ex0, ey0)
    if fit is None:
        return None
    m, c = fit
    a, b = affine
    p = np.column_stack(
        [
            df["x"].values + df[f"{prefix}_x"].values,
            df["y"].values + df[f"{prefix}_y"].values,
        ]
    )
    p = p @ a.T + b
    e = p @ m.T + c
    scale = np.sqrt((m**2).sum(axis=0))  # arcsec per pixel along x, y
    return e[:, 0], e[:, 1], scale


# --- linking ---------------------------------------------------------------


def link_by_position(x, y, radius):
    """Group labels for points linked by any chain of pairs closer than
    ``radius`` (connected components).
    """
    n = len(x)
    pairs = cKDTree(np.column_stack([x, y])).query_pairs(
        radius, output_type="ndarray"
    )
    if len(pairs):
        m = coo_matrix(
            (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)
        )
    else:
        m = coo_matrix((n, n))
    return connected_components(m, directed=False)[1]


def link(
    df: pd.DataFrame, prefix: str, link_px: float, duplicate_px: float
) -> pd.DataFrame:
    """Detections of one quantity linked across exposures: within an
    exposure the highest ``peak_abs_sigma`` within ``duplicate_px`` stays;
    across exposures detections within ``link_px`` form a group; groups
    seen in at least two exposures and at most once per exposure remain.
    ``df`` holds one detector with ``ex``, ``ey`` (arcsec) and
    ``pixel_arcsec`` columns.
    """
    d = df[np.isfinite(df.ex) & np.isfinite(df.ey)].copy()
    if len(d) < 2:
        return d.iloc[:0]
    keep = []
    for _, v in d.groupby("exposure_id", sort=False):
        g = link_by_position(
            v.ex.values, v.ey.values, duplicate_px * v.pixel_arcsec.values[0]
        )
        keep.append(
            v.assign(_g=g)
            .sort_values("peak_abs_sigma", ascending=False)
            .drop_duplicates("_g")
            .drop(columns="_g")
        )
    d = pd.concat(keep)
    pix = float(np.median(d.pixel_arcsec.values))
    d["group"] = [
        f"{d.ccd_id.iloc[0]}/{g}"
        for g in link_by_position(d.ex.values, d.ey.values, link_px * pix)
    ]
    size = d.groupby("group").exposure_id.agg(["size", "nunique"])
    ok = size.index[
        (size["size"] == size["nunique"]) & (size["nunique"] >= 2)
    ]
    return d[d.group.isin(ok)]


# --- statistics ------------------------------------------------------------


def arm_variance(df: pd.DataFrame, q: str, arm: str) -> np.ndarray:
    """Variance (squared catalog units) of quantity ``q`` under ``arm``:
    the full bar, the fit term alone, or the full bar with one term
    removed in quadrature.
    """
    full = df[f"{q}_err"].to_numpy(float) ** 2
    if arm == "full":
        return full
    if arm == "fit only":
        return df[f"{q}_err_fit"].to_numpy(float) ** 2
    if arm.startswith("without "):
        term = arm.removeprefix("without ")
        if term not in TERMS:
            raise ValueError(f"unknown term {term!r}")
        return full - df[f"{q}_err_{term}"].to_numpy(float) ** 2
    raise ValueError(f"unknown arm {arm!r}")


def repeat_chi2(group, x, y, var_x, var_y) -> dict:
    """Scatter of repeated measurements of the same source against their
    predicted variances: per group with ``k >= 2`` detections, chi2 is the
    sum over axes of ``sum((x - weighted mean)^2 / var)`` with ``2 (k - 1)``
    degrees of freedom. ``chi2_per_dof`` pools the groups;
    ``median_group_chi2_ratio`` is the median over groups of chi2 over the
    chi-square median for that group's degrees of freedom, 1 for correct
    variances at any group size.
    """
    df = pd.DataFrame(
        {"g": group, "x": x, "y": y, "vx": var_x, "vy": var_y}
    ).dropna()
    df = df[(df.vx > 0) & (df.vy > 0)]
    chi, dof, per = 0.0, 0, []
    for _, s in df.groupby("g"):
        if len(s) < 2:
            continue
        c = 0.0
        for a, va in (("x", "vx"), ("y", "vy")):
            w = 1.0 / s[va].values
            m = float((w * s[a].values).sum() / w.sum())
            c += float((((s[a].values - m) ** 2) / s[va].values).sum())
        k = 2 * (len(s) - 1)
        chi += c
        dof += k
        per.append(c / chi2_dist.ppf(0.5, k))
    return {
        "groups": len(per),
        "dof": dof,
        "chi2_per_dof": chi / dof if dof else None,
        "median_group_chi2_ratio": float(np.median(per)) if per else None,
    }


def score(linked: pd.DataFrame, prefix: str) -> dict:
    """Every arm on the linked detections whose full error bar is defined
    on both axes, in groups that keep at least two of them, so the arms are
    compared on the same sources.
    """
    ok = np.ones(len(linked), bool)
    for a in ("x", "y"):
        v = arm_variance(linked, f"{prefix}_{a}", "full")
        ok &= np.isfinite(v) & (v > 0)
    s = linked[ok]
    s = s[s.group.map(s.group.value_counts()) >= 2]
    out = {
        "n_groups": int(s.group.nunique()),
        "n_detections": int(len(s)),
        "n_dropped_undefined_error_bar": int((~ok).sum()),
        "exposures_per_group": {
            str(k): int(v)
            for k, v in s.groupby("group").size().value_counts().items()
        },
        "class_counts": {
            str(k): int(v) for k, v in s["class"].value_counts().items()
        },
        "arms": {},
        "by_class": {},
    }

    def rep(t, arm):
        return repeat_chi2(
            t.group.values,
            t.ex.values,
            t.ey.values,
            arm_variance(t, f"{prefix}_x", arm) * t.scale_x.values**2,
            arm_variance(t, f"{prefix}_y", arm) * t.scale_y.values**2,
        )

    for arm in ARMS:
        out["arms"][arm] = rep(s, arm)
    for k, t in s.groupby("class"):
        out["by_class"][str(k)] = {arm: rep(t, arm) for arm in ARMS}
    return out


# --- the report ------------------------------------------------------------


def load(paths) -> pd.DataFrame:
    """Concatenate catalogs; a ``summary.json`` beside a catalog gives its
    exposures their refine affine. Returns rows with ``affine_A``,
    ``affine_b`` columns holding the per-row matrices as lists.
    """
    frames = []
    for p in paths:
        p = Path(p)
        df = pd.read_parquet(p)
        a, b = read_affine(p.with_name("summary.json"))
        df["affine_A"] = [a.tolist()] * len(df)
        df["affine_b"] = [b.tolist()] * len(df)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["exposure_id"] = df["exposure_id"].astype(str)
    df["ccd_id"] = df["ccd_id"].astype(str)
    return df


def repeat(
    df: pd.DataFrame, *, link_px: float = 2.0, duplicate_px: float = 1.0
) -> dict:
    """The repeat test on the chain's rows of one or more detectors."""
    df = df.copy()
    df["class"] = classify(df)
    out: dict = {
        "link_px": link_px,
        "duplicate_px": duplicate_px,
        "n_detections": int(len(df)),
        "detectors": {},
        "blind_spots": list(BLIND_SPOTS),
        "quantities": {},
    }
    for det, d in df.groupby("ccd_id"):
        out["detectors"][str(det)] = {
            "exposures": sorted(d.exposure_id.unique().tolist()),
            "exposures_without_anchor_fit": [],
        }
    for name, (prefix, classes) in QUANTITIES.items():
        rows = []
        for det, d in df[df["class"].isin(classes)].groupby("ccd_id"):
            ra0, dec0 = float(np.median(d.ra)), float(np.median(d.dec))
            for exp, e in d.groupby("exposure_id"):
                anchors = df[(df.ccd_id == det) & (df.exposure_id == exp)]
                a = (
                    np.asarray(e.affine_A.iloc[0]),
                    np.asarray(e.affine_b.iloc[0]),
                )
                sky = sky_positions(anchors, prefix, a, ra0, dec0)
                if sky is None:
                    out["detectors"][str(det)][
                        "exposures_without_anchor_fit"
                    ].append(str(exp))
                    continue
                ex, ey, scale = sky
                sel = anchors.index.get_indexer(e.index)
                rows.append(
                    e.assign(
                        ex=ex[sel],
                        ey=ey[sel],
                        scale_x=scale[0],
                        scale_y=scale[1],
                        pixel_arcsec=float(scale.mean()),
                    )
                )
        if not rows:
            out["quantities"][name] = None
            continue
        linked = pd.concat(
            [
                link(d, prefix, link_px, duplicate_px)
                for _, d in pd.concat(rows).groupby("ccd_id")
            ]
        )
        out["quantities"][name] = (
            None if linked.empty else score(linked, prefix)
        )
    return out


def text_table(report: dict) -> str:
    """A short table: quantity, class, groups, arm, chi2/dof, median ratio."""
    lines = [
        "quantity         class        groups  arm             chi2/dof  "
        "median ratio",
    ]

    def fmt(v):
        return "   -  " if v is None or not math.isfinite(v) else f"{v:6.2f}"

    for name, q in report["quantities"].items():
        if q is None:
            lines.append(f"{name:16s} (no linked detections)")
            continue
        blocks = [("all", q["n_groups"], q["arms"])]
        for k, arms in q["by_class"].items():
            blocks.append((k, arms["full"]["groups"], arms))
        for cls, n, arms in blocks:
            for arm in ARMS:
                r = arms[arm]
                lines.append(
                    f"{name:16s} {cls:12s} {n:6d}  {arm:15s} "
                    f"{fmt(r['chi2_per_dof'])}    "
                    f"{fmt(r['median_group_chi2_ratio'])}"
                )
    lines.append("")
    lines.append("Not seen by this test:")
    lines += [f"- {s}" for s in report["blind_spots"]]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("catalogs", nargs="+", help="catalog.parquet files")
    ap.add_argument("--out", default=None, help="write the JSON report here")
    ap.add_argument(
        "--link-px",
        type=float,
        default=2.0,
        help="link radius across exposures in pixels (default %(default)s)",
    )
    ap.add_argument(
        "--duplicate-px",
        type=float,
        default=1.0,
        help="duplicate radius within an exposure in pixels "
        "(default %(default)s)",
    )
    a = ap.parse_args()
    report = {
        "catalogs": list(a.catalogs),
        **repeat(
            load(a.catalogs), link_px=a.link_px, duplicate_px=a.duplicate_px
        ),
    }
    if a.out:
        Path(a.out).write_text(
            json.dumps(report, indent=1, allow_nan=False) + "\n"
        )
    print(text_table(report))


if __name__ == "__main__":
    main()
