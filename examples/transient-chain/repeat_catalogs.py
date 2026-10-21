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
links the chain's detections across the exposures of each detector, drops
duplicates within an exposure, and scores three quantities: the dipole
center (``center_*``), the source position (``positive_*``) and the source
position of either lobe (``source_*``, the measured lobe whichever its
sign). Each is scored overall and per class, for the full error bar
``<q>_err``, for ``<q>_err_fit`` alone, and with each of the corr, kfit,
kmatch and mis terms dropped in turn (``ARMS``).

Classes come from ``quality_flags``, ``converged`` and ``uncertainty_valid``
(``classify``): ``resolved`` (two supported lobes), ``overlapping``,
``single`` (a brightening source, only the positive lobe supported),
``fading`` (only the negative lobe supported), ``failed`` (width out of
range or no lobe at the candidate) and ``invalid`` (no usable covariance).
The center is scored on resolved and overlapping candidates, the source
position on resolved and single ones, and the either-lobe position on
resolved, single and fading ones; a fading source whose one-lobe fit fell
back to the dipole has no ``source_*`` error bar and is dropped with the
other rows whose bar is undefined.

Positions reach the sky in two steps. The refine stage's affine from the
item's ``summary.json`` (read by ``load`` when it sits beside the catalog)
moves the sub-pixel position into the reference frame. The catalog's own
``ra``/``dec`` at the integer candidate pixel then anchors it: per exposure
an affine pixel-to-tangent-plane map is fitted to those anchors, and a
detection's sky position is its anchor's plus the linear part of that map
applied to the offset from the integer pixel. The header WCS's distortion
stays in the anchors, and dithers and rotations between exposures are not
scatter.

What the test cannot see: the reference's noise is the same in every
exposure of a detector and never scatters between them, so an arm that
includes this noise keeps a term the test does not measure; and a bias
common to all exposures (a reference position error, a systematic of the
fit) passes unseen.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.stats import chi2 as chi2_dist

TERMS = ("corr", "kfit", "kmatch", "mis")
# arm -> the terms whose squares, with the fit term's, make its variance
ARMS: dict[str, tuple[str, ...]] = {
    "full": TERMS,
    "fit only": (),
    **{f"without {t}": tuple(u for u in TERMS if u != t) for t in TERMS},
}
QUANTITIES = {
    "dipole center": ("center", ("resolved", "overlapping")),
    "source position": ("positive", ("resolved", "single")),
    "source position (either lobe)": (
        "source",
        ("resolved", "single", "fading"),
    ),
}
# the source_* columns copy the positive_* bar where it is defined, so their
# arms read the positive_* terms
TERM_PREFIX = {"source": "positive"}
CLASSES = ("invalid", "failed", "overlapping", "fading", "single", "resolved")
BLIND_SPOTS = (
    "The reference's noise is shared by every exposure of a detector and "
    "does not scatter between them: an arm that includes this noise can "
    "have a chi-square below 1 by that share even with correct error bars.",
    "A bias common to all exposures (a reference position error, a "
    "systematic of the fit) moves every detection alike and passes unseen.",
)
MIN_ANCHORS = 10
# of the anchor design on centered, unit-scaled pixel coordinates
MAX_ANCHOR_CONDITION = 1e3


# --- classes from the quality flags ----------------------------------------


def classify(df: pd.DataFrame) -> np.ndarray:
    """One class per row (``CLASSES``) from ``quality_flags``, ``converged``
    and ``uncertainty_valid``.
    """
    from cuphoton.xfit import DipoleQualityFlag as F

    flags = df["quality_flags"].to_numpy(np.int64)
    neg = (flags & int(F.NEGATIVE_LOBE_UNSUPPORTED)) != 0
    pos = (flags & int(F.POSITIVE_LOBE_UNSUPPORTED)) != 0
    kind = np.full(len(df), "resolved", dtype=object)
    kind[neg & ~pos] = "single"
    kind[pos & ~neg] = "fading"
    kind[(flags & int(F.LOBES_OVERLAP)) != 0] = "overlapping"
    kind[
        (neg & pos)
        | (
            (
                flags
                & int(
                    F.WIDTH_OUT_OF_RANGE
                    | F.NO_LOBE_AT_CANDIDATE
                    | F.LOBE_OUTSIDE_STAMP
                )
            )
            != 0
        )
    ] = "failed"
    valid = df["converged"].to_numpy(bool) & df["uncertainty_valid"].to_numpy(
        bool
    )
    kind[~valid] = "invalid"
    if "fit_model" in df:
        accepted_single = df["fit_model"].to_numpy() == "single"
        kind[accepted_single & neg & ~pos] = "single"
        kind[accepted_single & pos & ~neg] = "fading"
    return kind


# --- positions in the reference frame, on the sky --------------------------


def tangent_plane(ra, dec, ra0, dec0):
    """Offsets in arcsec on the tangent plane at ``(ra0, dec0)``; the right
    ascension difference is wrapped into (-180, 180] degrees.
    """
    d = ((np.asarray(ra, float) - ra0 + 180.0) % 360.0) - 180.0
    return (
        d * np.cos(np.radians(dec0)) * 3600.0,
        (np.asarray(dec, float) - dec0) * 3600.0,
    )


@dataclass(frozen=True)
class AnchorFit:
    """Affine pixel-to-tangent-plane map ``[ex, ey] = matrix [x, y] +
    offset`` of one exposure, with the anchors used and the rms of their
    residuals in arcsec.
    """

    matrix: np.ndarray
    offset: np.ndarray
    anchors: int
    residual_arcsec: float

    @property
    def pixel_arcsec(self) -> np.ndarray:
        """Arcsec per pixel along the x and y pixel axes."""
        return np.sqrt((self.matrix**2).sum(axis=0))


def fit_anchor_affine(x, y, ex, ey) -> AnchorFit | None:
    """The affine map from the integer-pixel anchors of one exposure; None
    with fewer than ``MIN_ANCHORS`` finite anchors or a layout whose design on
    centered, unit-scaled coordinates is conditioned worse than
    ``MAX_ANCHOR_CONDITION`` (anchors along a line cannot fix the map).
    """
    anchors = np.column_stack([x, y, ex, ey]).astype(float)
    x, y, ex, ey = anchors[np.isfinite(anchors).all(axis=1)].T
    if len(x) < MIN_ANCHORS:
        return None
    centered = np.column_stack([x - x.mean(), y - y.mean()])
    spread = centered.std(axis=0).max()
    if not spread > 0:
        return None
    probe = np.column_stack([centered / spread, np.ones(len(x))])
    if np.linalg.cond(probe) > MAX_ANCHOR_CONDITION:
        return None
    g = np.column_stack([x, y, np.ones(len(x))])
    target = np.column_stack([ex, ey])
    coef, *_ = np.linalg.lstsq(g, target, rcond=None)
    residual = float(np.sqrt(((g @ coef - target) ** 2).sum(axis=1).mean()))
    return AnchorFit(coef[:2].T, coef[2], int(len(x)), residual)


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


def transform_kind(affine) -> str:
    """``"identity"``, ``"shift"`` or ``"affine"`` for a refine ``(A, b)``."""
    a, b = affine
    if np.allclose(a, np.eye(2)):
        return "identity" if np.allclose(b, 0.0) else "shift"
    return "affine"


def sky_positions(
    df: pd.DataFrame, prefix: str, affine, fit: AnchorFit, ra0, dec0
):
    """Tangent-plane offsets (arcsec) of ``x + <prefix>_x, y + <prefix>_y``
    for one exposure: the anchor's own offsets at the integer pixel plus the
    linear part of ``fit`` applied to the sub-pixel position, after
    ``affine`` has moved that position into the reference frame.
    """
    ex0, ey0 = tangent_plane(df["ra"].values, df["dec"].values, ra0, dec0)
    pixel = np.column_stack([df["x"].values, df["y"].values]).astype(float)
    p = pixel + np.column_stack(
        [df[f"{prefix}_x"].values, df[f"{prefix}_y"].values]
    )
    a, b = affine
    p = p @ a.T + b
    e = np.column_stack([ex0, ey0]) + (p - pixel) @ fit.matrix.T
    return e[:, 0], e[:, 1]


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
    df: pd.DataFrame, link_px: float, duplicate_px: float
) -> pd.DataFrame:
    """Detections linked across exposures: within an exposure the highest
    ``peak_abs_sigma`` within ``duplicate_px`` stays; across exposures
    detections within ``link_px`` form a group; groups seen in at least two
    exposures and at most once per exposure remain. ``df`` holds one
    detector with ``ex``, ``ey`` (arcsec) and ``pixel_arcsec`` columns.
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
    """Variance (squared catalog units) of quantity ``q`` under ``arm``: the
    fit term's square plus the squares of the terms ``ARMS[arm]`` keeps.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; one of {list(ARMS)}")
    var = df[f"{q}_err_fit"].to_numpy(float) ** 2
    prefix, axis = q.rsplit("_", 1)
    q = f"{TERM_PREFIX.get(prefix, prefix)}_{axis}"
    for t in ARMS[arm]:
        var = var + df[f"{q}_err_{t}"].to_numpy(float) ** 2
    return var


def repeat_chi2(group, x, y, var_x, var_y, cov_xy=None) -> dict:
    """Scatter of repeated measurements of the same source against their
    predicted variances: per group with ``k >= 2`` detections, chi2 is the
    the covariance-weighted squared residual about the fitted mean with
    ``2 (k - 1)`` degrees of freedom. ``cov_xy`` defaults to zero.
    ``chi2_per_dof`` pools the groups;
    ``median_group_chi2_ratio`` is the median over groups of chi2 over the
    chi-square median for that group's degrees of freedom, 1 for correct
    variances at any group size.
    """
    df = pd.DataFrame(
        {
            "g": group,
            "x": x,
            "y": y,
            "vx": var_x,
            "vy": var_y,
            "cxy": 0.0 if cov_xy is None else cov_xy,
        }
    ).dropna()
    finite = np.isfinite(df[["x", "y", "vx", "vy", "cxy"]]).all(axis=1)
    determinant = df.vx * df.vy - df.cxy**2
    df = df[finite & (df.vx > 0) & (df.vy > 0) & (determinant > 0)]
    chi, dof, per = 0.0, 0, []
    for _, s in df.groupby("g"):
        if len(s) < 2:
            continue
        covariance = np.zeros((len(s), 2, 2))
        covariance[:, 0, 0] = s.vx.values
        covariance[:, 1, 1] = s.vy.values
        covariance[:, 0, 1] = covariance[:, 1, 0] = s.cxy.values
        precision = np.linalg.inv(covariance)
        positions = s[["x", "y"]].to_numpy()
        mean = np.linalg.solve(
            precision.sum(axis=0),
            np.einsum("nij,nj->i", precision, positions),
        )
        residual = positions - mean
        c = float(np.einsum("ni,nij,nj->", residual, precision, residual))
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
    and positive on both axes, in groups that keep at least two of them.
    Each arm then excludes rows whose own covariance is unusable.
    """
    ok = np.ones(len(linked), bool)
    for a in ("x", "y"):
        v = linked[f"{prefix}_{a}_err"].to_numpy(float)
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
        variance_x = arm_variance(t, f"{prefix}_x", arm)
        variance_y = arm_variance(t, f"{prefix}_y", arm)
        covariance_xy = t.get(
            f"{TERM_PREFIX.get(prefix, prefix)}_xy_cov",
            pd.Series(0.0, index=t.index),
        )
        covariance_xy = covariance_xy.fillna(0.0).to_numpy(float)
        jacobian = t[
            ["jacobian_xx", "jacobian_xy", "jacobian_yx", "jacobian_yy"]
        ]
        jacobian = jacobian.to_numpy().reshape(-1, 2, 2)
        covariance = np.zeros((len(t), 2, 2))
        covariance[:, 0, 0] = variance_x
        covariance[:, 1, 1] = variance_y
        covariance[:, 0, 1] = covariance[:, 1, 0] = covariance_xy
        sky_covariance = jacobian @ covariance @ jacobian.transpose(0, 2, 1)
        return repeat_chi2(
            t.group.values,
            t.ex.values,
            t.ey.values,
            sky_covariance[:, 0, 0],
            sky_covariance[:, 1, 1],
            sky_covariance[:, 0, 1],
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


def place(df: pd.DataFrame, detectors: dict) -> pd.DataFrame:
    """Sky positions of both quantities for every row, the anchors fitted
    once per detector and exposure. Fills ``detectors`` with each
    exposure's anchor count, transform and fit (or lists it under
    ``exposures_without_anchor_fit``); returns the rows that were placed,
    with ``<prefix>_ex``, ``<prefix>_ey``, ``scale_x``, ``scale_y`` and
    ``pixel_arcsec``.
    """
    placed = []
    for det, d in df.groupby("ccd_id"):
        sky = d[np.isfinite(d[["ra", "dec"]]).all(axis=1)]
        angles = np.radians(sky.ra.to_numpy(float))
        ra0 = (
            float(
                np.degrees(
                    np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
                )
            )
            % 360.0
            if len(sky)
            else math.nan
        )
        dec0 = float(np.median(sky.dec)) if len(sky) else math.nan
        info = {
            "exposures": sorted(d.exposure_id.unique().tolist()),
            "exposures_without_anchor_fit": [],
            "anchors": {},
        }
        for exp, e in d.groupby("exposure_id"):
            affine = (
                np.asarray(e.affine_A.iloc[0], float),
                np.asarray(e.affine_b.iloc[0], float),
            )
            ex0, ey0 = tangent_plane(e.ra.values, e.dec.values, ra0, dec0)
            fit = fit_anchor_affine(e.x.values, e.y.values, ex0, ey0)
            rec = {
                "detections": int(len(e)),
                "transform": transform_kind(affine),
                "fit": None,
            }
            info["anchors"][str(exp)] = rec
            if fit is None:
                info["exposures_without_anchor_fit"].append(str(exp))
                continue
            scale = fit.pixel_arcsec
            rec["fit"] = {
                "anchors": fit.anchors,
                "residual_arcsec": fit.residual_arcsec,
                "pixel_arcsec": scale.tolist(),
            }
            cols = {}
            jacobian = fit.matrix @ affine[0]
            for row, axis in enumerate("xy"):
                for column, source_axis in enumerate("xy"):
                    cols[f"jacobian_{axis}{source_axis}"] = jacobian[
                        row, column
                    ]
            for prefix, _ in QUANTITIES.values():
                ex, ey = sky_positions(e, prefix, affine, fit, ra0, dec0)
                cols[f"{prefix}_ex"], cols[f"{prefix}_ey"] = ex, ey
            placed.append(
                e.assign(
                    **cols,
                    scale_x=scale[0],
                    scale_y=scale[1],
                    pixel_arcsec=float(scale.mean()),
                )
            )
        detectors[str(det)] = info
    return pd.concat(placed) if placed else df.iloc[:0]


def repeat(
    df: pd.DataFrame, *, link_px: float = 2.0, duplicate_px: float = 1.0
) -> dict:
    """The repeat test on the chain's rows of one or more detectors, as
    :func:`load` returns them.
    """
    if not {"affine_A", "affine_b"} <= set(df.columns):
        raise ValueError(
            "rows carry no affine_A and affine_b columns: read the catalogs "
            "with load(), which adds each item's refine affine"
        )
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
    placed = place(df, out["detectors"])
    for name, (prefix, classes) in QUANTITIES.items():
        rows = placed[placed["class"].isin(classes)]
        if rows.empty:
            out["quantities"][name] = None
            continue
        rows = rows.assign(ex=rows[f"{prefix}_ex"], ey=rows[f"{prefix}_ey"])
        linked = pd.concat(
            [
                link(d, link_px, duplicate_px)
                for _, d in rows.groupby("ccd_id")
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
