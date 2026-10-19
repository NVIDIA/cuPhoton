# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The repeat test on synthetic catalogs of repeated exposures: static
sources with known scatter, duplicates, and the term arithmetic.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import repeat_catalogs as rc
from astropy.wcs import WCS
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
PIX = 0.2 / 3600  # degrees per pixel
NX = NY = 2000


def _wcs(dx, dy, rot_deg=0.0):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [30.0, -10.0]
    w.wcs.crpix = [NX / 2 + dx, NY / 2 + dy]
    a = np.radians(rot_deg)
    w.wcs.cd = PIX * np.array(
        [[-np.cos(a), np.sin(a)], [np.sin(a), np.cos(a)]]
    )
    return w


def _catalog(
    rng,
    sky,
    exposure,
    sigma,
    *,
    err_scale=1.0,
    terms=(0.2, 0.2, 0.0, 0.1),
    flags=None,
    wcs=None,
    duplicates=(),
):
    """One exposure's catalog of static sources at ``sky`` (deg), detected
    at the nearest pixel with sub-pixel offsets scattered by ``sigma`` px
    per axis; the quoted error bar is ``sigma * err_scale``, split into the
    fit term and the four others by ``terms`` (shares of the variance).
    """
    w = wcs or _wcs(*rng.uniform(-30, 30, 2))
    px = w.all_world2pix(sky, 0)
    n = len(px)
    ix, iy = np.rint(px[:, 0]).astype(int), np.rint(px[:, 1]).astype(int)
    true = px - np.column_stack([ix, iy])
    rows = {
        "x": ix,
        "y": iy,
        "peak_abs_sigma": rng.uniform(8, 40, n),
        "pixel_count": np.full(n, 9),
        "peak_value": np.full(n, 100.0),
        "sign": np.ones(n, int),
        "noise_match_factor": np.ones(n, np.float32),
        "converged": np.ones(n, bool),
        "chi_square": np.ones(n),
        "reduced_chi_square": np.ones(n),
        "uncertainty_valid": np.ones(n, bool),
        "evaluations": np.full(n, 10),
        "valid_pixel_count": np.full(n, 2601),
        "quality_flags": np.zeros(n, np.int64) if flags is None else flags,
    }
    for name in NAMES:
        rows[name] = np.ones(n)
        rows[f"{name}_err"] = np.ones(n)
    for name in ("separation", "lobe_flux"):
        rows[name] = np.ones(n)
        rows[f"{name}_err"] = np.ones(n)
    bar = sigma * err_scale
    share = np.array(terms, float)
    fit = bar * np.sqrt(1.0 - share.sum())
    for prefix in ("center", "positive"):
        off = true + rng.normal(0.0, sigma, (n, 2))
        for j, a in enumerate("xy"):
            rows[f"{prefix}_{a}"] = off[:, j]
            rows[f"{prefix}_{a}_err_fit"] = np.full(n, fit)
            rows[f"{prefix}_{a}_err"] = np.full(n, bar)
            for t, s in zip(rc.TERMS, share, strict=True):
                rows[f"{prefix}_{a}_err_{t}"] = np.full(n, bar * np.sqrt(s))
        rows[f"{prefix}_xy_cov"] = np.zeros(n)
    rows["uq_valid"] = np.ones(n, bool)
    rows["sky_err_arcsec"] = np.full(n, np.nan)
    rows["sky_err_source"] = np.array([None] * n, dtype=object)
    sky_int = w.all_pix2world(np.column_stack([ix, iy]).astype(float), 0)
    rows["ra"], rows["dec"] = sky_int[:, 0], sky_int[:, 1]
    df = catalog.finish_table(
        rows,
        n,
        {"exposure_id": str(exposure), "ccd_id": "D1", "band": "r"},
        "header-wcs-approx",
        str(exposure),
        "D1",
        NAMES,
    )
    for j, peak in duplicates:  # a fainter second detection 0.8 px away
        d = df.iloc[[j]].copy()
        d["positive_x"] += 0.8
        d["center_x"] += 0.8
        d["peak_abs_sigma"] = peak
        df = pd.concat([df, d], ignore_index=True)
    df["affine_A"] = [np.eye(2).tolist()] * len(df)
    df["affine_b"] = [[0.0, 0.0]] * len(df)
    return df


def _sky(rng, n=150):
    w = _wcs(0.0, 0.0)
    px = np.column_stack(
        [rng.uniform(100, NX - 100, n), rng.uniform(100, NY - 100, n)]
    )
    return w.all_pix2world(px, 0)


@pytest.mark.parametrize("err_scale, want", [(1.0, 1.0), (0.5, 4.0)])
def test_honest_error_bars_give_unit_chi2_and_halved_ones_four(
    err_scale, want
):
    rng = np.random.default_rng(0)
    sky = _sky(rng)
    df = pd.concat(
        [
            _catalog(rng, sky, e, 0.05, err_scale=err_scale)
            for e in (101, 102, 103, 104)
        ],
        ignore_index=True,
    )
    rep = rc.repeat(df)
    for name in ("dipole center", "source position"):
        q = rep["quantities"][name]
        assert q["n_groups"] == 150 and q["n_detections"] == 600
        assert q["exposures_per_group"] == {"4": 150}
        full = q["arms"]["full"]
        assert full["chi2_per_dof"] == pytest.approx(want, rel=0.12)
        assert full["median_group_chi2_ratio"] == pytest.approx(want, rel=0.2)
        # the arms shrink the variance by the term's share: 50% fit alone
        assert q["arms"]["fit only"]["chi2_per_dof"] == pytest.approx(
            full["chi2_per_dof"] / 0.5, rel=1e-9
        )
        assert q["arms"]["without kmatch"]["chi2_per_dof"] == pytest.approx(
            full["chi2_per_dof"], rel=1e-9
        )
        assert q["by_class"]["resolved"]["full"]["groups"] == 150
    assert rep["blind_spots"] and rep["detectors"]["D1"]["exposures"] == [
        "101",
        "102",
        "103",
        "104",
    ]


def test_rotated_and_dithered_exposures_link_through_the_anchors():
    """A 15 degree rotation and a 200 px dither between exposures do not
    add scatter: the anchor fit per exposure carries the positions to one
    tangent plane.
    """
    rng = np.random.default_rng(2)
    sky = _sky(rng)
    df = pd.concat(
        [
            _catalog(rng, sky, 1, 0.05, wcs=_wcs(0.0, 0.0)),
            _catalog(rng, sky, 2, 0.05, wcs=_wcs(200.0, -150.0, 15.0)),
            _catalog(rng, sky, 3, 0.05, wcs=_wcs(-120.0, 80.0, -7.0)),
        ],
        ignore_index=True,
    )
    q = rc.repeat(df)["quantities"]["source position"]
    assert q["n_groups"] == 150
    assert q["arms"]["full"]["chi2_per_dof"] == pytest.approx(1.0, rel=0.15)


def test_duplicates_within_an_exposure_are_dropped_and_ambiguity_rejected():
    rng = np.random.default_rng(3)
    sky = _sky(rng, n=40)
    a = _catalog(rng, sky, 1, 0.05, duplicates=[(0, 3.0), (5, 2.0)])
    b = _catalog(rng, sky, 2, 0.05)
    # source 7 gets a second, brighter detection 1.5 px away in exposure 2:
    # within link_px but beyond duplicate_px, so its group is ambiguous
    extra = b.iloc[[7]].copy()
    extra["positive_y"] += 1.5
    extra["center_y"] += 1.5
    extra["peak_abs_sigma"] = 99.0
    b = pd.concat([b, extra], ignore_index=True)
    q = rc.repeat(pd.concat([a, b], ignore_index=True))["quantities"][
        "source position"
    ]
    assert q["n_groups"] == 39 and q["n_detections"] == 78
    assert q["arms"]["full"]["chi2_per_dof"] == pytest.approx(1.0, rel=0.3)


def test_classes_select_the_quantity():
    from cuphoton.xfit import DipoleQualityFlag as F

    rng = np.random.default_rng(4)
    sky = _sky(rng, n=60)
    flags = np.zeros(60, np.int64)
    flags[:20] = int(F.LOBES_OVERLAP)
    flags[20:40] = int(F.NEGATIVE_LOBE_UNSUPPORTED)
    flags[40:45] = int(F.WIDTH_OUT_OF_RANGE)
    df = pd.concat(
        [_catalog(rng, sky, e, 0.05, flags=flags) for e in (1, 2, 3)],
        ignore_index=True,
    )
    df.loc[df.index[:3], "converged"] = False
    rep = rc.repeat(df)
    center = rep["quantities"]["dipole center"]
    source = rep["quantities"]["source position"]
    assert set(center["by_class"]) == {"overlapping", "resolved"}
    assert set(source["by_class"]) == {"single", "resolved"}
    # the three invalid rows are overlapping sources in exposure 1; their
    # groups keep two exposures
    assert center["class_counts"]["overlapping"] == 60 - 3
    assert center["n_groups"] == 35
    assert source["class_counts"]["single"] == 60
    assert source["class_counts"]["resolved"] == 45
    assert source["n_groups"] == 35


def test_arm_variances_by_hand():
    df = pd.DataFrame(
        {
            "positive_x_err_fit": [1.0],
            "positive_x_err_corr": [2.0],
            "positive_x_err_kfit": [3.0],
            "positive_x_err_kmatch": [4.0],
            "positive_x_err_mis": [5.0],
            "positive_x_err": [np.sqrt(55.0)],
        }
    )
    want = {
        "full": 55.0,
        "fit only": 1.0,
        "without corr": 51.0,
        "without kfit": 46.0,
        "without kmatch": 39.0,
        "without mis": 30.0,
    }
    for arm, v in want.items():
        assert rc.arm_variance(df, "positive_x", arm)[0] == pytest.approx(v)
    with pytest.raises(ValueError):
        rc.arm_variance(df, "positive_x", "without sky")


def test_repeat_chi2_by_hand():
    """Two exposures of one source, x = 0 and 1 with variances 1 and 3, y
    equal: the inverse-variance mean is 0.25 and chi2 = 0.25 on 2 degrees
    of freedom; singletons and non-positive variances are skipped.
    """
    from scipy.stats import chi2

    r = rc.repeat_chi2(["a", "a"], [0.0, 1.0], [5.0, 5.0], [1.0, 3.0], [2, 2])
    assert r["groups"] == 1 and r["dof"] == 2
    assert r["chi2_per_dof"] == pytest.approx(0.125)
    assert r["median_group_chi2_ratio"] == pytest.approx(
        0.25 / chi2.ppf(0.5, 2)
    )
    r = rc.repeat_chi2(
        ["a", "b", "b", "c", "c"],
        [0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
    )
    assert r["groups"] == 1 and r["chi2_per_dof"] == pytest.approx(0.25)


def test_load_reads_the_affine_beside_each_catalog(tmp_path):
    rng = np.random.default_rng(5)
    sky = _sky(rng, n=30)
    for e in (1, 2):
        d = tmp_path / f"item{e}"
        d.mkdir()
        df = _catalog(rng, sky, e, 0.05).drop(
            columns=["affine_A", "affine_b"]
        )
        df.to_parquet(d / "catalog.parquet", index=False)
        if e == 2:
            (d / "summary.json").write_text(
                json.dumps(
                    {
                        "diagnostics": {
                            "align": {
                                "refine": {
                                    "affine": {
                                        "A": [[1.0, 0.0], [0.0, 1.0]],
                                        "b": [0.3, -0.2],
                                    }
                                }
                            }
                        }
                    }
                )
            )
    df = rc.load(sorted(tmp_path.glob("*/catalog.parquet")))
    assert df.affine_b.iloc[0] == [0.0, 0.0]
    assert df[df.exposure_id == "2"].affine_b.iloc[0] == [0.3, -0.2]
    # the shift moves exposure 2 by (0.3, -0.2) px against exposure 1
    q = rc.repeat(df)["quantities"]["source position"]
    assert q["arms"]["full"]["chi2_per_dof"] > 5
    text = rc.text_table({**rc.repeat(df), "catalogs": []})
    assert "Not seen by this test" in text and "source position" in text
