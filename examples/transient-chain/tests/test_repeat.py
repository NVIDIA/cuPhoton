# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The repeat test on synthetic catalogs of repeated exposures: static
sources with known scatter, a shared reference term, unrecorded shifts,
distorted and rotated frames, duplicates, and the term arithmetic.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import repeat_catalogs as rc
from astropy.wcs import WCS, Sip
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
DEC0 = -10.0
SIGMA = (0.05, 0.08)  # scatter per axis in pixels, the anisotropic fixture


def _wcs(dx, dy, rot_deg=0.0, *, pointing=False, sip=False):
    """A TAN WCS dithered by ``(dx, dy)`` pixels and rotated: through the
    reference pixel by default, through the pointing (``crval``) with
    ``pointing``, which keeps a SIP distortion fixed on the detector.
    """
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [30.0, DEC0]
    w.wcs.crpix = [NX / 2, NY / 2]
    if pointing:
        w.wcs.crval = [
            30.0 + dx * PIX / np.cos(np.radians(DEC0)),
            DEC0 + dy * PIX,
        ]
    else:
        w.wcs.crpix = [NX / 2 + dx, NY / 2 + dy]
    a = np.radians(rot_deg)
    w.wcs.cd = PIX * np.array(
        [[-np.cos(a), np.sin(a)], [np.sin(a), np.cos(a)]]
    )
    if sip:  # cubic terms worth 2 px at the corners
        w.wcs.ctype = ["RA---TAN-SIP", "DEC--TAN-SIP"]
        ca, cb = np.zeros((4, 4)), np.zeros((4, 4))
        ca[3, 0], ca[0, 3], ca[2, 1] = 2e-9, -1e-9, 1.5e-9
        cb[0, 3], cb[3, 0], cb[1, 2] = 2e-9, 1e-9, -1.5e-9
        w.sip = Sip(ca, cb, None, None, w.wcs.crpix)
    return w


def _catalog(
    rng,
    sky,
    exposure,
    sigma=SIGMA,
    *,
    err_scale=1.0,
    terms=(0.2, 0.2, 0.0, 0.1),
    flags=None,
    wcs=None,
    duplicates=(),
    shared=None,
    shift=(0.0, 0.0),
    frame=None,
):
    """One exposure's catalog of static sources at ``sky`` (deg), detected
    at the nearest pixel with sub-pixel offsets scattered by ``sigma`` px
    per axis (one value or a pair); the quoted error bar is ``sigma *
    err_scale``, split into the fit term and the four others by ``terms``
    (shares of the variance). ``shared`` (n, 2) px is added to every
    exposure alike (a reference term), ``shift`` px is an unrecorded
    pointing shift, and ``frame`` an ``(A, b)`` the exposure's positions are
    taken through in reverse and recorded as its refine affine.
    """
    w = wcs or _wcs(*rng.uniform(-30, 30, 2))
    px = w.all_world2pix(sky, 0)
    n = len(px)
    sigma = np.broadcast_to(np.asarray(sigma, float), (2,))
    a, b = (np.eye(2), np.zeros(2)) if frame is None else frame
    true = px + np.asarray(shift, float)
    if shared is not None:
        true = true + shared
    true = (true - b) @ np.linalg.inv(a).T  # the target frame
    ix, iy = np.rint(true[:, 0]).astype(int), np.rint(true[:, 1]).astype(int)
    true = true - np.column_stack([ix, iy])
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
        off = true + rng.normal(0.0, 1.0, (n, 2)) * sigma
        for j, ax in enumerate("xy"):
            rows[f"{prefix}_{ax}"] = off[:, j]
            rows[f"{prefix}_{ax}_err_fit"] = np.full(n, fit[j])
            rows[f"{prefix}_{ax}_err"] = np.full(n, bar[j])
            for t, s in zip(rc.TERMS, share, strict=True):
                rows[f"{prefix}_{ax}_err_{t}"] = np.full(
                    n, bar[j] * np.sqrt(s)
                )
        rows[f"{prefix}_xy_cov"] = np.zeros(n)
    rows["uq_valid"] = np.ones(n, bool)
    rows["uq_window_masked"] = np.zeros(n, bool)
    rows["uq_corr_measured"] = np.zeros(n, bool)
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
    df["affine_A"] = [a.tolist()] * len(df)
    df["affine_b"] = [b.tolist()] * len(df)
    return df


def _sky(rng, n=150):
    w = _wcs(0.0, 0.0)
    px = np.column_stack(
        [rng.uniform(100, NX - 100, n), rng.uniform(100, NY - 100, n)]
    )
    return w.all_pix2world(px, 0)


def _exposures(rng, sky, n_exposures=4, **kw):
    return pd.concat(
        [_catalog(rng, sky, e, **kw) for e in range(101, 101 + n_exposures)],
        ignore_index=True,
    )


@pytest.mark.parametrize("err_scale, want", [(1.0, 1.0), (0.5, 4.0)])
def test_honest_error_bars_give_unit_chi2_and_halved_ones_four(
    err_scale, want
):
    """Four exposures of 150 sources, 600 detections and 900 degrees of
    freedom per quantity, scatter 0.05 px in x and 0.08 px in y."""
    rng = np.random.default_rng(0)
    df = _exposures(rng, _sky(rng), err_scale=err_scale)
    rep = rc.repeat(df)
    for name in ("dipole center", "source position"):
        q = rep["quantities"][name]
        assert q["n_groups"] == 150 and q["n_detections"] == 600
        assert q["exposures_per_group"] == {"4": 150}
        full = q["arms"]["full"]
        assert full["dof"] == 900
        assert full["chi2_per_dof"] == pytest.approx(want, rel=0.05)
        assert full["median_group_chi2_ratio"] == pytest.approx(want, rel=0.1)
        # the arms shrink the variance by the term's share: 50% fit alone
        assert q["arms"]["fit only"]["chi2_per_dof"] == pytest.approx(
            full["chi2_per_dof"] / 0.5, rel=1e-9
        )
        assert q["arms"]["without kmatch"]["chi2_per_dof"] == pytest.approx(
            full["chi2_per_dof"], rel=1e-9
        )
        assert q["by_class"]["resolved"]["full"]["groups"] == 150
    assert rep["blind_spots"]
    d1 = rep["detectors"]["D1"]
    assert d1["exposures"] == ["101", "102", "103", "104"]
    assert d1["exposures_without_anchor_fit"] == []
    for exp in d1["exposures"]:
        rec = d1["anchors"][exp]
        assert rec["detections"] == 150 and rec["transform"] == "identity"
        assert rec["fit"]["anchors"] == 150
        # the gnomonic projection is not affine over 2000 px: 0.05 px rms
        assert rec["fit"]["residual_arcsec"] < 0.02
        np.testing.assert_allclose(rec["fit"]["pixel_arcsec"], 0.2, rtol=1e-4)


def test_a_shared_reference_term_lowers_the_chi2_by_its_share():
    """A reference position error drawn once per source moves every
    exposure alike; the bars carry it as the corr term with half the
    variance, so the chi-square is 1/2 and the "without corr" arm is 1."""
    rng = np.random.default_rng(1)
    sky = _sky(rng)
    shared = rng.normal(0.0, 1.0, (150, 2)) * SIGMA
    df = _exposures(
        rng,
        sky,
        shared=shared,
        err_scale=np.sqrt(2.0),
        terms=(0.5, 0.0, 0.0, 0.0),
    )
    q = rc.repeat(df)["quantities"]["source position"]
    assert q["arms"]["full"]["chi2_per_dof"] == pytest.approx(0.5, rel=0.1)
    assert q["arms"]["without corr"]["chi2_per_dof"] == pytest.approx(
        1.0, rel=0.1
    )


def test_an_unrecorded_pointing_shift_raises_the_chi2():
    """Two exposures, the second shifted by 0.1 px in x without a recorded
    affine: the x residuals gain shift^2 / 2 per pair, so the pooled
    chi-square per degree of freedom over both axes is 1 + shift^2 / (4
    sigma_x^2) = 2."""
    rng = np.random.default_rng(2)
    sky = _sky(rng)
    df = pd.concat(
        [
            _catalog(rng, sky, 1, 0.05),
            _catalog(rng, sky, 2, 0.05, shift=(0.1, 0.0)),
        ],
        ignore_index=True,
    )
    q = rc.repeat(df)["quantities"]["source position"]
    assert q["arms"]["full"]["dof"] == 300
    assert q["arms"]["full"]["chi2_per_dof"] == pytest.approx(2.0, rel=0.1)


def test_ra_zero_crossing_preserves_the_anchor_scale():
    rng = np.random.default_rng(23)
    frame = _exposures(rng, _sky(rng))
    frame["ra"] = (frame["ra"] - frame["ra"].median()) % 360.0
    report = rc.repeat(frame)
    for exposure in report["detectors"]["D1"]["anchors"].values():
        np.testing.assert_allclose(
            exposure["fit"]["pixel_arcsec"], 0.2, rtol=1e-4
        )
    for quantity in report["quantities"].values():
        assert quantity["arms"]["full"]["chi2_per_dof"] == pytest.approx(
            1.0, rel=0.1
        )


def test_unsupported_and_outside_lobes_are_not_scored():
    from cuphoton.xfit import DipoleQualityFlag

    flags = [
        DipoleQualityFlag.NEGATIVE_LOBE_UNSUPPORTED
        | DipoleQualityFlag.POSITIVE_LOBE_UNSUPPORTED,
        DipoleQualityFlag.LOBE_OUTSIDE_STAMP,
        DipoleQualityFlag.LOBE_OUTSIDE_STAMP
        | DipoleQualityFlag.LOBES_OVERLAP,
    ]
    frame = pd.DataFrame(
        {"quality_flags": flags, "converged": True, "uncertainty_valid": True}
    )
    assert rc.classify(frame).tolist() == ["failed"] * len(flags)


def test_rotated_and_dithered_exposures_link_through_the_anchors():
    """A 15 degree rotation and a 200 px dither between exposures do not
    add scatter: the anchor fit per exposure carries the positions to one
    tangent plane.
    """
    rng = np.random.default_rng(2)
    sky = _sky(rng)
    df = pd.concat(
        [
            _catalog(rng, sky, 1, wcs=_wcs(0.0, 0.0)),
            _catalog(rng, sky, 2, wcs=_wcs(200.0, -150.0, 15.0)),
            _catalog(rng, sky, 3, wcs=_wcs(-120.0, 80.0, -7.0)),
        ],
        ignore_index=True,
    )
    q = rc.repeat(df)["quantities"]["source position"]
    assert q["n_groups"] == 150
    assert q["arms"]["full"]["chi2_per_dof"] == pytest.approx(1.0, rel=0.05)


@pytest.mark.parametrize("angles", [(90, 90, 90, 90), (0, 45, 90, -45)])
def test_anisotropic_errors_follow_the_exposure_rotation(angles):
    rng = np.random.default_rng(72)
    sky = _sky(rng, n=500)
    frames = [
        _catalog(
            rng,
            sky,
            exposure,
            sigma=(0.01, 0.1),
            wcs=_wcs(0, 0, angle),
        )
        for exposure, angle in enumerate(angles)
    ]
    report = rc.repeat(pd.concat(frames, ignore_index=True))
    for quantity in report["quantities"].values():
        full = quantity["arms"]["full"]
        assert full["groups"] >= 490
        assert full["chi2_per_dof"] == pytest.approx(1.0, rel=0.08)
        assert full["median_group_chi2_ratio"] == pytest.approx(1.0, rel=0.1)


def test_repeat_chi2_uses_cross_axis_covariance():
    report = rc.repeat_chi2(
        [0, 0],
        [0.0, 1.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [1.0, 1.0],
        [0.5, 0.5],
    )
    assert report["chi2_per_dof"] == pytest.approx(1.0 / 3.0)


def test_distorted_frames_keep_their_distortion_in_the_anchors():
    """Cubic SIP terms worth 2 px at the corners, fixed on the detector,
    with the pointing dithered by up to 150 px and rotated between
    exposures: the sky position is the anchor's plus the linear map of the
    sub-pixel offset, so the distortion does not become scatter.
    """
    rng = np.random.default_rng(6)
    sky = _sky(rng)
    df = pd.concat(
        [
            _catalog(
                rng,
                sky,
                e,
                wcs=_wcs(
                    *rng.uniform(-150, 150, 2), r, pointing=True, sip=True
                ),
            )
            for e, r in ((1, 0.0), (2, 12.0), (3, -5.0), (4, 3.0))
        ],
        ignore_index=True,
    )
    q = rc.repeat(df)["quantities"]["source position"]
    assert q["n_groups"] == 150
    assert q["arms"]["full"]["chi2_per_dof"] == pytest.approx(1.0, rel=0.05)


def test_the_refine_affine_is_applied_in_the_recorded_direction():
    """Exposure 2's positions are taken through the inverse of a recorded
    ``(A, b)`` with a rotation, a scale and a shift: applying the affine as
    recorded restores them, the identity or the inverse does not.
    """
    rng = np.random.default_rng(7)
    sky = _sky(rng)
    th = np.radians(0.3)
    a = 1.002 * np.array(
        [[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]
    )
    b = np.array([0.4, -0.3])
    first = _catalog(rng, sky, 1)
    second = _catalog(rng, sky, 2, frame=(a, b))
    q = rc.repeat(pd.concat([first, second], ignore_index=True))["quantities"]
    assert q["source position"]["arms"]["full"][
        "chi2_per_dof"
    ] == pytest.approx(1.0, rel=0.1)
    rep = rc.repeat(pd.concat([first, second], ignore_index=True))
    assert rep["detectors"]["D1"]["anchors"]["2"]["transform"] == "affine"
    for wrong_a, wrong_b in (
        (np.eye(2), np.zeros(2)),
        (np.linalg.inv(a), -b),
    ):
        other = second.copy()
        other["affine_A"] = [wrong_a.tolist()] * len(other)
        other["affine_b"] = [wrong_b.tolist()] * len(other)
        q = rc.repeat(pd.concat([first, other], ignore_index=True))[
            "quantities"
        ]
        assert q["source position"]["arms"]["full"]["chi2_per_dof"] > 5


def test_duplicates_within_an_exposure_are_dropped_and_ambiguity_rejected():
    rng = np.random.default_rng(3)
    sky = _sky(rng, n=40)
    a = _catalog(rng, sky, 1, duplicates=[(0, 3.0), (5, 2.0)])
    b = _catalog(rng, sky, 2)
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
    sky = _sky(rng, n=80)
    flags = np.zeros(80, np.int64)
    flags[:20] = int(F.LOBES_OVERLAP)
    flags[20:40] = int(F.NEGATIVE_LOBE_UNSUPPORTED)  # brightening singles
    flags[40:45] = int(F.WIDTH_OUT_OF_RANGE)
    flags[60:80] = int(F.POSITIVE_LOBE_UNSUPPORTED)  # fading singles
    df = _exposures(rng, sky, 3, flags=flags)
    df.loc[df.index[:3], "converged"] = False
    kinds = rc.classify(df)
    assert set(kinds) == {
        "invalid",
        "overlapping",
        "single",
        "failed",
        "resolved",
        "fading",
    }
    assert (kinds[60:80] == "fading").all() and (
        kinds[20:40] == "single"
    ).all()
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
    # fading singles are in neither quantity
    assert sum(center["class_counts"].values()) == 102
    assert sum(source["class_counts"].values()) == 105


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
    assert list(rc.ARMS) == list(want)
    for arm, v in want.items():
        assert rc.arm_variance(df, "positive_x", arm)[0] == pytest.approx(v)
    with pytest.raises(ValueError, match="unknown arm"):
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


def test_tangent_plane_wraps_the_right_ascension():
    ex, ey = rc.tangent_plane([359.9, 0.3], [0.0, 0.0], 0.1, 0.0)
    np.testing.assert_allclose(ex, [-720.0, 720.0])
    np.testing.assert_allclose(ey, 0.0)


def test_anchor_fit_needs_enough_well_spread_anchors():
    rng = np.random.default_rng(8)
    x, y = rng.uniform(0, 1000, 12), rng.uniform(0, 1000, 12)
    fit = rc.fit_anchor_affine(x, y, 0.2 * x, -0.2 * y)
    np.testing.assert_allclose(
        fit.matrix, [[0.2, 0.0], [0.0, -0.2]], atol=1e-9
    )
    assert fit.anchors == 12 and fit.residual_arcsec < 1e-9
    np.testing.assert_allclose(fit.pixel_arcsec, 0.2)
    assert rc.fit_anchor_affine(x[:9], y[:9], x[:9], y[:9]) is None
    line = np.linspace(0, 1000, 12)
    assert rc.fit_anchor_affine(line, 0.5 * line + 1e-6, line, line) is None
    # an exposure whose anchors lie on a line is reported and skipped
    sky = _sky(rng, n=12)
    w = _wcs(0.0, 0.0)
    px = np.column_stack([line, 1000 + 0.5 * line])
    df = pd.concat(
        [
            _catalog(rng, sky, 1),
            _catalog(rng, w.all_pix2world(px, 0), 2, wcs=w),
        ],
        ignore_index=True,
    )
    rep = rc.repeat(df)
    assert rep["detectors"]["D1"]["exposures_without_anchor_fit"] == ["2"]
    assert rep["detectors"]["D1"]["anchors"]["2"]["fit"] is None
    assert rep["quantities"]["source position"] is None


@pytest.mark.parametrize(
    "column,value", [("ra", np.nan), ("dec", np.inf), ("x", np.nan)]
)
def test_repeat_uses_finite_anchors_without_losing_valid_sources(
    column, value
):
    rng = np.random.default_rng(112)
    sky = _sky(rng, n=30)
    df = pd.concat([_catalog(rng, sky, e) for e in (1, 2)], ignore_index=True)
    df.loc[0, column] = value
    rep = rc.repeat(df)
    detector = rep["detectors"]["D1"]
    assert detector["exposures_without_anchor_fit"] == []
    assert detector["anchors"]["1"]["fit"]["anchors"] == 29
    assert detector["anchors"]["2"]["fit"]["anchors"] == 30
    for quantity in rep["quantities"].values():
        assert quantity["n_groups"] == 29
        assert np.isfinite(quantity["arms"]["full"]["chi2_per_dof"])


@pytest.mark.parametrize("bad_count,skipped", [(25, ["1"]), (60, ["1", "2"])])
def test_repeat_reports_exposures_without_enough_finite_anchors(
    bad_count, skipped
):
    rng = np.random.default_rng(112)
    sky = _sky(rng, n=30)
    df = pd.concat([_catalog(rng, sky, e) for e in (1, 2)], ignore_index=True)
    df.loc[: bad_count - 1, "ra"] = np.nan
    rep = rc.repeat(df)
    detector = rep["detectors"]["D1"]
    assert detector["exposures_without_anchor_fit"] == skipped
    assert all(detector["anchors"][e]["fit"] is None for e in skipped)
    assert all(quantity is None for quantity in rep["quantities"].values())


def test_repeat_needs_the_rows_load_returns():
    rng = np.random.default_rng(9)
    df = _catalog(rng, _sky(rng, n=12), 1).drop(columns=["affine_A"])
    with pytest.raises(ValueError, match="load"):
        rc.repeat(df)


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
    rep = rc.repeat(df)
    q = rep["quantities"]["source position"]
    assert q["arms"]["full"]["chi2_per_dof"] > 5
    anchors = rep["detectors"]["D1"]["anchors"]
    assert anchors["1"]["transform"] == "identity"
    assert anchors["2"]["transform"] == "shift"
    text = rc.text_table({**rep, "catalogs": []})
    assert "Not seen by this test" in text and "source position" in text
