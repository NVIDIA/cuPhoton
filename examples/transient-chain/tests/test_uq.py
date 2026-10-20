# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The error budget on a NumPy stand-in for one item: a reference with
stars, a Gaussian kernel, a target with transients and white noise.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.signal import fftconvolve
from transient_chain import config, ops, uq

SIZE = 200
PSF = 1.8
NOISE = 3.0
FIELD = np.array([0.1, -0.05])  # constant reference-minus-target offset
TAU2 = np.array([0.01, 0.0225])  # misalignment variance per axis
MIS_VARIANCE = TAU2 + FIELD**2
QUANTITIES = uq.QUANTITIES
MAX_REACH = config.SCHEMA["uq"]["max_lobe_reach_px"][0]


def _gaussians(xs, ys, fluxes, sigma=PSF):
    yy, xx = np.mgrid[:SIZE, :SIZE]
    img = np.zeros((SIZE, SIZE))
    for x, y, f in zip(xs, ys, fluxes, strict=True):
        img += (
            f
            * np.exp(-0.5 * ((xx - x) ** 2 + (yy - y) ** 2) / sigma**2)
            / (2 * np.pi * sigma**2)
        )
    return img


def _kernel(sigma=1.0, size=7):
    u = np.arange(size) - size // 2
    g = np.exp(-0.5 * (u / sigma) ** 2)
    k = np.outer(g, g)
    return k / k.sum()


class Scene:
    """Reference stars, a kernel, transients and the arrays an item holds."""

    def __init__(self, seed=0, mismatch=0.0, n_stars=24, cand=None):
        rng = np.random.default_rng(seed)
        self.star_xy = np.column_stack(
            [
                rng.uniform(25, SIZE - 25, n_stars),
                rng.uniform(25, SIZE - 25, n_stars),
            ]
        )
        flux = rng.uniform(2.0e4, 6.0e4, n_stars)
        self.reference = 50.0 + _gaussians(
            self.star_xy[:, 0], self.star_xy[:, 1], flux
        )
        self.kernel = _kernel()
        self.matched = fftconvolve(self.reference, self.kernel, mode="same")
        self.cand = np.array(
            [[60, 70], [140, 50], [100, 150], [40, 160]]
            if cand is None
            else cand
        )
        self.flux = np.array([3000.0, 5000.0, 2500.0, 4000.0])
        target = self.matched + _gaussians(
            self.cand[:, 0] + 0.3, self.cand[:, 1] - 0.2, self.flux
        )
        # a mismatched kernel leaves a fraction of every star in the
        # residual, here a different fraction per star with scatter
        # ``mismatch`` (the budget's eta is that scatter)
        fraction = mismatch * rng.normal(size=n_stars)
        target += fftconvolve(
            _gaussians(
                self.star_xy[:, 0], self.star_xy[:, 1], flux * fraction
            ),
            self.kernel,
            mode="same",
        )
        self.target = target + rng.normal(0.0, NOISE, target.shape)
        self.residual = self.target - self.matched
        self.variance = np.full((SIZE, SIZE), NOISE**2)
        # a masked block over most of the last candidate's windows (more
        # than half of their pixels)
        self.variance[150:170, 30:46] = np.nan
        self.residual[150:170, 30:46] = np.nan
        mask = np.ones((SIZE, SIZE), bool)
        mask[:4] = mask[-4:] = mask[:, :4] = mask[:, -4:] = False
        mask &= np.isfinite(self.variance)
        self.fit = SimpleNamespace(
            basis_kernels=self.kernel[None],
            fit_mask=mask,
            background_coefficients=np.zeros(1),
            kernel=self.kernel,
        )
        self.misalignment = ops.Misalignment(
            np.eye(2),
            FIELD,
            np.zeros((2, 3, 3)),
            "rms",
            TAU2,
            np.ones(5, bool),
            np.zeros((5, 2)),
            (SIZE, SIZE),
        )

    def context(self, **kw):
        args = dict(
            kernel=self.kernel,
            reference=self.reference,
            matched=self.matched,
            residual=self.residual,
            variance=self.variance,
            psf_sigma=PSF,
            misalignment=self.misalignment,
            star_xy=self.star_xy,
            fit=self.fit,
        )
        args.update(kw)
        return uq.build_context(**args)

    def inputs(self, single=None, err=0.05):
        n = len(self.cand)
        err_fit = {q: np.full(n, err) for q in QUANTITIES}
        err_fit["psf_flux"] = np.full(n, 40.0)
        return dict(
            max_lobe_reach_px=MAX_REACH,
            cand_x=self.cand[:, 0],
            cand_y=self.cand[:, 1],
            # a resolved dipole: lobes 2.5 px either side of the center
            offsets={
                "center": (np.full(n, 0.3), np.full(n, -0.2)),
                "positive": (np.full(n, 2.8), np.full(n, -0.2)),
            },
            err_fit=err_fit,
            lobe_flux=self.flux,
            single=np.zeros(n, bool) if single is None else single,
            photometry_xy=np.column_stack(
                [np.full(n, 0.3), np.full(n, -0.2)]
            ),
        )


@pytest.fixture(scope="module")
def scene():
    return Scene()


@pytest.mark.parametrize("bad_offset", [np.nan, -1000.0])
@pytest.mark.parametrize("photometry", [True, False])
@pytest.mark.parametrize("psf_sigma", [PSF, None])
def test_invalid_lobe_position_does_not_validate_its_fallback(
    scene, bad_offset, photometry, psf_sigma
):
    context = scene.context(
        fit=None, star_xy=None, misalignment=None, psf_sigma=psf_sigma
    )
    inputs = scene.inputs()
    if not photometry:
        inputs["photometry_xy"] = None
    baseline = uq.budget(context, **inputs).columns
    inputs["offsets"]["positive"][0][0] = bad_offset

    columns = uq.budget(context, **inputs).columns

    assert not columns["uq_valid"][0]
    assert columns["uq_valid"][1:3].all()
    for quantity in QUANTITIES:
        if quantity == "psf_flux" and photometry:
            assert np.isfinite(columns[f"{quantity}_err"][0])
        else:
            assert np.isnan(columns[f"{quantity}_err"][0])
        np.testing.assert_allclose(
            columns[f"{quantity}_err"][1:],
            baseline[f"{quantity}_err"][1:],
            equal_nan=True,
        )


@pytest.mark.parametrize("bad_offset", [np.nan, -1000.0])
@pytest.mark.parametrize("psf_sigma", [PSF, None])
def test_invalid_photometry_position_does_not_invalidate_lobes(
    scene, bad_offset, psf_sigma
):
    context = scene.context(
        fit=None, star_xy=None, misalignment=None, psf_sigma=psf_sigma
    )
    inputs = scene.inputs()
    inputs["photometry_xy"][0, 0] = bad_offset

    columns = uq.budget(context, **inputs).columns

    assert not columns["uq_valid"][0]
    assert columns["uq_valid"][1:3].all()
    assert np.isnan(columns["psf_flux_err"][0])
    for quantity in QUANTITIES:
        if quantity != "psf_flux":
            assert np.isfinite(columns[f"{quantity}_err"][0])


def test_terms_and_fallbacks_are_recorded(scene):
    ctx = scene.context()
    assert ctx.terms == ("kfit", "kmatch", "mis")
    assert ctx.excluded == {"corr": "no reference variance plane"}
    assert ctx.kernel_covariance.form == "plain"
    assert ctx.window == 2 * 9 + 1
    bare = scene.context(fit=None, misalignment=None, star_xy=None)
    assert bare.terms == ()
    assert set(bare.excluded) == {"corr", "kfit", "kmatch", "mis"}
    blind = scene.context(psf_sigma=None)
    assert blind.terms == ("mis",) and blind.window is None


def test_total_is_the_quadrature_and_never_below_the_fit_term(scene):
    ctx = scene.context()
    res = uq.budget(ctx, **scene.inputs())
    c = res.columns
    valid = c["uq_valid"]
    assert valid.tolist() == [True, True, True, False]
    for q in QUANTITIES:
        fit = scene.inputs()["err_fit"][q]
        total = c[f"{q}_err"]
        assert np.all(total[valid] >= fit[valid])
        assert np.isnan(total[~valid]).all()
        squares = sum(c[f"{q}_err_{t}"][valid] ** 2 for t in uq.TERMS)
        np.testing.assert_allclose(
            total[valid] ** 2, fit[valid] ** 2 + squares, rtol=1e-12
        )
        np.testing.assert_array_equal(c[f"{q}_err_corr"][valid], 0.0)
    # misalignment: 1/4 of the model for a center, the model for a lobe
    for a, t2 in zip("xy", MIS_VARIANCE, strict=True):
        np.testing.assert_allclose(c[f"center_{a}_err_mis"] ** 2, 0.25 * t2)
        np.testing.assert_allclose(c[f"positive_{a}_err_mis"] ** 2, t2)
        np.testing.assert_array_equal(c[f"center_{a}_err_kmatch"], 0.0)
        np.testing.assert_array_equal(c[f"positive_{a}_err_kmatch"], 0.0)
    assert np.all(c["psf_flux_err_mis"] == 0.0)
    assert np.all(c["psf_flux_err_kfit"][valid] > 0)
    assert np.all(c["center_x_err_kfit"][valid] > 0)
    assert res.diagnostics["invalid_window"] == 1
    assert res.diagnostics["rotated_grid"] == 0


def test_kernel_fit_term_is_excluded_without_target_noise_variance(scene):
    context = scene.context(variance=None)
    assert context.excluded["kfit"] == "no target variance plane"
    assert context.kernel_covariance is None
    columns = uq.budget(context, **scene.inputs()).columns
    for quantity in QUANTITIES:
        np.testing.assert_array_equal(columns[f"{quantity}_err_kfit"], 0.0)


def test_flux_kernel_term_uses_the_published_photometry_position():
    import dataclasses

    from cuphoton.xpois import psf_weights

    scene = Scene(cand=[[80, 80], [140, 50], [100, 150], [40, 160]])
    star = _gaussians([80.3], [79.8], [1e6])
    scene.reference += star
    scene.matched = fftconvolve(scene.reference, scene.kernel, mode="same")
    covariance = 1e-6
    context = dataclasses.replace(
        scene.context(star_xy=None),
        kernel_covariance=_known_covariance(scene, covariance),
    )
    inputs = scene.inputs(single=np.array([True, False, False, False]))
    result = uq.budget(context, **inputs)
    assert result.diagnostics["sensitivity_capped"] == 1
    position_x, position_y = (scene.cand + inputs["photometry_xy"])[0]
    offset = np.array(
        [[position_x - np.rint(position_x), position_y - np.rint(position_y)]]
    )
    weights = psf_weights(PSF, context.window, offset=offset)[0]
    expected = (
        covariance * _hand_sum(scene, weights, position_x, position_y) ** 2
    )
    assert result.columns["psf_flux_err_kfit"][0] ** 2 == pytest.approx(
        expected, rel=1e-9
    )


def test_worked_quadrature_by_hand(scene):
    """Kernel fit off, no stars, no reference variance: only misalignment
    adds to the fit term, so the numbers follow by hand.
    """
    ctx = scene.context(fit=None, star_xy=None)
    assert ctx.terms == ("mis",)
    c = uq.budget(ctx, **scene.inputs(err=0.05)).columns
    assert c["center_x_err"][0] == pytest.approx(
        np.sqrt(0.05**2 + 0.25 * MIS_VARIANCE[0]), rel=1e-12
    )
    assert c["positive_y_err"][1] == pytest.approx(
        np.sqrt(0.05**2 + MIS_VARIANCE[1]), rel=1e-12
    )
    assert c["psf_flux_err"][0] == pytest.approx(40.0, rel=1e-12)
    assert c["center_x_err_mis"][0] == pytest.approx(
        0.5 * np.sqrt(MIS_VARIANCE[0])
    )
    assert c["positive_x_err_mis"][0] == pytest.approx(
        np.sqrt(MIS_VARIANCE[0])
    )


def test_nonfinite_fit_error_invalidates_only_its_quantity(scene):
    inputs = scene.inputs()
    inputs["err_fit"]["positive_x"][0] = np.nan
    result = uq.budget(scene.context(fit=None, star_xy=None), **inputs)
    columns = result.columns
    assert columns["uq_valid"].tolist() == [False, True, True, False]
    assert np.isnan(columns["positive_x_err"][0])
    for quantity in QUANTITIES:
        if quantity != "positive_x":
            assert np.isfinite(columns[f"{quantity}_err"][0])
    assert result.diagnostics["invalid_window"] == 1
    assert result.diagnostics["rotated_grid"] == 0


def test_kernel_mismatch_is_recovered_from_the_stars():
    clean = Scene(seed=3)
    assert clean.context().kernel_mismatch["eta"] < 0.01
    off = Scene(seed=3, mismatch=0.03)
    km = off.context().kernel_mismatch
    assert km["n"] >= 20
    assert km["eta"] == pytest.approx(0.03, rel=0.3)
    # the flux term carries eta times the reference flux at the candidate
    c = uq.budget(off.context(), **off.inputs()).columns
    valid = c["uq_valid"]
    assert np.all(c["psf_flux_err_kmatch"][valid] > 0)


@pytest.mark.parametrize("gain, capped", [(0.5, False), (200.0, True)])
def test_single_source_sensitivity_scales_the_lobe_misalignment(
    scene, gain, capped
):
    """A single source's lobe moves by lambda = (D + T) / D times the
    misalignment; |lambda| is capped at SENSITIVITY_CAP.
    """
    # the first candidate sits where the scene has no star of its own
    s = Scene(seed=1, cand=[[15, 100], [140, 50], [100, 150], [40, 160]])
    # a static star under the first candidate, in both images: T = gain * D
    star = _gaussians(
        [s.cand[0, 0] + 0.3], [s.cand[0, 1] - 0.2], [gain * s.flux[0]]
    )
    s.reference = s.reference + star
    s.matched = fftconvolve(s.reference, s.kernel, mode="same")
    s.target = s.target + fftconvolve(star, s.kernel, mode="same")
    s.residual = s.target - s.matched
    single = np.zeros(len(s.cand), bool)
    single[0] = True
    res = uq.budget(s.context(), **s.inputs(single=single))
    c = res.columns
    lam2 = c["positive_x_err_mis"][0] ** 2 / MIS_VARIANCE[0]
    if capped:
        assert lam2 == pytest.approx(uq.SENSITIVITY_CAP**2, rel=1e-9)
        assert res.diagnostics["sensitivity_capped"] == 1
    else:
        # the matched star is the PSF broadened by the kernel, so its forced
        # flux through the PSF is gain * flux * 2 p^2 / (2 p^2 + k^2)
        seen = gain * 2 * PSF**2 / (2 * PSF**2 + 1.0)
        assert lam2 == pytest.approx((1 + seen) ** 2, rel=0.05)
        assert res.diagnostics["sensitivity_capped"] == 0
    # resolved dipoles keep lambda = 1
    np.testing.assert_allclose(
        c["positive_x_err_mis"][1:] ** 2, MIS_VARIANCE[0]
    )
    np.testing.assert_allclose(
        c["center_x_err_mis"][0] ** 2, 0.25 * MIS_VARIANCE[0]
    )


def test_reference_variance_adds_the_correlation_term(scene):
    from cuphoton.xrep import resampling_weights_at

    def mapping(c):  # a half-pixel translation of the reference grid
        return np.asarray(c, dtype=float) + 0.5

    def resampling(x, y):
        return resampling_weights_at(mapping, x, y, interpolation="lanczos3")

    plane = np.full((SIZE, SIZE), 4.0)
    ctx = scene.context(reference_variance=plane, resampling=resampling)
    assert ctx.terms == ("corr", "kfit", "kmatch", "mis")
    assert ctx.kernel_covariance.form == "sandwich"
    res = uq.budget(ctx, **scene.inputs())
    c = res.columns
    valid = c["uq_valid"]
    assert valid.tolist() == [True, True, True, False]
    for q in QUANTITIES:
        assert np.all(c[f"{q}_err_corr"][valid] > 0)
        assert np.all(c[f"{q}_err"][valid] > c[f"{q}_err_corr"][valid])

    def rotated(c):
        a = np.radians(10.0)
        r = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        return np.asarray(c, dtype=float) @ r.T

    ctx = scene.context(
        reference_variance=plane,
        resampling=lambda x, y: resampling_weights_at(
            rotated, x, y, interpolation="lanczos3"
        ),
    )
    res = uq.budget(ctx, **scene.inputs())
    assert not res.columns["uq_valid"].any()
    assert res.diagnostics["rotated_grid"] == len(scene.cand)
    with pytest.raises(ValueError, match="given together"):
        scene.context(reference_variance=plane)


def test_corr_term_follows_the_template_provider_variance(scene):
    """The chain hands the provider's variance and resampling weights to the
    budget: the corr column is non-zero only when the provider has a plane.
    """
    from transient_chain.surveys.base import GridTemplate

    image = scene.reference.astype(np.float32)
    without = GridTemplate(image, None, np)
    without.build()
    assert without.variance() is None
    ctx = scene.context(reference_variance=None, resampling=None)
    c = uq.budget(ctx, **scene.inputs()).columns
    assert "corr" in ctx.excluded
    for q in QUANTITIES:
        np.testing.assert_array_equal(c[f"{q}_err_corr"][c["uq_valid"]], 0.0)

    plane = np.full(scene.reference.shape, 4.0)
    with_plane = GridTemplate(image, None, np, variance=plane)
    with_plane.build()
    ctx = scene.context(
        reference_variance=with_plane.variance(),
        resampling=with_plane.resampling,
    )
    assert "corr" in ctx.terms
    c = uq.budget(ctx, **scene.inputs()).columns
    valid = c["uq_valid"]
    assert valid.tolist() == [True, True, True, False]
    for q in QUANTITIES:
        assert np.all(c[f"{q}_err_corr"][valid] > 0)


def test_fit_only_writes_zero_terms(scene):
    err_fit = scene.inputs()["err_fit"]
    c = uq.fit_only(err_fit).columns
    for q in QUANTITIES:
        np.testing.assert_array_equal(c[f"{q}_err"], err_fit[q])
        for t in uq.TERMS:
            np.testing.assert_array_equal(c[f"{q}_err_{t}"], 0.0)
    assert c["uq_valid"].all() and c["uq_valid"].dtype == bool


def test_reference_noise_is_relative_to_the_variance_used_by_xfit(scene):
    def resampling(position_x, position_y):
        return SimpleNamespace(
            weights=np.ones((len(position_x), 1, 1)),
            is_translation=np.ones(len(position_x), dtype=bool),
        )

    context = scene.context(
        kernel=np.ones((1, 1)),
        fit=None,
        misalignment=None,
        star_xy=None,
        reference_variance=np.full_like(scene.variance, NOISE**2),
        resampling=resampling,
    )
    inputs = scene.inputs()
    columns = uq.budget(context, **inputs).columns
    valid = columns["uq_valid"]
    assert valid.tolist() == [True, True, True, False]
    for quantity in QUANTITIES:
        np.testing.assert_allclose(
            columns[f"{quantity}_err"][valid],
            np.sqrt(2.0) * inputs["err_fit"][quantity][valid],
        )


def test_forced_flux_matches_the_planted_flux(scene):
    half = scene.context().window // 2
    flux, err = uq.forced_flux(
        scene.residual,
        scene.variance,
        scene.cand[:3, 0] + 0.3,
        scene.cand[:3, 1] - 0.2,
        PSF,
        half,
    )
    np.testing.assert_allclose(flux, scene.flux[:3], rtol=0.05)
    assert np.all(err > 0) and np.all(np.abs(flux - scene.flux[:3]) < 4 * err)


def _known_covariance(scene, c):
    """A kernel-fit covariance with variance ``c`` on the one kernel
    coefficient and none on the background, so the kernel-fit term of a
    weighted sum is ``c`` times the squared sum of the weights over the
    kernel-matched reference.
    """
    from cuphoton.xpois import KernelFitCovariance

    return KernelFitCovariance(
        covariance=np.diag([c, 0.0]),
        basis_kernels=scene.kernel[None],
        basis_count=1,
        background_degree=0,
        form="plain",
    )


def _hand_sum(scene, stamp, x, y):
    """``sum(w (R * K))`` over the window at ``rint(x, y)`` with the
    normalized weights ``w = stamp / (V sum(stamp^2 / V))``, by hand.
    """
    h = stamp.shape[-1] // 2
    cx, cy = int(np.rint(x)), int(np.rint(y))
    conv = scene.matched[cy - h : cy + h + 1, cx - h : cx + h + 1]
    var = scene.variance[cy - h : cy + h + 1, cx - h : cx + h + 1]
    w = stamp / var
    w /= (w * stamp).sum()
    return (w * conv).sum()


def test_kernel_fit_terms_match_a_known_covariance_by_hand(scene):
    import dataclasses

    from cuphoton.xpois import psf_weights

    c = 1e-6
    ctx = dataclasses.replace(
        scene.context(star_xy=None),
        kernel_covariance=_known_covariance(scene, c),
    )
    inp = scene.inputs()
    res = uq.budget(ctx, **inp)
    cols = res.columns
    size = ctx.window
    for i in range(3):
        cx, cy = scene.cand[i]
        # the flux: the PSF matched filter at the photometry position
        fx, fy = cx + 0.3, cy - 0.2
        off = np.array([[fx - np.rint(fx), fy - np.rint(fy)]])
        psf = psf_weights(PSF, size, offset=off)[0]
        expect = c * _hand_sum(scene, psf, fx, fy) ** 2
        assert cols["psf_flux_err_kfit"][i] ** 2 == pytest.approx(
            expect, rel=1e-9
        )
        # a lobe: the derivative matched filter, divided by the lobe flux
        px, py = cx + 2.8, cy - 0.2
        off = np.array([[px - np.rint(px), py - np.rint(py)]])
        dg = psf_weights(PSF, size, offset=off, derivative="x")[0]
        s_pos = _hand_sum(scene, dg, px, py)
        expect = c * (s_pos / scene.flux[i]) ** 2
        assert cols["positive_x_err_kfit"][i] ** 2 == pytest.approx(
            expect, rel=1e-9
        )
        # the center: the mean of the two lobes' shifts, cross term kept
        nx, ny = cx - 2.2, cy - 0.2
        off = np.array([[nx - np.rint(nx), ny - np.rint(ny)]])
        dg = psf_weights(PSF, size, offset=off, derivative="x")[0]
        s_neg = _hand_sum(scene, dg, nx, ny)
        expect = c * (0.5 * (s_pos - s_neg) / scene.flux[i]) ** 2
        assert cols["center_x_err_kfit"][i] ** 2 == pytest.approx(
            expect, rel=1e-5
        )
    assert res.diagnostics["center_window_px"] == size + 2 * 3


def test_lobe_kernel_fit_term_follows_the_sub_pixel_position():
    """A lobe centered on a reference star has no kernel-fit position term
    (the derivative weights are antisymmetric about the star); pixel-centered
    weights would give it the term of a lobe 0.4 px off the star.
    """
    s = Scene(seed=4, cand=[[80, 80], [140, 50], [100, 150], [40, 160]])
    star = _gaussians([80.4], [80.0], [5.0e4])
    s.reference = s.reference + star
    s.matched = fftconvolve(s.reference, s.kernel, mode="same")
    ctx = s.context(star_xy=None, misalignment=None)
    n = len(s.cand)
    on, off = [], []
    for dx, out in ((0.4, on), (1.4, off)):
        inp = s.inputs()
        inp["offsets"] = {
            "center": (np.full(n, dx), np.zeros(n)),
            "positive": (np.full(n, dx), np.zeros(n)),
        }
        inp["photometry_xy"] = np.column_stack([np.full(n, dx), np.zeros(n)])
        out.append(uq.budget(ctx, **inp).columns["positive_x_err_kfit"][0])
    assert off[0] > 0
    assert on[0] < 1e-3 * off[0]


def test_kernel_mismatch_term_is_eta_times_the_reference_flux(scene):
    import dataclasses

    ctx = dataclasses.replace(
        scene.context(),
        kernel_mismatch={"eta": 0.1, "n": 24},
        terms=("kfit", "kmatch", "mis"),
    )
    inp = scene.inputs()
    cols = uq.budget(ctx, **inp).columns
    half = ctx.window // 2
    # T is the matched reference at the lobe plus the modeled offset
    t, _ = uq.forced_flux(
        scene.matched,
        scene.variance,
        scene.cand[:, 0] + 0.3 + FIELD[0],
        scene.cand[:, 1] - 0.2 + FIELD[1],
        PSF,
        half,
        background="median",
    )
    np.testing.assert_allclose(
        cols["psf_flux_err_kmatch"], 0.1 * np.abs(t), rtol=1e-12
    )
    np.testing.assert_array_equal(cols["center_x_err_kmatch"], 0.0)


def test_validity_is_per_quantity_and_independent_of_terms():
    """The masked block covers more than half of the first candidate's flux
    and negative-lobe windows but under half of its positive-lobe window:
    psf_flux and the center are invalid, the positive lobe stays valid with
    the masked flag, whatever terms are on.
    """
    s = Scene(cand=[[44, 160], [140, 50], [100, 150], [60, 70]])
    for kw in (
        {},
        {"fit": None, "star_xy": None},
        {"fit": None, "star_xy": None, "misalignment": None},
    ):
        res = uq.budget(s.context(**kw), **s.inputs())
        c = res.columns
        assert c["uq_valid"].tolist() == [False, True, True, True]
        assert c["uq_window_masked"].tolist() == [True, False, False, False]
        assert res.diagnostics["invalid_window"] == 1
        assert res.diagnostics["masked_window"] == 1
        assert np.isnan(c["psf_flux_err"][0])
        assert np.isnan(c["center_x_err"][0])
        assert np.isnan(c["center_y_err"][0])
        assert np.isfinite(c["positive_x_err"][0])
        assert np.isfinite(c["positive_y_err"][0])
        # the term columns keep their values on the invalid row
        for q in QUANTITIES:
            for t in uq.TERMS:
                assert np.isfinite(c[f"{q}_err_{t}"][0])
        assert np.isfinite(c["positive_x_err"]).all()


def test_masked_window_keeps_the_error_bar_and_sets_the_flag():
    """A candidate whose windows hold a few masked pixels (3 of 19 columns
    of its flux window, 6 of the negative lobe's, none of the positive
    lobe's) keeps every error bar, built over the usable pixels, and is
    flagged; the same candidate on a clean scene is not flagged and its
    bars agree within the masked share.
    """
    from cuphoton.xrep import resampling_weights_at

    def resampling(x, y):
        return resampling_weights_at(
            lambda c: np.asarray(c, dtype=float) + 0.5,
            x,
            y,
            interpolation="lanczos3",
        )

    plane = np.full((SIZE, SIZE), 4.0)
    cand = [[52, 160], [140, 50], [100, 150], [60, 70]]
    masked = Scene(cand=cand)
    clean = Scene(cand=cand)
    clean.variance[150:170, 30:46] = NOISE**2
    clean.residual[150:170, 30:46] = (
        clean.target[150:170, 30:46] - (clean.matched[150:170, 30:46])
    )
    out = {}
    for name, s in (("masked", masked), ("clean", clean)):
        ctx = s.context(reference_variance=plane, resampling=resampling)
        assert ctx.terms == ("corr", "kfit", "kmatch", "mis")
        out[name] = uq.budget(ctx, **s.inputs())
    m, c = out["masked"].columns, out["clean"].columns
    assert m["uq_valid"].all() and c["uq_valid"].all()
    assert m["uq_window_masked"].tolist() == [True, False, False, False]
    assert not c["uq_window_masked"].any()
    assert out["masked"].diagnostics["masked_window"] == 1
    assert out["masked"].diagnostics["invalid_window"] == 0
    for q in QUANTITIES:
        assert np.isfinite(m[f"{q}_err"]).all()
        assert np.isfinite(m[f"{q}_err_corr"]).all()
        assert m[f"{q}_err_corr"][0] > 0
        # the masked row's bar stays close to the clean one (the factor
        # moves by under 20% with a mask this far from the center)
        assert m[f"{q}_err"][0] == pytest.approx(c[f"{q}_err"][0], rel=0.2)
        np.testing.assert_allclose(m[f"{q}_err"][1:], c[f"{q}_err"][1:])
    # the positive lobe's window is clean: its sums are the clean ones
    for a in "xy":
        assert m[f"positive_{a}_err_corr"][0] == pytest.approx(
            c[f"positive_{a}_err_corr"][0], rel=1e-9
        )


def test_correlation_factor_is_not_evaluated_when_excluded(
    scene, monkeypatch
):
    def boom(*a, **k):
        raise AssertionError("correlation_factor called")

    monkeypatch.setattr(uq, "correlation_factor", boom)
    ctx = scene.context()
    assert "corr" not in ctx.terms
    res = uq.budget(ctx, **scene.inputs())
    assert res.columns["uq_valid"].tolist() == [True, True, True, False]
    np.testing.assert_array_equal(res.columns["psf_flux_err_corr"], 0.0)


@pytest.mark.parametrize("enabled", [True, False])
def test_error_budget_runs_step_8c(scene, enabled):
    from transient_chain import catalog

    n = len(scene.cand)
    err = np.full(n, 0.05)
    derived = SimpleNamespace(
        center_x=np.full(n, 0.3),
        center_y=np.full(n, -0.2),
        positive_x=np.full(n, 2.8),
        positive_y=np.full(n, -0.2),
        center_x_error=err,
        center_y_error=err,
        positive_x_error=err,
        positive_y_error=err,
        lobe_flux=scene.flux,
    )
    u_cfg = {
        "enabled": enabled,
        "kernel_fit": True,
        "max_lobe_reach_px": MAX_REACH,
    }
    res, summary = uq.error_budget(
        u_cfg,
        cand_x=scene.cand[:, 0],
        cand_y=scene.cand[:, 1],
        derived=derived,
        single=np.zeros(n, bool),
        psf_flux_error=np.full(n, 40.0),
        photometry_xy=np.column_stack([np.full(n, 0.3), np.full(n, -0.2)]),
        fit=scene.fit,
        kernel=scene.kernel,
        reference=scene.reference,
        matched=scene.matched,
        residual=scene.residual,
        variance=scene.variance,
        psf_sigma=PSF,
        misalignment=scene.misalignment,
        star_xy=scene.star_xy,
    )
    cols = res.host_columns()
    budget_columns = {
        k for k in catalog.BUDGET_COLUMNS if not k.startswith("sky_err")
    }
    assert set(cols) == budget_columns | {f"{q}_err" for q in QUANTITIES}
    assert all(v.shape == (n,) for v in cols.values())
    assert cols["uq_valid"].dtype == bool
    assert cols["uq_window_masked"].dtype == bool
    if enabled:
        assert summary["terms"] == ["fit", "kfit", "kmatch", "mis"]
        assert summary["excluded"] == {"corr": "no reference variance plane"}
        assert summary["kernel_fit_form"] == "plain"
        assert summary["invalid_window"] == 1
        assert summary["masked_window"] == 0
        mismatch_bytes = 3 * summary["kernel_mismatch"]["n"] * 8
        assert res.control_bytes == 3 * n * 2 * 8 + 5 * n + mismatch_bytes
        assert "control_bytes" not in summary["kernel_mismatch"]
        assert np.all(cols["center_x_err"][:3] > 0.05)
    else:
        assert summary == {"terms": ["fit"], "excluded": {}, "disabled": True}
        assert res.control_bytes == 0
        np.testing.assert_array_equal(cols["center_x_err"], 0.05)
        np.testing.assert_array_equal(cols["psf_flux_err"], 40.0)
        assert cols["uq_valid"].all() and not cols["uq_window_masked"].any()


@pytest.mark.parametrize("stars", ["kept", "too_few", "rejected"])
def test_control_bytes_include_mismatch_downloads(scene, monkeypatch, stars):
    downloaded = []
    to_host = ops.to_host

    def record_download(array):
        host = to_host(array)
        downloaded.append(host.nbytes)
        return host

    monkeypatch.setattr(ops, "to_host", record_download)
    star_xy = scene.star_xy
    if stars == "too_few":
        star_xy = star_xy[:1]
    elif stars == "rejected":
        star_xy = np.zeros_like(star_xy)
    context = scene.context(star_xy=star_xy)
    result = uq.budget(context, **scene.inputs())
    assert result.control_bytes == sum(downloaded)
    assert "control_bytes" not in result.diagnostics["kernel_mismatch"]
    if stars == "kept":
        assert context.control_bytes > 0
    else:
        assert context.control_bytes == 0


def test_far_lobe_is_capped_out_of_the_center_stamp(scene):
    """One row whose positive lobe sits 120 px from its center (a diverged
    dipole fit; its mirrored lobe falls off the image) leaves the center
    stamp size as it is without it, loses its own center bars and is
    counted; every other row and its own positive-lobe and flux quantities
    are what they are without the cap.
    """
    ctx = scene.context()
    near = scene.inputs()
    far = scene.inputs()
    far["offsets"]["positive"][0][1] = -119.7
    base = uq.budget(ctx, **near)
    capped = uq.budget(ctx, **far)
    uncapped = uq.budget(ctx, **(far | {"max_lobe_reach_px": 1000.0}))
    assert base.diagnostics["center_reach_exceeded"] == 0
    assert capped.diagnostics["center_reach_exceeded"] == 1
    assert capped.diagnostics["max_lobe_reach_px"] == MAX_REACH
    assert (
        capped.diagnostics["center_window_px"]
        == base.diagnostics["center_window_px"]
        == ctx.window + 2 * 3
    )
    assert uncapped.diagnostics["center_window_px"] > ctx.window + 2 * 120
    assert uncapped.diagnostics["center_reach_exceeded"] == 0
    c, b, u = capped.columns, base.columns, uncapped.columns
    assert c["uq_valid"].tolist() == [True, False, True, False]
    assert b["uq_valid"].tolist() == [True, True, True, False]
    assert capped.diagnostics["invalid_window"] == 2
    for a in ("x", "y"):
        assert np.isnan(c[f"center_{a}_err"][1])
        assert np.isfinite(c[f"positive_{a}_err"][1])
    keep = [0, 2, 3]
    for name in b:
        np.testing.assert_array_equal(c[name][keep], b[name][keep])
    for q in ("positive_x", "positive_y", "psf_flux"):
        for col in [f"{q}_err_{t}" for t in uq.TERMS] + [f"{q}_err"]:
            np.testing.assert_array_equal(c[col], u[col])
