# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The error budget per candidate (worker step 8c): the terms beyond xFit's
covariance, combined into the error bar the catalog quotes.

For ``center_x``, ``center_y``, ``positive_x``, ``positive_y`` and
``psf_flux`` the budget adds to the fit term ``<q>_err_fit``:

- ``corr``: the fit term times ``sqrt(f - 1)``, ``f`` the correlation factor
  of the quantity's weighted sum against the diagonal variance plane the
  fit used (``cuphoton.xpois.correlation_factor``).
- ``kfit``: the variance the kernel fit's own uncertainty adds to the same
  weighted sum (``cuphoton.xpois.kernel_fit_variance``), in the quantity's
  units.
- ``kmatch``: the kernel mismatch ``eta`` measured at the chain's stars,
  times the kernel-matched reference flux at the candidate; flux only.
- ``mis``: the misalignment variance at the position, times 1/4 for a center
  and times the squared single-source sensitivity for the positive lobe.

The weighted sums stand in for xFit's estimators: the PSF-weighted sum at
the photometry position for the flux (the forced-photometry estimator), the
PSF-derivative-weighted sum at a lobe divided by the lobe flux for the
lobe's position, and the mean of the two lobes' sums for the center, built
as one weight stamp over both lobes so their kernel-fit covariance is kept.
Every weight stamp is centered on the sub-pixel position it measures.

Every term is a variance in the quantity's squared units; ``<q>_err`` is
their quadrature with the fit term. Arrays follow one array module (NumPy or
CuPy); only the misalignment model and the kernel-mismatch solve run on the
host, on arrays with one row per candidate or per star, and the device
arrays cross to the host three times per item: the positions the model
needs, the validity flags, and the stacked columns.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from cuphoton.xpois import (
    correlation_factor,
    kernel_fit_covariance,
    kernel_fit_variance,
    psf_weights,
    window_stamps,
)

from . import ops
from .catalog import BUDGET_QUANTITIES, BUDGET_TERMS

QUANTITIES = tuple(BUDGET_QUANTITIES)
POSITIONS = ("center", "positive")
TERMS = tuple(BUDGET_TERMS)
# beyond this the difference flux is under a tenth of the target flux: the
# lobe is the remainder of two nearly equal images and the sensitivity is set
# by the noise rather than by the source
SENSITIVITY_CAP = 10.0
MIN_MISMATCH_STARS = 5


@dataclass(frozen=True)
class BudgetContext:
    """Per-item inputs every candidate shares; see :func:`build_context`."""

    kernel: Any
    reference: Any
    matched: Any
    residual: Any
    weights_plane: Any  # 1 / this weights the sums; NaN where no weight
    target_variance: Any  # the same plane with 0 where no weight
    validity_plane: Any  # NaN where a window pixel invalidates the row
    psf_sigma: float | None
    window: int | None  # side of the window a weighted sum covers, px
    misalignment: ops.Misalignment | None
    kernel_covariance: Any | None
    kernel_mismatch: dict[str, Any]
    reference_variance: Any | None
    resampling: Callable[[Any, Any], Any] | None
    terms: tuple[str, ...]
    excluded: dict[str, str]


@dataclass(frozen=True)
class BudgetResult:
    """The budget columns as one ``(len(names), N)`` device array ``values``
    plus the host ``valid`` flags (``uq_valid``), the terms in the bars, the
    per-item diagnostics and the host bytes the budget moved.
    """

    names: tuple[str, ...]
    values: Any
    valid: np.ndarray
    terms: tuple[str, ...]
    diagnostics: dict[str, Any]
    control_bytes: int = 0

    @property
    def columns(self) -> dict[str, Any]:
        """One array per column in the values' array module."""
        cols = {name: self.values[i] for i, name in enumerate(self.names)}
        cols["uq_valid"] = self.valid
        return cols

    def host_columns(self) -> dict[str, np.ndarray]:
        """The columns on the host, one transfer for every float column."""
        values = ops.to_host(self.values)
        cols = {name: values[i] for i, name in enumerate(self.names)}
        cols["uq_valid"] = self.valid
        return cols


def window_side(psf_sigma: float) -> int:
    """Side of the window a PSF-weighted sum covers: 5 sigma each way."""
    return 2 * int(math.ceil(5.0 * float(psf_sigma))) + 1


def build_context(
    *,
    kernel,
    reference,
    matched,
    residual,
    variance,
    psf_sigma: float | None,
    misalignment: ops.Misalignment | None,
    star_xy: np.ndarray | None,
    fit=None,
    reference_variance=None,
    resampling: Callable[[Any, Any], Any] | None = None,
) -> BudgetContext:
    """Compute once per item what every candidate's budget shares.

    ``kernel``, ``reference`` (NaN where bad), ``matched`` and ``residual``
    are the subtraction's; ``variance`` the target variance plane (NaN where
    bad) or None. ``fit`` is the constant-kernel fit result when the kernel
    fit term is wanted, None to skip it. ``star_xy`` (N, 2) are the stars the
    refine stage kept, where the kernel mismatch is measured.
    ``reference_variance`` is the resampled reference's variance plane on the
    target grid (resampled units, as ``cuphoton.xpois.correlation_factor``
    takes it) and ``resampling(x, y)`` returns the
    ``cuphoton.xrep.ResamplingWeights`` at target pixels, for instance
    ``functools.partial(resampling_weights_at, spec, prepared=prepared)``
    with the template's ``ReprojectionSpec``; both None when the reference
    has no variance plane, which leaves the correlation term at zero. A term
    that cannot be computed is left out of ``terms`` with its reason in
    ``excluded``.
    """
    xp = ops.array_module(residual)
    excluded: dict[str, str] = {}
    finite = xp.isfinite(residual)
    if variance is None:
        weights_plane = xp.ones(residual.shape, dtype=xp.float64)
        validity = xp.where(finite, 1.0, xp.nan)
    else:
        weights_plane = xp.where(
            xp.isfinite(variance) & (variance > 0), variance, xp.nan
        )
        validity = xp.where(finite, weights_plane, xp.nan)
    target_variance = xp.where(xp.isfinite(weights_plane), weights_plane, 0.0)
    window = None if psf_sigma is None else window_side(psf_sigma)

    if (reference_variance is None) != (resampling is None):
        raise ValueError(
            "reference_variance and resampling are given together"
        )
    if reference_variance is None:
        excluded["corr"] = "no reference variance plane"
    if variance is None:
        excluded.setdefault("corr", "no target variance plane")
        excluded["kfit"] = "no target variance plane"
        excluded["kmatch"] = "no target variance plane"
    if psf_sigma is None:
        excluded.setdefault("corr", "no PSF width")
        excluded["kfit"] = "no PSF width"
        excluded.setdefault("kmatch", "no PSF width")

    cov = None
    if fit is None:
        excluded.setdefault("kfit", "kernel fit term disabled")
    elif "kfit" not in excluded:
        sandwich = {}
        if reference_variance is not None:
            ny, nx = residual.shape
            at = resampling(
                xp.asarray([(nx - 1) / 2.0]), xp.asarray([(ny - 1) / 2.0])
            )
            sandwich = dict(
                reference_variance=reference_variance,
                resampling_weights=at.weights[0],
            )
        cov = kernel_fit_covariance(
            fit,
            reference=reference,
            target_variance=weights_plane,
            **sandwich,
        )

    if misalignment is None:
        excluded["mis"] = "no misalignment model"
    if star_xy is None or len(star_xy) < MIN_MISMATCH_STARS:
        excluded.setdefault(
            "kmatch", f"fewer than {MIN_MISMATCH_STARS} matched stars"
        )

    ctx = BudgetContext(
        kernel=kernel,
        reference=reference,
        matched=matched,
        residual=residual,
        weights_plane=weights_plane,
        target_variance=target_variance,
        validity_plane=validity,
        psf_sigma=psf_sigma,
        window=window,
        misalignment=misalignment,
        kernel_covariance=cov,
        kernel_mismatch={"eta": 0.0, "n": 0},
        reference_variance=reference_variance,
        resampling=resampling,
        terms=tuple(t for t in TERMS if t not in excluded),
        excluded=excluded,
    )
    if "kmatch" in ctx.terms:
        mismatch = kernel_mismatch(ctx, np.asarray(star_xy, dtype=np.float64))
        ctx = dataclasses.replace(ctx, kernel_mismatch=mismatch)
    return ctx


# --- windows and weighted sums ---------------------------------------------


def _positions(xp, x, y):
    return xp.stack(
        [xp.asarray(x, dtype=xp.float64), xp.asarray(y, dtype=xp.float64)],
        axis=1,
    )


def _valid_positions(xp, x, y, shape):
    ny, nx = shape
    return (
        xp.isfinite(x)
        & xp.isfinite(y)
        & (x >= 0)
        & (x <= nx - 1)
        & (y >= 0)
        & (y <= ny - 1)
    )


def _safe_positions(xp, x, y, cx, cy, shape):
    """Positions for image lookups: ``(x, y)`` where finite and inside the
    image, else the candidate pixel ``(cx, cy)``.
    """
    ok = _valid_positions(xp, x, y, shape)
    return xp.where(ok, x, cx), xp.where(ok, y, cy)


def _valid_windows(ctx: BudgetContext, pos):
    """True where every pixel of the window at ``pos`` (N, 2) is valid."""
    xp = ops.array_module(ctx.residual)
    stamps = window_stamps(ctx.validity_plane, pos, ctx.window)
    return xp.isfinite(stamps).all(axis=(1, 2))


def _inverse_variance(ctx: BudgetContext, pos, size: int):
    """``1 / variance`` on the windows at ``pos``, 0 where no weight."""
    xp = ops.array_module(ctx.residual)
    plane = window_stamps(ctx.weights_plane, pos, size)
    good = xp.isfinite(plane)
    return xp.where(good, 1.0 / xp.where(good, plane, 1.0), 0.0)


def _matched_filter(xp, inverse_variance, stamp):
    """Weights ``w = inverse_variance * stamp / sum(inverse_variance *
    stamp**2)`` of the least-squares amplitude of ``stamp`` on each window,
    and whether the normalization was usable (0 weights otherwise).
    """
    w = inverse_variance * stamp
    norm = (w * stamp).sum(axis=(1, 2))
    ok = xp.isfinite(norm) & (norm > 0)
    return w / xp.where(ok, norm, 1.0)[:, None, None], ok


def _estimator_terms(ctx: BudgetContext, weights, pos):
    """Correlation factor and kernel-fit variance of the weighted sums
    ``sum(weights * D)`` on the windows centered at ``pos`` (N, 2).

    The factor is 1 when the correlation term is excluded, else
    ``correlation_factor`` on the planes: NaN where the grid mapping is not
    locally a translation or the reprojection filled the reference there.
    The kernel-fit variance is 0 without a kernel covariance or where the
    stamp's kernel footprint leaves the image.
    """
    xp = ops.array_module(ctx.residual)
    n = int(pos.shape[0])
    if "corr" in ctx.terms:
        at = ctx.resampling(pos[:, 0], pos[:, 1])
        factor = correlation_factor(
            weights,
            kernel=ctx.kernel,
            target_variance=ctx.target_variance,
            reference_variance=ctx.reference_variance,
            resampling_weights=at.weights,
            positions=pos,
            residual_variance=ctx.target_variance,
            is_translation=at.is_translation,
        ).factor
    else:
        factor = xp.ones(n, dtype=xp.float64)
    kfit = xp.zeros(n, dtype=xp.float64)
    if ctx.kernel_covariance is not None:
        kfit = kernel_fit_variance(
            ctx.kernel_covariance,
            reference=ctx.reference,
            positions=pos,
            weights=weights,
        )
        kfit = xp.where(xp.isfinite(kfit), kfit, 0.0)
    return factor, kfit


# --- forced fluxes ---------------------------------------------------------


def forced_flux(image, variance, x, y, psf_sigma, half, *, background=None):
    """Forced PSF flux and its error at ``(x, y)``: the weighted least
    squares of a unit-flux Gaussian on windows of ``image``, ``w = 1 /
    variance`` on the finite pixels, ``flux = sum(w P D) / sum(w P^2)``.
    ``background="median"`` subtracts each window's median first (a local
    sky for the matched reference). NaN where no pixel is usable.

    The same estimator as ``cuphoton.xfit.forced_psf_photometry``, which
    measures stamps already cut around the candidates; this one cuts windows
    of a full plane at any positions (the stars, the matched reference) and
    takes the local sky option.
    """
    xp = ops.array_module(image)
    pos = _positions(xp, x, y)
    size = 2 * half + 1
    d = window_stamps(image, pos, size)
    v = window_stamps(variance, pos, size)
    if background == "median":
        d = d - xp.nanmedian(d, axis=(1, 2), keepdims=True)
    good = xp.isfinite(d) & xp.isfinite(v) & (v > 0)
    w = xp.where(good, 1.0 / xp.where(good, v, 1.0), 0.0)
    psf = psf_weights(psf_sigma, size, offset=pos - xp.rint(pos), xp=xp)
    information = (w * psf * psf).sum(axis=(1, 2))
    ok = information > 0
    safe = xp.where(ok, information, 1.0)
    flux = (w * psf * xp.where(good, d, 0.0)).sum(axis=(1, 2)) / safe
    return (
        xp.where(ok, flux, xp.nan),
        xp.where(ok, 1.0 / xp.sqrt(safe), xp.nan),
    )


# --- the misalignment model on the host -----------------------------------


def misalignment_field(m: ops.Misalignment | None, x, y) -> np.ndarray:
    """Modeled reference-minus-target offset (N, 2) at host positions: the
    affine field left after refinement plus the fitted excess field (zero in
    rms mode and without a model).
    """
    xy = np.column_stack([np.ravel(x), np.ravel(y)]).astype(np.float64)
    if m is None:
        return np.zeros_like(xy)
    return xy @ m.matrix.T + m.offset - xy + m.excess(xy[:, 0], xy[:, 1])[0]


def misalignment_variance(m: ops.Misalignment | None, x, y) -> np.ndarray:
    """Per-axis misalignment variance (N, 2) at host positions, zero without
    a model.
    """
    if m is None:
        return np.zeros((np.size(x), 2))
    return m.variance(np.ravel(x), np.ravel(y))


# --- the terms --------------------------------------------------------------


def kernel_mismatch(ctx: BudgetContext, star_xy: np.ndarray) -> dict:
    """``eta``: the fraction of a static star's kernel-matched reference flux
    that the subtraction leaves in the residual beyond noise, from the
    chain's stars. At each star the forced flux ``D`` of the residual has
    predicted variance ``uD^2 f + kfit`` when the kernel is right; ``eta^2``
    solves ``median(D^2 / (uD^2 f + kfit + eta^2 T^2))`` equal to the
    chi-square(1) median, so a variable star or two among the stars does not
    set it. Stars whose window is not fully valid are left out.
    """
    xp = ops.array_module(ctx.residual)
    half = ctx.window // 2
    pos = xp.asarray(star_xy, dtype=xp.float64)
    x, y = pos[:, 0], pos[:, 1]
    out = {"eta": 0.0, "n": int(len(star_xy)), "median_ratio": None}
    if len(star_xy) < MIN_MISMATCH_STARS:
        return out
    field = xp.asarray(misalignment_field(ctx.misalignment, *star_xy.T))
    d, ud = forced_flux(
        ctx.residual, ctx.weights_plane, x, y, ctx.psf_sigma, half
    )
    tx, ty = _safe_positions(
        xp, x + field[:, 0], y + field[:, 1], x, y, ctx.residual.shape
    )
    t, _ = forced_flux(
        ctx.matched,
        ctx.weights_plane,
        tx,
        ty,
        ctx.psf_sigma,
        half,
        background="median",
    )
    psf = psf_weights(
        ctx.psf_sigma, ctx.window, offset=pos - xp.rint(pos), xp=xp
    )
    w, _ = _matched_filter(xp, _inverse_variance(ctx, pos, ctx.window), psf)
    f, kfit = _estimator_terms(ctx, w, pos)
    base = ud**2 * xp.where(f > 1.0, f, 1.0) + kfit
    ok = (
        _valid_windows(ctx, pos)
        & xp.isfinite(d)
        & xp.isfinite(base)
        & xp.isfinite(t)
        & (t > 0)
        & (base > 0)
    )
    d, t, base = ops.to_host(xp.stack([d, t, base])[:, ok])
    out["n"] = int(len(d))
    if len(d) < MIN_MISMATCH_STARS:
        return out
    eta2 = ops.chi2_median_scale(d**2, base, t**2)
    out.update(
        eta=float(math.sqrt(eta2)),
        median_ratio=float(np.median(d / t)),
        chi2_median_without=float(np.median(d**2 / base)),
        reference_flux_median=float(np.median(t)),
    )
    return out


def sensitivity(ctx: BudgetContext, pos, cx, cy, single, field):
    """Single-source sensitivity ``lambda = (D + T) / D`` of a lobe to the
    misalignment, with the forced fluxes ``D`` of the residual and ``T`` of
    the kernel-matched reference at the lobe plus the modeled offset
    ``field`` (N, 2); 1 for a resolved dipole. A single source's target
    image sits at the lobe plus ``(lambda - 1)`` times the offset, so the
    fluxes are re-measured there (three passes). ``|lambda|`` is capped at
    ``SENSITIVITY_CAP``. Returns ``(lambda, raw lambda, T, positions)`` at
    the final positions.
    """
    xp = ops.array_module(ctx.residual)
    half = ctx.window // 2
    shape = ctx.residual.shape
    x, y = pos[:, 0], pos[:, 1]
    xf, yf = x, y
    for it in range(3):
        d, _ = forced_flux(
            ctx.residual, ctx.weights_plane, xf, yf, ctx.psf_sigma, half
        )
        tx, ty = _safe_positions(
            xp, xf + field[:, 0], yf + field[:, 1], cx, cy, shape
        )
        t, _ = forced_flux(
            ctx.matched,
            ctx.weights_plane,
            tx,
            ty,
            ctx.psf_sigma,
            half,
            background="median",
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = xp.where(
                single & xp.isfinite(d + t) & (d != 0), (d + t) / d, 1.0
            )
        lam = xp.clip(raw, -SENSITIVITY_CAP, SENSITIVITY_CAP)
        if it < 2:
            xf, yf = _safe_positions(
                xp,
                x + (lam - 1.0) * field[:, 0],
                y + (lam - 1.0) * field[:, 1],
                cx,
                cy,
                shape,
            )
    return lam, raw, t, _positions(xp, xf, yf)


def budget(
    ctx: BudgetContext,
    *,
    cand_x,
    cand_y,
    offsets: dict[str, tuple[Any, Any]],
    err_fit: dict[str, Any],
    lobe_flux,
    single,
    max_lobe_reach_px: float,
    photometry_xy=None,
) -> BudgetResult:
    """Every budget term and the full error bar per candidate.

    ``cand_x``, ``cand_y`` are the candidate pixels; ``offsets`` maps
    ``"center"`` and ``"positive"`` to ``(dx, dy)`` arrays relative to them;
    ``err_fit`` maps each quantity in :data:`QUANTITIES` to its fit term
    (``psf_flux`` absent or None when forced photometry did not run);
    ``lobe_flux`` is the fitted lobe flux and ``single`` marks candidates
    with one supported lobe. ``photometry_xy`` (N, 2) is the offset forced
    photometry measured at, None without it. ``max_lobe_reach_px`` caps
    each lobe's distance from the center pixel; rows beyond it get no
    center error bar and cannot inflate every candidate's center stamp.

    Returns the ``<q>_err_{corr,kfit,kmatch,mis}`` and ``<q>_err`` columns
    and ``uq_valid``. A quantity is valid where every pixel of its window
    (both lobes' windows for a center) is valid and its correlation factor
    is defined; ``<q>_err`` is NaN elsewhere while the term columns keep
    their values (``<q>_err_corr`` NaN with the factor). ``uq_valid`` is
    true where every computed quantity is valid.
    """
    xp = ops.array_module(ctx.residual)
    cx = xp.asarray(cand_x, dtype=xp.float64)
    cy = xp.asarray(cand_y, dtype=xp.float64)
    n = int(cx.shape[0])
    shape = ctx.residual.shape
    single = xp.asarray(single, dtype=bool)
    flux = xp.asarray(lobe_flux, dtype=xp.float64)
    quantities = [q for q in QUANTITIES if err_fit.get(q) is not None]
    zeros = xp.zeros(n, dtype=xp.float64)
    ones = xp.ones(n, dtype=xp.float64)
    true = xp.ones(n, dtype=bool)
    var = {q: {t: zeros for t in TERMS} for q in quantities}
    factor = {q: ones for q in quantities}
    window_ok = {q: true for q in quantities}
    diag: dict[str, Any] = {
        "window_px": ctx.window,
        "kernel_fit_form": None
        if ctx.kernel_covariance is None
        else ctx.kernel_covariance.form,
        "kernel_mismatch": ctx.kernel_mismatch,
    }

    # positions on the device, the candidate pixel where a fit gave none
    pos = {}
    for name in POSITIONS:
        dx, dy = offsets[name]
        pos[name] = _positions(
            xp,
            cx + xp.asarray(dx, dtype=xp.float64),
            cy + xp.asarray(dy, dtype=xp.float64),
        )
    pos["negative"] = 2.0 * pos["center"] - pos["positive"]
    if photometry_xy is None:
        pos["flux"] = pos["positive"]
    else:
        pxy = xp.asarray(photometry_xy, dtype=xp.float64)
        pos["flux"] = _positions(xp, cx + pxy[:, 0], cy + pxy[:, 1])
    position_ok = {
        name: _valid_positions(xp, at[:, 0], at[:, 1], shape)
        for name, at in pos.items()
    }
    for axis in ("x", "y"):
        window_ok[f"center_{axis}"] = (
            position_ok["positive"] & position_ok["negative"]
        )
        window_ok[f"positive_{axis}"] = position_ok["positive"]
    if "psf_flux" in quantities:
        window_ok["psf_flux"] = position_ok["flux"]
    pos = {
        name: _positions(
            xp, *_safe_positions(xp, at[:, 0], at[:, 1], cx, cy, shape)
        )
        for name, at in pos.items()
    }
    # one transfer: the host-side misalignment model's inputs
    host = ops.to_host(
        xp.stack([pos["center"], pos["positive"], pos["flux"]])
    )
    control_bytes = int(host.nbytes)
    model = xp.asarray(
        np.concatenate(
            [
                misalignment_field(ctx.misalignment, *host[2].T),
                misalignment_variance(ctx.misalignment, *host[0].T),
                misalignment_variance(ctx.misalignment, *host[1].T),
            ],
            axis=1,
        )
    )
    field, mis_center, mis_positive = (
        model[:, :2],
        model[:, 2:4],
        model[:, 4:],
    )
    # the center's stamp holds both lobes: this far from the center pixel
    center_pixel = np.rint(host[0])
    reach = 0
    lobes = np.abs(
        np.stack([host[1], 2.0 * host[0] - host[1]]) - center_pixel
    )
    distance = lobes.max(axis=(0, 2)) if n else np.zeros(0)
    exceeded = distance > max_lobe_reach_px
    within = distance[~exceeded]
    if within.size and np.isfinite(within).any():
        reach = int(math.ceil(min(np.nanmax(within), max_lobe_reach_px)))
    for axis in ("x", "y"):
        window_ok[f"center_{axis}"] &= ~xp.asarray(exceeded)

    lam = ones
    capped = xp.zeros(n, dtype=bool)
    if ctx.window is not None:
        valid_at = {
            name: _valid_windows(ctx, pos[name])
            for name in ("positive", "negative", "flux")
        }
        for a in ("x", "y"):
            window_ok[f"center_{a}"] = (
                window_ok[f"center_{a}"]
                & valid_at["positive"]
                & valid_at["negative"]
            )
            window_ok[f"positive_{a}"] = (
                window_ok[f"positive_{a}"] & valid_at["positive"]
            )
        if "psf_flux" in quantities:
            window_ok["psf_flux"] = window_ok["psf_flux"] & valid_at["flux"]
        lam, raw, t_ref, _ = sensitivity(
            ctx, pos["flux"], cx, cy, single, field
        )
        capped = xp.abs(raw) > SENSITIVITY_CAP
        sums = "corr" in ctx.terms or "kfit" in ctx.terms
        if sums:
            # a lobe: PSF-derivative weights at the lobe, divided by the lobe
            # flux so the sum is a shift in pixels
            flux_ok = xp.isfinite(flux) & (flux != 0)
            gain = xp.where(
                flux_ok, 1.0 / xp.where(flux_ok, flux, 1.0) ** 2, 0
            )
            lobe_inv = _inverse_variance(ctx, pos["positive"], ctx.window)
            lobe_off = pos["positive"] - xp.rint(pos["positive"])
            # the center: the mean of both lobes' shifts, one stamp over both
            big = ctx.window + 2 * reach
            c0 = xp.asarray(center_pixel)
            center_inv = _inverse_variance(ctx, c0, big)
            off = {
                "positive": pos["positive"] - c0,
                "negative": pos["negative"] - c0,
            }
            for a in ("x", "y"):
                dg = psf_weights(
                    ctx.psf_sigma,
                    ctx.window,
                    offset=lobe_off,
                    derivative=a,
                    xp=xp,
                )
                w, ok = _matched_filter(xp, lobe_inv, dg)
                f, kfit = _estimator_terms(ctx, w, pos["positive"])
                factor[f"positive_{a}"] = f
                var[f"positive_{a}"]["kfit"] = kfit * gain * ok
                stamps = {
                    name: _matched_filter(
                        xp,
                        center_inv,
                        psf_weights(
                            ctx.psf_sigma,
                            big,
                            offset=off[name],
                            derivative=a,
                            xp=xp,
                        ),
                    )
                    for name in ("positive", "negative")
                }
                w = 0.5 * (stamps["positive"][0] - stamps["negative"][0])
                f, kfit = _estimator_terms(ctx, w, c0)
                factor[f"center_{a}"] = f
                var[f"center_{a}"]["kfit"] = (
                    kfit
                    * gain
                    * (stamps["positive"][1] & stamps["negative"][1])
                )
        if "psf_flux" in quantities:
            if sums:
                psf = psf_weights(
                    ctx.psf_sigma,
                    ctx.window,
                    offset=pos["flux"] - xp.rint(pos["flux"]),
                    xp=xp,
                )
                w, ok = _matched_filter(
                    xp, _inverse_variance(ctx, pos["flux"], ctx.window), psf
                )
                f, kfit = _estimator_terms(ctx, w, pos["flux"])
                factor["psf_flux"] = f
                var["psf_flux"]["kfit"] = kfit * ok
            if "kmatch" in ctx.terms:
                eta = float(ctx.kernel_mismatch.get("eta", 0.0))
                var["psf_flux"]["kmatch"] = (
                    eta * xp.where(xp.isfinite(t_ref), xp.abs(t_ref), 0.0)
                ) ** 2
    if "mis" in ctx.terms:
        for name, gain, mv in (
            ("center", 0.25 * ones, mis_center),
            ("positive", lam**2, mis_positive),
        ):
            for j, a in enumerate(("x", "y")):
                var[f"{name}_{a}"]["mis"] = gain * mv[:, j]

    names: list[str] = []
    rows: list[Any] = []
    valid = windows = true
    undefined = xp.zeros(n, dtype=bool)
    for q in quantities:
        fit = xp.asarray(err_fit[q], dtype=xp.float64)
        f = factor[q]
        defined = xp.isfinite(f)
        f = xp.where(defined & (f > 1.0), f, 1.0)
        windows = windows & window_ok[q]
        undefined = undefined | ~defined
        corr = fit**2 * (f - 1.0)
        total = xp.sqrt(
            fit**2 + corr + var[q]["kfit"] + var[q]["kmatch"] + var[q]["mis"]
        )
        ok = window_ok[q] & defined & xp.isfinite(total)
        valid = valid & ok
        names += [f"{q}_err_{t}" for t in TERMS] + [f"{q}_err"]
        rows += [
            xp.where(defined, xp.sqrt(corr), xp.nan),
            xp.sqrt(var[q]["kfit"]),
            xp.sqrt(var[q]["kmatch"]),
            xp.sqrt(var[q]["mis"]),
            xp.where(ok, total, xp.nan),
        ]
    values = xp.stack(rows) if rows else xp.zeros((0, n), dtype=xp.float64)
    # one transfer: the flags the summary counts
    flags = ops.to_host(xp.stack([valid, windows, undefined, capped]))
    control_bytes += int(flags.nbytes)
    diag.update(
        max_lobe_reach_px=max_lobe_reach_px,
        center_reach_exceeded=int(exceeded.sum()),
        invalid_window=int((~flags[1]).sum()),
        rotated_grid=int(flags[2].sum()),
        sensitivity_capped=int(flags[3].sum()),
        center_window_px=None
        if ctx.window is None
        else ctx.window + 2 * reach,
    )
    return BudgetResult(
        tuple(names), values, flags[0], ctx.terms, diag, control_bytes
    )


def fit_only(err_fit: dict[str, Any]) -> BudgetResult:
    """The columns of :func:`budget` when the budget is disabled: every term
    zero and ``<q>_err`` equal to the fit term.
    """
    names: list[str] = []
    rows: list[Any] = []
    xp, n = np, 0
    for q in QUANTITIES:
        fit = err_fit.get(q)
        if fit is None:
            continue
        xp = ops.array_module(fit)
        fit = xp.asarray(fit, dtype=xp.float64)
        n = int(fit.shape[0])
        names += [f"{q}_err_{t}" for t in TERMS] + [f"{q}_err"]
        rows += [xp.zeros(n, dtype=xp.float64)] * len(TERMS) + [fit]
    values = xp.stack(rows) if rows else np.zeros((0, 0))
    return BudgetResult(
        tuple(names), values, np.ones(n, dtype=bool), (), {"disabled": True}
    )


def error_budget(
    u_cfg: Mapping[str, Any],
    *,
    cand_x,
    cand_y,
    derived,
    single,
    psf_flux_error=None,
    photometry_xy=None,
    fit=None,
    **context,
) -> tuple[BudgetResult, dict[str, Any]]:
    """Worker step 8c as one call: the budget per the ``uq`` configuration.

    ``cand_x``, ``cand_y`` are the host candidate pixels; ``derived`` the
    dipole quantities (``center_x`` ... ``positive_y_error`` and
    ``lobe_flux`` on the device); ``single`` marks candidates with one
    supported lobe; ``psf_flux_error`` and ``photometry_xy`` come from
    forced photometry (None without it); ``fit`` and ``context`` are
    :func:`build_context`'s inputs. With ``uq.enabled`` false the result is
    :func:`fit_only`. Returns the result and the ``uq`` entry of the item
    summary.
    """
    err_fit = {
        f"{name}_{axis}": getattr(derived, f"{name}_{axis}_error")
        for name in POSITIONS
        for axis in ("x", "y")
    }
    if psf_flux_error is not None:
        err_fit["psf_flux"] = psf_flux_error
    if not u_cfg["enabled"]:
        res = fit_only(err_fit)
        return res, {"terms": ["fit"], "excluded": {}, **res.diagnostics}
    ctx = build_context(fit=fit if u_cfg["kernel_fit"] else None, **context)
    xp = ops.array_module(derived.center_x)
    cand = xp.asarray(np.stack([cand_x, cand_y]).astype(np.float64))
    res = budget(
        ctx,
        cand_x=cand[0],
        cand_y=cand[1],
        offsets={
            name: (
                getattr(derived, f"{name}_x"),
                getattr(derived, f"{name}_y"),
            )
            for name in POSITIONS
        },
        err_fit=err_fit,
        lobe_flux=derived.lobe_flux,
        single=single,
        max_lobe_reach_px=float(u_cfg["max_lobe_reach_px"]),
        photometry_xy=photometry_xy,
    )
    return res, {
        "terms": ["fit", *res.terms],
        "excluded": ctx.excluded,
        **res.diagnostics,
    }
