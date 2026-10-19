# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The error budget per candidate (worker step 8c): the terms beyond xFit's
covariance, combined into the error bar the catalog quotes.

For ``center_x``, ``center_y``, ``positive_x``, ``positive_y`` and
``psf_flux`` the budget adds to the fit term ``<q>_err_fit``:

- ``corr``: the fit term times ``sqrt(f - 1)``, ``f`` the correlation factor
  of a PSF-weighted sum (flux) or a PSF-derivative-weighted sum (position)
  against the diagonal variance plane the fit used
  (``cuphoton.xpois.correlation_factor``).
- ``kfit``: the variance the kernel fit's own uncertainty adds to the same
  weighted sum (``cuphoton.xpois.kernel_fit_variance``), in the quantity's
  units: positions use the derivative weights normalized by the lobe flux.
- ``kmatch``: the kernel mismatch ``eta`` measured at the chain's stars,
  times the kernel-matched reference flux at the candidate; flux only.
- ``mis``: the misalignment variance at the position, times 1/4 for a center
  and times the squared single-source sensitivity for the positive lobe.

Every term is a variance in the quantity's squared units; ``<q>_err`` is
their quadrature with the fit term. Arrays follow one array module (NumPy or
CuPy); only the misalignment model and the kernel-mismatch solve run on the
host, on arrays with one row per candidate or per star.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import ops

QUANTITIES = ("center_x", "center_y", "positive_x", "positive_y", "psf_flux")
POSITIONS = ("center", "positive")
TERMS = ("corr", "kfit", "kmatch", "mis")
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
    weights_plane: Any  # 1 / this weights the sums (ones without a plane)
    validity_plane: Any  # NaN where a window pixel invalidates the factor
    psf_sigma: float | None
    window: int  # side of the correlation window, px
    misalignment: ops.Misalignment | None
    kernel_covariance: Any | None
    kernel_mismatch: dict[str, Any]
    reference_variance: Any | None
    resampling: Callable[[Any, Any], Any] | None
    terms: tuple[str, ...]
    excluded: dict[str, str]


@dataclass(frozen=True)
class BudgetResult:
    """Catalog columns (one array per candidate) and per-item diagnostics."""

    columns: dict[str, Any]
    terms: tuple[str, ...]
    diagnostics: dict[str, Any]


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
    target grid and ``resampling(x, y)`` returns the
    ``cuphoton.xrep.ResamplingWeights`` at target pixels; both None when the
    reference has no variance plane, which leaves the correlation term at
    zero. A term that cannot be computed is left out of ``terms`` with its
    reason in ``excluded``.
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
    window = None if psf_sigma is None else window_side(psf_sigma)

    if (reference_variance is None) != (resampling is None):
        raise ValueError(
            "reference_variance and resampling are given together"
        )
    if reference_variance is None:
        excluded["corr"] = "no reference variance plane"
    if psf_sigma is None:
        excluded.setdefault("corr", "no PSF width")
        excluded["kfit"] = "no PSF width"
        excluded["kmatch"] = "no PSF width"

    cov = None
    if fit is None:
        excluded.setdefault("kfit", "kernel fit term disabled")
    elif "kfit" not in excluded:
        from cuphoton.xpois import kernel_fit_covariance

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
    if variance is None:
        excluded.setdefault("kmatch", "no target variance plane")
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


# --- forced fluxes ---------------------------------------------------------


def _psf_stamps(xp, dx, dy, sigma, half):
    """Unit-flux point-sampled Gaussians, (N, s, s), centered ``(dx, dy)``
    from the window's center pixel.
    """
    u = xp.arange(-half, half + 1, dtype=xp.float64)
    gx = xp.exp(-0.5 * ((u[None, :] - dx[:, None]) / sigma) ** 2)
    gy = xp.exp(-0.5 * ((u[None, :] - dy[:, None]) / sigma) ** 2)
    return gy[:, :, None] * gx[:, None, :] / (2.0 * math.pi * sigma**2)


def forced_flux(image, variance, x, y, psf_sigma, half, *, background=None):
    """Forced PSF flux and its error at ``(x, y)``: the weighted least
    squares of a unit-flux Gaussian on windows of ``image``, ``w = 1 /
    variance`` on the finite pixels, ``flux = sum(w P D) / sum(w P^2)``.
    ``background="median"`` subtracts each window's median first (a local
    sky for the matched reference). NaN where no pixel is usable.
    """
    xp = ops.array_module(image)
    x = xp.asarray(x, dtype=xp.float64)
    y = xp.asarray(y, dtype=xp.float64)
    d = ops._patches(image, x, y, half)
    v = ops._patches(variance, x, y, half)
    if background == "median":
        d = d - xp.nanmedian(d, axis=(1, 2), keepdims=True)
    good = xp.isfinite(d) & xp.isfinite(v) & (v > 0)
    w = xp.where(good, 1.0 / xp.where(good, v, 1.0), 0.0)
    psf = _psf_stamps(xp, x - xp.rint(x), y - xp.rint(y), psf_sigma, half)
    information = (w * psf * psf).sum(axis=(1, 2))
    ok = information > 0
    safe = xp.where(ok, information, 1.0)
    flux = (w * psf * xp.where(good, d, 0.0)).sum(axis=(1, 2)) / safe
    return (
        xp.where(ok, flux, xp.nan),
        xp.where(ok, 1.0 / xp.sqrt(safe), xp.nan),
    )


def _safe_positions(xp, x, y, cx, cy, shape):
    """Positions for image lookups: ``(x, y)`` where finite and inside the
    image, else the candidate pixel ``(cx, cy)``.
    """
    ny, nx = shape
    ok = (
        xp.isfinite(x)
        & xp.isfinite(y)
        & (x >= 0)
        & (x <= nx - 1)
        & (y >= 0)
        & (y <= ny - 1)
    )
    return xp.where(ok, x, cx), xp.where(ok, y, cy)


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


def _weighted_sum_terms(ctx: BudgetContext, weights, x, y, *, scale=None):
    """Correlation factor and kernel-fit variance of the weighted sums
    ``sum(weights * D)`` on windows centered at ``rint(x), rint(y)``.

    ``weights`` is ``(N, s, s)`` or ``(s, s)``; it is multiplied by the
    inverse weights plane (the fit's weights) and, for the kernel-fit term,
    by ``scale`` (N,) that converts the sum to the quantity's units. Returns
    ``(factor, kfit_variance, rotated)``: the factor is NaN where the window
    holds an invalid pixel or, with a reference variance plane, where the
    grid mapping is not locally a translation (``rotated`` true); the
    kernel-fit variance is 0 where unavailable.
    """
    from cuphoton.xpois import correlation_factor, kernel_fit_variance

    xp = ops.array_module(ctx.residual)
    half = ctx.window // 2
    x = xp.asarray(x, dtype=xp.float64)
    y = xp.asarray(y, dtype=xp.float64)
    n = int(x.shape[0])
    validity = ops._patches(ctx.validity_plane, x, y, half)
    plane = ops._patches(ctx.weights_plane, x, y, half)
    good = xp.isfinite(plane)
    w = xp.where(good, 1.0 / xp.where(good, plane, 1.0), 0.0)
    w = w * xp.broadcast_to(xp.asarray(weights, dtype=xp.float64), w.shape)

    rotated = xp.zeros(n, dtype=bool)
    if ctx.reference_variance is None:
        source = 0.0
        stamps = xp.ones((1, 1), dtype=xp.float64)
    else:
        at = ctx.resampling(x, y)
        stamps = at.weights
        rotated = ~xp.asarray(at.is_translation, dtype=bool)
        source = ops._patches(ctx.reference_variance, x, y, half)
    factor = correlation_factor(
        w,
        kernel=ctx.kernel,
        target_variance=validity,
        reference_variance=source,
        resampling_weights=stamps,
    ).factor
    factor = xp.where(rotated, xp.nan, factor)

    kfit = xp.zeros(n, dtype=xp.float64)
    if ctx.kernel_covariance is not None:
        if scale is not None:
            w = w * xp.asarray(scale, dtype=xp.float64)[:, None, None]
        kfit = kernel_fit_variance(
            ctx.kernel_covariance,
            reference=ctx.reference,
            positions=xp.stack([xp.rint(x), xp.rint(y)], axis=1),
            weights=w,
        )
        kfit = xp.where(xp.isfinite(kfit), kfit, 0.0)
    return factor, kfit, rotated


def _derivative_scale(xp, weights_plane_windows, dg, flux):
    """Factor turning ``sum(w dg D)`` into a position shift in pixels:
    ``1 / (F sum(w dg^2))``, 0 where the lobe flux is not usable.
    """
    good = xp.isfinite(weights_plane_windows)
    w = xp.where(good, 1.0 / xp.where(good, weights_plane_windows, 1.0), 0.0)
    norm = xp.abs(flux) * (w * dg**2).sum(axis=(1, 2))
    ok = xp.isfinite(norm) & (norm > 0)
    return xp.where(ok, 1.0 / xp.where(ok, norm, 1.0), 0.0)


def kernel_mismatch(ctx: BudgetContext, star_xy: np.ndarray) -> dict:
    """``eta``: the fraction of a static star's kernel-matched reference flux
    that the subtraction leaves in the residual beyond noise, from the
    chain's stars. At each star the forced flux ``D`` of the residual has
    predicted variance ``uD^2 f + kfit`` when the kernel is right; ``eta^2``
    solves ``median(D^2 / (uD^2 f + kfit + eta^2 T^2))`` equal to the
    chi-square(1) median, so a variable star or two among the stars does not
    set it. Stars whose correlation window is not fully valid are left out.
    """
    from cuphoton.xpois import psf_weights

    xp = ops.array_module(ctx.residual)
    half = ctx.window // 2
    x = xp.asarray(star_xy[:, 0])
    y = xp.asarray(star_xy[:, 1])
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
    psf = xp.asarray(psf_weights(ctx.psf_sigma, ctx.window))
    plane = ops._patches(ctx.weights_plane, x, y, half)
    scale = _derivative_scale(xp, plane, psf, xp.ones(x.shape[0]))
    f, kfit, _ = _weighted_sum_terms(ctx, psf, x, y, scale=scale)
    base = ud**2 * xp.where(f < 1.0, 1.0, f) + kfit
    ok = (
        xp.isfinite(d)
        & xp.isfinite(base)
        & xp.isfinite(t)
        & (t > 0)
        & (base > 0)
    )
    d, t, base = (ops.to_host(a[ok]) for a in (d, t, base))
    out["n"] = int(len(d))
    if len(d) < MIN_MISMATCH_STARS:
        return out
    eta2 = ops._chi2_median_scale(d**2, base, t**2)
    out.update(
        eta=float(math.sqrt(eta2)),
        median_ratio=float(np.median(d / t)),
        chi2_median_without=float(np.median(d**2 / base)),
        reference_flux_median=float(np.median(t)),
    )
    return out


def sensitivity(ctx: BudgetContext, x, y, cx, cy, single):
    """Single-source sensitivity ``lambda = (D + T) / D`` of a lobe to the
    misalignment, with the forced fluxes ``D`` of the residual and ``T`` of
    the kernel-matched reference at the lobe plus the modeled offset; 1 for
    a resolved dipole. A single source's target image sits at the lobe plus
    ``(lambda - 1)`` times the offset, so the fluxes are re-measured there
    (three passes). ``|lambda|`` is capped at ``SENSITIVITY_CAP``. Returns
    ``(lambda, raw lambda, T, flux x, flux y)`` at the final position.
    """
    xp = ops.array_module(ctx.residual)
    half = ctx.window // 2
    shape = ctx.residual.shape
    field = xp.asarray(
        misalignment_field(ctx.misalignment, ops.to_host(x), ops.to_host(y))
    )
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
    return lam, raw, t, xf, yf


def budget(
    ctx: BudgetContext,
    *,
    cand_x,
    cand_y,
    offsets: dict[str, tuple[Any, Any]],
    err_fit: dict[str, Any],
    lobe_flux,
    single,
    photometry_xy=None,
) -> BudgetResult:
    """Every budget term and the full error bar per candidate.

    ``cand_x``, ``cand_y`` are the candidate pixels; ``offsets`` maps
    ``"center"`` and ``"positive"`` to ``(dx, dy)`` arrays relative to them;
    ``err_fit`` maps each quantity in :data:`QUANTITIES` to its fit term
    (``psf_flux`` absent or None when forced photometry did not run);
    ``lobe_flux`` is the fitted lobe flux and ``single`` marks candidates
    with one supported lobe. ``photometry_xy`` (N, 2) is the offset forced
    photometry measured at, None without it.

    Returns the ``<q>_err_{corr,kfit,kmatch,mis}``, ``<q>_err`` and
    ``uq_valid`` columns. ``<q>_err`` is NaN where the quantity's window
    holds an invalid pixel; ``uq_valid`` is true where every computed
    quantity's error bar is defined.
    """
    from cuphoton.xpois import psf_weights

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
    var = {q: {t: zeros for t in TERMS} for q in quantities}
    factor = {q: ones for q in quantities}
    rotated = xp.zeros(n, dtype=bool)
    diag: dict[str, Any] = {
        "window_px": ctx.window,
        "kernel_fit_form": None
        if ctx.kernel_covariance is None
        else ctx.kernel_covariance.form,
        "kernel_mismatch": ctx.kernel_mismatch,
        "sensitivity_capped": 0,
    }

    positions = {}
    for name in POSITIONS:
        dx, dy = offsets[name]
        px = cx + xp.asarray(dx, dtype=xp.float64)
        py = cy + xp.asarray(dy, dtype=xp.float64)
        safe = _safe_positions(xp, px, py, cx, cy, shape)
        positions[name] = (px, py, *safe)

    lam = ones
    if photometry_xy is not None:
        pxy = xp.asarray(photometry_xy, dtype=xp.float64)
        fx, fy = _safe_positions(
            xp, cx + pxy[:, 0], cy + pxy[:, 1], cx, cy, shape
        )
    else:
        fx, fy = positions["positive"][2:]
    if ctx.window is not None:
        lam, raw, t_ref, fx, fy = sensitivity(ctx, fx, fy, cx, cy, single)
        diag["sensitivity_capped"] = int(
            (xp.abs(raw) > SENSITIVITY_CAP).sum()
        )

    # positions: PSF-derivative weights, scaled to pixels by the lobe flux
    if ctx.window is not None and (
        "corr" in ctx.terms or "kfit" in ctx.terms
    ):
        half = ctx.window // 2
        dg = {
            a: xp.asarray(
                psf_weights(ctx.psf_sigma, ctx.window, derivative=a)
            )
            for a in ("x", "y")
        }
        for name in POSITIONS:
            sx, sy = positions[name][2:]
            plane = ops._patches(ctx.weights_plane, sx, sy, half)
            f_axis = []
            for a in ("x", "y"):
                scale = _derivative_scale(xp, plane, dg[a], flux)
                f, kfit, rot = _weighted_sum_terms(
                    ctx, dg[a], sx, sy, scale=scale
                )
                var[f"{name}_{a}"]["kfit"] = kfit
                f_axis.append(f)
                rotated |= rot
            # one window per position: both axes share its validity
            both = f_axis[0] + f_axis[1]
            for a, f in zip(("x", "y"), f_axis, strict=True):
                factor[f"{name}_{a}"] = xp.where(xp.isfinite(both), f, xp.nan)
    # the flux: forced photometry's weights
    if "psf_flux" in quantities and ctx.window is not None:
        psf = xp.asarray(psf_weights(ctx.psf_sigma, ctx.window))
        half = ctx.window // 2
        plane = ops._patches(ctx.weights_plane, fx, fy, half)
        scale = _derivative_scale(xp, plane, psf, ones)
        f, kfit, rot = _weighted_sum_terms(ctx, psf, fx, fy, scale=scale)
        factor["psf_flux"] = f
        var["psf_flux"]["kfit"] = kfit
        rotated |= rot
        if "kmatch" in ctx.terms:
            eta = float(ctx.kernel_mismatch.get("eta", 0.0))
            var["psf_flux"]["kmatch"] = (
                eta * xp.where(xp.isfinite(t_ref), xp.abs(t_ref), 0.0)
            ) ** 2
    # misalignment
    if "mis" in ctx.terms:
        for name, gain in (("center", 0.25 * ones), ("positive", lam**2)):
            px, py = positions[name][:2]
            mv = xp.asarray(
                misalignment_variance(
                    ctx.misalignment, ops.to_host(px), ops.to_host(py)
                )
            )
            for j, a in enumerate(("x", "y")):
                var[f"{name}_{a}"]["mis"] = gain * mv[:, j]

    columns: dict[str, Any] = {}
    valid = xp.ones(n, dtype=bool)
    for q in quantities:
        fit = xp.asarray(err_fit[q], dtype=xp.float64)
        f = factor[q]
        f = xp.where(f < 1.0, 1.0, f)
        valid &= xp.isfinite(f)
        columns[f"{q}_err_corr"] = fit * xp.sqrt(f - 1.0)
        for t in ("kfit", "kmatch", "mis"):
            columns[f"{q}_err_{t}"] = xp.sqrt(var[q][t])
        columns[f"{q}_err"] = xp.sqrt(
            fit**2 * f + var[q]["kfit"] + var[q]["kmatch"] + var[q]["mis"]
        )
    columns["uq_valid"] = valid
    diag["invalid_window"] = int((~valid & ~rotated).sum())
    diag["rotated_grid"] = int(rotated.sum())
    return BudgetResult(columns, ctx.terms, diag)


def fit_only(err_fit: dict[str, Any]) -> BudgetResult:
    """The columns of :func:`budget` when the budget is disabled: every term
    zero and ``<q>_err`` equal to the fit term.
    """
    columns: dict[str, Any] = {}
    n = None
    for q in QUANTITIES:
        fit = err_fit.get(q)
        if fit is None:
            continue
        xp = ops.array_module(fit)
        n = int(fit.shape[0])
        for t in TERMS:
            columns[f"{q}_err_{t}"] = xp.zeros(n, dtype=xp.float64)
        columns[f"{q}_err"] = xp.asarray(fit, dtype=xp.float64)
    columns["uq_valid"] = np.ones(0 if n is None else n, dtype=bool)
    return BudgetResult(columns, (), {"disabled": True})
