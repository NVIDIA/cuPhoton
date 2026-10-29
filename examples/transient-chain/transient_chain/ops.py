# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Stage operations for the GPU-resident chain, the same for every survey.

Every function takes arrays from one array module (NumPy on the host for
tests,
CuPy on the device in production) and returns arrays from the same module, or
small host scalars where a decision needs one. Nothing here touches a file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def array_module(a):
    try:
        import cupy

        return cupy.get_array_module(a)
    except ImportError:
        return np


def ndimage_for(xp):
    if xp is np:
        from scipy import ndimage

        return ndimage
    from cupyx.scipy import ndimage

    return ndimage


def to_host(a) -> np.ndarray:
    return np.asarray(a.get() if hasattr(a, "get") else a)


def robust_sigma(values) -> float:
    """1.4826 times the median absolute deviation of the finite values (one
    scalar sync).
    """
    xp = array_module(values)
    a = values.ravel()
    a = a[xp.isfinite(a)]
    if a.size == 0:
        return float("nan")
    return float(1.4826 * xp.median(xp.abs(a - xp.median(a))))


def xcorr_offset(a, b) -> tuple[float, float]:
    """Cross-correlation peak of two equal windows, sub-pixel by parabola:
    (dx, dy) pixels.
    """
    xp = array_module(a)
    a = xp.where(xp.isfinite(a), a, 0.0)
    a = a - xp.median(a)
    b = xp.where(xp.isfinite(b), b, 0.0)
    b = b - xp.median(b)
    cc = xp.fft.fftshift(
        xp.fft.irfft2(xp.fft.rfft2(a) * xp.conj(xp.fft.rfft2(b)), s=a.shape)
    )
    cc = to_host(cc)
    iy, ix = np.unravel_index(np.argmax(cc), cc.shape)
    cy, cx = a.shape[0] // 2, a.shape[1] // 2

    def sub(m, z, p):
        d = m - 2 * z + p
        return 0.0 if d == 0 else 0.5 * (m - p) / d

    dy = (
        sub(cc[iy - 1, ix], cc[iy, ix], cc[iy + 1, ix])
        if 0 < iy < cc.shape[0] - 1
        else 0.0
    )
    dx = (
        sub(cc[iy, ix - 1], cc[iy, ix], cc[iy, ix + 1])
        if 0 < ix < cc.shape[1] - 1
        else 0.0
    )
    return float(ix - cx + dx), float(iy - cy + dy)


@dataclass(frozen=True)
class SmoothedImage:
    """One image prepared for the star finder, arrays of the image's array
    module: ``finite`` mask and its count, ``median`` and robust ``sigma`` of
    the finite pixels (scalars of the array module), ``work`` the image minus
    the median with non-finite pixels zero, ``smooth`` its Gaussian-smoothed
    copy with FWHM ``fwhm``.

    ``find_stars`` and ``star_measurements`` take one through ``smoothed`` so
    an image is smoothed once per item, not once per call.
    """

    fwhm: float
    finite: Any
    n_finite: int
    median: Any
    sigma: Any
    work: Any
    smooth: Any


def smooth_image(img, fwhm: float) -> SmoothedImage:
    """Median-subtract ``img`` and smooth it with a Gaussian of FWHM ``fwhm``
    (one filter, one count sync).
    """
    xp = array_module(img)
    finite = xp.isfinite(img)
    n_finite = int(finite.sum())
    if n_finite == 0:
        zero = xp.zeros(img.shape, dtype=xp.float64)
        return SmoothedImage(float(fwhm), finite, 0, 0.0, 0.0, zero, zero)
    vals = img[finite]
    med = xp.median(vals)
    sigma = 1.4826 * xp.median(xp.abs(vals - med))
    work = xp.where(finite, img - med, 0.0).astype(xp.float64)
    smooth = ndimage_for(xp).gaussian_filter(
        work, sigma=fwhm / 2.3548, mode="nearest"
    )
    return SmoothedImage(
        float(fwhm), finite, n_finite, med, sigma, work, smooth
    )


def _smoothed(img, fwhm: float, smoothed: SmoothedImage | None):
    if smoothed is None:
        return smooth_image(img, fwhm)
    if smoothed.fwhm != float(fwhm):
        raise ValueError(
            f"smoothed image has fwhm {smoothed.fwhm}, the call asks {fwhm}"
        )
    return smoothed


def find_stars(
    img,
    *,
    fwhm: float = 4.0,
    nsigma: float = 20.0,
    max_n: int = 600,
    box: int = 15,
    iterations: int = 4,
    smoothed: SmoothedImage | None = None,
) -> np.ndarray:
    """Bright, isolated peaks of a Gaussian-smoothed image with windowed
    centroids.

    Replaces DAOStarFinder for alignment. Threshold = nsigma times the robust
    sigma of the
    finite pixels above their median; peaks are local maxima of the smoothed
    image over a
    (2*box//2+1) window. Each centroid is refined by ``iterations`` passes of
    a Gaussian-windowed
    flux-weighted mean (window sigma = fwhm, the SExtractor XWIN estimator) on
    the
    background-subtracted image inside a fixed box around the peak
    (``iterations=0`` returns the peak pixels). Returns
    (N, 2) host x, y
    sorted by box flux, brightest first (when more than ``2*max_n`` peaks
    exist, the pre-cut uses
    the same box flux, so the final list is the true top ``max_n``). Only the
    star list crosses
    to the host. ``smoothed`` is ``smooth_image(img, fwhm)`` when the caller
    already has it.
    """
    xp = array_module(img)
    nd = ndimage_for(xp)
    sm = _smoothed(img, fwhm, smoothed)
    if sm.n_finite < 1000:
        return np.empty((0, 2))
    finite, work, smooth = sm.finite, sm.work, sm.smooth
    thr = nsigma * sm.sigma
    half = box // 2
    peaks = (
        (
            smooth
            == nd.maximum_filter(smooth, size=2 * half + 1, mode="nearest")
        )
        & (smooth > thr)
        & finite
    )
    ys, xs = xp.nonzero(peaks)
    if ys.size == 0:
        return np.empty((0, 2))
    ok = (
        (ys >= half)
        & (ys < img.shape[0] - half)
        & (xs >= half)
        & (xs < img.shape[1] - half)
    )
    ys, xs = ys[ok], xs[ok]
    off = xp.arange(-half, half + 1, dtype=xp.float64)
    wy = ys[:, None, None] + off[None, :, None]
    wx = xs[:, None, None] + off[None, None, :]
    w = xp.clip(work[wy.astype(xp.int64), wx.astype(xp.int64)], 0.0, None)
    if (
        ys.size > max_n * 2
    ):  # pre-cut on the ranking quantity itself (box flux), never on the peak
        order = xp.argsort(-w.sum(axis=(1, 2)))[: max_n * 2]
        ys, xs, wy, wx, w = (
            ys[order],
            xs[order],
            wy[order],
            wx[order],
            w[order],
        )
    cy, cx = ys.astype(xp.float64), xs.astype(xp.float64)
    sw2 = 2.0 * fwhm * fwhm
    for _ in range(iterations):
        g = w * xp.exp(
            -((wy - cy[:, None, None]) ** 2 + (wx - cx[:, None, None]) ** 2)
            / sw2
        )
        flux = g.sum(axis=(1, 2))
        good = flux > 0
        cy = xp.where(
            good, (g * wy).sum(axis=(1, 2)) / xp.where(good, flux, 1.0), cy
        )
        cx = xp.where(
            good, (g * wx).sum(axis=(1, 2)) / xp.where(good, flux, 1.0), cx
        )
    flux = w.sum(axis=(1, 2))
    keep = (flux > 0) & (xp.abs(cy - ys) < half) & (xp.abs(cx - xs) < half)
    order = xp.argsort(-flux[keep])[:max_n]
    return np.column_stack(
        [to_host(cx[keep][order]), to_host(cy[keep][order])]
    ).astype(float)


def find_stars_daofind(
    img, *, fwhm: float = 4.0, nsigma: float = 20.0, max_n: int = 600
) -> np.ndarray:
    """The reference chain's DAOStarFinder path on a host copy of the image
    (one device-to-host copy, no file).
    """
    from astropy.stats import sigma_clipped_stats
    from photutils.detection import DAOStarFinder

    a = to_host(img).astype(np.float64)
    finite = np.isfinite(a)
    if finite.sum() < 1000:
        return np.empty((0, 2))
    _, med, std = sigma_clipped_stats(a[finite], sigma=3.0)
    tbl = DAOStarFinder(
        fwhm=fwhm, threshold=nsigma * std, exclude_border=True
    )(np.where(finite, a - med, 0.0))
    if tbl is None or len(tbl) == 0:
        return np.empty((0, 2))
    tbl.sort("flux", reverse=True)
    xc, yc = (
        ("x_centroid", "y_centroid")
        if "x_centroid" in tbl.colnames
        else ("xcentroid", "ycentroid")
    )
    return np.column_stack([tbl[xc], tbl[yc]])[:max_n].astype(float)


def match_stars(
    a: np.ndarray, b: np.ndarray, r: float
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree

    if len(a) == 0 or len(b) == 0:
        return np.empty((0, 2)), np.empty((0, 2))
    d, j = cKDTree(b).query(a, distance_upper_bound=r)
    candidates = np.flatnonzero(np.isfinite(d))
    order = candidates[np.argsort(d[candidates], kind="stable")]
    _, first = np.unique(j[order], return_index=True)
    keep = np.sort(order[first])
    return a[keep], b[j[keep]]


def fit_affine(src: np.ndarray, dst: np.ndarray):
    """Least squares dst = A @ src + b. Returns A (2x2), b (2,), residual rms
    in pixels.
    """
    G = np.column_stack([src, np.ones(len(src))])
    cx, *_ = np.linalg.lstsq(G, dst[:, 0], rcond=None)
    cy, *_ = np.linalg.lstsq(G, dst[:, 1], rcond=None)
    A, b = (
        np.array([[cx[0], cx[1]], [cy[0], cy[1]]]),
        np.array([cx[2], cy[2]]),
    )
    return (
        A,
        b,
        float(np.sqrt(np.mean(np.sum((src @ A.T + b - dst) ** 2, axis=1)))),
    )


def hotspots(residual, *, robust_sigma: float, threshold_sigma: float = 5.0):
    """Connected components of |residual| / robust_sigma >= threshold, as in
    cuphoton.xpois.review.identify_residual_hotspots.

    Returns a dict of host arrays sorted by (peak_abs_sigma, pixel_count)
    descending, ties
    in label order: pixel_count, peak_y, peak_x, centroid_y, centroid_x,
    peak_abs_sigma,
    peak_residual, mean_residual. Empty arrays when nothing crosses the
    threshold.
    """
    xp = array_module(residual)
    nd = ndimage_for(xp)
    if robust_sigma <= 0.0:
        raise ValueError("robust_sigma must be positive")
    residual = residual.astype(xp.float64, copy=False)
    sigma = xp.abs(residual) / robust_sigma
    mask = xp.isfinite(residual) & (sigma >= threshold_sigma)
    empty = {
        k: np.empty(0)
        for k in (
            "pixel_count",
            "peak_y",
            "peak_x",
            "centroid_y",
            "centroid_x",
            "peak_abs_sigma",
            "peak_residual",
            "mean_residual",
        )
    }
    if not bool(mask.any()):
        return empty
    labels, count = nd.label(mask)
    count = int(count)
    flat = xp.flatnonzero(mask)
    lab, s, r = (
        labels.ravel()[flat],
        sigma.ravel()[flat],
        residual.ravel()[flat],
    )
    ys, xs = xp.divmod(flat, residual.shape[1])
    ids = xp.arange(1, count + 1)
    n = xp.bincount(lab, minlength=count + 1)[1:]
    cy = xp.bincount(lab, weights=ys, minlength=count + 1)[1:] / n
    cx = xp.bincount(lab, weights=xs, minlength=count + 1)[1:] / n
    mean_r = xp.bincount(lab, weights=r, minlength=count + 1)[1:] / n
    order = xp.lexsort(xp.stack((flat, -s, lab)))
    pk = order[xp.searchsorted(lab[order], ids)]
    peak_s, peak_r = s[pk], r[pk]
    final = xp.lexsort(xp.stack((ids, -n, -peak_s)))
    out = {
        "pixel_count": n,
        "peak_y": ys[pk],
        "peak_x": xs[pk],
        "centroid_y": cy,
        "centroid_x": cx,
        "peak_abs_sigma": peak_s,
        "peak_residual": peak_r,
        "mean_residual": mean_r,
    }
    return {k: to_host(v[final]) for k, v in out.items()}


def compact_source_stamp_mask(
    image,
    *,
    variance=None,
    valid_mask=None,
    stamp_size: int = 31,
    max_stamps: int = 5,
    peak_percentile: float = 99.5,
    background_filter_size: int = 15,
    peak_filter_size: int = 9,
    min_separation: int | None = None,
    compact_core_size: int = 3,
):
    """Port of cuphoton.xpois.build_compact_source_stamp_mask that stays on
    the array module.

    Same selection rule: median-filter background, local maxima of the
    high-pass image above a
    percentile, per-peak significance and compactness ranking, greedy
    separation. Returns the
    boolean fit mask and a small host diagnostics dict.
    """
    xp = array_module(image)
    nd = ndimage_for(xp)
    a = image.astype(xp.float64, copy=False)
    base_valid = xp.isfinite(a)
    if variance is not None:
        variance = variance.astype(xp.float64, copy=False)
        base_valid &= xp.isfinite(variance) & (variance > 0)
    if valid_mask is not None:
        base_valid &= valid_mask
    if not bool(base_valid.any()):
        raise ValueError(
            "no finite pixels remain for compact-source selection"
        )
    fill = xp.median(a[base_valid])
    background = nd.median_filter(
        xp.where(base_valid, a, fill),
        size=background_filter_size,
        mode="nearest",
    )
    highpass = xp.where(base_valid, a - background, -xp.inf)
    threshold = xp.percentile(highpass[base_valid], peak_percentile)
    maxima = nd.maximum_filter(
        highpass, size=peak_filter_size, mode="nearest"
    )
    py, px = xp.nonzero(
        (highpass == maxima) & base_valid & (highpass > threshold)
    )
    if py.size == 0:
        raise ValueError(
            "no compact-source peaks found for the requested mask"
        )
    half = stamp_size // 2
    inside = (
        (py >= half)
        & (py + half + 1 <= a.shape[0])
        & (px >= half)
        & (px + half + 1 <= a.shape[1])
    )
    py, px = py[inside], px[inside]
    off = xp.arange(-half, half + 1)
    wy = py[:, None, None] + off[None, :, None]
    wx = px[:, None, None] + off[None, None, :]
    all_valid = base_valid[wy, wx].all(axis=(1, 2))
    py, px, wy, wx = (
        py[all_valid],
        px[all_valid],
        wy[all_valid],
        wx[all_valid],
    )
    stamps = a[wy, wx]
    local_bg = xp.median(stamps.reshape(len(py), -1), axis=1)
    weights = xp.clip(stamps - local_bg[:, None, None], 0.0, None)
    total = weights.sum(axis=(1, 2))
    ch = compact_core_size // 2
    core = weights[
        :, half - ch : half + ch + 1, half - ch : half + ch + 1
    ].sum(axis=(1, 2))
    yy, xx = xp.meshgrid(off, off, indexing="ij")
    radial = (weights * (xx * xx + yy * yy)[None]).sum(axis=(1, 2))
    signal = stamps[:, half, half] - local_bg
    score = signal if variance is None else signal / xp.sqrt(variance[py, px])
    keep = total > 0
    cand = np.rec.fromarrays(
        [
            to_host(score[keep]),
            to_host(core[keep] / total[keep]),
            -to_host(radial[keep] / total[keep]),
            to_host(py[keep]).astype(int),
            to_host(px[keep]).astype(int),
        ],
        names="score,compact,negradial,y,x",
    )
    if len(cand) == 0:
        raise ValueError(
            "no valid compact-source stamps remain after finite-mask "
            "filtering"
        )
    order = np.lexsort(
        (-cand.x, -cand.y, -cand.negradial, -cand.compact, -cand.score)
    )
    cand = cand[order]
    min_d2 = (
        min_separation if min_separation is not None else max(1, half)
    ) ** 2
    selected = []
    for c in cand:
        if any(
            (c.y - s.y) ** 2 + (c.x - s.x) ** 2 < min_d2 for s in selected
        ):
            continue
        selected.append(c)
        if len(selected) >= max_stamps:
            break
    mask = xp.zeros(a.shape, dtype=bool)
    for s in selected:
        mask[s.y - half : s.y + half + 1, s.x - half : s.x + half + 1] = True
    return mask, {
        "centers": [(int(s.y), int(s.x)) for s in selected],
        "scores": [float(s.score) for s in selected],
        "threshold": float(threshold),
        "peaks_considered": int(len(cand)),
    }


def extract_stamps(search, residual, cx, cy, size: int, variance=None):
    """Cut (K, 3, size, size) float32 stamps: search, template = search minus
    difference, difference.

    Non-finite pixels of the search and difference planes become zero before
    the template is
    formed, as ``cuphoton.xscan.lsstcomcam._finalize_stamp`` does; the planes
    are rounded to
    float32 first, matching a builder that read float32 FITS planes. Also
    returns the float64
    difference stamps for xFit, rounded through float32 the same way, the
    boolean mask of the pixels xFit may fit (finite residual, and finite
    positive ``variance`` when given) and the float64 variance stamps (None
    without ``variance``). Zeroed pixels stay out of the fit through that
    mask. ``size`` must be odd.
    """
    if size % 2 == 0:
        raise ValueError(f"stamp size must be odd, got {size}")
    xp = array_module(search)
    half = size // 2
    off = xp.arange(-half, half + 1)
    cy = xp.asarray(cy)[:, None, None] + off[None, :, None]
    cx = xp.asarray(cx)[:, None, None] + off[None, None, :]
    # Same order of operations as cuphoton.xscan.lsstcomcam's builder: zero
    # the non-finite pixels of the
    # search and difference planes first, then form the template from the
    # zeroed planes, so a NaN residual
    # pixel leaves the search value in the template instead of a zero.
    s = xp.nan_to_num(
        search[cy, cx].astype(xp.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    raw = residual[cy, cx].astype(xp.float32)
    valid = xp.isfinite(raw)
    d = xp.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    stamps = xp.empty((len(cy), 3, size, size), dtype=xp.float32)
    stamps[:, 0] = s
    stamps[:, 1] = s - d
    stamps[:, 2] = d
    difference = d.astype(xp.float64)
    var = None
    if variance is not None:
        var = variance[cy, cx].astype(xp.float64)
        valid &= xp.isfinite(var) & (var > 0)
    return stamps, difference, valid, var


def noise_match_factors(search_stamps, reference_sigma: float):
    """Per-stamp factor making the search stamp's robust sigma equal the
    reference value.

    A flat stamp (zero MAD) gets the batch median factor, or 1.0 when every
    stamp is flat.
    Returns the float32 factors and the number of stamps that took the
    fallback.
    """
    xp = array_module(search_stamps)
    flat = search_stamps.reshape(len(search_stamps), -1).astype(xp.float64)
    med = xp.median(flat, axis=1, keepdims=True)
    sigma = 1.4826 * xp.median(xp.abs(flat - med), axis=1)
    factor = xp.where(
        sigma > 0, reference_sigma / xp.where(sigma > 0, sigma, 1.0), xp.nan
    )
    bad = ~xp.isfinite(factor)
    n_bad = int(bad.sum())
    fallback = float(xp.nanmedian(factor)) if n_bad < factor.size else 1.0
    return xp.where(bad, fallback, factor).astype(xp.float32), n_bad


# --- star measurements, misalignment and PSF width --------------------------


def patches(image, x, y, half):
    """(N, 2 half + 1, 2 half + 1) float64 windows of ``image`` centered on
    the pixels rint(x), rint(y); NaN where a window leaves the image.
    """
    xp = array_module(image)
    ny, nx = image.shape
    cx = xp.rint(xp.asarray(x, dtype=xp.float64)).astype(xp.int64)
    cy = xp.rint(xp.asarray(y, dtype=xp.float64)).astype(xp.int64)
    i = xp.arange(-half, half + 1)
    yy = cy[:, None, None] + i[None, :, None]
    xx = cx[:, None, None] + i[None, None, :]
    inside = (yy >= 0) & (yy < ny) & (xx >= 0) & (xx < nx)
    v = image[xp.clip(yy, 0, ny - 1), xp.clip(xx, 0, nx - 1)]
    return xp.where(inside, v.astype(xp.float64), xp.nan)


def _star_boxes(image, variance, xy, box, fwhm, smoothed=None):
    """Background-subtracted pixels, variances and offsets from the centroid
    in the box ``find_stars`` used: centered on the brightest pixel of the
    smoothed image within the box around the centroid, with the global median
    as background. Pixels off the image or not finite count as zero signal
    and zero variance, as ``find_stars`` counts them. ``smoothed`` is
    ``smooth_image(image, fwhm)`` when the caller already has it.
    """
    xp = array_module(image)
    sm = _smoothed(image, fwhm, smoothed)
    work, smooth = sm.work, sm.smooth
    if variance is None:
        v = xp.zeros(image.shape, dtype=xp.float64)
    else:
        v = xp.where(sm.finite & xp.isfinite(variance), variance, 0.0)
    half = box // 2
    c = xp.asarray(xy, dtype=xp.float64).reshape(-1, 2)
    s = xp.nan_to_num(
        patches(smooth, c[:, 0], c[:, 1], half), nan=-xp.inf
    ).reshape(len(c), -1)
    k = xp.argmax(s, axis=1)
    width = 2 * half + 1
    px = xp.rint(c[:, 0]).astype(xp.int64) + k % width - half
    py = xp.rint(c[:, 1]).astype(xp.int64) + k // width - half
    off = xp.arange(-half, half + 1)
    wy = py[:, None, None] + off[None, :, None]
    wx = px[:, None, None] + off[None, None, :]
    return (
        xp.nan_to_num(patches(work, px, py, half)),
        xp.nan_to_num(patches(v, px, py, half)),
        wx - c[:, 0, None, None],
        wy - c[:, 1, None, None],
    )


def windowed_centroid_error(d, v, ux, uy, window_sigma: float):
    """Standard error per axis, (N, 2), of the windowed centroid
    ``find_stars`` returns, by linearizing its fixed point.

    ``find_stars`` iterates c = sum(g x) / sum(g) with g = W(x - c) max(d, 0)
    and W a Gaussian of width ``window_sigma``. At convergence
    F(c) = sum W max(d, 0) (x - c) = 0, so a pixel-noise perturbation moves c
    by dF / (-dF/dc):
    sigma_c^2 = sum W^2 u^2 V 1[d > 0] / (sum W max(d, 0) (1 - u^2 / s_w^2))^2
    per axis, with ``d`` the background-subtracted pixels, ``v`` their
    variances and ``ux``, ``uy`` their offsets from the centroid.
    """
    xp = array_module(d)
    w = xp.clip(d, 0.0, None)
    W = xp.exp(-(ux**2 + uy**2) / (2.0 * window_sigma**2))
    on = d > 0
    out = xp.empty((d.shape[0], 2), dtype=xp.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        for a, u in enumerate((ux, uy)):
            num = (W**2 * u**2 * v * on).sum(axis=(1, 2))
            den = (W * w * (1.0 - u**2 / window_sigma**2)).sum(axis=(1, 2))
            out[:, a] = xp.sqrt(num) / xp.abs(den)  # NaN without flux
    return out


def second_moment_width(
    d, ux, uy, *, start: float = 2.0, iterations: int = 6
):
    """Adaptive Gaussian-weighted second-moment width per star, (N,), averaged
    over the axes and corrected for the weight.

    With weight width s_w, a Gaussian of width s has weighted moment
    m = 1 / (1/s^2 + 1/s_w^2), so s^2 = 1 / (1/m - 1/s_w^2), and the weight
    follows s. The width is the pixel-level one (the PSF convolved with the
    pixel), the width a point-sampled Gaussian PSF model needs; the optical
    width is sqrt(s^2 - 1/12). NaN where the iteration fails (no positive
    flux, or a moment wider than the weight): ``psf_width`` drops those.
    """
    xp = array_module(d)
    s = xp.full(d.shape[0], float(start))
    r2 = 0.5 * (ux**2 + uy**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        for _ in range(iterations):
            W = xp.exp(-(ux**2 + uy**2) / (2.0 * s[:, None, None] ** 2)) * d
            m = (W * r2).sum(axis=(1, 2)) / W.sum(axis=(1, 2))
            inv = 1.0 / m - 1.0 / s**2
            ok = inv > 0
            s = xp.where(
                ok,
                xp.clip(1.0 / xp.sqrt(xp.where(ok, inv, 1.0)), 0.3, 20.0),
                xp.nan,
            )
    return s


@dataclass(frozen=True)
class StarMeasurements:
    """Per-star measurements at the centroids ``find_stars`` returned, arrays
    of the image's array module.

    ``centroid_error`` (N, 2) is the standard error per axis of the windowed
    centroid, propagated from the pixel variance; ``flux`` (N,) the positive
    box flux; ``width`` (N,) the pixel-level second-moment width; ``snr`` (N,)
    the PSF-weighted signal-to-noise (NaN without a variance plane).
    """

    centroid_error: Any
    flux: Any
    width: Any
    snr: Any


def star_measurements(
    image,
    variance,
    xy,
    *,
    box: int,
    psf_sigma: float | None = None,
    fwhm: float = 4.0,
    smoothed: SmoothedImage | None = None,
) -> StarMeasurements:
    """Centroid standard error, box flux, width and PSF-weighted
    signal-to-noise of the stars at ``xy`` (N, 2).

    ``box`` and ``fwhm`` are the ones ``find_stars`` used: the windowed
    estimator's window is a Gaussian of width ``fwhm``. The signal-to-noise
    is a matched filter with a circular Gaussian of width ``psf_sigma``, the
    star's own width when None. ``variance`` None counts every pixel as
    noiseless: the centroid errors are zero and the signal-to-noise NaN.
    ``smoothed`` is ``smooth_image(image, fwhm)`` when the caller already has
    it.
    """
    d, v, ux, uy = _star_boxes(
        image, variance, xy, int(box), float(fwhm), smoothed
    )
    xp = array_module(d)
    centroid_error = windowed_centroid_error(d, v, ux, uy, float(fwhm))
    width = second_moment_width(d, ux, uy)
    s = width if psf_sigma is None else xp.full(d.shape[0], float(psf_sigma))
    s2 = s[:, None, None] ** 2
    psf = xp.exp(-0.5 * (ux**2 + uy**2) / s2) / (2.0 * np.pi * s2)
    w = xp.where(v > 0, 1.0 / xp.where(v > 0, v, 1.0), 0.0)
    norm = (w * psf * psf).sum(axis=(1, 2))
    snr = xp.where(
        norm > 0,
        (w * psf * d).sum(axis=(1, 2))
        / xp.sqrt(xp.where(norm > 0, norm, 1.0)),
        xp.nan,
    )
    return StarMeasurements(
        centroid_error, xp.clip(d, 0.0, None).sum(axis=(1, 2)), width, snr
    )


NEIGHBOR_PEAKS = dict(nsigma=5.0, max_n=10**6, box=9, iterations=0)
"""``find_stars`` settings of the low-threshold peak list ``neighbor_free``
compares against: every peak above 5 robust sigma, at its pixel."""


def neighbor_free(peaks, xy, radius: float) -> np.ndarray:
    """True where no other peak of ``peaks`` (M, 2), the host list
    ``find_stars(image, fwhm=fwhm, **NEIGHBOR_PEAKS)`` returns, lies within
    ``radius`` pixels of the host positions ``xy`` (N, 2): a neighbor's light
    moves a windowed centroid, and by different amounts in two images with
    different PSFs. The one peak allowed within ``radius`` is the star's own.
    """
    from scipy.spatial import cKDTree

    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    peaks = np.asarray(peaks, dtype=np.float64).reshape(-1, 2)
    if len(peaks) == 0 or len(xy) == 0:
        return np.ones(len(xy), bool)
    return (
        cKDTree(peaks).query_ball_point(xy, radius, return_length=True) <= 1
    )


def _affine_design(xy):
    return np.column_stack([xy, np.ones(len(xy))])


def _quad_design(xy, shape):
    center = ((shape[1] - 1) / 2.0, (shape[0] - 1) / 2.0)
    half = max(shape) / 2.0
    X, Y = (xy[:, 0] - center[0]) / half, (xy[:, 1] - center[1]) / half
    return np.column_stack([X * X, X * Y, Y * Y])


def _quadratic_form(g, cov):
    return np.stack(
        [np.einsum("ni,ij,nj->n", g, cov[a], g) for a in (0, 1)], axis=1
    )


@dataclass(frozen=True)
class Misalignment:
    """Residual misalignment after the affine re-alignment, per axis, anywhere
    on the target grid; host arrays.

    ``variance(x, y)`` = affine-fit variance at (x, y) + the squared
    modeled residual displacement (affine and quadratic together) + the
    excess variance the field leaves. This affine is measured after the
    template rebuild and is not applied to the catalog positions.
    The field's estimation variance
    is not added: the estimate's square already exceeds the true field's
    square by it on average. With fewer than ``min_poly`` stars there is no
    field (``mode == "rms"``) and the excess is one constant per axis.
    """

    matrix: np.ndarray  # (2, 2) affine A of dst = A @ src + b
    offset: np.ndarray  # (2,) affine b
    affine_covariance: (
        np.ndarray
    )  # (2, 3, 3) per axis, parameters (a_x, a_y, b)
    mode: str  # "polynomial" or "rms"
    excess_variance: np.ndarray  # (2,) beyond the model, pixels squared
    used: np.ndarray  # (N_input,) pairs the fit used
    residual: np.ndarray  # (N_used, 2) dst minus affine prediction, pixels
    shape: tuple[int, int]  # target grid
    field_coefficients: np.ndarray | None = None  # (2, 3) per axis
    field_covariance: np.ndarray | None = None  # (2, 3, 3)
    field_projection: np.ndarray | None = (
        None  # (3, 3) affine part of the field
    )

    @property
    def residual_rms(self) -> float:
        """Rms per axis of the residuals of the stars the fit used."""
        return float(np.sqrt(np.mean(np.sum(self.residual**2, axis=1)) / 2.0))

    def affine_variance(self, x, y) -> np.ndarray:
        """Variance per axis, (N, 2), of the affine prediction at (x, y)."""
        g = _affine_design(np.column_stack([np.ravel(x), np.ravel(y)]))
        return _quadratic_form(g, self.affine_covariance)

    def excess(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        """Modeled misalignment field (N, 2) and its estimation variance
        (N, 2); zero field in ``rms`` mode.
        """
        xy = np.column_stack([np.ravel(x), np.ravel(y)])
        if self.mode != "polynomial":
            return np.zeros((len(xy), 2)), np.zeros((len(xy), 2))
        q = _quad_design(xy, self.shape) - _affine_design(xy) @ (
            self.field_projection
        )
        return q @ self.field_coefficients.T, _quadratic_form(
            q, self.field_covariance
        )

    def variance(self, x, y) -> np.ndarray:
        """Misalignment variance per axis, pixels squared, (N, 2) at the
        positions (x, y) on the target grid.
        """
        positions = np.column_stack([np.ravel(x), np.ravel(y)])
        displacement = (
            positions @ self.matrix.T
            + self.offset
            - positions
            + self.excess(x, y)[0]
        )
        return (
            self.affine_variance(x, y)
            + displacement**2
            + self.excess_variance[None, :]
        )


def _weights(var):
    w = np.where(var > 0, 1.0 / np.where(var > 0, var, 1.0), 0.0)
    return w if w.any() else np.ones_like(w)


def _wls(G, y, w):
    Gw = G * w[:, None]
    cov = np.linalg.inv(G.T @ Gw)
    return cov @ (Gw.T @ y), cov


def _misalignment_model(
    src_xy, dst_xy, noise_var, used, shape, min_poly, iterations=3
) -> Misalignment:
    """One fit on the star set ``used``: ``fit_affine`` (unweighted least
    squares), its covariance by the sandwich formula with the centroid noise,
    then the excess as a weighted quadratic field plus a constant, or as the
    constant alone. The constant is the excess over the centroid noise with
    leverage-corrected degrees of freedom,
    (sum r^2 - sum V (1 - h)) / (n - p), refitted a fixed ``iterations``
    times with the field.
    """
    s, t, nv = src_xy[used], dst_xy[used], noise_var[used]
    A, b, _ = fit_affine(s, t)
    e = t - (s @ A.T + b)
    n = len(e)
    G = _affine_design(s)
    Ginv = np.linalg.pinv(G)  # (3, N): least-squares coefficients
    # per-star scatter beyond the model moves the fit like noise does

    def cov(t2):
        return np.stack(
            [Ginv @ ((nv[:, a] + t2[a])[:, None] * Ginv.T) for a in (0, 1)]
        )

    if n >= min_poly:
        proj = Ginv @ _quad_design(s, shape)
        Q = _quad_design(s, shape) - G @ proj  # what the affine leaves
        G6 = np.column_stack([G, Q])
        lev6 = np.einsum("ij,ji->i", G6, np.linalg.pinv(G6))
        beta, cov_b, tau2 = np.zeros((2, 3)), np.zeros((2, 3, 3)), np.zeros(2)
        for a in (0, 1):
            t2 = 0.0
            for _ in range(iterations):
                bb, cb = _wls(Q, e[:, a], _weights(nv[:, a] + t2))
                r = e[:, a] - Q @ bb
                t2 = max(
                    0.0,
                    float(
                        (np.sum(r**2) - np.sum(nv[:, a] * (1.0 - lev6)))
                        / (n - 6)
                    ),
                )
            beta[a], cov_b[a], tau2[a] = bb, cb, t2
        return Misalignment(
            A,
            b,
            cov(tau2),
            "polynomial",
            tau2,
            used,
            e,
            tuple(shape),
            beta,
            cov_b,
            proj,
        )
    lev = np.einsum("ij,ji->i", G, Ginv)
    tau2 = np.array(
        [
            max(
                0.0,
                float(
                    (np.sum(e[:, a] ** 2) - np.sum(nv[:, a] * (1.0 - lev)))
                    / max(n - 3, 1)
                ),
            )
            for a in (0, 1)
        ]
    )
    return Misalignment(A, b, cov(tau2), "rms", tau2, used, e, tuple(shape))


def chi2_median_scale(num, base, scale) -> float:
    """t >= 0 with median(num / (base + t scale)) at the chi-square(1) median,
    0 when the median is already at or below it: the extra variance a sample
    of squared deviations asks for, which a few outliers do not set.
    """
    from scipy.optimize import brentq
    from scipy.stats import chi2

    target = float(chi2.ppf(0.5, 1))
    num, base, scale = (
        np.asarray(a, dtype=np.float64) for a in (num, base, scale)
    )

    def med(t):
        return float(np.median(num / (base + t * scale))) - target

    if med(0.0) <= 0.0:
        return 0.0
    return float(brentq(med, 0.0, float(np.max(num / scale)) / target))


def fit_misalignment(
    src_xy,
    dst_xy,
    noise_var,
    *,
    shape,
    clip: float | None = 5.0,
    min_poly: int = 30,
    max_rounds: int = 5,
) -> Misalignment:
    """Model the misalignment from matched stars: ``src_xy`` and ``dst_xy``
    (N, 2) host positions on the target grid, ``noise_var`` (N, 2) the
    centroid variance per axis of each pair (both sides added).

    Pairs whose residual is more than ``clip`` standard deviations from the
    model (blended or mismatched stars, whose centroid errors the noise model
    does not describe) are rejected and the model refitted until the kept set
    is stable or ``max_rounds`` fits are done; the result is always the last
    fit and the stars it used. The default 5 keeps every Gaussian residual
    (3 drops the tail of a thousand good stars and biases the excess low by
    a few percent) while a blend, which sits tens of sigma out, still goes.
    ``clip=None`` keeps every pair, as ``fit_affine`` does. With fewer than
    ``min_poly`` stars the excess is one constant per axis instead of a
    quadratic field.
    """
    src_xy, dst_xy, noise_var = (
        np.asarray(a, dtype=np.float64) for a in (src_xy, dst_xy, noise_var)
    )
    used = np.ones(len(src_xy), bool)
    for _ in range(max_rounds):
        m = _misalignment_model(
            src_xy, dst_xy, noise_var, used, shape, min_poly
        )
        if clip is None:
            break
        e = dst_xy - (src_xy @ m.matrix.T + m.offset)
        x, y = src_xy[:, 0], src_xy[:, 1]
        p, vp = m.excess(x, y)
        base = noise_var + vp + m.affine_variance(x, y)
        # the scale every residual is judged by, set robustly so one outlier
        # cannot hide itself
        t2 = np.array(
            [
                chi2_median_scale((e - p)[used, a] ** 2, base[used, a], 1.0)
                for a in (0, 1)
            ]
        )
        new = np.all(
            np.abs(e - p) <= clip * np.sqrt(base + t2[None, :]), axis=1
        )
        if new.sum() < 4 or (new == used).all():
            break
        used = new
    return m


@dataclass(frozen=True)
class PsfWidth:
    """PSF width of an image from its stars: ``sigma_px`` the clipped median
    of the pixel-level second-moment widths, ``optical_sigma_px`` the same
    with the pixel removed, sqrt(sigma^2 - 1/12), ``standard_error_px`` the
    standard error of the median, ``scatter_px`` the robust scatter of the
    kept widths, ``n_used`` of ``n`` finite widths kept by the clipping.
    """

    sigma_px: float
    optical_sigma_px: float
    standard_error_px: float
    scatter_px: float
    n_used: int
    n: int


def psf_width(widths, *, clip: float = 3.0, iterations: int = 5) -> PsfWidth:
    """Median of the stars' pixel-level widths, clipped at ``clip`` robust
    sigmas for at most ``iterations`` rounds, with the standard error of a
    median, sqrt(pi / 2) sigma / sqrt(n).
    """
    w = np.asarray(to_host(widths), dtype=np.float64).ravel()
    w = w[np.isfinite(w)]
    if len(w) == 0:
        raise ValueError("psf_width needs at least one finite width")
    keep = np.ones(len(w), bool)
    for _ in range(iterations):
        med, s = np.median(w[keep]), robust_sigma(w[keep])
        new = np.abs(w - med) <= clip * s if s > 0 else keep
        if new.sum() < 3 or (new == keep).all():
            break
        keep = new
    m, s = float(np.median(w[keep])), robust_sigma(w[keep])
    return PsfWidth(
        m,
        float(np.sqrt(max(m * m - 1.0 / 12.0, 0.0))),
        float(np.sqrt(np.pi / 2.0) * s / np.sqrt(keep.sum())),
        s,
        int(keep.sum()),
        int(len(w)),
    )


@dataclass(frozen=True)
class MatchedStars:
    """What ``measure_matched_stars`` found: ``diagnostics`` (host scalars for
    the run summary), the ``misalignment`` model or None, the target's PSF
    width from the kept stars or None when no width was finite, the host
    bytes of the two low-threshold peak lists (``control_bytes``) and the
    target positions ``star_xy`` (N, 2) of the kept stars: the model's used
    set when it was fitted, every neighbor-free finite pair otherwise.
    """

    diagnostics: dict[str, Any]
    misalignment: Misalignment | None
    psf: PsfWidth | None
    control_bytes: int
    star_xy: np.ndarray


def measure_matched_stars(
    target,
    template,
    variance,
    target_xy,
    template_xy,
    *,
    box: int,
    fwhm: float,
    min_stars: int,
    skip_model: str | None = None,
    target_smoothed: SmoothedImage | None = None,
    template_smoothed: SmoothedImage | None = None,
) -> MatchedStars:
    """Measure the star pairs ``target_xy``, ``template_xy`` (N, 2, host, on
    the target grid) that the refinement matched and model what they show.

    Pairs with another peak within ``box`` pixels in either image are dropped
    (``neighbor_free``), both centroids are measured
    (``star_measurements``: the template has no variance plane, so its
    centroid noise lands in the excess), pairs with a non-finite centroid
    error are dropped and counted, and the misalignment model is fitted
    (``fit_misalignment``) unless ``skip_model`` gives the reason not to or
    fewer than ``min_stars`` pairs remain. The PSF width (``psf_width``) comes
    from the kept target stars: the model's used set when it was fitted,
    every neighbor-free pair otherwise. ``diagnostics["model"]`` is the
    model's mode and ``diagnostics["reason"]`` why there is none.
    """
    target_xy = np.asarray(target_xy, dtype=np.float64).reshape(-1, 2)
    template_xy = np.asarray(template_xy, dtype=np.float64).reshape(-1, 2)
    diag: dict[str, Any] = {
        "stars_matched": int(len(target_xy)),
        "model": None,
        "reason": None,
    }
    if len(target_xy) < 3:
        diag["reason"] = "fewer than 3 matched stars"
        return MatchedStars(diag, None, None, 0, target_xy)
    peaks_t = find_stars(
        target, fwhm=fwhm, smoothed=target_smoothed, **NEIGHBOR_PEAKS
    )
    peaks_r = find_stars(
        template, fwhm=fwhm, smoothed=template_smoothed, **NEIGHBOR_PEAKS
    )
    control_bytes = int(peaks_t.nbytes + peaks_r.nbytes)
    ok = neighbor_free(peaks_t, target_xy, float(box))
    ok &= neighbor_free(peaks_r, template_xy, float(box))
    mv, mt = target_xy[ok], template_xy[ok]
    diag["stars_neighbor_free"] = int(len(mv))
    if len(mv) < 3:
        diag["reason"] = "fewer than 3 neighbor-free stars"
        return MatchedStars(diag, None, None, control_bytes, mv)
    tm = star_measurements(
        target, variance, mv, box=box, fwhm=fwhm, smoothed=target_smoothed
    )
    rm = star_measurements(
        template, None, mt, box=box, fwhm=fwhm, smoothed=template_smoothed
    )
    noise = to_host(tm.centroid_error) ** 2 + to_host(rm.centroid_error) ** 2
    widths = to_host(tm.width)
    finite = np.isfinite(noise).all(axis=1)
    diag["stars_nonfinite_dropped"] = int((~finite).sum())
    mv, mt, noise, widths = (
        mv[finite],
        mt[finite],
        noise[finite],
        widths[finite],
    )
    model = None
    if skip_model is not None:
        diag["reason"] = skip_model
    elif len(mv) < min_stars:
        diag["reason"] = f"fewer than {min_stars} stars"
    else:
        model = fit_misalignment(mv, mt, noise, shape=target.shape)
        ny, nx = target.shape
        center = model.variance([(nx - 1) / 2.0], [(ny - 1) / 2.0])[0]
        diag.update(
            model=model.mode,
            stars_used=int(model.used.sum()),
            residual_rms_px=model.residual_rms,
            excess_sigma_px=np.sqrt(model.excess_variance).tolist(),
            center_sigma_px=np.sqrt(center).tolist(),
        )
        widths = widths[model.used]
        mv = mv[model.used]
    psf = None
    if np.isfinite(widths).any():
        psf = psf_width(widths)
        diag["psf_width"] = {
            "sigma_px": psf.sigma_px,
            "standard_error_px": psf.standard_error_px,
            "scatter_px": psf.scatter_px,
            "n_used": psf.n_used,
            "n": psf.n,
        }
    return MatchedStars(diag, model, psf, control_bytes, mv)
