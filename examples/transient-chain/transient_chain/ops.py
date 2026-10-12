# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Stage operations for the GPU-resident Rubin chain.

Every function takes arrays from one array module (NumPy on the host for
tests,
CuPy on the device in production) and returns arrays from the same module, or
small host scalars where a decision needs one. Nothing here touches a file.
"""

from __future__ import annotations

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


def find_stars(
    img,
    *,
    fwhm: float = 4.0,
    nsigma: float = 20.0,
    max_n: int = 600,
    box: int = 15,
    iterations: int = 4,
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
    background-subtracted image inside a fixed box around the peak. Returns
    (N, 2) host x, y
    sorted by box flux, brightest first (when more than ``2*max_n`` peaks
    exist, the pre-cut uses
    the same box flux, so the final list is the true top ``max_n``). Only the
    star list crosses
    to the host.
    """
    xp = array_module(img)
    nd = ndimage_for(xp)
    finite = xp.isfinite(img)
    if int(finite.sum()) < 1000:
        return np.empty((0, 2))
    vals = img[finite]
    med = xp.median(vals)
    std = 1.4826 * xp.median(xp.abs(vals - med))
    work = xp.where(finite, img - med, 0.0).astype(xp.float64)
    smooth = nd.gaussian_filter(work, sigma=fwhm / 2.3548, mode="nearest")
    thr = nsigma * std
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
    ok = np.isfinite(d)
    return a[ok], b[j[ok]]


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
