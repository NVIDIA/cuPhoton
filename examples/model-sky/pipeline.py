# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Workshop measurements using cuPhoton. Never reads injection truth."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy import ndimage
from scipy.optimize import least_squares

from cuphoton.xfit import (
    StampDipoleModel,
    fit_dipoles,
    fit_single_sources,
    forced_psf_photometry,
)
from cuphoton.xpois import (
    GaussianBasisComponent,
    solve_constant_kernel,
    standardize_constant_kernel_residual,
)
from cuphoton.xrep import (
    BBox,
    ReprojectionSpec,
    make_wcs_pair_transform,
    reproject_masked_array,
)


def gpu():
    """Require CUDA rather than silently demonstrate a CPU fallback."""
    import cupy as cp

    cp.cuda.runtime.getDeviceCount()
    cp.cuda.Device().use()
    return cp


def load_bundle(root):
    root = Path(root).resolve()
    return root, json.loads((root / "bundle.json").read_text())


def bundle_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("bundle paths must remain within the bundle")
    return path


def read_exposure(path):
    with fits.open(path) as hdus:
        return {
            "science": np.asarray(hdus["SCI"].data, dtype=float),
            "variance": np.asarray(hdus["VAR"].data, dtype=float),
            "mask": np.asarray(hdus["MASK"].data, dtype=np.uint32),
            "header": hdus["SCI"].header.copy(),
        }


def write_exposure(path, exposure):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(
                exposure["science"], exposure["header"], name="SCI"
            ),
            fits.ImageHDU(exposure["variance"], name="VAR"),
            fits.ImageHDU(exposure["mask"].astype(np.uint32), name="MASK"),
        ]
    ).writeto(path, overwrite=True)


def good_pixels(exposure):
    return (
        (exposure["mask"] == 0)
        & np.isfinite(exposure["science"])
        & np.isfinite(exposure["variance"])
        & (exposure["variance"] > 0)
    )


def stamp(image, xy, size=41):
    x, y = np.rint(xy).astype(int)
    half = size // 2
    part = image[y - half : y + half + 1, x - half : x + half + 1]
    if part.shape != (size, size):
        raise ValueError("source too close to an image boundary")
    return part.copy()


def local_sky(image):
    yy, xx = np.indices(image.shape)
    radius = np.hypot(
        xx - (image.shape[1] - 1) / 2, yy - (image.shape[0] - 1) / 2
    )
    annulus = image[
        (radius > image.shape[0] * 0.34) & (radius < image.shape[0] * 0.49)
    ]
    return float(np.nanmedian(annulus))


def select_stars(exposure, *, exclude=(), count=12, margin=28):
    """Example helper: separated compact peaks selected from the image."""
    data = exposure["science"]
    smoothed = ndimage.gaussian_filter(np.nan_to_num(data), 1.2)
    noise = np.sqrt(np.nanmedian(exposure["variance"]))
    peaks = smoothed == ndimage.maximum_filter(smoothed, size=15)
    peaks &= smoothed > np.nanmedian(smoothed) + 10 * noise
    peaks &= good_pixels(exposure)
    peaks[:margin] = peaks[-margin:] = False
    peaks[:, :margin] = peaks[:, -margin:] = False
    ys, xs = np.where(peaks)
    selected = []
    for i in np.argsort(smoothed[ys, xs])[::-1]:
        xy = np.array([xs[i], ys[i]], dtype=float)
        if any(np.linalg.norm(xy - p) < 32 for p in [*exclude, *selected]):
            continue
        patch = stamp(data, xy, 31)
        signal = np.maximum(patch - local_sky(patch), 0)
        yy, xx = np.indices(signal.shape)
        radius2 = (xx - 15) ** 2 + (yy - 15) ** 2
        signal[radius2 > 8**2] = 0
        if signal.sum() <= 0:
            continue
        sigma = np.sqrt((signal * radius2).sum() / signal.sum() / 2)
        if 1.2 < sigma < 4.5 and stamp(good_pixels(exposure), xy, 17).all():
            selected.append(xy)
        if len(selected) == count:
            break
    if len(selected) < 3:
        raise ValueError("need three independent compact field stars")
    positions, widths, converged = fit_positions(exposure, np.array(selected))
    # Extended galaxies can pass a peak/moment cut. The bright, narrow locus
    # supplies a data-derived stellar width; retain only consistent sources.
    locus = np.median(widths[:5][converged[:5]])
    keep = converged & (widths > 0.8 * locus) & (widths < 1.25 * locus)
    if keep.sum() < 3:
        raise ValueError("need three sources in the stellar width locus")
    return positions[keep]


def fit_positions(exposure, positions):
    cp = gpu()
    images = np.stack([stamp(exposure["science"], p) for p in positions])
    images -= np.array([local_sky(s) for s in images])[:, None, None]
    variance = np.stack([stamp(exposure["variance"], p) for p in positions])
    good = np.stack([stamp(good_pixels(exposure), p) for p in positions])
    fit = fit_single_sources(
        cp.asarray(images),
        cp.asarray(variance),
        cp.asarray(positions - np.rint(positions)),
        mask=cp.asarray(good),
    )
    xy = np.rint(positions) + cp.asnumpy(cp.stack([fit.x, fit.y], axis=1))
    sigma = cp.asnumpy(cp.sqrt(fit.parameters[:, 1] * fit.parameters[:, 2]))
    return xy, sigma, cp.asnumpy(fit.converged)


def register(science, reference, *, anchors=None):
    """Warp all planes with xRep; refine translation on stationary stars."""
    mapping = make_wcs_pair_transform(
        WCS(reference["header"]), WCS(science["header"])
    )
    height, width = reference["science"].shape

    def warp(transform):
        result = reproject_masked_array(
            science["science"],
            science["variance"],
            science["mask"],
            ReprojectionSpec(
                mapping=transform,
                output_bbox=BBox(0, 0, width, height),
                mapping_grid_step=32,
            ),
            backend="cupy",
        )
        return {
            "science": result.image,
            "variance": result.variance,
            "mask": result.mask,
            "header": reference["header"].copy(),
            "registration": result.metadata,
        }

    coarse = warp(mapping)
    offset, floor = np.zeros(2), 0.0
    if anchors is not None:
        ref_xy, _, ref_ok = fit_positions(reference, anchors)
        sci_xy, _, sci_ok = fit_positions(coarse, anchors)
        valid = ref_ok & sci_ok & np.isfinite(sci_xy).all(axis=1)
        delta = sci_xy[valid] - ref_xy[valid]
        if len(delta) < 3:
            raise ValueError("registration needs three stationary stars")
        offset = np.median(delta, axis=0)
        floor = float(np.sqrt(np.mean((delta - offset) ** 2)))
        if np.linalg.norm(offset) > 2:
            raise ValueError("registration correction exceeds two pixels")
        coarse = warp(lambda xy: mapping(xy + offset))
    coarse["registration_offset"] = offset
    coarse["registration_floor"] = floor
    return coarse


def exclusion_disks(shape, positions, radius=22):
    yy, xx = np.indices(shape)
    excluded = np.zeros(shape, dtype=bool)
    for x, y in positions:
        excluded |= (xx - x) ** 2 + (yy - y) ** 2 < radius**2
    return excluded


def detect_trail(difference, variance, good=None, *, width_scale=1.0):
    """CPU line voting proposes a mask; this is not a cuPhoton detector."""
    from skimage.transform import hough_line, hough_line_peaks

    good = np.ones(difference.shape, bool) if good is None else good
    snr = np.divide(
        difference,
        np.sqrt(variance),
        out=np.zeros_like(difference),
        where=good & (variance > 0),
    )
    votes, angles, distances = hough_line(good & (snr > 5))
    peaks, theta, rho = hough_line_peaks(
        votes,
        angles,
        distances,
        num_peaks=1,
        threshold=max(40, min(difference.shape) * 0.25),
    )
    mask = np.zeros_like(good)
    if not len(peaks):
        return mask, {"detected": False}
    yy, xx = np.indices(difference.shape)
    perpendicular = xx * np.cos(theta[0]) + yy * np.sin(theta[0]) - rho[0]
    # A localized elongated source can win a line vote. Require persistence
    # along most of a field-crossing track before proposing a trail mask.
    along = -xx * np.sin(theta[0]) + yy * np.cos(theta[0])
    corridor = good & (np.abs(perpendicular) < 3)
    bins = np.linspace(along[corridor].min(), along[corridor].max(), 13)
    supported, _ = np.histogram(along[corridor & (snr > 5)], bins)
    if np.count_nonzero(supported >= 3) < 9:
        return mask, {"detected": False}
    coordinate = np.arange(-35, 36)
    profile = np.array(
        [
            np.nanmedian(
                np.where(
                    good & (np.abs(perpendicular - t) < 0.5),
                    difference,
                    np.nan,
                )
            )
            for t in coordinate
        ]
    )
    base = np.nanmedian(profile[np.abs(coordinate) > 25])
    positive = np.maximum(profile - base, 0)
    center = coordinate[np.nanargmax(positive)]
    half = coordinate[positive > np.nanmax(positive) / 2]
    sigma = max(2.0, (half.max() - half.min()) / 2.355)
    mask = np.abs(perpendicular - center) < 3.5 * sigma * width_scale
    return mask, {
        "detected": True,
        "theta": float(theta[0]),
        "rho": float(rho[0] + center),
        "sigma_pixels": float(sigma),
        "votes": int(peaks[0]),
        "coordinate": coordinate,
        "profile": profile,
    }


def subtract(reference, science, *, exclude=(), trail_width=1.0):
    """Fit, detect a trail, exclude it, then refit the xPois kernel."""
    usable = good_pixels(reference) & good_pixels(science)
    fitting = usable & ~exclusion_disks(usable.shape, exclude)
    components = [
        GaussianBasisComponent(sigma=0.7, degree=2),
        GaussianBasisComponent(sigma=1.6, degree=2),
        GaussianBasisComponent(sigma=3.0, degree=0),
    ]

    def solve(mask):
        return solve_constant_kernel(
            reference["science"],
            science["science"],
            components,
            variance=science["variance"],
            fit_mask=mask,
            kernel_shape=(21, 21),
            background_degree=0,
            backend="cupy",
        )

    preliminary = solve(fitting)
    trail, profile = detect_trail(
        preliminary.residual,
        science["variance"] + reference["variance"],
        usable,
        width_scale=trail_width,
    )
    fit = solve(fitting & ~trail) if trail.any() else preliminary
    noise = standardize_constant_kernel_residual(
        fit.residual,
        kernel=fit.kernel,
        target_variance=science["variance"],
        reference_variance=reference["variance"],
        valid_mask=usable & ~trail,
    )
    return {
        "science": science["science"],
        "reference": reference["science"],
        "matched": fit.matched,
        "difference": fit.residual,
        "variance": np.asarray(noise.marginal_residual_variance),
        "valid": np.asarray(noise.valid_mask),
        "trail_mask": trail,
        "trail_profile": profile,
        "kernel": fit.kernel,
        "background": fit.background,
        "fit_mask": fit.fit_mask,
        "backend": fit.backend,
        "preliminary": preliminary.residual,
        "registered": science,
    }


def measure_motion(reference, science, old_guess, *, search_radius=15):
    anchors = select_stars(reference, exclude=[old_guess])
    registered = register(science, reference, anchors=anchors)
    subtraction = subtract(reference, registered, exclude=[old_guess])
    size = 2 * int(search_radius) + 21
    image = stamp(subtraction["difference"], old_guess, size)
    variance = stamp(subtraction["variance"], old_guess, size)
    valid = stamp(subtraction["valid"], old_guess, size)
    # Measure the sampled PSF in the matched reference. A free Gaussian
    # width can trade against dipole separation when the lobes overlap.
    matched_exposure = {**reference, "science": subtraction["matched"]}
    basis_xy, _, _ = fit_positions(matched_exposure, np.array([old_guess]))
    basis = stamp(subtraction["matched"], basis_xy[0])
    basis -= local_sky(basis)
    basis = ndimage.shift(
        basis, (np.rint(basis_xy[0]) - basis_xy[0])[::-1], order=3
    )
    basis = np.maximum(basis, 0)
    basis /= basis.sum()
    # A native-pixel bilinear interpolant broadens fractional shifts and
    # can trade that broadening against dipole separation. Sample the
    # measured PSF on quarter-pixel centers before xFit evaluates it.
    factor = 4
    yy, xx = np.meshgrid(
        np.arange((basis.shape[0] - 1) * factor + 1) / factor,
        np.arange((basis.shape[1] - 1) * factor + 1) / factor,
        indexing="ij",
    )
    basis = np.maximum(ndimage.map_coordinates(basis, [yy, xx], order=3), 0)
    basis /= basis.sum()
    basis_scale = 1 / factor
    fit = fit_dipoles(
        image[None],
        model=StampDipoleModel(
            basis, image_shape=image.shape, scale=basis_scale, backend="cupy"
        ),
        variance=variance[None],
        mask=valid[None],
        backend="cupy",
    )
    center = np.rint(old_guess)
    positive = center + fit.parameters[0, :2]
    negative = center + fit.parameters[0, 2:4]
    # A stationary target can leave only noise while an unrelated object
    # supplies residual lobes. Associate the fading lobe with the star
    # measured in the old image, and keep the new lobe in the search area.
    if (
        np.linalg.norm(negative - basis_xy[0]) > 2
        or np.linalg.norm(positive - basis_xy[0]) > search_radius
    ):
        raise RuntimeError("dipole is not associated with the searched star")
    offset = positive - negative
    separation = float(np.linalg.norm(offset))
    if not np.isfinite(separation) or separation == 0:
        raise RuntimeError("moving-star displacement is not constrained")
    gradient = np.r_[offset, -offset, 0.0] / separation
    covariance = fit.covariance[0]
    separation_variance = float(gradient @ covariance @ gradient)
    flux_variance = float(covariance[4, 4])
    if separation_variance <= 0 or flux_variance <= 0:
        raise RuntimeError("moving-star covariance is not constrained")
    correlation = float(
        (gradient @ covariance[:, 4])
        / np.sqrt(separation_variance * flux_variance)
    )
    # Closely overlapping lobes mostly constrain flux times displacement.
    # This conservative example guard avoids quoting a misleading local
    # Hessian error along that degeneracy; it is not a resolution limit.
    if abs(correlation) > 0.98:
        raise RuntimeError(
            "overlapping dipole lobes trade flux against separation; "
            "use a longer baseline or direct epoch astrometry"
        )
    statistical = float(np.sqrt(separation_variance))
    floor = registered["registration_floor"]
    error = float(np.hypot(statistical, np.sqrt(2) * floor))
    if (
        not fit.converged[0]
        or not np.isfinite(error)
        or error > separation / 2
    ):
        raise RuntimeError("moving-star displacement is not constrained")
    return {
        **subtraction,
        "old_xy": negative,
        "new_xy": positive,
        "separation_pixels": separation,
        "separation_error_pixels": error,
        "registration_floor_pixels": floor,
        "flux_separation_correlation": correlation,
        "fit": fit,
        "fit_input": {
            "images": image[None],
            "variance": variance[None],
            "mask": valid[None],
            "stamp_basis": basis[None],
            "stamp_scale": np.asarray(basis_scale),
        },
        "anchors": anchors,
    }


def empirical_psf(exposure, anchors, size=41):
    xy, sigma, converged = fit_positions(exposure, anchors)
    samples = []
    for point, good, width in zip(xy, converged, sigma, strict=True):
        if not good or not 1 < width < 6:
            continue
        patch = stamp(exposure["science"], point, size)
        patch -= local_sky(patch)
        patch = ndimage.shift(patch, (np.rint(point) - point)[::-1], order=3)
        if patch.sum() > 0:
            samples.append(patch / patch.sum())
    if len(samples) < 3:
        raise ValueError("too few field stars to estimate an empirical PSF")
    psf = np.median(samples, axis=0)
    yy, xx = np.indices(psf.shape)
    psf[np.hypot(xx - size // 2, yy - size // 2) > size * 0.34] = 0
    psf = np.maximum(psf, 0)
    return psf / psf.sum()


def photometry(subtraction, positions, psf):
    """F_ref + delta F, using identical linear weights for both terms.

    The reference contribution cancels in the total, leaving science-image
    formal variance. Position, PSF, sky and resampling errors are conditional;
    calibrate them with check stars and independently simulated exposures.
    """
    cp = gpu()
    science = subtraction["registered"]
    reference, delta, models, variance, valid = [], [], [], [], []
    for point in positions:
        target = stamp(science["science"], point)
        matched = stamp(subtraction["matched"], point)
        sky_target, sky_matched = local_sky(target), local_sky(matched)
        reference.append(matched - sky_matched)
        delta.append(target - matched - (sky_target - sky_matched))
        models.append(
            ndimage.shift(psf, (point - np.rint(point))[::-1], order=3)
        )
        variance.append(stamp(science["variance"], point))
        valid.append(stamp(subtraction["valid"], point))
    weights = cp.asarray(np.array(variance))
    mask = cp.asarray(np.array(valid))
    models = cp.asarray(np.array(models))
    offsets = cp.asarray(positions - np.rint(positions))
    reference_flux = forced_psf_photometry(
        cp.asarray(np.array(reference)),
        weights,
        offsets,
        1,
        mask=mask,
        psf=models,
    )
    difference_flux = forced_psf_photometry(
        cp.asarray(np.array(delta)),
        weights,
        offsets,
        1,
        mask=mask,
        psf=models,
    )
    # A formal fit to the surviving wings is not reliable when a trail
    # removes the core. Require at least 90% of the measured PSF mass.
    retained = cp.sum(cp.maximum(models, 0) * mask, axis=(1, 2))
    retained /= cp.sum(cp.maximum(models, 0), axis=(1, 2))
    reliable = difference_flux.valid & (retained >= 0.9)
    total = reference_flux.flux + difference_flux.flux
    return (
        cp.asnumpy(cp.where(reliable, total, cp.nan)),
        cp.asnumpy(cp.where(reliable, difference_flux.flux_error, cp.nan)),
        cp.asnumpy(reliable),
        cp.asnumpy(cp.where(reliable, difference_flux.flux, cp.nan)),
    )


def transit_shape(times, radius_ratio, midtime, assumptions):
    import batman

    params = batman.TransitParams()
    params.t0 = midtime / 1440
    params.per = assumptions.get("period_days", 1.580404531)
    params.rp = radius_ratio
    params.a = assumptions.get("a_over_rstar", 14.97)
    params.inc = assumptions.get("inclination_deg", 88.980)
    params.ecc, params.w = assumptions.get("eccentricity", 0), 90
    params.u = assumptions.get("limb_darkening", [0.6, 0.15])
    params.limb_dark = "quadratic"
    model = batman.TransitModel(
        params,
        np.asarray(times) / 1440,
        supersample_factor=7,
        exp_time=assumptions.get("exposure_s", 30) / 86400,
    )
    return model.light_curve(params)


def fit_transit(times, flux, errors, valid, assumptions):
    valid = valid & np.isfinite(flux) & np.isfinite(errors) & (errors > 0)
    t, y, err = times[valid], flux[valid], errors[valid]
    if len(t) < 30:
        raise ValueError("not enough valid epochs for a transit fit")

    def model(p, x):
        return (p[2] + p[3] * x / 60) * transit_shape(
            x, p[0], p[1], assumptions
        )

    result = least_squares(
        lambda p: (model(p, t) - y) / err,
        [0.1, 0, 1, 0],
        bounds=([0.01, -25, 0.95, -0.05], [0.3, 25, 1.05, 0.05]),
    )
    covariance = np.linalg.pinv(result.jac.T @ result.jac)
    return {
        "parameters": result.x,
        "covariance": covariance,
        "model": model(result.x, times),
        "success": bool(result.success),
        "radius_ratio": float(result.x[0]),
        "radius_ratio_error": float(np.sqrt(covariance[0, 0])),
        "geometric_depth": float(result.x[0] ** 2),
        "midtime_min": float(result.x[1]),
        "midtime_error_min": float(np.sqrt(covariance[1, 1])),
    }


def measure_transit(
    root,
    manifest,
    target_xy,
    *,
    comparison_count=5,
    trail_width=1.0,
    progress=None,
):
    reference = read_exposure(
        bundle_path(root, manifest["files"]["reference_current"])
    )
    # Motion finds the star; refine its fixed position on the independent
    # current reference, rather than fitting each noisy transit frame.
    target_fit, _, target_ok = fit_positions(reference, np.array([target_xy]))
    if not target_ok[0]:
        raise RuntimeError(
            "current-reference target position is not measured"
        )
    target_xy = target_fit[0]
    stars = select_stars(reference, exclude=[target_xy], count=12)
    check = stars[0]
    comparisons = stars[1 : comparison_count + 1]
    anchors = stars[1:]
    positions = np.vstack([target_xy, check, comparisons])
    rows, errors, good, deltas = [], [], [], []
    stamps, differences, flags = [], [], []
    started = perf_counter()
    for index, epoch in enumerate(manifest["epochs"]):
        science = read_exposure(bundle_path(root, epoch["path"]))
        aligned = register(science, reference, anchors=anchors)
        difference = subtract(
            reference,
            aligned,
            exclude=[target_xy, check],
            trail_width=trail_width,
        )
        psf = empirical_psf(aligned, anchors)
        f, e, ok, df = photometry(difference, positions, psf)
        rows.append(f)
        errors.append(e)
        good.append(ok)
        deltas.append(df)
        stamps.append(stamp(aligned["science"], target_xy, 65))
        differences.append(stamp(difference["difference"], target_xy, 65))
        flags.append(int(difference["trail_mask"].sum()))
        if progress and (
            index % 10 == 0 or index == len(manifest["epochs"]) - 1
        ):
            progress(index + 1, len(manifest["epochs"]))
    gpu().cuda.get_current_stream().synchronize()
    flux, error, good = np.array(rows), np.array(errors), np.array(good)
    medians = np.nanmedian(np.where(good, flux, np.nan), axis=0)
    relative, relative_error = flux / medians, error / medians
    comp_good = good[:, 2:] & np.isfinite(relative[:, 2:])
    weight = np.where(comp_good, 1 / relative_error[:, 2:] ** 2, 0)
    denom = weight.sum(axis=1)
    ensemble = (
        np.sum(np.where(comp_good, relative[:, 2:], 0) * weight, axis=1)
        / denom
    )
    ensemble_error = np.sqrt(1 / denom)
    target, check_flux = relative[:, 0] / ensemble, relative[:, 1] / ensemble
    check_error = np.hypot(
        relative_error[:, 1] / ensemble,
        check_flux * ensemble_error / ensemble,
    )
    check_good = good[:, 1] & (denom > 0) & np.isfinite(check_flux)
    checks = check_flux[check_good]
    scatter = 1.4826 * np.median(np.abs(checks - np.median(checks)))
    scale = max(1.0, float(scatter / np.nanmedian(check_error[check_good])))
    formal = np.hypot(
        relative_error[:, 0] / ensemble, target * ensemble_error / ensemble
    )
    target_error = formal * scale
    valid = good[:, 0] & (denom > 0) & np.isfinite(target)
    baseline = np.nanpercentile(target[valid], 70)
    target /= baseline
    target_error /= baseline
    times = np.array([e["time_min"] for e in manifest["epochs"]])
    fit = fit_transit(
        times, target, target_error, valid, manifest.get("assumptions", {})
    )
    return {
        "times_minutes": times,
        "flux": target,
        "flux_error": target_error,
        "formal_error": formal / baseline,
        "valid": valid,
        "check_flux": check_flux,
        "check_error": check_error * scale,
        "error_scale": scale,
        "raw_flux": flux,
        "raw_error": error,
        "difference_flux": np.array(deltas),
        "positions": positions,
        "stamps": np.array(stamps),
        "difference_stamps": np.array(differences),
        "trail_mask_pixels": np.array(flags),
        "fit": fit,
        "reference_stamp": stamp(reference["science"], target_xy, 65),
        "elapsed_seconds": perf_counter() - started,
        "backend": "cupy",
    }
