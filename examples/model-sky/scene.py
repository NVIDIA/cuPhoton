# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Model Sky exposures; injection truth lives separately."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter, maximum_filter, shift


@dataclass(frozen=True)
class SceneConfig:
    """One fictional M dwarf, with GJ 1214 inspired geometry."""

    seed: int = 260611
    n_epochs: int = 120
    exposure_s: float = 30.0
    cadence_s: float = 60.0
    reference_exposures: int = 16
    distance_pc: float = 146.42
    proper_motion_ra_mas_yr: float = 58.0202
    proper_motion_dec_mas_yr: float = -74.9713
    baseline_years: float = 10.0
    target_flux_njy: float = 348775.3294253341
    stellar_radius_solar: float = 0.2162
    stellar_mass_solar: float = 0.1820
    stellar_temperature_k: float = 3100.0
    period_days: float = 1.580404531
    radius_ratio: float = 0.11589
    a_over_rstar: float = 14.97
    inclination_deg: float = 88.980
    eccentricity: float = 0.0
    # Declared illustrative quadratic limb-darkening coefficients.
    limb_darkening_u1: float = 0.6
    limb_darkening_u2: float = 0.15
    reference_blur_sigma_px: float = 1.0
    trail_width_arcsec: float = 2.8
    trail_epoch: int = 43
    simulation_midpoint_mjd: float = 61354.25


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def transit_flux(time_min, config):
    """Exposure-integrated quadratic limb-darkened relative flux."""
    import batman

    params = batman.TransitParams()
    params.t0 = 0.0
    params.per = config.period_days
    params.rp = config.radius_ratio
    params.a = config.a_over_rstar
    params.inc = config.inclination_deg
    params.ecc = config.eccentricity
    params.w = 90.0
    params.u = [config.limb_darkening_u1, config.limb_darkening_u2]
    params.limb_dark = "quadratic"
    times = np.atleast_1d(np.asarray(time_min, dtype=float)) / 1440.0
    model = batman.TransitModel(
        params,
        times,
        supersample_factor=11,
        exp_time=config.exposure_s / 86400.0,
    )
    return model.light_curve(params)


def draw_exposure(model, sky_variance, noise_scale, rng, n_average=1):
    """Draw calibrated pixels with source variance a*M and sky/read variance.

    a is nJy per effective electron: a*Poisson(max(M,0)/a). Negative
    model values retain their mean but contribute no source Poisson noise.
    n_average generates the mean of independent equal-duration exposures.
    """
    model = np.asarray(model, dtype=float)
    sky_variance = np.broadcast_to(sky_variance, model.shape)
    noise_scale = np.broadcast_to(noise_scale, model.shape)
    if not np.all(np.isfinite(model)):
        raise ValueError("latent scene must be finite")
    if not np.all(np.isfinite(noise_scale) & (noise_scale > 0)):
        raise ValueError(
            "noise_scale must be positive nJy/effective-electron"
        )
    if not np.all(np.isfinite(sky_variance) & (sky_variance >= 0)):
        raise ValueError("sky_variance must be finite and nonnegative")
    if not isinstance(n_average, int) or n_average < 1:
        raise ValueError("n_average must be a positive integer")
    positive = np.maximum(model, 0)
    electrons = rng.poisson(positive * n_average / noise_scale)
    image = model + electrons * noise_scale / n_average - positive
    image += rng.normal(size=model.shape) * np.sqrt(sky_variance / n_average)
    variance = (sky_variance + noise_scale * positive) / n_average
    return image.astype(np.float32), variance.astype(np.float32)


def normalized_psf(base_psf, blur_sigma_px):
    base = np.asarray(base_psf, dtype=float)
    if base.ndim != 2 or not np.all(np.isfinite(base)):
        raise ValueError("base_psf must be a finite two-dimensional kernel")
    # Empirical interpolation may leave small unphysical negative wings.
    base = np.maximum(base, 0)
    if base.sum() <= 0:
        raise ValueError("base_psf must have positive integrated flux")
    padding = int(np.ceil(5 * blur_sigma_px)) + 3
    base = np.pad(base, padding)
    base = np.pad(base, [(0, 1 - size % 2) for size in base.shape])
    psf = gaussian_filter(base, blur_sigma_px, mode="constant")
    return psf / psf.sum()


def add_star(image, psf, xy, flux):
    """Add an integrated-flux source at zero-based x,y coordinates."""
    xy = np.asarray(xy, dtype=float)
    center = np.rint(xy).astype(int)
    shifted = shift(psf, (xy - center)[::-1], order=3, mode="constant")
    shifted = np.maximum(shifted, 0)
    shifted *= flux / shifted.sum()
    x0, y0 = center - np.asarray(psf.shape[::-1]) // 2
    x1, y1 = x0 + psf.shape[1], y0 + psf.shape[0]
    left, right = max(0, x0), min(image.shape[1], x1)
    bottom, top = max(0, y0), min(image.shape[0], y1)
    if right > left and top > bottom:
        image[bottom:top, left:right] += shifted[
            bottom - y0 : top - y0, left - x0 : right - x0
        ]


def satellite_trail(
    shape,
    target_xy,
    pixel_scale_arcsec,
    width_arcsec=2.8,
    peak_njy=1800,
):
    """Finite-distance, double-lobed cross section across a target wing.

    An analytic teaching profile, not an optical ray trace or a prediction
    for a particular satellite.
    """
    yy, xx = np.indices(shape, dtype=float)
    angle = np.deg2rad(29.0)
    dx, dy = xx - target_xy[0], yy - target_xy[1]
    width = width_arcsec / pixel_scale_arcsec
    across = -np.sin(angle) * dx + np.cos(angle) * dy - 0.48 * width
    along = np.cos(angle) * dx + np.sin(angle) * dy
    sigma = 0.21 * width
    profile = np.exp(-0.5 * ((across - 0.25 * width) / sigma) ** 2)
    profile += np.exp(-0.5 * ((across + 0.25 * width) / sigma) ** 2)
    profile /= profile.max()
    return peak_njy * profile * (1 + 0.08 * np.sin(2 * np.pi * along / 170))


def binary_quality_mask(raw, metadata):
    """Translate documented bad flags; DETECTED alone is not a bad pixel."""
    raw = np.asarray(raw)
    if metadata.get("quality_mask_is_binary", False):
        return (raw != 0).astype(np.uint8)
    bad_bits = metadata.get("bad_mask_bits")
    if bad_bits is None and "mask_planes" in metadata:
        names = (
            "BAD",
            "SAT",
            "CR",
            "EDGE",
            "NO_DATA",
            "INTRP",
            "SUSPECT",
            "STREAK",
        )
        bad_bits = sum(
            1 << int(metadata["mask_planes"][name])
            for name in names
            if name in metadata["mask_planes"]
        )
    if bad_bits is not None:
        return ((raw.astype(np.uint64) & int(bad_bits)) != 0).astype(np.uint8)
    if not np.any(raw):
        return np.zeros(raw.shape, dtype=np.uint8)
    raise ValueError("nonzero quality_mask needs its documented bit mapping")


def _source_positions(background, bad):
    """Place isolated sources; their identities are not measurement priors."""
    anchors = [
        (0.52, 0.48),
        (0.23, 0.23),
        (0.76, 0.22),
        (0.24, 0.76),
        (0.77, 0.76),
        (0.51, 0.20),
    ]
    score = gaussian_filter(np.abs(background), 4)
    score[maximum_filter(bad, size=25) != 0] = np.inf
    radius = max(4, min(background.shape) // 30)
    positions = []
    for fx, fy in anchors:
        cx, cy = int(fx * background.shape[1]), int(fy * background.shape[0])
        y0, y1, x0, x1 = (
            cy - radius,
            cy + radius + 1,
            cx - radius,
            cx + radius + 1,
        )
        region = score[y0:y1, x0:x1]
        if not np.any(np.isfinite(region)):
            raise ValueError("no clean injection location near an anchor")
        y, x = np.unravel_index(np.argmin(region), region.shape)
        positions.append([float(x + x0), float(y + y0)])
    return np.asarray(positions)


def _motion_offset(wcs, xy, config):
    coordinate = wcs.pixel_to_world(*xy)
    moved = coordinate.spherical_offsets_by(
        config.proper_motion_ra_mas_yr * config.baseline_years * u.mas,
        config.proper_motion_dec_mas_yr * config.baseline_years * u.mas,
    )
    return np.asarray(wcs.world_to_pixel(moved)) - xy


def _write_exposure(
    path,
    science,
    variance,
    mask,
    wcs_header,
    time_min,
    config,
    dither=(0.0, 0.0),
    n_average=1,
    modeled_scene=False,
):
    header = wcs_header.copy()
    header["CRPIX1"] += float(dither[0])
    header["CRPIX2"] += float(dither[1])
    mjd = config.simulation_midpoint_mjd + time_min / 1440.0
    header["SIMULATE"] = (True, "Model Sky simulated exposure")
    header["SIMMJD"] = (mjd, "SIMULATED midpoint MJD, UTC")
    span_s = (n_average - 1) * config.cadence_s + config.exposure_s
    header["MJD-OBS"] = (
        mjd - span_s / 172800,
        "SIMULATED exposure/block start",
    )
    header["DATE-OBS"] = Time(header["MJD-OBS"], format="mjd").isot
    header["TIMESYS"] = "UTC"
    header["EXPTIME"] = (
        config.exposure_s,
        "Seconds per constituent exposure",
    )
    header["NCOMBINE"] = n_average
    header["TOTALEXP"] = config.exposure_s * n_average
    header["TREL_MIN"] = (
        float(time_min),
        "Minutes from simulated series center",
    )
    header["BUNIT"] = "nJy"
    header.add_history("All sky and instrument inputs are analytic models.")
    header.add_history(
        "MASK: 0 usable; 1 bad. Injected trails are not pre-masked."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList(
        [
            fits.PrimaryHDU(header=header),
            fits.ImageHDU(science, header=header, name="SCI"),
            fits.ImageHDU(
                variance, header=fits.Header({"BUNIT": "nJy2"}), name="VAR"
            ),
            fits.ImageHDU(mask.astype(np.uint8), name="MASK"),
        ]
    ).writeto(path, overwrite=True, checksum=True)


def load_exposure(path):
    """Read measurement planes without opening any truth product."""
    with fits.open(path) as hdus:
        return {
            "science": np.array(hdus["SCI"].data),
            "variance": np.array(hdus["VAR"].data),
            "mask": np.array(hdus["MASK"].data),
            "header": hdus["SCI"].header.copy(),
        }


def prepare_bundle(
    calibration_dir, output_dir, *, seed=260611, force=False, config=None
):
    """Generate Model Sky FITS exposures from analytic calibration.

    Preparation generates the expected sky, then draws independent exposure
    noise. Measurement opens the generated planes without injection truth.
    """
    calibration_dir, output_dir = Path(calibration_dir), Path(output_dir)
    config = config or SceneConfig(seed=seed)
    inputs = [
        calibration_dir / f"calibration.{suffix}"
        for suffix in ("npz", "fits", "json")
    ]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    fingerprint = {
        "calibration_sha256": {path.name: sha256(path) for path in inputs},
        "generator_sha256": sha256(Path(__file__)),
        # Keep injected values out of the measurement manifest.
        "config_sha256": hashlib.sha256(
            json.dumps(asdict(config), sort_keys=True).encode()
        ).hexdigest(),
    }
    manifest_path = output_dir / "bundle.json"
    if manifest_path.exists() and not force:
        previous = json.loads(manifest_path.read_text())
        if previous.get("fingerprint") != fingerprint:
            raise ValueError(
                "bundle inputs changed; use --force to regenerate"
            )
        for relative, expected in previous["sha256"].items():
            path = output_dir / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            if sha256(path) != expected:
                raise ValueError(f"bundle file hash mismatch: {relative}")
        return previous
    metadata = json.loads(inputs[2].read_text())
    with np.load(inputs[0]) as arrays:
        background = arrays["background"].astype(float)
        base_psf = arrays["base_psf"].astype(float)
        sky_variance = arrays["sky_variance"].astype(float)
        noise_scale = arrays["noise_scale"].astype(float)
        bad = binary_quality_mask(arrays["quality_mask"], metadata)
    if background.ndim != 2 or min(background.shape) < 96:
        raise ValueError("calibration scene must be at least 96 by 96 pixels")
    if bad.shape != background.shape:
        raise ValueError("quality_mask and background must share their grid")
    with fits.open(inputs[1]) as hdus:
        header = hdus["SCI"].header if "SCI" in hdus else hdus[0].header
        wcs_header = WCS(header).celestial.to_header(relax=True)
    wcs = WCS(wcs_header)
    pixel_scale = float(metadata["pixel_scale_arcsec"])
    if pixel_scale <= 0:
        raise ValueError("pixel_scale_arcsec must be positive")
    border = min(14, min(background.shape) // 10)
    bad[:border] = bad[-border:] = 1
    bad[:, :border] = bad[:, -border:] = 1
    source_xy = _source_positions(background, bad)
    old_xy = source_xy[0]
    offset = _motion_offset(wcs, old_xy, config)
    current_xy = old_xy + offset
    stable_xy = source_xy[1:]
    stable_flux = config.target_flux_njy * np.array(
        [0.72, 1.08, 0.58, 0.92, 0.80]
    )
    rng = np.random.default_rng(config.seed)
    times = (
        (np.arange(config.n_epochs) - (config.n_epochs - 1) / 2)
        * config.cadence_s
        / 60
    )
    relative_flux = transit_flux(times, config)

    def render(
        time_min,
        target_xy,
        blur,
        throughput=1.0,
        dither=(0.0, 0.0),
        n_average=1,
        with_trail=False,
    ):
        psf = normalized_psf(base_psf, blur)
        scene = gaussian_filter(background, blur, mode="reflect")
        scene = shift(scene, dither[::-1], order=3, mode="reflect")
        location = target_xy + np.asarray(dither)
        constituent_times = (
            time_min
            + (np.arange(n_average) - (n_average - 1) / 2)
            * config.cadence_s
            / 60
        )
        factor = float(np.mean(transit_flux(constituent_times, config)))
        add_star(scene, psf, location, config.target_flux_njy * factor)
        for xy, flux in zip(stable_xy, stable_flux, strict=True):
            add_star(scene, psf, xy + np.asarray(dither), flux)
        scene *= throughput
        trail = np.zeros(background.shape)
        if with_trail:
            trail = satellite_trail(
                background.shape,
                location,
                pixel_scale,
                config.trail_width_arcsec,
            )
            scene += trail
        image, variance = draw_exposure(
            scene, sky_variance, noise_scale, rng, n_average
        )
        shifted_bad = shift(
            bad, dither[::-1], order=0, mode="constant", cval=1
        )
        mask = np.maximum(bad, shifted_bad)
        return image, variance, mask, psf, trail

    files = {
        "reference_old": "refs/reference_old.fits",
        "reference_current": "refs/reference_current.fits",
        "motion": "motion/science.fits",
        "trail": "trail/science.fits",
        "epochs": [
            f"epochs/epoch_{index:03d}.fits"
            for index in range(config.n_epochs)
        ],
    }
    specifications = [
        (
            "reference_old",
            -config.baseline_years * 365.25 * 1440 - 120,
            old_xy,
            config.reference_blur_sigma_px,
            config.reference_exposures,
            False,
        ),
        (
            "reference_current",
            -120,
            current_xy,
            config.reference_blur_sigma_px,
            config.reference_exposures,
            False,
        ),
        ("motion", -90, current_xy, 1.4, 1, False),
        ("trail", -45, current_xy, 1.4, 1, True),
    ]
    reference_psfs, all_paths = [], []
    # An interrupted forced rebuild must not leave a completion marker for
    # a mixture of old and newly written exposures.
    manifest_path.unlink(missing_ok=True)
    for key, time_min, xy, blur, n_average, with_trail in specifications:
        image, variance, mask, psf, trail = render(
            time_min,
            xy,
            blur,
            n_average=n_average,
            with_trail=with_trail,
        )
        path = output_dir / files[key]
        _write_exposure(
            path,
            image,
            variance,
            mask,
            wcs_header,
            time_min,
            config,
            modeled_scene=metadata.get("modeled_scene", True),
            n_average=n_average,
        )
        all_paths.append(files[key])
        if key.startswith("reference_"):
            reference_psfs.append(psf)
        if with_trail:
            standalone_trail = trail
    epochs, epoch_psfs, nuisance = [], [], []
    for index, time_min in enumerate(times):
        phase = index / max(config.n_epochs - 1, 1)
        blur = 1.2 + 0.35 * (1 + np.sin(2 * np.pi * phase + 0.3)) / 2
        throughput = (
            1
            + 0.018 * np.sin(2 * np.pi * phase + 0.7)
            + 0.006 * (phase - 0.5)
        )
        dither = (
            0.24 * np.sin(4 * np.pi * phase),
            0.22 * np.cos(3 * np.pi * phase),
        )
        image, variance, mask, psf, _ = render(
            time_min,
            current_xy,
            blur,
            throughput,
            dither,
            with_trail=index == config.trail_epoch,
        )
        relative = files["epochs"][index]
        _write_exposure(
            output_dir / relative,
            image,
            variance,
            mask,
            wcs_header,
            time_min,
            config,
            modeled_scene=metadata.get("modeled_scene", True),
            dither=dither,
        )
        all_paths.append(relative)
        epochs.append(
            {
                "index": index,
                "path": relative,
                "time_min": float(time_min),
                "exposure_s": config.exposure_s,
                "mjd_sim": config.simulation_midpoint_mjd + time_min / 1440,
            }
        )
        epoch_psfs.append(psf)
        nuisance.append([throughput, blur, *dither])
    nuisance = np.asarray(nuisance)
    psfs = reference_psfs + epoch_psfs
    max_shape = np.max([psf.shape for psf in psfs], axis=0)
    padded_psfs = np.array(
        [
            np.pad(
                psf,
                [
                    (int((outer - inner) // 2), int((outer - inner) // 2))
                    for outer, inner in zip(max_shape, psf.shape, strict=True)
                ],
            )
            for psf in psfs
        ]
    )
    truth_paths = {"json": "truth/truth.json", "npz": "truth/truth.npz"}
    (output_dir / "truth").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / truth_paths["npz"],
        psf_reference=padded_psfs[:2],
        psfs_epochs=padded_psfs[2:],
        relative_flux=relative_flux,
        epoch_target_flux=config.target_flux_njy
        * relative_flux
        * nuisance[:, 0],
        epoch_target_xy=current_xy[None, :] + nuisance[:, 2:],
        throughput=nuisance[:, 0],
        blur_sigma_px=nuisance[:, 1],
        dither_xy=nuisance[:, 2:],
        old_target_xy=old_xy,
        current_target_xy=current_xy,
        stable_source_xy=stable_xy,
        stable_source_flux=stable_flux,
        trail_mask=standalone_trail > 0.01 * standalone_trail.max(),
    )
    motion_arcsec_yr = float(
        np.hypot(
            config.proper_motion_ra_mas_yr, config.proper_motion_dec_mas_yr
        )
        / 1000
    )
    truth = {
        "purpose": (
            "Injection recovery validation only; not a measurement input."
        ),
        "physical_system": asdict(config),
        "old_target_xy": old_xy.tolist(),
        "current_target_xy": current_xy.tolist(),
        "displacement_xy_pixels": offset.tolist(),
        "proper_motion_arcsec_yr": motion_arcsec_yr,
        "transverse_velocity_km_s": 4.74047
        * motion_arcsec_yr
        * config.distance_pc,
        "contaminated_epoch": config.trail_epoch,
    }
    (output_dir / truth_paths["json"]).write_text(
        json.dumps(truth, indent=2) + "\n"
    )
    all_paths.extend(truth_paths.values())
    observed = {}
    (output_dir / "provenance").mkdir(parents=True, exist_ok=True)
    for source in inputs:
        relative = f"provenance/{source.name}"
        shutil.copy2(source, output_dir / relative)
        all_paths.append(relative)
    versions = {
        package: importlib.metadata.version(package)
        for package in ("numpy", "scipy", "astropy", "batman-package")
    }
    manifest = {
        "schema_version": 1,
        "files": files,
        "observed": observed,
        "calibration_dir": "provenance",
        "epochs": epochs,
        "shape": list(background.shape),
        "units": "nJy",
        "pixel_scale_arcsec": pixel_scale,
        "target_search": {"old_xy": old_xy.tolist(), "radius_pixels": 18},
        "assumptions": {
            "period_days": config.period_days,
            "a_over_rstar": config.a_over_rstar,
            "inclination_deg": config.inclination_deg,
            "eccentricity": config.eccentricity,
            "limb_darkening": [
                config.limb_darkening_u1,
                config.limb_darkening_u2,
            ],
            "stellar_radius_solar": config.stellar_radius_solar,
            "exposure_s": config.exposure_s,
            "baseline_years": config.baseline_years,
            "description": (
                "Supplied geometry priors; fit radius ratio and time."
            ),
        },
        "truth": truth_paths,
        "fingerprint": fingerprint,
        "dependencies": versions,
        "provenance": metadata,
        "sha256": {
            relative: sha256(output_dir / relative) for relative in all_paths
        },
        "simulation_notes": metadata.get("simulation_notes", []),
        "references": metadata.get("references", {}),
        "photometric_model": metadata.get("photometric_model", {}),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
