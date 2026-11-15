# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent analytic sky and instrument for Model Sky.

No downloads, observed catalogs, measured PSFs or empirical calibrations.
Only preparation reads the physical model; pipeline.py measures image planes.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.constants import R_sun, c, h, k_B, pc
from astropy.io import fits
from astropy.wcs import WCS
from scene import SceneConfig as BaseConfig
from scene import _source_positions, add_star, prepare_bundle, sha256


def brightness():
    """Photon-weighted AB flux and electrons, using SI Planck radiance."""
    wavelength = np.linspace(560e-9, 700e-9, 4097)
    response = np.full(wavelength.shape, 0.4)
    dilution = (0.2162 * R_sun.value / (146.42 * pc.value)) ** 2
    surface = (2 * h.value * c.value**2 / wavelength**5) / np.expm1(
        h.value * c.value / (wavelength * k_B.value * 3100)
    )
    flam = np.pi * surface * dilution  # W m^-2 m^-1 at Earth
    fnu = np.trapezoid(flam * wavelength * response, wavelength) / (
        c.value * np.trapezoid(response / wavelength, wavelength)
    )
    area = np.pi * (4.0 / 2) ** 2  # modeled unobstructed 4 m telescope
    electrons_per_njy = (
        area
        * 30
        * 1e-35
        / h.value
        * np.trapezoid(response / wavelength, wavelength)
    )
    return {
        "flux_njy": float(fnu / 1e-35),
        "ab_magnitude": float(-2.5 * np.log10(fnu / (3631e-26))),
        "electrons_per_njy": float(electrons_per_njy),
        "nJy_per_electron": float(1 / electrons_per_njy),
        "temperature_k": 3100,
        "radius_solar": 0.2162,
        "distance_pc": 146.42,
        "effective_area_m2": area,
        "exposure_s": 30,
        "bandpass": "modeled 560-700 nm top hat; total efficiency 0.4",
        "spectrum": "Planck blackbody; pi B_lambda (R/d)^2; zero extinction",
        "limb_darkening": [0.6, 0.15],
        "limb_darkening_scope": (
            "Explicit illustrative quadratic intensity law; not an "
            "atmosphere table or derived from the blackbody."
        ),
        "limitations": (
            "An M dwarf has molecular absorption absent from a "
            "blackbody. This is a declared idealized instrument and "
            "star, not calibrated r-band photometry or a prediction for"
            " GJ 1214."
        ),
    }


@dataclass(frozen=True)
class SceneConfig(BaseConfig):
    target_flux_njy: float = brightness()["flux_njy"]
    limb_darkening_u1: float = 0.6
    limb_darkening_u2: float = 0.15


def prepare(output, *, force=False):
    output = Path(output)
    if (
        output.exists()
        and not force
        and (not output.is_dir() or any(output.iterdir()))
    ):
        raise ValueError(
            "output must be absent or empty; use --force to regenerate"
        )
    calibration = output / "provenance"
    calibration.mkdir(parents=True, exist_ok=True)
    audit = brightness()
    shape = (480, 480)
    y, x = np.indices((41, 41))
    psf = np.exp(-0.5 * ((x - 20) ** 2 + (y - 20) ** 2) / 1.5**2)
    psf /= psf.sum()
    background = np.zeros(shape)
    bad = np.zeros(shape, np.uint8)
    # Reserve the same geometric anchor regions for injected target/standards.
    anchors = _source_positions(background, bad)
    rng = np.random.default_rng(260630)
    positions = []
    for _ in range(120):
        point = rng.uniform(36, 444, 2)
        if np.min(np.linalg.norm(anchors - point, axis=1)) < 48:
            continue
        if (
            positions
            and np.min(np.linalg.norm(np.array(positions) - point, axis=1))
            < 24
        ):
            continue
        positions.append(point)
        add_star(
            background,
            psf,
            point,
            audit["flux_njy"] * rng.uniform(0.25, 0.95),
        )
    # Sky of 20.5 AB mag/arcsec^2 in the modeled band; mean sky subtracted.
    sky_njy_pixel = 3631e9 * 10 ** (-0.4 * 20.5) * 0.2**2
    noise_scale = audit["nJy_per_electron"]
    sky_variance = sky_njy_pixel * noise_scale + (5 * noise_scale) ** 2
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [240.5, 240.5]
    wcs.wcs.crval = [210, -10]
    wcs.wcs.cdelt = [-0.2 / 3600, 0.2 / 3600]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    np.savez_compressed(
        calibration / "calibration.npz",
        background=background,
        base_psf=psf,
        sky_variance=np.full(shape, sky_variance),
        noise_scale=np.full(shape, noise_scale),
        quality_mask=bad,
    )
    fits.PrimaryHDU(
        background.astype(np.float32), header=wcs.to_header()
    ).writeto(calibration / "calibration.fits", overwrite=True, checksum=True)
    metadata = {
        "edition": "Model Sky",
        "modeled_scene": True,
        "quality_mask_is_binary": True,
        "pixel_scale_arcsec": 0.2,
        "origin": "Analytic models in model_sky.py; fixed RNG seed 260630",
        "license": "NVIDIA-generated modeled arrays; Apache-2.0",
        "catalog": "Uniform isolated fictional stars; no observed catalog",
        "wcs": (
            "Declared TAN grid at RA 210 deg, Dec -10 deg, 0.2 "
            "arcsec/pixel; no observed solution"
        ),
        "psf": (
            "Normalized analytic Gaussian sigma 1.5 pixels; identical "
            "base for background and target, common seeing blur per "
            "exposure"
        ),
        "sky_ab_mag_arcsec2": 20.5,
        "read_noise_electrons": 5,
        "background_stars": len(positions),
        "photometric_model": audit,
        "simulation_notes": [
            (
                "All pixels, WCS, stars, masks, PSFs, variance and "
                "throughput are generated independently of "
                "observational data."
            ),
            (
                "30 second exposures once per minute, 120 epochs; "
                "separate old/current 16-exposure references and a "
                "ten-year proper-motion baseline."
            ),
            (
                "Field, target and comparison stars share analytic base"
                " PSF and per-exposure seeing, transparency and dither."
            ),
            (
                "Source Poisson electrons and independent Gaussian "
                "sky/read noise: variance [nJy^2] = a * source [nJy] + "
                "sky_variance; a is nJy/electron. References divide "
                "variance by 16."
            ),
            (
                "Sky Gaussian approximation applies to the high "
                "sky-electron count; sky mean is subtracted. Constant "
                "gain, no saturation, no scintillation or color terms "
                "are modeled."
            ),
            (
                "The fixed latent sky is noiseless; independent "
                "exposure noise is sampled after transformation. Truth "
                "is written separately; empirical PSFs and comparisons "
                "are selected from images."
            ),
            (
                "Transit uses batman, 11 subexposures per 30 second "
                "integration, declared quadratic limb darkening "
                "(0.6,0.15), circular orbit and GJ 1214-inspired "
                "geometry. This is not an observed planet discovery."
            ),
            (
                "Trail is an analytic double-lobed teaching model with "
                "width 2.8 arcsec, not an instrument measurement or "
                "satellite ray trace."
            ),
            (
                "Blackbody flux is computed at 146.42 pc, radius 0.2162"
                " solar, 3100 K, through modeled 560-700 nm response. "
                "Molecular absorption is omitted; no empirical r-band "
                "calibration is claimed."
            ),
            (
                "xScan omitted: no trained classifier or performance "
                "claim is included."
            ),
        ],
        "references": {
            "gj1214_discovery": "https://arxiv.org/abs/0912.3229",
            "system_parameters": "https://arxiv.org/html/2402.05991v2",
            "batman": "https://arxiv.org/abs/1507.08285",
            "subtraction": "https://arxiv.org/abs/astro-ph/9712287",
        },
        "generator_sha256": sha256(Path(__file__)),
    }
    (calibration / "calibration.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    # Stage beside the output so provenance copies use distinct paths.
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        for path in calibration.glob("calibration.*"):
            shutil.copy2(path, Path(temporary) / path.name)
        manifest = prepare_bundle(
            temporary, output, config=SceneConfig(), force=force
        )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    prepare(args.output, force=args.force)
    print(json.dumps(brightness(), indent=2))
