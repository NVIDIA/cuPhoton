# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Visible workflow adapters; truth is used only by evaluation functions."""

from pathlib import Path

import numpy as np
import yaml
from scene import pixel_gaussian, write_bundle
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d

ROI = (slice(12, 60), slice(12, 60))


def reconstruct(
    root,
    arrays,
    *,
    name,
    algorithm="l2",
    masked=False,
    device="cpu",
    iterations=75,
    delta=2.0,
):
    """Fit eight frames and save artifacts and secondary diagnostics."""
    from cuphoton.ximage.data import load_ximage_bundle, validate_bundle
    from cuphoton.ximage.evaluation import evaluate_reconstruction_run
    from cuphoton.ximage.workflows import run_reconstruction

    root = Path(root)
    bundle_path = root / f"{name}-bundle"
    write_bundle(bundle_path, arrays)
    with load_ximage_bundle(bundle_path) as bundle:
        errors = validate_bundle(bundle)
        if errors:
            raise ValueError(errors)
    config = {
        "backend": "torch",
        "device": device,
        "dtype": "float64",
        "algorithm": algorithm,
        "data": {
            "base": str(bundle_path.resolve()),
            "exposure_indices": list(range(8)),
        },
        "output": {"root_dir": str((root / "runs").resolve()), "name": name},
        "roi": {"height": 48, "width": 48},
        "mask": {"mode": "provided" if masked else "none"},
        "padding": "VALID",
        "srf": 1,
        "max_iter": iterations,
        "stop_tol": 1e-12,
        "rho_loss": "huber",
        "delta_rho_start": delta,
        "delta_rho_end": delta,
    }
    config_path = root / f"{name}.yaml"
    config_path.write_text(yaml.safe_dump(config))
    run = run_reconstruction(config_path)
    secondary = evaluate_reconstruction_run(
        run.run_dir, metrics_backend="cpu", write_artifacts=True
    )
    image = np.load(run.run_dir / "artifacts/image.npy").squeeze()
    return image, run, secondary


def calibrate(
    root, *, seed=300, method="mm", device="cpu", target_override=None
):
    """Measure separate calibration stars and solve reference PSFs.

    Unit-flux stars have known locations and total flux 100,000 electrons.
    Gaussian read noise is independent of every science noise realization.
    These star stamps are the observations; analytic widths are never passed
    to the calibration solver. Background is known and subtracted. Negative
    noisy samples are clipped before unit-sum normalization; quantify this
    processing bias against the perfect-kernel control.
    """
    from cuphoton.ximage.workflows import run_psf_solve

    rng = np.random.default_rng(seed)
    widths = np.linspace(0.9, 1.7, 12)
    stars = np.stack(
        [pixel_gaussian(15, 7, 7, width) for width in widths], axis=-1
    )
    counts = stars * 100000 + rng.normal(0, 2, stars.shape)
    measured = np.maximum(counts, 0)
    measured /= measured.sum(axis=(0, 1), keepdims=True)
    if target_override is not None:
        measured = np.asarray(target_override, dtype=float).copy()
    path = Path(root) / f"calibration-{method}"
    # Loader requires only the explicitly requested PSF artifact here.
    path.mkdir(parents=True, exist_ok=False)
    np.save(path / "psfs.npy", measured[:, :, None, :])
    config = {
        "backend": "torch",
        "device": device,
        "dtype": "float64",
        "method": method,
        "data": {"base": str(path.resolve())},
        "output": {
            "root_dir": str((Path(root) / "runs").resolve()),
            "name": f"calibration-{method}",
        },
        "sigma_g": 0.55,
        "padding": "VALID",
        "max_iter": 100,
        "num_steps": 500,
        "lr": 0.01,
    }
    cfg = Path(root) / f"calibration-{method}.yaml"
    cfg.write_text(yaml.safe_dump(config))
    run = run_psf_solve(cfg)
    return run, measured


def common_resolution(image, sigma):
    """Blur to a declared attainable Gaussian target."""
    return gaussian_filter(image, sigma, mode="constant", cval=2)


def aperture_measurements(image, sources, *, origin=12, radius=3):
    """Fixed circular-aperture flux, centroid and radial light profile.

    Flux includes neighbors inside the aperture and excludes known constant
    background. Compare with the same aperture on the appropriate target
    truth, rather than labeling this a deblended total source flux. Exclude
    apertures less than radius+2 from the measurement boundary.
    """
    y, x = np.indices(image.shape)
    rows = []
    for source in sources:
        cx, cy = source.x - origin, source.y - origin
        if (
            min(cx, cy, image.shape[1] - 1 - cx, image.shape[0] - 1 - cy)
            < radius + 2
        ):
            continue
        r = np.hypot(x - cx, y - cy)
        light = (image - 2) * (r <= radius)
        flux = light.sum()
        rows.append(
            {
                "kind": source.kind,
                "input_flux": source.flux,
                "flux": float(flux),
                "x": float((light * x).sum() / flux + origin),
                "y": float((light * y).sum() / flux + origin),
                "profile": [
                    float((image - 2)[(r >= a) & (r < a + 1)].mean())
                    for a in range(5)
                ],
            }
        )
    return rows


def forward_prediction(image, psfs):
    """Predict the valid interior using real surrounding pixels."""
    return np.stack(
        [
            convolve2d(image, psfs[:, :, 0, i], mode="valid")
            for i in range(12)
        ],
        axis=-1,
    )
