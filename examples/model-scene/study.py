# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Execute scene experiments and save data-only page diagnostics."""

import hashlib
import json
from pathlib import Path

import numpy as np
from experiment import calibrate, reconstruct
from measure import evaluate, paired_summary
from scene import (
    HELDOUT_FRAMES,
    TRAIN_FRAMES,
    generate,
    kernels,
    pixel_gaussian,
    science_window,
)
from scipy.signal import convolve2d


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run_case(
    root,
    arrays,
    truth,
    *,
    device,
    iterations,
    masked=False,
    reference_sigma=0.0,
):
    root = Path(root)
    root.mkdir(exist_ok=False, parents=True)
    images, runs = {}, {}
    for label, algorithm in [("l2", "l2"), ("huber", "rho")]:
        image, run, _ = reconstruct(
            root,
            arrays,
            name=label,
            algorithm=algorithm,
            masked=masked,
            device=device,
            iterations=iterations,
        )
        images[label] = image
        runs[label] = run
    result, products, target = evaluate(
        arrays, truth, images, reference_sigma=reference_sigma
    )
    np.savez_compressed(
        root / "page.npz",
        truth=target,
        coadd=products["coadd"],
        l2=products["l2"],
        huber=products["huber"],
        l2_native=images["l2"],
        huber_native=images["huber"],
        observations=science_window(arrays["images"][0]),
        variances=science_window(arrays["variances"][0]),
        kernels=arrays["psfs"],
        train_frames=np.array(TRAIN_FRAMES),
        l2_loss=np.load(runs["l2"].run_dir / "trace/loss.npy"),
        huber_loss=np.load(runs["huber"].run_dir / "trace/loss.npy"),
    )
    write_json(root / "measurements.json", result)
    return result


def validate_protocol(protocol):
    """Reject descriptive protocol fields that contradict this fixed model."""
    expected = {
        "huber_delta": 2.0,
        "stop_tolerance": 1e-12,
        "background_electrons_per_pixel": 2.0,
        "common_gaussian_sigma_pixels": 2.4,
        "aperture_radius_pixels": 3,
        "edge_exclusion_pixels": 5,
        "image_size": 120,
        "fit_size": 96,
        "science_window_size": 48,
        "science_window_origin": 36,
        "train_frames": list(TRAIN_FRAMES),
        "heldout_frames": list(HELDOUT_FRAMES),
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"Protocol contradicts fixed model: {key}")


def execute_study(root, protocol, *, device="cpu", quick=False):
    """Run a previously frozen protocol; evaluate truth after fitting."""
    validate_protocol(protocol)
    root = Path(root)
    if root.exists() or root.is_symlink():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    iterations = 10 if quick else protocol["iterations"]
    count = 2 if quick else protocol["noise_realizations"]
    write_json(root / "protocol.json", protocol)
    trials = []

    def make_data(seed, **kwargs):
        return generate(seed, development=quick, **kwargs)

    science_seed = (
        protocol["development_seeds"][0]
        if quick
        else protocol["test_seed_start"]
    )
    calibration_seed = 2000 if quick else protocol["calibration_seed_start"]
    example_seed = 2099 if quick else protocol["calibration_example_seed"]
    for index in range(count):
        arrays, truth = make_data(science_seed + index)
        trials.append(
            run_case(
                root / f"trial-{index:03}",
                arrays,
                truth,
                device=device,
                iterations=iterations,
            )
        )
        print(
            f"Completed paired noise realization {index + 1}/{count}",
            flush=True,
        )
        write_json(root / "trials.json", trials)
    write_json(root / "summary.json", paired_summary(trials))
    # Gaussian cases share noise; other noise models use separate draws.
    seed = science_seed
    for case in ("outliers", "photon-read", "transient"):
        arrays, truth = make_data(seed, case=case)
        run_case(
            root / case, arrays, truth, device=device, iterations=iterations
        )
        if case == "outliers":
            run_case(
                root / "explicit-mask",
                arrays,
                truth,
                device=device,
                iterations=iterations,
                masked=True,
            )
    for factor in (0.95, 1.05):
        arrays, truth = make_data(seed)
        arrays["psfs"] = kernels(truth["widths"] * factor)[:, :, None, :]
        run_case(
            root / f"psf-width-{factor}",
            arrays,
            truth,
            device=device,
            iterations=iterations,
        )
    arrays, truth = make_data(seed)
    arrays["variances"] *= 4
    run_case(
        root / "variance-times-four",
        arrays,
        truth,
        device=device,
        iterations=iterations,
    )
    arrays, truth = make_data(seed, compact=True)
    run_case(
        root / "boundary-failure",
        arrays,
        truth,
        device=device,
        iterations=iterations,
    )
    arrays, truth = make_data(seed, matched=True)
    run_case(
        root / "matched-operator",
        arrays,
        truth,
        device=device,
        iterations=iterations,
    )
    arrays, truth = make_data(seed)
    arrays["images"] = truth["expectation"][None].copy()
    run_case(
        root / "noiseless-physical",
        arrays,
        truth,
        device=device,
        iterations=iterations,
    )
    # PSF factorization is separate from estimating the observed star stamp.
    calibration_root = root / "calibration"
    calibration_root.mkdir()
    calibration_report = {}
    for method in ("mm", "adam"):
        run, measured = calibrate(
            calibration_root,
            method=method,
            device=device,
            seed=example_seed,
        )
        solved = np.load(run.run_dir / "artifacts/psfs.npy")
        np.save(calibration_root / f"{method}-measured.npy", measured)
        arrays, truth = make_data(seed)
        arrays["psfs"] = solved
        if method == "mm":
            run_case(
                root / "calibrated-mm",
                arrays,
                truth,
                device=device,
                iterations=iterations,
                reference_sigma=0.55,
            )
        # Adam permits negative kernels. Keep this factorization diagnostic
        # separate from multiplicative reconstruction with positive kernels.
        reference = np.load(
            run.run_dir / "artifacts/gaussian_psf.npy"
        ).squeeze()
        reproduced = np.stack(
            [
                convolve2d(solved[:, :, 0, i], reference, mode="same")
                for i in range(12)
            ],
            axis=-1,
        )
        np.save(calibration_root / f"{method}-reproduced.npy", reproduced)
        y, x = np.indices(solved.shape[:2])
        total = solved.sum(axis=(0, 1, 2))
        centroid_x = (solved[:, :, 0] * x[..., None]).sum(axis=(0, 1)) / total
        centroid_y = (solved[:, :, 0] * y[..., None]).sum(axis=(0, 1)) / total
        calibration_report[method] = {
            "target_reproduction_rms_by_frame": np.sqrt(
                np.mean((reproduced - measured) ** 2, axis=(0, 1))
            ).tolist(),
            "centroid_xy_by_frame": np.stack(
                [centroid_x, centroid_y], axis=-1
            ).tolist(),
            "minimum": float(solved.min()),
            "sums": solved.sum(axis=(0, 1, 2)).tolist(),
            "shape": list(solved.shape),
        }
    # Noise-free integrated stamps isolate finite calibration uncertainty.
    exact_root = calibration_root / "perfect"
    exact_root.mkdir()
    run, _ = calibrate(
        exact_root,
        device=device,
        target_override=np.stack(
            [
                pixel_gaussian(15, 7, 7, width)
                for width in np.linspace(0.9, 1.7, 12)
            ],
            axis=-1,
        ),
    )
    arrays, truth = make_data(seed)
    arrays["psfs"] = np.load(run.run_dir / "artifacts/psfs.npy")
    arrays["images"] = truth["expectation"][None].copy()
    perfect_result = run_case(
        root / "perfect-factorization",
        arrays,
        truth,
        device=device,
        iterations=iterations,
        reference_sigma=0.55,
    )
    finite_calibration = []
    for index in range(2 if quick else 50):
        draw_root = calibration_root / f"draw-{index:03}"
        draw_root.mkdir()
        run, measured = calibrate(
            draw_root,
            seed=calibration_seed + index,
            device=device,
        )
        arrays, truth = make_data(seed)
        arrays["images"] = truth["expectation"][None].copy()
        arrays["psfs"] = np.load(run.run_dir / "artifacts/psfs.npy")
        result = run_case(
            draw_root / "scene",
            arrays,
            truth,
            device=device,
            iterations=iterations,
            reference_sigma=0.55,
        )
        for method in ("l2", "huber"):
            for measured, reference in zip(
                result[method]["sources"],
                perfect_result[method]["sources"],
                strict=True,
            ):
                measured["calibration_flux_error_increment"] = (
                    measured["fractional_flux_error"]
                    - reference["fractional_flux_error"]
                )
        finite_calibration.append(result)
        write_json(
            calibration_root / "finite-calibration.json", finite_calibration
        )
        print(f"Calibration draw {index + 1}", flush=True)
    write_json(calibration_root / "diagnostics.json", calibration_report)
    hashes = {
        p.relative_to(root).as_posix(): hashlib.sha256(
            p.read_bytes()
        ).hexdigest()
        for p in root.rglob("page.npz")
    }
    write_json(root / "page-hashes.json", hashes)
    return paired_summary(trials)
