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
from scene import generate, kernels
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
        observations=arrays["images"][0, 12:60, 12:60],
        variances=arrays["variances"][0, 12:60, 12:60],
        kernels=arrays["psfs"],
        l2_loss=np.load(runs["l2"].run_dir / "trace/loss.npy"),
        huber_loss=np.load(runs["huber"].run_dir / "trace/loss.npy"),
    )
    write_json(root / "measurements.json", result)
    return result


def execute_study(root, protocol, *, device="cpu", quick=False):
    """Run a previously frozen protocol; evaluate truth after fitting."""
    root = Path(root)
    if root.exists() or root.is_symlink():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    iterations = 10 if quick else protocol["iterations"]
    count = 2 if quick else protocol["noise_realizations"]
    write_json(root / "protocol.json", protocol)
    trials = []
    for index in range(count):
        arrays, truth = generate(protocol["test_seed_start"] + index)
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
    # Same first noise realization is paired across every stress case.
    seed = protocol["test_seed_start"]
    for case in ("outliers", "photon-read", "transient"):
        arrays, truth = generate(seed, case=case)
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
        arrays, truth = generate(seed)
        arrays["psfs"] = kernels(truth["widths"] * factor)[:, :, None, :]
        run_case(
            root / f"psf-width-{factor}",
            arrays,
            truth,
            device=device,
            iterations=iterations,
        )
    arrays, truth = generate(seed)
    arrays["variances"] *= 4
    run_case(
        root / "variance-times-four",
        arrays,
        truth,
        device=device,
        iterations=iterations,
    )
    arrays, truth = generate(seed, matched=True)
    run_case(
        root / "matched-operator",
        arrays,
        truth,
        device=device,
        iterations=iterations,
    )
    arrays, truth = generate(seed)
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
            calibration_root, method=method, device=device
        )
        solved = np.load(run.run_dir / "artifacts/psfs.npy")
        np.save(calibration_root / f"{method}-measured.npy", measured)
        arrays, truth = generate(seed)
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
    # Exact-kernel factorization isolates reference-solve error.
    exact_root = calibration_root / "perfect"
    exact_root.mkdir()
    run, _ = calibrate(
        exact_root,
        device=device,
        target_override=kernels(np.linspace(0.9, 1.7, 12)),
    )
    arrays, truth = generate(seed)
    arrays["psfs"] = np.load(run.run_dir / "artifacts/psfs.npy")
    run_case(
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
            draw_root, seed=45000 + index, device=device
        )
        arrays, truth = generate(seed)
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
