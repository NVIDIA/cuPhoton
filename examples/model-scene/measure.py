# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Declared evaluation protocol; never used by the reconstruction solver."""

import numpy as np
from experiment import aperture_measurements, forward_prediction
from scene import (
    HELDOUT_FRAMES,
    TRAIN_FRAMES,
    catalog,
    render,
    science_window,
)
from scipy.ndimage import gaussian_filter

TARGET_SIGMA = 2.4


def comparison_products(arrays, truth, reconstructions, reference_sigma=0.0):
    """Bring all products to a Gaussian target broader than every input PSF.

    Coadd is an ordinary arithmetic mean. For its comparison version, blur
    each constituent exposure to the common target before averaging. This
    avoids treating the mixture of exposure PSFs as a single Gaussian.
    Discrete Gaussian filtering approximates continuous smoothing; measure
    its noiseless discrepancy explicitly rather than claiming exact equality.
    """
    widths = truth["widths"][list(TRAIN_FRAMES)]
    data = arrays["images"][0]
    common_coadd = science_window(
        np.mean(
            [
                gaussian_filter(
                    data[..., TRAIN_FRAMES[i]],
                    np.sqrt(TARGET_SIGMA**2 - width**2),
                    mode="constant",
                    cval=2,
                )
                for i, width in enumerate(widths)
            ],
            axis=0,
        )
    )
    products = {"coadd": common_coadd}
    for name, image in reconstructions.items():
        expected_size = arrays["images"].shape[1] - 10
        if image.shape != (expected_size, expected_size):
            raise ValueError(
                f"Unexpected VALID restoration shape {image.shape}"
            )
        products[name] = science_window(
            gaussian_filter(
                image,
                np.sqrt(TARGET_SIGMA**2 - reference_sigma**2),
                mode="constant",
                cval=2,
            )
        )
    return products


def evaluate(
    arrays, truth, reconstructions, *, development=None, reference_sigma=0.0
):
    """Compare apertures and frame predictions with independent truth."""
    if development is None:
        development = truth.get("development", False)
    sources = catalog(development, offset=truth["source_offset"])
    origin = truth["view_origin"]
    target = science_window(
        gaussian_filter(truth["truth"], TARGET_SIGMA, mode="constant", cval=2)
        if truth.get("matched", False)
        else render(
            sources, sigma=TARGET_SIGMA, size=arrays["images"].shape[1]
        )
    )
    if not truth.get("boundary_failure", False):
        last_fit_pixel = arrays["images"].shape[1] - 13
        for source in sources:
            margin = min(
                source.x - 12,
                source.y - 12,
                last_fit_pixel - source.x,
                last_fit_pixel - source.y,
            )
            if margin < 13:
                raise ValueError(
                    "Insufficient fitting context for blur and aperture"
                )
    expected = aperture_measurements(target, sources, origin=origin)
    products = comparison_products(
        arrays, truth, reconstructions, reference_sigma=reference_sigma
    )
    results = {}
    for name, image in products.items():
        measured = aperture_measurements(image, sources, origin=origin)
        errors = []
        for got, reference in zip(measured, expected, strict=True):
            errors.append(
                {
                    "kind": reference["kind"],
                    "input_flux": reference["input_flux"],
                    "fractional_flux_error": got["flux"] / reference["flux"]
                    - 1,
                    "centroid_error_px": float(
                        np.hypot(
                            got["x"] - reference["x"],
                            got["y"] - reference["y"],
                        )
                    ),
                    "profile_error": (
                        np.asarray(got["profile"]) - reference["profile"]
                    ).tolist(),
                }
            )
        blend = [row for row in measured if row["kind"] == "blend"]
        source_blend = [
            source for source in sources if source.kind == "blend"
        ]
        separation = float(
            np.hypot(
                blend[0]["x"] - blend[1]["x"], blend[0]["y"] - blend[1]["y"]
            )
        )
        intrinsic = float(
            np.hypot(
                source_blend[0].x - source_blend[1].x,
                source_blend[0].y - source_blend[1].y,
            )
        )
        results[name] = {
            "sources": errors,
            "blend_aperture_centroid_separation_px": separation,
            "intrinsic_blend_separation_px": intrinsic,
            "separation_error_px": separation - intrinsic,
            "separation_estimator": "overlapping aperture centroid distance",
        }
        if name in reconstructions:
            prediction = forward_prediction(
                reconstructions[name], arrays["psfs"]
            )
            full_residual = arrays["images"][0, 12:-12, 12:-12] - prediction
            results[name]["context_prediction_mse_by_frame"] = np.mean(
                full_residual**2, axis=(0, 1)
            ).tolist()
            residual = science_window(full_residual)
            variance = science_window(arrays["variances"][0])
            results[name]["prediction_mse_by_frame"] = np.mean(
                residual**2, axis=(0, 1)
            ).tolist()
            results[name]["standardized_residual_ms_by_frame"] = np.mean(
                residual**2 / variance, axis=(0, 1)
            ).tolist()
            results[name]["heldout_mean_residual"] = float(
                residual[..., list(HELDOUT_FRAMES)].mean()
            )
    return results, products, target


def paired_summary(trials):
    """Estimate uncertainty over independent realizations.

    Aggregate source classes within each realization first.
    sources within a trial are not treated as independent replicates.
    """
    summary = {}
    for method in trials[0]:
        summary[method] = {}
        for kind in ("isolated", "blend", "extended"):
            values = np.array(
                [
                    np.mean(
                        [
                            source["fractional_flux_error"]
                            for source in trial[method]["sources"]
                            if source["kind"] == kind
                        ]
                    )
                    for trial in trials
                ]
            )
            summary[method][kind] = {
                "mean_fractional_bias": float(values.mean()),
                "realization_scatter": float(values.std(ddof=1)),
                "bias_standard_error": float(
                    values.std(ddof=1) / np.sqrt(len(values))
                ),
                "n_independent_noise_realizations": len(values),
            }
    for method in trials[0]:
        summary[method]["by_source"] = []
        for index, source in enumerate(trials[0][method]["sources"]):
            errors = np.array(
                [
                    trial[method]["sources"][index]["fractional_flux_error"]
                    for trial in trials
                ]
            )
            standard_error = float(errors.std(ddof=1) / np.sqrt(len(errors)))
            summary[method]["by_source"].append(
                {
                    "kind": source["kind"],
                    "input_flux": source["input_flux"],
                    "mean_fractional_bias": float(errors.mean()),
                    "scatter": float(errors.std(ddof=1)),
                    "bias_standard_error": standard_error,
                    "approx_80pct_power_detectable_bias": 2.8
                    * standard_error,
                }
            )
    for method in ("l2", "huber"):
        summary[method]["forward_prediction"] = {}
        if "standardized_residual_ms_by_frame" in trials[0][method]:
            for label, indices in (
                ("fitted", TRAIN_FRAMES),
                ("heldout", HELDOUT_FRAMES),
            ):
                values = np.array(
                    [
                        np.mean(
                            np.asarray(
                                t[method]["standardized_residual_ms_by_frame"]
                            )[list(indices)]
                        )
                        for t in trials
                    ]
                )
                summary[method]["forward_prediction"][label] = {
                    "mean_standardized_residual_ms": float(values.mean()),
                    "standard_error": float(
                        values.std(ddof=1) / np.sqrt(len(values))
                    ),
                }
        differences = np.array(
            [
                np.mean(
                    [
                        s["fractional_flux_error"]
                        for s in trial[method]["sources"]
                    ]
                )
                - np.mean(
                    [
                        s["fractional_flux_error"]
                        for s in trial["coadd"]["sources"]
                    ]
                )
                for trial in trials
            ]
        )
        summary[method]["paired_difference_vs_coadd"] = {
            "mean": float(differences.mean()),
            "standard_error": float(
                differences.std(ddof=1) / np.sqrt(len(differences))
            ),
        }
    return summary
