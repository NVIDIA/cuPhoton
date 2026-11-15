# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Declared evaluation protocol; never used by the reconstruction solver."""

import numpy as np
from experiment import aperture_measurements, forward_prediction
from scene import catalog, render
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
    widths = truth["widths"][:8]
    data = arrays["images"][0]
    common_coadd = np.mean(
        [
            gaussian_filter(
                data[..., i],
                np.sqrt(TARGET_SIGMA**2 - width**2),
                mode="constant",
                cval=2,
            )
            for i, width in enumerate(widths)
        ],
        axis=0,
    )[12:60, 12:60]
    products = {"coadd": common_coadd}
    for name, image in reconstructions.items():
        if image.shape != (62, 62):
            raise ValueError(
                f"Unexpected VALID restoration shape {image.shape}"
            )
        products[name] = gaussian_filter(
            image,
            np.sqrt(TARGET_SIGMA**2 - reference_sigma**2),
            mode="constant",
            cval=2,
        )[7:-7, 7:-7]
    return products


def evaluate(
    arrays, truth, reconstructions, *, development=False, reference_sigma=0.0
):
    """Compare apertures and frame predictions with independent truth."""
    sources = catalog(development)
    target = (
        gaussian_filter(
            truth["truth"], TARGET_SIGMA, mode="constant", cval=2
        )[12:60, 12:60]
        if truth.get("matched", False)
        else render(sources, sigma=TARGET_SIGMA)[12:60, 12:60]
    )
    expected = aperture_measurements(target, sources)
    products = comparison_products(
        arrays, truth, reconstructions, reference_sigma=reference_sigma
    )
    results = {}
    for name, image in products.items():
        measured = aperture_measurements(image, sources)
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
            residual = arrays["images"][0, 12:60, 12:60] - prediction
            variance = arrays["variances"][0, 12:60, 12:60]
            results[name]["prediction_mse_by_frame"] = np.mean(
                residual**2, axis=(0, 1)
            ).tolist()
            results[name]["standardized_residual_ms_by_frame"] = np.mean(
                residual**2 / variance, axis=(0, 1)
            ).tolist()
            results[name]["heldout_mean_residual"] = float(
                residual[..., 8:].mean()
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
