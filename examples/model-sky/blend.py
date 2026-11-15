# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Optional experiment: locate the fading light in an unresolved blend.

Preparation owns injection settings. Measurement opens only the generated
science/variance/mask planes and locates sources from image evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from bokeh.layouts import column
from bokeh.models import Div
from bokeh.plotting import figure
from pipeline import (
    bundle_path,
    empirical_psf,
    fit_positions,
    good_pixels,
    gpu,
    load_bundle,
    local_sky,
    read_exposure,
    select_stars,
    stamp,
    write_exposure,
)
from scene import (
    SceneConfig,
    add_star,
    draw_exposure,
    normalized_psf,
    transit_flux,
)
from scipy import ndimage
from views import export_html, image_triptych

from cuphoton.xfit import fit_single_sources, forced_psf_photometry


def _brightest_near(exposure, seed, radius=18):
    patch = stamp(exposure["science"], seed, 2 * radius + 1)
    usable = stamp(good_pixels(exposure), seed, 2 * radius + 1)
    y, x = np.unravel_index(
        np.argmax(np.where(usable, patch, -np.inf)), patch.shape
    )
    return np.rint(seed) + [x - radius, y - radius]


def _prepare(bundle, output, config=None):
    """Construct independent in/out means; write truth for validation only."""
    root, manifest = load_bundle(bundle)
    reference = read_exposure(
        bundle_path(root, manifest["files"]["reference_current"])
    )
    target_guess = _brightest_near(
        reference, manifest["target_search"]["old_xy"]
    )
    target_xy, _, ok = fit_positions(reference, np.array([target_guess]))
    if not ok[0]:
        raise RuntimeError(
            "the independent reference does not locate the target"
        )
    target_xy = target_xy[0]
    # Three bright, isolated image-selected stars supply independent PSF data.
    controls = select_stars(reference, exclude=[target_xy], count=3)
    controls, _, controls_ok = fit_positions(reference, controls)
    if not controls_ok.all():
        raise RuntimeError(
            "the reference does not constrain three control stars"
        )
    calibration_dir = bundle_path(
        root, manifest.get("calibration_dir", "provenance")
    )
    with np.load(calibration_dir / "calibration.npz") as arrays:
        background = arrays["background"].astype(float)
        psf = normalized_psf(arrays["base_psf"], 1.4)
        sky_variance = arrays["sky_variance"]
        noise_scale = arrays["noise_scale"]
        bad = arrays["quality_mask"].astype(np.uint8)
    config = config or SceneConfig()
    target_flux = config.target_flux_njy
    neighbor_flux = target_flux * 0.05
    neighbor_xy = target_xy + [1.5, 0.7]
    loss = target_flux * (1 - float(transit_flux([0], config)[0]))
    base = ndimage.gaussian_filter(background, 1.4, mode="reflect")
    for index, point in enumerate(controls):
        add_star(base, psf, point, target_flux * (0.8 + 0.1 * index))
    out_model = base.copy()
    add_star(out_model, psf, target_xy, target_flux)
    add_star(out_model, psf, neighbor_xy, neighbor_flux)
    cases = {}
    rng = np.random.default_rng(260614)
    output.mkdir(parents=True, exist_ok=True)
    for name, position in (("planet", target_xy), ("blend", neighbor_xy)):
        in_model = out_model.copy()
        add_star(in_model, psf, position, -loss)
        paths = {}
        for state, model in (("out", out_model), ("in", in_model)):
            data, variance = draw_exposure(
                model, sky_variance, noise_scale, rng, n_average=16
            )
            header = reference["header"].copy()
            for key in ("SIMMJD", "MJD-OBS", "DATE-OBS", "TREL_MIN"):
                header.remove(key, ignore_missing=True)
            header["SIMULATE"] = True
            header["NCOMBINE"] = 16
            header["TOTALEXP"] = 480.0
            header["SIMSTATE"] = state
            header.add_history(
                "Independent 16-exposure mean at a held light level."
            )
            path = output / name / f"{state}.fits"
            write_exposure(
                path,
                {
                    "science": data,
                    "variance": variance,
                    "mask": bad,
                    "header": header,
                },
            )
            paths[state] = str(path.relative_to(output))
        cases[name] = paths
    definition = {
        "schema_version": 1,
        "cases": cases,
        "target_search": manifest["target_search"],
        "pixel_scale_arcsec": manifest["pixel_scale_arcsec"],
        "description": (
            "Independent 16-exposure means at fixed in/out light levels, "
            "with equal lost total flux in both scenarios. These are "
            "controlled snapshots, not a time-sampled eclipse model."
        ),
        "references": {
            "centroid_validation": "https://arxiv.org/abs/1303.0052",
            "gj1214_discovery": "https://arxiv.org/abs/0912.3229",
        },
    }
    (output / "cases.json").write_text(
        json.dumps(definition, indent=2) + "\n"
    )
    (output / "injection-truth.json").write_text(
        json.dumps(
            {
                "purpose": "Validation only; never read by measure_case.",
                "target_xy": target_xy.tolist(),
                "neighbor_xy": neighbor_xy.tolist(),
                "target_flux_njy": target_flux,
                "neighbor_flux_njy": neighbor_flux,
                "lost_flux_njy": loss,
                "neighbor_eclipse_fraction": loss / neighbor_flux,
                "combined_aperture_depth": loss
                / (target_flux + neighbor_flux),
            },
            indent=2,
        )
        + "\n"
    )
    return definition


def measure_case(out_path, in_path, seed):
    """Use image evidence and real xFit GPU fits, without injection truth."""
    cp = gpu()
    out_frame, in_frame = read_exposure(out_path), read_exposure(in_path)
    guess = _brightest_near(out_frame, seed)
    center = np.rint(guess)
    out_stamp = stamp(out_frame["science"], center)
    in_stamp = stamp(in_frame["science"], center)
    out_stamp -= local_sky(out_stamp)
    in_stamp -= local_sky(in_stamp)
    variance_out = stamp(out_frame["variance"], center)
    variance_in = stamp(in_frame["variance"], center)
    valid_out = stamp(good_pixels(out_frame), center)
    valid_in = stamp(good_pixels(in_frame), center)
    difference = in_stamp - out_stamp
    variance_difference = variance_in + variance_out
    images = np.stack([out_stamp, in_stamp, difference])
    variances = np.stack([variance_out, variance_in, variance_difference])
    usable = np.stack([valid_out, valid_in, valid_out & valid_in])
    fit = fit_single_sources(
        cp.asarray(images),
        cp.asarray(variances),
        cp.zeros((3, 2)),
        mask=cp.asarray(usable),
        sign=cp.asarray([1, 1, -1]),
    )
    if not bool(cp.all(fit.converged & fit.uncertainty_valid)):
        raise RuntimeError(
            "xFit did not constrain the three source centroids"
        )
    measured_xy = center + cp.asnumpy(cp.stack([fit.x, fit.y], axis=1))
    errors_xy = cp.asnumpy(
        cp.stack([fit.position_error_x, fit.position_error_y], axis=1)
    )
    # Build the forced-photometry model from independent isolated field stars.
    anchors = select_stars(out_frame, exclude=[measured_xy[0]], count=3)
    empirical = empirical_psf(out_frame, anchors)
    psf_model = ndimage.shift(
        empirical, (measured_xy[2] - center)[::-1], order=3
    )
    flux_fit = forced_psf_photometry(
        cp.asarray(difference[None]),
        cp.asarray(variance_difference[None]),
        cp.asarray((measured_xy[2] - center)[None]),
        1,
        mask=cp.asarray((valid_out & valid_in)[None]),
        psf=cp.asarray(psf_model[None]),
    )
    if not bool(flux_fit.valid[0]):
        raise RuntimeError("forced photometry has no usable information")
    yy, xx = np.indices(out_stamp.shape)
    aperture = (xx - 20) ** 2 + (yy - 20) ** 2 < 10**2
    aperture &= valid_out & valid_in
    flux_out, flux_in = out_stamp[aperture].sum(), in_stamp[aperture].sum()
    if flux_out <= 0:
        raise RuntimeError("aperture has no positive baseline flux")
    ratio = flux_in / flux_out
    depth_error = (
        np.sqrt(
            variance_in[aperture].sum()
            + ratio**2 * variance_out[aperture].sum()
        )
        / flux_out
    )
    cp.cuda.get_current_stream().synchronize()
    return {
        "out_centroid_xy": measured_xy[0],
        "in_centroid_xy": measured_xy[1],
        "difference_centroid_xy": measured_xy[2],
        "centroid_shift_xy_pixels": measured_xy[1] - measured_xy[0],
        "centroid_shift_error_xy_pixels": np.hypot(
            errors_xy[0], errors_xy[1]
        ),
        "fading_source_offset_xy_pixels": measured_xy[2] - measured_xy[0],
        "difference_centroid_error_xy_pixels": errors_xy[2],
        "lost_flux_njy": -float(flux_fit.flux[0]),
        "lost_flux_error_njy": float(flux_fit.flux_error[0]),
        "aperture_depth": float(1 - ratio),
        "aperture_depth_error": float(depth_error),
        "fit_converged": cp.asnumpy(fit.converged),
        "backend": "cupy",
        "science": in_stamp,
        "reference": out_stamp,
        "difference": difference,
        "valid": valid_out & valid_in,
        "stamp_center_xy": center,
        "uncertainty_scope": (
            "Formal errors condition on the chosen Gaussian centroid model, "
            "empirical PSF and estimated sky; they exclude PSF-model and "
            "sky-estimation uncertainty. Difference-source and out-image "
            "positions share pixels; no independent sigma claim is made "
            "for their separation."
        ),
    }


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    return value


def run(bundle: Path, output: Path, *, config=None, sources=None) -> dict:
    """Prepare and measure two controlled scenarios; save offline HTML."""
    output = Path(output)
    definition = _prepare(Path(bundle), output, config=config)
    results, panels = {}, []
    for key, paths in definition["cases"].items():
        measured = measure_case(
            output / paths["out"],
            output / paths["in"],
            definition["target_search"]["old_xy"],
        )
        results[key] = measured
        offset = np.linalg.norm(measured["fading_source_offset_xy_pixels"])
        shift_value = np.linalg.norm(measured["centroid_shift_xy_pixels"])
        panels.append(
            image_triptych(
                measured["science"],
                measured["reference"],
                measured["difference"],
                mask=~measured["valid"],
                title="Planet on the target"
                if key == "planet"
                else "Nearby eclipsing blend",
                pixel_scale=definition["pixel_scale_arcsec"],
                context=(
                    "Measured aperture dip: "
                    f"{100 * measured['aperture_depth']:.2f}% "
                    f"± {100 * measured['aperture_depth_error']:.2f}% "
                    "formal. "
                    f"In/out photocenter shift: {shift_value:.3f} pixels. "
                    "Fading-light offset from baseline photocenter: "
                    f"{offset:.2f} pixels. The cyan ring marks the baseline "
                    "photocenter; the orange cross locates fading light."
                ),
            )
        )
        center = measured["stamp_center_xy"]
        baseline = measured["out_centroid_xy"] - center + 20
        fading = measured["difference_centroid_xy"] - center + 20
        for plot in panels[-1].select({"type": figure}):
            if "−" not in plot.title.text:
                continue
            plot.scatter(
                [baseline[0]],
                [baseline[1]],
                marker="circle",
                size=18,
                fill_alpha=0,
                line_color="#00a9d6",
                line_width=2,
            )
            plot.scatter(
                [fading[0]],
                [fading[1]],
                marker="x",
                size=17,
                line_color="#ff9500",
                line_width=2,
            )
    layout = column(
        Div(
            text=(
                "<h2>Where did the light disappear?</h2><p>A similar dip can "
                "come from a small planet crossing the target or a faint "
                "neighboring star undergoing a deeper eclipse. The negative "
                "difference source helps locate the missing light. These "
                "controlled snapshots average 16 independent exposures "
                "each; they do not establish periodicity or model an "
                "entire eclipse.</p>"
            )
        ),
        *panels,
        Div(
            text=(
                "<p><b>Interpretation:</b> a displaced fading source argues "
                "against a transit on the target. An unresolved source "
                "consistent with the target is not proof of a planet; "
                "radial velocities, repeated transits, color information "
                "and other follow-up remain relevant. Formal uncertainties "
                "condition on the fitted PSF and sky.</p>"
                "<p><small>Read more: "
                "<a href='https://arxiv.org/abs/1303.0052'>Kepler centroid "
                "vetting</a>; <a href='https://arxiv.org/abs/0912.3229'>"
                "GJ 1214 discovery and blend rejection</a>.</small></p>"
            )
        ),
        sizing_mode="stretch_width",
        max_width=1200,
    )
    html_path = output / "planet-or-blend.html"
    export_html(
        layout,
        html_path,
        title="cuPhoton · Planet or blended variable?",
        sources=sources,
    )
    arrays = {"science", "reference", "difference", "valid"}
    measurements = {
        case: {
            key: value for key, value in result.items() if key not in arrays
        }
        for case, result in results.items()
    }
    (output / "measurements.json").write_text(
        json.dumps(_json_safe(measurements), indent=2) + "\n"
    )
    return {
        "measurements": measurements,
        "cases": results,
        "layout": layout,
        "html": html_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    options = parser.parse_args()
    result = run(options.bundle, options.output)
    print(result["html"])
