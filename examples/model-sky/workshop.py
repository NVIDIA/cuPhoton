# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Command-line entry points for the example's measurements and artifacts."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pipeline


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, default=json_default) + "\n"
    )


def save_motion(output, motion, reference):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pipeline.write_exposure(
        output / "motion_registered.fits", motion["registered"]
    )
    pipeline.write_exposure(output / "motion_reference.fits", reference)
    np.save(
        output / "motion_fit_mask.npy", motion["fit_mask"], allow_pickle=False
    )
    np.savez(
        output / "motion_candidates.npz",
        candidate_id=np.array(["target"]),
        **{
            key: value
            for key, value in motion["fit_input"].items()
            if key != "stamp_scale"
        },
    )
    np.savez_compressed(
        output / "motion.npz",
        **{
            key: value
            for key, value in motion.items()
            if isinstance(value, np.ndarray)
        },
    )
    summary = {
        key: motion[key]
        for key in (
            "old_xy",
            "new_xy",
            "separation_pixels",
            "separation_error_pixels",
            "registration_floor_pixels",
            "backend",
        )
    }
    summary["stamp_scale"] = float(motion["fit_input"]["stamp_scale"])
    write_json(output / "motion.json", summary)
    return summary


def save_transit(output, transit):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    arrays = {
        key: value
        for key, value in transit.items()
        if isinstance(value, np.ndarray)
    }
    arrays["model"] = transit["fit"]["model"]
    np.savez_compressed(output / "transit.npz", **arrays)
    summary = {
        key: value
        for key, value in transit["fit"].items()
        if key not in {"model", "covariance", "parameters"}
    }
    summary.update(
        error_scale=transit["error_scale"],
        elapsed_seconds=transit["elapsed_seconds"],
        backend=transit["backend"],
        valid_epochs=int(transit["valid"].sum()),
        trail_epochs=np.flatnonzero(transit["trail_mask_pixels"]).tolist(),
    )
    write_json(output / "transit.json", summary)
    np.savetxt(
        output / "light_curve.csv",
        np.column_stack(
            [
                transit["times_minutes"],
                transit["flux"],
                transit["flux_error"],
                transit["valid"],
                transit["check_flux"],
                transit["fit"]["model"],
            ]
        ),
        delimiter=",",
        header="time_min,relative_flux,error,valid,check_flux,model",
        comments="",
    )
    return summary


def export_views(output, manifest, trail, motion, transit, *, png=False):
    import views
    from astropy.wcs import WCS

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    scale = manifest["pixel_scale_arcsec"]
    image_wcs = WCS(motion["registered"]["header"])
    x, y = np.rint(transit["positions"][0]).astype(int)
    half = transit["stamps"].shape[-1] // 2
    stamp_wcs = image_wcs.slice(
        (slice(y - half, y + half + 1), slice(x - half, x + half + 1))
    )
    profile_data = trail.get("trail_profile", {})
    profile = (
        dict(
            x=profile_data["coordinate"],
            difference=profile_data["profile"],
            x_label="Perpendicular distance from trail [pixel]",
            y_label="Median residual [nJy]",
        )
        if profile_data.get("detected")
        else None
    )
    factories = {
        "01_satellite": lambda: views.trail_view(
            trail["science"],
            trail["matched"],
            trail["difference"],
            trail["trail_mask"],
            pixel_scale=scale,
            profile=profile,
            wcs=image_wcs,
            context="Model Sky · simulated satellite trail.",
        ),
        "02_motion": lambda: views.motion_view(
            motion["matched"],
            motion["science"],
            motion["difference"],
            old_xy=motion["old_xy"],
            new_xy=motion["new_xy"],
            baseline_years=manifest["assumptions"]["baseline_years"],
            pixel_scale=scale,
            wcs=image_wcs,
            context=(
                f"Displacement {motion['separation_pixels']:.3f} ± "
                f"{motion['separation_error_pixels']:.3f} pixel; "
                "uncertainty includes the registration floor."
            ),
        ),
        "03_transit": lambda: views.transit_view(
            transit["times_minutes"],
            transit["flux"],
            transit["flux_error"],
            transit["fit"]["model"],
            transit_model=transit["fit"]["transit_model"],
            stamps=transit["stamps"],
            difference_stamps=transit["difference_stamps"],
            valid=transit["valid"],
            check_flux=transit["check_flux"],
            reference_stamp=transit["reference_stamp"],
            pixel_scale=scale,
            wcs=stamp_wcs,
            summary={
                "Geometric depth (Rp/Rstar)²": (
                    f"{100 * transit['fit']['geometric_depth']:.3f}%"
                ),
                "Measurement": "Model Sky · simulated system and field",
            },
        ),
    }
    for name, factory in factories.items():
        views.export_html(
            factory(),
            output / f"{name}.html",
            sources=Path(__file__).with_name("sources.md"),
        )
        if png:
            views.export_png(factory(), output / f"{name}.png")


def run(bundle, output, *, stage="all", png=False):
    import cuphoton

    root, manifest = pipeline.load_bundle(bundle)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cp = pipeline.gpu()
    properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    write_json(
        output / "runtime.json",
        {
            "cuphoton": cuphoton.__version__,
            "python": platform.python_version(),
            "machine": platform.machine(),
            "host": platform.node(),
            "gpu": properties["name"].decode(),
            "cupy": cp.__version__,
            "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
            "cuda_driver": cp.cuda.runtime.driverGetVersion(),
            "backend": "cupy",
            "truth_used_for_measurement": False,
        },
    )
    old = pipeline.read_exposure(
        pipeline.bundle_path(root, manifest["files"]["reference_old"])
    )
    current = pipeline.read_exposure(
        pipeline.bundle_path(root, manifest["files"]["reference_current"])
    )
    seed = np.asarray(manifest["target_search"]["old_xy"])
    motion = pipeline.measure_motion(
        old,
        pipeline.read_exposure(
            pipeline.bundle_path(root, manifest["files"]["motion"])
        ),
        seed,
        search_radius=manifest["target_search"].get("radius_pixels", 18),
    )
    summary = {"motion": save_motion(output, motion, old)}
    if stage == "motion":
        return summary
    anchors = pipeline.select_stars(current, exclude=[motion["new_xy"]])
    trail = pipeline.subtract(
        current,
        pipeline.register(
            pipeline.read_exposure(
                pipeline.bundle_path(root, manifest["files"]["trail"])
            ),
            current,
            anchors=anchors,
        ),
        exclude=[motion["new_xy"]],
    )
    np.savez_compressed(
        output / "trail.npz",
        **{
            key: value
            for key, value in trail.items()
            if isinstance(value, np.ndarray)
        },
    )
    if stage == "trail":
        return {**summary, "trail_pixels": int(trail["trail_mask"].sum())}
    transit = pipeline.measure_transit(
        root,
        manifest,
        motion["new_xy"],
        progress=lambda n, total: print(
            f"Measured {n}/{total} exposures", flush=True
        ),
    )
    summary["transit"] = save_transit(output, transit)
    export_views(output, manifest, trail, motion, transit, png=png)
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=["motion", "trail", "transit", "all"]
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--png", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.bundle, args.output, stage=args.stage, png=args.png),
            indent=2,
            default=json_default,
        )
    )


if __name__ == "__main__":
    main()
