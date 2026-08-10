#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run deterministic, data-independent cuPhoton workflow smoke tests."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from astropy.io import fits
from scipy.signal import fftconvolve

from cuphoton import __version__

SEED = 20260701
COMPONENTS = ("xfit", "xpois", "xscan", "xrep", "xray")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _relative(path: Path, root: Path) -> str:
    return path.expanduser().resolve().relative_to(root.resolve()).as_posix()


def _finite_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def _torch_hardware() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"available": False, "cuda_available": False}

    cuda_available = bool(torch.cuda.is_available())
    payload: dict[str, Any] = {
        "available": True,
        "version": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "cuda_available": cuda_available,
        "devices": [],
    }
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            payload["devices"].append(
                {
                    "index": index,
                    "name": properties.name,
                    "compute_capability": (
                        f"{properties.major}.{properties.minor}"
                    ),
                    "total_memory_bytes": int(properties.total_memory),
                }
            )
    return payload


def _cupy_hardware() -> dict[str, Any]:
    try:
        import cupy as cp
    except ImportError:
        return {"available": False, "cuda_available": False}

    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
        devices = []
        for index in range(device_count):
            properties = cp.cuda.runtime.getDeviceProperties(index)
            name = properties.get("name", "unknown")
            if isinstance(name, bytes):
                name = name.decode(errors="replace")
            devices.append({"index": index, "name": str(name)})
        return {
            "available": True,
            "version": str(cp.__version__),
            "cuda_available": device_count > 0,
            "driver_version": int(cp.cuda.runtime.driverGetVersion()),
            "runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
            "devices": devices,
        }
    except Exception as exc:  # pragma: no cover - runtime-specific
        return {
            "available": True,
            "version": str(cp.__version__),
            "cuda_available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _numba_cuda_available() -> bool:
    try:
        from numba import cuda

        return bool(cuda.is_available())
    except Exception:
        return False


def detect_hardware() -> dict[str, Any]:
    """Return JSON-compatible runtime and accelerator metadata."""

    torch = _torch_hardware()
    cupy = _cupy_hardware()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuphoton": __version__,
        "torch": torch,
        "cupy": cupy,
        "numba_cuda_available": _numba_cuda_available(),
        "gpu_available": bool(
            torch.get("cuda_available")
            or cupy.get("cuda_available")
            or _numba_cuda_available()
        ),
    }


def _torch_device(
    *,
    profile: str,
    require_gpu: bool,
    hardware: dict[str, Any],
    component: str,
) -> str:
    cuda_available = bool(hardware["torch"].get("cuda_available"))
    if profile == "cpu":
        return "cpu"
    if cuda_available:
        return "cuda"
    if require_gpu:
        raise RuntimeError(
            f"{component} requires a CUDA-capable PyTorch runtime"
        )
    return "cpu"


def _array_backend(
    *,
    profile: str,
    require_gpu: bool,
    hardware: dict[str, Any],
    component: str,
    allow_torch: bool = False,
) -> str:
    if profile == "cpu":
        return "cpu"
    if hardware["cupy"].get("cuda_available"):
        return "cupy"
    if component == "xpois" and hardware["numba_cuda_available"]:
        return "numba-cuda"
    if allow_torch and hardware["torch"].get("cuda_available"):
        return "torch"
    if require_gpu:
        raise RuntimeError(f"{component} has no usable CUDA backend")
    return "cpu"


def _run_xpois(
    root: Path,
    component_dir: Path,
    *,
    profile: str,
    require_gpu: bool,
    hardware: dict[str, Any],
) -> dict[str, Any]:
    from cuphoton.xpois import (
        GaussianBasisComponent,
        build_gaussian_polynomial_basis,
        solve_constant_kernel,
    )

    input_dir = component_dir / "inputs"
    artifact_dir = component_dir / "artifacts"
    input_dir.mkdir(parents=True)
    artifact_dir.mkdir()
    y, x = np.mgrid[:32, :32]
    reference = 3.0 * np.exp(-((x - 10.0) ** 2 + (y - 11.0) ** 2) / 8.0)
    reference += 2.0 * np.exp(-((x - 22.0) ** 2 + (y - 20.0) ** 2) / 12.0)
    components = [GaussianBasisComponent(sigma=1.5, degree=0)]
    basis, _ = build_gaussian_polynomial_basis(
        (9, 9),
        components,
        flux_conserve=True,
    )
    target = fftconvolve(reference, basis[0], mode="same") + 0.02
    reference_path = input_dir / "reference.npy"
    target_path = input_dir / "target.npy"
    np.save(reference_path, reference, allow_pickle=False)
    np.save(target_path, target, allow_pickle=False)

    requested_backend = _array_backend(
        profile=profile,
        require_gpu=require_gpu,
        hardware=hardware,
        component="xpois",
    )
    result = solve_constant_kernel(
        reference,
        target,
        components,
        kernel_shape=(9, 9),
        background_degree=0,
        flux_conserve=True,
        backend=requested_backend,
    )
    kernel_path = artifact_dir / "kernel.npy"
    matched_path = artifact_dir / "matched.npy"
    residual_path = artifact_dir / "residual.npy"
    np.save(kernel_path, result.kernel, allow_pickle=False)
    np.save(matched_path, result.matched, allow_pickle=False)
    np.save(residual_path, result.residual, allow_pickle=False)
    device = "cuda" if result.backend != "cpu" else "cpu"
    valid = np.isfinite(result.matched) & np.isfinite(result.residual)
    return {
        "backend": result.backend,
        "requested_backend": requested_backend,
        "device": device,
        "dtype": str(result.kernel.dtype),
        "inputs": {
            "reference": _relative(reference_path, root),
            "target": _relative(target_path, root),
        },
        "artifacts": {
            "kernel": _relative(kernel_path, root),
            "matched": _relative(matched_path, root),
            "residual": _relative(residual_path, root),
        },
        "metrics": {
            "kernel_sum": float(result.kernel.sum()),
            "residual_rms": float(
                np.sqrt(np.mean(result.residual[valid] ** 2))
            ),
            "fit_pixel_count": int(result.fit_pixel_count),
            "finite": bool(np.isfinite(result.kernel).all() and valid.any()),
            "finite_output_fraction": float(valid.mean()),
        },
    }


def _run_xfit(
    root: Path,
    component_dir: Path,
    *,
    profile: str,
    require_gpu: bool,
    hardware: dict[str, Any],
) -> dict[str, Any]:
    from cuphoton.xfit import GaussianDipoleModel, fit_dipoles

    input_dir = component_dir / "inputs"
    artifact_dir = component_dir / "artifacts"
    input_dir.mkdir(parents=True)
    artifact_dir.mkdir()
    model = GaussianDipoleModel((17, 21), dtype=np.float64)
    true_parameters = np.asarray(
        [
            [7.0, 1.7, 1.3, 0.20, -2.3, -1.8, 2.5, 1.4],
            [5.5, 1.4, 1.8, -0.15, -3.1, 1.6, 2.2, -1.4],
            [8.0, 1.9, 1.5, 0.35, -1.8, -2.0, 3.2, 2.1],
        ],
        dtype=np.float64,
    )
    images = np.asarray(
        model.evaluate(true_parameters, mode="difference"),
        dtype=np.float64,
    )
    initial = true_parameters.copy()
    initial[:, 0] *= 0.9
    initial[:, 4:8] += np.asarray([0.25, -0.20, -0.20, 0.25])
    candidate_id = np.asarray(
        [f"synthetic-{index}" for index in range(images.shape[0])]
    )
    input_path = input_dir / "dipoles.npz"
    np.savez_compressed(
        input_path,
        candidate_id=candidate_id,
        images=images,
        initial=initial,
    )

    selected_backend = _array_backend(
        profile=profile,
        require_gpu=require_gpu,
        hardware=hardware,
        component="xfit",
    )
    requested_backend = "numpy" if selected_backend == "cpu" else "cupy"
    result = fit_dipoles(
        images,
        model=model,
        initial=initial,
        backend=requested_backend,
    )
    artifact_path = artifact_dir / "fit-arrays.npz"
    np.savez_compressed(
        artifact_path,
        parameters=result.parameters,
        covariance=result.covariance,
        standard_errors=result.standard_errors,
        residuals=result.residuals,
        status=result.status,
        converged=result.converged,
    )
    parameter_error = np.abs(result.parameters - true_parameters)
    finite = bool(
        np.isfinite(result.parameters).all()
        and np.isfinite(result.residual_norm).all()
    )
    return {
        "backend": result.backend,
        "requested_backend": requested_backend,
        "device": result.device,
        "dtype": result.dtype,
        "inputs": {"dipoles": _relative(input_path, root)},
        "artifacts": {"fit_arrays": _relative(artifact_path, root)},
        "metrics": {
            "candidate_count": int(images.shape[0]),
            "converged_count": int(np.count_nonzero(result.converged)),
            "max_absolute_parameter_error": float(parameter_error.max()),
            "max_residual_norm": float(np.max(result.residual_norm)),
            "finite": finite,
        },
    }


def _run_xscan(
    root: Path,
    component_dir: Path,
    *,
    profile: str,
    require_gpu: bool,
    hardware: dict[str, Any],
) -> dict[str, Any]:
    from cuphoton.xscan.config import ModelConfig, TrainingConfig
    from cuphoton.xscan.training import train_classifier

    dataset_dir = component_dir / "dataset"
    run_dir = component_dir / "run"
    dataset_dir.mkdir(parents=True)
    run_dir.mkdir()
    rng = np.random.default_rng(SEED)
    sample_count = 8
    search = rng.normal(0.0, 0.05, (sample_count, 9, 9)).astype(np.float32)
    template = rng.normal(0.0, 0.05, (sample_count, 9, 9)).astype(np.float32)
    labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    split = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    for index, label in enumerate(labels):
        if label:
            search[index, 3:6, 3:6] += 1.0
    np.save(dataset_dir / "search.npy", search, allow_pickle=False)
    np.save(dataset_dir / "template.npy", template, allow_pickle=False)
    np.save(dataset_dir / "labels.npy", labels, allow_pickle=False)
    np.save(dataset_dir / "split.npy", split, allow_pickle=False)
    split_names = ("train", "val")
    metadata = []
    for index, label in enumerate(labels.tolist()):
        metadata.append(
            {
                "candidate_id": f"synthetic-{index}",
                "label": label,
                "label_source": "synthetic_quickstart",
                "split": split_names[int(split[index])],
                "split_group": f"synthetic-{index}",
                "exposure_id": index,
                "ccd_id": "synthetic",
                "band": "synthetic",
                "x": 4,
                "y": 4,
            }
        )
    metadata_path = dataset_dir / "metadata.jsonl"
    metadata_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in metadata) + "\n",
        encoding="utf-8",
    )

    device = _torch_device(
        profile=profile,
        require_gpu=require_gpu,
        hardware=hardware,
        component="xscan",
    )
    config = TrainingConfig(
        dataset_dir=str(dataset_dir),
        epochs=1,
        batch_size=2,
        seed=SEED,
        device=device,
        train_split="train",
        val_split="val",
        model=ModelConfig(
            input_mode="pair",
            image_size=9,
            depths=[1, 1],
            num_heads=[1, 1],
            embed_dims=[8, 8],
            decoder_embedding_dim=8,
            pos_dim=8,
            output_nc=2,
            drop_rate=0.0,
            attn_drop=0.0,
        ),
    )
    training = train_classifier(config, run_dir=run_dir)
    training_summary_path = run_dir / "training-summary.json"
    _write_json(training_summary_path, training)
    best_val_roc_auc = _finite_float_or_none(training["best_val_roc_auc"])
    return {
        "backend": "torch",
        "device": str(training["device"]),
        "dtype": "float32",
        "inputs": {
            "search": _relative(dataset_dir / "search.npy", root),
            "template": _relative(dataset_dir / "template.npy", root),
            "labels": _relative(dataset_dir / "labels.npy", root),
            "split": _relative(dataset_dir / "split.npy", root),
            "metadata": _relative(metadata_path, root),
        },
        "artifacts": {
            "checkpoint": _relative(run_dir / "checkpoint.pt", root),
            "config": _relative(run_dir / "config.yaml", root),
            "history": _relative(run_dir / "history.json", root),
            "training_summary": _relative(training_summary_path, root),
        },
        "metrics": {
            "sample_count": sample_count,
            "epochs_completed": int(training["epochs_completed"]),
            "best_val_roc_auc": best_val_roc_auc,
            "finite": best_val_roc_auc is not None,
        },
    }


def _run_xrep(
    root: Path,
    component_dir: Path,
    *,
    profile: str,
    require_gpu: bool,
    hardware: dict[str, Any],
) -> dict[str, Any]:
    from cuphoton.xrep import make_north_up_wcs
    from cuphoton.xrep.workflows import run_reproject_image

    input_dir = component_dir / "inputs"
    input_dir.mkdir(parents=True)
    y, x = np.mgrid[:16, :16]
    image = np.exp(-((x - 7.5) ** 2 + (y - 7.5) ** 2) / 8.0).astype(
        np.float32
    )
    wcs = make_north_up_wcs(
        (150.0, 2.0),
        shape=image.shape,
        pixel_scale_arcsec=0.2,
    )
    input_path = input_dir / "image.fits"
    fits.PrimaryHDU(data=image, header=wcs.to_header()).writeto(input_path)
    backend = _array_backend(
        profile=profile,
        require_gpu=require_gpu,
        hardware=hardware,
        component="xrep",
        allow_torch=True,
    )
    result = run_reproject_image(
        input_path=input_path,
        output_root=component_dir / "runs",
        name="reproject",
        hdu=0,
        mask_path=None,
        mask_hdu=None,
        backend=backend,
        interpolation="bilinear",
        grid_crval_ra=None,
        grid_crval_dec=None,
        pixel_scale_arcsec=None,
        mapping_grid_step=4,
        area_scaling=False,
        write_fits=True,
    )
    native_summary = result.run_dir / "summary.json"
    saved = {
        key: _relative(result.run_dir / value, root)
        for key, value in result.summary["saved"].items()
    }
    saved["workflow_summary"] = _relative(native_summary, root)
    reprojected = np.load(result.run_dir / result.summary["saved"]["image"])
    finite = np.isfinite(reprojected)
    return {
        "backend": result.summary["backend"],
        "device": "cuda" if backend != "cpu" else "cpu",
        "dtype": str(reprojected.dtype),
        "inputs": {"fits": _relative(input_path, root)},
        "artifacts": saved,
        "metrics": {
            "input_shape": list(image.shape),
            "output_shape": list(reprojected.shape),
            "finite": bool(finite.any()),
            "finite_output_fraction": float(finite.mean()),
        },
    }


def _write_xray_hdf5(
    path: Path,
    *,
    delay: np.ndarray,
    image: np.ndarray,
) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset("ROI", data=np.ones((2, 2), dtype=np.float64))
        handle.create_dataset("bin_count", data=np.ones(delay.size))
        handle.create_dataset("i0", data=np.ones(delay.size))
        handle.create_dataset("i0_ipm3", data=np.ones(delay.size))
        handle.create_dataset("imgs", data=image)
        handle.create_dataset("scan_var", data=delay)


def _run_xray(
    root: Path,
    component_dir: Path,
    *,
    profile: str,
    require_gpu: bool,
    hardware: dict[str, Any],
) -> dict[str, Any]:
    from cuphoton.xray.hdf5 import load_hdf5_pair_trace
    from cuphoton.xray.linear_prediction import (
        linear_prediction_cupy,
        linear_prediction_numpy,
        synthetic_trace,
    )

    input_dir = component_dir / "inputs"
    artifact_dir = component_dir / "artifacts"
    input_dir.mkdir(parents=True)
    artifact_dir.mkdir()
    delay, source_trace = synthetic_trace(32)
    source_trace = 0.05 * np.asarray(source_trace, dtype=np.float64)
    off = np.ones((delay.size, 2, 2), dtype=np.float64)
    on = (1.0 + source_trace)[:, None, None] * off
    on_path = input_dir / "on.h5"
    off_path = input_dir / "off.h5"
    _write_xray_hdf5(on_path, delay=delay, image=on)
    _write_xray_hdf5(off_path, delay=delay, image=off)
    pair = load_hdf5_pair_trace(
        h5dir=input_dir,
        fon=on_path.name,
        foff=off_path.name,
        drop_leading=0,
        chunk_frames=8,
        reference_shift=False,
    )
    backend = _array_backend(
        profile=profile,
        require_gpu=require_gpu,
        hardware=hardware,
        component="xray",
    )
    if backend == "cupy":
        result = linear_prediction_cupy(pair.delay, pair.ratio_minus_one, 4)
    elif backend == "cpu":
        result = linear_prediction_numpy(pair.delay, pair.ratio_minus_one, 4)
    else:
        raise RuntimeError(
            f"XRay does not support quickstart backend {backend}"
        )

    delay_path = artifact_dir / "delay.npy"
    trace_path = artifact_dir / "ratio_minus_one.npy"
    reconstruction_path = artifact_dir / "reconstruction.npy"
    spectrum_path = artifact_dir / "spectrum.npy"
    np.save(delay_path, pair.delay, allow_pickle=False)
    np.save(trace_path, pair.ratio_minus_one, allow_pickle=False)
    np.save(reconstruction_path, result.reconstruction, allow_pickle=False)
    np.save(spectrum_path, result.spectrum_total, allow_pickle=False)
    return {
        "backend": result.backend,
        "device": "cuda" if result.backend == "gpu" else "cpu",
        "dtype": str(result.reconstruction.dtype),
        "inputs": {
            "on_hdf5": _relative(on_path, root),
            "off_hdf5": _relative(off_path, root),
        },
        "artifacts": {
            "delay": _relative(delay_path, root),
            "ratio_minus_one": _relative(trace_path, root),
            "reconstruction": _relative(reconstruction_path, root),
            "spectrum": _relative(spectrum_path, root),
        },
        "metrics": {
            "samples": int(pair.delay.size),
            "selected_model_order": int(result.selected_model_order),
            "chi2": float(result.chi2),
            "finite": bool(np.isfinite(result.reconstruction).all()),
        },
    }


Runner = Callable[..., dict[str, Any]]
RUNNERS: dict[str, Runner] = {
    "xfit": _run_xfit,
    "xpois": _run_xpois,
    "xscan": _run_xscan,
    "xrep": _run_xrep,
    "xray": _run_xray,
}


def run_quickstarts(
    *,
    output_dir: Path,
    profile: str,
    require_gpu: bool,
    components: Sequence[str],
) -> dict[str, Any]:
    """Run selected synthetic workflows and return their root summary."""

    if profile not in {"auto", "cpu"}:
        raise ValueError("profile must be 'auto' or 'cpu'")
    if profile == "cpu" and require_gpu:
        raise ValueError("--profile cpu and --require-gpu are incompatible")
    selected = tuple(dict.fromkeys(components))
    if not selected:
        selected = COMPONENTS
    unknown = sorted(set(selected) - set(COMPONENTS))
    if unknown:
        raise ValueError("unknown components: " + ", ".join(unknown))

    root = output_dir.expanduser().resolve()
    if root.exists():
        raise FileExistsError(
            f"output directory already exists; remove it first: {root}"
        )
    root.mkdir(parents=True)
    hardware = detect_hardware()
    summary: dict[str, Any] = {
        "schema_version": 1,
        "seed": SEED,
        "profile": profile,
        "require_gpu": require_gpu,
        "components": list(selected),
        "hardware": hardware,
        "results": {},
        "status": "running",
    }
    summary_path = root / "summary.json"

    for component in selected:
        component_dir = root / component
        component_dir.mkdir()
        try:
            result = RUNNERS[component](
                root,
                component_dir,
                profile=profile,
                require_gpu=require_gpu,
                hardware=hardware,
            )
            result = {"status": "ok", **result}
            component_summary_path = component_dir / "summary.json"
            result.setdefault("artifacts", {})["summary"] = _relative(
                component_summary_path,
                root,
            )
            _write_json(component_summary_path, result)
            summary["results"][component] = result
        except Exception as exc:
            summary["results"][component] = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            summary["status"] = "failed"
            _write_json(summary_path, summary)
            raise
        _write_json(summary_path, summary)

    summary["status"] = "ok"
    _write_json(summary_path, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("auto", "cpu"),
        default="auto",
        help="Execution policy; auto prefers a GPU and visibly falls back.",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail if any selected component cannot run on a GPU.",
    )
    parser.add_argument(
        "--component",
        choices=COMPONENTS,
        action="append",
        default=[],
        help="Component to run; repeat the option to select several.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("quickstart-output"),
        help="New directory that will own inputs, artifacts, and summaries.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.profile == "cpu" and args.require_gpu:
        parser.error("--profile cpu and --require-gpu are incompatible")
    try:
        summary = run_quickstarts(
            output_dir=args.output_dir,
            profile=args.profile,
            require_gpu=args.require_gpu,
            components=args.component,
        )
    except Exception as exc:
        print(
            f"quickstart failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        summary_path = args.output_dir.expanduser().resolve() / "summary.json"
        if summary_path.is_file():
            print(summary_path.read_text(encoding="utf-8"), end="")
        return 1
    print(json.dumps(summary, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
