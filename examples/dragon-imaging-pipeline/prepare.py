# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Prepare the public-image Dragon demonstration before the presentation.

The checkpoint is trained on controlled, labeled injections. It is not an
astronomically calibrated classifier. Completed preparation stages are reused
only after their input identity and retained output hashes have been checked.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import yaml

STAMP_SIZE = 51
SPLITS = {"train": 0, "val": 1, "test": 2}
XPOIS = {
    "kernel_shape": (15, 15),
    "basis_sigmas": (1.5,),
    "basis_degrees": (0,),
    "background_degree": 0,
    "flux_conserve": False,
}
XFIT = {"solver_version": "v0.2", "max_evaluations": 400}


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def checked_stage(
    root: Path,
    name: str,
    identity: Any,
    outputs: list[Path],
    operation: Callable[[], None],
) -> None:
    """Resume complete stages; preserve incomplete attempts for inspection."""
    receipt = root / "preparation-receipts" / f"{name}.json"
    canonical = json.loads(json.dumps(identity, sort_keys=True))
    if receipt.exists():
        previous = json.loads(receipt.read_text(encoding="utf-8"))
        if previous["input"] != canonical:
            raise ValueError(
                f"{name}: inputs changed; use a new output directory"
            )
        for relative, digest in previous["outputs"].items():
            path = root / relative
            if not path.is_file() or sha256(path) != digest:
                raise ValueError(f"{name}: retained artifact changed: {path}")
        print(f"Verified completed preparation stage: {name}", flush=True)
        return
    occupied = [str(path) for path in outputs if path.exists()]
    if occupied:
        raise FileExistsError(
            f"{name}: incomplete artifacts are preserved: {occupied}; "
            "inspect them and select a new output directory"
        )
    print(f"Preparing {name}", flush=True)
    operation()
    files = []
    for path in outputs:
        if path.is_dir():
            files.extend(
                sorted(item for item in path.rglob("*") if item.is_file())
            )
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(
                f"{name}: missing completed output: {path}"
            )
    if not files:
        raise ValueError(f"{name}: no completed artifacts")
    write_json(
        receipt,
        {
            "input": canonical,
            "outputs": {
                str(path.relative_to(root)): sha256(path) for path in files
            },
        },
    )


def command(*arguments: str) -> None:
    subprocess.run(
        [sys.executable, "-m", "cuphoton", *arguments],
        check=True,
    )


def source_identity(scenes: list[dict[str, Any]]) -> dict[str, str]:
    return {
        str(path): sha256(path)
        for path in sorted(
            {
                Path(scene[field])
                for scene in scenes
                for field in ("reference", "target", "variance", "fit_mask")
            }
        )
    }


def validate_splits(rows: list[dict[str, Any]], *, training: bool) -> None:
    """Require honest, balanced group-separated supervised inputs."""
    groups: dict[str, set[str]] = {}
    identifiers = set()
    for row in rows:
        if row["candidate_id"] in identifiers:
            raise ValueError("duplicate candidate identity")
        identifiers.add(row["candidate_id"])
        groups.setdefault(str(row["split_group"]), set()).add(row["split"])
        if not row.get("label_source", "").startswith("synthetic_"):
            raise ValueError(
                "the demo requires explicit synthetic label provenance"
            )
    if any(len(splits) != 1 for splits in groups.values()):
        raise ValueError(
            "a spatial background group occurs in multiple splits"
        )
    expected = (
        {"train": 320, "val": 96, "test": 96}
        if training
        else {"test": len(rows)}
    )
    observed = {
        split: sum(row["split"] == split for row in rows)
        for split in expected
    }
    if observed != expected or sum(observed.values()) != len(rows):
        raise ValueError(
            f"unexpected split counts: {observed}, expected {expected}"
        )
    for split, count in expected.items():
        labels = [row["label"] for row in rows if row["split"] == split]
        if set(labels) != {0, 1} or sum(labels) * 2 != count:
            raise ValueError(
                f"{split}: the controlled classes must be balanced"
            )


def prepare_dataset(
    root: Path, scenes: list[dict[str, Any]], *, training: bool
) -> None:
    # Match the pipeline's CUDA-library initialization order.
    import cupy as cp
    import torch
    from astropy.io import fits

    from cuphoton.xpois import (
        GaussianBasisComponent,
        solve_constant_kernel_device,
    )
    from cuphoton.xscan.dataset import validate_dataset_dir
    from cuphoton.xscan.training import require_training_label_provenance

    torch.cuda.set_device(0)
    cp.cuda.Device(0).use()
    root.mkdir(parents=True)
    image_rows: dict[str, list[np.ndarray]] = {
        key: [] for key in ("search", "template", "difference")
    }
    rows: list[dict[str, Any]] = []
    half = STAMP_SIZE // 2
    components = [GaussianBasisComponent(sigma=1.5, degree=0)]
    for scene in scenes:
        hdu = scene.get("hdu", 1)
        planes = {
            field: np.asarray(
                fits.getdata(scene[field], hdu), dtype=np.float64
            )
            for field in ("reference", "target", "variance", "fit_mask")
        }
        result = solve_constant_kernel_device(
            planes["reference"],
            planes["target"],
            components,
            kernel_shape=XPOIS["kernel_shape"],
            variance=planes["variance"],
            fit_mask=planes["fit_mask"].astype(bool),
            background_degree=0,
            flux_conserve=False,
        )
        residual = cp.asnumpy(result.residual)
        diagnostics = root / "subtractions" / scene["item_id"]
        diagnostics.mkdir(parents=True)
        np.save(diagnostics / "residual.npy", residual, allow_pickle=False)
        np.save(
            diagnostics / "matched.npy",
            cp.asnumpy(result.matched),
            allow_pickle=False,
        )
        np.save(
            diagnostics / "kernel.npy",
            cp.asnumpy(result.kernel),
            allow_pickle=False,
        )
        # The standalone xPois CLI accepts NPY fit masks; the complete
        # pipeline also accepts the original FITS mask descriptor.
        np.save(
            diagnostics / "fit-mask.npy",
            planes["fit_mask"].astype(bool),
            allow_pickle=False,
        )
        write_json(
            diagnostics / "summary.json",
            {
                "item_id": scene["item_id"],
                "backend": result.backend,
                "chi2": float(result.chi2),
                "dof": int(result.dof),
                "fit_pixel_count": int(result.fit_pixel_count),
            },
        )
        for candidate in scene["candidates"]:
            x, y = int(candidate["x"]), int(candidate["y"])
            section = (
                slice(y - half, y + half + 1),
                slice(x - half, x + half + 1),
            )
            for key, plane in (
                ("search", planes["target"]),
                ("template", planes["reference"]),
                ("difference", residual),
            ):
                stamp = np.array(plane[section], dtype=np.float64, copy=True)
                if (
                    stamp.shape != (STAMP_SIZE, STAMP_SIZE)
                    or not np.isfinite(stamp).all()
                ):
                    raise ValueError(
                        f"invalid stamp: {candidate['candidate_id']}"
                    )
                image_rows[key].append(stamp)
            rows.append(
                {
                    **candidate,
                    "exposure_id": scene["item_id"],
                    "ccd_id": "public-image-crop",
                    "band": "demonstration",
                    "target_label_available": True,
                }
            )
        del result
    validate_splits(rows, training=training)
    dataset = root / "dataset"
    dataset.mkdir()
    for key, stamps in image_rows.items():
        np.save(dataset / f"{key}.npy", np.stack(stamps), allow_pickle=False)
    np.save(
        dataset / "labels.npy",
        np.asarray([row["label"] for row in rows], dtype=np.int64),
        allow_pickle=False,
    )
    np.save(
        dataset / "split.npy",
        np.asarray([SPLITS[row["split"]] for row in rows], dtype=np.int64),
        allow_pickle=False,
    )
    (dataset / "metadata.jsonl").write_text(
        "".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    write_json(
        dataset / "summary.json",
        {
            "purpose": "Controlled moving-source injection labels",
            "validation": validate_dataset_dir(dataset),
            "label_provenance": require_training_label_provenance(dataset),
            "channel_order": [
                "target",
                "original_reference",
                "xpois_residual",
            ],
            "xpois": XPOIS,
        },
    )


def prepare_features(root: Path, collection: Path) -> None:
    dataset = collection / "dataset"
    dataset_hashes = {
        path.name: sha256(path)
        for path in sorted(dataset.iterdir())
        if path.is_file()
    }
    archive = collection / "dipoles.npz"
    prefix = collection.name
    checked_stage(
        root,
        f"{prefix}-export",
        dataset_hashes,
        [archive],
        lambda: command(
            "xscan",
            "data-export-xfit-input",
            "--dataset-dir",
            str(dataset),
            "--output",
            str(archive),
        ),
    )
    fitted = collection / "xfit"
    checked_stage(
        root,
        f"{prefix}-fit",
        {"input": sha256(archive), **XFIT},
        [fitted],
        lambda: command(
            "xfit",
            "fit-dipoles",
            "--input",
            str(archive),
            "--output-dir",
            str(fitted),
            "--model",
            "gaussian",
            "--mode",
            "difference",
            "--backend",
            "cupy",
            "--compute-dtype",
            "float64",
            "--solver-version",
            XFIT["solver_version"],
            "--max-evaluations",
            str(XFIT["max_evaluations"]),
        ),
    )
    features = collection / "features"
    checked_stage(
        root,
        f"{prefix}-features",
        {
            "dataset": dataset_hashes,
            "fit": sha256(fitted / "summary.json"),
        },
        [features],
        lambda: command(
            "xscan",
            "data-build-xfit-features",
            "--dataset-dir",
            str(dataset),
            "--xfit-run-dir",
            str(fitted),
            "--output-dir",
            str(features),
            "--missing-policy",
            "error",
        ),
    )


def evaluate_model(training: Path, output: Path) -> None:
    import torch

    from cuphoton.xscan.dataset import StampDataset
    from cuphoton.xscan.metrics import (
        evaluate_predictions,
        select_threshold_for_scores,
        threshold_metrics,
    )
    from cuphoton.xscan.training import (
        load_model_from_checkpoint,
        predict_dataset,
    )

    torch.set_num_threads(1)
    device = torch.device("cuda:0")
    model, _, performance = load_model_from_checkpoint(
        training / "model", device=device
    )
    predictions = {}
    for split in ("val", "test"):
        dataset = StampDataset(
            training / "dataset",
            input_mode="triplet",
            split=split,
            xfit_feature_dir=training / "features",
        )
        predictions[split] = predict_dataset(
            model=model,
            dataset=dataset,
            batch_size=32,
            device=device,
            performance=performance,
        )
    validation = predictions["val"]
    selection = select_threshold_for_scores(
        validation["labels"],
        validation["probabilities"],
        metric="accuracy",
        source_split="val",
    )
    threshold = float(selection["threshold"])
    reports = {}
    for split, prediction in predictions.items():
        reports[split] = {
            **evaluate_predictions(
                y_true=prediction["labels"],
                logits=prediction["logits"],
                metadata_rows=prediction["metadata_rows"],
                threshold=threshold,
            ),
            **threshold_metrics(
                prediction["labels"],
                prediction["probabilities"],
                threshold=threshold,
            ),
        }
    test = reports["test"]
    majority = (
        max(test["positive_count"], test["negative_count"])
        / test["sample_count"]
    )
    accepted = bool(
        test["accuracy"] > majority
        and test["roc_auc"] is not None
        and test["roc_auc"] > 0.5
    )
    write_json(
        output,
        {
            "purpose": "Held-out controlled-injection classification",
            "accepted": accepted,
            "majority_class_baseline": majority,
            "decision_threshold": threshold,
            "threshold_selection": selection,
            "splits": reports,
        },
    )
    if not accepted:
        raise RuntimeError(
            f"model did not exceed the held-out baseline; inspect {output}"
        )


def make_manifests(
    root: Path, scenes: list[dict[str, Any]], threshold: float
) -> None:
    from cuphoton.core.fits_io import inspect_fits_image
    from cuphoton.xscan.device_pipeline import (
        DevicePipelineCandidate,
        DevicePipelineConfig,
        DevicePipelineItem,
        DeviceXFitPipelineConfig,
        DeviceXPOISPipelineConfig,
        FitsArrayDescriptor,
    )

    def describe(raw: str, hdu: int) -> FitsArrayDescriptor:
        path = Path(raw).resolve()
        info = inspect_fits_image(path, hdu)
        return FitsArrayDescriptor(
            path=str(path),
            sha256=sha256(path),
            hdu=info.hdu,
            shape=info.shape,
            dtype=info.dtype.str,
            reader="astropy",
        )

    model = root / "training" / "model"
    schema = root / "training" / "features" / "schema.json"
    common = dict(
        device="cuda:0",
        checkpoint_dir=str(model),
        checkpoint_sha256=sha256(model / "checkpoint.pt"),
        feature_schema_path=str(schema),
        feature_schema_sha256=sha256(schema),
        stamp_shape=(STAMP_SIZE, STAMP_SIZE),
        decision_threshold=threshold,
        xpois=DeviceXPOISPipelineConfig(**XPOIS),
    )
    items = [
        DevicePipelineItem(
            item_id=scene["item_id"],
            reference=describe(scene["reference"], scene.get("hdu", 1)),
            target=describe(scene["target"], scene.get("hdu", 1)),
            variance=describe(scene["variance"], scene.get("hdu", 1)),
            fit_mask=describe(scene["fit_mask"], scene.get("hdu", 1)),
            candidates=tuple(
                DevicePipelineCandidate(
                    candidate_id=row["candidate_id"],
                    center_x=int(row["x"]),
                    center_y=int(row["y"]),
                    source_index=index,
                )
                for index, row in enumerate(scene["candidates"])
            ),
        ).to_payload()
        for scene in scenes
    ]
    for backend, filename in (
        ("cupy", "pipeline.json"),
        ("native", "pipeline-native.json"),
    ):
        config = DevicePipelineConfig(
            **common, xfit=DeviceXFitPipelineConfig(backend=backend, **XFIT)
        )
        write_json(
            root / filename,
            {
                "schema": "cuphoton.xscan.pipeline-manifest/v1",
                "configuration": config.to_payload(),
                "items": items,
            },
        )
    pairs = [
        {
            "id": scene["item_id"],
            **{
                field: scene[field]
                for field in ("reference", "target", "variance")
            },
            "fit_mask": str(
                root
                / "demo"
                / "subtractions"
                / scene["item_id"]
                / "fit-mask.npy"
            ),
            **{
                f"{field}_hdu": scene.get("hdu", 1)
                for field in ("reference", "target", "variance")
            },
        }
        for scene in scenes
    ]
    (root / "pairs.yaml").write_text(
        yaml.safe_dump(
            {"schema": "cuphoton.xpois.image-pairs/v2", "pairs": pairs},
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def make_transient_chain(root: Path, data: dict[str, Any]) -> None:
    """Use the existing richer example on one deliberately shifted image."""
    from astropy.io import fits

    example = Path(__file__).resolve().parents[1] / "transient-chain"
    configuration = yaml.safe_load(
        (example / "config-generic.yaml").read_text(encoding="utf-8")
    )
    configuration["io"]["reader"] = "astropy"
    parameters = configuration["survey"]["params"]
    parameters["target"].update(
        image_hdu=1, mask_hdu=1, variance_hdu=1, variance_kind="variance"
    )
    parameters["template"].update(image_hdu=1, on_target_grid=False)
    # WCS reprojection is real; fitting the injected motions as an
    # astrometric correction would change the controlled experiment.
    configuration["align"].update(refine=False, mapping_grid_step=32)
    configuration["subtract"].update(
        basis_sigmas=[1.5], basis_degrees=[0], auto_stamp_mask=False
    )
    configuration["xfit"].update(
        solver={"max_evaluations": XFIT["max_evaluations"]},
        psf_sigma_px=2.5,
    )
    configuration["uq"]["correlation_positions"] = 64
    configuration["classify"].update(
        checkpoint_dir=None,
        apply_noise_match=False,
        noise_reference_sigma=None,
    )
    destination = root / "transient-chain"
    destination.mkdir()
    reprojection = data["reprojection"]
    with fits.open(reprojection["raw_valid_mask"]) as hdus:
        bad = (hdus[1].data == 0).astype(np.uint8)
        header = hdus[1].header.copy()
    bad_path = destination / "raw-target-bad.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(bad, header=header),
        ]
    ).writeto(bad_path)
    (destination / "config.yaml").write_text(
        yaml.safe_dump(configuration, sort_keys=False), encoding="utf-8"
    )
    write_json(
        destination / "selections.json",
        {
            "selections": [
                {
                    "expnum": "demo-00",
                    "ccdnum": "M51",
                    "band": "IRAC1",
                    "field": "Controlled moving-source injections",
                    "image": reprojection["raw_target"],
                    "template": reprojection["reference"],
                    "weight": data["pairs"][0]["variance"],
                    "mask": str(bad_path),
                }
            ]
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source",
        type=Path,
        help="Optional already downloaded public FITS input",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--images", type=int, default=8)
    args = parser.parse_args()
    from data import acquire_image, prepare_images

    root = args.output.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    source = acquire_image(root, local_file=args.source)
    data = prepare_images(root, source, seed=args.seed, images=args.images)
    software = {
        name: importlib.metadata.version(name)
        for name in (
            "cuphoton",
            "numpy",
            "torch",
            "cupy-cuda13x",
            "dragonhpc",
        )
    }
    settings = {
        "seed": args.seed,
        "images": args.images,
        "stamp_size": STAMP_SIZE,
        "xpois": XPOIS,
        "xfit": XFIT,
        "software": software,
        "python": platform.python_version(),
    }
    for name, scenes in (
        ("training", data["corpus"]),
        ("demo", data["pairs"]),
    ):
        collection = root / name
        checked_stage(
            root,
            f"{name}-dataset",
            {
                "settings": settings,
                "scenes": scenes,
                "inputs": source_identity(scenes),
            },
            [collection / "dataset", collection / "subtractions"],
            lambda collection=collection, scenes=scenes, name=name: (
                prepare_dataset(
                    collection, scenes, training=name == "training"
                )
            ),
        )
        prepare_features(root, collection)
    training = root / "training"
    config = yaml.safe_load(
        Path(__file__)
        .with_name("train.example.yaml")
        .read_text(encoding="utf-8")
    )
    config.update(
        dataset_dir=str(training / "dataset"),
        output_root=str(training),
        xfit_feature_dir=str(training / "features"),
        seed=args.seed,
    )
    config_path = training / "training.yaml"
    # This is a generated copy; changing it invalidates the training receipt.
    config_text = yaml.safe_dump(config, sort_keys=False)
    if (
        config_path.exists()
        and config_path.read_text(encoding="utf-8") != config_text
    ):
        raise ValueError(
            "training configuration changed; use a new preparation directory"
        )
    config_path.write_text(config_text, encoding="utf-8")
    model = training / "model"
    checked_stage(
        root,
        "model-training",
        {
            "settings": settings,
            "config": config,
            "features": sha256(training / "features" / "schema.json"),
        },
        [model],
        lambda: command(
            "xscan",
            "train-inada-triplet",
            "--config",
            str(config_path),
        ),
    )
    evaluation = training / "evaluation.json"
    checked_stage(
        root,
        "model-evaluation",
        {
            "checkpoint": sha256(model / "checkpoint.pt"),
            "schema": sha256(training / "features" / "schema.json"),
        },
        [evaluation],
        lambda: evaluate_model(training, evaluation),
    )
    metrics = json.loads(evaluation.read_text(encoding="utf-8"))
    checked_stage(
        root,
        "manifests",
        {
            "checkpoint": sha256(model / "checkpoint.pt"),
            "schema": sha256(training / "features" / "schema.json"),
            "settings": settings,
            "threshold": metrics["decision_threshold"],
            "items": data["pairs"],
        },
        [
            root / "pipeline.json",
            root / "pipeline-native.json",
            root / "pairs.yaml",
        ],
        lambda: make_manifests(
            root, data["pairs"], metrics["decision_threshold"]
        ),
    )
    checked_stage(
        root,
        "transient-chain-inputs",
        {"reprojection": data["reprojection"], "xfit": XFIT},
        [root / "transient-chain"],
        lambda: make_transient_chain(root, data),
    )
    prepared = {
        "schema": "cuphoton.sc26-demo/v1",
        "root": str(root),
        "data_manifest": str(root / "data" / "manifest.json"),
        "pairs_manifest": str(root / "pairs.yaml"),
        "pipeline_manifest": str(root / "pipeline.json"),
        "native_pipeline_manifest": str(root / "pipeline-native.json"),
        "model_dir": str(model),
        "feature_schema": str(training / "features" / "schema.json"),
        "dataset_dir": str(root / "demo" / "dataset"),
        "xfit_input": str(root / "demo" / "dipoles.npz"),
        "xfit_run_dir": str(root / "demo" / "xfit"),
        "xfit_feature_dir": str(root / "demo" / "features"),
        "training_dataset_dir": str(training / "dataset"),
        "evaluation_path": str(evaluation),
        "items": data["pairs"],
        "metrics": metrics,
        "transient_chain": {
            "config": str(root / "transient-chain" / "config.yaml"),
            "selections": str(root / "transient-chain" / "selections.json"),
            "launcher": str(
                Path(__file__).resolve().parents[1]
                / "transient-chain"
                / "launch.py"
            ),
            "checkpoint_dir": None,
            "alignment": "WCS reprojection; stellar refinement disabled",
            "classification": "Omitted; requires an image-only checkpoint",
        },
        "config": {
            **settings,
            "inference_split": "test",
            "batch_size": 32,
            "decision_threshold": metrics["decision_threshold"],
        },
    }
    target = root / "prepared.json"
    if target.exists() and json.loads(
        target.read_text(encoding="utf-8")
    ) != json.loads(json.dumps(prepared)):
        raise ValueError("prepared.json disagrees with validated preparation")
    write_json(target, prepared)
    print(target, flush=True)


if __name__ == "__main__":
    main()
