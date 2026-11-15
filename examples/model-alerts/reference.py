# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Development, freeze and independent evaluation for Model Alerts."""

from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
from baselines import LogisticBaseline, measured_features
from controls import (
    learning_curve_subset,
    permutation_control,
    rejected_split_control,
)
from evaluation import confusion, field_rate_intervals, paired_field_bootstrap
from fusion import fit_feature_bundle
from generator import Design, detect, field_exposures, write_dataset
from run_study import pilot
from study import prepare_training, train_development
from views import gallery_indices


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def rows_at(path):
    return [
        json.loads(line)
        for line in (Path(path) / "metadata.jsonl").read_text().splitlines()
    ]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def content_hashes(dataset):
    """Hash full raw triplets across partitions, independent of row IDs."""
    arrays = [
        np.load(Path(dataset) / f"{name}.npy", mmap_mode="r")
        for name in ("search", "template", "difference")
    ]
    return [
        hashlib.sha256(np.stack(triplet).tobytes()).hexdigest()
        for triplet in zip(*arrays, strict=True)
    ]


def infer(run, dataset, mode, *, feature_dir=None, device="cuda:0"):
    import torch

    from cuphoton.xscan.dataset import StampDataset
    from cuphoton.xscan.training import (
        load_model_from_checkpoint,
        predict_dataset,
    )

    model, _, performance = load_model_from_checkpoint(
        Path(run), device=torch.device(device)
    )
    data = StampDataset(
        Path(dataset),
        input_mode=mode,
        split="test",
        xfit_feature_dir=feature_dir,
    )
    result = predict_dataset(
        model=model,
        dataset=data,
        batch_size=64,
        device=torch.device(device),
        performance=performance,
    )
    expected_rows = rows_at(dataset)
    expected_indices = data.indices.tolist()
    np.testing.assert_array_equal(
        result["labels"],
        np.load(Path(dataset) / "labels.npy")[expected_indices],
    )
    if [r["candidate_id"] for r in result["metadata_rows"]] != [
        expected_rows[i]["candidate_id"] for i in expected_indices
    ]:
        raise AssertionError("native inference row identities changed order")
    return result["probabilities"]


def _experiment(root, *, smoke, device):
    design = Design()
    agreement = {}
    if device.startswith("cuda"):
        exposure = next(field_exposures(9000, design=design))
        cpu_stamps, cpu_rows, cpu_fit, _ = detect(
            exposure, design=design, backend="cpu"
        )
        gpu_stamps, gpu_rows, gpu_fit, _ = detect(
            exposure, design=design, backend="cupy"
        )
        valid = np.isfinite(cpu_fit.residual)
        np.testing.assert_array_equal(valid, np.isfinite(gpu_fit.residual))
        np.testing.assert_allclose(
            cpu_fit.residual[valid],
            gpu_fit.residual[valid],
            rtol=1e-8,
            atol=1e-7,
        )
        if [(r["x"], r["y"]) for r in cpu_rows] != [
            (r["x"], r["y"]) for r in gpu_rows
        ]:
            raise AssertionError(
                "CPU/CUDA detected candidate locations disagree"
            )
        np.testing.assert_allclose(
            cpu_stamps, gpu_stamps, rtol=1e-8, atol=1e-7
        )
        agreement = dict(
            backend_pair=["cpu", "cupy"],
            field=9000,
            candidate_count=len(cpu_rows),
            valid_residual_pixels=int(valid.sum()),
            relative_tolerance=1e-8,
            absolute_tolerance=1e-7,
            maximum_residual_difference=float(
                np.max(
                    np.abs(cpu_fit.residual[valid] - gpu_fit.residual[valid])
                )
            ),
        )
    multiplier = 1
    while True:
        development = root / f"development-{multiplier}"
        result = pilot(
            development, smoke=smoke, device=device, multiplier=multiplier
        )
        if smoke or not result["precision_requires_expansion"]:
            break
        print(
            "Development precision requires doubling fields before freeze.",
            flush=True,
        )
        multiplier *= 2
        if multiplier > 16:
            raise RuntimeError(
                "precision remains inadequate; diagnose development study"
            )
    prepared, raw = development / "prepared", development / "raw"
    epochs = 1 if smoke else 12
    controls = dict(
        registration_diagnostic=result["registration_diagnostic"],
        cpu_cuda_agreement=agreement,
        invalid_split=rejected_split_control(
            prepared, root / "invalid-split"
        ),
    )
    permuted = root / "permuted"
    controls["permutation_design"] = permutation_control(prepared, permuted)
    controls["permutation"] = train_development(
        permuted,
        root / "permutation-run",
        mode="triplet",
        seed=11,
        epochs=epochs,
        device=device,
    )
    controls["permutation_review_required"] = (
        abs(controls["permutation"]["validation_roc_auc"] - 0.5) > 0.15
    )
    if not smoke and controls["permutation_review_required"]:
        raise RuntimeError(
            "permutation control unexpectedly predictive; "
            "diagnose before final freeze, do not change its seed"
        )
    controls["learning_curve"] = []
    for divisor in (4, 2):
        count = max(2, result["train_fields"] // divisor)
        subset_raw, subset = (
            root / f"curve-raw-{divisor}",
            root / f"curve-prepared-{divisor}",
        )
        learning_curve_subset(raw, subset_raw, count)
        prepare_training(subset_raw, subset)
        curve = train_development(
            subset,
            root / f"curve-run-{divisor}",
            mode="triplet",
            seed=11,
            epochs=epochs,
            device=device,
        )
        controls["learning_curve"].append(
            dict(training_fields=count, **curve)
        )
    feature_dir, fit_coverage = fit_feature_bundle(
        prepared,
        root / "fusion-development",
        backend="cupy" if device.startswith("cuda") else "numpy",
    )
    fusion_result = train_development(
        prepared,
        root / "fusion-run",
        mode="triplet",
        seed=11,
        epochs=epochs,
        device=device,
        xfit_feature_dir=feature_dir,
    )
    normalization = json.loads((prepared / "normalization.json").read_text())
    test_count = 3 if smoke else 60 * multiplier
    shift_count = 3 if smoke else 60
    test_base = int(
        os.environ.get(
            "CUPHOTON_MODEL_ALERTS_TEST_SEED_BASE",
            "40000" if smoke else "20000",
        )
    )
    shift_base = int(
        os.environ.get(
            "CUPHOTON_MODEL_ALERTS_SHIFT_SEED_BASE",
            "50000" if smoke else "30000",
        )
    )
    # Read every generated field, including zero-detection fields and earlier
    # precision pilots. Candidate metadata alone would miss empty fields.
    development_fields = {9000}
    for generation_path in root.glob("development-*/raw/generation.json"):
        generation = json.loads(generation_path.read_text())
        development_fields.update(map(int, generation["fields"]))
    test_fields = set(range(test_base, test_base + test_count))
    shift_fields = set(range(shift_base, shift_base + shift_count))
    if (
        min(test_base, shift_base) < 0
        or test_fields & shift_fields
        or (test_fields | shift_fields) & development_fields
    ):
        raise ValueError(
            "reserved seed blocks overlap development or each other"
        )
    streams = [
        0,
        1,
        *range(10, 10 + design.epochs),
        1000,
        *range(1100, 1100 + design.epochs),
    ]
    if len(streams) != len(set(streams)):
        raise ValueError("generator stream roles overlap")
    # Reserved blocks are declared before their first generation. Separate
    # Smoke uses separate blocks; invalidated full reruns need fresh blocks.
    freeze = dict(
        design=asdict(design),
        seed_audit=dict(
            development_fields=sorted(development_fields),
            development_field_count=len(development_fields),
            stream_roles=streams,
            seed_scheme="SeedSequence(root_seed, field, stream)",
            reserved_blocks_disjoint=True,
        ),
        train_fields=result["train_fields"],
        validation_fields=result["validation_fields"],
        test_fields=test_count,
        shifted_fields=shift_count,
        test_seed_block=[test_base, test_base + test_count],
        shift_seed_block=[shift_base, shift_base + shift_count],
        normalization=normalization,
        operating_points=result["outcomes"],
        baselines=result["baselines"],
        fusion_operating_point=fusion_result,
        checkpoints={
            str(p.relative_to(root)): sha(p)
            for p in root.rglob("checkpoint.pt")
        },
        source_sha256={
            str(p.relative_to(Path(__file__).parent)): sha(p)
            for p in Path(__file__).parent.rglob("*")
            if p.is_file()
            and p.suffix in (".py", ".js", ".css", ".html", ".ipynb")
            and "__pycache__" not in p.parts
        },
    )
    write_json(root / "freeze.json", freeze)
    # Nothing above this line generates or infers the reserved populations.
    print(
        "Frozen models and thresholds; generating reserved populations.",
        flush=True,
    )
    report = dict(
        development_directory=str(development.relative_to(root)),
        design=dict(
            profile="quick-smoke" if smoke else "full-reference",
            scientific_performance_qualified=not smoke,
            **asdict(design),
            train_fields=result["train_fields"],
            validation_fields=result["validation_fields"],
            test_fields=test_count,
            shifted_fields=shift_count,
        ),
        training=result["outcomes"],
        baselines=result["baselines"],
        controls=controls,
        fusion=dict(
            development_coverage=fit_coverage, validation=fusion_result
        ),
        freeze=freeze,
        coverage={
            split: dict(
                all_detected=len(split_rows),
                eligible=sum(
                    not r["ambiguous"] and not r["challenge"]
                    for r in split_rows
                ),
                ambiguous=sum(r["ambiguous"] for r in split_rows),
                overlapping_causes=sum(
                    r["overlapping_causes"] for r in split_rows
                ),
                moving_challenge=sum(
                    r["challenge"] and not r["ambiguous"] for r in split_rows
                ),
            )
            for split in ("train", "val")
            for split_rows in [
                [r for r in rows_at(raw) if r["split"] == split]
            ]
        },
        endpoints={},
        uncertainty={},
        shift={},
        moving_source_challenge={},
        breakdowns={"test": {}, "shift": {}},
        high_confidence_mistakes={"test": {}, "shift": {}},
    )
    seen_content = set(content_hashes(raw))
    content_audit = {"development": len(seen_content)}
    for population, fields, shifted in (
        ("test", {test_base + i: "test" for i in range(test_count)}, False),
        ("shift", {shift_base + i: "test" for i in range(shift_count)}, True),
    ):
        population_raw, population_data = (
            root / f"{population}-raw",
            root / f"{population}-prepared",
        )
        write_dataset(
            population_raw,
            fields,
            design=design,
            shifted=shifted,
            backend="cpu",
        )
        digests = content_hashes(population_raw)
        if seen_content.intersection(digests) or len(set(digests)) != len(
            digests
        ):
            raise ValueError(
                "duplicate candidate content across evaluation partitions"
            )
        seen_content.update(digests)
        content_audit[population] = len(digests)
        prepare_training(
            population_raw,
            population_data,
            normalization=normalization,
            eligible_only=False,
        )
        population_features, coverage = fit_feature_bundle(
            population_data,
            root / f"fusion-{population}",
            backend="cupy" if device.startswith("cuda") else "numpy",
        )
        rows = rows_at(population_data)
        labels = np.load(population_data / "labels.npy")
        groups = np.array([r["field"] for r in rows])
        eligible = np.array(
            [not r["ambiguous"] and not r["challenge"] for r in rows]
        )
        movers = np.array(
            [r["challenge"] and not r["ambiguous"] for r in rows]
        )
        raw_images = np.stack(
            [
                np.load(population_raw / f"{n}.npy", mmap_mode="r")
                for n in ("search", "template", "difference")
            ],
            axis=1,
        )
        scores, thresholds = {}, {}
        for outcome in result["outcomes"]:
            key = f"{outcome['mode']}-{outcome['seed']}"
            scores[key] = infer(
                development / key,
                population_data,
                outcome["mode"],
                device=device,
            )
            thresholds[key] = outcome["threshold"]
        if population == "test" and device.startswith("cuda"):
            cpu_scores = infer(
                development / "triplet-11",
                population_data,
                "triplet",
                device="cpu",
            )
            gpu_scores = scores["triplet-11"]
            np.testing.assert_allclose(
                cpu_scores, gpu_scores, rtol=2e-4, atol=2e-5
            )
            report["controls"]["cpu_cuda_inference"] = dict(
                mode="triplet",
                seed=11,
                rows=len(cpu_scores),
                maximum_score_difference=float(
                    np.max(np.abs(cpu_scores - gpu_scores))
                ),
                threshold_decision_flips=int(
                    np.sum(
                        (cpu_scores >= thresholds["triplet-11"])
                        != (gpu_scores >= thresholds["triplet-11"])
                    )
                ),
                relative_tolerance=2e-4,
                absolute_tolerance=2e-5,
            )
        scores["fusion-11"] = infer(
            root / "fusion-run",
            population_data,
            "triplet",
            feature_dir=population_features,
            device=device,
        )
        thresholds["fusion-11"] = fusion_result["threshold"]
        measured = measured_features(raw_images, [r["snr"] for r in rows])
        for name, baseline in result["baselines"].items():
            model = LogisticBaseline()
            for attribute in ("center", "scale", "weights"):
                setattr(
                    model, attribute, np.array(baseline["model"][attribute])
                )
            columns = (
                [0] if name == "snr_only" else list(range(measured.shape[1]))
            )
            scores[name] = model.predict(measured[:, columns])
            thresholds[name] = baseline["threshold"]
        metrics = {
            key: confusion(labels[eligible], value[eligible], thresholds[key])
            for key, value in scores.items()
        }
        report["coverage"][population] = dict(
            all_detected=len(rows),
            eligible=int(eligible.sum()),
            ambiguous=sum(r["ambiguous"] for r in rows),
            overlapping_causes=sum(r["overlapping_causes"] for r in rows),
            moving_challenge=int(movers.sum()),
            fit=coverage,
        )
        report["moving_source_challenge"][population] = {
            key: dict(
                count=int(movers.sum()),
                retained=int(np.sum(value[movers] >= thresholds[key])),
            )
            for key, value in scores.items()
        }
        generation = json.loads(
            (population_raw / "generation.json").read_text()
        )
        injection_families = {
            identity: family
            for recovery in generation["recovery"]
            for identity, family in recovery["injection_families"].items()
        }
        injected = sum(r["injected"] for r in generation["recovery"])
        detected = sum(r["detected"] for r in generation["recovery"])
        detected_ids = {
            identity for r in rows for identity in r["associated_injections"]
        }
        eligible_ids = {
            identity
            for i, r in enumerate(rows)
            if eligible[i]
            for identity in r["associated_injections"]
        }
        if len(detected_ids) != detected:
            raise AssertionError(
                "recovery and candidate association disagree"
            )
        for key, value in scores.items():
            retained_ids = {
                identity
                for i, r in enumerate(rows)
                if eligible[i] and value[i] >= thresholds[key]
                for identity in r["associated_injections"]
            }
            metrics[key]["injections"] = dict(
                injected=injected,
                detected=detected,
                eligible_detected=len(eligible_ids),
                ineligible_only_detected=len(detected_ids - eligible_ids),
                classifier_rejected_eligible=len(eligible_ids - retained_ids),
                retained=len(retained_ids),
                detection_recovery=detected / injected,
                retained_fraction=len(retained_ids) / injected,
                by_family={
                    family: dict(
                        injected=len(identities),
                        detected=len(identities & detected_ids),
                        eligible_detected=len(identities & eligible_ids),
                        ineligible_only_detected=len(
                            identities & (detected_ids - eligible_ids)
                        ),
                        classifier_rejected_eligible=len(
                            identities & (eligible_ids - retained_ids)
                        ),
                        retained=len(identities & retained_ids),
                        detection_recovery=len(identities & detected_ids)
                        / len(identities),
                        retained_fraction=len(identities & retained_ids)
                        / len(identities),
                    )
                    for family in sorted(set(injection_families.values()))
                    for identities in [
                        {
                            identity
                            for identity, f in injection_families.items()
                            if f == family
                        }
                    ]
                },
            )
        target = "endpoints" if population == "test" else "shift"
        report[target] = metrics
        report["uncertainty"][population + "_model_field_intervals"] = {
            key: field_rate_intervals(
                labels[eligible],
                value[eligible],
                groups[eligible],
                thresholds[key],
            )
            for key, value in scores.items()
        }
        if population == "test":
            for outcome in result["outcomes"]:
                if outcome["mode"] != "pair":
                    continue
                seed = outcome["seed"]
                a = f"pair-{seed}"
                b = f"triplet-{seed}"
                report["uncertainty"][str(seed)] = paired_field_bootstrap(
                    labels[eligible],
                    scores[a][eligible],
                    scores[b][eligible],
                    groups[eligible],
                    thresholds[a],
                    thresholds[b],
                )
            selected_rows = [
                r for r, keep in zip(rows, eligible, strict=True) if keep
            ]
            selected_scores = {
                key: value[eligible] for key, value in scores.items()
            }
            selected_labels = labels[eligible]
            score_key = "triplet-11"
            gallery = gallery_indices(
                selected_rows,
                selected_labels,
                selected_scores[score_key],
                thresholds[score_key],
            )
            np.savez(
                root / "evaluation.npz",
                labels=selected_labels,
                snr=np.array([r["snr"] for r in selected_rows]),
                candidate_id=np.array(
                    [str(r["candidate_id"]) for r in selected_rows]
                ),
                family=np.array([r["family"] for r in selected_rows]),
                **selected_scores,
            )
            np.savez(
                root / "gallery.npz",
                images=raw_images[eligible][gallery],
                labels=selected_labels[gallery],
                scores=selected_scores[score_key][gallery],
            )
            write_json(
                root / "gallery-rows.json",
                [selected_rows[i] for i in gallery],
            )
            report["score_arrays"] = {key: key for key in scores}
            report["inspector_threshold"] = thresholds[score_key]
            report["inspector_score_key"] = score_key
            report["gallery_rule"] = dict(
                indices=gallery.tolist(),
                maximum=96,
                rule="round-robin confusion/SNR/family, candidate-ID order",
            )
        selected_rows = [
            r for r, keep in zip(rows, eligible, strict=True) if keep
        ]
        selected_scores = {
            key: value[eligible] for key, value in scores.items()
        }
        selected_labels = labels[eligible]
        for key, value in selected_scores.items():
            by_family = {}
            for family in sorted({r["family"] for r in selected_rows}):
                mask = np.array(
                    [r["family"] == family for r in selected_rows]
                )
                by_family[family] = confusion(
                    selected_labels[mask], value[mask], thresholds[key]
                )
            by_snr = {}
            snr = np.array([r["snr"] for r in selected_rows])
            for lo, hi in ((4, 5), (5, 10), (10, 20), (20, float("inf"))):
                mask = (snr >= lo) & (snr < hi)
                by_snr[f"{lo}:{hi}"] = (
                    confusion(
                        selected_labels[mask],
                        value[mask],
                        thresholds[key],
                    )
                    if mask.any()
                    else dict(count=0)
                )
            report["breakdowns"][population][key] = dict(
                family=by_family,
                snr=by_snr,
                overlap={
                    str(overlap): confusion(
                        selected_labels[mask], value[mask], thresholds[key]
                    )
                    for overlap in (False, True)
                    for mask in [
                        np.array(
                            [
                                r["overlapping_causes"] == overlap
                                for r in selected_rows
                            ]
                        )
                    ]
                    if mask.any()
                },
            )
            wrong = (value >= thresholds[key]) != selected_labels.astype(bool)
            order = np.argsort(
                -np.abs(value - selected_labels), kind="stable"
            )
            report["high_confidence_mistakes"][population][key] = [
                dict(
                    candidate_id=selected_rows[i]["candidate_id"],
                    score=float(value[i]),
                    label=int(selected_labels[i]),
                    family=selected_rows[i]["family"],
                )
                for i in order
                if wrong[i]
            ][:12]
    import torch

    report["provenance"] = dict(
        hardware={
            "accelerator": torch.cuda.get_device_name(0)
            if device.startswith("cuda")
            else "CPU"
        },
        source_sha256=freeze["source_sha256"],
        cuphoton_revision=os.environ.get("CUPHOTON_MODEL_ALERTS_REVISION")
        or subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        dependencies={
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "scipy",
                "torch",
                "astropy",
                "bokeh",
                "pyarrow",
            )
        },
        artifact_sha256={
            str(p.relative_to(root)): sha(p) for p in root.glob("*.npz")
        },
        content_audit=dict(
            partition_counts=content_audit, all_triplets_unique=True
        ),
        network_required_after_provisioning=False,
    )
    if device.startswith("cuda"):
        report["provenance"]["dependencies"]["cupy-cuda13x"] = (
            importlib.metadata.version("cupy-cuda13x")
        )
    report["artifact_sizes"] = {
        str(p.relative_to(root)): p.stat().st_size
        for p in root.glob("*")
        if p.is_file()
    }
    write_json(root / "report.json", report)
    return report


def execute_reference(output, *, profile="quick-smoke", device="cuda:0"):
    """Execute once in a new directory, retaining the shared GPU lease."""
    if profile not in ("quick-smoke", "full-reference"):
        raise ValueError("profile must be quick-smoke or full-reference")
    if profile == "full-reference" and not device.startswith("cuda"):
        raise ValueError(
            "full-reference qualification requires CUDA execution"
        )
    root = Path(output).resolve()
    source = Path(__file__).resolve().parent
    if repo := os.environ.get("CUPHOTON_MODEL_ALERTS_SOURCE_REPO"):
        source = Path(repo).resolve()
    else:
        # Extracted ZIPs may have no Git ancestor. Never assume a fixed depth.
        source = next(
            (p for p in source.parents if (p / ".git").exists()), source
        )
    if (
        root == source
        or root.is_relative_to(source)
        or root == Path(root.anchor)
    ):
        raise ValueError(
            "output must be outside example sources and filesystem root"
        )
    if root.exists():
        raise ValueError("reference output must not exist")
    lock_path = os.environ.get("CUPHOTON_MODEL_ALERTS_GPU_LOCK")
    if device.startswith("cuda") and not lock_path:
        raise ValueError(
            "CUDA execution requires CUPHOTON_MODEL_ALERTS_GPU_LOCK"
        )
    root.mkdir(parents=True)
    if device.startswith("cuda"):
        with Path(lock_path).open("a") as lock:
            print("Waiting for shared GPU lock", flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
            print("Acquired shared GPU lock", flush=True)
            return _experiment(
                root, smoke=profile == "quick-smoke", device=device
            )
    return _experiment(root, smoke=profile == "quick-smoke", device=device)
