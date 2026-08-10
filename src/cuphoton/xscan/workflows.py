# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Workflow helpers for XScan."""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

from cuphoton.core.cli import ApplicationContext

from .butler import build_hsc_fits_registry
from .config import dump_config, load_training_config
from .dataset import (
    StampDataset,
    build_dataset_from_manifest,
    build_hsc_dataset_from_manifest,
    build_hsc_synthetic_dataset,
    inspect_dataset_dir,
    load_metadata_rows,
    merge_dataset_dirs,
    validate_dataset_dir,
)
from .des import (
    build_autoscan_dataset_from_raw,
    build_nodiff_dataset_from_raw,
    build_nodiff_release_dataset_from_manifest,
)
from .lsstcomcam import (
    build_lsstcomcam_smoke_dataset_from_manifest,
    check_lsstcomcam_candidate_catalog_from_manifest,
    plan_lsstcomcam_staging_from_manifest,
    stage_lsstcomcam_fits_from_manifest,
)
from .metrics import (
    ensure_binary_labels,
    evaluate_predictions,
    select_threshold_for_scores,
)
from .review import (
    aggregate_entity_review_annotations,
    aggregate_review_annotations,
    apply_review_annotations,
    build_dataset_review_queue,
    build_entity_review_queue,
    build_review_queue,
    entity_review_bokeh_server_summary,
    export_review_annotation_template,
    export_review_contact_sheets,
    import_review_annotations_from_csv,
    review_bokeh_server_summary,
    summarize_review_status,
)
from .training import (
    XFIT_COVERAGE_MISMATCH_THRESHOLD,
    check_training_label_provenance,
    load_model_from_checkpoint,
    predict_dataset,
    resolve_device,
    train_classifier,
    xfit_fit_coverage,
)
from .types import InputMode, MissingPolicy

if TYPE_CHECKING:
    from .xfit_features import XFitFeatureMatrix


@dataclass(slots=True)
class WorkflowResult:
    """Persisted XScan workflow result.

    Attributes
    ----------
    run_dir
        Directory owning the workflow artifacts.
    summary
        JSON-compatible workflow inputs, resolved execution details, and
        artifact paths.
    """

    run_dir: Path
    summary: dict[str, Any]


_SWEEP_VARIANT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_HSC_XPOIS_SWEEP_REQUIRED_FIELDS = (
    "difference_context_size",
    "xpois_kernel_shape",
    "xpois_basis_sigmas",
    "xpois_basis_degrees",
    "xpois_background_degree",
    "xpois_flux_conserve",
    "xpois_use_variance",
)
STRATIFIED_BREAKDOWN_KEYS = (
    "center_source_breakdown",
    "catalog_pool_role_breakdown",
    "catalog_morphology_breakdown",
    "negative_difficulty_breakdown",
    "mask_pressure_breakdown",
)
PAIR_TRIPLET_CONTROL_FIELDS = (
    "xfit_feature_dir",
    "use_xfit_features",
    "epochs",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "device",
    "training_mode",
    "pretrain_checkpoint",
    "freeze_encoder_stages",
    "train_split",
    "val_split",
    "eval_split",
    "early_stopping_metric",
    "early_stopping_patience",
    "early_stopping_min_delta",
)
STRATIFIED_BREAKDOWN_TITLES = {
    "center_source_breakdown": "Center Source",
    "catalog_pool_role_breakdown": "Catalog Pool Role",
    "catalog_morphology_breakdown": "Catalog Morphology",
    "negative_difficulty_breakdown": "Negative Difficulty",
    "mask_pressure_breakdown": "Mask Pressure",
}
COMPARISON_EVAL_METRIC_KEYS = (
    "sample_count",
    "positive_count",
    "negative_count",
    "threshold",
    "tpr_at_fpr_1pct",
    "tpr_at_fpr_5pct",
    "brier_score",
    "metric_undefined_reason",
    "confusion",
    "calibration",
    "threshold_diagnostics",
    "threshold_selection_undefined_reason",
)


def default_output_root() -> Path:
    return ApplicationContext.for_component("xscan").runs_dir


def resolve_run_dir(
    output_root: Path | None,
    run_name: str | None,
    prefix: str,
) -> Path:
    root = (output_root or default_output_root()).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    label = (
        Path(run_name).name
        if run_name
        else datetime.now().strftime(f"{prefix}-%Y%m%d-%H%M%S-%f")
    )
    run_dir = root / label
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_summary(run_dir: Path, summary: dict[str, Any]) -> None:
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def check_lsstcomcam_candidates_workflow(
    *,
    manifest_path: Path,
    strict: bool = False,
) -> dict[str, Any]:
    return check_lsstcomcam_candidate_catalog_from_manifest(
        manifest_path=manifest_path,
        strict=strict,
    )


def plan_lsstcomcam_staging_workflow(
    *,
    manifest_path: Path,
    sample_count: int | None = None,
) -> dict[str, Any]:
    return plan_lsstcomcam_staging_from_manifest(
        manifest_path=manifest_path,
        sample_count=sample_count,
    )


def stage_lsstcomcam_fits_workflow(
    *,
    manifest_path: Path,
    source_prefix: str,
    target_prefix: Path,
    search_roots: list[Path],
    sample_count: int | None = None,
    link_mode: str = "symlink",
    duplicate_policy: str = "same-size",
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    return stage_lsstcomcam_fits_from_manifest(
        manifest_path=manifest_path,
        source_prefix=source_prefix,
        target_prefix=target_prefix,
        search_roots=search_roots,
        sample_count=sample_count,
        link_mode=link_mode,
        duplicate_policy=duplicate_policy,
        force=force,
        dry_run=dry_run,
    )


def check_training_labels_workflow(*, dataset_dir: Path) -> dict[str, Any]:
    return check_training_label_provenance(dataset_dir.expanduser().resolve())


def load_hsc_xpois_sweep_variants(
    sweep_config_path: Path,
) -> list[dict[str, Any]]:
    payload = (
        yaml.safe_load(sweep_config_path.read_text(encoding="utf-8")) or {}
    )
    if isinstance(payload, list):
        variant_rows = payload
    elif isinstance(payload, dict):
        variant_rows = payload.get("variants")
    else:
        raise ValueError(
            "sweep config must be a YAML list or a mapping with a "
            "'variants' list"
        )
    if not isinstance(variant_rows, list) or not variant_rows:
        raise ValueError("sweep config must define at least one variant")

    variants: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(variant_rows):
        if not isinstance(row, dict):
            raise ValueError(
                f"sweep variant at index {index} must be a mapping"
            )
        name = str(row.get("name", "")).strip()
        if not name:
            raise ValueError(
                f"sweep variant at index {index} is missing required "
                "field 'name'"
            )
        if not _SWEEP_VARIANT_NAME_RE.fullmatch(name):
            raise ValueError(
                "variant names must match "
                f"{_SWEEP_VARIANT_NAME_RE.pattern!r}; got {name!r}"
            )
        if name in seen:
            raise ValueError(f"duplicate sweep variant name: {name}")
        missing = [
            key for key in _HSC_XPOIS_SWEEP_REQUIRED_FIELDS if key not in row
        ]
        if missing:
            raise ValueError(
                f"sweep variant {name!r} is missing required fields: "
                + ", ".join(sorted(missing))
            )
        variants.append(
            {
                "name": name,
                **{key: row[key] for key in _HSC_XPOIS_SWEEP_REQUIRED_FIELDS},
            }
        )
        seen.add(name)
    return variants


def compute_difference_diagnostics(
    *,
    reference_dataset_dir: Path,
    dataset_dir: Path,
) -> dict[str, Any]:
    reference = np.load(
        reference_dataset_dir / "difference.npy",
        allow_pickle=False,
    )
    current = np.load(
        dataset_dir / "difference.npy",
        allow_pickle=False,
    )
    if reference.shape != current.shape:
        raise ValueError(
            "difference diagnostics require matching array shapes; got "
            f"{reference.shape} and {current.shape}"
        )
    if not bool(np.isfinite(reference).all()):
        invalid = int(np.size(reference) - np.isfinite(reference).sum())
        raise ValueError(
            f"reference difference array contains {invalid} non-finite values"
        )
    if not bool(np.isfinite(current).all()):
        invalid = int(np.size(current) - np.isfinite(current).sum())
        raise ValueError(
            f"difference array contains {invalid} non-finite values"
        )
    delta = np.abs(
        np.asarray(reference, dtype=np.float32)
        - np.asarray(current, dtype=np.float32)
    )
    per_sample_mean = delta.reshape(delta.shape[0], -1).mean(axis=1)
    return {
        "reference_variant": "triplet_simple",
        "sample_count": int(delta.shape[0]),
        "mean_abs_delta": float(delta.mean()),
        "median_abs_delta": float(np.median(delta)),
        "p95_per_sample_mean_abs_delta": float(
            np.percentile(per_sample_mean, 95)
        ),
        "allclose": bool(np.allclose(reference, current)),
    }


def summarize_hsc_job_runs(
    *,
    input_mode: str,
    training_mode: str,
    job_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    # Runs whose eval split was single-class have roc_auc/pr_auc == None
    # (undefined). Legacy summaries can also contain non-finite values.
    # Normalize both cases before aggregation and ranking.
    job_runs = normalize_run_metrics(job_runs)
    accuracy_values = finite_numeric_values(job_runs, "accuracy")
    defined_runs = [
        run
        for run in job_runs
        if run.get("roc_auc") is not None and run.get("pr_auc") is not None
    ]
    roc_values = finite_numeric_values(defined_runs, "roc_auc")
    pr_values = finite_numeric_values(defined_runs, "pr_auc")
    defined_accuracy_values = finite_numeric_values(defined_runs, "accuracy")
    result = {
        "input_mode": input_mode,
        "training_mode": training_mode,
        "runs": job_runs,
        "defined_run_count": len(defined_runs),
        "mean_roc_auc": float(np.mean(roc_values)) if roc_values else None,
        "std_roc_auc": float(np.std(roc_values)) if roc_values else None,
        "mean_pr_auc": float(np.mean(pr_values)) if pr_values else None,
        "std_pr_auc": float(np.std(pr_values)) if pr_values else None,
        "mean_accuracy": (
            float(np.mean(accuracy_values)) if accuracy_values else None
        ),
        "std_accuracy": (
            float(np.std(accuracy_values)) if accuracy_values else None
        ),
        "mean_defined_auc_accuracy": (
            float(np.mean(defined_accuracy_values))
            if defined_accuracy_values
            else None
        ),
        "best_run_dir": (
            max(
                defined_runs,
                key=lambda item: (
                    float(item["roc_auc"]),
                    float(item["pr_auc"]),
                ),
            )["run_dir"]
            if defined_runs
            else None
        ),
        "stratified_breakdowns": summarize_grouped_breakdowns(job_runs),
    }
    result["mean_fixed_threshold_accuracy"] = result["mean_accuracy"]
    result["std_fixed_threshold_accuracy"] = result["std_accuracy"]
    add_optional_run_stat(result, job_runs, "brier_score")
    add_optional_run_stat(result, job_runs, "tpr_at_fpr_1pct")
    add_optional_run_stat(result, job_runs, "tpr_at_fpr_5pct")
    confusion = summarize_confusion_totals(job_runs)
    if confusion is not None:
        result["confusion_totals"] = confusion
    calibration = summarize_calibration(job_runs)
    if calibration is not None:
        result["calibration_summary"] = calibration
    thresholds = finite_numeric_values(job_runs, "threshold")
    if thresholds:
        result["thresholds"] = thresholds
    return result


def summarize_reproduction_runs(
    job_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    job_runs = normalize_run_metrics(job_runs)
    roc_values = finite_numeric_values(job_runs, "roc_auc")
    accuracy_values = finite_numeric_values(job_runs, "accuracy")
    return {
        "runs": job_runs,
        "mean_roc_auc": float(np.mean(roc_values)) if roc_values else None,
        "std_roc_auc": float(np.std(roc_values)) if roc_values else None,
        "mean_accuracy": (
            float(np.mean(accuracy_values)) if accuracy_values else None
        ),
        "std_accuracy": (
            float(np.std(accuracy_values)) if accuracy_values else None
        ),
    }


def extract_comparison_eval_metrics(
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        key: summary[key]
        for key in COMPARISON_EVAL_METRIC_KEYS
        if summary.get(key) is not None
    }


def add_optional_run_stat(
    result: dict[str, Any],
    runs: list[dict[str, Any]],
    field: str,
) -> None:
    values = finite_numeric_values(runs, field)
    if not values:
        return
    result[f"mean_{field}"] = float(np.mean(values))
    result[f"std_{field}"] = float(np.std(values))


def _descending_key(value: Any) -> float:
    numeric = finite_numeric_value(value)
    return -numeric if numeric is not None else float("inf")


def rank_hsc_jobs(
    jobs: dict[str, dict[str, Any]],
) -> tuple[list[str], list[str]]:
    ranked = sorted(
        (
            (label, payload)
            for label, payload in jobs.items()
            if finite_numeric_value(payload.get("mean_roc_auc")) is not None
            and finite_numeric_value(payload.get("mean_pr_auc")) is not None
            and int(payload.get("defined_run_count", 0)) > 0
        ),
        key=lambda item: (
            _descending_key(item[1].get("mean_roc_auc")),
            _descending_key(item[1].get("mean_pr_auc")),
            _descending_key(item[1].get("mean_defined_auc_accuracy")),
        ),
    )
    ranking = [label for label, _payload in ranked]
    ranked_labels = set(ranking)
    unranked = [label for label in jobs if label not in ranked_labels]
    return ranking, unranked


def finite_numeric_values(
    rows: list[dict[str, Any]],
    field: str,
) -> list[float]:
    values = []
    for row in rows:
        numeric = finite_numeric_value(row.get(field))
        if numeric is not None:
            values.append(numeric)
    return values


def finite_numeric_value(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def normalize_run_metrics(
    runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized_runs = []
    for run in runs:
        normalized = {
            key: normalize_nonfinite_numbers(value)
            for key, value in run.items()
        }
        for field in ("roc_auc", "pr_auc", "accuracy"):
            if field in normalized:
                normalized[field] = finite_numeric_value(normalized[field])
        normalized_runs.append(normalized)
    return normalized_runs


def normalize_nonfinite_numbers(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: normalize_nonfinite_numbers(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [normalize_nonfinite_numbers(item) for item in value]
    if isinstance(value, np.generic):
        return normalize_nonfinite_numbers(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def summarize_confusion_totals(
    runs: list[dict[str, Any]],
) -> dict[str, int] | None:
    totals = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    count = 0
    for run in runs:
        confusion = run.get("confusion")
        if not isinstance(confusion, dict):
            continue
        try:
            values = {key: int(confusion[key]) for key in totals}
        except (KeyError, TypeError, ValueError):
            continue
        for key, value in values.items():
            totals[key] += value
        count += 1
    if count == 0:
        return None
    totals["runs"] = count
    return totals


def summarize_calibration(
    runs: list[dict[str, Any]],
) -> dict[str, float | int] | None:
    rows = [
        row["calibration"]
        for row in runs
        if isinstance(row.get("calibration"), dict)
    ]
    if not rows:
        return None
    result: dict[str, float | int] = {"runs": len(rows)}
    for field in (
        "brier_score",
        "expected_calibration_error",
        "max_calibration_error",
    ):
        values = finite_numeric_values(rows, field)
        if values:
            result[f"mean_{field}"] = float(np.mean(values))
            result[f"std_{field}"] = float(np.std(values))
    return result


def summarize_grouped_breakdowns(
    job_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in STRATIFIED_BREAKDOWN_KEYS:
        grouped: dict[str, list[dict[str, Any]]] = {}
        field = None
        for run in job_runs:
            payload = run.get(key)
            if not isinstance(payload, dict):
                continue
            field = field or payload.get("field")
            for group_name, group_payload in (
                payload.get("groups") or {}
            ).items():
                if isinstance(group_payload, dict):
                    grouped.setdefault(str(group_name), []).append(
                        group_payload
                    )
        if not grouped:
            continue
        result[key] = {
            "field": field,
            "groups": {
                group_name: summarize_breakdown_group(group_runs)
                for group_name, group_runs in sorted(grouped.items())
            },
        }
    return result


def summarize_breakdown_group(
    group_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "runs": len(group_runs),
        "total_count": int(
            sum(int(row.get("count", 0)) for row in group_runs)
        ),
        "mean_count": mean_numeric_field(group_runs, "count"),
        "mean_positive_count": mean_numeric_field(
            group_runs, "positive_count"
        ),
        "mean_negative_count": mean_numeric_field(
            group_runs, "negative_count"
        ),
        "mean_probability": mean_numeric_field(
            group_runs, "mean_probability"
        ),
        "mean_accuracy": mean_numeric_field(group_runs, "accuracy"),
        "mean_positive_recovery_rate": mean_numeric_field(
            group_runs, "positive_recovery_rate"
        ),
        "mean_negative_false_positive_rate": mean_numeric_field(
            group_runs, "negative_false_positive_rate"
        ),
    }


def mean_numeric_field(
    rows: list[dict[str, Any]],
    field: str,
) -> float | None:
    values = []
    for row in rows:
        value = row.get(field)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            values.append(numeric)
    if not values:
        return None
    return float(np.mean(values))


def inspect_dataset_workflow(
    *,
    dataset_dir: Path,
    dataset_kind: str | None,
) -> dict[str, Any]:
    return inspect_dataset_dir(dataset_dir, dataset_kind=dataset_kind)


def validate_dataset_workflow(
    *,
    dataset_dir: Path,
    dataset_kind: str | None,
) -> dict[str, Any]:
    return validate_dataset_dir(dataset_dir, dataset_kind=dataset_kind)


def export_xfit_input_workflow(
    *, dataset_dir: Path, output_path: Path
) -> dict[str, Any]:
    """Export exact XScan difference stamps for the xFit CLI."""

    from .xfit_features import export_xfit_input

    return export_xfit_input(
        dataset_dir=dataset_dir,
        output_path=output_path,
    )


def build_xfit_feature_bundle_workflow(
    *,
    dataset_dir: Path,
    xfit_run_dir: Path,
    output_dir: Path,
    missing_policy: MissingPolicy = "error",
) -> WorkflowResult:
    """Convert portable xFit artifacts into an ordered XScan sidecar."""

    from .xfit_features import build_xfit_feature_bundle

    bundle = build_xfit_feature_bundle(
        dataset_dir=dataset_dir,
        xfit_run_dir=xfit_run_dir,
        output_dir=output_dir,
        missing_policy=missing_policy,
    )
    resolved_output = output_dir.expanduser().resolve()
    summary = {
        "workflow": "build-xfit-features",
        "output_dir": str(resolved_output),
        **bundle,
    }
    return WorkflowResult(
        run_dir=resolved_output,
        summary=summary,
    )


def merge_dataset_workflow(
    *,
    dataset_dirs: list[Path],
    output_dir: Path,
    dataset_kind: str | None = None,
) -> WorkflowResult:
    result = merge_dataset_dirs(
        dataset_dirs=dataset_dirs,
        output_dir=output_dir,
        dataset_kind=dataset_kind,
    )
    return WorkflowResult(run_dir=result.output_dir, summary=result.summary)


def build_prepared_dataset_workflow(
    *,
    manifest_path: Path,
    output_dir: Path,
    dataset_kind: str,
) -> WorkflowResult:
    result = build_dataset_from_manifest(
        manifest_path=manifest_path,
        output_dir=output_dir,
        dataset_kind=dataset_kind,
    )
    return WorkflowResult(run_dir=result.output_dir, summary=result.summary)


def build_raw_autoscan_workflow(
    *,
    manifest_path: Path,
    output_dir: Path,
) -> WorkflowResult:
    result = build_autoscan_dataset_from_raw(
        manifest_path=manifest_path,
        output_dir=output_dir,
    )
    return WorkflowResult(run_dir=result.output_dir, summary=result.summary)


def build_raw_nodiff_workflow(
    *,
    manifest_path: Path,
    output_dir: Path,
) -> WorkflowResult:
    result = build_nodiff_dataset_from_raw(
        manifest_path=manifest_path,
        output_dir=output_dir,
    )
    return WorkflowResult(run_dir=result.output_dir, summary=result.summary)


def build_nodiff_release_workflow(
    *,
    manifest_path: Path,
    output_dir: Path,
) -> WorkflowResult:
    result = build_nodiff_release_dataset_from_manifest(
        manifest_path=manifest_path,
        output_dir=output_dir,
    )
    return WorkflowResult(run_dir=result.output_dir, summary=result.summary)


def build_hsc_workflow(
    *,
    manifest_path: Path,
    output_dir: Path,
) -> WorkflowResult:
    result = build_hsc_dataset_from_manifest(
        manifest_path=manifest_path,
        output_dir=output_dir,
    )
    return WorkflowResult(run_dir=result.output_dir, summary=result.summary)


def build_lsstcomcam_smoke_workflow(
    *,
    manifest_path: Path,
    output_dir: Path,
) -> WorkflowResult:
    result = build_lsstcomcam_smoke_dataset_from_manifest(
        manifest_path=manifest_path,
        output_dir=output_dir,
    )
    return WorkflowResult(run_dir=result.output_dir, summary=result.summary)


def build_hsc_registry_workflow(
    *,
    fits_root: Path,
    output_path: Path,
    hsc_npy_dir: Path | None,
    butler_run: str,
) -> WorkflowResult:
    result = build_hsc_fits_registry(
        fits_root=fits_root,
        output_path=output_path,
        hsc_npy_dir=hsc_npy_dir,
        butler_run=butler_run,
    )
    return WorkflowResult(
        run_dir=result.registry_path.parent,
        summary=result.summary,
    )


def build_experimental_hsc_workflow(
    *,
    base: str | Path,
    output_dir: Path,
    positive_count: int,
    negative_count: int,
    stamp_size: int,
    seed: int,
    template_source: str,
    include_difference: bool,
    tile_size: int,
) -> WorkflowResult:
    result = build_hsc_synthetic_dataset(
        base=base,
        output_dir=output_dir,
        positive_count=positive_count,
        negative_count=negative_count,
        stamp_size=stamp_size,
        seed=seed,
        tile_size=tile_size,
        template_source=template_source,
        include_difference=include_difference,
    )
    return WorkflowResult(run_dir=result.output_dir, summary=result.summary)


def train_workflow(
    config_path: Path,
    *,
    input_mode_override: InputMode | None = None,
) -> WorkflowResult:
    config = load_training_config(config_path)
    if input_mode_override is not None:
        config.model.input_mode = input_mode_override
    output_root = (
        Path(config.output_root).expanduser()
        if config.output_root is not None
        else None
    )
    run_prefix = (
        f"train-inada-{config.model.input_mode}"
        if input_mode_override is not None
        else "train"
    )
    run_dir = resolve_run_dir(output_root, config.run_name, run_prefix)
    summary = train_classifier(config, run_dir=run_dir)
    summary.update(
        {
            "workflow": "train",
            "config_path": str(config_path.expanduser().resolve()),
            "input_mode": config.model.input_mode,
        }
    )
    write_summary(run_dir, summary)
    return WorkflowResult(run_dir=run_dir, summary=summary)


def _checkpoint_dataset(
    *,
    checkpoint: dict[str, Any],
    dataset_dir: Path,
    split: str,
    xfit_feature_matrix: XFitFeatureMatrix | None,
) -> StampDataset:
    model_config = checkpoint["model_config"]
    feature_names = tuple(model_config.get("xfit_feature_names") or ())
    dataset = StampDataset(
        dataset_dir,
        input_mode=model_config["input_mode"],
        split=split,
        xfit_feature_matrix=xfit_feature_matrix,
        xfit_feature_names=feature_names or None,
    )
    return dataset


def _checkpoint_xfit_feature_matrix(
    *,
    checkpoint: dict[str, Any],
    dataset_dir: Path,
    xfit_feature_dir: Path | None,
    use_xfit_features: bool,
) -> XFitFeatureMatrix | None:
    """Resolve the explicit fusion opt-in and load its bundle once."""

    if not isinstance(use_xfit_features, bool):
        raise ValueError("use_xfit_features must be a boolean")
    model_config = checkpoint["model_config"]
    feature_names = tuple(model_config.get("xfit_feature_names") or ())
    if use_xfit_features and xfit_feature_dir is None:
        raise ValueError("--use-xfit-features requires --xfit-feature-dir")
    if not use_xfit_features and xfit_feature_dir is not None:
        raise ValueError("--xfit-feature-dir requires --use-xfit-features")
    if feature_names and not use_xfit_features:
        raise ValueError(
            "fusion checkpoint requires --use-xfit-features and "
            "--xfit-feature-dir for this dataset"
        )
    if not feature_names and use_xfit_features:
        raise ValueError(
            "--use-xfit-features cannot be used with an image-only checkpoint"
        )
    if not use_xfit_features:
        return None
    from .xfit_features import load_xfit_feature_matrix

    assert xfit_feature_dir is not None
    return load_xfit_feature_matrix(
        dataset_dir=dataset_dir,
        feature_dir=xfit_feature_dir,
        expected_feature_names=feature_names,
    )


def infer_workflow(
    *,
    run_dir: Path,
    dataset_dir: Path,
    split: str,
    batch_size: int = 32,
    xfit_feature_dir: Path | None = None,
    use_xfit_features: bool = False,
) -> WorkflowResult:
    device = resolve_device("auto")
    run_dir = run_dir.expanduser().resolve()
    dataset_dir = dataset_dir.expanduser().resolve()
    if xfit_feature_dir is not None:
        xfit_feature_dir = xfit_feature_dir.expanduser().resolve()
    model, checkpoint, performance = load_model_from_checkpoint(
        run_dir,
        device=device,
    )
    xfit_feature_matrix = _checkpoint_xfit_feature_matrix(
        checkpoint=checkpoint,
        dataset_dir=dataset_dir,
        xfit_feature_dir=xfit_feature_dir,
        use_xfit_features=use_xfit_features,
    )
    dataset = _checkpoint_dataset(
        checkpoint=checkpoint,
        dataset_dir=dataset_dir,
        split=split,
        xfit_feature_matrix=xfit_feature_matrix,
    )
    payload = predict_dataset(
        model=model,
        dataset=dataset,
        batch_size=batch_size,
        device=device,
        performance=performance,
    )
    output_dir = run_dir / "inference" / split
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "logits.npy", payload["logits"], allow_pickle=False)
    np.save(output_dir / "labels.npy", payload["labels"], allow_pickle=False)
    np.save(
        output_dir / "probabilities.npy",
        payload["probabilities"],
        allow_pickle=False,
    )
    summary = {
        "workflow": "infer",
        "run_dir": str(run_dir),
        "dataset_dir": str(dataset_dir),
        "split": split,
        "xfit_features": {
            "enabled": use_xfit_features,
            "feature_dir": (
                str(xfit_feature_dir)
                if xfit_feature_dir is not None
                else None
            ),
            "feature_names": list(dataset.xfit_feature_names),
            "bundle_identity": dataset.xfit_feature_bundle_identity,
            "join_diagnostics": dataset.xfit_join_diagnostics,
            "fit_coverage": (
                xfit_fit_coverage(dataset, split=split)
                if use_xfit_features
                else None
            ),
        },
        "saved": {
            "logits": str((output_dir / "logits.npy").relative_to(run_dir)),
            "labels": str((output_dir / "labels.npy").relative_to(run_dir)),
            "probabilities": str(
                (output_dir / "probabilities.npy").relative_to(run_dir)
            ),
        },
    }
    write_summary(output_dir, summary)
    return WorkflowResult(run_dir=output_dir, summary=summary)


def evaluate_workflow(
    *,
    run_dir: Path,
    dataset_dir: Path,
    split: str,
    batch_size: int = 32,
    xfit_feature_dir: Path | None = None,
    use_xfit_features: bool = False,
    allow_xfit_coverage_mismatch: bool = False,
) -> WorkflowResult:
    device = resolve_device("auto")
    run_dir = run_dir.expanduser().resolve()
    dataset_dir = dataset_dir.expanduser().resolve()
    if xfit_feature_dir is not None:
        xfit_feature_dir = xfit_feature_dir.expanduser().resolve()
    if not isinstance(allow_xfit_coverage_mismatch, bool):
        raise ValueError("allow_xfit_coverage_mismatch must be a boolean")
    if allow_xfit_coverage_mismatch and not use_xfit_features:
        raise ValueError(
            "--allow-xfit-coverage-mismatch requires --use-xfit-features"
        )
    model, checkpoint, performance = load_model_from_checkpoint(
        run_dir,
        device=device,
    )
    xfit_feature_matrix = _checkpoint_xfit_feature_matrix(
        checkpoint=checkpoint,
        dataset_dir=dataset_dir,
        xfit_feature_dir=xfit_feature_dir,
        use_xfit_features=use_xfit_features,
    )
    dataset = _checkpoint_dataset(
        checkpoint=checkpoint,
        dataset_dir=dataset_dir,
        split=split,
        xfit_feature_matrix=xfit_feature_matrix,
    )
    train_config = checkpoint.get("train_config")
    calibration_split = (
        train_config.get("val_split")
        if isinstance(train_config, dict)
        else None
    )
    if not isinstance(calibration_split, str) or not calibration_split:
        raise ValueError("checkpoint train_config.val_split is invalid")
    threshold_selection = None
    threshold_selection_undefined_reason = None
    val_dataset = dataset if split == calibration_split else None
    if split != calibration_split:
        val_dataset = _checkpoint_dataset(
            checkpoint=checkpoint,
            dataset_dir=dataset_dir,
            split=calibration_split,
            xfit_feature_matrix=xfit_feature_matrix,
        )
        if len(val_dataset) > 0:
            val_payload = predict_dataset(
                model=model,
                dataset=val_dataset,
                batch_size=batch_size,
                device=device,
                performance=performance,
            )
            val_labels = ensure_binary_labels(
                val_payload["labels"], scores=val_payload["probabilities"]
            )
            if np.unique(val_labels).size < 2:
                threshold_selection_undefined_reason = (
                    "single_class_validation_split"
                )
            else:
                threshold_selection = select_threshold_for_scores(
                    val_labels,
                    val_payload["probabilities"],
                    metric="accuracy",
                    source_split=calibration_split,
                )
        else:
            threshold_selection_undefined_reason = "empty_validation_split"
    fit_coverage = None
    if use_xfit_features:
        assert val_dataset is not None
        validation_coverage = xfit_fit_coverage(
            val_dataset,
            split=calibration_split,
        )
        evaluated_coverage = xfit_fit_coverage(dataset, split=split)
        absolute_difference = abs(
            validation_coverage["fit_coverage"]
            - evaluated_coverage["fit_coverage"]
        )
        missing_policy = (
            dataset.xfit_feature_schema["missing_policy"]
            if dataset.xfit_feature_schema is not None
            else None
        )
        materially_mismatched = bool(
            missing_policy == "indicator"
            and absolute_difference > XFIT_COVERAGE_MISMATCH_THRESHOLD
        )
        fit_coverage = {
            "validation": validation_coverage,
            "evaluated": evaluated_coverage,
            "absolute_difference": absolute_difference,
            "material_threshold": XFIT_COVERAGE_MISMATCH_THRESHOLD,
            "materially_mismatched": materially_mismatched,
            "mismatch_allowed": allow_xfit_coverage_mismatch,
            "missing_policy": missing_policy,
        }
        if (
            threshold_selection is not None
            and materially_mismatched
            and not allow_xfit_coverage_mismatch
        ):
            raise ValueError(
                "refusing to apply validation-calibrated threshold across "
                "materially mismatched xFit coverage: "
                f"validation={validation_coverage['fit_coverage']:.3f}, "
                f"evaluated={evaluated_coverage['fit_coverage']:.3f}, "
                "allowed absolute difference="
                f"{XFIT_COVERAGE_MISMATCH_THRESHOLD:.3f}; pass "
                "--allow-xfit-coverage-mismatch to opt in"
            )
    payload = predict_dataset(
        model=model,
        dataset=dataset,
        batch_size=batch_size,
        device=device,
        performance=performance,
    )
    output_dir = run_dir / "evaluation" / split
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_predictions(
        y_true=payload["labels"],
        logits=payload["logits"],
        metadata_rows=payload["metadata_rows"],
        threshold_selection=threshold_selection,
        threshold_selection_undefined_reason=(
            threshold_selection_undefined_reason
        ),
        output_dir=output_dir,
    )
    metrics.update(
        {
            "workflow": "evaluate",
            "run_dir": str(run_dir),
            "dataset_dir": str(dataset_dir),
            "split": split,
            "input_mode": checkpoint["model_config"]["input_mode"],
            "xfit_features": {
                "enabled": use_xfit_features,
                "feature_dir": (
                    str(xfit_feature_dir)
                    if xfit_feature_dir is not None
                    else None
                ),
                "feature_names": list(dataset.xfit_feature_names),
                "bundle_identity": dataset.xfit_feature_bundle_identity,
                "join_diagnostics": dataset.xfit_join_diagnostics,
                "fit_coverage": fit_coverage,
            },
        }
    )
    write_summary(output_dir, metrics)
    return WorkflowResult(run_dir=output_dir, summary=metrics)


def review_queue_workflow(
    *,
    run_dir: Path,
    dataset_dir: Path,
    split: str,
    output_dir: Path | None = None,
    compare_run_dirs: list[Path] | None = None,
    max_items: int = 200,
    strategy: str = "hybrid",
) -> WorkflowResult:
    result = build_review_queue(
        run_dir=run_dir,
        dataset_dir=dataset_dir,
        split=split,
        output_dir=output_dir,
        compare_run_dirs=compare_run_dirs,
        max_items=max_items,
        strategy=strategy,
    )
    return WorkflowResult(run_dir=result.review_dir, summary=result.summary)


def dataset_review_queue_workflow(
    *,
    dataset_dir: Path,
    split: str = "all",
    output_dir: Path | None = None,
    max_items: int = 200,
) -> WorkflowResult:
    result = build_dataset_review_queue(
        dataset_dir=dataset_dir,
        split=split,
        output_dir=output_dir,
        max_items=max_items,
    )
    return WorkflowResult(run_dir=result.review_dir, summary=result.summary)


def review_queue_splits_workflow(
    *,
    run_dir: Path,
    dataset_dir: Path,
    output_root: Path,
    splits: list[str],
    compare_run_dirs: list[Path] | None = None,
    max_items: int = 200,
    strategy: str = "hybrid",
) -> WorkflowResult:
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    split_results = {}
    for split in splits:
        clean_split = split.strip()
        if not clean_split:
            continue
        result = build_review_queue(
            run_dir=run_dir,
            dataset_dir=dataset_dir,
            split=clean_split,
            output_dir=output_root
            / f"{clean_split}-{strategy}-{int(max_items)}",
            compare_run_dirs=compare_run_dirs,
            max_items=max_items,
            strategy=strategy,
        )
        split_results[clean_split] = result.summary
    if not split_results:
        raise ValueError("at least one split must be requested")
    summary = {
        "workflow": "review-queue-splits",
        "run_dir": str(run_dir.expanduser().resolve()),
        "dataset_dir": str(dataset_dir.expanduser().resolve()),
        "output_root": str(output_root),
        "splits": list(split_results),
        "max_items": max_items,
        "strategy": strategy,
        "split_results": split_results,
    }
    return WorkflowResult(run_dir=output_root, summary=summary)


def review_bokeh_workflow(
    *,
    review_dir: Path,
    host: str = "localhost",
    port: int = 0,
    show_url_only: bool = False,
) -> WorkflowResult:
    summary = review_bokeh_server_summary(
        review_dir=review_dir,
        host=host,
        port=port,
        show_url_only=show_url_only,
    )
    return WorkflowResult(
        run_dir=review_dir.expanduser().resolve(),
        summary=summary,
    )


def review_status_workflow(
    *,
    review_dir: Path,
    min_reviewers: int = 2,
    min_actionable_reviewers: int = 2,
    consensus_rule: str = "unanimous",
    include_decisions: bool = False,
) -> WorkflowResult:
    result = summarize_review_status(
        review_dir=review_dir,
        min_reviewers=min_reviewers,
        min_actionable_reviewers=min_actionable_reviewers,
        consensus_rule=consensus_rule,
        include_decisions=include_decisions,
    )
    return WorkflowResult(run_dir=result.review_dir, summary=result.summary)


def review_contact_sheet_workflow(
    *,
    review_dir: Path,
    output_dir: Path,
    max_items: int = 64,
    items_per_page: int = 16,
    columns: int = 4,
    stamp_size: int = 96,
    overwrite: bool = False,
) -> WorkflowResult:
    result = export_review_contact_sheets(
        review_dir=review_dir,
        output_dir=output_dir,
        max_items=max_items,
        items_per_page=items_per_page,
        columns=columns,
        stamp_size=stamp_size,
        overwrite=overwrite,
    )
    return WorkflowResult(run_dir=result.output_dir, summary=result.summary)


def review_annotation_template_workflow(
    *,
    review_dir: Path,
    output_csv: Path,
    reviewer: str | None = None,
    overwrite: bool = False,
) -> WorkflowResult:
    result = export_review_annotation_template(
        review_dir=review_dir,
        output_csv=output_csv,
        reviewer=reviewer,
        overwrite=overwrite,
    )
    return WorkflowResult(
        run_dir=result.output_csv.parent, summary=result.summary
    )


def review_import_annotations_workflow(
    *,
    review_dir: Path,
    input_csv: Path,
    reviewer: str | None = None,
    dry_run: bool = False,
    require_all: bool = False,
) -> WorkflowResult:
    result = import_review_annotations_from_csv(
        review_dir=review_dir,
        input_csv=input_csv,
        reviewer=reviewer,
        dry_run=dry_run,
        require_all=require_all,
    )
    return WorkflowResult(run_dir=result.review_dir, summary=result.summary)


def review_aggregate_workflow(
    *,
    review_dir: Path,
    output_report: Path | None = None,
    min_reviewers: int = 2,
    min_actionable_reviewers: int = 2,
    consensus_rule: str = "unanimous",
) -> WorkflowResult:
    result = aggregate_review_annotations(
        review_dir=review_dir,
        output_report=output_report,
        min_reviewers=min_reviewers,
        min_actionable_reviewers=min_actionable_reviewers,
        consensus_rule=consensus_rule,
    )
    return WorkflowResult(run_dir=result.review_dir, summary=result.summary)


def review_apply_workflow(
    *,
    dataset_dir: Path,
    review_dir: Path,
    output_dir: Path,
    aggregation_report: Path | None = None,
) -> WorkflowResult:
    result = apply_review_annotations(
        dataset_dir=dataset_dir,
        review_dir=review_dir,
        output_dir=output_dir,
        aggregation_report=aggregation_report,
    )
    return WorkflowResult(run_dir=result.output_dir, summary=result.summary)


def entity_review_queue_workflow(
    *,
    source_review_dirs: list[Path],
    output_dir: Path,
) -> WorkflowResult:
    result = build_entity_review_queue(
        source_review_dirs=source_review_dirs,
        output_dir=output_dir,
    )
    return WorkflowResult(run_dir=result.review_dir, summary=result.summary)


def entity_review_bokeh_workflow(
    *,
    review_dir: Path,
    host: str = "localhost",
    port: int = 5007,
    show_url_only: bool = False,
) -> WorkflowResult:
    summary = entity_review_bokeh_server_summary(
        review_dir=review_dir,
        host=host,
        port=port,
        show_url_only=show_url_only,
    )
    return WorkflowResult(
        run_dir=review_dir.expanduser().resolve(),
        summary=summary,
    )


def entity_review_aggregate_workflow(
    *,
    review_dir: Path,
    output_report: Path | None = None,
    min_reviewers: int = 1,
    consensus_rule: str = "unanimous",
) -> WorkflowResult:
    result = aggregate_entity_review_annotations(
        review_dir=review_dir,
        output_report=output_report,
        min_reviewers=min_reviewers,
        consensus_rule=consensus_rule,
    )
    return WorkflowResult(run_dir=result.review_dir, summary=result.summary)


def compare_inputs_workflow(run_dirs: list[Path]) -> dict[str, Any]:
    rows = []
    for run_dir in run_dirs:
        resolved = run_dir.expanduser().resolve()
        summary_path = resolved / "evaluation" / "test" / "summary.json"
        if not summary_path.exists():
            summary_path = resolved / "evaluation" / "val" / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(
                f"Missing evaluation summary for {resolved}"
            )
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "run_dir": str(resolved),
                "input_mode": payload.get("input_mode"),
                "roc_auc": finite_numeric_value(payload["roc_auc"]),
                "pr_auc": finite_numeric_value(payload["pr_auc"]),
                "accuracy": finite_numeric_value(payload["accuracy"]),
                "sample_count": payload["sample_count"],
            }
        )
    rows.sort(
        key=lambda row: (
            not (row["roc_auc"] is not None and row["pr_auc"] is not None),
            _descending_key(row["roc_auc"]),
            _descending_key(row["pr_auc"]),
        )
    )
    best_row = (
        rows[0]
        if rows
        and rows[0]["roc_auc"] is not None
        and rows[0]["pr_auc"] is not None
        else None
    )
    return {
        "workflow": "compare_inputs",
        "runs": rows,
        "best_run_dir": best_row["run_dir"] if best_row else None,
        "leaderboard_markdown": build_compare_markdown(rows),
    }


def reproduce_hsc_comparison_workflow(
    *,
    manifest_path: Path,
    pair_config: Path,
    triplet_config: Path,
    seeds: list[int],
    pair_pretrain_checkpoint: Path | None = None,
    triplet_pretrain_checkpoint: Path | None = None,
    output_root: Path | None = None,
    run_name: str | None = None,
) -> WorkflowResult:
    manifest_path = manifest_path.expanduser().resolve()
    pair_config = pair_config.expanduser().resolve()
    triplet_config = triplet_config.expanduser().resolve()
    base_manifest = json.loads(
        json.dumps(
            yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        )
    )
    comparison_dir = resolve_run_dir(
        output_root,
        run_name,
        "reproduce-hsc-comparison",
    )
    manifests_dir = comparison_dir / "manifests"
    datasets_dir = comparison_dir / "datasets"
    models_dir = comparison_dir / "model-runs"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    variants = [
        (
            "pair",
            {
                **base_manifest,
                "include_difference": False,
                "difference_mode": "simple",
            },
            "pair",
            pair_config,
        ),
        (
            "triplet_simple",
            {
                **base_manifest,
                "include_difference": True,
                "difference_mode": "simple",
            },
            "triplet",
            triplet_config,
        ),
        (
            "triplet_xpois",
            {
                **base_manifest,
                "include_difference": True,
                "difference_mode": "xpois",
            },
            "triplet",
            triplet_config,
        ),
    ]

    aggregate: dict[str, Any] = {
        "workflow": "reproduce-hsc-comparison",
        "manifest_path": str(manifest_path),
        "pair_config": str(pair_config),
        "triplet_config": str(triplet_config),
        "seeds": seeds,
        "benchmark_regime_name": base_manifest.get("benchmark_regime_name"),
        "pretrain_checkpoints": {
            "pair": (
                str(pair_pretrain_checkpoint.expanduser().resolve())
                if pair_pretrain_checkpoint is not None
                else None
            ),
            "triplet": (
                str(triplet_pretrain_checkpoint.expanduser().resolve())
                if triplet_pretrain_checkpoint is not None
                else None
            ),
        },
        "datasets": {},
        "jobs": {},
    }

    dataset_refs: dict[str, Path] = {}
    for label, payload, _, _config_path in variants:
        generated_manifest = manifests_dir / f"{label}.yaml"
        dump_config(generated_manifest, payload)
        dataset_dir = datasets_dir / label
        result = build_hsc_dataset_from_manifest(
            manifest_path=generated_manifest,
            output_dir=dataset_dir,
        )
        dataset_refs[label] = result.output_dir
        aggregate["datasets"][label] = result.summary

    aggregate["alignment"] = compare_hsc_dataset_alignment(
        [
            dataset_refs["pair"],
            dataset_refs["triplet_simple"],
            dataset_refs["triplet_xpois"],
        ]
    )

    training_variants = [
        (
            "pair",
            "pair",
            "pair",
            "scratch",
            None,
            pair_config,
        ),
        (
            "triplet_simple",
            "triplet_simple",
            "triplet",
            "scratch",
            None,
            triplet_config,
        ),
        (
            "triplet_xpois",
            "triplet_xpois",
            "triplet",
            "scratch",
            None,
            triplet_config,
        ),
    ]
    if pair_pretrain_checkpoint is not None:
        training_variants.append(
            (
                "pair_finetune",
                "pair",
                "pair",
                "fine_tune",
                pair_pretrain_checkpoint.expanduser().resolve(),
                pair_config,
            )
        )
    if triplet_pretrain_checkpoint is not None:
        training_variants.extend(
            [
                (
                    "triplet_simple_finetune",
                    "triplet_simple",
                    "triplet",
                    "fine_tune",
                    triplet_pretrain_checkpoint.expanduser().resolve(),
                    triplet_config,
                ),
                (
                    "triplet_xpois_finetune",
                    "triplet_xpois",
                    "triplet",
                    "fine_tune",
                    triplet_pretrain_checkpoint.expanduser().resolve(),
                    triplet_config,
                ),
            ]
        )

    benchmark_regime_name = base_manifest.get("benchmark_regime_name")
    for (
        job_label,
        dataset_label,
        input_mode,
        training_mode,
        pretrain_checkpoint,
        config_path,
    ) in training_variants:
        base_config = load_training_config(config_path)
        job_runs = []
        for seed in seeds:
            seeded = replace(
                base_config,
                seed=seed,
                dataset_dir=str(dataset_refs[dataset_label]),
                output_root=str(models_dir),
                run_name=f"{job_label}-seed-{seed}",
                benchmark_regime_name=benchmark_regime_name,
                training_mode=training_mode,
                pretrain_checkpoint=(
                    str(pretrain_checkpoint)
                    if pretrain_checkpoint is not None
                    else None
                ),
            )
            seeded.model = replace(base_config.model, input_mode=input_mode)
            run_dir = resolve_run_dir(
                Path(seeded.output_root),
                seeded.run_name,
                job_label,
            )
            train_summary = train_classifier(seeded, run_dir=run_dir)
            write_summary(
                run_dir,
                {
                    **train_summary,
                    "workflow": "train",
                    "config_path": str(config_path),
                    "input_mode": seeded.model.input_mode,
                    "comparison_label": job_label,
                },
            )
            eval_split = resolve_available_split(
                dataset_dir=Path(seeded.dataset_dir),
                input_mode=seeded.model.input_mode,
                preferred=seeded.eval_split,
            )
            eval_result = evaluate_workflow(
                run_dir=run_dir,
                dataset_dir=Path(seeded.dataset_dir),
                split=eval_split,
                batch_size=seeded.batch_size,
                xfit_feature_dir=(
                    Path(seeded.xfit_feature_dir)
                    if seeded.xfit_feature_dir is not None
                    else None
                ),
                use_xfit_features=seeded.use_xfit_features,
            )
            job_runs.append(
                {
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "dataset_dir": str(dataset_refs[dataset_label]),
                    "eval_split": eval_split,
                    "input_mode": input_mode,
                    "training_mode": training_mode,
                    "benchmark_regime_name": benchmark_regime_name,
                    "pretrain_checkpoint": (
                        str(pretrain_checkpoint)
                        if pretrain_checkpoint is not None
                        else None
                    ),
                    "roc_auc": eval_result.summary["roc_auc"],
                    "pr_auc": eval_result.summary["pr_auc"],
                    "accuracy": eval_result.summary["accuracy"],
                    **extract_comparison_eval_metrics(eval_result.summary),
                    **{
                        key: eval_result.summary.get(key)
                        for key in STRATIFIED_BREAKDOWN_KEYS
                    },
                }
            )
        aggregate["jobs"][job_label] = summarize_hsc_job_runs(
            input_mode=input_mode,
            training_mode=training_mode,
            job_runs=job_runs,
        )

    aggregate["summary_markdown"] = build_hsc_comparison_markdown(aggregate)
    aggregate["saved"] = {
        "summary_json": "summary.json",
        "summary_markdown": "summary.md",
    }
    write_summary(comparison_dir, aggregate)
    (comparison_dir / "summary.md").write_text(
        aggregate["summary_markdown"],
        encoding="utf-8",
    )
    return WorkflowResult(run_dir=comparison_dir, summary=aggregate)


def reproduce_hsc_xpois_sweep_workflow(
    *,
    manifest_path: Path,
    pair_config: Path,
    triplet_config: Path,
    sweep_config: Path,
    seeds: list[int],
    output_root: Path | None = None,
    run_name: str | None = None,
) -> WorkflowResult:
    manifest_path = manifest_path.expanduser().resolve()
    pair_config = pair_config.expanduser().resolve()
    triplet_config = triplet_config.expanduser().resolve()
    sweep_config = sweep_config.expanduser().resolve()
    base_manifest = json.loads(
        json.dumps(
            yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        )
    )
    sweep_variants = load_hsc_xpois_sweep_variants(sweep_config)
    comparison_dir = resolve_run_dir(
        output_root,
        run_name,
        "reproduce-hsc-xpois-sweep",
    )
    manifests_dir = comparison_dir / "manifests"
    datasets_dir = comparison_dir / "datasets"
    models_dir = comparison_dir / "model-runs"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    base_variants = [
        (
            "pair",
            {
                **base_manifest,
                "include_difference": False,
                "difference_mode": "simple",
            },
            "pair",
            pair_config,
            "baseline",
        ),
        (
            "triplet_simple",
            {
                **base_manifest,
                "include_difference": True,
                "difference_mode": "simple",
            },
            "triplet",
            triplet_config,
            "baseline",
        ),
    ]
    xpois_variants = [
        (
            f"triplet_xpois_{variant['name']}",
            {
                **base_manifest,
                "include_difference": True,
                "difference_mode": "xpois",
                **{
                    key: value
                    for key, value in variant.items()
                    if key != "name"
                },
            },
            "triplet",
            triplet_config,
            "xpois",
            variant,
        )
        for variant in sweep_variants
    ]

    aggregate: dict[str, Any] = {
        "workflow": "reproduce-hsc-xpois-sweep",
        "manifest_path": str(manifest_path),
        "pair_config": str(pair_config),
        "triplet_config": str(triplet_config),
        "sweep_config": str(sweep_config),
        "seeds": seeds,
        "datasets": {},
        "jobs": {},
        "variant_builds": {},
    }

    dataset_refs: dict[str, Path] = {}
    for (
        label,
        payload,
        input_mode,
        _config_path,
        variant_type,
    ) in base_variants:
        generated_manifest = manifests_dir / f"{label}.yaml"
        dump_config(generated_manifest, payload)
        dataset_dir = datasets_dir / label
        build_start = time.perf_counter()
        result = build_hsc_dataset_from_manifest(
            manifest_path=generated_manifest,
            output_dir=dataset_dir,
        )
        build_time_seconds = float(time.perf_counter() - build_start)
        dataset_refs[label] = result.output_dir
        aggregate["datasets"][label] = result.summary
        aggregate["variant_builds"][label] = {
            "variant_name": label,
            "variant_type": variant_type,
            "input_mode": input_mode,
            "stability_status": "stable",
            "manifest_path": str(generated_manifest),
            "dataset_dir": str(result.output_dir),
            "build_time_seconds": build_time_seconds,
            "requested_overrides": {},
            "dataset_summary": result.summary,
            "difference_diagnostics": (
                {
                    "reference_variant": "triplet_simple",
                    "sample_count": int(result.summary["positive_count"])
                    + int(result.summary["negative_count"]),
                    "mean_abs_delta": 0.0,
                    "median_abs_delta": 0.0,
                    "p95_per_sample_mean_abs_delta": 0.0,
                    "allclose": True,
                }
                if label == "triplet_simple"
                else None
            ),
            "failure": None,
        }

    for (
        label,
        payload,
        input_mode,
        _config_path,
        variant_type,
        variant,
    ) in xpois_variants:
        generated_manifest = manifests_dir / f"{label}.yaml"
        dump_config(generated_manifest, payload)
        dataset_dir = datasets_dir / label
        build_start = time.perf_counter()
        try:
            result = build_hsc_dataset_from_manifest(
                manifest_path=generated_manifest,
                output_dir=dataset_dir,
            )
            diagnostics = compute_difference_diagnostics(
                reference_dataset_dir=dataset_refs["triplet_simple"],
                dataset_dir=result.output_dir,
            )
        except (RuntimeError, ValueError) as exc:
            build_time_seconds = float(time.perf_counter() - build_start)
            aggregate["variant_builds"][label] = {
                "variant_name": label,
                "variant_type": variant_type,
                "input_mode": input_mode,
                "stability_status": "unstable",
                "manifest_path": str(generated_manifest),
                "dataset_dir": str(dataset_dir.expanduser().resolve()),
                "build_time_seconds": build_time_seconds,
                "requested_overrides": {
                    key: value
                    for key, value in variant.items()
                    if key != "name"
                },
                "dataset_summary": None,
                "difference_diagnostics": None,
                "failure": {
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
            }
            continue
        build_time_seconds = float(time.perf_counter() - build_start)
        dataset_refs[label] = result.output_dir
        aggregate["datasets"][label] = result.summary
        aggregate["variant_builds"][label] = {
            "variant_name": label,
            "variant_type": variant_type,
            "input_mode": input_mode,
            "stability_status": "stable",
            "manifest_path": str(generated_manifest),
            "dataset_dir": str(result.output_dir),
            "build_time_seconds": build_time_seconds,
            "requested_overrides": {
                key: value for key, value in variant.items() if key != "name"
            },
            "dataset_summary": result.summary,
            "difference_diagnostics": diagnostics,
            "failure": None,
        }

    aggregate["stable_variants"] = [
        label
        for label, payload in aggregate["variant_builds"].items()
        if payload["stability_status"] == "stable"
    ]
    aggregate["unstable_variants"] = [
        label
        for label, payload in aggregate["variant_builds"].items()
        if payload["stability_status"] != "stable"
    ]
    aggregate["alignment"] = compare_hsc_dataset_alignment(
        [dataset_refs[label] for label in aggregate["stable_variants"]]
    )

    training_variants = [
        (
            "pair",
            "pair",
            pair_config,
        ),
        (
            "triplet_simple",
            "triplet",
            triplet_config,
        ),
    ] + [
        (
            label,
            "triplet",
            triplet_config,
        )
        for label in aggregate["stable_variants"]
        if label.startswith("triplet_xpois_")
    ]

    for label, input_mode, config_path in training_variants:
        base_config = load_training_config(config_path)
        job_runs = []
        training_mode = str(base_config.training_mode)
        for seed in seeds:
            seeded = replace(
                base_config,
                seed=seed,
                dataset_dir=str(dataset_refs[label]),
                output_root=str(models_dir),
                run_name=f"{label}-seed-{seed}",
            )
            seeded.model = replace(base_config.model, input_mode=input_mode)
            run_dir = resolve_run_dir(
                Path(seeded.output_root),
                seeded.run_name,
                label,
            )
            train_summary = train_classifier(seeded, run_dir=run_dir)
            write_summary(
                run_dir,
                {
                    **train_summary,
                    "workflow": "train",
                    "config_path": str(config_path),
                    "input_mode": seeded.model.input_mode,
                    "comparison_label": label,
                },
            )
            eval_split = resolve_available_split(
                dataset_dir=Path(seeded.dataset_dir),
                input_mode=seeded.model.input_mode,
                preferred=seeded.eval_split,
            )
            eval_result = evaluate_workflow(
                run_dir=run_dir,
                dataset_dir=Path(seeded.dataset_dir),
                split=eval_split,
                batch_size=seeded.batch_size,
                xfit_feature_dir=(
                    Path(seeded.xfit_feature_dir)
                    if seeded.xfit_feature_dir is not None
                    else None
                ),
                use_xfit_features=seeded.use_xfit_features,
            )
            job_runs.append(
                {
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "dataset_dir": str(dataset_refs[label]),
                    "eval_split": eval_split,
                    "input_mode": input_mode,
                    "training_mode": training_mode,
                    "roc_auc": eval_result.summary["roc_auc"],
                    "pr_auc": eval_result.summary["pr_auc"],
                    "accuracy": eval_result.summary["accuracy"],
                    **extract_comparison_eval_metrics(eval_result.summary),
                    **{
                        key: eval_result.summary.get(key)
                        for key in STRATIFIED_BREAKDOWN_KEYS
                    },
                }
            )
        aggregate["jobs"][label] = summarize_hsc_job_runs(
            input_mode=input_mode,
            training_mode=training_mode,
            job_runs=job_runs,
        )

    ranking, unranked = rank_hsc_jobs(aggregate["jobs"])
    aggregate["ranking"] = ranking
    aggregate["unranked"] = unranked
    _best = aggregate["jobs"][ranking[0]] if ranking else None
    aggregate["best_run_dir"] = (
        _best["best_run_dir"] if _best is not None else None
    )
    aggregate["summary_markdown"] = build_hsc_xpois_sweep_markdown(aggregate)
    aggregate["saved"] = {
        "summary_json": "summary.json",
        "summary_markdown": "summary.md",
    }
    write_summary(comparison_dir, aggregate)
    (comparison_dir / "summary.md").write_text(
        aggregate["summary_markdown"],
        encoding="utf-8",
    )
    return WorkflowResult(run_dir=comparison_dir, summary=aggregate)


def reproduce_inada_workflow(
    *,
    pair_config: Path | None,
    triplet_config: Path | None,
    nodiff_pair_config: Path | None,
    seeds: list[int],
) -> dict[str, Any]:
    jobs = []
    if pair_config is not None:
        jobs.append(("autoscan_pair", pair_config, "pair"))
    if triplet_config is not None:
        jobs.append(("autoscan_triplet", triplet_config, "triplet"))
    if nodiff_pair_config is not None:
        jobs.append(("nodiff_pair", nodiff_pair_config, "pair"))
    if not jobs:
        raise ValueError(
            "At least one of pair_config, triplet_config, or "
            "nodiff_pair_config must be provided"
        )

    aggregate: dict[str, Any] = {
        "workflow": "reproduce-inada",
        "jobs": {},
        "seeds": seeds,
    }
    for label, config_path, input_mode in jobs:
        config = load_training_config(config_path)
        job_runs = []
        for seed in seeds:
            seeded = replace(
                config, seed=seed, run_name=f"{label}-seed-{seed}"
            )
            seeded.model = replace(config.model, input_mode=input_mode)
            output_root = (
                Path(seeded.output_root).expanduser()
                if seeded.output_root is not None
                else None
            )
            run_dir = resolve_run_dir(output_root, seeded.run_name, label)
            train_summary = train_classifier(seeded, run_dir=run_dir)
            write_summary(
                run_dir,
                {
                    **train_summary,
                    "workflow": "train",
                    "config_path": str(config_path.expanduser().resolve()),
                    "input_mode": seeded.model.input_mode,
                },
            )
            eval_result = evaluate_workflow(
                run_dir=run_dir,
                dataset_dir=Path(seeded.dataset_dir),
                split=seeded.eval_split,
                batch_size=seeded.batch_size,
                xfit_feature_dir=(
                    Path(seeded.xfit_feature_dir)
                    if seeded.xfit_feature_dir is not None
                    else None
                ),
                use_xfit_features=seeded.use_xfit_features,
            )
            job_runs.append(
                {
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "roc_auc": eval_result.summary["roc_auc"],
                    "pr_auc": eval_result.summary["pr_auc"],
                    "accuracy": eval_result.summary["accuracy"],
                }
            )
        aggregate["jobs"][label] = summarize_reproduction_runs(job_runs)
    aggregate["summary_markdown"] = build_reproduction_markdown(aggregate)
    return aggregate


def validate_pair_triplet_comparison_controls(
    pair_config: Any,
    triplet_config: Any,
) -> dict[str, Any]:
    """Require pair/triplet comparisons to differ only by input mode."""

    control_values: dict[str, Any] = {}
    differences: dict[str, dict[str, Any]] = {}
    for field in PAIR_TRIPLET_CONTROL_FIELDS:
        pair_value = getattr(pair_config, field)
        triplet_value = getattr(triplet_config, field)
        if pair_value != triplet_value:
            differences[field] = {
                "pair": pair_value,
                "triplet": triplet_value,
            }
        control_values[field] = pair_value

    pair_model = asdict(pair_config.model)
    triplet_model = asdict(triplet_config.model)
    pair_input_mode = pair_model.pop("input_mode", None)
    triplet_input_mode = triplet_model.pop("input_mode", None)
    if pair_input_mode != "pair":
        differences["model.input_mode[pair_config]"] = {
            "expected": "pair",
            "actual": pair_input_mode,
        }
    if triplet_input_mode != "triplet":
        differences["model.input_mode[triplet_config]"] = {
            "expected": "triplet",
            "actual": triplet_input_mode,
        }
    if pair_model != triplet_model:
        differences["model"] = {
            "pair": pair_model,
            "triplet": triplet_model,
        }

    pair_performance = asdict(pair_config.performance)
    triplet_performance = asdict(triplet_config.performance)
    if pair_performance != triplet_performance:
        differences["performance"] = {
            "pair": pair_performance,
            "triplet": triplet_performance,
        }

    if differences:
        names = ", ".join(sorted(differences))
        raise ValueError(
            "pair/triplet comparison controls differ; use the same "
            "dataset, split policy, optimizer budget, model architecture, "
            "and runtime controls except input mode. Differing fields: "
            f"{names}"
        )

    return {
        "ok": True,
        "matched_fields": list(PAIR_TRIPLET_CONTROL_FIELDS),
        "control_values": control_values,
        "model_without_input_mode": pair_model,
        "performance": pair_performance,
        "pair_input_mode": pair_input_mode,
        "triplet_input_mode": triplet_input_mode,
    }


def reproduce_pair_triplet_workflow(
    *,
    dataset_dir: Path,
    pair_config: Path,
    triplet_config: Path,
    seeds: list[int],
    output_root: Path | None = None,
    run_name: str | None = None,
) -> WorkflowResult:
    dataset_dir = dataset_dir.expanduser().resolve()
    pair_config = pair_config.expanduser().resolve()
    triplet_config = triplet_config.expanduser().resolve()
    validation = validate_dataset_dir(dataset_dir)
    if not (dataset_dir / "difference.npy").exists():
        raise ValueError(
            "pair/triplet comparison requires difference.npy in the "
            f"reviewed dataset: {dataset_dir}"
        )
    label_provenance = check_training_label_provenance(dataset_dir)
    if not label_provenance["ok"]:
        raise ValueError(
            "training label provenance check failed: "
            + ", ".join(label_provenance["errors"])
        )
    pair_base_config = load_training_config(pair_config)
    triplet_base_config = load_training_config(triplet_config)
    comparison_controls = validate_pair_triplet_comparison_controls(
        pair_base_config,
        triplet_base_config,
    )

    comparison_dir = resolve_run_dir(
        output_root,
        run_name,
        "reproduce-pair-triplet",
    )
    models_dir = comparison_dir / "model-runs"
    models_dir.mkdir(parents=True, exist_ok=True)
    aggregate: dict[str, Any] = {
        "workflow": "reproduce-pair-triplet",
        "dataset_dir": str(dataset_dir),
        "dataset_validation": validation,
        "label_provenance": label_provenance,
        "pair_config": str(pair_config),
        "triplet_config": str(triplet_config),
        "seeds": seeds,
        "comparison_controls": comparison_controls,
        "jobs": {},
    }

    training_variants = [
        ("pair", "pair", pair_config, pair_base_config),
        ("triplet", "triplet", triplet_config, triplet_base_config),
    ]
    for label, input_mode, config_path, base_config in training_variants:
        job_runs = []
        training_mode = str(base_config.training_mode)
        for seed in seeds:
            seeded = replace(
                base_config,
                seed=seed,
                dataset_dir=str(dataset_dir),
                output_root=str(models_dir),
                run_name=f"{label}-seed-{seed}",
            )
            seeded.model = replace(base_config.model, input_mode=input_mode)
            run_dir = resolve_run_dir(
                Path(seeded.output_root),
                seeded.run_name,
                label,
            )
            train_summary = train_classifier(seeded, run_dir=run_dir)
            write_summary(
                run_dir,
                {
                    **train_summary,
                    "workflow": "train",
                    "config_path": str(config_path),
                    "input_mode": seeded.model.input_mode,
                    "comparison_label": label,
                },
            )
            eval_split = resolve_available_split(
                dataset_dir=dataset_dir,
                input_mode=seeded.model.input_mode,
                preferred=seeded.eval_split,
            )
            eval_result = evaluate_workflow(
                run_dir=run_dir,
                dataset_dir=dataset_dir,
                split=eval_split,
                batch_size=seeded.batch_size,
                xfit_feature_dir=(
                    Path(seeded.xfit_feature_dir)
                    if seeded.xfit_feature_dir is not None
                    else None
                ),
                use_xfit_features=seeded.use_xfit_features,
            )
            job_runs.append(
                {
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "dataset_dir": str(dataset_dir),
                    "eval_split": eval_split,
                    "input_mode": input_mode,
                    "training_mode": training_mode,
                    "roc_auc": eval_result.summary["roc_auc"],
                    "pr_auc": eval_result.summary["pr_auc"],
                    "accuracy": eval_result.summary["accuracy"],
                    **extract_comparison_eval_metrics(eval_result.summary),
                    **{
                        key: eval_result.summary.get(key)
                        for key in STRATIFIED_BREAKDOWN_KEYS
                    },
                }
            )
        aggregate["jobs"][label] = summarize_hsc_job_runs(
            input_mode=input_mode,
            training_mode=training_mode,
            job_runs=job_runs,
        )

    ranking, unranked = rank_hsc_jobs(aggregate["jobs"])
    aggregate["ranking"] = ranking
    aggregate["unranked"] = unranked
    _best = aggregate["jobs"][ranking[0]] if ranking else None
    aggregate["best_run_dir"] = (
        _best["best_run_dir"] if _best is not None else None
    )
    aggregate["summary_markdown"] = build_pair_triplet_markdown(aggregate)
    aggregate["saved"] = {
        "summary_json": "summary.json",
        "summary_markdown": "summary.md",
    }
    write_summary(comparison_dir, aggregate)
    (comparison_dir / "summary.md").write_text(
        aggregate["summary_markdown"],
        encoding="utf-8",
    )
    return WorkflowResult(run_dir=comparison_dir, summary=aggregate)


def build_compare_markdown(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "# XScan Compare Inputs\n\nNo runs provided.\n"
    lines = [
        "# XScan Compare Inputs",
        "",
        "| Run Dir | Input Mode | ROC AUC | PR AUC | Accuracy | Samples |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            (
                "| {run_dir} | {input_mode} | {roc_auc} | "
                "{pr_auc} | {accuracy} | {sample_count} |"
            ).format(
                run_dir=row["run_dir"],
                input_mode=row.get("input_mode") or "unknown",
                roc_auc=format_optional_float(row["roc_auc"]),
                pr_auc=format_optional_float(row["pr_auc"]),
                accuracy=format_optional_float(row["accuracy"]),
                sample_count=int(row["sample_count"]),
            )
        )
    return "\n".join(lines) + "\n"


def build_pair_triplet_markdown(aggregate: dict[str, Any]) -> str:
    lines = [
        "# XScan Pair/Triplet Comparison Summary",
        "",
        f"- Dataset: `{aggregate['dataset_dir']}`",
        f"- Pair config: `{aggregate['pair_config']}`",
        f"- Triplet config: `{aggregate['triplet_config']}`",
        f"- Seeds: `{','.join(str(seed) for seed in aggregate['seeds'])}`",
        "- Pair/triplet controls matched: "
        f"`{aggregate['comparison_controls']['ok']}`",
        "- Placeholder labels: "
        f"{aggregate['label_provenance']['placeholder_label_count']}",
    ]
    unranked = aggregate.get("unranked", [])
    if unranked:
        lines.append(
            "- Unranked (no run with defined ROC and PR AUC): "
            + ", ".join(unranked)
        )
    lines.extend(
        [
            "",
            "| Variant | Input Mode | Training Mode | Mean ROC AUC | "
            "Std ROC AUC | Mean PR AUC | Fixed Accuracy | TPR @ 1% FPR | "
            "TPR @ 5% FPR | Mean Brier | Confusion TP/TN/FP/FN | "
            "AUC Runs / Runs | Best Run |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|",
        ]
    )
    for label in [*aggregate["ranking"], *unranked]:
        payload = aggregate["jobs"][label]
        lines.append(
            (
                "| {label} | {input_mode} | {training_mode} | "
                "{mean_roc_auc} | {std_roc_auc} | "
                "{mean_pr_auc} | {mean_accuracy} | "
                "{tpr_at_fpr_1pct} | {tpr_at_fpr_5pct} | {brier_score} | "
                "{confusion} | {runs} | {best_run_dir} |"
            ).format(
                label=label,
                input_mode=payload["input_mode"],
                training_mode=payload["training_mode"],
                mean_roc_auc=format_optional_float(payload["mean_roc_auc"]),
                std_roc_auc=format_optional_float(payload["std_roc_auc"]),
                mean_pr_auc=format_optional_float(payload["mean_pr_auc"]),
                mean_accuracy=format_optional_float(payload["mean_accuracy"]),
                tpr_at_fpr_1pct=format_optional_float(
                    payload.get("mean_tpr_at_fpr_1pct")
                ),
                tpr_at_fpr_5pct=format_optional_float(
                    payload.get("mean_tpr_at_fpr_5pct")
                ),
                brier_score=format_optional_float(
                    payload.get("mean_brier_score")
                ),
                confusion=format_confusion_totals(
                    payload.get("confusion_totals")
                ),
                runs=(
                    f"{payload['defined_run_count']}/{len(payload['runs'])}"
                ),
                best_run_dir=format_optional_text(payload["best_run_dir"]),
            )
        )
    append_stratified_breakdown_markdown(lines, aggregate)
    return "\n".join(lines) + "\n"


def build_reproduction_markdown(aggregate: dict[str, Any]) -> str:
    lines = [
        "# XScan Reproduction Summary",
        "",
        "| Job | Mean ROC AUC | Std ROC AUC | Mean Accuracy | "
        "Std Accuracy | Runs |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, payload in aggregate["jobs"].items():
        lines.append(
            (
                "| {label} | {mean_roc_auc} | "
                "{std_roc_auc} | {mean_accuracy} | "
                "{std_accuracy} | {runs} |"
            ).format(
                label=label,
                mean_roc_auc=format_optional_float(payload["mean_roc_auc"]),
                std_roc_auc=format_optional_float(payload["std_roc_auc"]),
                mean_accuracy=format_optional_float(payload["mean_accuracy"]),
                std_accuracy=format_optional_float(payload["std_accuracy"]),
                runs=len(payload["runs"]),
            )
        )
    return "\n".join(lines) + "\n"


def resolve_available_split(
    *,
    dataset_dir: Path,
    input_mode: InputMode,
    preferred: str,
) -> str:
    for split in [preferred, "test", "val", "train"]:
        dataset = StampDataset(
            dataset_dir,
            input_mode=input_mode,
            split=split,
        )
        if len(dataset) > 0:
            return split
    raise ValueError(
        f"dataset {dataset_dir} has no non-empty splits for "
        f"input_mode={input_mode}"
    )


def compare_hsc_dataset_alignment(dataset_dirs: list[Path]) -> dict[str, Any]:
    payloads = []
    for dataset_dir in dataset_dirs:
        rows = [
            {
                key: row.get(key)
                for key in (
                    "candidate_id",
                    "label",
                    "split",
                    "x",
                    "y",
                    "split_group",
                    "center_source",
                    "catalog_object_id",
                    "center_offset_radius",
                )
            }
            for row in load_metadata_rows(dataset_dir)
        ]
        payloads.append((str(dataset_dir), rows))

    reference_dir, reference_rows = payloads[0]
    mismatches = []
    for other_dir, other_rows in payloads[1:]:
        aligned = reference_rows == other_rows
        mismatches.append(
            {
                "reference_dataset_dir": reference_dir,
                "dataset_dir": other_dir,
                "aligned": bool(aligned),
            }
        )
    return {
        "dataset_count": len(payloads),
        "all_aligned": (
            all(item["aligned"] for item in mismatches)
            if mismatches
            else True
        ),
        "checks": mismatches,
    }


def build_hsc_comparison_markdown(aggregate: dict[str, Any]) -> str:
    lines = [
        "# XScan HSC Comparison Summary",
        "",
        f"- Manifest: `{aggregate['manifest_path']}`",
        f"- Pair config: `{aggregate['pair_config']}`",
        f"- Triplet config: `{aggregate['triplet_config']}`",
        f"- Seeds: `{','.join(str(seed) for seed in aggregate['seeds'])}`",
        f"- Dataset alignment: {aggregate['alignment']['all_aligned']}",
    ]
    benchmark_regime_name = aggregate.get("benchmark_regime_name")
    if benchmark_regime_name:
        lines.append(f"- Benchmark regime: `{benchmark_regime_name}`")
    if aggregate.get("pretrain_checkpoints", {}).get("pair") is not None:
        lines.append(
            "- Pair pretrain checkpoint: "
            f"`{aggregate['pretrain_checkpoints']['pair']}`"
        )
    if aggregate.get("pretrain_checkpoints", {}).get("triplet") is not None:
        lines.append(
            "- Triplet pretrain checkpoint: "
            f"`{aggregate['pretrain_checkpoints']['triplet']}`"
        )
    lines.extend(
        [
            "",
            (
                "| Variant | Input Mode | Training Mode | Mean ROC AUC | "
                "Std ROC AUC | Mean PR AUC | Mean Accuracy | "
                "AUC Runs / Runs | Best Run |"
            ),
            "|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for label, payload in aggregate["jobs"].items():
        lines.append(
            (
                "| {label} | {input_mode} | {training_mode} | "
                "{mean_roc_auc} | {std_roc_auc} | "
                "{mean_pr_auc} | {mean_accuracy} | {runs} | "
                "{best_run_dir} |"
            ).format(
                label=label,
                input_mode=payload["input_mode"],
                training_mode=payload["training_mode"],
                mean_roc_auc=format_optional_float(payload["mean_roc_auc"]),
                std_roc_auc=format_optional_float(payload["std_roc_auc"]),
                mean_pr_auc=format_optional_float(payload["mean_pr_auc"]),
                mean_accuracy=format_optional_float(payload["mean_accuracy"]),
                runs=(
                    f"{payload['defined_run_count']}/{len(payload['runs'])}"
                ),
                best_run_dir=format_optional_text(payload["best_run_dir"]),
            )
        )
    append_stratified_breakdown_markdown(lines, aggregate)
    return "\n".join(lines) + "\n"


def append_stratified_breakdown_markdown(
    lines: list[str],
    aggregate: dict[str, Any],
) -> None:
    jobs = aggregate.get("jobs") or {}
    for key in STRATIFIED_BREAKDOWN_KEYS:
        rows = []
        for label, payload in jobs.items():
            breakdowns = payload.get("stratified_breakdowns") or {}
            breakdown = breakdowns.get(key)
            if not isinstance(breakdown, dict):
                continue
            for group_name, group_payload in (
                breakdown.get("groups") or {}
            ).items():
                rows.append((label, group_name, group_payload))
        if not rows:
            continue
        lines.extend(
            [
                "",
                f"## {STRATIFIED_BREAKDOWN_TITLES[key]} Breakdown",
                "",
                "| Variant | Group | Runs | Mean Count | Mean Accuracy | "
                "Mean Recovery | Mean FPR | Mean Probability |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for label, group_name, group_payload in rows:
            lines.append(
                (
                    "| {label} | {group_name} | {runs} | {mean_count} | "
                    "{mean_accuracy} | {mean_recovery} | {mean_fpr} | "
                    "{mean_probability} |"
                ).format(
                    label=label,
                    group_name=group_name,
                    runs=group_payload["runs"],
                    mean_count=format_optional_float(
                        group_payload.get("mean_count")
                    ),
                    mean_accuracy=format_optional_float(
                        group_payload.get("mean_accuracy")
                    ),
                    mean_recovery=format_optional_float(
                        group_payload.get("mean_positive_recovery_rate")
                    ),
                    mean_fpr=format_optional_float(
                        group_payload.get("mean_negative_false_positive_rate")
                    ),
                    mean_probability=format_optional_float(
                        group_payload.get("mean_probability")
                    ),
                )
            )


def format_optional_float(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.6f}"


def format_optional_text(value: Any) -> str:
    return "-" if value is None else str(value)


def format_confusion_totals(value: Any) -> str:
    if not isinstance(value, dict):
        return "-"
    keys = ("tp", "tn", "fp", "fn")
    if not all(key in value for key in keys):
        return "-"
    return "/".join(str(int(value[key])) for key in keys)


def build_hsc_xpois_sweep_markdown(aggregate: dict[str, Any]) -> str:
    def format_stat_mean(stat: dict[str, Any] | None) -> str:
        if not stat or int(stat.get("count", 0)) == 0:
            return "-"
        return f"{float(stat['mean']):.6f}"

    def top_rejected_plane(
        mask_diagnostics: dict[str, Any],
        *,
        region: str,
    ) -> str:
        overlaps = mask_diagnostics.get("rejected_plane_overlaps", {}).get(
            region, {}
        )
        if not overlaps:
            return "-"
        plane, payload = max(
            overlaps.items(),
            key=lambda item: int(
                item[1].get("rejected_stamps_with_plane", 0)
            ),
        )
        count = int(payload.get("rejected_stamps_with_plane", 0))
        if count == 0:
            return "-"
        mean_fraction = float(payload.get("mean_fraction_when_present", 0.0))
        return f"{plane} ({count}, mean={mean_fraction:.3f})"

    lines = [
        "# XScan HSC XPOIS Sweep Summary",
        "",
        f"- Manifest: `{aggregate['manifest_path']}`",
        f"- Pair config: `{aggregate['pair_config']}`",
        f"- Triplet config: `{aggregate['triplet_config']}`",
        f"- Sweep config: `{aggregate['sweep_config']}`",
        f"- Seeds: `{','.join(str(seed) for seed in aggregate['seeds'])}`",
        f"- Dataset alignment: {aggregate['alignment']['all_aligned']}",
        f"- Stable variants: `{len(aggregate['stable_variants'])}`",
        f"- Unstable variants: `{len(aggregate['unstable_variants'])}`",
        "",
        "## Results",
        "",
    ]
    unranked = aggregate.get("unranked", [])
    if unranked:
        lines.extend(
            [
                "Unranked because no run has defined ROC and PR AUC: "
                + ", ".join(unranked),
                "",
            ]
        )
    lines.extend(
        [
            "| Variant | Input Mode | Mean ROC AUC | Std ROC AUC | "
            "Mean PR AUC | "
            "Std PR AUC | Mean Accuracy | AUC Runs / Runs | Best Run |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for label in [*aggregate["ranking"], *unranked]:
        payload = aggregate["jobs"][label]
        lines.append(
            (
                "| {label} | {input_mode} | {mean_roc_auc} | "
                "{std_roc_auc} | {mean_pr_auc} | "
                "{std_pr_auc} | {mean_accuracy} | {runs} | "
                "{best_run_dir} |"
            ).format(
                label=label,
                input_mode=payload["input_mode"],
                mean_roc_auc=format_optional_float(payload["mean_roc_auc"]),
                std_roc_auc=format_optional_float(payload["std_roc_auc"]),
                mean_pr_auc=format_optional_float(payload["mean_pr_auc"]),
                std_pr_auc=format_optional_float(payload["std_pr_auc"]),
                mean_accuracy=format_optional_float(payload["mean_accuracy"]),
                runs=(
                    f"{payload['defined_run_count']}/{len(payload['runs'])}"
                ),
                best_run_dir=format_optional_text(payload["best_run_dir"]),
            )
        )
    append_stratified_breakdown_markdown(lines, aggregate)
    lines.extend(
        [
            "",
            "## Variant Builds",
            "",
            "| Variant | Status | Type | Build Time (s) | Mean Abs Delta | "
            "P95 Sample Mean Abs Delta | Failure |",
            "|---|---|---|---:|---:|---:|---|",
        ]
    )
    for label, payload in aggregate["variant_builds"].items():
        diagnostics = payload.get("difference_diagnostics") or {}
        failure = payload.get("failure") or {}
        lines.append(
            (
                "| {label} | {status} | {variant_type} | "
                "{build_time_seconds:.3f} | {mean_abs_delta} | {p95} | "
                "{failure_message} |"
            ).format(
                label=label,
                status=payload["stability_status"],
                variant_type=payload["variant_type"],
                build_time_seconds=float(payload["build_time_seconds"]),
                mean_abs_delta=(
                    f"{float(diagnostics['mean_abs_delta']):.6f}"
                    if "mean_abs_delta" in diagnostics
                    else "-"
                ),
                p95=(
                    f"{float(diagnostics['p95_per_sample_mean_abs_delta']):.6f}"
                    if "p95_per_sample_mean_abs_delta" in diagnostics
                    else "-"
                ),
                failure_message=failure.get("message", "-"),
            )
        )
    if any(
        (payload.get("dataset_summary") or {}).get("mask_diagnostics")
        for payload in aggregate["variant_builds"].values()
    ):
        lines.extend(
            [
                "",
                "## Mask Diagnostics",
                "",
                "| Variant | Enabled | Global Valid Fraction | Exposure "
                "Attempts | Stamp Rejections | Context Rejections | "
                "Centers Without "
                "Valid Exposure | Accepted Search Mean | Accepted Context "
                "Mean | Top Rejected Stamp Plane | Top Rejected Context "
                "Plane |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
            ]
        )
        for label, payload in aggregate["variant_builds"].items():
            dataset_summary = payload.get("dataset_summary") or {}
            mask_diagnostics = dataset_summary.get("mask_diagnostics") or {}
            rejections = mask_diagnostics.get("exposure_rejections") or {}
            global_fraction = mask_diagnostics.get(
                "valid_mask_global_fraction"
            )
            lines.append(
                (
                    "| {label} | {enabled} | {global_fraction} | "
                    "{attempts} | {stamp_rejections} | "
                    "{context_rejections} | {center_rejections} | "
                    "{search_mean} | {context_mean} | {top_stamp} | "
                    "{top_context} |"
                ).format(
                    label=label,
                    enabled=mask_diagnostics.get("enabled", "-"),
                    global_fraction=(
                        f"{float(global_fraction):.6f}"
                        if global_fraction is not None
                        else "-"
                    ),
                    attempts=mask_diagnostics.get("exposure_attempts", "-"),
                    stamp_rejections=rejections.get(
                        "search_stamp_low_valid_fraction", "-"
                    ),
                    context_rejections=rejections.get(
                        "search_context_low_valid_fraction", "-"
                    ),
                    center_rejections=rejections.get(
                        "centers_without_valid_exposure", "-"
                    ),
                    search_mean=format_stat_mean(
                        mask_diagnostics.get("accepted_search_valid_fraction")
                    ),
                    context_mean=format_stat_mean(
                        mask_diagnostics.get(
                            "accepted_difference_context_valid_fraction"
                        )
                    ),
                    top_stamp=top_rejected_plane(
                        mask_diagnostics,
                        region="search_stamp",
                    ),
                    top_context=top_rejected_plane(
                        mask_diagnostics,
                        region="search_context",
                    ),
                )
            )
    return "\n".join(lines) + "\n"
