# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Class-based CLI commands for XScan."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, TypeVar

from cuphoton.core.cli import (
    BoolInvariant,
    CommandError,
    InvariantAwareCommand,
    NonNegativeIntegerInvariant,
    PositiveIntegerInvariant,
    SetInvariant,
    StringInvariant,
)

T = TypeVar("T")
_DEFAULT_REVIEW_DIR = os.environ.get("CUPHOTON_XSCAN_REVIEW_DIR")
_DEFAULT_RAW_COMPARE_HOST = os.environ.get(
    "CUPHOTON_XSCAN_COMPARE_HOST", "localhost"
)
_DEFAULT_RAW_COMPARE_PORT = int(
    os.environ.get("CUPHOTON_XSCAN_COMPARE_PORT", "5011")
)
_DEFAULT_ALARD_LUPTON_HOST = os.environ.get(
    "CUPHOTON_XSCAN_ALARD_LUPTON_HOST", "localhost"
)
_DEFAULT_ALARD_LUPTON_PORT = int(
    os.environ.get("CUPHOTON_XSCAN_ALARD_LUPTON_PORT", "5012")
)


def _load_workflow(name: str) -> Callable[..., Any]:
    """Defer Torch-backed imports until a workflow is actually invoked."""

    def invoke(*args: Any, **kwargs: Any) -> Any:
        try:
            from . import workflows
        except ModuleNotFoundError as exc:
            if exc.name != "torch":
                raise
            raise ModuleNotFoundError(
                "XScan workflows require PyTorch; run "
                "'uv sync --extra torch' for development or install "
                "'cuphoton[torch]'"
            ) from exc
        return getattr(workflows, name)(*args, **kwargs)

    return invoke


def run_raw_compare_review_server(
    *,
    review_dir: Path,
    host: str,
    port: int,
) -> None:
    """Lazily start the raw-comparison review server."""

    from .raw_compare_review import run_server

    run_server(review_dir=review_dir, host=host, port=port)


def run_alard_lupton_review_server(
    *,
    review_dir: Path,
    host: str,
    port: int,
) -> None:
    """Lazily start the Alard-Lupton display-lab server."""

    from .alard_lupton_experiment_review import run_server

    run_server(review_dir=review_dir, host=host, port=port)


build_experimental_hsc_workflow = _load_workflow(
    "build_experimental_hsc_workflow"
)
build_hsc_registry_workflow = _load_workflow("build_hsc_registry_workflow")
build_hsc_workflow = _load_workflow("build_hsc_workflow")
build_lsstcomcam_smoke_workflow = _load_workflow(
    "build_lsstcomcam_smoke_workflow"
)
build_nodiff_release_workflow = _load_workflow(
    "build_nodiff_release_workflow"
)
build_prepared_dataset_workflow = _load_workflow(
    "build_prepared_dataset_workflow"
)
build_raw_autoscan_workflow = _load_workflow("build_raw_autoscan_workflow")
build_raw_nodiff_workflow = _load_workflow("build_raw_nodiff_workflow")
build_xfit_feature_bundle_workflow = _load_workflow(
    "build_xfit_feature_bundle_workflow"
)
export_xfit_input_workflow = _load_workflow("export_xfit_input_workflow")
check_lsstcomcam_candidates_workflow = _load_workflow(
    "check_lsstcomcam_candidates_workflow"
)
check_training_labels_workflow = _load_workflow(
    "check_training_labels_workflow"
)
compare_inputs_workflow = _load_workflow("compare_inputs_workflow")
dataset_review_queue_workflow = _load_workflow(
    "dataset_review_queue_workflow"
)
entity_review_aggregate_workflow = _load_workflow(
    "entity_review_aggregate_workflow"
)
entity_review_bokeh_workflow = _load_workflow("entity_review_bokeh_workflow")
entity_review_queue_workflow = _load_workflow("entity_review_queue_workflow")
evaluate_workflow = _load_workflow("evaluate_workflow")
infer_workflow = _load_workflow("infer_workflow")
inspect_dataset_workflow = _load_workflow("inspect_dataset_workflow")
merge_dataset_workflow = _load_workflow("merge_dataset_workflow")
plan_lsstcomcam_staging_workflow = _load_workflow(
    "plan_lsstcomcam_staging_workflow"
)
reproduce_hsc_xpois_sweep_workflow = _load_workflow(
    "reproduce_hsc_xpois_sweep_workflow"
)
reproduce_hsc_comparison_workflow = _load_workflow(
    "reproduce_hsc_comparison_workflow"
)
reproduce_inada_workflow = _load_workflow("reproduce_inada_workflow")
reproduce_pair_triplet_workflow = _load_workflow(
    "reproduce_pair_triplet_workflow"
)
review_aggregate_workflow = _load_workflow("review_aggregate_workflow")
review_annotation_template_workflow = _load_workflow(
    "review_annotation_template_workflow"
)
review_apply_workflow = _load_workflow("review_apply_workflow")
review_bokeh_workflow = _load_workflow("review_bokeh_workflow")
review_contact_sheet_workflow = _load_workflow(
    "review_contact_sheet_workflow"
)
review_import_annotations_workflow = _load_workflow(
    "review_import_annotations_workflow"
)
review_queue_splits_workflow = _load_workflow("review_queue_splits_workflow")
review_queue_workflow = _load_workflow("review_queue_workflow")
review_status_workflow = _load_workflow("review_status_workflow")
stage_lsstcomcam_fits_workflow = _load_workflow(
    "stage_lsstcomcam_fits_workflow"
)
train_workflow = _load_workflow("train_workflow")
validate_dataset_workflow = _load_workflow("validate_dataset_workflow")

_KNOWN_COMMAND_EXCEPTIONS = (
    FileExistsError,
    FileNotFoundError,
    ImportError,
    NotImplementedError,
    OSError,
    RuntimeError,
    ValueError,
)


def _emit_json_payload(payload: Any, *, out: Callable[[str], None]) -> None:
    out(json.dumps(payload, indent=2, default=str))


class PathSpecInvariant(StringInvariant):
    _type_desc = "path"
    _minlen = 1
    _maxlen = 4096


class CsvIntInvariant(StringInvariant):
    expected = "a comma-separated list of integer values"


class DatasetKindInvariant(SetInvariant):
    _set = {
        "autoscan",
        "nodiff",
        "experimental-hsc-synthetic",
        "hsc-synthetic",
        "lsstcomcam-smoke",
    }


class ReviewStrategyInvariant(SetInvariant):
    _set = {"hybrid", "uncertainty", "known-errors"}


class ReviewConsensusRuleInvariant(SetInvariant):
    _set = {"unanimous", "majority"}


class StageLinkModeInvariant(SetInvariant):
    _set = {"symlink", "copy", "hardlink"}


class StageDuplicatePolicyInvariant(SetInvariant):
    _set = {"fail", "first", "same-size"}


class XFitMissingPolicyInvariant(SetInvariant):
    _set = {"error", "indicator"}


class XScanCommand(InvariantAwareCommand):
    def _call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        try:
            return func(*args, **kwargs)
        except _KNOWN_COMMAND_EXCEPTIONS as exc:
            raise CommandError(str(exc)) from exc

    def _path(self, value: str | None) -> Path | None:
        if value is None:
            return None
        return Path(value).expanduser()

    def _emit_json(self, payload: Any) -> None:
        _emit_json_payload(payload, out=self._out)

    def _csv_ints(self, value: str) -> list[int]:
        try:
            items = [int(item) for item in value.split(",") if item.strip()]
        except ValueError as exc:
            raise CommandError(
                "expected a comma-separated list of integers"
            ) from exc
        if not items:
            raise CommandError("expected at least one integer value")
        return items

    def _csv_paths(self, value: str) -> list[Path]:
        parts = [item.strip() for item in value.split(",") if item.strip()]
        if not parts:
            raise CommandError("expected at least one path")
        return [Path(item).expanduser() for item in parts]


class DataInspectCommand(XScanCommand):
    """Inspect a packaged XScan dataset directory."""

    dataset_dir = None
    dataset_kind = None

    class DatasetDirArg(PathSpecInvariant):
        _arg = "--dataset-dir"
        _help = "Packaged dataset directory to inspect."
        _mandatory = True

    class DatasetKindArg(DatasetKindInvariant):
        _arg = "--dataset-kind"
        _help = "Optional expected dataset kind."
        _mandatory = False
        _default = None

    def run(self) -> None:
        payload = self._call(
            inspect_dataset_workflow,
            dataset_dir=Path(self.dataset_dir).expanduser(),
            dataset_kind=self.dataset_kind,
        )
        self._emit_json(payload)


class DataValidateCommand(XScanCommand):
    """Validate a packaged XScan dataset directory."""

    dataset_dir = None
    dataset_kind = None

    class DatasetDirArg(PathSpecInvariant):
        _arg = "--dataset-dir"
        _help = "Packaged dataset directory to validate."
        _mandatory = True

    class DatasetKindArg(DatasetKindInvariant):
        _arg = "--dataset-kind"
        _help = "Optional expected dataset kind."
        _mandatory = False
        _default = None

    def run(self) -> None:
        payload = self._call(
            validate_dataset_workflow,
            dataset_dir=Path(self.dataset_dir).expanduser(),
            dataset_kind=self.dataset_kind,
        )
        self._emit_json(payload)


class DataBuildXFitFeaturesCommand(XScanCommand):
    """Build a candidate-keyed xFit feature bundle for XScan fusion."""

    _name_ = "data-build-xfit-features"

    dataset_dir = None
    x_fit_run_dir = None
    output_dir = None
    missing_policy = None

    class DatasetDirArg(PathSpecInvariant):
        _arg = "--dataset-dir"
        _help = (
            "Packaged XScan dataset whose candidate order is authoritative."
        )
        _mandatory = True

    class XFitRunDirArg(PathSpecInvariant):
        _arg = "--xfit-run-dir"
        _help = "Completed xFit run containing portable fit artifacts."
        _mandatory = True

    class OutputDirArg(PathSpecInvariant):
        _arg = "--output-dir"
        _help = "New directory for the XScan xFit feature bundle."
        _mandatory = True

    class MissingPolicyArg(XFitMissingPolicyInvariant):
        _arg = "--missing-policy"
        _help = (
            "Missing candidate policy: error or indicator. "
            "[default: %default]"
        )
        _mandatory = False
        _default = "error"

    def run(self) -> None:
        result = self._call(
            build_xfit_feature_bundle_workflow,
            dataset_dir=Path(self.dataset_dir).expanduser(),
            xfit_run_dir=Path(self.x_fit_run_dir).expanduser(),
            output_dir=Path(self.output_dir).expanduser(),
            missing_policy=self.missing_policy,
        )
        self._emit_json(result.summary)


class DataExportXFitInputCommand(XScanCommand):
    """Export exact stamps in a pickle-free xFit input NPZ."""

    _name_ = "data-export-xfit-input"

    dataset_dir = None
    output_path = None

    class DatasetDirArg(PathSpecInvariant):
        _arg = "--dataset-dir"
        _help = "Packaged XScan dataset containing difference.npy."
        _mandatory = True

    class OutputPathArg(PathSpecInvariant):
        _arg = "--output"
        _help = "New .npz path for exact, unique xFit input stamps."
        _mandatory = True

    def run(self) -> None:
        payload = self._call(
            export_xfit_input_workflow,
            dataset_dir=Path(self.dataset_dir).expanduser(),
            output_path=Path(self.output_path).expanduser(),
        )
        self._emit_json(payload)


class DataMergeCommand(XScanCommand):
    """Merge compatible packaged XScan datasets."""

    dataset_dirs = None
    output_dir = None
    dataset_kind = None

    class DatasetDirsArg(StringInvariant):
        _arg = "--dataset-dirs"
        _help = "Comma-separated packaged dataset directories to merge."
        _mandatory = True

    class OutputDirArg(PathSpecInvariant):
        _arg = "--output-dir"
        _help = "Merged output dataset directory."
        _mandatory = True

    class DatasetKindArg(DatasetKindInvariant):
        _arg = "--dataset-kind"
        _help = "Optional expected dataset kind."
        _mandatory = False
        _default = None

    def run(self) -> None:
        result = self._call(
            merge_dataset_workflow,
            dataset_dirs=self._csv_paths(self.dataset_dirs),
            output_dir=Path(self.output_dir).expanduser(),
            dataset_kind=self.dataset_kind,
        )
        self._emit_json(result.summary)


class _PreparedDatasetBuildCommand(XScanCommand):
    manifest = None
    output_dir = None
    dataset_kind = None

    class ManifestArg(PathSpecInvariant):
        _arg = "--manifest"
        _help = "Manifest JSON/YAML describing prepared dataset arrays."
        _mandatory = True

    class OutputDirArg(PathSpecInvariant):
        _arg = "--output-dir"
        _help = "Output dataset directory."
        _mandatory = True


class DataBuildAutoscanCommand(_PreparedDatasetBuildCommand):
    """Package a prepared autoScan-style DES dataset into canonical layout."""

    def run(self) -> None:
        result = self._call(
            build_prepared_dataset_workflow,
            manifest_path=Path(self.manifest).expanduser(),
            output_dir=Path(self.output_dir).expanduser(),
            dataset_kind="autoscan",
        )
        self._emit_json(result.summary)


class DataBuildAutoscanRawCommand(_PreparedDatasetBuildCommand):
    """Build a canonical autoScan dataset from raw detection records."""

    def run(self) -> None:
        result = self._call(
            build_raw_autoscan_workflow,
            manifest_path=Path(self.manifest).expanduser(),
            output_dir=Path(self.output_dir).expanduser(),
        )
        self._emit_json(result.summary)


class DataBuildNodiffCommand(_PreparedDatasetBuildCommand):
    """Package a prepared no-Diff DES dataset into canonical layout."""

    def run(self) -> None:
        result = self._call(
            build_prepared_dataset_workflow,
            manifest_path=Path(self.manifest).expanduser(),
            output_dir=Path(self.output_dir).expanduser(),
            dataset_kind="nodiff",
        )
        self._emit_json(result.summary)


class DataBuildNodiffRawCommand(_PreparedDatasetBuildCommand):
    """Build a canonical no-Diff dataset from raw search/template images."""

    def run(self) -> None:
        result = self._call(
            build_raw_nodiff_workflow,
            manifest_path=Path(self.manifest).expanduser(),
            output_dir=Path(self.output_dir).expanduser(),
        )
        self._emit_json(result.summary)


class DataBuildNodiffReleaseCommand(_PreparedDatasetBuildCommand):
    """Build a canonical no-Diff dataset from released stamp shards."""

    _shortname_ = "dbnrel"

    def run(self) -> None:
        result = self._call(
            build_nodiff_release_workflow,
            manifest_path=Path(self.manifest).expanduser(),
            output_dir=Path(self.output_dir).expanduser(),
        )
        self._emit_json(result.summary)


class DataBuildHscCommand(_PreparedDatasetBuildCommand):
    """Build an HSC-domain synthetic dataset from real HSC image products."""

    def run(self) -> None:
        result = self._call(
            build_hsc_workflow,
            manifest_path=Path(self.manifest).expanduser(),
            output_dir=Path(self.output_dir).expanduser(),
        )
        self._emit_json(result.summary)


class DataBuildLsstcomcamSmokeCommand(_PreparedDatasetBuildCommand):
    """Build a tiny registry-backed LSSTComCam smoke dataset."""

    def run(self) -> None:
        result = self._call(
            build_lsstcomcam_smoke_workflow,
            manifest_path=Path(self.manifest).expanduser(),
            output_dir=Path(self.output_dir).expanduser(),
        )
        self._emit_json(result.summary)


class DataCheckLsstcomcamCandidatesCommand(XScanCommand):
    """Preflight LSSTComCam candidate catalogs before stamp building."""

    manifest = None
    strict = None
    require_ok = None

    class ManifestArg(PathSpecInvariant):
        _arg = "--manifest"
        _help = "Manifest JSON/YAML with registry and candidate_catalog."
        _mandatory = True

    class StrictArg(BoolInvariant):
        _arg = "--strict"
        _help = (
            "Require every candidate row to match compatible FITS pairs and "
            "require non-empty, unique candidate IDs."
        )
        _mandatory = False
        _default = False

    class RequireOkArg(BoolInvariant):
        _arg = "--require-ok"
        _help = "Exit with an error when the candidate preflight fails."
        _mandatory = False
        _default = False

    def run(self) -> None:
        result = self._call(
            check_lsstcomcam_candidates_workflow,
            manifest_path=Path(self.manifest).expanduser(),
            strict=bool(self.strict),
        )
        self._emit_json(result)
        if bool(self.require_ok) and not result.get("ok"):
            raise CommandError(
                "LSSTComCam candidate catalog check failed: "
                + ", ".join(result.get("errors") or ["ok=false"])
            )


class DataPlanLsstcomcamStagingCommand(XScanCommand):
    """List FITS files needed by an LSSTComCam smoke manifest."""

    manifest = None
    sample_count = None

    class ManifestArg(PathSpecInvariant):
        _arg = "--manifest"
        _help = "Manifest JSON/YAML with registry and optional candidates."
        _mandatory = True

    class SampleCountArg(PositiveIntegerInvariant):
        _arg = "--sample-count"
        _help = (
            "Override manifest sample_count for staging. [default: manifest]"
        )
        _mandatory = False
        _default = None

    def run(self) -> None:
        result = self._call(
            plan_lsstcomcam_staging_workflow,
            manifest_path=Path(self.manifest).expanduser(),
            sample_count=(
                int(self.sample_count)
                if self.sample_count is not None
                else None
            ),
        )
        self._emit_json(result)


class DataStageLsstcomcamFitsCommand(XScanCommand):
    """Stage exact FITS inputs needed by an LSSTComCam smoke manifest."""

    manifest = None
    source_prefix = None
    target_prefix = None
    search_roots = None
    sample_count = None
    link_mode = None
    duplicate_policy = None
    force = None
    dry_run = None

    class ManifestArg(PathSpecInvariant):
        _arg = "--manifest"
        _help = "Manifest JSON/YAML with registry and optional candidates."
        _mandatory = True

    class SourcePrefixArg(PathSpecInvariant):
        _arg = "--source-prefix"
        _help = "Original registry path prefix to stage from."
        _mandatory = True

    class TargetPrefixArg(PathSpecInvariant):
        _arg = "--target-prefix"
        _help = "Local staged FITS prefix to populate."
        _mandatory = True

    class SearchRootsArg(StringInvariant):
        _arg = "--search-roots"
        _help = "Comma-separated roots containing local FITS candidates."
        _mandatory = True

    class SampleCountArg(PositiveIntegerInvariant):
        _arg = "--sample-count"
        _help = (
            "Override manifest sample_count for staging. [default: manifest]"
        )
        _mandatory = False
        _default = None

    class LinkModeArg(StageLinkModeInvariant):
        _arg = "--link-mode"
        _help = "How to stage resolved FITS files. [default: %default]"
        _mandatory = False
        _default = "symlink"

    class DuplicatePolicyArg(StageDuplicatePolicyInvariant):
        _arg = "--duplicate-policy"
        _help = "How to resolve duplicate local matches. [default: %default]"
        _mandatory = False
        _default = "same-size"

    class ForceArg(BoolInvariant):
        _arg = "--force"
        _help = "Overwrite existing staged files or links."
        _mandatory = False
        _default = False

    class DryRunArg(BoolInvariant):
        _arg = "--dry-run"
        _help = "Resolve matches without writing staged files."
        _mandatory = False
        _default = False

    def run(self) -> None:
        result = self._call(
            stage_lsstcomcam_fits_workflow,
            manifest_path=Path(self.manifest).expanduser(),
            source_prefix=str(self.source_prefix),
            target_prefix=Path(self.target_prefix).expanduser(),
            search_roots=self._csv_paths(self.search_roots),
            sample_count=(
                int(self.sample_count)
                if self.sample_count is not None
                else None
            ),
            link_mode=self.link_mode,
            duplicate_policy=self.duplicate_policy,
            force=bool(self.force),
            dry_run=bool(self.dry_run),
        )
        self._emit_json(result)


class DataCheckTrainingLabelsCommand(XScanCommand):
    """Preflight label provenance before training a real-bogus model."""

    dataset_dir = None
    require_ok = None

    class DatasetDirArg(PathSpecInvariant):
        _arg = "--dataset-dir"
        _help = "Packaged dataset directory to check."
        _mandatory = True

    class RequireOkArg(BoolInvariant):
        _arg = "--require-ok"
        _help = "Exit with an error when label provenance is not trainable."
        _mandatory = False
        _default = False

    def run(self) -> None:
        result = self._call(
            check_training_labels_workflow,
            dataset_dir=Path(self.dataset_dir).expanduser(),
        )
        self._emit_json(result)
        if bool(self.require_ok) and not result.get("ok"):
            raise CommandError(
                "training label provenance check failed: "
                + ", ".join(result.get("errors") or ["ok=false"])
            )


class DataBuildHscRegistryCommand(XScanCommand):
    """Build a Butler-style registry from local HSC FITS products."""

    fits_root = None
    output_path = None
    hsc_npy_dir = None
    butler_run = None

    class FitsRootArg(PathSpecInvariant):
        _arg = "--fits-root"
        _help = "Root directory containing HSC/FITS products."
        _mandatory = True

    class OutputPathArg(PathSpecInvariant):
        _arg = "--output-path"
        _help = "Registry parquet/csv/json/jsonl output path."
        _mandatory = True

    class HscNpyDirArg(PathSpecInvariant):
        _arg = "--hsc-npy-dir"
        _help = "Optional HSC_npy directory for exposure-count validation."
        _mandatory = False
        _default = None

    class ButlerRunArg(StringInvariant):
        _arg = "--butler-run"
        _help = "Run label stored in registry rows. [default: %default]"
        _mandatory = False
        _default = "local-hsc-fits"

    def run(self) -> None:
        result = self._call(
            build_hsc_registry_workflow,
            fits_root=Path(self.fits_root).expanduser(),
            output_path=Path(self.output_path).expanduser(),
            hsc_npy_dir=self._path(self.hsc_npy_dir),
            butler_run=self.butler_run,
        )
        self._emit_json(result.summary)


class ExperimentalBuildHscSyntheticCommand(XScanCommand):
    """Build a lightweight experimental HSC synthetic dataset."""

    base = None
    output_dir = None
    positive_count = None
    negative_count = None
    stamp_size = None
    tile_size = None
    seed = None
    template_source = None
    no_difference = None

    class BaseArg(PathSpecInvariant):
        _arg = "--base"
        _help = "Base directory containing HSC_npy."
        _mandatory = True

    class OutputDirArg(PathSpecInvariant):
        _arg = "--output-dir"
        _help = "Output dataset directory."
        _mandatory = True

    class PositiveCountArg(PositiveIntegerInvariant):
        _arg = "--positive-count"
        _help = "Number of positive samples. [default: %default]"
        _mandatory = False
        _default = 64

    class NegativeCountArg(PositiveIntegerInvariant):
        _arg = "--negative-count"
        _help = "Number of negative samples. [default: %default]"
        _mandatory = False
        _default = 64

    class StampSizeArg(PositiveIntegerInvariant):
        _arg = "--stamp-size"
        _help = "Odd stamp width/height. [default: %default]"
        _mandatory = False
        _default = 51

    class TileSizeArg(PositiveIntegerInvariant):
        _arg = "--tile-size"
        _help = "Split grouping tile size. [default: %default]"
        _mandatory = False
        _default = 256

    class SeedArg(NonNegativeIntegerInvariant):
        _arg = "--seed"
        _help = "Random seed. [default: %default]"
        _mandatory = False
        _default = 0

    class TemplateSourceArg(StringInvariant):
        _arg = "--template-source"
        _help = "Template source mode. [default: %default]"
        _mandatory = False
        _default = "median"

    class NoDifferenceArg(BoolInvariant):
        _arg = "--no-difference"
        _help = "Skip writing difference.npy."
        _mandatory = False
        _default = False

    def run(self) -> None:
        result = self._call(
            build_experimental_hsc_workflow,
            base=self.base,
            output_dir=Path(self.output_dir).expanduser(),
            positive_count=self.positive_count,
            negative_count=self.negative_count,
            stamp_size=self.stamp_size,
            seed=self.seed,
            template_source=self.template_source,
            include_difference=not bool(self.no_difference),
            tile_size=self.tile_size,
        )
        self._emit_json(result.summary)


class _TrainCommand(XScanCommand):
    config = None

    class ConfigArg(PathSpecInvariant):
        _arg = "--config"
        _help = "Training YAML configuration file."
        _mandatory = True


class TrainInadaPairCommand(_TrainCommand):
    """Train the faithful Inada pair model."""

    def run(self) -> None:
        result = self._call(
            train_workflow,
            Path(self.config).expanduser(),
            input_mode_override="pair",
        )
        self._emit_json({"run_dir": str(result.run_dir), **result.summary})


class TrainInadaTripletCommand(_TrainCommand):
    """Train the faithful Inada triplet model."""

    def run(self) -> None:
        result = self._call(
            train_workflow,
            Path(self.config).expanduser(),
            input_mode_override="triplet",
        )
        self._emit_json({"run_dir": str(result.run_dir), **result.summary})


class InferRealBogusCommand(XScanCommand):
    """Run inference for a trained real-bogus model on one split."""

    run_dir = None
    dataset_dir = None
    split = None
    batch_size = None
    use_x_fit_features = None
    x_fit_feature_dir = None

    class RunDirArg(PathSpecInvariant):
        _arg = "--run-dir"
        _help = "Training run directory."
        _mandatory = True

    class DatasetDirArg(PathSpecInvariant):
        _arg = "--dataset-dir"
        _help = "Packaged dataset directory."
        _mandatory = True

    class SplitArg(StringInvariant):
        _arg = "--split"
        _help = "Dataset split to score. [default: %default]"
        _mandatory = False
        _default = "test"

    class BatchSizeArg(PositiveIntegerInvariant):
        _arg = "--batch-size"
        _help = "Inference batch size. [default: %default]"
        _mandatory = False
        _default = 32

    class UseXFitFeaturesArg(BoolInvariant):
        _arg = "--use-xfit-features"
        _help = (
            "Enable xFit feature fusion; requires --xfit-feature-dir. "
            "[default: %default]"
        )
        _mandatory = False
        _default = False

    class XFitFeatureDirArg(PathSpecInvariant):
        _arg = "--xfit-feature-dir"
        _help = "Feature bundle location used with --use-xfit-features."
        _mandatory = False
        _default = None

    def run(self) -> None:
        result = self._call(
            infer_workflow,
            run_dir=Path(self.run_dir).expanduser(),
            dataset_dir=Path(self.dataset_dir).expanduser(),
            split=self.split,
            batch_size=self.batch_size,
            use_xfit_features=bool(self.use_x_fit_features),
            xfit_feature_dir=self._path(self.x_fit_feature_dir),
        )
        self._emit_json(result.summary)


class EvaluateRealBogusCommand(XScanCommand):
    """Evaluate a trained real-bogus model on one split."""

    run_dir = None
    dataset_dir = None
    split = None
    batch_size = None
    use_x_fit_features = None
    x_fit_feature_dir = None
    allow_x_fit_coverage_mismatch = None

    class RunDirArg(PathSpecInvariant):
        _arg = "--run-dir"
        _help = "Training run directory."
        _mandatory = True

    class DatasetDirArg(PathSpecInvariant):
        _arg = "--dataset-dir"
        _help = "Packaged dataset directory."
        _mandatory = True

    class SplitArg(StringInvariant):
        _arg = "--split"
        _help = "Dataset split to evaluate. [default: %default]"
        _mandatory = False
        _default = "test"

    class BatchSizeArg(PositiveIntegerInvariant):
        _arg = "--batch-size"
        _help = "Evaluation batch size. [default: %default]"
        _mandatory = False
        _default = 32

    class UseXFitFeaturesArg(BoolInvariant):
        _arg = "--use-xfit-features"
        _help = (
            "Enable xFit feature fusion; requires --xfit-feature-dir. "
            "[default: %default]"
        )
        _mandatory = False
        _default = False

    class XFitFeatureDirArg(PathSpecInvariant):
        _arg = "--xfit-feature-dir"
        _help = "Feature bundle location used with --use-xfit-features."
        _mandatory = False
        _default = None

    class AllowXFitCoverageMismatchArg(BoolInvariant):
        _arg = "--allow-xfit-coverage-mismatch"
        _help = (
            "Allow evaluated-split fit_present coverage to differ materially "
            "from validation. [default: %default]"
        )
        _mandatory = False
        _default = False

    def run(self) -> None:
        result = self._call(
            evaluate_workflow,
            run_dir=Path(self.run_dir).expanduser(),
            dataset_dir=Path(self.dataset_dir).expanduser(),
            split=self.split,
            batch_size=self.batch_size,
            use_xfit_features=bool(self.use_x_fit_features),
            xfit_feature_dir=self._path(self.x_fit_feature_dir),
            allow_xfit_coverage_mismatch=bool(
                self.allow_x_fit_coverage_mismatch
            ),
        )
        self._emit_json(result.summary)


class ReviewQueueCommand(XScanCommand):
    """Build a prioritized human-review queue from model predictions."""

    run_dir = None
    dataset_dir = None
    split = None
    output_dir = None
    compare_run_dirs = None
    max_items = None
    strategy = None

    class RunDirArg(PathSpecInvariant):
        _arg = "--run-dir"
        _help = (
            "Training run directory containing evaluation/inference outputs."
        )
        _mandatory = True

    class DatasetDirArg(PathSpecInvariant):
        _arg = "--dataset-dir"
        _help = "Packaged dataset directory that was scored."
        _mandatory = True

    class SplitArg(StringInvariant):
        _arg = "--split"
        _help = "Dataset split to review. [default: %default]"
        _mandatory = False
        _default = "test"

    class OutputDirArg(PathSpecInvariant):
        _arg = "--output-dir"
        _help = "Review output directory. Defaults under the run directory."
        _mandatory = False
        _default = None

    class CompareRunDirsArg(StringInvariant):
        _arg = "--compare-run-dirs"
        _help = "Optional comma-separated run dirs for disagreement ranking."
        _mandatory = False
        _default = None

    class MaxItemsArg(PositiveIntegerInvariant):
        _arg = "--max-items"
        _help = "Maximum queued review items. [default: %default]"
        _mandatory = False
        _default = 200

    class StrategyArg(ReviewStrategyInvariant):
        _arg = "--strategy"
        _help = "Queue ranking strategy. [default: %default]"
        _mandatory = False
        _default = "hybrid"

    def run(self) -> None:
        compare_run_dirs = (
            self._csv_paths(self.compare_run_dirs)
            if self.compare_run_dirs
            else None
        )
        result = self._call(
            review_queue_workflow,
            run_dir=Path(self.run_dir).expanduser(),
            dataset_dir=Path(self.dataset_dir).expanduser(),
            split=self.split,
            output_dir=self._path(self.output_dir),
            compare_run_dirs=compare_run_dirs,
            max_items=self.max_items,
            strategy=self.strategy,
        )
        self._emit_json(result.summary)


class ReviewQueueDatasetCommand(XScanCommand):
    """Build a human-review queue directly from dataset samples."""

    dataset_dir = None
    split = None
    output_dir = None
    max_items = None

    class DatasetDirArg(PathSpecInvariant):
        _arg = "--dataset-dir"
        _help = "Packaged dataset directory to review."
        _mandatory = True

    class SplitArg(StringInvariant):
        _arg = "--split"
        _help = (
            "Dataset split to review: all, train, val, or test. "
            "[default: %default]"
        )
        _mandatory = False
        _default = "all"

    class OutputDirArg(PathSpecInvariant):
        _arg = "--output-dir"
        _help = (
            "Review output directory. Defaults under the dataset directory."
        )
        _mandatory = False
        _default = None

    class MaxItemsArg(PositiveIntegerInvariant):
        _arg = "--max-items"
        _help = "Maximum queued review items. [default: %default]"
        _mandatory = False
        _default = 200

    def run(self) -> None:
        result = self._call(
            dataset_review_queue_workflow,
            dataset_dir=Path(self.dataset_dir).expanduser(),
            split=self.split,
            output_dir=self._path(self.output_dir),
            max_items=self.max_items,
        )
        self._emit_json(result.summary)


class ReviewQueueSplitsCommand(XScanCommand):
    """Build review queues for multiple dataset splits."""

    run_dir = None
    dataset_dir = None
    output_root = None
    splits = None
    compare_run_dirs = None
    max_items = None
    strategy = None

    class RunDirArg(PathSpecInvariant):
        _arg = "--run-dir"
        _help = (
            "Training run directory containing evaluation/inference outputs."
        )
        _mandatory = True

    class DatasetDirArg(PathSpecInvariant):
        _arg = "--dataset-dir"
        _help = "Packaged dataset directory that was scored."
        _mandatory = True

    class OutputRootArg(PathSpecInvariant):
        _arg = "--output-root"
        _help = "Directory where per-split review queues are written."
        _mandatory = True

    class SplitsArg(StringInvariant):
        _arg = "--splits"
        _help = "Comma-separated splits. [default: %default]"
        _mandatory = False
        _default = "train,val,test"

    class CompareRunDirsArg(StringInvariant):
        _arg = "--compare-run-dirs"
        _help = "Optional comma-separated run dirs for disagreement ranking."
        _mandatory = False
        _default = None

    class MaxItemsArg(PositiveIntegerInvariant):
        _arg = "--max-items"
        _help = "Maximum queued review items per split. [default: %default]"
        _mandatory = False
        _default = 200

    class StrategyArg(ReviewStrategyInvariant):
        _arg = "--strategy"
        _help = "Queue ranking strategy. [default: %default]"
        _mandatory = False
        _default = "hybrid"

    def run(self) -> None:
        compare_run_dirs = (
            self._csv_paths(self.compare_run_dirs)
            if self.compare_run_dirs
            else None
        )
        result = self._call(
            review_queue_splits_workflow,
            run_dir=Path(self.run_dir).expanduser(),
            dataset_dir=Path(self.dataset_dir).expanduser(),
            output_root=Path(self.output_root).expanduser(),
            splits=[
                item.strip()
                for item in self.splits.split(",")
                if item.strip()
            ],
            compare_run_dirs=compare_run_dirs,
            max_items=self.max_items,
            strategy=self.strategy,
        )
        self._emit_json(result.summary)


class ReviewBokehCommand(XScanCommand):
    """Run the local Bokeh server for a saved review queue."""

    review_dir = None
    host = None
    port = None
    show_url_only = None

    class ReviewDirArg(PathSpecInvariant):
        _arg = "--review-dir"
        _help = "Review directory containing manifest.json and queue.jsonl."
        _mandatory = True

    class HostArg(StringInvariant):
        _arg = "--host"
        _help = "Bokeh server host. [default: %default]"
        _mandatory = False
        _default = "localhost"

    class PortArg(PositiveIntegerInvariant):
        _arg = "--port"
        _help = "Bokeh server port. [default: %default]"
        _mandatory = False
        _default = 5006

    class ShowUrlOnlyArg(BoolInvariant):
        _arg = "--show-url-only"
        _help = "Validate inputs and emit server metadata without blocking."
        _mandatory = False
        _default = False

    def run(self) -> None:
        result = self._call(
            review_bokeh_workflow,
            review_dir=Path(self.review_dir).expanduser(),
            host=self.host,
            port=self.port,
            show_url_only=self.show_url_only,
        )
        if self.show_url_only:
            self._emit_json(result.summary)


class ReviewRawCompareCommand(XScanCommand):
    """Run the raw-array comparison Bokeh server for a review queue."""

    _shortname_ = "rrc"
    review_dir = None
    host = None
    port = None

    class ReviewDirArg(PathSpecInvariant):
        _arg = "--review-dir"
        _help = "Review directory containing manifest.json and queue.jsonl."
        _mandatory = _DEFAULT_REVIEW_DIR is None
        _default = _DEFAULT_REVIEW_DIR

    class HostArg(StringInvariant):
        _arg = "--host"
        _help = "Bokeh server host. [default: %default]"
        _mandatory = False
        _default = _DEFAULT_RAW_COMPARE_HOST

    class PortArg(PositiveIntegerInvariant):
        _arg = "--port"
        _help = "Bokeh server port. [default: %default]"
        _mandatory = False
        _default = _DEFAULT_RAW_COMPARE_PORT

    def run(self) -> None:
        if self.review_dir is None:
            raise CommandError(
                "--review-dir or CUPHOTON_XSCAN_REVIEW_DIR is required"
            )
        self._call(
            run_raw_compare_review_server,
            review_dir=Path(self.review_dir).expanduser(),
            host=self.host,
            port=self.port,
        )


class ReviewAlardLuptonCommand(XScanCommand):
    """Run the Alard-Lupton display-lab Bokeh server."""

    _shortname_ = "ral"
    review_dir = None
    host = None
    port = None

    class ReviewDirArg(PathSpecInvariant):
        _arg = "--review-dir"
        _help = "Review directory containing manifest.json and queue.jsonl."
        _mandatory = _DEFAULT_REVIEW_DIR is None
        _default = _DEFAULT_REVIEW_DIR

    class HostArg(StringInvariant):
        _arg = "--host"
        _help = "Bokeh server host. [default: %default]"
        _mandatory = False
        _default = _DEFAULT_ALARD_LUPTON_HOST

    class PortArg(PositiveIntegerInvariant):
        _arg = "--port"
        _help = "Bokeh server port. [default: %default]"
        _mandatory = False
        _default = _DEFAULT_ALARD_LUPTON_PORT

    def run(self) -> None:
        if self.review_dir is None:
            raise CommandError(
                "--review-dir or CUPHOTON_XSCAN_REVIEW_DIR is required"
            )
        self._call(
            run_alard_lupton_review_server,
            review_dir=Path(self.review_dir).expanduser(),
            host=self.host,
            port=self.port,
        )


class ReviewStatusCommand(XScanCommand):
    """Summarize whether a saved review queue is ready for review-apply."""

    review_dir = None
    min_reviewers = None
    min_actionable_reviewers = None
    consensus_rule = None
    include_decisions = None
    require_ready = None

    class ReviewDirArg(PathSpecInvariant):
        _arg = "--review-dir"
        _help = "Review directory containing manifest.json and queue.jsonl."
        _mandatory = True

    class MinReviewersArg(PositiveIntegerInvariant):
        _arg = "--min-reviewers"
        _help = (
            "Minimum latest reviewer annotations per item. "
            "[default: %default]"
        )
        _mandatory = False
        _default = 2

    class MinActionableReviewersArg(PositiveIntegerInvariant):
        _arg = "--min-actionable-reviewers"
        _help = (
            "Minimum non-unsure latest reviewer annotations per item. "
            "[default: %default]"
        )
        _mandatory = False
        _default = 2

    class ConsensusRuleArg(ReviewConsensusRuleInvariant):
        _arg = "--consensus-rule"
        _help = "Consensus rule for actionable labels. [default: %default]"
        _mandatory = False
        _default = "unanimous"

    class IncludeDecisionsArg(BoolInvariant):
        _arg = "--include-decisions"
        _help = "Include per-queue aggregation decisions in the JSON output."
        _mandatory = False
        _default = False

    class RequireReadyArg(BoolInvariant):
        _arg = "--require-ready"
        _help = "Exit non-zero unless every queued item is actionable."
        _mandatory = False
        _default = False

    def run(self) -> None:
        result = self._call(
            review_status_workflow,
            review_dir=Path(self.review_dir).expanduser(),
            min_reviewers=self.min_reviewers,
            min_actionable_reviewers=self.min_actionable_reviewers,
            consensus_rule=self.consensus_rule,
            include_decisions=self.include_decisions,
        )
        self._emit_json(result.summary)
        if (
            self.require_ready
            and not result.summary["ready_for_review_apply"]
        ):
            raise CommandError(
                "review annotations are not ready for review-apply"
            )


class ReviewContactSheetCommand(XScanCommand):
    """Export static PNG contact sheets for a saved review queue."""

    review_dir = None
    output_dir = None
    max_items = None
    items_per_page = None
    columns = None
    stamp_size = None
    overwrite = None

    class ReviewDirArg(PathSpecInvariant):
        _arg = "--review-dir"
        _help = "Review directory containing manifest.json and queue.jsonl."
        _mandatory = True

    class OutputDirArg(PathSpecInvariant):
        _arg = "--output-dir"
        _help = "Output directory for contact-sheet PNGs and index.json."
        _mandatory = True

    class MaxItemsArg(PositiveIntegerInvariant):
        _arg = "--max-items"
        _help = "Maximum queued review items to export. [default: %default]"
        _mandatory = False
        _default = 64

    class ItemsPerPageArg(PositiveIntegerInvariant):
        _arg = "--items-per-page"
        _help = "Maximum queued items per PNG page. [default: %default]"
        _mandatory = False
        _default = 16

    class ColumnsArg(PositiveIntegerInvariant):
        _arg = "--columns"
        _help = "Review items per contact-sheet row. [default: %default]"
        _mandatory = False
        _default = 4

    class StampSizeArg(PositiveIntegerInvariant):
        _arg = "--stamp-size"
        _help = "Rendered stamp size in pixels. [default: %default]"
        _mandatory = False
        _default = 96

    class OverwriteArg(BoolInvariant):
        _arg = "--overwrite"
        _help = "Replace prior contact-sheet PNGs and index.json."
        _mandatory = False
        _default = False

    def run(self) -> None:
        result = self._call(
            review_contact_sheet_workflow,
            review_dir=Path(self.review_dir).expanduser(),
            output_dir=Path(self.output_dir).expanduser(),
            max_items=self.max_items,
            items_per_page=self.items_per_page,
            columns=self.columns,
            stamp_size=self.stamp_size,
            overwrite=self.overwrite,
        )
        self._emit_json(result.summary)


class ReviewAnnotationTemplateCommand(XScanCommand):
    """Write a CSV template for offline review annotations."""

    review_dir = None
    output_csv = None
    reviewer = None
    overwrite = None

    class ReviewDirArg(PathSpecInvariant):
        _arg = "--review-dir"
        _help = "Review directory containing manifest.json and queue.jsonl."
        _mandatory = True

    class OutputCsvArg(PathSpecInvariant):
        _arg = "--output-csv"
        _help = "Output CSV path for offline annotations."
        _mandatory = True

    class ReviewerArg(StringInvariant):
        _arg = "--reviewer"
        _help = "Optional reviewer name to prefill in the CSV."
        _mandatory = False
        _default = None

    class OverwriteArg(BoolInvariant):
        _arg = "--overwrite"
        _help = "Replace an existing annotation-template CSV."
        _mandatory = False
        _default = False

    def run(self) -> None:
        result = self._call(
            review_annotation_template_workflow,
            review_dir=Path(self.review_dir).expanduser(),
            output_csv=Path(self.output_csv).expanduser(),
            reviewer=self.reviewer,
            overwrite=self.overwrite,
        )
        self._emit_json(result.summary)


class ReviewImportAnnotationsCommand(XScanCommand):
    """Validate and append offline review annotations from CSV."""

    review_dir = None
    input_csv = None
    reviewer = None
    dry_run = None
    require_all = None

    class ReviewDirArg(PathSpecInvariant):
        _arg = "--review-dir"
        _help = "Review directory containing manifest.json and queue.jsonl."
        _mandatory = True

    class InputCsvArg(PathSpecInvariant):
        _arg = "--input-csv"
        _help = "Filled annotation CSV to validate and import."
        _mandatory = True

    class ReviewerArg(StringInvariant):
        _arg = "--reviewer"
        _help = "Reviewer name override for all imported rows."
        _mandatory = False
        _default = None

    class DryRunArg(BoolInvariant):
        _arg = "--dry-run"
        _help = "Validate rows without appending annotations.jsonl."
        _mandatory = False
        _default = False

    class RequireAllArg(BoolInvariant):
        _arg = "--require-all"
        _help = "Require a non-blank annotation for every queued item."
        _mandatory = False
        _default = False

    def run(self) -> None:
        result = self._call(
            review_import_annotations_workflow,
            review_dir=Path(self.review_dir).expanduser(),
            input_csv=Path(self.input_csv).expanduser(),
            reviewer=self.reviewer,
            dry_run=self.dry_run,
            require_all=self.require_all,
        )
        self._emit_json(result.summary)


class ReviewAggregateCommand(XScanCommand):
    """Aggregate latest per-reviewer annotations into explicit decisions."""

    _shortname_ = "rag"
    review_dir = None
    output_report = None
    min_reviewers = None
    min_actionable_reviewers = None
    consensus_rule = None

    class ReviewDirArg(PathSpecInvariant):
        _arg = "--review-dir"
        _help = "Review directory containing annotations.jsonl."
        _mandatory = True

    class OutputReportArg(PathSpecInvariant):
        _arg = "--output-report"
        _help = "Optional JSON report path. Omit for dry-run summary only."
        _mandatory = False
        _default = None

    class MinReviewersArg(PositiveIntegerInvariant):
        _arg = "--min-reviewers"
        _help = (
            "Minimum latest reviewer annotations per item. "
            "[default: %default]"
        )
        _mandatory = False
        _default = 2

    class MinActionableReviewersArg(PositiveIntegerInvariant):
        _arg = "--min-actionable-reviewers"
        _help = (
            "Minimum non-unsure latest reviewer annotations per item. "
            "[default: %default]"
        )
        _mandatory = False
        _default = 2

    class ConsensusRuleArg(ReviewConsensusRuleInvariant):
        _arg = "--consensus-rule"
        _help = "Consensus rule for actionable labels. [default: %default]"
        _mandatory = False
        _default = "unanimous"

    def run(self) -> None:
        result = self._call(
            review_aggregate_workflow,
            review_dir=Path(self.review_dir).expanduser(),
            output_report=self._path(self.output_report),
            min_reviewers=self.min_reviewers,
            min_actionable_reviewers=self.min_actionable_reviewers,
            consensus_rule=self.consensus_rule,
        )
        self._emit_json(result.summary)


class ReviewApplyCommand(XScanCommand):
    """Apply reviewed binary labels to a new packaged dataset."""

    dataset_dir = None
    review_dir = None
    output_dir = None
    aggregation_report = None

    class DatasetDirArg(PathSpecInvariant):
        _arg = "--dataset-dir"
        _help = "Source packaged dataset directory."
        _mandatory = True

    class ReviewDirArg(PathSpecInvariant):
        _arg = "--review-dir"
        _help = "Review directory containing annotations.jsonl."
        _mandatory = True

    class OutputDirArg(PathSpecInvariant):
        _arg = "--output-dir"
        _help = "Output reviewed dataset directory."
        _mandatory = True

    class AggregationReportArg(PathSpecInvariant):
        _arg = "--aggregation-report"
        _help = "review-aggregate JSON report for multi-reviewer labels."
        _mandatory = False
        _default = None

    def run(self) -> None:
        result = self._call(
            review_apply_workflow,
            dataset_dir=Path(self.dataset_dir).expanduser(),
            review_dir=Path(self.review_dir).expanduser(),
            output_dir=Path(self.output_dir).expanduser(),
            aggregation_report=self._path(self.aggregation_report),
        )
        self._emit_json(result.summary)


class EntityReviewQueueCommand(XScanCommand):
    """Build an entity-class review queue from binary real annotations."""

    source_review_dirs = None
    output_dir = None

    class SourceReviewDirsArg(StringInvariant):
        _arg = "--source-review-dirs"
        _help = "Comma-separated binary review directories to mine."
        _mandatory = True

    class OutputDirArg(PathSpecInvariant):
        _arg = "--output-dir"
        _help = "Entity review output directory."
        _mandatory = True

    def run(self) -> None:
        result = self._call(
            entity_review_queue_workflow,
            source_review_dirs=self._csv_paths(self.source_review_dirs),
            output_dir=Path(self.output_dir).expanduser(),
        )
        self._emit_json(result.summary)


class EntityReviewBokehCommand(XScanCommand):
    """Run the local Bokeh server for an entity-class review queue."""

    _shortname_ = "erbk"
    review_dir = None
    host = None
    port = None
    show_url_only = None

    class ReviewDirArg(PathSpecInvariant):
        _arg = "--review-dir"
        _help = "Entity review directory with manifest.json and queue.jsonl."
        _mandatory = True

    class HostArg(StringInvariant):
        _arg = "--host"
        _help = "Bokeh server host. [default: %default]"
        _mandatory = False
        _default = "localhost"

    class PortArg(PositiveIntegerInvariant):
        _arg = "--port"
        _help = "Bokeh server port. [default: %default]"
        _mandatory = False
        _default = 5007

    class ShowUrlOnlyArg(BoolInvariant):
        _arg = "--show-url-only"
        _help = "Validate inputs and emit server metadata without blocking."
        _mandatory = False
        _default = False

    def run(self) -> None:
        result = self._call(
            entity_review_bokeh_workflow,
            review_dir=Path(self.review_dir).expanduser(),
            host=self.host,
            port=self.port,
            show_url_only=self.show_url_only,
        )
        if self.show_url_only:
            self._emit_json(result.summary)


class EntityReviewAggregateCommand(XScanCommand):
    """Aggregate entity-class annotations into consensus reports."""

    review_dir = None
    output_report = None
    min_reviewers = None
    consensus_rule = None

    class ReviewDirArg(PathSpecInvariant):
        _arg = "--review-dir"
        _help = "Entity review directory containing entity_annotations.jsonl."
        _mandatory = True

    class OutputReportArg(PathSpecInvariant):
        _arg = "--output-report"
        _help = "Optional JSON report path. Omit for dry-run summary only."
        _mandatory = False
        _default = None

    class MinReviewersArg(PositiveIntegerInvariant):
        _arg = "--min-reviewers"
        _help = (
            "Minimum latest reviewer annotations per item. "
            "[default: %default]"
        )
        _mandatory = False
        _default = 1

    class ConsensusRuleArg(ReviewConsensusRuleInvariant):
        _arg = "--consensus-rule"
        _help = "Consensus rule for entity labels. [default: %default]"
        _mandatory = False
        _default = "unanimous"

    def run(self) -> None:
        result = self._call(
            entity_review_aggregate_workflow,
            review_dir=Path(self.review_dir).expanduser(),
            output_report=self._path(self.output_report),
            min_reviewers=self.min_reviewers,
            consensus_rule=self.consensus_rule,
        )
        self._emit_json(result.summary)


class CompareInputsCommand(XScanCommand):
    """Compare evaluation summaries from multiple run directories."""

    run_dirs = None

    class RunDirsArg(StringInvariant):
        _arg = "--run-dirs"
        _help = "Comma-separated run directories to compare."
        _mandatory = True

    def run(self) -> None:
        result = self._call(
            compare_inputs_workflow,
            self._csv_paths(self.run_dirs),
        )
        self._emit_json(result)


class ReproduceInadaCommand(XScanCommand):
    """Run the multi-seed faithful Inada reproduction orchestration."""

    pair_config = None
    triplet_config = None
    nodiff_pair_config = None
    seeds = None

    class PairConfigArg(PathSpecInvariant):
        _arg = "--pair-config"
        _help = "Training config for the autoScan pair model."
        _mandatory = False
        _default = None

    class TripletConfigArg(PathSpecInvariant):
        _arg = "--triplet-config"
        _help = "Training config for the autoScan triplet model."
        _mandatory = False
        _default = None

    class NodiffPairConfigArg(PathSpecInvariant):
        _arg = "--nodiff-pair-config"
        _help = "Training config for the no-Diff pair model."
        _mandatory = False
        _default = None

    class SeedsArg(CsvIntInvariant):
        _arg = "--seeds"
        _help = "Comma-separated random seeds. [default: %default]"
        _mandatory = False
        _default = "0,1,2,3,4"

    def run(self) -> None:
        result = self._call(
            reproduce_inada_workflow,
            pair_config=self._path(self.pair_config),
            triplet_config=self._path(self.triplet_config),
            nodiff_pair_config=self._path(self.nodiff_pair_config),
            seeds=self._csv_ints(self.seeds),
        )
        self._emit_json(result)


class ReproducePairTripletCommand(XScanCommand):
    """Train matched pair and triplet models on one reviewed dataset."""

    dataset_dir = None
    pair_config = None
    triplet_config = None
    seeds = None
    output_dir = None
    run_name = None

    class DatasetDirArg(PathSpecInvariant):
        _arg = "--dataset-dir"
        _help = "Reviewed packaged dataset directory."
        _mandatory = True

    class PairConfigArg(PathSpecInvariant):
        _arg = "--pair-config"
        _help = "Training config template for the pair model."
        _mandatory = True

    class TripletConfigArg(PathSpecInvariant):
        _arg = "--triplet-config"
        _help = "Training config template for the triplet model."
        _mandatory = True

    class SeedsArg(CsvIntInvariant):
        _arg = "--seeds"
        _help = "Comma-separated random seeds. [default: %default]"
        _mandatory = False
        _default = "0"

    class OutputDirArg(PathSpecInvariant):
        _arg = "--output-dir"
        _help = "Optional output root for the comparison run."
        _mandatory = False
        _default = None

    class RunNameArg(StringInvariant):
        _arg = "--name"
        _help = "Optional comparison run directory name override."
        _mandatory = False
        _default = None
        _minlen = 0

    def run(self) -> None:
        result = self._call(
            reproduce_pair_triplet_workflow,
            dataset_dir=Path(self.dataset_dir).expanduser(),
            pair_config=Path(self.pair_config).expanduser(),
            triplet_config=Path(self.triplet_config).expanduser(),
            seeds=self._csv_ints(self.seeds),
            output_root=self._path(self.output_dir),
            run_name=self.run_name or None,
        )
        self._emit_json({"run_dir": str(result.run_dir), **result.summary})


class ReproduceHscComparisonCommand(XScanCommand):
    """Build and compare pair/simple-triplet/xpois-triplet HSC runs."""

    manifest = None
    pair_config = None
    triplet_config = None
    pair_pretrain_checkpoint = None
    triplet_pretrain_checkpoint = None
    seeds = None
    output_dir = None
    run_name = None

    class ManifestArg(PathSpecInvariant):
        _arg = "--manifest"
        _help = "Base HSC manifest used to derive the comparison datasets."
        _mandatory = True

    class PairConfigArg(PathSpecInvariant):
        _arg = "--pair-config"
        _help = "Training config template for the pair model."
        _mandatory = True

    class TripletConfigArg(PathSpecInvariant):
        _arg = "--triplet-config"
        _help = "Training config template for the triplet model."
        _mandatory = True

    class PairPretrainCheckpointArg(PathSpecInvariant):
        _arg = "--pair-pretrain-checkpoint"
        _help = "Optional checkpoint used for pair fine-tuning variants."
        _mandatory = False
        _default = None

    class TripletPretrainCheckpointArg(PathSpecInvariant):
        _arg = "--triplet-pretrain-checkpoint"
        _help = "Optional checkpoint used for triplet fine-tuning variants."
        _mandatory = False
        _default = None

    class SeedsArg(CsvIntInvariant):
        _arg = "--seeds"
        _help = "Comma-separated random seeds. [default: %default]"
        _mandatory = False
        _default = "0"

    class OutputDirArg(PathSpecInvariant):
        _arg = "--output-dir"
        _help = "Optional output root for the comparison run."
        _mandatory = False
        _default = None

    class RunNameArg(StringInvariant):
        _arg = "--name"
        _help = "Optional comparison run directory name override."
        _mandatory = False
        _default = None
        _minlen = 0

    def run(self) -> None:
        result = self._call(
            reproduce_hsc_comparison_workflow,
            manifest_path=Path(self.manifest).expanduser(),
            pair_config=Path(self.pair_config).expanduser(),
            triplet_config=Path(self.triplet_config).expanduser(),
            seeds=self._csv_ints(self.seeds),
            pair_pretrain_checkpoint=self._path(
                self.pair_pretrain_checkpoint
            ),
            triplet_pretrain_checkpoint=self._path(
                self.triplet_pretrain_checkpoint
            ),
            output_root=self._path(self.output_dir),
            run_name=self.run_name or None,
        )
        self._emit_json({"run_dir": str(result.run_dir), **result.summary})


class ReproduceHscXPOISSweepCommand(XScanCommand):
    """Build and compare HSC pair/simple-triplet/xpois sweep variants."""

    _name_ = "reproduce-hsc-xpois-sweep"
    _shortname_ = "rhxs"

    manifest = None
    pair_config = None
    triplet_config = None
    sweep_config = None
    seeds = None
    output_dir = None
    run_name = None

    class ManifestArg(PathSpecInvariant):
        _arg = "--manifest"
        _help = "Base HSC manifest used to derive the sweep datasets."
        _mandatory = True

    class PairConfigArg(PathSpecInvariant):
        _arg = "--pair-config"
        _help = "Training config template for the pair model."
        _mandatory = True

    class TripletConfigArg(PathSpecInvariant):
        _arg = "--triplet-config"
        _help = "Training config template for the triplet model."
        _mandatory = True

    class SweepConfigArg(PathSpecInvariant):
        _arg = "--sweep-config"
        _help = "YAML config describing the xpois variant sweep."
        _mandatory = True

    class SeedsArg(CsvIntInvariant):
        _arg = "--seeds"
        _help = "Comma-separated random seeds. [default: %default]"
        _mandatory = False
        _default = "0"

    class OutputDirArg(PathSpecInvariant):
        _arg = "--output-dir"
        _help = "Optional output root for the sweep run."
        _mandatory = False
        _default = None

    class RunNameArg(StringInvariant):
        _arg = "--name"
        _help = "Optional sweep run directory name override."
        _mandatory = False
        _default = None
        _minlen = 0

    def run(self) -> None:
        result = self._call(
            reproduce_hsc_xpois_sweep_workflow,
            manifest_path=Path(self.manifest).expanduser(),
            pair_config=Path(self.pair_config).expanduser(),
            triplet_config=Path(self.triplet_config).expanduser(),
            sweep_config=Path(self.sweep_config).expanduser(),
            seeds=self._csv_ints(self.seeds),
            output_root=self._path(self.output_dir),
            run_name=self.run_name or None,
        )
        self._emit_json({"run_dir": str(result.run_dir), **result.summary})
