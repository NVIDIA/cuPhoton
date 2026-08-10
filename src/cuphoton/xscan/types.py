# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared static contracts for XScan configuration and feature artifacts."""

from __future__ import annotations

from typing import Literal, TypeAlias, TypedDict

from torch import Tensor

InputMode: TypeAlias = Literal["pair", "triplet"]
TrainingMode: TypeAlias = Literal["scratch", "fine_tune"]
EarlyStoppingMetric: TypeAlias = Literal[
    "val_roc_auc", "val_pr_auc", "val_loss"
]
MissingPolicy: TypeAlias = Literal["error", "indicator"]
XFitModel: TypeAlias = Literal["gaussian", "stamp"]

# ``default_collate`` preserves tensors and currently represents a collated
# two-tensor tuple as a list. Keep the individual-sample and collated forms
# distinct so fusion callers cannot silently pass arbitrary objects.
ModelFeatures: TypeAlias = Tensor | tuple[Tensor, Tensor]
CollatedModelFeatures: TypeAlias = (
    Tensor | list[Tensor] | tuple[Tensor, Tensor]
)
DatasetSample: TypeAlias = tuple[ModelFeatures, Tensor]
# ``default_collate`` returns an outer list for sequence samples. Its two
# positions contain the collated model features and labels, respectively.
CollatedDatasetBatch: TypeAlias = list[CollatedModelFeatures | Tensor]


class XFitArtifactDigest(TypedDict):
    """Named artifact with a validated SHA-256 digest."""

    name: str
    sha256: str


class XFitSchemaArtifact(TypedDict):
    """Named schema artifact whose file digest defines bundle identity."""

    name: str


class XFitBundleArtifacts(TypedDict):
    """Artifacts directly owned by an xFit feature bundle."""

    candidate_id: XFitArtifactDigest
    features: XFitArtifactDigest
    input_image_sha256: XFitArtifactDigest
    schema: XFitSchemaArtifact


class XFitSourceArtifacts(TypedDict):
    """Validated portable xFit artifacts used to build a feature bundle."""

    summary: XFitArtifactDigest
    fits: XFitArtifactDigest
    fit_arrays: XFitArtifactDigest


class XFitFeatureSource(TypedDict):
    """Validated source-run properties that affect feature semantics."""

    model: XFitModel
    mode: Literal["difference"]
    image_shape: list[int]
    mask_present: bool
    variance_present: bool
    input_archive_sha256: str


class XFitFeatureDefinition(TypedDict):
    """One ordered scalar feature and its canonical transform."""

    index: int
    name: str
    range: list[float]
    transform: str


class XFitJoinDiagnostics(TypedDict):
    """Validated dataset-to-fit alignment counts."""

    dataset_row_count: int
    dataset_unique_candidate_count: int
    duplicate_dataset_row_count: int
    fit_row_count: int
    matched_dataset_row_count: int
    missing_dataset_row_count: int
    reused_fit_row_count: int
    extra_fit_row_count: int
    missing_candidate_ids: list[int | str]
    extra_fit_candidate_ids: list[int | str]


class XFitCoverage(TypedDict):
    """Fit-availability rate for one named dataset split."""

    split: str
    sample_count: int
    fit_present_count: int
    fit_coverage: float


class XFitFeatureSchema(TypedDict):
    """Fully validated schema for the v1 xFit-to-XScan feature contract."""

    schema_version: int
    artifact: str
    feature_names: list[str]
    feature_dtype: str
    candidate_id_dtype: str
    features: list[XFitFeatureDefinition]
    transform_constants: dict[str, float]
    missing_policy: MissingPolicy
    source: XFitFeatureSource
    join_diagnostics: XFitJoinDiagnostics
    source_artifacts: XFitSourceArtifacts
    artifacts: XFitBundleArtifacts


class XFitBundleIdentity(TypedDict):
    """Immutable identity recorded in model checkpoints."""

    schema_sha256: str
    feature_sha256: str
    source_artifacts: XFitSourceArtifacts


__all__ = [
    "CollatedModelFeatures",
    "CollatedDatasetBatch",
    "DatasetSample",
    "EarlyStoppingMetric",
    "InputMode",
    "MissingPolicy",
    "ModelFeatures",
    "TrainingMode",
    "XFitArtifactDigest",
    "XFitBundleArtifacts",
    "XFitBundleIdentity",
    "XFitCoverage",
    "XFitFeatureDefinition",
    "XFitFeatureSchema",
    "XFitFeatureSource",
    "XFitJoinDiagnostics",
    "XFitModel",
    "XFitSchemaArtifact",
    "XFitSourceArtifacts",
]
