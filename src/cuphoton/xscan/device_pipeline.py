# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Persistent, same-process xPois-to-xScan device inference seam.

This module deliberately stops before process orchestration and artifact
publication. A caller binds one CUDA device, initializes one
:class:`DeviceWorkerContext`, and reuses it for serial item execution. Work
items contain only compact file, identity, shape, and candidate descriptors;
device arrays never leave :func:`run_device_pipeline_item`.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import re
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

import numpy as np

from cuphoton.core.artifacts import file_sha256
from cuphoton.core.bulk import json_mapping, validate_identifier

DEVICE_PIPELINE_CONFIG_SCHEMA = "cuphoton.xscan.device-pipeline.config/v2"
DEVICE_PIPELINE_ITEM_SCHEMA = "cuphoton.xscan.device-pipeline.item/v1"
DEVICE_PIPELINE_RESULT_SCHEMA = "cuphoton.xscan.device-pipeline.result/v2"
DEVICE_PIPELINE_EVIDENCE_SCHEMA = (
    "cuphoton.xscan.device-pipeline.scientific-evidence/v1"
)

_CUDA_DEVICE = re.compile(r"cuda:(0|[1-9][0-9]*)")
_TIMING_STAGES = (
    "input_read",
    "input_h2d",
    "xpois",
    "stamp_extraction",
    "xfit",
    "feature_transform",
    "evidence_pack",
    "dlpack_handoff",
    "xscan",
    "terminal_pack",
    "terminal_d2h",
)


def _strict_inference_policy_payload() -> dict[str, Any]:
    """Return the fixed inference performance policy for this pipeline."""

    return {
        "amp_dtype": "off",
        "allow_tf32": False,
        "cudnn_benchmark": False,
        "compile": False,
        "compile_mode": None,
        "compile_backend": None,
        "compile_threads": None,
        "compile_worker_start_method": None,
        "num_workers": 0,
        "worker_start_method": None,
        "worker_cpu_threads": 1,
        "pin_memory": False,
        "persistent_workers": False,
        "non_blocking_transfers": False,
    }


def _require_exact_fields(
    payload: Mapping[str, Any],
    *,
    expected: frozenset[str],
    field_name: str,
) -> dict[str, Any]:
    values = json_mapping(payload, field=field_name)
    missing = sorted(expected - set(values))
    unknown = sorted(set(values) - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise ValueError(
            f"{field_name} fields are invalid: {'; '.join(details)}"
        )
    return values


def _require_sha256(value: str, *, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _require_shape(
    value: tuple[int, int], *, field_name: str, odd: bool = False
) -> tuple[int, int]:
    try:
        shape = tuple(value)
    except TypeError as exc:
        raise TypeError(
            f"{field_name} must be a two-element sequence"
        ) from exc
    if len(shape) != 2:
        raise ValueError(f"{field_name} must contain two dimensions")
    if any(
        isinstance(size, bool) or not isinstance(size, int) for size in shape
    ):
        raise TypeError(f"{field_name} dimensions must be integers")
    if any(size <= 0 for size in shape):
        raise ValueError(f"{field_name} dimensions must be positive")
    if odd and any(size % 2 == 0 for size in shape):
        raise ValueError(f"{field_name} dimensions must be odd")
    return shape


def _require_finite(
    value: int | float, *, field_name: str, minimum: float | None = None
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}")
    return result


def _stable_payload_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class NpyArrayDescriptor:
    """One absolute NPY path with its exact content and array identity."""

    path: str
    sha256: str
    shape: tuple[int, int]
    dtype: str

    def __post_init__(self) -> None:
        if not isinstance(self.path, str):
            raise TypeError("NPY descriptor path must be a string")
        if not isinstance(self.dtype, str):
            raise TypeError("NPY descriptor dtype must be a string")
        path = Path(self.path)
        if not path.is_absolute():
            raise ValueError("NPY descriptor path must be absolute")
        if path.suffix.lower() != ".npy":
            raise ValueError("NPY descriptor path must end in .npy")
        _require_sha256(self.sha256, field_name="NPY descriptor sha256")
        object.__setattr__(
            self,
            "shape",
            _require_shape(self.shape, field_name="NPY descriptor shape"),
        )
        try:
            dtype = np.dtype(self.dtype)
        except (TypeError, ValueError) as exc:
            raise TypeError("NPY descriptor dtype is invalid") from exc
        if (
            dtype.hasobject
            or dtype.fields is not None
            or dtype.subdtype is not None
        ):
            raise TypeError(
                "NPY descriptor dtype must be a scalar numeric dtype"
            )
        object.__setattr__(self, "dtype", dtype.str)

    def to_payload(self) -> dict[str, Any]:
        """Return the compact JSON-compatible descriptor."""

        return {
            "path": self.path,
            "sha256": self.sha256,
            "shape": list(self.shape),
            "dtype": self.dtype,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> NpyArrayDescriptor:
        """Restore one strict descriptor from JSON-compatible values."""

        values = _require_exact_fields(
            payload,
            expected=frozenset({"path", "sha256", "shape", "dtype"}),
            field_name="NPY descriptor",
        )
        return cls(
            path=values["path"],
            sha256=values["sha256"],
            shape=tuple(values["shape"]),
            dtype=values["dtype"],
        )


@dataclass(frozen=True, slots=True)
class DevicePipelineCandidate:
    """One order-preserving candidate center inside an image pair."""

    candidate_id: int | str
    center_x: int
    center_y: int
    source_index: int

    def __post_init__(self) -> None:
        if isinstance(self.candidate_id, bool) or not isinstance(
            self.candidate_id, (int, str)
        ):
            raise TypeError("candidate_id must be an integer or string")
        if isinstance(self.candidate_id, str) and not self.candidate_id:
            raise ValueError("candidate_id must not be empty")
        for name in ("center_x", "center_y", "source_index"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.source_index < 0:
            raise ValueError("source_index must be non-negative")

    def to_payload(self) -> dict[str, Any]:
        """Return this candidate's JSON-compatible descriptor."""

        return asdict(self)

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any]
    ) -> DevicePipelineCandidate:
        """Restore one candidate from JSON-compatible values."""

        values = _require_exact_fields(
            payload,
            expected=frozenset(
                {"candidate_id", "center_x", "center_y", "source_index"}
            ),
            field_name="device pipeline candidate",
        )
        return cls(
            candidate_id=values["candidate_id"],
            center_x=values["center_x"],
            center_y=values["center_y"],
            source_index=values["source_index"],
        )


@dataclass(frozen=True, slots=True)
class DevicePipelineItem:
    """One compact, independently executable image-pair descriptor."""

    item_id: str
    reference: NpyArrayDescriptor
    target: NpyArrayDescriptor
    candidates: tuple[DevicePipelineCandidate, ...]
    variance: NpyArrayDescriptor | None = None
    fit_mask: NpyArrayDescriptor | None = None
    schema: Literal["cuphoton.xscan.device-pipeline.item/v1"] = field(
        init=False,
        default=DEVICE_PIPELINE_ITEM_SCHEMA,
    )

    def __post_init__(self) -> None:
        validate_identifier(self.item_id, field="item_id")
        for name in ("reference", "target"):
            if not isinstance(getattr(self, name), NpyArrayDescriptor):
                raise TypeError(f"{name} must be an NpyArrayDescriptor")
        for name in ("variance", "fit_mask"):
            value = getattr(self, name)
            if value is not None and not isinstance(
                value, NpyArrayDescriptor
            ):
                raise TypeError(
                    f"{name} must be an NpyArrayDescriptor or None"
                )
        candidates = tuple(self.candidates)
        if not candidates:
            raise ValueError("device pipeline item must contain candidates")
        if not all(
            isinstance(value, DevicePipelineCandidate) for value in candidates
        ):
            raise TypeError(
                "candidates must be DevicePipelineCandidate values"
            )
        object.__setattr__(self, "candidates", candidates)
        typed_ids = [
            (type(value.candidate_id), value.candidate_id)
            for value in candidates
        ]
        if len(set(typed_ids)) != len(typed_ids):
            raise ValueError(
                "candidate_id values must be unique within an item"
            )
        source_indices = [value.source_index for value in candidates]
        if len(set(source_indices)) != len(source_indices):
            raise ValueError("candidate source_index values must be unique")

    def to_payload(self) -> dict[str, Any]:
        """Return the compact JSON-compatible work descriptor."""

        return {
            "schema": self.schema,
            "item_id": self.item_id,
            "reference": self.reference.to_payload(),
            "target": self.target.to_payload(),
            "variance": (
                None if self.variance is None else self.variance.to_payload()
            ),
            "fit_mask": (
                None if self.fit_mask is None else self.fit_mask.to_payload()
            ),
            "candidates": [value.to_payload() for value in self.candidates],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> DevicePipelineItem:
        """Restore one strict item from JSON-compatible values."""

        values = _require_exact_fields(
            payload,
            expected=frozenset(
                {
                    "schema",
                    "item_id",
                    "reference",
                    "target",
                    "variance",
                    "fit_mask",
                    "candidates",
                }
            ),
            field_name="device pipeline item",
        )
        if values["schema"] != DEVICE_PIPELINE_ITEM_SCHEMA:
            raise ValueError("unsupported device pipeline item schema")
        candidates = values["candidates"]
        if not isinstance(candidates, list):
            raise TypeError("device pipeline candidates must be a list")

        def optional_descriptor(value: Any) -> NpyArrayDescriptor | None:
            if value is None:
                return None
            if not isinstance(value, Mapping):
                raise TypeError("optional NPY descriptors must be mappings")
            return NpyArrayDescriptor.from_payload(value)

        return cls(
            item_id=values["item_id"],
            reference=NpyArrayDescriptor.from_payload(values["reference"]),
            target=NpyArrayDescriptor.from_payload(values["target"]),
            variance=optional_descriptor(values["variance"]),
            fit_mask=optional_descriptor(values["fit_mask"]),
            candidates=tuple(
                DevicePipelineCandidate.from_payload(value)
                for value in candidates
            ),
        )


@dataclass(frozen=True, slots=True)
class DeviceXPOISPipelineConfig:
    """Constant-kernel xPois settings owned by a device worker."""

    kernel_shape: tuple[int, int]
    basis_sigmas: tuple[float, ...]
    basis_degrees: tuple[int, ...]
    background_degree: int = 0
    flux_conserve: bool = False
    flux_reference_index: int = 0
    solver: Literal["constant"] = "constant"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "kernel_shape",
            _require_shape(
                self.kernel_shape,
                field_name="xpois kernel_shape",
                odd=True,
            ),
        )
        sigmas = tuple(self.basis_sigmas)
        degrees = tuple(self.basis_degrees)
        if not sigmas or len(sigmas) != len(degrees):
            raise ValueError(
                "xpois basis_sigmas and basis_degrees must be nonempty "
                "and aligned"
            )
        normalized_sigmas = tuple(
            _require_finite(
                value,
                field_name="xpois basis sigma",
                minimum=np.finfo(float).tiny,
            )
            for value in sigmas
        )
        for value in degrees:
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(
                    "xpois basis degrees must be non-negative integers"
                )
        if (
            isinstance(self.background_degree, bool)
            or not isinstance(self.background_degree, int)
            or self.background_degree < 0
        ):
            raise ValueError("xpois background_degree must be non-negative")
        if not isinstance(self.flux_conserve, bool):
            raise TypeError("xpois flux_conserve must be boolean")
        if (
            isinstance(self.flux_reference_index, bool)
            or not isinstance(self.flux_reference_index, int)
            or self.flux_reference_index < 0
        ):
            raise ValueError(
                "xpois flux_reference_index must be non-negative"
            )
        if self.solver != "constant":
            raise ValueError("xpois solver must be 'constant'")
        object.__setattr__(self, "basis_sigmas", normalized_sigmas)
        object.__setattr__(self, "basis_degrees", degrees)

    def to_payload(self) -> dict[str, Any]:
        """Return JSON-compatible xPois settings."""

        return {
            "kernel_shape": list(self.kernel_shape),
            "basis_sigmas": list(self.basis_sigmas),
            "basis_degrees": list(self.basis_degrees),
            "background_degree": self.background_degree,
            "flux_conserve": self.flux_conserve,
            "flux_reference_index": self.flux_reference_index,
            "solver": self.solver,
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any]
    ) -> DeviceXPOISPipelineConfig:
        """Restore strict xPois settings from JSON-compatible values."""

        values = _require_exact_fields(
            payload,
            expected=frozenset(
                {
                    "kernel_shape",
                    "basis_sigmas",
                    "basis_degrees",
                    "background_degree",
                    "flux_conserve",
                    "flux_reference_index",
                    "solver",
                }
            ),
            field_name="device pipeline xpois config",
        )
        return cls(
            kernel_shape=tuple(values["kernel_shape"]),
            basis_sigmas=tuple(values["basis_sigmas"]),
            basis_degrees=tuple(values["basis_degrees"]),
            background_degree=values["background_degree"],
            flux_conserve=values["flux_conserve"],
            flux_reference_index=values["flux_reference_index"],
            solver=values["solver"],
        )


@dataclass(frozen=True, slots=True)
class DeviceXFitPipelineConfig:
    """Levenberg-Marquardt controls for Gaussian difference fitting."""

    model: Literal["gaussian"] = "gaussian"
    mode: Literal["difference"] = "difference"
    f_tol: float | None = None
    x_tol: float | None = None
    g_tol: float | None = None
    max_evaluations: int | None = None
    initial_damping: float = 1.0e-3
    damping_increase: float = 10.0
    damping_decrease: float = 0.3
    finite_difference_step: float | None = None
    use_finite_difference: bool = False

    def __post_init__(self) -> None:
        if self.model != "gaussian":
            raise ValueError("xfit model must be 'gaussian'")
        if self.mode != "difference":
            raise ValueError("xfit mode must be 'difference'")
        for name in ("f_tol", "x_tol", "g_tol", "finite_difference_step"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    _require_finite(
                        value,
                        field_name=f"xfit {name}",
                        minimum=0.0,
                    ),
                )
        if self.max_evaluations is not None and (
            isinstance(self.max_evaluations, bool)
            or not isinstance(self.max_evaluations, int)
            or self.max_evaluations <= 0
        ):
            raise ValueError("xfit max_evaluations must be positive")
        for name in (
            "initial_damping",
            "damping_increase",
            "damping_decrease",
        ):
            object.__setattr__(
                self,
                name,
                _require_finite(
                    getattr(self, name),
                    field_name=f"xfit {name}",
                    minimum=np.finfo(float).tiny,
                ),
            )
        if self.damping_increase <= 1.0:
            raise ValueError("xfit damping_increase must be greater than one")
        if self.damping_decrease >= 1.0:
            raise ValueError("xfit damping_decrease must be less than one")
        if (
            self.finite_difference_step is not None
            and self.finite_difference_step <= 0.0
        ):
            raise ValueError(
                "xfit finite_difference_step must be greater than zero"
            )
        if not isinstance(self.use_finite_difference, bool):
            raise TypeError("xfit use_finite_difference must be boolean")

    def to_payload(self) -> dict[str, Any]:
        """Return JSON-compatible LM settings."""

        return asdict(self)

    def solver_payload(self) -> dict[str, Any]:
        """Return only fields accepted by :class:`LMConfig`."""

        payload = self.to_payload()
        del payload["model"]
        del payload["mode"]
        return payload

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any]
    ) -> DeviceXFitPipelineConfig:
        """Restore exact LM settings from JSON-compatible values."""

        expected = frozenset(value.name for value in fields(cls))
        values = _require_exact_fields(
            payload,
            expected=expected,
            field_name="device pipeline xfit config",
        )
        return cls(**values)


@dataclass(frozen=True, slots=True)
class DevicePipelineConfig:
    """Compact worker initialization descriptor for one CUDA device."""

    device: str
    checkpoint_dir: str
    checkpoint_sha256: str
    feature_schema_path: str
    feature_schema_sha256: str
    stamp_shape: tuple[int, int]
    decision_threshold: float
    xpois: DeviceXPOISPipelineConfig
    xfit: DeviceXFitPipelineConfig = field(
        default_factory=DeviceXFitPipelineConfig
    )
    schema: Literal["cuphoton.xscan.device-pipeline.config/v2"] = field(
        init=False,
        default=DEVICE_PIPELINE_CONFIG_SCHEMA,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.device, str) or not _CUDA_DEVICE.fullmatch(
            self.device
        ):
            raise ValueError(
                "device must be an explicit CUDA device like cuda:0"
            )
        if not isinstance(self.checkpoint_dir, str):
            raise TypeError("checkpoint_dir must be a string")
        checkpoint_dir = Path(self.checkpoint_dir)
        if not checkpoint_dir.is_absolute():
            raise ValueError("checkpoint_dir must be absolute")
        if not isinstance(self.feature_schema_path, str):
            raise TypeError("feature_schema_path must be a string")
        feature_schema_path = Path(self.feature_schema_path)
        if not feature_schema_path.is_absolute():
            raise ValueError("feature_schema_path must be absolute")
        _require_sha256(
            self.checkpoint_sha256,
            field_name="checkpoint_sha256",
        )
        _require_sha256(
            self.feature_schema_sha256,
            field_name="feature_schema_sha256",
        )
        stamp_shape = _require_shape(
            self.stamp_shape,
            field_name="stamp_shape",
            odd=True,
        )
        if stamp_shape[0] != stamp_shape[1]:
            raise ValueError("stamp_shape must be square for the xScan model")
        threshold = _require_finite(
            self.decision_threshold,
            field_name="decision_threshold",
            minimum=0.0,
        )
        if threshold > 1.0:
            raise ValueError("decision_threshold must be at most 1")
        if not isinstance(self.xpois, DeviceXPOISPipelineConfig):
            raise TypeError("xpois must be a DeviceXPOISPipelineConfig")
        if not isinstance(self.xfit, DeviceXFitPipelineConfig):
            raise TypeError("xfit must be a DeviceXFitPipelineConfig")
        object.__setattr__(self, "stamp_shape", stamp_shape)
        object.__setattr__(self, "decision_threshold", threshold)

    @property
    def device_id(self) -> int:
        """Return the explicit local CUDA ordinal."""

        return int(self.device.removeprefix("cuda:"))

    @property
    def configuration_sha256(self) -> str:
        """Return the stable hash of this complete worker descriptor."""

        return _stable_payload_sha256(self.to_payload())

    @property
    def inference_policy(self) -> dict[str, Any]:
        """Return the fixed deterministic inference policy."""

        return _strict_inference_policy_payload()

    def to_payload(self) -> dict[str, Any]:
        """Return the compact JSON-compatible worker descriptor."""

        return {
            "schema": self.schema,
            "device": self.device,
            "checkpoint_dir": self.checkpoint_dir,
            "checkpoint_sha256": self.checkpoint_sha256,
            "feature_schema_path": self.feature_schema_path,
            "feature_schema_sha256": self.feature_schema_sha256,
            "stamp_shape": list(self.stamp_shape),
            "decision_threshold": self.decision_threshold,
            "xpois": self.xpois.to_payload(),
            "xfit": self.xfit.to_payload(),
            "inference_policy": self.inference_policy,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> DevicePipelineConfig:
        """Restore one worker descriptor from JSON-compatible values."""

        values = _require_exact_fields(
            payload,
            expected=frozenset(
                {
                    "schema",
                    "device",
                    "checkpoint_dir",
                    "checkpoint_sha256",
                    "feature_schema_path",
                    "feature_schema_sha256",
                    "stamp_shape",
                    "decision_threshold",
                    "xpois",
                    "xfit",
                    "inference_policy",
                }
            ),
            field_name="device pipeline config",
        )
        if values["schema"] != DEVICE_PIPELINE_CONFIG_SCHEMA:
            raise ValueError("unsupported device pipeline config schema")
        inference_policy = json_mapping(
            values["inference_policy"],
            field="device pipeline inference_policy",
        )
        if inference_policy != _strict_inference_policy_payload():
            raise ValueError(
                "device pipeline inference_policy must match the strict "
                "deterministic policy"
            )
        return cls(
            device=values["device"],
            checkpoint_dir=values["checkpoint_dir"],
            checkpoint_sha256=values["checkpoint_sha256"],
            feature_schema_path=values["feature_schema_path"],
            feature_schema_sha256=values["feature_schema_sha256"],
            stamp_shape=tuple(values["stamp_shape"]),
            decision_threshold=values["decision_threshold"],
            xpois=DeviceXPOISPipelineConfig.from_payload(values["xpois"]),
            xfit=DeviceXFitPipelineConfig.from_payload(values["xfit"]),
        )


@dataclass(frozen=True, slots=True)
class DevicePipelinePrediction:
    """One compact terminal xScan prediction in source order."""

    candidate_id: int | str
    center_x: int
    center_y: int
    source_index: int
    logit: float
    probability: float
    decision: bool

    def to_payload(self) -> dict[str, Any]:
        """Return this prediction as JSON-compatible scalar values."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class DevicePipelineEvidenceSegment:
    """One named slice of the canonical packed evidence vector."""

    name: str
    offset: int
    shape: tuple[int, ...]
    source_dtype: str

    def to_payload(self) -> dict[str, Any]:
        """Return the strict layout descriptor."""

        return {
            "name": self.name,
            "offset": self.offset,
            "shape": list(self.shape),
            "source_dtype": self.source_dtype,
        }


@dataclass(frozen=True, slots=True)
class DevicePipelineScientificEvidence:
    """One-copy numerical evidence for released-versus-candidate parity."""

    candidate_count: int
    parameter_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    layout: tuple[DevicePipelineEvidenceSegment, ...]
    packed_element_count: int
    packed_byte_count: int
    packed_sha256: str
    packed_base64: str
    xfit_schema: str
    xfit_backend: str
    xfit_solver: str
    xfit_model: str
    xfit_mode: str
    xfit_dtype: str
    xfit_status_code_names: tuple[tuple[int, str], ...]
    xfit_uncertainty_reason_code_names: tuple[tuple[int, str], ...]
    xpois_schema: str
    xpois_backend: str
    xpois_solver: str
    xpois_kernel_shape: tuple[int, int]
    xpois_flux_conserve: bool
    xpois_basis_terms: tuple[tuple[int, float, int, int, bool], ...]
    schema: Literal[
        "cuphoton.xscan.device-pipeline.scientific-evidence/v1"
    ] = field(init=False, default=DEVICE_PIPELINE_EVIDENCE_SCHEMA)
    encoding: Literal["base64"] = field(init=False, default="base64")
    packed_dtype: Literal["<f8"] = field(init=False, default="<f8")

    def to_payload(self) -> dict[str, Any]:
        """Return strict JSON containing canonical little-endian bytes."""

        return {
            "schema": self.schema,
            "encoding": self.encoding,
            "packed_dtype": self.packed_dtype,
            "candidate_count": self.candidate_count,
            "parameter_names": list(self.parameter_names),
            "feature_names": list(self.feature_names),
            "layout": [segment.to_payload() for segment in self.layout],
            "packed_element_count": self.packed_element_count,
            "packed_byte_count": self.packed_byte_count,
            "packed_sha256": self.packed_sha256,
            "packed_base64": self.packed_base64,
            "xfit": {
                "schema": self.xfit_schema,
                "backend": self.xfit_backend,
                "solver": self.xfit_solver,
                "model": self.xfit_model,
                "mode": self.xfit_mode,
                "dtype": self.xfit_dtype,
                "status_code_names": {
                    str(code): name
                    for code, name in self.xfit_status_code_names
                },
                "uncertainty_reason_code_names": {
                    str(code): name
                    for code, name in (
                        self.xfit_uncertainty_reason_code_names
                    )
                },
            },
            "xpois": {
                "schema": self.xpois_schema,
                "backend": self.xpois_backend,
                "solver": self.xpois_solver,
                "kernel_shape": list(self.xpois_kernel_shape),
                "flux_conserve": self.xpois_flux_conserve,
                "basis_terms": [
                    {
                        "component_index": component_index,
                        "sigma": sigma,
                        "poly_u_degree": poly_u_degree,
                        "poly_v_degree": poly_v_degree,
                        "zero_sum": zero_sum,
                    }
                    for (
                        component_index,
                        sigma,
                        poly_u_degree,
                        poly_v_degree,
                        zero_sum,
                    ) in self.xpois_basis_terms
                ],
            },
        }


def _device_pipeline_evidence_layout_contract(
    values: Mapping[str, Any],
) -> tuple[tuple[str, tuple[int, ...], str], ...]:
    """Validate v1 evidence metadata and return its exact packed layout."""

    from cuphoton.xfit.api import DipoleFitUncertaintyReason
    from cuphoton.xfit.solver import LMStatus

    from .xfit_features import FEATURE_NAMES

    parameter_names = values["parameter_names"]
    if (
        not isinstance(parameter_names, list)
        or tuple(parameter_names) != _GAUSSIAN_PARAMETER_NAMES
    ):
        raise ValueError(
            "scientific evidence parameter_names must be canonical and "
            "ordered"
        )
    feature_names = values["feature_names"]
    if (
        not isinstance(feature_names, list)
        or tuple(feature_names) != FEATURE_NAMES
    ):
        raise ValueError(
            "scientific evidence feature_names must be canonical and ordered"
        )

    xfit = _require_exact_fields(
        values["xfit"],
        expected=frozenset(
            {
                "schema",
                "backend",
                "solver",
                "model",
                "mode",
                "dtype",
                "status_code_names",
                "uncertainty_reason_code_names",
            }
        ),
        field_name="scientific evidence xfit",
    )
    expected_xfit = {
        "schema": "cuphoton.xfit.device-fit-result/v1",
        "backend": "cupy",
        "solver": "levenberg-marquardt",
        "model": "gaussian",
        "mode": "difference",
        "dtype": "float64",
    }
    for name, expected in expected_xfit.items():
        if xfit[name] != expected:
            raise ValueError(
                f"scientific evidence xfit.{name} must be {expected!r}"
            )
    expected_status_names = {
        str(int(status)): status.name for status in LMStatus
    }
    status_names = json_mapping(
        xfit["status_code_names"],
        field="scientific evidence xfit.status_code_names",
    )
    if status_names != expected_status_names:
        raise ValueError("scientific evidence xfit.status_code_names changed")
    expected_reason_names = {
        str(int(reason)): reason.name for reason in DipoleFitUncertaintyReason
    }
    reason_names = json_mapping(
        xfit["uncertainty_reason_code_names"],
        field=("scientific evidence xfit.uncertainty_reason_code_names"),
    )
    if reason_names != expected_reason_names:
        raise ValueError(
            "scientific evidence xfit uncertainty reason codes changed"
        )

    xpois = _require_exact_fields(
        values["xpois"],
        expected=frozenset(
            {
                "schema",
                "backend",
                "solver",
                "kernel_shape",
                "flux_conserve",
                "basis_terms",
            }
        ),
        field_name="scientific evidence xpois",
    )
    expected_xpois = {
        "schema": "cuphoton.xpois.device-fit-result/v1",
        "backend": "cupy",
        "solver": "constant",
    }
    for name, expected in expected_xpois.items():
        if xpois[name] != expected:
            raise ValueError(
                f"scientific evidence xpois.{name} must be {expected!r}"
            )
    kernel_shape = xpois["kernel_shape"]
    if not isinstance(kernel_shape, list):
        raise TypeError(
            "scientific evidence xpois.kernel_shape must be a list"
        )
    _require_shape(
        tuple(kernel_shape),
        field_name="scientific evidence xpois.kernel_shape",
        odd=True,
    )
    if not isinstance(xpois["flux_conserve"], bool):
        raise TypeError(
            "scientific evidence xpois.flux_conserve must be boolean"
        )
    basis_terms = xpois["basis_terms"]
    if not isinstance(basis_terms, list) or not basis_terms:
        raise ValueError(
            "scientific evidence xpois.basis_terms must be a nonempty list"
        )
    basis_identities: list[tuple[int, float, int, int]] = []
    zero_sum_values: list[bool] = []
    for index, raw_term in enumerate(basis_terms):
        term = _require_exact_fields(
            raw_term,
            expected=frozenset(
                {
                    "component_index",
                    "sigma",
                    "poly_u_degree",
                    "poly_v_degree",
                    "zero_sum",
                }
            ),
            field_name=f"scientific evidence xpois.basis_terms[{index}]",
        )
        component_index = term["component_index"]
        if (
            isinstance(component_index, bool)
            or not isinstance(component_index, int)
            or component_index < 0
        ):
            raise ValueError(
                "scientific evidence basis component_index must be "
                "non-negative"
            )
        degrees: list[int] = []
        for name in ("poly_u_degree", "poly_v_degree"):
            degree = term[name]
            if (
                isinstance(degree, bool)
                or not isinstance(degree, int)
                or degree < 0
            ):
                raise ValueError(
                    f"scientific evidence basis {name} must be non-negative"
                )
            degrees.append(degree)
        sigma = _require_finite(
            term["sigma"],
            field_name="scientific evidence basis sigma",
            minimum=np.finfo(float).tiny,
        )
        if not isinstance(term["zero_sum"], bool):
            raise TypeError(
                "scientific evidence basis zero_sum must be boolean"
            )
        basis_identities.append(
            (component_index, sigma, degrees[0], degrees[1])
        )
        zero_sum_values.append(term["zero_sum"])
    if len(set(basis_identities)) != len(basis_identities):
        raise ValueError("scientific evidence xPois basis terms repeat")
    if xpois["flux_conserve"]:
        expected_zero_sum = [False, *([True] * (len(basis_terms) - 1))]
    else:
        from cuphoton.xpois import (
            GaussianBasisComponent,
            build_gaussian_polynomial_basis,
        )

        expected_zero_sum = []
        for _, sigma, degree_u, degree_v in basis_identities:
            _, terms = build_gaussian_polynomial_basis(
                tuple(kernel_shape),
                [GaussianBasisComponent(sigma, degree_u + degree_v)],
            )
            expected_zero_sum.append(
                next(
                    term.zero_sum
                    for term in terms
                    if (term.poly_u_degree, term.poly_v_degree)
                    == (degree_u, degree_v)
                )
            )
    if zero_sum_values != expected_zero_sum:
        raise ValueError(
            "scientific evidence basis zero_sum flags do not match "
            "flux_conserve"
        )

    layout = values["layout"]
    expected_segment_count = 2 + len(_XFIT_SCALAR_EVIDENCE_FIELDS) + 6
    if not isinstance(layout, list) or len(layout) != expected_segment_count:
        raise ValueError(
            "scientific evidence layout must contain the exact v1 segments"
        )
    background_segment = _require_exact_fields(
        layout[1],
        expected=frozenset({"name", "offset", "shape", "source_dtype"}),
        field_name="scientific evidence layout[1]",
    )
    background_shape = background_segment["shape"]
    if (
        background_segment["name"] != "xpois.background_coefficients"
        or not isinstance(background_shape, list)
        or len(background_shape) != 1
        or isinstance(background_shape[0], bool)
        or not isinstance(background_shape[0], int)
        or background_shape[0] < 1
    ):
        raise ValueError(
            "scientific evidence xPois background coefficient shape is "
            "invalid"
        )
    background_count = background_shape[0]
    discriminant_root = math.isqrt(8 * background_count + 1)
    if (
        discriminant_root * discriminant_root != 8 * background_count + 1
        or (discriminant_root - 3) % 2 != 0
    ):
        raise ValueError(
            "scientific evidence xPois background coefficient count is not "
            "triangular"
        )

    candidate_count = values["candidate_count"]
    parameter_count = len(_GAUSSIAN_PARAMETER_NAMES)
    feature_count = len(FEATURE_NAMES)
    scalar_dtypes = {
        "status_codes": np.dtype(np.int8).str,
        "converged": np.dtype(bool).str,
        "evaluations": np.dtype(np.int64).str,
        "residual_norm": np.dtype(np.float64).str,
        "chi_square": np.dtype(np.float64).str,
        "valid_pixel_count": np.dtype(np.int64).str,
        "valid_pixel_fraction": np.dtype(np.float64).str,
        "null_chi_square": np.dtype(np.float64).str,
        "delta_chi_square": np.dtype(np.float64).str,
        "fractional_null_improvement": np.dtype(np.float64).str,
        "degrees_of_freedom": np.dtype(np.int64).str,
        "reduced_chi_square": np.dtype(np.float64).str,
        "uncertainty_valid": np.dtype(bool).str,
        "uncertainty_reason_codes": np.dtype(np.int8).str,
    }
    contract: list[tuple[str, tuple[int, ...], str]] = [
        (
            "xpois.kernel_coefficients",
            (len(basis_terms),),
            np.dtype(np.float64).str,
        ),
        (
            "xpois.background_coefficients",
            (background_count,),
            np.dtype(np.float64).str,
        ),
    ]
    contract.extend(
        (
            f"xfit.{name}",
            (candidate_count,),
            scalar_dtypes[name],
        )
        for name in _XFIT_SCALAR_EVIDENCE_FIELDS
    )
    contract.extend(
        (
            name,
            shape,
            dtype,
        )
        for name, shape, dtype in (
            (
                "xfit.parameters",
                (candidate_count, parameter_count),
                np.dtype(np.float64).str,
            ),
            (
                "xfit.standard_errors",
                (candidate_count, parameter_count),
                np.dtype(np.float64).str,
            ),
            (
                "xfit.covariance",
                (candidate_count, parameter_count, parameter_count),
                np.dtype(np.float64).str,
            ),
            (
                "xfit.features",
                (candidate_count, feature_count),
                np.dtype(np.float32).str,
            ),
            (
                "xscan.logits",
                (candidate_count,),
                np.dtype(np.float32).str,
            ),
            (
                "xscan.probabilities",
                (candidate_count,),
                np.dtype(np.float32).str,
            ),
        )
    )
    return tuple(contract)


def _validate_device_pipeline_evidence_values(
    decoded: Mapping[str, np.ndarray],
) -> None:
    """Validate the host-side numerical semantics of v1 evidence."""

    from cuphoton.xfit.api import DipoleFitUncertaintyReason
    from cuphoton.xfit.solver import LMStatus

    from .xfit_features import (
        FEATURE_NAMES,
        LOG_DELTA_CHI_SQUARE_SCALE,
        LOG_REDUCED_CHI_SQUARE_SCALE,
        _validate_feature_values,
    )

    def require_finite(name: str) -> np.ndarray:
        result = decoded[name]
        if not np.isfinite(result).all():
            raise ValueError(
                f"scientific evidence {name} contains non-finite values"
            )
        return result

    def require_integral(name: str, dtype: np.dtype[Any]) -> np.ndarray:
        result = require_finite(name)
        limits = np.iinfo(dtype)
        if (
            np.any(result != np.trunc(result))
            or np.any(result < limits.min)
            or np.any(result > limits.max)
        ):
            raise ValueError(
                f"scientific evidence {name} is not valid {dtype.name} data"
            )
        return result

    def require_boolean(name: str) -> np.ndarray:
        result = require_finite(name)
        if not np.all((result == 0.0) | (result == 1.0)):
            raise ValueError(
                f"scientific evidence {name} is not valid boolean data"
            )
        return result.astype(bool)

    for name in (
        "xpois.kernel_coefficients",
        "xpois.background_coefficients",
        "xfit.parameters",
        "xfit.residual_norm",
        "xfit.chi_square",
        "xfit.valid_pixel_fraction",
        "xfit.null_chi_square",
        "xfit.delta_chi_square",
        "xscan.logits",
        "xscan.probabilities",
    ):
        require_finite(name)
    status_codes = require_integral(
        "xfit.status_codes", np.dtype(np.int8)
    ).astype(np.int8)
    evaluations = require_integral("xfit.evaluations", np.dtype(np.int64))
    valid_pixel_count = require_integral(
        "xfit.valid_pixel_count", np.dtype(np.int64)
    )
    degrees_of_freedom = require_integral(
        "xfit.degrees_of_freedom", np.dtype(np.int64)
    )
    reason_codes = require_integral(
        "xfit.uncertainty_reason_codes", np.dtype(np.int8)
    ).astype(np.int8)
    converged = require_boolean("xfit.converged")
    uncertainty_valid = require_boolean("xfit.uncertainty_valid")

    allowed_statuses = {int(status) for status in LMStatus}
    if any(int(code) not in allowed_statuses for code in status_codes):
        raise ValueError(
            "scientific evidence contains an unknown xFit status"
        )
    if np.any(status_codes == int(LMStatus.ACTIVE)):
        raise ValueError("scientific evidence contains an active xFit status")
    converged_statuses = np.isin(
        status_codes,
        [
            int(LMStatus.CONVERGED_F_TOL),
            int(LMStatus.CONVERGED_X_TOL),
            int(LMStatus.CONVERGED_G_TOL),
        ],
    )
    if not np.array_equal(converged, converged_statuses):
        raise ValueError(
            "scientific evidence xFit converged flags do not match statuses"
        )
    if np.any(evaluations < 1):
        raise ValueError(
            "scientific evidence xFit evaluations must be positive"
        )
    if np.any(valid_pixel_count < 1):
        raise ValueError(
            "scientific evidence xFit valid_pixel_count must be positive"
        )
    valid_fraction = decoded["xfit.valid_pixel_fraction"]
    if np.any((valid_fraction <= 0.0) | (valid_fraction > 1.0)):
        raise ValueError(
            "scientific evidence xFit valid_pixel_fraction is invalid"
        )
    parameter_count = decoded["xfit.parameters"].shape[1]
    if not np.array_equal(
        degrees_of_freedom,
        valid_pixel_count - parameter_count,
    ):
        raise ValueError(
            "scientific evidence xFit degrees_of_freedom is inconsistent"
        )

    parameters = decoded["xfit.parameters"]
    if np.any(parameters[:, 1:3] <= 0.0):
        raise ValueError(
            "scientific evidence Gaussian widths must be positive"
        )
    theta = parameters[:, 3]
    if np.any((theta < -np.pi / 2.0) | (theta >= np.pi / 2.0)):
        raise ValueError(
            "scientific evidence Gaussian theta is outside its canonical "
            "interval"
        )
    residual_norm = decoded["xfit.residual_norm"]
    chi_square = decoded["xfit.chi_square"]
    null_chi_square = decoded["xfit.null_chi_square"]
    delta_chi_square = decoded["xfit.delta_chi_square"]
    if (
        np.any(residual_norm < 0.0)
        or np.any(chi_square < 0.0)
        or np.any(null_chi_square < 0.0)
    ):
        raise ValueError(
            "scientific evidence xFit norms and chi-square must be "
            "non-negative"
        )
    if not np.allclose(
        residual_norm * residual_norm,
        chi_square,
        rtol=2.0e-12,
        atol=2.0e-12,
    ):
        raise ValueError(
            "scientific evidence xFit residual_norm is inconsistent"
        )
    expected_delta = null_chi_square - chi_square
    if not np.allclose(
        delta_chi_square,
        expected_delta,
        rtol=2.0e-12,
        atol=2.0e-12,
    ):
        raise ValueError(
            "scientific evidence xFit delta_chi_square is inconsistent"
        )

    fractional_improvement = decoded["xfit.fractional_null_improvement"]
    positive_null = null_chi_square > 0.0
    if (
        not np.isfinite(fractional_improvement[positive_null]).all()
        or not np.isnan(fractional_improvement[~positive_null]).all()
    ):
        raise ValueError(
            "scientific evidence xFit fractional_null_improvement has "
            "invalid non-finite values"
        )
    if not np.allclose(
        fractional_improvement[positive_null],
        1.0 - chi_square[positive_null] / null_chi_square[positive_null],
        rtol=2.0e-12,
        atol=2.0e-12,
    ):
        raise ValueError(
            "scientific evidence xFit fractional_null_improvement is "
            "inconsistent"
        )
    reduced_chi_square = decoded["xfit.reduced_chi_square"]
    positive_dof = degrees_of_freedom > 0.0
    if (
        not np.isfinite(reduced_chi_square[positive_dof]).all()
        or not np.isnan(reduced_chi_square[~positive_dof]).all()
    ):
        raise ValueError(
            "scientific evidence xFit reduced_chi_square has invalid "
            "non-finite values"
        )
    if not np.allclose(
        reduced_chi_square[positive_dof],
        chi_square[positive_dof] / degrees_of_freedom[positive_dof],
        rtol=2.0e-12,
        atol=2.0e-12,
    ):
        raise ValueError(
            "scientific evidence xFit reduced_chi_square is inconsistent"
        )

    allowed_reasons = {int(reason) for reason in DipoleFitUncertaintyReason}
    if any(int(code) not in allowed_reasons for code in reason_codes):
        raise ValueError(
            "scientific evidence contains an unknown uncertainty reason"
        )
    expected_uncertainty_valid = reason_codes == int(
        DipoleFitUncertaintyReason.VALID
    )
    if not np.array_equal(uncertainty_valid, expected_uncertainty_valid):
        raise ValueError(
            "scientific evidence uncertainty flags do not match reasons"
        )
    if np.any(
        (~converged)
        & (reason_codes != int(DipoleFitUncertaintyReason.FIT_NOT_CONVERGED))
    ):
        raise ValueError(
            "scientific evidence non-converged fits have invalid uncertainty "
            "reasons"
        )
    if np.any(
        converged
        & (degrees_of_freedom <= 0)
        & (
            reason_codes
            != int(DipoleFitUncertaintyReason.INSUFFICIENT_DEGREES_OF_FREEDOM)
        )
    ):
        raise ValueError(
            "scientific evidence non-positive degrees of freedom have an "
            "invalid uncertainty reason"
        )
    covariance = decoded["xfit.covariance"]
    standard_errors = decoded["xfit.standard_errors"]
    valid_covariance = covariance[uncertainty_valid]
    valid_errors = standard_errors[uncertainty_valid]
    if (
        not np.isfinite(valid_covariance).all()
        or not np.isfinite(valid_errors).all()
        or np.any(valid_errors < 0.0)
    ):
        raise ValueError(
            "scientific evidence valid uncertainties must be finite and "
            "non-negative"
        )
    if valid_covariance.size:
        diagonal = np.diagonal(valid_covariance, axis1=1, axis2=2)
        if np.any(diagonal < 0.0) or not np.allclose(
            valid_errors,
            np.sqrt(diagonal),
            rtol=2.0e-8,
            atol=2.0e-8,
        ):
            raise ValueError(
                "scientific evidence standard errors do not match covariance"
            )
        if not np.allclose(
            valid_covariance,
            np.swapaxes(valid_covariance, 1, 2),
            rtol=2.0e-8,
            atol=2.0e-8,
        ):
            raise ValueError(
                "scientific evidence covariance must be symmetric"
            )
    if (
        not np.isnan(covariance[~uncertainty_valid]).all()
        or not np.isnan(standard_errors[~uncertainty_valid]).all()
    ):
        raise ValueError(
            "scientific evidence unavailable uncertainties must be all-NaN"
        )

    features = require_finite("xfit.features")
    logits = require_finite("xscan.logits")
    probabilities = require_finite("xscan.probabilities")
    for name, array in (
        ("xfit.features", features),
        ("xscan.logits", logits),
        ("xscan.probabilities", probabilities),
    ):
        roundtrip = array.astype(np.float32).astype(np.float64)
        if not np.array_equal(array, roundtrip):
            raise ValueError(
                f"scientific evidence {name} is not exact float32 data"
            )
    feature_values = features.astype(np.float32)
    feature_index = {name: index for index, name in enumerate(FEATURE_NAMES)}
    if not np.all(feature_values[:, feature_index["fit_present"]] == 1.0):
        raise ValueError(
            "scientific evidence live xFit features must all be present"
        )
    variance_values = feature_values[:, feature_index["variance_weighted"]]
    if np.all(variance_values == 0.0):
        variance_present = False
    elif np.all(variance_values == 1.0):
        variance_present = True
    else:
        raise ValueError(
            "scientific evidence variance_weighted must be item-consistent"
        )
    _validate_feature_values(
        feature_values,
        variance_present=variance_present,
        model="gaussian",
    )
    feature_fit_valid = feature_values[:, feature_index["fit_valid"]] == 1.0
    if np.any(feature_fit_valid & (~converged | (degrees_of_freedom <= 0))):
        raise ValueError(
            "scientific evidence feature fit_valid requires a converged fit "
            "with positive degrees of freedom"
        )
    expected_feature_uncertainty = feature_fit_valid & uncertainty_valid
    if not np.array_equal(
        feature_values[:, feature_index["uncertainty_valid"]] == 1.0,
        expected_feature_uncertainty,
    ):
        raise ValueError(
            "scientific evidence feature uncertainty does not match xFit"
        )
    expected_fraction = valid_fraction.astype(np.float32)
    if not np.array_equal(
        feature_values[:, feature_index["valid_pixel_fraction"]],
        expected_fraction,
    ):
        raise ValueError(
            "scientific evidence valid-pixel feature does not match xFit"
        )
    expected_improvement = np.where(
        np.isfinite(fractional_improvement),
        np.clip(fractional_improvement, -1.0, 1.0),
        0.0,
    ).astype(np.float32)
    if not np.array_equal(
        feature_values[:, feature_index["fractional_null_improvement"]],
        expected_improvement,
    ):
        raise ValueError(
            "scientific evidence improvement feature does not match xFit"
        )
    if variance_present:
        expected_log_delta = np.clip(
            np.sign(delta_chi_square)
            * np.log1p(np.abs(delta_chi_square))
            / LOG_DELTA_CHI_SQUARE_SCALE,
            -1.0,
            1.0,
        ).astype(np.float32)
        valid_reduced = np.isfinite(reduced_chi_square) & (
            reduced_chi_square > 0.0
        )
        expected_log_reduced = np.zeros_like(
            reduced_chi_square, dtype=np.float32
        )
        expected_log_reduced[valid_reduced] = np.clip(
            np.log(reduced_chi_square[valid_reduced])
            / LOG_REDUCED_CHI_SQUARE_SCALE,
            -1.0,
            1.0,
        ).astype(np.float32)
        if not np.allclose(
            feature_values[:, feature_index["log_delta_chi_square"]],
            expected_log_delta,
            rtol=1.0e-6,
            atol=1.0e-7,
        ) or not np.allclose(
            feature_values[:, feature_index["log_reduced_chi_square"]],
            expected_log_reduced,
            rtol=1.0e-6,
            atol=1.0e-7,
        ):
            raise ValueError(
                "scientific evidence variance-weighted features do not "
                "match xFit"
            )
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError(
            "scientific evidence xScan probabilities are outside [0, 1]"
        )
    expected_probabilities = (
        1.0 / (1.0 + np.exp(-logits.astype(np.float32)))
    ).astype(np.float32)
    if not np.allclose(
        probabilities,
        expected_probabilities.astype(np.float64),
        rtol=1.0e-6,
        atol=1.0e-7,
    ):
        raise ValueError(
            "scientific evidence xScan probabilities do not match logits"
        )


def _canonical_xpois_basis_terms(
    config: DevicePipelineConfig,
) -> tuple[tuple[int, float, int, int, bool], ...]:
    from cuphoton.xpois import (
        GaussianBasisComponent,
        build_gaussian_polynomial_basis,
    )

    _, terms = build_gaussian_polynomial_basis(
        config.xpois.kernel_shape,
        [
            GaussianBasisComponent(sigma=sigma, degree=degree)
            for sigma, degree in zip(
                config.xpois.basis_sigmas,
                config.xpois.basis_degrees,
                strict=True,
            )
        ],
        flux_conserve=config.xpois.flux_conserve,
        flux_reference_index=config.xpois.flux_reference_index,
    )
    return tuple(
        (
            term.component_index,
            term.sigma,
            term.poly_u_degree,
            term.poly_v_degree,
            term.zero_sum,
        )
        for term in terms
    )


def _validate_device_pipeline_evidence_config(
    values: Mapping[str, Any],
    decoded: Mapping[str, np.ndarray],
    *,
    config: DevicePipelineConfig,
) -> None:
    """Bind valid generic evidence to one exact pipeline configuration."""

    from .xfit_features import FEATURE_NAMES, _canonical_feature_row

    if not isinstance(config, DevicePipelineConfig):
        raise TypeError("config must be a DevicePipelineConfig")
    xpois = json_mapping(values["xpois"], field="scientific evidence xpois")
    if tuple(xpois["kernel_shape"]) != config.xpois.kernel_shape:
        raise ValueError(
            "scientific evidence xPois kernel_shape does not match config"
        )
    if xpois["flux_conserve"] is not config.xpois.flux_conserve:
        raise ValueError(
            "scientific evidence xPois flux_conserve does not match config"
        )
    actual_basis_terms = tuple(
        (
            term["component_index"],
            float(term["sigma"]),
            term["poly_u_degree"],
            term["poly_v_degree"],
            term["zero_sum"],
        )
        for term in xpois["basis_terms"]
    )
    if actual_basis_terms != _canonical_xpois_basis_terms(config):
        raise ValueError(
            "scientific evidence xPois basis terms do not match config"
        )
    expected_background_count = (
        (config.xpois.background_degree + 1)
        * (config.xpois.background_degree + 2)
        // 2
    )
    if decoded["xpois.background_coefficients"].shape != (
        expected_background_count,
    ):
        raise ValueError(
            "scientific evidence xPois background coefficients do not match "
            "config"
        )

    features = decoded["xfit.features"].astype(np.float32)
    feature_index = {name: index for index, name in enumerate(FEATURE_NAMES)}
    if np.any(features[:, feature_index["variance_weighted"]] != 0.0):
        raise ValueError(
            "scientific evidence xFit features do not match the frozen "
            "unweighted schema"
        )
    parameters = decoded["xfit.parameters"]
    standard_errors = decoded["xfit.standard_errors"]
    covariance = decoded["xfit.covariance"]
    columns: dict[str, np.ndarray] = {
        "converged": decoded["xfit.converged"].astype(bool),
        "degrees_of_freedom": decoded["xfit.degrees_of_freedom"].astype(
            np.int64
        ),
        "valid_pixel_fraction": decoded["xfit.valid_pixel_fraction"],
        "fractional_null_improvement": decoded[
            "xfit.fractional_null_improvement"
        ],
        "delta_chi_square": decoded["xfit.delta_chi_square"],
        "reduced_chi_square": decoded["xfit.reduced_chi_square"],
        "uncertainty_valid": decoded["xfit.uncertainty_valid"].astype(bool),
    }
    for parameter_index, parameter_name in enumerate(
        _GAUSSIAN_PARAMETER_NAMES
    ):
        columns[parameter_name] = parameters[:, parameter_index]
        columns[f"{parameter_name}_standard_error"] = standard_errors[
            :, parameter_index
        ]
    expected_features = np.empty_like(features)
    for row_index in range(features.shape[0]):
        expected_features[row_index] = _canonical_feature_row(
            columns,
            row_index,
            covariance[row_index],
            model="gaussian",
            parameter_names=_GAUSSIAN_PARAMETER_NAMES,
            image_shape=config.stamp_shape,
            variance_present=False,
        )
    if not np.array_equal(features, expected_features):
        raise ValueError(
            "scientific evidence xFit features do not match the canonical "
            "config transform"
        )


def decode_device_pipeline_evidence(
    evidence: DevicePipelineScientificEvidence | Mapping[str, Any],
    *,
    config: DevicePipelineConfig | None = None,
) -> dict[str, np.ndarray]:
    """Validate and decode evidence, optionally binding it to a config."""

    payload = (
        evidence.to_payload()
        if isinstance(evidence, DevicePipelineScientificEvidence)
        else evidence
    )
    values = _require_exact_fields(
        payload,
        expected=frozenset(
            {
                "schema",
                "encoding",
                "packed_dtype",
                "candidate_count",
                "parameter_names",
                "feature_names",
                "layout",
                "packed_element_count",
                "packed_byte_count",
                "packed_sha256",
                "packed_base64",
                "xfit",
                "xpois",
            }
        ),
        field_name="device pipeline scientific evidence",
    )
    if values["schema"] != DEVICE_PIPELINE_EVIDENCE_SCHEMA:
        raise ValueError("unsupported device pipeline evidence schema")
    if values["encoding"] != "base64" or values["packed_dtype"] != "<f8":
        raise ValueError("unsupported device pipeline evidence encoding")
    for name in (
        "candidate_count",
        "packed_element_count",
        "packed_byte_count",
    ):
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"scientific evidence {name} must be positive")
    _require_sha256(
        values["packed_sha256"],
        field_name="scientific evidence packed_sha256",
    )
    if not isinstance(values["packed_base64"], str):
        raise TypeError("scientific evidence packed_base64 must be a string")
    try:
        raw = base64.b64decode(values["packed_base64"], validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError(
            "scientific evidence packed_base64 is invalid"
        ) from exc
    if len(raw) != values["packed_byte_count"]:
        raise ValueError("scientific evidence packed byte count changed")
    if len(raw) != values["packed_element_count"] * 8:
        raise ValueError("scientific evidence packed shape changed")
    if hashlib.sha256(raw).hexdigest() != values["packed_sha256"]:
        raise ValueError("scientific evidence packed SHA-256 changed")
    layout_contract = _device_pipeline_evidence_layout_contract(values)
    expected_element_count = sum(
        math.prod(shape) for _, shape, _ in layout_contract
    )
    if values["packed_element_count"] != expected_element_count:
        raise ValueError("scientific evidence packed element count changed")
    if values["packed_byte_count"] != expected_element_count * 8:
        raise ValueError("scientific evidence packed byte count changed")
    layout = values["layout"]
    packed = np.frombuffer(raw, dtype=np.dtype("<f8"))
    decoded: dict[str, np.ndarray] = {}
    next_offset = 0
    for index, (raw_segment, expected_segment) in enumerate(
        zip(layout, layout_contract, strict=True)
    ):
        segment = _require_exact_fields(
            raw_segment,
            expected=frozenset({"name", "offset", "shape", "source_dtype"}),
            field_name=f"scientific evidence layout[{index}]",
        )
        name, expected_shape, expected_dtype = expected_segment
        if segment["name"] != name:
            raise ValueError(
                "scientific evidence segment names/order changed at "
                f"layout[{index}]"
            )
        if segment["offset"] != next_offset or isinstance(
            segment["offset"], bool
        ):
            raise ValueError("scientific evidence layout must be contiguous")
        if segment["shape"] != list(expected_shape):
            raise ValueError(
                f"scientific evidence segment {name} shape changed"
            )
        if segment["source_dtype"] != expected_dtype:
            raise ValueError(
                f"scientific evidence segment {name} source dtype changed"
            )
        shape = segment["shape"]
        offset = next_offset
        size = math.prod(shape)
        next_offset += size
        if next_offset > packed.size:
            raise ValueError(
                "scientific evidence segment exceeds packed data"
            )
        decoded[name] = packed[offset:next_offset].reshape(shape).copy()
    if next_offset != packed.size:
        raise ValueError(
            "scientific evidence layout does not cover packed data"
        )
    _validate_device_pipeline_evidence_values(decoded)
    if config is not None:
        _validate_device_pipeline_evidence_config(
            values,
            decoded,
            config=config,
        )
    return decoded


@dataclass(frozen=True, slots=True)
class DevicePipelineTransferReceipt:
    """Exact pipeline-owned transfer accounting for one item."""

    input_h2d_bytes: int
    terminal_d2h_bytes: int
    dlpack_shared_bytes: int
    pipeline_full_array_d2h_bytes: Literal[0] = 0
    pipeline_compact_h2d_bytes: Literal[0] = 0
    terminal_d2h_calls: Literal[1] = 1
    accounting_scope: Literal["pipeline-owned"] = "pipeline-owned"
    device_stage_internal_transfers: Literal["not-instrumented"] = (
        "not-instrumented"
    )

    def to_payload(self) -> dict[str, Any]:
        """Return deterministic transfer counters and their scope."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class DevicePipelineTimingReceipt:
    """Host-elapsed stage timings without device-wide synchronization."""

    stages_seconds: tuple[tuple[str, float], ...]
    total_seconds: float
    semantics: Literal["host-elapsed; terminal-d2h-is-blocking"] = (
        "host-elapsed; terminal-d2h-is-blocking"
    )

    def to_payload(self) -> dict[str, Any]:
        """Return fixed-name timing values in execution order."""

        return {
            "stages_seconds": dict(self.stages_seconds),
            "total_seconds": self.total_seconds,
            "semantics": self.semantics,
        }


@dataclass(frozen=True, slots=True)
class DevicePipelineResult:
    """Compact, JSON-compatible successful result of one device item."""

    item_id: str
    device: str
    checkpoint_sha256: str
    feature_schema_sha256: str
    configuration_sha256: str
    input_sha256: tuple[tuple[str, str], ...]
    xpois_chi2: float
    xpois_dof: int
    xpois_fit_pixel_count: int
    predictions: tuple[DevicePipelinePrediction, ...]
    scientific_evidence: DevicePipelineScientificEvidence
    transfers: DevicePipelineTransferReceipt
    timings: DevicePipelineTimingReceipt
    result_sha256: str
    schema: Literal["cuphoton.xscan.device-pipeline.result/v2"] = field(
        init=False,
        default=DEVICE_PIPELINE_RESULT_SCHEMA,
    )

    def to_payload(self) -> dict[str, Any]:
        """Return a result containing no device or array-valued objects."""

        return {
            "schema": self.schema,
            "item_id": self.item_id,
            "device": self.device,
            "checkpoint_sha256": self.checkpoint_sha256,
            "feature_schema_sha256": self.feature_schema_sha256,
            "configuration_sha256": self.configuration_sha256,
            "input_sha256": dict(self.input_sha256),
            "xpois": {
                "chi2": self.xpois_chi2,
                "dof": self.xpois_dof,
                "fit_pixel_count": self.xpois_fit_pixel_count,
            },
            "predictions": [value.to_payload() for value in self.predictions],
            "scientific_evidence": self.scientific_evidence.to_payload(),
            "transfers": self.transfers.to_payload(),
            "timings": self.timings.to_payload(),
            "result_sha256": self.result_sha256,
        }


_GAUSSIAN_PARAMETER_NAMES = (
    "amplitude",
    "sigma_x",
    "sigma_y",
    "theta",
    "x_pos",
    "y_pos",
    "x_neg",
    "y_neg",
)
_XFIT_SCALAR_EVIDENCE_FIELDS = (
    "status_codes",
    "converged",
    "evaluations",
    "residual_norm",
    "chi_square",
    "valid_pixel_count",
    "valid_pixel_fraction",
    "null_chi_square",
    "delta_chi_square",
    "fractional_null_improvement",
    "degrees_of_freedom",
    "reduced_chi_square",
    "uncertainty_valid",
    "uncertainty_reason_codes",
)


@dataclass(frozen=True, slots=True)
class _PackedScientificEvidence:
    values: Any
    layout: tuple[DevicePipelineEvidenceSegment, ...]
    candidate_count: int
    parameter_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    xfit_schema: str
    xfit_backend: str
    xfit_solver: str
    xfit_model: str
    xfit_mode: str
    xfit_dtype: str
    xfit_status_code_names: tuple[tuple[int, str], ...]
    xfit_uncertainty_reason_code_names: tuple[tuple[int, str], ...]
    xpois_schema: str
    xpois_backend: str
    xpois_solver: str
    xpois_kernel_shape: tuple[int, int]
    xpois_flux_conserve: bool
    xpois_basis_terms: tuple[tuple[int, float, int, int, bool], ...]


@dataclass(frozen=True, slots=True)
class _EvidenceTorchView:
    tensor: Any
    _owner: Any = field(repr=False)


def _pack_scientific_evidence(
    *,
    cp: Any,
    xpois_result: Any,
    xfit_result: Any,
    features: Any,
    candidate_count: int,
    kernel_shape: tuple[int, int],
    flux_conserve: bool,
) -> _PackedScientificEvidence:
    """Pack all parity evidence on the producer stream as float64."""

    from cuphoton.xfit.api import DipoleFitUncertaintyReason
    from cuphoton.xfit.solver import LMStatus

    parameter_names = tuple(xfit_result.parameter_names)
    if parameter_names != _GAUSSIAN_PARAMETER_NAMES:
        raise ValueError(
            "device pipeline requires canonical Gaussian xFit parameters"
        )
    feature_names = tuple(features.feature_names)
    from .xfit_features import FEATURE_NAMES, _device_float_cast

    if feature_names != FEATURE_NAMES:
        raise ValueError(
            "device pipeline requires canonical ordered xFit features"
        )
    xfit_contract = (
        str(xfit_result.backend),
        str(xfit_result.solver),
        str(xfit_result.model),
        str(xfit_result.mode),
        str(xfit_result.dtype),
    )
    if xfit_contract != (
        "cupy",
        "levenberg-marquardt",
        "gaussian",
        "difference",
        "float64",
    ):
        raise ValueError("device pipeline xFit evidence contract changed")
    if (
        str(xpois_result.backend) != "cupy"
        or str(xpois_result.solver) != "constant"
        or bool(xpois_result.flux_conserve) != flux_conserve
    ):
        raise ValueError("device pipeline xPois evidence contract changed")
    parameter_count = len(parameter_names)
    pieces: list[Any] = []
    layout: list[DevicePipelineEvidenceSegment] = []
    offset = 0
    active_device = int(cp.cuda.runtime.getDevice())

    def append(
        name: str,
        values: Any,
        *,
        expected_shape: tuple[int, ...],
    ) -> None:
        nonlocal offset
        if not isinstance(values, cp.ndarray):
            raise TypeError(
                f"scientific evidence {name} must be a CuPy array"
            )
        if int(values.device.id) != active_device:
            raise ValueError(
                f"scientific evidence {name} is on the wrong CUDA device"
            )
        shape = tuple(int(size) for size in values.shape)
        if shape != expected_shape:
            raise ValueError(
                f"scientific evidence {name} must have shape "
                f"{expected_shape}, got {shape}"
            )
        source_dtype = np.dtype(values.dtype).str
        # Preserve float32 feature subnormals while widening the evidence.
        # Ordinary CuPy casts may flush them before the terminal copy.
        normalized = cp.ascontiguousarray(
            _device_float_cast(values, cp.float64)
        ).reshape(-1)
        pieces.append(normalized)
        layout.append(
            DevicePipelineEvidenceSegment(
                name=name,
                offset=offset,
                shape=shape,
                source_dtype=source_dtype,
            )
        )
        offset += math.prod(shape)

    kernel_coefficients = xpois_result.kernel_coefficients
    background_coefficients = xpois_result.background_coefficients
    if getattr(kernel_coefficients, "ndim", None) != 1:
        raise ValueError("xPois kernel coefficients must be one-dimensional")
    if getattr(background_coefficients, "ndim", None) != 1:
        raise ValueError(
            "xPois background coefficients must be one-dimensional"
        )
    kernel_count = int(kernel_coefficients.shape[0])
    background_count = int(background_coefficients.shape[0])
    if kernel_count < 1 or background_count < 1:
        raise ValueError("xPois coefficient evidence must be nonempty")
    append(
        "xpois.kernel_coefficients",
        kernel_coefficients,
        expected_shape=(kernel_count,),
    )
    append(
        "xpois.background_coefficients",
        background_coefficients,
        expected_shape=(background_count,),
    )
    for name in _XFIT_SCALAR_EVIDENCE_FIELDS:
        append(
            f"xfit.{name}",
            getattr(xfit_result, name),
            expected_shape=(candidate_count,),
        )
    append(
        "xfit.parameters",
        xfit_result.parameters,
        expected_shape=(candidate_count, parameter_count),
    )
    append(
        "xfit.standard_errors",
        xfit_result.standard_errors,
        expected_shape=(candidate_count, parameter_count),
    )
    append(
        "xfit.covariance",
        xfit_result.covariance,
        expected_shape=(candidate_count, parameter_count, parameter_count),
    )
    append(
        "xfit.features",
        features.values,
        expected_shape=(candidate_count, len(feature_names)),
    )
    if not pieces:
        raise RuntimeError("scientific evidence packing produced no segments")
    packed = cp.concatenate(pieces)
    if packed.dtype != cp.float64 or not packed.flags.c_contiguous:
        raise RuntimeError("scientific evidence packing changed its contract")
    if int(packed.size) != offset:
        raise RuntimeError("scientific evidence packing changed its size")

    basis_terms: list[tuple[int, float, int, int, bool]] = []
    for term in xpois_result.basis_terms:
        sigma = float(term.sigma)
        if not math.isfinite(sigma):
            raise ValueError("xPois basis-term sigma must be finite")
        basis_terms.append(
            (
                int(term.component_index),
                sigma,
                int(term.poly_u_degree),
                int(term.poly_v_degree),
                bool(term.zero_sum),
            )
        )
    if len(basis_terms) != kernel_count:
        raise ValueError("xPois basis terms must align with coefficients")
    return _PackedScientificEvidence(
        values=packed,
        layout=tuple(layout),
        candidate_count=candidate_count,
        parameter_names=parameter_names,
        feature_names=feature_names,
        xfit_schema=str(xfit_result.schema),
        xfit_backend=str(xfit_result.backend),
        xfit_solver=str(xfit_result.solver),
        xfit_model=str(xfit_result.model),
        xfit_mode=str(xfit_result.mode),
        xfit_dtype=str(xfit_result.dtype),
        xfit_status_code_names=tuple(
            (int(status), status.name) for status in LMStatus
        ),
        xfit_uncertainty_reason_code_names=tuple(
            (int(reason), reason.name)
            for reason in DipoleFitUncertaintyReason
        ),
        xpois_schema=str(xpois_result.schema),
        xpois_backend=str(xpois_result.backend),
        xpois_solver=str(xpois_result.solver),
        xpois_kernel_shape=kernel_shape,
        xpois_flux_conserve=bool(xpois_result.flux_conserve),
        xpois_basis_terms=tuple(basis_terms),
    )


def _append_prediction_evidence_layout(
    evidence: _PackedScientificEvidence,
) -> tuple[DevicePipelineEvidenceSegment, ...]:
    offset = int(evidence.values.size)
    candidate_count = evidence.candidate_count
    return (
        *evidence.layout,
        DevicePipelineEvidenceSegment(
            name="xscan.logits",
            offset=offset,
            shape=(candidate_count,),
            source_dtype=np.dtype(np.float32).str,
        ),
        DevicePipelineEvidenceSegment(
            name="xscan.probabilities",
            offset=offset + candidate_count,
            shape=(candidate_count,),
            source_dtype=np.dtype(np.float32).str,
        ),
    )


def _cupy_float64_to_torch(
    values: Any,
    *,
    device: Any,
    cp: Any,
    torch: Any,
) -> _EvidenceTorchView:
    """Borrow one contiguous float64 CuPy vector through DLPack."""

    if device.type != "cuda" or device.index is None:
        raise ValueError(
            "scientific evidence requires an explicit CUDA device"
        )
    if not isinstance(values, cp.ndarray):
        raise TypeError("scientific evidence must be a CuPy array")
    active_device = int(cp.cuda.runtime.getDevice())
    if active_device != device.index:
        raise ValueError(
            f"active CuPy device is cuda:{active_device}, expected {device}"
        )
    if int(values.device.id) != device.index:
        raise ValueError(
            f"scientific evidence is on cuda:{int(values.device.id)}, "
            f"expected {device}"
        )
    if values.ndim != 1 or int(values.size) < 1:
        raise ValueError("scientific evidence must be a nonempty vector")
    if np.dtype(values.dtype) != np.dtype(np.float64):
        raise TypeError("scientific evidence must have float64 dtype")
    if not values.flags.c_contiguous:
        raise ValueError("scientific evidence must be C-contiguous")
    tensor = torch.utils.dlpack.from_dlpack(values, copy=False)
    if tensor.device != device:
        raise RuntimeError("evidence DLPack handoff changed the CUDA device")
    if tensor.dtype != torch.float64:
        raise RuntimeError("evidence DLPack handoff changed the dtype")
    if (
        tuple(tensor.shape) != (int(values.size),)
        or not tensor.is_contiguous()
    ):
        raise RuntimeError("evidence DLPack handoff changed the shape")
    if tensor.data_ptr() != int(values.data.ptr):
        raise RuntimeError("evidence DLPack handoff copied the storage")
    return _EvidenceTorchView(tensor=tensor, _owner=values)


def _scientific_evidence_receipt(
    evidence: _PackedScientificEvidence,
    *,
    layout: tuple[DevicePipelineEvidenceSegment, ...],
    host_values: np.ndarray,
) -> DevicePipelineScientificEvidence:
    if host_values.ndim != 1 or host_values.dtype != np.float64:
        raise RuntimeError("terminal scientific evidence must be float64")
    expected_count = sum(math.prod(segment.shape) for segment in layout)
    if int(host_values.size) != expected_count:
        raise RuntimeError("terminal scientific evidence size changed")
    canonical = np.ascontiguousarray(host_values, dtype=np.dtype("<f8"))
    packed_bytes = canonical.tobytes(order="C")
    return DevicePipelineScientificEvidence(
        candidate_count=evidence.candidate_count,
        parameter_names=evidence.parameter_names,
        feature_names=evidence.feature_names,
        layout=layout,
        packed_element_count=int(canonical.size),
        packed_byte_count=len(packed_bytes),
        packed_sha256=hashlib.sha256(packed_bytes).hexdigest(),
        packed_base64=base64.b64encode(packed_bytes).decode("ascii"),
        xfit_schema=evidence.xfit_schema,
        xfit_backend=evidence.xfit_backend,
        xfit_solver=evidence.xfit_solver,
        xfit_model=evidence.xfit_model,
        xfit_mode=evidence.xfit_mode,
        xfit_dtype=evidence.xfit_dtype,
        xfit_status_code_names=evidence.xfit_status_code_names,
        xfit_uncertainty_reason_code_names=(
            evidence.xfit_uncertainty_reason_code_names
        ),
        xpois_schema=evidence.xpois_schema,
        xpois_backend=evidence.xpois_backend,
        xpois_solver=evidence.xpois_solver,
        xpois_kernel_shape=evidence.xpois_kernel_shape,
        xpois_flux_conserve=evidence.xpois_flux_conserve,
        xpois_basis_terms=evidence.xpois_basis_terms,
    )


def _host_evidence_segment(
    host_values: np.ndarray,
    *,
    layout: tuple[DevicePipelineEvidenceSegment, ...],
    name: str,
) -> np.ndarray:
    segment = next((value for value in layout if value.name == name), None)
    if segment is None:
        raise RuntimeError(f"terminal evidence is missing {name}")
    size = math.prod(segment.shape)
    return host_values[segment.offset : segment.offset + size].reshape(
        segment.shape
    )


@dataclass(frozen=True, slots=True)
class _PipelineFunctions:
    gaussian_basis_component: type[Any]
    solve_constant_kernel_device: Callable[..., Any]
    fit_dipoles_device: Callable[..., Any]
    transform_xfit_result_features_device: Callable[..., Any]
    cupy_to_torch: Callable[..., Any]
    predict_tensors: Callable[..., Mapping[str, Any]]


def _load_pipeline_functions() -> _PipelineFunctions:
    from cuphoton.xfit import fit_dipoles_device
    from cuphoton.xpois import (
        GaussianBasisComponent,
        solve_constant_kernel_device,
    )

    from .training import _cupy_to_torch, predict_tensors
    from .xfit_features import transform_xfit_result_features_device

    return _PipelineFunctions(
        gaussian_basis_component=GaussianBasisComponent,
        solve_constant_kernel_device=solve_constant_kernel_device,
        fit_dipoles_device=fit_dipoles_device,
        transform_xfit_result_features_device=(
            transform_xfit_result_features_device
        ),
        cupy_to_torch=_cupy_to_torch,
        predict_tensors=predict_tensors,
    )


class DeviceWorkerContext:
    """One post-placement CUDA context reused for serial pipeline items.

    Use :meth:`initialize` only after both CuPy and Torch have been bound to
    the explicit device in ``config``. Initialization verifies the checkpoint
    digest, loads the model exactly once, and creates one producer and one
    consumer stream. The context and every device-stage value are strictly
    same-process objects and cannot be pickled.
    """

    __slots__ = (
        "config",
        "cp",
        "torch",
        "model",
        "performance",
        "feature_variance_present",
        "device",
        "producer_stream",
        "consumer_stream",
        "solver_config",
        "load_seconds",
        "_item_lock",
        "_poisoned_reason",
    )

    def __init__(
        self,
        *,
        config: DevicePipelineConfig,
        cp: Any,
        torch: Any,
        model: Any,
        performance: Any,
        feature_variance_present: bool,
        device: Any,
        producer_stream: Any,
        consumer_stream: Any,
        solver_config: Any,
        load_seconds: float,
    ) -> None:
        self.config = config
        self.cp = cp
        self.torch = torch
        self.model = model
        self.performance = performance
        self.feature_variance_present = feature_variance_present
        self.device = device
        self.producer_stream = producer_stream
        self.consumer_stream = consumer_stream
        self.solver_config = solver_config
        self.load_seconds = load_seconds
        self._item_lock = threading.Lock()
        self._poisoned_reason: str | None = None

    @classmethod
    def initialize(cls, config: DevicePipelineConfig) -> DeviceWorkerContext:
        """Initialize after the caller has bound one CUDA device."""

        if not isinstance(config, DevicePipelineConfig):
            raise TypeError("config must be a DevicePipelineConfig")
        started = time.perf_counter()
        try:
            import cupy as cp
        except ImportError as exc:
            raise ImportError(
                "device pipeline execution requires CuPy; install "
                "'cuphoton[gpu]'"
            ) from exc
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "device pipeline execution requires Torch; install "
                "'cuphoton[gpu]'"
            ) from exc

        from cuphoton.xfit import LMConfig

        from .config import PerformanceConfig
        from .training import (
            _validate_model_device,
            load_model_from_checkpoint,
        )
        from .xfit_features import FEATURE_NAMES

        device_id = config.device_id
        try:
            cupy_count = int(cp.cuda.runtime.getDeviceCount())
            active_cupy_device = int(cp.cuda.runtime.getDevice())
        except Exception as exc:
            raise RuntimeError(
                "device pipeline requires usable CuPy CUDA"
            ) from exc
        if cupy_count <= device_id:
            raise ValueError(
                f"CUDA device {config.device} is not visible to CuPy"
            )
        if active_cupy_device != device_id:
            raise ValueError(
                "CuPy must be bound before worker initialization: "
                f"active cuda:{active_cupy_device}, expected {config.device}"
            )
        if (
            not torch.cuda.is_available()
            or torch.cuda.device_count() <= device_id
        ):
            raise ValueError(
                f"CUDA device {config.device} is not visible to Torch"
            )
        active_torch_device = int(torch.cuda.current_device())
        if active_torch_device != device_id:
            raise ValueError(
                "Torch must be bound before worker initialization: "
                f"active cuda:{active_torch_device}, expected {config.device}"
            )

        checkpoint_dir = Path(config.checkpoint_dir)
        checkpoint_path = checkpoint_dir / "checkpoint.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"xScan checkpoint does not exist: {checkpoint_path}"
            )
        if file_sha256(checkpoint_path) != config.checkpoint_sha256:
            raise RuntimeError("xScan checkpoint changed before worker load")
        feature_schema = _load_feature_schema_contract(
            Path(config.feature_schema_path),
            expected_sha256=config.feature_schema_sha256,
        )
        feature_variance_present = _validate_feature_schema_source(
            feature_schema,
            config=config,
            feature_names=FEATURE_NAMES,
        )

        device = torch.device(config.device)
        model, checkpoint, performance = load_model_from_checkpoint(
            checkpoint_dir,
            device=device,
            performance_override=PerformanceConfig(**config.inference_policy),
        )
        if asdict(performance) != config.inference_policy:
            raise RuntimeError(
                "loaded xScan inference policy changed during normalization"
            )
        if file_sha256(checkpoint_path) != config.checkpoint_sha256:
            raise RuntimeError("xScan checkpoint changed during worker load")
        if (
            file_sha256(config.feature_schema_path)
            != config.feature_schema_sha256
        ):
            raise RuntimeError(
                "xFit feature schema changed during worker load"
            )
        _validate_checkpoint_contract(
            checkpoint,
            stamp_shape=config.stamp_shape,
            feature_names=FEATURE_NAMES,
            feature_schema_sha256=config.feature_schema_sha256,
            feature_schema=feature_schema,
        )
        _validate_loaded_model_contract(model, feature_names=FEATURE_NAMES)
        _validate_model_device(model, device=device)
        solver_config = LMConfig(**config.xfit.solver_payload())

        producer_stream = cp.cuda.Stream(non_blocking=True)
        consumer_stream = torch.cuda.Stream(device=device)
        model_ready = torch.cuda.Event()
        model_ready.record(torch.cuda.current_stream(device))
        consumer_stream.wait_event(model_ready)
        return cls(
            config=config,
            cp=cp,
            torch=torch,
            model=model,
            performance=performance,
            feature_variance_present=feature_variance_present,
            device=device,
            producer_stream=producer_stream,
            consumer_stream=consumer_stream,
            solver_config=solver_config,
            load_seconds=time.perf_counter() - started,
        )

    def __reduce_ex__(self, protocol: int) -> Any:
        raise TypeError(
            "DeviceWorkerContext cannot be pickled; initialize it inside "
            "the bound worker process"
        )

    def _validate_active_device(self) -> None:
        if self._poisoned_reason is not None:
            raise RuntimeError(
                "DeviceWorkerContext is unusable after failed cleanup: "
                + self._poisoned_reason
            )
        cupy_device = int(self.cp.cuda.runtime.getDevice())
        torch_device = int(self.torch.cuda.current_device())
        if cupy_device != self.config.device_id:
            raise RuntimeError(
                f"active CuPy device changed to cuda:{cupy_device}; "
                f"expected {self.config.device}"
            )
        if torch_device != self.config.device_id:
            raise RuntimeError(
                f"active Torch device changed to cuda:{torch_device}; "
                f"expected {self.config.device}"
            )

    @contextmanager
    def _claim_item(self, item_id: str) -> Iterator[None]:
        if not self._item_lock.acquire(blocking=False):
            raise RuntimeError(
                "DeviceWorkerContext only supports serial item execution"
            )
        try:
            self._validate_active_device()
            yield
        finally:
            self._item_lock.release()


def _validate_checkpoint_contract(
    checkpoint: Mapping[str, Any],
    *,
    stamp_shape: tuple[int, int],
    feature_names: tuple[str, ...],
    feature_schema_sha256: str,
    feature_schema: Mapping[str, Any],
) -> None:
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, Mapping):
        raise ValueError("xScan checkpoint is missing model_config")
    if model_config.get("input_mode") != "triplet":
        raise ValueError(
            "device pipeline requires a triplet xScan checkpoint"
        )
    image_size = model_config.get("image_size")
    if isinstance(image_size, bool) or not isinstance(image_size, int):
        raise ValueError("xScan checkpoint image_size must be an integer")
    if stamp_shape != (image_size, image_size):
        raise ValueError(
            "xScan checkpoint image_size does not match pipeline stamp_shape"
        )
    if model_config.get("xfit_feature_names") != list(feature_names):
        raise ValueError(
            "xScan checkpoint must use the canonical ordered xFit features"
        )
    raw_bundle_identity = checkpoint.get("xfit_feature_bundle")
    if not isinstance(raw_bundle_identity, Mapping):
        raise ValueError("xScan checkpoint is missing xfit_feature_bundle")
    bundle_identity = _require_exact_fields(
        raw_bundle_identity,
        expected=frozenset(
            {"schema_sha256", "feature_sha256", "source_artifacts"}
        ),
        field_name="xScan checkpoint xfit_feature_bundle",
    )
    if bundle_identity.get("schema_sha256") != feature_schema_sha256:
        raise ValueError(
            "xScan checkpoint and live xFit feature schema identities differ"
        )
    artifacts = feature_schema.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("xFit feature schema is missing artifact metadata")
    feature_artifact = artifacts.get("features")
    if not isinstance(feature_artifact, Mapping):
        raise ValueError(
            "xFit feature schema is missing its feature identity"
        )
    if bundle_identity.get("feature_sha256") != feature_artifact.get(
        "sha256"
    ):
        raise ValueError(
            "xScan checkpoint feature identity does not match its schema"
        )
    source_artifacts = feature_schema.get("source_artifacts")
    if not isinstance(source_artifacts, Mapping):
        raise ValueError(
            "xFit feature schema is missing source artifact identities"
        )
    if bundle_identity.get("source_artifacts") != source_artifacts:
        raise ValueError(
            "xScan checkpoint source artifacts do not match "
            "its feature schema"
        )


def _load_feature_schema_contract(
    path: Path, *, expected_sha256: str
) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"xFit feature schema does not exist: {path}")
    if file_sha256(path) != expected_sha256:
        raise RuntimeError("xFit feature schema changed before worker load")
    from .xfit_features import _load_json_object, _validate_canonical_schema

    return _validate_canonical_schema(
        _load_json_object(path, description="xFit feature schema")
    )


def _validate_feature_schema_source(
    feature_schema: Mapping[str, Any],
    *,
    config: DevicePipelineConfig,
    feature_names: tuple[str, ...],
) -> bool:
    if feature_schema.get("feature_names") != list(feature_names):
        raise ValueError("xFit feature schema uses unexpected feature names")
    if feature_schema.get("feature_dtype") != "float32":
        raise ValueError("xFit feature schema must use float32 features")
    source = feature_schema.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("xFit feature schema is missing source metadata")
    expected_source = {
        "model": config.xfit.model,
        "mode": config.xfit.mode,
        "image_shape": list(config.stamp_shape),
        "mask_present": False,
        "variance_present": False,
    }
    for name, expected in expected_source.items():
        if source.get(name) != expected:
            raise ValueError(
                f"xFit feature schema source.{name} must be {expected!r}"
            )
    return bool(source["variance_present"])


def _validate_loaded_model_contract(
    model: Any, *, feature_names: tuple[str, ...]
) -> None:
    original_model = getattr(model, "_orig_mod", model)
    if getattr(original_model, "input_mode", None) != "triplet":
        raise ValueError("loaded xScan model must use triplet inputs")
    configured_names = tuple(
        getattr(original_model, "xfit_feature_names", ())
    )
    if configured_names != feature_names:
        raise ValueError(
            "loaded xScan model must use the canonical ordered xFit features"
        )


def _item_inputs(
    item: DevicePipelineItem,
) -> tuple[tuple[str, NpyArrayDescriptor], ...]:
    values: list[tuple[str, NpyArrayDescriptor]] = [
        ("reference", item.reference),
        ("target", item.target),
    ]
    if item.variance is not None:
        values.append(("variance", item.variance))
    if item.fit_mask is not None:
        values.append(("fit_mask", item.fit_mask))
    return tuple(values)


def _validate_item_contract(
    item: DevicePipelineItem, *, config: DevicePipelineConfig
) -> None:
    image_shape = item.reference.shape
    for name, descriptor in _item_inputs(item):
        if descriptor.shape != image_shape:
            raise ValueError(
                f"{name} shape {descriptor.shape} does not match "
                f"reference shape {image_shape}"
            )
        dtype = np.dtype(descriptor.dtype)
        if name == "fit_mask":
            if dtype != np.dtype(bool):
                raise TypeError(
                    "fit_mask NPY descriptor must use boolean dtype"
                )
        elif dtype.kind not in "iuf":
            raise TypeError(f"{name} NPY descriptor must use a numeric dtype")
    kernel_half_y = config.xpois.kernel_shape[0] // 2
    kernel_half_x = config.xpois.kernel_shape[1] // 2
    stamp_half_y = config.stamp_shape[0] // 2
    stamp_half_x = config.stamp_shape[1] // 2
    height, width = image_shape
    for candidate in item.candidates:
        if (
            candidate.center_x - stamp_half_x < kernel_half_x
            or candidate.center_x + stamp_half_x >= width - kernel_half_x
            or candidate.center_y - stamp_half_y < kernel_half_y
            or candidate.center_y + stamp_half_y >= height - kernel_half_y
        ):
            raise ValueError(
                f"candidate {candidate.candidate_id!r} stamp crosses the "
                "image or xPois kernel edge"
            )


def _read_item_inputs(
    item: DevicePipelineItem,
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for name, descriptor in _item_inputs(item):
        path = Path(descriptor.path)
        if not path.is_file():
            raise FileNotFoundError(
                f"{name} NPY input does not exist: {path}"
            )
        if file_sha256(path) != descriptor.sha256:
            raise RuntimeError(f"{name} NPY input changed before execution")
        values = np.load(path, allow_pickle=False)
        if not isinstance(values, np.ndarray):
            raise TypeError(f"{name} NPY input did not contain an array")
        if tuple(values.shape) != descriptor.shape:
            raise RuntimeError(f"{name} NPY shape changed before execution")
        if values.dtype.str != descriptor.dtype:
            raise RuntimeError(f"{name} NPY dtype changed before execution")
        if not values.flags.c_contiguous:
            raise ValueError(f"{name} NPY input must be C-contiguous")
        dtype = np.bool_ if name == "fit_mask" else np.float64
        arrays[name] = np.asarray(values, dtype=dtype, order="C")
    return arrays


def _verify_item_hashes(item: DevicePipelineItem, *, when: str) -> None:
    for name, descriptor in _item_inputs(item):
        if file_sha256(descriptor.path) != descriptor.sha256:
            raise RuntimeError(f"{name} NPY input changed {when}")


def _extract_stamps(
    *,
    cp: Any,
    target: Any,
    reference: Any,
    residual: Any,
    candidates: tuple[DevicePipelineCandidate, ...],
    stamp_shape: tuple[int, int],
) -> tuple[Any, Any]:
    stamps = cp.empty(
        (len(candidates), 3, *stamp_shape),
        dtype=cp.float32,
    )
    difference = cp.empty(
        (len(candidates), *stamp_shape),
        dtype=cp.float64,
    )
    half_y = stamp_shape[0] // 2
    half_x = stamp_shape[1] // 2
    for index, candidate in enumerate(candidates):
        y_slice = slice(
            candidate.center_y - half_y, candidate.center_y + half_y + 1
        )
        x_slice = slice(
            candidate.center_x - half_x, candidate.center_x + half_x + 1
        )
        stamps[index, 0] = target[y_slice, x_slice]
        stamps[index, 1] = reference[y_slice, x_slice]
        stamps[index, 2] = residual[y_slice, x_slice]
        difference[index] = residual[y_slice, x_slice]
    if not stamps.flags.c_contiguous or not difference.flags.c_contiguous:
        raise RuntimeError(
            "device stamp extraction returned noncontiguous data"
        )
    return stamps, difference


def _timed(
    timings: dict[str, float],
    name: str,
    operation: Callable[[], Any],
) -> Any:
    started = time.perf_counter()
    try:
        return operation()
    finally:
        timings[name] += time.perf_counter() - started


def _complete_failed_device_work(
    context: DeviceWorkerContext, original_error: BaseException
) -> None:
    cleanup_errors: list[str] = []
    for name, stream in (
        ("CuPy producer", context.producer_stream),
        ("Torch consumer", context.consumer_stream),
    ):
        try:
            stream.synchronize()
        except BaseException as cleanup_error:
            detail = (
                f"{name}: {type(cleanup_error).__name__}: {cleanup_error}"
            )
            cleanup_errors.append(detail)
            original_error.add_note(
                "device pipeline cleanup failed: " + detail
            )
    if cleanup_errors:
        context._poisoned_reason = "; ".join(cleanup_errors)


def _result_identity_payload(
    *,
    item: DevicePipelineItem,
    context: DeviceWorkerContext,
    xpois_result: Any,
    predictions: tuple[DevicePipelinePrediction, ...],
    scientific_evidence: DevicePipelineScientificEvidence,
    transfers: DevicePipelineTransferReceipt,
) -> dict[str, Any]:
    return {
        "schema": DEVICE_PIPELINE_RESULT_SCHEMA,
        "item_id": item.item_id,
        "device": context.config.device,
        "checkpoint_sha256": context.config.checkpoint_sha256,
        "feature_schema_sha256": context.config.feature_schema_sha256,
        "configuration_sha256": context.config.configuration_sha256,
        "input_sha256": {
            name: descriptor.sha256 for name, descriptor in _item_inputs(item)
        },
        "xpois": {
            "chi2": float(xpois_result.chi2),
            "dof": int(xpois_result.dof),
            "fit_pixel_count": int(xpois_result.fit_pixel_count),
        },
        "predictions": [value.to_payload() for value in predictions],
        "scientific_evidence": scientific_evidence.to_payload(),
        "transfers": transfers.to_payload(),
    }


def run_device_pipeline_item(
    item: DevicePipelineItem,
    context: DeviceWorkerContext,
) -> DevicePipelineResult:
    """Run one item through resident xPois, xFit, and xScan stages.

    No device result is returned or published. CuPy-to-Torch owner leases stay
    live through one blocking terminal copy containing xPois coefficients,
    full compact xFit diagnostics, features, logits, and probabilities.
    Pipeline-owned full-array D2H and post-ingress compact H2D transfers are
    forbidden; scalar/control transfers internal to the existing device-stage
    APIs are outside this function's byte instrumentation.
    """

    if not isinstance(item, DevicePipelineItem):
        raise TypeError("item must be a DevicePipelineItem")
    if not isinstance(context, DeviceWorkerContext):
        raise TypeError("context must be a DeviceWorkerContext")

    started = time.perf_counter()
    timings = {name: 0.0 for name in _TIMING_STAGES}
    functions = _load_pipeline_functions()
    with context._claim_item(item.item_id):
        _validate_item_contract(item, config=context.config)
        if context.feature_variance_present is not False:
            raise RuntimeError(
                "device pipeline context must use the frozen unweighted "
                "xFit feature schema"
            )
        host_inputs = _timed(
            timings,
            "input_read",
            lambda: _read_item_inputs(item),
        )
        input_h2d_bytes = sum(value.nbytes for value in host_inputs.values())

        leases: list[Any] = []
        try:
            with context.producer_stream:
                device_inputs = _timed(
                    timings,
                    "input_h2d",
                    lambda: {
                        name: context.cp.asarray(value)
                        for name, value in host_inputs.items()
                    },
                )
                components = [
                    functions.gaussian_basis_component(
                        sigma=sigma, degree=degree
                    )
                    for sigma, degree in zip(
                        context.config.xpois.basis_sigmas,
                        context.config.xpois.basis_degrees,
                        strict=True,
                    )
                ]
                xpois_result = _timed(
                    timings,
                    "xpois",
                    lambda: functions.solve_constant_kernel_device(
                        device_inputs["reference"],
                        device_inputs["target"],
                        components,
                        kernel_shape=context.config.xpois.kernel_shape,
                        variance=device_inputs.get("variance"),
                        fit_mask=device_inputs.get("fit_mask"),
                        background_degree=(
                            context.config.xpois.background_degree
                        ),
                        flux_conserve=context.config.xpois.flux_conserve,
                        flux_reference_index=(
                            context.config.xpois.flux_reference_index
                        ),
                    ),
                )
                stamps, difference = _timed(
                    timings,
                    "stamp_extraction",
                    lambda: _extract_stamps(
                        cp=context.cp,
                        target=device_inputs["target"],
                        reference=device_inputs["reference"],
                        residual=xpois_result.residual,
                        candidates=item.candidates,
                        stamp_shape=context.config.stamp_shape,
                    ),
                )
                xfit_result = _timed(
                    timings,
                    "xfit",
                    lambda: functions.fit_dipoles_device(
                        difference,
                        model=context.config.xfit.model,
                        mode=context.config.xfit.mode,
                        config=context.solver_config,
                    ),
                )
                features = _timed(
                    timings,
                    "feature_transform",
                    lambda: functions.transform_xfit_result_features_device(
                        xfit_result,
                        image_shape=context.config.stamp_shape,
                        variance_present=context.feature_variance_present,
                    ),
                )
                evidence_device = _timed(
                    timings,
                    "evidence_pack",
                    lambda: _pack_scientific_evidence(
                        cp=context.cp,
                        xpois_result=xpois_result,
                        xfit_result=xfit_result,
                        features=features,
                        candidate_count=len(item.candidates),
                        kernel_shape=context.config.xpois.kernel_shape,
                        flux_conserve=context.config.xpois.flux_conserve,
                    ),
                )
                evidence_layout = _append_prediction_evidence_layout(
                    evidence_device
                )

                with context.torch.cuda.stream(context.consumer_stream):
                    image_view = _timed(
                        timings,
                        "dlpack_handoff",
                        lambda: functions.cupy_to_torch(
                            stamps,
                            device=context.device,
                        ),
                    )
                    leases.append(image_view)
                    feature_view = _timed(
                        timings,
                        "dlpack_handoff",
                        lambda: functions.cupy_to_torch(
                            features.values,
                            device=context.device,
                        ),
                    )
                    leases.append(feature_view)
                    evidence_view = _timed(
                        timings,
                        "dlpack_handoff",
                        lambda: _cupy_float64_to_torch(
                            evidence_device.values,
                            device=context.device,
                            cp=context.cp,
                            torch=context.torch,
                        ),
                    )
                    leases.append(evidence_view)
                    prediction = _timed(
                        timings,
                        "xscan",
                        lambda: functions.predict_tensors(
                            model=context.model,
                            images=image_view.tensor,
                            xfit_features=feature_view.tensor,
                            device=context.device,
                            performance=context.performance,
                        ),
                    )
                    packed = _timed(
                        timings,
                        "terminal_pack",
                        lambda: context.torch.cat(
                            (
                                evidence_view.tensor,
                                prediction["logits"].to(
                                    dtype=context.torch.float64
                                ),
                                prediction["probabilities"].to(
                                    dtype=context.torch.float64
                                ),
                            ),
                            dim=0,
                        ),
                    )
                    terminal_d2h_bytes = int(
                        packed.numel() * packed.element_size()
                    )
                    host_evidence = _timed(
                        timings,
                        "terminal_d2h",
                        lambda: packed.detach().cpu().numpy(),
                    )
                    # The blocking CPU copy completes this consumer stream.
                    # Keep all owner-bearing views reachable through it.
                    if len(leases) != 3:
                        raise RuntimeError(
                            "DLPack owner leases were not retained"
                        )
        except BaseException as exc:
            _complete_failed_device_work(context, exc)
            raise

        expected_count = sum(
            math.prod(segment.shape) for segment in evidence_layout
        )
        if host_evidence.shape != (expected_count,):
            raise RuntimeError(
                "terminal scientific evidence shape changed: expected "
                f"{(expected_count,)}, got {host_evidence.shape}"
            )
        if host_evidence.dtype != np.float64:
            raise RuntimeError("terminal scientific evidence must be float64")
        host_logits = _host_evidence_segment(
            host_evidence,
            layout=evidence_layout,
            name="xscan.logits",
        )
        host_probabilities = _host_evidence_segment(
            host_evidence,
            layout=evidence_layout,
            name="xscan.probabilities",
        )
        if (
            not np.isfinite(host_logits).all()
            or not np.isfinite(host_probabilities).all()
        ):
            raise ValueError(
                "terminal xScan predictions contain non-finite values"
            )
        xpois_chi2 = float(xpois_result.chi2)
        if not math.isfinite(xpois_chi2):
            raise ValueError("terminal xPois chi2 is non-finite")
        _verify_item_hashes(item, when="during execution")
        scientific_evidence = _scientific_evidence_receipt(
            evidence_device,
            layout=evidence_layout,
            host_values=host_evidence,
        )
        # Seal a successful receipt only after the exact public evidence
        # decoder accepts both its metadata and its host-side science values.
        decode_device_pipeline_evidence(
            scientific_evidence,
            config=context.config,
        )

        predictions = tuple(
            DevicePipelinePrediction(
                candidate_id=candidate.candidate_id,
                center_x=candidate.center_x,
                center_y=candidate.center_y,
                source_index=candidate.source_index,
                logit=float(host_logits[index]),
                probability=float(host_probabilities[index]),
                decision=bool(
                    host_probabilities[index]
                    >= context.config.decision_threshold
                ),
            )
            for index, candidate in enumerate(item.candidates)
        )
        transfers = DevicePipelineTransferReceipt(
            input_h2d_bytes=input_h2d_bytes,
            terminal_d2h_bytes=terminal_d2h_bytes,
            dlpack_shared_bytes=int(
                stamps.nbytes
                + features.values.nbytes
                + evidence_device.values.nbytes
            ),
        )
        identity_payload = _result_identity_payload(
            item=item,
            context=context,
            xpois_result=xpois_result,
            predictions=predictions,
            scientific_evidence=scientific_evidence,
            transfers=transfers,
        )
        timing_receipt = DevicePipelineTimingReceipt(
            stages_seconds=tuple(
                (name, float(timings[name])) for name in _TIMING_STAGES
            ),
            total_seconds=time.perf_counter() - started,
        )
        return DevicePipelineResult(
            item_id=item.item_id,
            device=context.config.device,
            checkpoint_sha256=context.config.checkpoint_sha256,
            feature_schema_sha256=context.config.feature_schema_sha256,
            configuration_sha256=context.config.configuration_sha256,
            input_sha256=tuple(
                (name, descriptor.sha256)
                for name, descriptor in _item_inputs(item)
            ),
            xpois_chi2=xpois_chi2,
            xpois_dof=int(xpois_result.dof),
            xpois_fit_pixel_count=int(xpois_result.fit_pixel_count),
            predictions=predictions,
            scientific_evidence=scientific_evidence,
            transfers=transfers,
            timings=timing_receipt,
            result_sha256=_stable_payload_sha256(identity_payload),
        )


__all__ = [
    "DEVICE_PIPELINE_CONFIG_SCHEMA",
    "DEVICE_PIPELINE_EVIDENCE_SCHEMA",
    "DEVICE_PIPELINE_ITEM_SCHEMA",
    "DEVICE_PIPELINE_RESULT_SCHEMA",
    "DevicePipelineCandidate",
    "DevicePipelineConfig",
    "DevicePipelineEvidenceSegment",
    "DevicePipelineItem",
    "DevicePipelinePrediction",
    "DevicePipelineResult",
    "DevicePipelineScientificEvidence",
    "DevicePipelineTimingReceipt",
    "DevicePipelineTransferReceipt",
    "DeviceWorkerContext",
    "DeviceXFitPipelineConfig",
    "DeviceXPOISPipelineConfig",
    "NpyArrayDescriptor",
    "decode_device_pipeline_evidence",
    "run_device_pipeline_item",
]
