# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Dragon adapter tests for the persistent device pipeline."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import queue
import socket
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from cuphoton.core.artifacts import file_sha256
from cuphoton.xfit.api import DipoleFitUncertaintyReason
from cuphoton.xfit.solver import LMStatus
from cuphoton.xpois import (
    GaussianBasisComponent,
    build_gaussian_polynomial_basis,
    triangular_degree_pairs,
)
from cuphoton.xpois import dragon as dragon_module
from cuphoton.xscan import dragon_pipeline
from cuphoton.xscan.device_pipeline import (
    DEVICE_PIPELINE_EVIDENCE_SCHEMA,
    DEVICE_PIPELINE_RESULT_SCHEMA,
    DevicePipelineCandidate,
    DevicePipelineConfig,
    DevicePipelineItem,
    DevicePipelineTransferReceipt,
    DeviceXFitPipelineConfig,
    DeviceXPOISPipelineConfig,
    NpyArrayDescriptor,
)
from cuphoton.xscan.xfit_features import FEATURE_NAMES, _canonical_feature_row

_XFIT_SCALAR_EVIDENCE_SEGMENTS = (
    "xfit.status_codes",
    "xfit.converged",
    "xfit.evaluations",
    "xfit.residual_norm",
    "xfit.chi_square",
    "xfit.valid_pixel_count",
    "xfit.valid_pixel_fraction",
    "xfit.null_chi_square",
    "xfit.delta_chi_square",
    "xfit.fractional_null_improvement",
    "xfit.degrees_of_freedom",
    "xfit.reduced_chi_square",
    "xfit.uncertainty_valid",
    "xfit.uncertainty_reason_codes",
)
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


def _descriptor(path: Path) -> NpyArrayDescriptor:
    values = np.load(path, allow_pickle=False)
    return NpyArrayDescriptor(
        path=str(path.resolve()),
        sha256=file_sha256(path),
        shape=tuple(values.shape),
        dtype=values.dtype.str,
    )


def _config(
    tmp_path: Path,
    *,
    device: str = "cuda:0",
    xpois: DeviceXPOISPipelineConfig | None = None,
) -> DevicePipelineConfig:
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    checkpoint_path = checkpoint_dir / "checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    feature_schema_path = tmp_path / "feature-schema.json"
    feature_schema_path.write_text("{}\n", encoding="utf-8")
    return DevicePipelineConfig(
        device=device,
        checkpoint_dir=str(checkpoint_dir.resolve()),
        checkpoint_sha256=file_sha256(checkpoint_path),
        feature_schema_path=str(feature_schema_path.resolve()),
        feature_schema_sha256=file_sha256(feature_schema_path),
        stamp_shape=(11, 11),
        decision_threshold=0.5,
        xpois=xpois
        or DeviceXPOISPipelineConfig(
            kernel_shape=(3, 3),
            basis_sigmas=(0.8,),
            basis_degrees=(0,),
        ),
        xfit=DeviceXFitPipelineConfig(max_evaluations=4),
    )


def _item(tmp_path: Path, item_id: str) -> DevicePipelineItem:
    reference_path = tmp_path / f"{item_id}-reference.npy"
    target_path = tmp_path / f"{item_id}-target.npy"
    np.save(reference_path, np.zeros((31, 31), dtype=np.float32))
    np.save(target_path, np.ones((31, 31), dtype=np.float64))
    return DevicePipelineItem(
        item_id=item_id,
        reference=_descriptor(reference_path),
        target=_descriptor(target_path),
        candidates=(
            DevicePipelineCandidate(
                candidate_id=f"{item_id}-candidate",
                center_x=15,
                center_y=15,
                source_index=0,
            ),
        ),
    )


def _payload_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _fake_scientific_evidence(
    *,
    config: DevicePipelineConfig,
    candidate_count: int,
    logit: float,
    probability: float,
    xpois: DeviceXPOISPipelineConfig | None = None,
    variance_present: bool = False,
) -> dict[str, Any]:
    xpois = xpois or config.xpois
    parameter_names = [
        "amplitude",
        "sigma_x",
        "sigma_y",
        "theta",
        "x_pos",
        "y_pos",
        "x_neg",
        "y_neg",
    ]
    feature_names = list(FEATURE_NAMES)
    parameters = np.tile(
        np.asarray(
            [[2.0, 1.5, 1.25, 0.0, 1.0, 0.0, -1.0, 0.0]],
            dtype="<f8",
        ),
        (candidate_count, 1),
    )
    standard_errors = np.ones(
        (candidate_count, len(parameter_names)), dtype="<f8"
    )
    covariance = np.tile(
        np.eye(len(parameter_names), dtype="<f8"),
        (candidate_count, 1, 1),
    )
    columns: dict[str, np.ndarray] = {
        "converged": np.ones((candidate_count,), dtype=bool),
        "degrees_of_freedom": np.full((candidate_count,), 113, dtype="<i8"),
        "valid_pixel_fraction": np.ones((candidate_count,), dtype="<f8"),
        "fractional_null_improvement": np.full(
            (candidate_count,), 0.75, dtype="<f8"
        ),
        "delta_chi_square": np.full((candidate_count,), 0.75, dtype="<f8"),
        "reduced_chi_square": np.full(
            (candidate_count,), 0.25 / 113.0, dtype="<f8"
        ),
        "uncertainty_valid": np.ones((candidate_count,), dtype=bool),
    }
    for index, name in enumerate(parameter_names):
        columns[name] = parameters[:, index]
        columns[f"{name}_standard_error"] = standard_errors[:, index]
    features = np.empty((candidate_count, len(feature_names)), dtype="<f4")
    for row_index in range(candidate_count):
        features[row_index] = _canonical_feature_row(
            columns,
            row_index,
            covariance[row_index],
            model="gaussian",
            parameter_names=parameter_names,
            image_shape=config.stamp_shape,
            variance_present=variance_present,
        )
    _, expanded_basis_terms = build_gaussian_polynomial_basis(
        xpois.kernel_shape,
        tuple(
            GaussianBasisComponent(sigma=sigma, degree=degree)
            for sigma, degree in zip(
                xpois.basis_sigmas,
                xpois.basis_degrees,
                strict=True,
            )
        ),
        flux_conserve=xpois.flux_conserve,
        flux_reference_index=xpois.flux_reference_index,
    )
    background_count = len(triangular_degree_pairs(xpois.background_degree))
    arrays: list[tuple[str, np.ndarray]] = [
        (
            "xpois.kernel_coefficients",
            np.ones((len(expanded_basis_terms),), dtype="<f8"),
        ),
        (
            "xpois.background_coefficients",
            np.zeros((background_count,), dtype="<f8"),
        ),
        (
            "xfit.status_codes",
            np.full((candidate_count,), LMStatus.CONVERGED_F_TOL, dtype="i1"),
        ),
        ("xfit.converged", np.ones((candidate_count,), dtype="?")),
        ("xfit.evaluations", np.full((candidate_count,), 4, dtype="<i8")),
        (
            "xfit.residual_norm",
            np.full((candidate_count,), 0.5, dtype="<f8"),
        ),
        (
            "xfit.chi_square",
            np.full((candidate_count,), 0.25, dtype="<f8"),
        ),
        (
            "xfit.valid_pixel_count",
            np.full((candidate_count,), 121, dtype="<i8"),
        ),
        (
            "xfit.valid_pixel_fraction",
            np.ones((candidate_count,), dtype="<f8"),
        ),
        (
            "xfit.null_chi_square",
            np.ones((candidate_count,), dtype="<f8"),
        ),
        (
            "xfit.delta_chi_square",
            np.full((candidate_count,), 0.75, dtype="<f8"),
        ),
        (
            "xfit.fractional_null_improvement",
            np.full((candidate_count,), 0.75, dtype="<f8"),
        ),
        (
            "xfit.degrees_of_freedom",
            np.full((candidate_count,), 113, dtype="<i8"),
        ),
        (
            "xfit.reduced_chi_square",
            np.full((candidate_count,), 0.25 / 113.0, dtype="<f8"),
        ),
        (
            "xfit.uncertainty_valid",
            np.ones((candidate_count,), dtype="?"),
        ),
        (
            "xfit.uncertainty_reason_codes",
            np.full(
                (candidate_count,),
                DipoleFitUncertaintyReason.VALID,
                dtype="i1",
            ),
        ),
        ("xfit.parameters", parameters),
        ("xfit.standard_errors", standard_errors),
        ("xfit.covariance", covariance),
        ("xfit.features", features),
        (
            "xscan.logits",
            np.full((candidate_count,), logit, dtype="<f4"),
        ),
        (
            "xscan.probabilities",
            np.full((candidate_count,), probability, dtype="<f4"),
        ),
    ]
    layout = []
    offset = 0
    for name, values in arrays:
        layout.append(
            {
                "name": name,
                "offset": offset,
                "shape": list(values.shape),
                "source_dtype": values.dtype.str,
            }
        )
        offset += values.size
    packed = np.concatenate(
        [values.astype("<f8").reshape(-1) for _, values in arrays]
    )
    packed_bytes = packed.tobytes(order="C")
    return {
        "schema": DEVICE_PIPELINE_EVIDENCE_SCHEMA,
        "encoding": "base64",
        "packed_dtype": "<f8",
        "candidate_count": candidate_count,
        "parameter_names": parameter_names,
        "feature_names": feature_names,
        "layout": layout,
        "packed_element_count": packed.size,
        "packed_byte_count": len(packed_bytes),
        "packed_sha256": hashlib.sha256(packed_bytes).hexdigest(),
        "packed_base64": base64.b64encode(packed_bytes).decode("ascii"),
        "xfit": {
            "schema": "cuphoton.xfit.device-fit-result/v1",
            "backend": "cupy",
            "solver": "levenberg-marquardt",
            "model": "gaussian",
            "mode": "difference",
            "dtype": "float64",
            "status_code_names": {
                str(int(status)): status.name for status in LMStatus
            },
            "uncertainty_reason_code_names": {
                str(int(reason)): reason.name
                for reason in DipoleFitUncertaintyReason
            },
        },
        "xpois": {
            "schema": "cuphoton.xpois.device-fit-result/v1",
            "backend": "cupy",
            "solver": "constant",
            "kernel_shape": list(xpois.kernel_shape),
            "flux_conserve": xpois.flux_conserve,
            "basis_terms": [
                {
                    "component_index": term.component_index,
                    "sigma": float(term.sigma),
                    "poly_u_degree": term.poly_u_degree,
                    "poly_v_degree": term.poly_v_degree,
                    "zero_sum": term.zero_sum,
                }
                for term in expanded_basis_terms
            ],
        },
    }


def _seal_fake_result(payload: dict[str, Any]) -> dict[str, Any]:
    identity = {
        name: value
        for name, value in payload.items()
        if name not in {"timings", "result_sha256"}
    }
    payload["result_sha256"] = _payload_sha256(identity)
    return payload


def _fake_result_payload(
    item: DevicePipelineItem,
    config: DevicePipelineConfig,
    *,
    evidence_xpois: DeviceXPOISPipelineConfig | None = None,
    evidence_variance_present: bool = False,
) -> dict[str, Any]:
    logit = float(np.float32(0.25))
    probability = float(np.float32(1.0 / (1.0 + np.exp(-np.float32(logit)))))
    evidence = _fake_scientific_evidence(
        config=config,
        candidate_count=len(item.candidates),
        logit=logit,
        probability=probability,
        xpois=evidence_xpois,
        variance_present=evidence_variance_present,
    )
    background_count = next(
        segment
        for segment in evidence["layout"]
        if segment["name"] == "xpois.background_coefficients"
    )["shape"][0]
    coefficient_count = (
        len(evidence["xpois"]["basis_terms"]) + background_count
    )
    fit_pixel_count = coefficient_count + 9
    input_descriptors = [
        ("reference", item.reference),
        ("target", item.target),
    ]
    if item.variance is not None:
        input_descriptors.append(("variance", item.variance))
    if item.fit_mask is not None:
        input_descriptors.append(("fit_mask", item.fit_mask))
    input_h2d_bytes = sum(
        math.prod(descriptor.shape)
        * (1 if name == "fit_mask" else np.dtype(np.float64).itemsize)
        for name, descriptor in input_descriptors
    )
    evidence_device_elements = evidence["packed_element_count"] - 2 * len(
        item.candidates
    )
    dlpack_shared_bytes = (
        len(item.candidates)
        * 3
        * math.prod(config.stamp_shape)
        * np.dtype(np.float32).itemsize
        + len(item.candidates)
        * len(FEATURE_NAMES)
        * np.dtype(np.float32).itemsize
        + evidence_device_elements * np.dtype(np.float64).itemsize
    )
    payload = {
        "schema": DEVICE_PIPELINE_RESULT_SCHEMA,
        "item_id": item.item_id,
        "device": config.device,
        "checkpoint_sha256": config.checkpoint_sha256,
        "feature_schema_sha256": config.feature_schema_sha256,
        "configuration_sha256": config.configuration_sha256,
        "input_sha256": {
            name: descriptor.sha256 for name, descriptor in input_descriptors
        },
        "xpois": {
            "chi2": 1.0,
            "dof": 9,
            "fit_pixel_count": fit_pixel_count,
        },
        "predictions": [
            {
                **candidate.to_payload(),
                "logit": logit,
                "probability": probability,
                "decision": True,
            }
            for candidate in item.candidates
        ],
        "transfers": DevicePipelineTransferReceipt(
            input_h2d_bytes=input_h2d_bytes,
            terminal_d2h_bytes=8,
            dlpack_shared_bytes=dlpack_shared_bytes,
        ).to_payload(),
        "timings": {
            "stages_seconds": {
                name: 0.01 if name == "input_read" else 0.0
                for name in _TIMING_STAGES
            },
            "total_seconds": 0.02,
            "semantics": "host-elapsed; terminal-d2h-is-blocking",
        },
        "scientific_evidence": evidence,
    }
    payload["transfers"]["terminal_d2h_bytes"] = payload[
        "scientific_evidence"
    ]["packed_byte_count"]
    return _seal_fake_result(payload)


def _fake_result(
    item: DevicePipelineItem, config: DevicePipelineConfig
) -> Any:
    payload = _fake_result_payload(item, config)
    return SimpleNamespace(to_payload=lambda: payload)


class _FakeQueue(queue.Queue):
    def close(self) -> None:
        return None


class _FakePolicy:
    Placement = SimpleNamespace(HOST_NAME="host-name")

    def __init__(
        self,
        *,
        placement: Any,
        host_name: str,
        gpu_affinity: list[int],
    ):
        self.placement = placement
        self.host_name = host_name
        self.gpu_affinity = gpu_affinity


class _FakeTemplate:
    created: list[_FakeTemplate] = []

    def __init__(self, *, target: Any, args: tuple[Any, ...], policy: Any):
        self.target = target
        self.args = args
        self.policy = policy
        self.argdata = b"x" * 512
        type(self).created.append(self)


class _FakeGroup:
    init_args: tuple[Any, ...] | None = None
    init_kwargs: dict[str, Any] | None = None

    def __init__(
        self, *, restart: bool, ignore_error_on_exit: bool, walltime: float
    ) -> None:
        assert restart is False
        assert ignore_error_on_exit is False
        assert walltime > 0
        self.templates: list[_FakeTemplate] = []
        self.exit_status: list[tuple[int, int]] = []

    def add_process(self, *, nproc: int, template: _FakeTemplate) -> None:
        assert nproc == 1
        self.templates.append(template)

    def init(self, *args: Any, **kwargs: Any) -> None:
        type(self).init_args = args
        type(self).init_kwargs = kwargs

    def start(self) -> None:
        original = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            for index, template in enumerate(self.templates):
                os.environ["CUDA_VISIBLE_DEVICES"] = str(
                    template.policy.gpu_affinity[0]
                )
                template.target(*template.args)
                self.exit_status.append((1000 + index, 0))
        finally:
            if original is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = original

    def join(self, timeout: float | None = None) -> None:
        assert timeout is not None and timeout > 0

    def stop(self, patience: float = 5.0) -> None:
        assert patience == 5.0

    def close(self, patience: float = 5.0) -> None:
        assert patience == 5.0

    @property
    def inactive_puids(self) -> list[tuple[int, int]]:
        return self.exit_status


class _CorruptingSummaryGroup(_FakeGroup):
    def start(self) -> None:
        super().start()
        descriptor_path = Path(self.templates[0].args[1])
        run_dir = descriptor_path.parent.parent
        descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
        item_id = descriptor["items"][0]["item_id"]
        summary_path = run_dir / "items" / item_id / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["scientific_evidence"]["candidate_count"] += 1
        _seal_fake_result(summary)
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        record_path = run_dir / "records" / f"{item_id}.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        record["device_pipeline"]["summary_sha256"] = file_sha256(
            summary_path
        )
        record["device_pipeline"]["result_sha256"] = summary["result_sha256"]
        record_path.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


class _CorruptingRecordDeviceGroup(_FakeGroup):
    def start(self) -> None:
        super().start()
        descriptor_path = Path(self.templates[0].args[1])
        run_dir = descriptor_path.parent.parent
        descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
        item_id = descriptor["items"][0]["item_id"]
        summary_path = run_dir / "items" / item_id / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["device"] = "cuda:7"
        _seal_fake_result(summary)
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        record_path = run_dir / "records" / f"{item_id}.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        record["device"] = "cuda:7"
        record["device_pipeline"]["summary_sha256"] = file_sha256(
            summary_path
        )
        record["device_pipeline"]["result_sha256"] = summary["result_sha256"]
        record_path.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


class _FakeSystem:
    nodes = (1,)


class _FakeNode:
    hostname = socket.gethostname()
    gpus = [3]

    def __init__(self, node_id: int):
        assert node_id == 1


def _install_fake_dragon(
    monkeypatch: pytest.MonkeyPatch,
    *,
    context_type: type[Any],
    group_type: type[Any] = _FakeGroup,
) -> None:
    api = dragon_module._DragonAPI(
        System=_FakeSystem,
        Node=_FakeNode,
        Policy=_FakePolicy,
        ProcessGroup=group_type,
        ProcessTemplate=_FakeTemplate,
        Queue=_FakeQueue,
    )
    monkeypatch.setattr(dragon_module, "_load_dragon_api", lambda: api)
    monkeypatch.setattr(
        dragon_pipeline,
        "_collect_device_pipeline_gpu_identity",
        lambda backend: {
            "backend": backend,
            "device_index": 0,
            "name": "fake GPU",
            "uuid": "GPU-fake",
            "pci_bus_id": "00000000:03:00.0",
            "identity_error": None,
            "identity_warnings": [],
        },
    )
    execute_shard = dragon_module._execute_shard

    def execute_without_import_check(**kwargs: Any) -> dict[str, Any]:
        return execute_shard(require_clean_cuda_imports=False, **kwargs)

    monkeypatch.setattr(
        dragon_module, "_execute_shard", execute_without_import_check
    )
    monkeypatch.setattr(dragon_pipeline, "DeviceWorkerContext", context_type)


def test_device_pipeline_gpu_identity_loads_torch_before_cupy_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []
    identity = {
        "backend": "cupy",
        "device_index": 0,
        "name": "fake GPU",
    }

    monkeypatch.setattr(
        dragon_pipeline.importlib,
        "import_module",
        lambda name: calls.append(("import", name)),
    )

    def collect(backend: str) -> dict[str, Any]:
        calls.append(("identity", backend))
        return identity

    monkeypatch.setattr(dragon_module, "_collect_gpu_identity", collect)

    assert (
        dragon_pipeline._collect_device_pipeline_gpu_identity("cupy")
        is identity
    )
    assert calls == [("import", "torch"), ("identity", "cupy")]


def test_device_pipeline_gpu_identity_preserves_probe_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        dragon_pipeline.importlib,
        "import_module",
        lambda _name: object(),
    )

    def fail(_backend: str) -> dict[str, Any]:
        raise RuntimeError("identity failed")

    monkeypatch.setattr(dragon_module, "_collect_gpu_identity", fail)

    with pytest.raises(RuntimeError, match="identity failed"):
        dragon_pipeline._collect_device_pipeline_gpu_identity("cupy")


def test_dragon_pipeline_reuses_one_context_with_released_dragon_init(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    items = (_item(tmp_path, "pair-0"), _item(tmp_path, "pair-1"))
    contexts: list[Any] = []
    calls: list[tuple[str, Any]] = []

    class FakeContext:
        load_seconds = 0.125

        def __init__(self, initialized_config: DevicePipelineConfig):
            self.config = initialized_config

        @classmethod
        def initialize(cls, initialized_config: DevicePipelineConfig) -> Any:
            context = cls(initialized_config)
            contexts.append(context)
            return context

    _FakeTemplate.created = []
    _FakeGroup.init_args = None
    _FakeGroup.init_kwargs = None
    _install_fake_dragon(monkeypatch, context_type=FakeContext)

    def run_item(item: DevicePipelineItem, context: Any) -> Any:
        calls.append((item.item_id, context))
        return _fake_result(item, context.config)

    monkeypatch.setattr(dragon_pipeline, "run_device_pipeline_item", run_item)
    result = dragon_pipeline.run_dragon_device_pipeline(
        items=items,
        config=config,
        output_root=tmp_path / "runs",
        run_id="adapter-test",
        max_workers=1,
        result_timeout_sec=1.0,
        worker_timeout_sec=10.0,
    )

    assert result.status == "success"
    assert len(contexts) == 1
    assert [item_id for item_id, _ in calls] == ["pair-0", "pair-1"]
    assert all(context is contexts[0] for _, context in calls)
    assert _FakeGroup.init_args == ()
    assert _FakeGroup.init_kwargs == {}
    assert len(_FakeTemplate.created) == 1
    assert len(_FakeTemplate.created[0].args) == 5
    assert "candidates" not in repr(_FakeTemplate.created[0].args[:4])
    launch = json.loads(
        (result.run_dir / "launch" / "worker-0000.json").read_text()
    )
    assert launch["worker_target"] == {
        "module": "cuphoton.xscan.dragon_pipeline",
        "qualname": "_dragon_device_pipeline_worker",
    }
    assert launch["options"]["configuration"] == config.to_payload()
    assert len(launch["items"]) == 2
    record = json.loads(
        (result.run_dir / "records" / "pair-0.json").read_text()
    )
    assert record["device_pipeline"]["candidate_count"] == 1
    assert record["device_pipeline"]["positive_count"] == 1
    item_summary = json.loads(
        (result.run_dir / "items" / "pair-0" / "summary.json").read_text()
    )
    assert item_summary["schema"] == DEVICE_PIPELINE_RESULT_SCHEMA
    evidence = item_summary["scientific_evidence"]
    assert evidence["schema"] == DEVICE_PIPELINE_EVIDENCE_SCHEMA
    assert (
        record["device_pipeline"]["scientific_evidence_sha256"]
        == (evidence["packed_sha256"])
    )
    assert (
        record["device_pipeline"]["scientific_evidence_bytes"]
        == evidence["packed_byte_count"]
    )
    queued = json.dumps(result.summary["shard_results"], sort_keys=True)
    assert "scientific_evidence" not in queued
    assert "packed_base64" not in queued
    assert result.summary["terminal_record_errors"] == []


def test_dragon_pipeline_caches_context_initialization_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    items = (_item(tmp_path, "pair-0"), _item(tmp_path, "pair-1"))
    attempts = 0

    class FailedContext:
        @classmethod
        def initialize(cls, initialized_config: DevicePipelineConfig) -> Any:
            nonlocal attempts
            del initialized_config
            attempts += 1
            raise RuntimeError("context load failed")

    _install_fake_dragon(monkeypatch, context_type=FailedContext)
    monkeypatch.setattr(
        dragon_pipeline,
        "run_device_pipeline_item",
        lambda *_args: pytest.fail(
            "items must not run after context failure"
        ),
    )
    result = dragon_pipeline.run_dragon_device_pipeline(
        items=items,
        config=config,
        output_root=tmp_path / "runs",
        run_id="context-failure",
        max_workers=1,
        result_timeout_sec=1.0,
        worker_timeout_sec=10.0,
    )

    assert result.status == "failed"
    assert attempts == 1
    assert result.summary["terminal_record_audit"]["failed_item_ids"] == [
        "pair-0",
        "pair-1",
    ]
    for item in items:
        record = json.loads(
            (result.run_dir / "records" / f"{item.item_id}.json").read_text()
        )
        assert record["error"]["message"] == "context load failed"
        assert not (result.run_dir / "items" / item.item_id).exists()


def test_dragon_pipeline_rejects_nonlocal_device_before_dragon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, device="cuda:1")
    item = _item(tmp_path, "pair-0")
    monkeypatch.setattr(
        dragon_module,
        "run_dragon_work_items",
        lambda **_kwargs: pytest.fail("Dragon must not load"),
    )

    with pytest.raises(ValueError, match="local CUDA ordinal 0"):
        dragon_pipeline.run_dragon_device_pipeline(
            items=(item,),
            config=config,
            output_root=tmp_path / "runs",
        )


def test_dragon_pipeline_rejects_changed_input_before_dragon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    item = _item(tmp_path, "pair-0")
    np.save(item.reference.path, np.ones((31, 31), dtype=np.float32))
    monkeypatch.setattr(
        dragon_module,
        "run_dragon_work_items",
        lambda **_kwargs: pytest.fail("Dragon must not load"),
    )

    with pytest.raises(RuntimeError, match="changed before Dragon launch"):
        dragon_pipeline.run_dragon_device_pipeline(
            items=(item,),
            config=config,
            output_root=tmp_path / "runs",
        )


def test_worker_options_and_nested_item_identity_are_strict(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    options = {
        "schema": dragon_pipeline.DRAGON_DEVICE_PIPELINE_OPTIONS_SCHEMA,
        "backend": "cupy",
        "configuration": config.to_payload(),
        "configuration_sha256": config.configuration_sha256,
    }
    assert dragon_pipeline._strict_worker_options(options) == config
    with pytest.raises(ValueError, match="invalid fields"):
        dragon_pipeline._strict_worker_options({**options, "extra": True})

    item = _item(tmp_path, "pair-0")
    work_item = dragon_module.WorkItem(
        item_id="different", payload=item.to_payload(), weight_bytes=1
    )
    context = SimpleNamespace(load_seconds=0.0)
    with pytest.raises(ValueError, match="IDs differ"):
        dragon_pipeline._publish_item_result(
            work_item=work_item,
            item_dir=tmp_path / "item-output",
            config=config,
            context=context,
        )


def test_publish_accepts_exact_transfer_bytes_for_all_input_roles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    base_item = _item(tmp_path, "pair-0")
    variance_path = tmp_path / "variance.npy"
    fit_mask_path = tmp_path / "fit-mask.npy"
    np.save(variance_path, np.ones((31, 31), dtype=np.float32))
    np.save(fit_mask_path, np.ones((31, 31), dtype=bool))
    item = DevicePipelineItem(
        item_id=base_item.item_id,
        reference=base_item.reference,
        target=base_item.target,
        variance=_descriptor(variance_path),
        fit_mask=_descriptor(fit_mask_path),
        candidates=(
            *base_item.candidates,
            DevicePipelineCandidate(
                candidate_id="pair-0-candidate-2",
                center_x=16,
                center_y=15,
                source_index=1,
            ),
        ),
    )
    payload = _fake_result_payload(item, config)
    monkeypatch.setattr(
        dragon_pipeline,
        "run_device_pipeline_item",
        lambda *_args: SimpleNamespace(to_payload=lambda: payload),
    )

    metadata = dragon_pipeline._publish_item_result(
        work_item=dragon_module.WorkItem(
            item_id=item.item_id,
            payload=item.to_payload(),
            weight_bytes=1,
        ),
        item_dir=tmp_path / "item-output",
        config=config,
        context=SimpleNamespace(load_seconds=0.0),
    )

    transfers = metadata["device_pipeline"]["transfers"]
    assert transfers["input_h2d_bytes"] == 3 * 31 * 31 * 8 + 31 * 31
    evidence_elements = payload["scientific_evidence"]["packed_element_count"]
    assert transfers["dlpack_shared_bytes"] == (
        2 * 3 * 11 * 11 * 4
        + 2 * len(FEATURE_NAMES) * 4
        + (evidence_elements - 4) * 8
    )


@pytest.mark.parametrize(
    ("corruption", "error_type", "message"),
    [
        ("top-level", ValueError, "result has invalid fields"),
        ("prediction", ValueError, "prediction 0 has invalid fields"),
        ("prediction-type", TypeError, "prediction 0 logit must be a float"),
        ("prediction-value", ValueError, "prediction decision differs"),
        (
            "prediction-alignment",
            ValueError,
            "predictions and scientific evidence differ",
        ),
        ("xpois", ValueError, "XPOIS result has invalid fields"),
        ("xpois-type", TypeError, "XPOIS dof must be an integer"),
        ("timing", ValueError, "timing receipt has invalid fields"),
        ("timing-value", ValueError, "total timing must be at least 0.0"),
        ("transfer", ValueError, "transfer receipt has invalid fields"),
        ("transfer-value", ValueError, "transfer receipt differs"),
        ("input-h2d-bytes", ValueError, "input H2D byte count differs"),
        ("dlpack-bytes", ValueError, "DLPack shared byte count differs"),
        ("evidence", ValueError, "packed SHA-256 changed"),
        ("result", ValueError, "result SHA-256 differs"),
    ],
)
def test_publish_rejects_corrupt_v2_result_before_creating_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
    error_type: type[Exception],
    message: str,
) -> None:
    config = _config(tmp_path)
    item = _item(tmp_path, "pair-0")
    payload = _fake_result_payload(item, config)
    if corruption == "top-level":
        payload["unexpected"] = True
        _seal_fake_result(payload)
    elif corruption == "prediction":
        payload["predictions"][0]["unexpected"] = True
        _seal_fake_result(payload)
    elif corruption == "prediction-type":
        payload["predictions"][0]["logit"] = 1
        _seal_fake_result(payload)
    elif corruption == "prediction-value":
        payload["predictions"][0]["decision"] = False
        _seal_fake_result(payload)
    elif corruption == "prediction-alignment":
        payload["predictions"][0]["logit"] = 0.5
        _seal_fake_result(payload)
    elif corruption == "xpois":
        payload["xpois"]["unexpected"] = True
        _seal_fake_result(payload)
    elif corruption == "xpois-type":
        payload["xpois"]["dof"] = True
        _seal_fake_result(payload)
    elif corruption == "timing":
        payload["timings"]["unexpected"] = True
    elif corruption == "timing-value":
        payload["timings"]["total_seconds"] = -1.0
    elif corruption == "transfer":
        payload["transfers"]["unexpected"] = True
        _seal_fake_result(payload)
    elif corruption == "transfer-value":
        payload["transfers"]["terminal_d2h_calls"] = 0
        _seal_fake_result(payload)
    elif corruption == "input-h2d-bytes":
        payload["transfers"]["input_h2d_bytes"] += 1
        _seal_fake_result(payload)
    elif corruption == "dlpack-bytes":
        payload["transfers"]["dlpack_shared_bytes"] += 1
        _seal_fake_result(payload)
    elif corruption == "evidence":
        payload["scientific_evidence"]["packed_sha256"] = "0" * 64
        _seal_fake_result(payload)
    else:
        payload["result_sha256"] = "0" * 64
    monkeypatch.setattr(
        dragon_pipeline,
        "run_device_pipeline_item",
        lambda *_args: SimpleNamespace(to_payload=lambda: payload),
    )
    item_dir = tmp_path / "item-output"

    with pytest.raises(error_type, match=message):
        dragon_pipeline._publish_item_result(
            work_item=dragon_module.WorkItem(
                item_id=item.item_id,
                payload=item.to_payload(),
                weight_bytes=1,
            ),
            item_dir=item_dir,
            config=config,
            context=SimpleNamespace(load_seconds=0.0),
        )

    assert not item_dir.exists()


@pytest.mark.parametrize(
    "corruption",
    [
        "kernel-shape",
        "flux-policy",
        "flux-reference-order",
        "background-count",
        "variance-feature",
    ],
)
def test_publish_rejects_rehashed_evidence_not_bound_to_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    if corruption == "flux-reference-order":
        xpois = DeviceXPOISPipelineConfig(
            kernel_shape=(3, 3),
            basis_sigmas=(0.8, 1.2),
            basis_degrees=(0, 0),
            flux_conserve=True,
            flux_reference_index=1,
        )
    elif corruption == "background-count":
        xpois = DeviceXPOISPipelineConfig(
            kernel_shape=(3, 3),
            basis_sigmas=(0.8,),
            basis_degrees=(0,),
            background_degree=1,
        )
    else:
        xpois = DeviceXPOISPipelineConfig(
            kernel_shape=(3, 3),
            basis_sigmas=(0.8,),
            basis_degrees=(0,),
        )
    config = _config(tmp_path, xpois=xpois)
    item = _item(tmp_path, "pair-0")
    evidence_xpois = None
    if corruption == "background-count":
        evidence_xpois = DeviceXPOISPipelineConfig(
            kernel_shape=(3, 3),
            basis_sigmas=(0.8,),
            basis_degrees=(0,),
            background_degree=0,
        )
    payload = _fake_result_payload(
        item,
        config,
        evidence_xpois=evidence_xpois,
        evidence_variance_present=corruption == "variance-feature",
    )
    evidence = payload["scientific_evidence"]
    if corruption == "kernel-shape":
        evidence["xpois"]["kernel_shape"] = [5, 5]
    elif corruption == "flux-policy":
        evidence["xpois"]["flux_conserve"] = True
    elif corruption == "flux-reference-order":
        terms = evidence["xpois"]["basis_terms"]
        assert [term["component_index"] for term in terms] == [1, 0]
        evidence["xpois"]["basis_terms"] = [
            {**terms[1], "zero_sum": False},
            {**terms[0], "zero_sum": True},
        ]
    _seal_fake_result(payload)
    monkeypatch.setattr(
        dragon_pipeline,
        "run_device_pipeline_item",
        lambda *_args: SimpleNamespace(to_payload=lambda: payload),
    )
    item_dir = tmp_path / "item-output"

    with pytest.raises(ValueError, match="scientific evidence"):
        dragon_pipeline._publish_item_result(
            work_item=dragon_module.WorkItem(
                item_id=item.item_id,
                payload=item.to_payload(),
                weight_bytes=1,
            ),
            item_dir=item_dir,
            config=config,
            context=SimpleNamespace(load_seconds=0.0),
        )

    assert not item_dir.exists()


def test_coordinator_revalidates_file_backed_v2_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    item = _item(tmp_path, "pair-0")

    class FakeContext:
        load_seconds = 0.125

        def __init__(self, initialized_config: DevicePipelineConfig):
            self.config = initialized_config

        @classmethod
        def initialize(cls, initialized_config: DevicePipelineConfig) -> Any:
            return cls(initialized_config)

    _install_fake_dragon(
        monkeypatch,
        context_type=FakeContext,
        group_type=_CorruptingSummaryGroup,
    )
    monkeypatch.setattr(
        dragon_pipeline,
        "run_device_pipeline_item",
        lambda value, context: _fake_result(value, context.config),
    )

    result = dragon_pipeline.run_dragon_device_pipeline(
        items=(item,),
        config=config,
        output_root=tmp_path / "runs",
        run_id="corrupt-file-backed-result",
        max_workers=1,
        result_timeout_sec=1.0,
        worker_timeout_sec=10.0,
    )

    assert result.status == "failed"
    assert result.summary["shard_result_audit"]["ok"] is True
    assert result.summary["process_exit_audit"]["ok"] is True
    assert result.summary["terminal_record_errors"] == [
        {
            "item_id": item.item_id,
            "type": "InvalidTerminalRecord",
            "message": "record 0 has invalid field(s): device_pipeline",
        }
    ]


def test_coordinator_rejects_record_device_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    item = _item(tmp_path, "pair-0")

    class FakeContext:
        load_seconds = 0.125

        def __init__(self, initialized_config: DevicePipelineConfig):
            self.config = initialized_config

        @classmethod
        def initialize(cls, initialized_config: DevicePipelineConfig) -> Any:
            return cls(initialized_config)

    _install_fake_dragon(
        monkeypatch,
        context_type=FakeContext,
        group_type=_CorruptingRecordDeviceGroup,
    )
    monkeypatch.setattr(
        dragon_pipeline,
        "run_device_pipeline_item",
        lambda value, context: _fake_result(value, context.config),
    )

    result = dragon_pipeline.run_dragon_device_pipeline(
        items=(item,),
        config=config,
        output_root=tmp_path / "runs",
        run_id="corrupt-record-device",
        max_workers=1,
        result_timeout_sec=1.0,
        worker_timeout_sec=10.0,
    )

    assert result.status == "failed"
    assert result.summary["shard_result_audit"]["ok"] is True
    assert result.summary["process_exit_audit"]["ok"] is True
    assert result.summary["terminal_record_errors"] == [
        {
            "item_id": item.item_id,
            "type": "InvalidTerminalRecord",
            "message": (
                "record 0 has invalid field(s): device, device_pipeline"
            ),
        }
    ]
