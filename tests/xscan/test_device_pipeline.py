# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Persistent same-process device pipeline contract tests."""

from __future__ import annotations

import base64
import gc
import hashlib
import json
import pickle
import subprocess
import sys
import weakref
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from cuphoton.core.artifacts import file_sha256
from cuphoton.xscan.device_pipeline import (
    DEVICE_PIPELINE_CONFIG_SCHEMA,
    DEVICE_PIPELINE_ITEM_SCHEMA,
    DevicePipelineCandidate,
    DevicePipelineConfig,
    DevicePipelineItem,
    DeviceWorkerContext,
    DeviceXFitPipelineConfig,
    DeviceXPOISPipelineConfig,
    NpyArrayDescriptor,
    _PipelineFunctions,
    _validate_checkpoint_contract,
    _validate_feature_schema_source,
    _validate_loaded_model_contract,
    decode_device_pipeline_evidence,
    run_device_pipeline_item,
)
from cuphoton.xscan.xfit_features import FEATURE_NAMES

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


def _descriptor(path: Path) -> NpyArrayDescriptor:
    values = np.load(path, allow_pickle=False)
    return NpyArrayDescriptor(
        path=str(path.resolve()),
        sha256=file_sha256(path),
        shape=tuple(values.shape),
        dtype=values.dtype.str,
    )


def _config(
    checkpoint_dir: Path,
    *,
    stamp_shape: tuple[int, int] = (11, 11),
    xpois: DeviceXPOISPipelineConfig | None = None,
) -> DevicePipelineConfig:
    feature_schema_path = checkpoint_dir / "feature-schema.json"
    if not feature_schema_path.exists():
        feature_schema_path.write_text("{}\n", encoding="utf-8")
    return DevicePipelineConfig(
        device="cuda:0",
        checkpoint_dir=str(checkpoint_dir.resolve()),
        checkpoint_sha256=file_sha256(checkpoint_dir / "checkpoint.pt"),
        feature_schema_path=str(feature_schema_path.resolve()),
        feature_schema_sha256=file_sha256(feature_schema_path),
        stamp_shape=stamp_shape,
        decision_threshold=0.5,
        xpois=(
            xpois
            if xpois is not None
            else DeviceXPOISPipelineConfig(
                kernel_shape=(3, 3),
                basis_sigmas=(0.8,),
                basis_degrees=(0,),
            )
        ),
        xfit=DeviceXFitPipelineConfig(max_evaluations=4),
    )


def _feature_schema(config: DevicePipelineConfig) -> dict[str, Any]:
    return {
        "feature_names": list(FEATURE_NAMES),
        "feature_dtype": "float32",
        "source": {
            "model": "gaussian",
            "mode": "difference",
            "image_shape": list(config.stamp_shape),
            "mask_present": False,
            "variance_present": False,
        },
        "source_artifacts": {"summary": {"sha256": "a" * 64}},
        "artifacts": {"features": {"sha256": "b" * 64}},
    }


def _checkpoint_contract(config: DevicePipelineConfig) -> dict[str, Any]:
    schema = _feature_schema(config)
    return {
        "model_config": {
            "input_mode": "triplet",
            "image_size": config.stamp_shape[0],
            "xfit_feature_names": list(FEATURE_NAMES),
        },
        "xfit_feature_bundle": {
            "schema_sha256": config.feature_schema_sha256,
            "feature_sha256": schema["artifacts"]["features"]["sha256"],
            "source_artifacts": schema["source_artifacts"],
        },
    }


def _item(
    reference_path: Path,
    target_path: Path,
    *,
    variance_path: Path | None = None,
) -> DevicePipelineItem:
    return DevicePipelineItem(
        item_id="pair-0",
        reference=_descriptor(reference_path),
        target=_descriptor(target_path),
        variance=(
            None if variance_path is None else _descriptor(variance_path)
        ),
        candidates=(
            DevicePipelineCandidate(
                candidate_id="candidate-b",
                center_x=20,
                center_y=20,
                source_index=9,
            ),
            DevicePipelineCandidate(
                candidate_id=17,
                center_x=12,
                center_y=12,
                source_index=3,
            ),
        ),
    )


def _gpu_modules() -> tuple[Any, Any]:
    torch = pytest.importorskip("torch")
    cp = pytest.importorskip("cupy")
    try:
        if not torch.cuda.is_available():
            pytest.skip("Torch CUDA is unavailable")
        if int(cp.cuda.runtime.getDeviceCount()) < 1:
            pytest.skip("CuPy CUDA is unavailable")
    except Exception as exc:
        pytest.skip(f"CUDA runtime is unavailable: {exc}")
    cp.cuda.Device(0).use()
    torch.cuda.set_device(0)
    return cp, torch


def _fake_xfit_result(cp: Any, candidate_count: int) -> Any:
    parameter_count = len(_GAUSSIAN_PARAMETER_NAMES)
    covariance = cp.full(
        (candidate_count, parameter_count, parameter_count),
        cp.nan,
        dtype=cp.float64,
    )
    standard_errors = cp.full(
        (candidate_count, parameter_count),
        cp.nan,
        dtype=cp.float64,
    )
    if candidate_count > 1:
        covariance[1:] = cp.eye(parameter_count, dtype=cp.float64)[None]
        standard_errors[1:] = 1.0
    uncertainty_valid = cp.ones(candidate_count, dtype=cp.bool_)
    uncertainty_valid[0] = False
    uncertainty_reason_codes = cp.zeros(candidate_count, dtype=cp.int8)
    uncertainty_reason_codes[0] = 5
    parameters = cp.asarray(
        [2.0, 1.5, 1.25, 0.0, 1.0, 0.0, -1.0, 0.0],
        dtype=cp.float64,
    )[None].repeat(candidate_count, axis=0)
    chi_square = cp.arange(candidate_count, dtype=cp.float64) + 0.2
    null_chi_square = chi_square + 0.4
    valid_pixel_count = cp.full(candidate_count, 121, dtype=cp.int64)
    degrees_of_freedom = valid_pixel_count - parameter_count
    return SimpleNamespace(
        parameters=parameters,
        parameter_names=_GAUSSIAN_PARAMETER_NAMES,
        status_codes=cp.ones(candidate_count, dtype=cp.int8),
        converged=cp.ones(candidate_count, dtype=cp.bool_),
        evaluations=cp.arange(candidate_count, dtype=cp.int64) + 4,
        residual_norm=cp.sqrt(chi_square),
        chi_square=chi_square,
        valid_pixel_count=valid_pixel_count,
        valid_pixel_fraction=cp.ones(candidate_count, dtype=cp.float64),
        null_chi_square=null_chi_square,
        delta_chi_square=null_chi_square - chi_square,
        fractional_null_improvement=(1.0 - chi_square / null_chi_square),
        degrees_of_freedom=degrees_of_freedom,
        reduced_chi_square=chi_square / degrees_of_freedom,
        covariance=covariance,
        standard_errors=standard_errors,
        uncertainty_valid=uncertainty_valid,
        uncertainty_reason_codes=uncertainty_reason_codes,
        schema="cuphoton.xfit.device-fit-result/v1",
        backend="cupy",
        solver="levenberg-marquardt",
        model="gaussian",
        mode="difference",
        dtype="float64",
    )


def _canonical_test_features(
    *,
    parameters: np.ndarray,
    standard_errors: np.ndarray,
    covariance: np.ndarray,
    converged: np.ndarray,
    degrees_of_freedom: np.ndarray,
    valid_pixel_fraction: np.ndarray,
    fractional_null_improvement: np.ndarray,
    delta_chi_square: np.ndarray,
    reduced_chi_square: np.ndarray,
    uncertainty_valid: np.ndarray,
) -> np.ndarray:
    from cuphoton.xscan.xfit_features import _canonical_feature_row

    columns: dict[str, np.ndarray] = {
        "converged": converged,
        "degrees_of_freedom": degrees_of_freedom,
        "valid_pixel_fraction": valid_pixel_fraction,
        "fractional_null_improvement": fractional_null_improvement,
        "delta_chi_square": delta_chi_square,
        "reduced_chi_square": reduced_chi_square,
        "uncertainty_valid": uncertainty_valid,
    }
    for parameter_index, parameter_name in enumerate(
        _GAUSSIAN_PARAMETER_NAMES
    ):
        columns[parameter_name] = parameters[:, parameter_index]
        columns[f"{parameter_name}_standard_error"] = standard_errors[
            :, parameter_index
        ]
    return np.stack(
        [
            _canonical_feature_row(
                columns,
                row_index,
                covariance[row_index],
                model="gaussian",
                parameter_names=_GAUSSIAN_PARAMETER_NAMES,
                image_shape=(11, 11),
                variance_present=False,
            )
            for row_index in range(parameters.shape[0])
        ]
    )


def _fake_features(
    cp: Any,
    candidate_count: int,
) -> Any:
    parameter_count = len(_GAUSSIAN_PARAMETER_NAMES)
    parameters = np.repeat(
        np.asarray(
            [[2.0, 1.5, 1.25, 0.0, 1.0, 0.0, -1.0, 0.0]],
            dtype=np.float64,
        ),
        candidate_count,
        axis=0,
    )
    covariance = np.full(
        (candidate_count, parameter_count, parameter_count),
        np.nan,
        dtype=np.float64,
    )
    standard_errors = np.full(
        (candidate_count, parameter_count), np.nan, dtype=np.float64
    )
    if candidate_count > 1:
        covariance[1:] = np.eye(parameter_count, dtype=np.float64)[None]
        standard_errors[1:] = 1.0
    chi_square = np.arange(candidate_count, dtype=np.float64) + 0.2
    null_chi_square = chi_square + 0.4
    uncertainty_valid = np.ones(candidate_count, dtype=bool)
    uncertainty_valid[0] = False
    values = _canonical_test_features(
        parameters=parameters,
        standard_errors=standard_errors,
        covariance=covariance,
        converged=np.ones(candidate_count, dtype=bool),
        degrees_of_freedom=np.full(candidate_count, 113, dtype=np.int64),
        valid_pixel_fraction=np.ones(candidate_count, dtype=np.float64),
        fractional_null_improvement=(1.0 - chi_square / null_chi_square),
        delta_chi_square=null_chi_square - chi_square,
        reduced_chi_square=chi_square / 113.0,
        uncertainty_valid=uncertainty_valid,
    )
    return SimpleNamespace(
        values=cp.asarray(values),
        feature_names=FEATURE_NAMES,
    )


def _fake_xpois_result(cp: Any, residual: Any, *, chi2: float) -> Any:
    return SimpleNamespace(
        residual=residual,
        kernel_coefficients=cp.asarray([1.25], dtype=cp.float64),
        background_coefficients=cp.asarray([0.5], dtype=cp.float64),
        basis_terms=(
            SimpleNamespace(
                component_index=0,
                sigma=0.8,
                poly_u_degree=0,
                poly_v_degree=0,
                zero_sum=False,
            ),
        ),
        chi2=chi2,
        dof=700,
        fit_pixel_count=702,
        flux_conserve=False,
        schema="cuphoton.xpois.device-fit-result/v1",
        backend="cupy",
        solver="constant",
    )


def _scientific_evidence_payload(
    *,
    basis_terms: list[dict[str, Any]] | None = None,
    background_count: int = 1,
    flux_conserve: bool = False,
) -> dict[str, Any]:
    from cuphoton.xfit import DipoleFitUncertaintyReason, LMStatus

    candidate_count = 2
    parameter_count = len(_GAUSSIAN_PARAMETER_NAMES)
    chi_square = np.asarray([0.25, 1.0], dtype=np.float64)
    null_chi_square = np.asarray([1.25, 2.0], dtype=np.float64)
    valid_pixel_count = np.full(candidate_count, 121, dtype=np.int64)
    degrees_of_freedom = valid_pixel_count - parameter_count
    parameters = np.repeat(
        np.asarray(
            [[2.0, 1.5, 1.25, 0.0, 1.0, 0.0, -1.0, 0.0]],
            dtype=np.float64,
        ),
        candidate_count,
        axis=0,
    )
    covariance = np.full(
        (candidate_count, parameter_count, parameter_count),
        np.nan,
        dtype=np.float64,
    )
    covariance[0] = np.eye(parameter_count, dtype=np.float64)
    standard_errors = np.full(
        (candidate_count, parameter_count), np.nan, dtype=np.float64
    )
    standard_errors[0] = 1.0
    converged = np.asarray([True, False], dtype=bool)
    valid_pixel_fraction = np.ones(candidate_count, dtype=np.float64)
    fractional_null_improvement = 1.0 - chi_square / null_chi_square
    delta_chi_square = null_chi_square - chi_square
    reduced_chi_square = chi_square / degrees_of_freedom
    uncertainty_valid = np.asarray([True, False], dtype=bool)
    features = _canonical_test_features(
        parameters=parameters,
        standard_errors=standard_errors,
        covariance=covariance,
        converged=converged,
        degrees_of_freedom=degrees_of_freedom,
        valid_pixel_fraction=valid_pixel_fraction,
        fractional_null_improvement=fractional_null_improvement,
        delta_chi_square=delta_chi_square,
        reduced_chi_square=reduced_chi_square,
        uncertainty_valid=uncertainty_valid,
    )
    if basis_terms is None:
        basis_terms = [
            {
                "component_index": 0,
                "sigma": 0.8,
                "poly_u_degree": 0,
                "poly_v_degree": 0,
                "zero_sum": False,
            }
        ]
    logits = np.asarray([-1.0, 1.0], dtype=np.float32)
    probabilities = (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
    arrays = (
        (
            "xpois.kernel_coefficients",
            np.arange(len(basis_terms), dtype=np.float64) + 1.25,
        ),
        (
            "xpois.background_coefficients",
            np.arange(background_count, dtype=np.float64) + 0.5,
        ),
        (
            "xfit.status_codes",
            np.asarray(
                [LMStatus.CONVERGED_F_TOL, LMStatus.MAX_EVALUATIONS],
                dtype=np.int8,
            ),
        ),
        ("xfit.converged", converged),
        ("xfit.evaluations", np.asarray([4, 4], dtype=np.int64)),
        ("xfit.residual_norm", np.sqrt(chi_square)),
        ("xfit.chi_square", chi_square),
        ("xfit.valid_pixel_count", valid_pixel_count),
        (
            "xfit.valid_pixel_fraction",
            valid_pixel_fraction,
        ),
        ("xfit.null_chi_square", null_chi_square),
        ("xfit.delta_chi_square", delta_chi_square),
        (
            "xfit.fractional_null_improvement",
            fractional_null_improvement,
        ),
        ("xfit.degrees_of_freedom", degrees_of_freedom),
        (
            "xfit.reduced_chi_square",
            reduced_chi_square,
        ),
        ("xfit.uncertainty_valid", uncertainty_valid),
        (
            "xfit.uncertainty_reason_codes",
            np.asarray(
                [
                    DipoleFitUncertaintyReason.VALID,
                    DipoleFitUncertaintyReason.FIT_NOT_CONVERGED,
                ],
                dtype=np.int8,
            ),
        ),
        ("xfit.parameters", parameters),
        ("xfit.standard_errors", standard_errors),
        ("xfit.covariance", covariance),
        ("xfit.features", features),
        ("xscan.logits", logits),
        ("xscan.probabilities", probabilities),
    )
    offset = 0
    layout: list[dict[str, Any]] = []
    packed_pieces: list[np.ndarray] = []
    for name, array in arrays:
        values = np.asarray(array)
        layout.append(
            {
                "name": name,
                "offset": offset,
                "shape": list(values.shape),
                "source_dtype": values.dtype.str,
            }
        )
        packed_pieces.append(values.astype(np.float64).reshape(-1))
        offset += int(values.size)
    packed = np.ascontiguousarray(
        np.concatenate(packed_pieces), dtype=np.dtype("<f8")
    )
    packed_bytes = packed.tobytes()
    return {
        "schema": ("cuphoton.xscan.device-pipeline.scientific-evidence/v1"),
        "encoding": "base64",
        "packed_dtype": "<f8",
        "candidate_count": candidate_count,
        "parameter_names": list(_GAUSSIAN_PARAMETER_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "layout": layout,
        "packed_element_count": int(packed.size),
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
            "kernel_shape": [3, 3],
            "flux_conserve": flux_conserve,
            "basis_terms": [dict(term) for term in basis_terms],
        },
    }


def _rewrite_evidence_segment(
    payload: dict[str, Any],
    name: str,
    index: tuple[int, ...],
    value: float,
) -> None:
    segment = next(item for item in payload["layout"] if item["name"] == name)
    packed = np.frombuffer(
        base64.b64decode(payload["packed_base64"]),
        dtype=np.dtype("<f8"),
    ).copy()
    size = int(np.prod(segment["shape"], dtype=np.int64))
    values = packed[segment["offset"] : segment["offset"] + size].reshape(
        segment["shape"]
    )
    values[index] = value
    packed_bytes = packed.tobytes()
    payload["packed_sha256"] = hashlib.sha256(packed_bytes).hexdigest()
    payload["packed_base64"] = base64.b64encode(packed_bytes).decode("ascii")


def _evidence_config(
    tmp_path: Path,
    *,
    xpois: DeviceXPOISPipelineConfig | None = None,
) -> DevicePipelineConfig:
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "checkpoint.pt").write_bytes(b"checkpoint")
    return _config(checkpoint_dir, xpois=xpois)


def test_device_pipeline_import_keeps_cuda_dependencies_lazy() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import cuphoton.xscan.device_pipeline; "
                "raise SystemExit(any(name in sys.modules for name in "
                "('cupy', 'torch')))"
            ),
        ],
        check=False,
    )
    assert result.returncode == 0


def test_scientific_evidence_v1_roundtrip_preserves_expected_nans() -> None:
    payload = _scientific_evidence_payload()

    json.dumps(payload, allow_nan=False)
    decoded = decode_device_pipeline_evidence(payload)

    assert np.isfinite(decoded["xfit.covariance"][0]).all()
    assert np.isnan(decoded["xfit.covariance"][1]).all()
    assert np.isfinite(decoded["xfit.standard_errors"][0]).all()
    assert np.isnan(decoded["xfit.standard_errors"][1]).all()


def test_scientific_evidence_validates_raw_basis_zero_sum_parity() -> None:
    payload = _scientific_evidence_payload()
    term = payload["xpois"]["basis_terms"][0]
    term["poly_v_degree"] = 1
    term["zero_sum"] = True

    decode_device_pipeline_evidence(payload)

    term["zero_sum"] = False
    with pytest.raises(ValueError, match="zero_sum"):
        decode_device_pipeline_evidence(payload)


def test_scientific_evidence_accepts_narrow_even_zero_sum_basis(
    tmp_path: Path,
) -> None:
    from dataclasses import asdict

    from cuphoton.xpois import (
        GaussianBasisComponent,
        build_gaussian_polynomial_basis,
    )

    # Even powers can have numerically zero sums for narrow kernels. Use
    # the solver's actual metadata instead of inferring it from parity.
    _, terms = build_gaussian_polynomial_basis(
        (3, 3), [GaussianBasisComponent(0.1, 2)]
    )
    assert terms[2].poly_v_degree == 2 and terms[2].zero_sum
    config = _evidence_config(
        tmp_path,
        xpois=DeviceXPOISPipelineConfig(
            kernel_shape=(3, 3), basis_sigmas=(0.1,), basis_degrees=(2,)
        ),
    )
    payload = _scientific_evidence_payload(
        basis_terms=[asdict(term) for term in terms]
    )
    decode_device_pipeline_evidence(payload, config=config)
    payload["xpois"]["basis_terms"][2]["zero_sum"] = False
    with pytest.raises(ValueError, match="zero_sum"):
        decode_device_pipeline_evidence(payload, config=config)


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    [
        (("xfit", "backend"), "numpy", "xfit.backend"),
        (
            ("xfit", "status_code_names"),
            {"0": "ACTIVE"},
            "status_code_names",
        ),
        (("xpois", "solver"), "spatial", "xpois.solver"),
        (
            ("parameter_names",),
            list(reversed(_GAUSSIAN_PARAMETER_NAMES)),
            "parameter_names",
        ),
        (
            ("feature_names",),
            list(reversed(FEATURE_NAMES)),
            "feature_names",
        ),
        (("candidate_count",), 3, "packed element count"),
        (("xpois", "kernel_shape"), [4, 3], "kernel_shape"),
        (
            ("xpois", "basis_terms", 0, "zero_sum"),
            True,
            "zero_sum",
        ),
    ],
)
def test_scientific_evidence_rejects_metadata_contract_drift(
    path: tuple[str | int, ...],
    replacement: Any,
    message: str,
) -> None:
    payload = _scientific_evidence_payload()
    target: Any = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement

    with pytest.raises((TypeError, ValueError), match=message):
        decode_device_pipeline_evidence(payload)


@pytest.mark.parametrize(
    ("segment_index", "field", "replacement", "message"),
    [
        (0, "name", "xpois.background_coefficients", "names/order"),
        (2, "shape", [3], "shape changed"),
        (1, "shape", [2], "not triangular"),
        (-1, "source_dtype", "<f8", "source dtype changed"),
        (4, "offset", 99, "contiguous"),
    ],
)
def test_scientific_evidence_rejects_layout_contract_drift(
    segment_index: int,
    field: str,
    replacement: Any,
    message: str,
) -> None:
    payload = _scientific_evidence_payload()
    payload["layout"][segment_index][field] = replacement

    with pytest.raises(ValueError, match=message):
        decode_device_pipeline_evidence(payload)


@pytest.mark.parametrize(
    ("name", "index", "replacement", "message"),
    [
        (
            "xfit.covariance",
            (0, 0, 0),
            np.nan,
            "valid uncertainties",
        ),
        (
            "xfit.covariance",
            (1, 0, 0),
            np.inf,
            "unavailable uncertainties",
        ),
        (
            "xfit.standard_errors",
            (0, 0),
            2.0,
            "standard errors",
        ),
        (
            "xfit.uncertainty_reason_codes",
            (1,),
            99.0,
            "unknown uncertainty reason",
        ),
        ("xfit.features", (0, 4), 2.0, "declared range"),
        (
            "xscan.probabilities",
            (0,),
            0.5,
            "do not match logits",
        ),
    ],
)
def test_scientific_evidence_rejects_invalid_host_semantics(
    name: str,
    index: tuple[int, ...],
    replacement: float,
    message: str,
) -> None:
    payload = _scientific_evidence_payload()
    _rewrite_evidence_segment(payload, name, index, replacement)

    with pytest.raises(ValueError, match=message):
        decode_device_pipeline_evidence(payload)


def test_scientific_evidence_fit_valid_requires_converged_fit() -> None:
    payload = _scientific_evidence_payload()
    _rewrite_evidence_segment(
        payload,
        "xfit.features",
        (1, FEATURE_NAMES.index("fit_valid")),
        1.0,
    )
    _rewrite_evidence_segment(
        payload,
        "xfit.features",
        (1, FEATURE_NAMES.index("gaussian_shape_available")),
        1.0,
    )

    with pytest.raises(ValueError, match="fit_valid requires a converged"):
        decode_device_pipeline_evidence(payload)


def test_scientific_evidence_config_binds_xpois_metadata_and_layout(
    tmp_path: Path,
) -> None:
    config = _evidence_config(tmp_path)
    decode_device_pipeline_evidence(
        _scientific_evidence_payload(),
        config=config,
    )

    kernel_drift = _scientific_evidence_payload()
    kernel_drift["xpois"]["kernel_shape"] = [5, 5]
    decode_device_pipeline_evidence(kernel_drift)
    with pytest.raises(ValueError, match="kernel_shape does not match"):
        decode_device_pipeline_evidence(kernel_drift, config=config)

    flux_drift = _scientific_evidence_payload(flux_conserve=True)
    decode_device_pipeline_evidence(flux_drift)
    with pytest.raises(ValueError, match="flux_conserve does not match"):
        decode_device_pipeline_evidence(flux_drift, config=config)

    basis_drift = _scientific_evidence_payload()
    basis_drift["xpois"]["basis_terms"][0]["sigma"] = 0.9
    decode_device_pipeline_evidence(basis_drift)
    with pytest.raises(ValueError, match="basis terms do not match"):
        decode_device_pipeline_evidence(basis_drift, config=config)

    background_config = replace(
        config,
        xpois=replace(config.xpois, background_degree=1),
    )
    with pytest.raises(
        ValueError,
        match="background coefficients do not match",
    ):
        decode_device_pipeline_evidence(
            _scientific_evidence_payload(),
            config=background_config,
        )


def test_scientific_evidence_config_binds_flux_basis_order_and_reference(
    tmp_path: Path,
) -> None:
    xpois = DeviceXPOISPipelineConfig(
        kernel_shape=(3, 3),
        basis_sigmas=(0.8, 1.5),
        basis_degrees=(1, 0),
        background_degree=1,
        flux_conserve=True,
        flux_reference_index=3,
    )
    config = _evidence_config(tmp_path, xpois=xpois)
    basis_terms = [
        {
            "component_index": 1,
            "sigma": 1.5,
            "poly_u_degree": 0,
            "poly_v_degree": 0,
            "zero_sum": False,
        },
        {
            "component_index": 0,
            "sigma": 0.8,
            "poly_u_degree": 0,
            "poly_v_degree": 0,
            "zero_sum": True,
        },
        {
            "component_index": 0,
            "sigma": 0.8,
            "poly_u_degree": 0,
            "poly_v_degree": 1,
            "zero_sum": True,
        },
        {
            "component_index": 0,
            "sigma": 0.8,
            "poly_u_degree": 1,
            "poly_v_degree": 0,
            "zero_sum": True,
        },
    ]
    payload = _scientific_evidence_payload(
        basis_terms=basis_terms,
        background_count=3,
        flux_conserve=True,
    )
    decode_device_pipeline_evidence(payload, config=config)

    reordered = _scientific_evidence_payload(
        basis_terms=[
            basis_terms[0],
            basis_terms[2],
            basis_terms[1],
            basis_terms[3],
        ],
        background_count=3,
        flux_conserve=True,
    )
    decode_device_pipeline_evidence(reordered)
    with pytest.raises(ValueError, match="basis terms do not match"):
        decode_device_pipeline_evidence(reordered, config=config)


def test_scientific_evidence_config_rejects_rehashed_feature_drift(
    tmp_path: Path,
) -> None:
    config = _evidence_config(tmp_path)
    payload = _scientific_evidence_payload()
    _rewrite_evidence_segment(
        payload,
        "xfit.features",
        (0, FEATURE_NAMES.index("separation_over_stamp")),
        0.25,
    )

    decode_device_pipeline_evidence(payload)
    with pytest.raises(ValueError, match="canonical config transform"):
        decode_device_pipeline_evidence(payload, config=config)


def test_evidence_packing_preserves_float32_subnormals() -> None:
    cp, _torch = _gpu_modules()
    import cuphoton.xscan.device_pipeline as pipeline

    # Exercise the actual packed evidence boundary on its producer stream.
    # Both signed subnormals and the smallest normal must survive widening.
    bits = np.asarray(
        [1, 2, 0x007FFFFF, 0x00800000, 0x80000001], dtype=np.uint32
    )
    expected = np.zeros((1, len(FEATURE_NAMES)), dtype=np.float32)
    expected[0, : len(bits)] = bits.view(np.float32)
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        packed = pipeline._pack_scientific_evidence(
            cp=cp,
            xpois_result=_fake_xpois_result(cp, cp.zeros((3, 3)), chi2=0.0),
            xfit_result=_fake_xfit_result(cp, 1),
            features=SimpleNamespace(
                values=cp.asarray(expected), feature_names=FEATURE_NAMES
            ),
            candidate_count=1,
            kernel_shape=(3, 3),
            flux_conserve=False,
        )
        actual = pipeline._host_evidence_segment(
            cp.asnumpy(packed.values),
            layout=packed.layout,
            name="xfit.features",
        )
    np.testing.assert_array_equal(actual, expected.astype(np.float64))


def test_float64_evidence_bridge_retains_owner_through_consumption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cp, torch = _gpu_modules()
    import cuphoton.xscan.device_pipeline as pipeline

    device = torch.device("cuda:0")
    values = cp.arange(32, dtype=cp.float64)
    owner_ref = weakref.ref(values)
    original = torch.utils.dlpack.from_dlpack
    producer_refs: list[Any] = []

    def record_producer(producer: Any, **kwargs: Any) -> Any:
        producer_refs.append(weakref.ref(producer))
        assert kwargs == {"copy": False}
        return original(producer, **kwargs)

    monkeypatch.setattr(torch.utils.dlpack, "from_dlpack", record_producer)
    view = pipeline._cupy_float64_to_torch(
        values,
        device=device,
        cp=cp,
        torch=torch,
    )
    output = view.tensor + 1.0

    assert producer_refs[0]() is values
    assert view.tensor.data_ptr() == int(values.data.ptr)
    assert view._owner is values
    del values
    gc.collect()
    assert owner_ref() is not None

    torch.cuda.current_stream(device).synchronize()
    del view
    gc.collect()
    assert owner_ref() is None
    assert output.device == device


def test_float64_evidence_bridge_orders_nondefault_streams() -> None:
    cp, torch = _gpu_modules()
    import cuphoton.xscan.device_pipeline as pipeline

    device = torch.device("cuda:0")
    producer = cp.cuda.Stream(non_blocking=True)
    release = cp.cuda.Stream(non_blocking=True)
    consumer = torch.cuda.Stream(device=device)
    gate = cp.cuda.Event()
    values = cp.zeros(32, dtype=cp.float64)
    delay = cp.RawKernel(
        r"""
        extern "C" __global__ void delay(unsigned long long cycles) {
            unsigned long long start = clock64();
            while (clock64() - start < cycles) {}
        }
        """,
        "delay",
    )

    with release:
        delay((1,), (1,), (100_000_000,))
        gate.record()
    assert not gate.done

    with producer:
        producer.wait_event(gate)
        values.fill(7.0)
        with torch.cuda.stream(consumer):
            view = pipeline._cupy_float64_to_torch(
                values,
                device=device,
                cp=cp,
                torch=torch,
            )
            output = view.tensor.clone()

    consumer.synchronize()

    assert gate.done
    assert torch.equal(output, torch.full_like(output, 7.0))
    assert view.tensor.data_ptr() == int(values.data.ptr)


def test_compact_descriptors_are_strict_json_roundtrips(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "checkpoint.pt").write_bytes(b"checkpoint")
    reference_path = tmp_path / "reference.npy"
    target_path = tmp_path / "target.npy"
    np.save(reference_path, np.zeros((31, 31), dtype=np.float32))
    np.save(target_path, np.ones((31, 31), dtype=np.float64))
    config = _config(checkpoint_dir)
    item = _item(reference_path, target_path)

    config_payload = config.to_payload()
    item_payload = item.to_payload()
    json.dumps(config_payload, allow_nan=False)
    json.dumps(item_payload, allow_nan=False)

    assert config_payload["schema"] == DEVICE_PIPELINE_CONFIG_SCHEMA
    assert config_payload["inference_policy"] == {
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
    assert item_payload["schema"] == DEVICE_PIPELINE_ITEM_SCHEMA
    assert DevicePipelineConfig.from_payload(config_payload) == config
    assert DevicePipelineItem.from_payload(item_payload) == item
    assert config.configuration_sha256 == (
        DevicePipelineConfig.from_payload(config_payload).configuration_sha256
    )
    with pytest.raises(ValueError, match="unknown unexpected"):
        DevicePipelineItem.from_payload(
            {**item_payload, "unexpected": "large-argdata"}
        )
    with pytest.raises(ValueError, match="strict deterministic policy"):
        DevicePipelineConfig.from_payload(
            {
                **config_payload,
                "inference_policy": {
                    **config_payload["inference_policy"],
                    "allow_tf32": True,
                },
            }
        )
    for field, value in (
        ("item_id", 12),
        ("reference.path", 12),
        ("reference.sha256", 12),
        ("config.device", 12),
        ("config.feature_schema_path", 12),
    ):
        if field.startswith("reference."):
            descriptor_field = field.removeprefix("reference.")
            payload = {
                **item_payload,
                "reference": {
                    **item_payload["reference"],
                    descriptor_field: value,
                },
            }
            restore = DevicePipelineItem.from_payload
        elif field.startswith("config."):
            config_field = field.removeprefix("config.")
            payload = {**config_payload, config_field: value}
            restore = DevicePipelineConfig.from_payload
        else:
            payload = {**item_payload, field: value}
            restore = DevicePipelineItem.from_payload
        with pytest.raises(
            (TypeError, ValueError), match=field.split(".")[-1]
        ):
            restore(payload)


@pytest.mark.parametrize(
    ("model_config", "message"),
    [
        (
            {
                "input_mode": "pair",
                "image_size": 11,
                "xfit_feature_names": list(FEATURE_NAMES),
            },
            "triplet",
        ),
        (
            {
                "input_mode": "triplet",
                "image_size": 13,
                "xfit_feature_names": list(FEATURE_NAMES),
            },
            "image_size",
        ),
        (
            {
                "input_mode": "triplet",
                "image_size": 11,
                "xfit_feature_names": list(reversed(FEATURE_NAMES)),
            },
            "canonical ordered",
        ),
    ],
)
def test_checkpoint_preflight_binds_triplet_shape_and_feature_order(
    model_config: dict[str, Any],
    message: str,
) -> None:
    feature_schema = {
        "artifacts": {"features": {"sha256": "b" * 64}},
        "source_artifacts": {"summary": {"sha256": "a" * 64}},
    }
    with pytest.raises(ValueError, match=message):
        _validate_checkpoint_contract(
            {
                "model_config": model_config,
                "xfit_feature_bundle": {
                    "schema_sha256": "c" * 64,
                    "feature_sha256": "b" * 64,
                    "source_artifacts": feature_schema["source_artifacts"],
                },
            },
            stamp_shape=(11, 11),
            feature_names=FEATURE_NAMES,
            feature_schema_sha256="c" * 64,
            feature_schema=feature_schema,
        )


@pytest.mark.parametrize(
    ("bundle_update", "schema_update", "message"),
    [
        ({"schema_sha256": "d" * 64}, {}, "schema identities"),
        ({"feature_sha256": "d" * 64}, {}, "feature identity"),
        (
            {"source_artifacts": {"summary": {"sha256": "d" * 64}}},
            {},
            "source artifacts",
        ),
        (
            {"unexpected": "identity"},
            {},
            "fields are invalid",
        ),
        ({}, {"artifacts": None}, "artifact metadata"),
        (
            {},
            {"source_artifacts": None},
            "source artifact identities",
        ),
    ],
)
def test_checkpoint_preflight_requires_exact_feature_bundle_identity(
    tmp_path: Path,
    bundle_update: dict[str, Any],
    schema_update: dict[str, Any],
    message: str,
) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "checkpoint.pt").write_bytes(b"checkpoint")
    config = _config(checkpoint_dir)
    feature_schema = {**_feature_schema(config), **schema_update}
    checkpoint = _checkpoint_contract(config)
    checkpoint["xfit_feature_bundle"] = {
        **checkpoint["xfit_feature_bundle"],
        **bundle_update,
    }

    with pytest.raises(ValueError, match=message):
        _validate_checkpoint_contract(
            checkpoint,
            stamp_shape=config.stamp_shape,
            feature_names=FEATURE_NAMES,
            feature_schema_sha256=config.feature_schema_sha256,
            feature_schema=feature_schema,
        )


@pytest.mark.parametrize(
    ("schema_update", "message"),
    [
        ({"feature_names": list(reversed(FEATURE_NAMES))}, "feature names"),
        ({"feature_dtype": "float64"}, "float32"),
        ({"source": None}, "source metadata"),
        ({"source": {"model": "stamp"}}, "source.model"),
        ({"source": {"mode": "split"}}, "source.mode"),
        ({"source": {"image_shape": [13, 13]}}, "source.image_shape"),
        ({"source": {"mask_present": True}}, "source.mask_present"),
        (
            {"source": {"variance_present": True}},
            "source.variance_present",
        ),
    ],
)
def test_feature_schema_preflight_binds_live_transform_contract(
    tmp_path: Path,
    schema_update: dict[str, Any],
    message: str,
) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "checkpoint.pt").write_bytes(b"checkpoint")
    config = _config(checkpoint_dir)
    feature_schema = _feature_schema(config)
    if "source" in schema_update and isinstance(
        schema_update["source"], dict
    ):
        feature_schema["source"] = {
            **feature_schema["source"],
            **schema_update["source"],
        }
    else:
        feature_schema.update(schema_update)

    with pytest.raises(ValueError, match=message):
        _validate_feature_schema_source(
            feature_schema,
            config=config,
            feature_names=FEATURE_NAMES,
        )


def test_candidate_preflight_includes_stamp_and_kernel_margin(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "checkpoint.pt").write_bytes(b"checkpoint")
    path = tmp_path / "image.npy"
    np.save(path, np.zeros((31, 31), dtype=np.float64))
    item = DevicePipelineItem(
        item_id="edge",
        reference=_descriptor(path),
        target=_descriptor(path),
        candidates=(
            DevicePipelineCandidate(
                candidate_id=0,
                center_x=5,
                center_y=15,
                source_index=0,
            ),
        ),
    )
    from cuphoton.xscan.device_pipeline import _validate_item_contract

    with pytest.raises(ValueError, match="kernel edge"):
        _validate_item_contract(item, config=_config(checkpoint_dir))


def test_persistent_context_pipeline_is_device_resident_and_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cp, torch = _gpu_modules()
    import cuphoton.xscan.device_pipeline as pipeline
    import cuphoton.xscan.training as training
    from cuphoton.xscan.config import PerformanceConfig

    class TinyTripletModel(torch.nn.Module):
        input_mode = "triplet"
        xfit_feature_names = FEATURE_NAMES

        def __init__(self) -> None:
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(0.125))

        def forward(
            self,
            images: Any,
            *,
            xfit_features: Any,
        ) -> Any:
            assert torch.cuda.current_stream(images.device) == (
                context.consumer_stream
            )
            assert images.dtype == torch.float32
            assert images.is_contiguous()
            assert tuple(images.shape[1:]) == (3, 11, 11)
            assert xfit_features.dtype == torch.float32
            assert xfit_features.is_contiguous()
            assert tuple(xfit_features.shape[1:]) == (len(FEATURE_NAMES),)
            stage_calls.append("predict")
            return (
                images[:, 0, 5, 5]
                + 2.0 * images[:, 1, 5, 5]
                + 3.0 * images[:, 2, 5, 5]
                + 0.01 * xfit_features[:, 0]
                + self.bias
            )

    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "checkpoint.pt").write_bytes(b"checkpoint")
    config = _config(checkpoint_dir)
    load_calls: list[tuple[Path, Any, PerformanceConfig]] = []

    def load_model(
        run_dir: Path,
        *,
        device: Any,
        performance_override: PerformanceConfig,
    ) -> tuple[Any, Any, Any]:
        load_calls.append((run_dir, device, performance_override))
        model = TinyTripletModel().to(device)
        checkpoint = _checkpoint_contract(config)
        return model, checkpoint, performance_override

    monkeypatch.setattr(training, "load_model_from_checkpoint", load_model)
    monkeypatch.setattr(
        pipeline,
        "_load_feature_schema_contract",
        lambda *_args, **_kwargs: _feature_schema(config),
    )
    context = DeviceWorkerContext.initialize(config)
    assert len(load_calls) == 1
    assert load_calls[0][:2] == (checkpoint_dir, torch.device("cuda:0"))
    assert load_calls[0][2] == PerformanceConfig()
    assert context.feature_variance_present is False
    with pytest.raises(TypeError, match="cannot be pickled"):
        pickle.dumps(context, protocol=pickle.HIGHEST_PROTOCOL)

    rng = np.random.default_rng(19)
    reference = rng.normal(0.0, 0.1, (31, 31)).astype(np.float32)
    target = reference.astype(np.float64)
    target[17:24, 17:24] += 0.5
    reference_path = tmp_path / "reference.npy"
    target_path = tmp_path / "target.npy"
    variance_path = tmp_path / "variance.npy"
    np.save(reference_path, reference)
    np.save(target_path, target)
    np.save(variance_path, np.full((31, 31), 2.0, dtype=np.float64))
    item = _item(
        reference_path,
        target_path,
        variance_path=variance_path,
    )

    stage_calls: list[str] = []
    pointer_pairs: list[tuple[int, int]] = []

    def solve(
        reference: Any, target: Any, _components: Any, **_kwargs: Any
    ) -> Any:
        assert int(cp.cuda.get_current_stream().ptr) == int(
            context.producer_stream.ptr
        )
        assert reference.dtype == cp.float64
        assert target.dtype == cp.float64
        assert _kwargs["variance"].dtype == cp.float64
        assert tuple(_kwargs["variance"].shape) == (31, 31)
        stage_calls.append("xpois")
        return _fake_xpois_result(cp, target - reference, chi2=2.5)

    def fit(images: Any, **kwargs: Any) -> Any:
        assert int(cp.cuda.get_current_stream().ptr) == int(
            context.producer_stream.ptr
        )
        assert images.dtype == cp.float64
        assert images.flags.c_contiguous
        assert tuple(images.shape) == (2, 11, 11)
        assert kwargs == {
            "model": "gaussian",
            "mode": "difference",
            "config": context.solver_config,
        }
        stage_calls.append("xfit")
        return _fake_xfit_result(cp, 2)

    def transform(result: Any, **kwargs: Any) -> Any:
        assert result is not None
        assert int(cp.cuda.get_current_stream().ptr) == int(
            context.producer_stream.ptr
        )
        assert kwargs == {
            "image_shape": (11, 11),
            "variance_present": context.feature_variance_present,
        }
        stage_calls.append("features")
        return _fake_features(cp, 2)

    def bridge(values: Any, *, device: Any) -> Any:
        view = training._cupy_to_torch(values, device=device)
        pointer_pairs.append((int(values.data.ptr), view.tensor.data_ptr()))
        return view

    functions = _PipelineFunctions(
        gaussian_basis_component=lambda **values: values,
        solve_constant_kernel_device=solve,
        fit_dipoles_device=fit,
        transform_xfit_result_features_device=transform,
        cupy_to_torch=bridge,
        predict_tensors=training.predict_tensors,
    )
    monkeypatch.setattr(
        pipeline, "_load_pipeline_functions", lambda: functions
    )
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda *_args, **_kwargs: pytest.fail("device-wide synchronize"),
    )
    original_cpu = torch.Tensor.cpu
    cpu_calls: list[tuple[int, ...]] = []

    def one_terminal_cpu(values: Any, *args: Any, **kwargs: Any) -> Any:
        cpu_calls.append(tuple(values.shape))
        return original_cpu(values, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", one_terminal_cpu)
    original_decode = pipeline.decode_device_pipeline_evidence
    validated_receipts: list[dict[str, np.ndarray]] = []

    def record_validation(
        evidence: Any,
        *,
        config: DevicePipelineConfig | None = None,
    ) -> dict[str, np.ndarray]:
        assert config is context.config
        decoded = original_decode(evidence, config=config)
        validated_receipts.append(decoded)
        return decoded

    monkeypatch.setattr(
        pipeline,
        "decode_device_pipeline_evidence",
        record_validation,
    )

    first = run_device_pipeline_item(item, context)
    second = run_device_pipeline_item(item, context)

    assert len(load_calls) == 1
    assert stage_calls == [
        "xpois",
        "xfit",
        "features",
        "predict",
        "xpois",
        "xfit",
        "features",
        "predict",
    ]
    assert all(producer == consumer for producer, consumer in pointer_pairs)
    assert len(pointer_pairs) == 4
    assert cpu_calls == [(228,), (228,)]
    assert len(validated_receipts) == 2
    assert tuple(value.candidate_id for value in first.predictions) == (
        "candidate-b",
        17,
    )
    assert tuple(value.source_index for value in first.predictions) == (9, 3)
    expected_logits = np.asarray(
        [
            3.0 * reference[20, 20] + 2.135,
            3.0 * reference[12, 12] + 0.135,
        ],
        dtype=np.float32,
    )
    assert np.allclose(
        [value.logit for value in first.predictions],
        expected_logits,
        rtol=0.0,
        atol=1.0e-6,
    )
    assert np.allclose(
        [value.probability for value in first.predictions],
        1.0 / (1.0 + np.exp(-expected_logits)),
        rtol=0.0,
        atol=1.0e-6,
    )
    assert first.result_sha256 == second.result_sha256
    assert first.transfers.input_h2d_bytes == 3 * 31 * 31 * 8
    assert first.transfers.terminal_d2h_bytes == 228 * 8
    assert first.transfers.dlpack_shared_bytes == (
        2 * 3 * 11 * 11 * 4 + 2 * len(FEATURE_NAMES) * 4 + 224 * 8
    )
    assert first.transfers.pipeline_full_array_d2h_bytes == 0
    assert first.transfers.pipeline_compact_h2d_bytes == 0
    assert first.transfers.terminal_d2h_calls == 1
    payload = first.to_payload()
    json.dumps(payload, allow_nan=False)
    assert item.variance is not None
    assert payload["input_sha256"] == {
        "reference": item.reference.sha256,
        "target": item.target.sha256,
        "variance": item.variance.sha256,
    }
    assert payload["configuration_sha256"] == config.configuration_sha256
    assert payload["checkpoint_sha256"] == config.checkpoint_sha256
    assert payload["feature_schema_sha256"] == (config.feature_schema_sha256)
    evidence = payload["scientific_evidence"]
    assert evidence["packed_element_count"] == 228
    assert evidence["packed_byte_count"] == 228 * 8
    assert evidence["parameter_names"] == list(_GAUSSIAN_PARAMETER_NAMES)
    assert evidence["feature_names"] == list(FEATURE_NAMES)
    assert evidence["xfit"]["backend"] == "cupy"
    assert evidence["xpois"]["kernel_shape"] == [3, 3]
    decoded = decode_device_pipeline_evidence(evidence)
    assert set(decoded) == {
        "xpois.kernel_coefficients",
        "xpois.background_coefficients",
        *(
            f"xfit.{name}"
            for name in (
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
        ),
        "xfit.parameters",
        "xfit.standard_errors",
        "xfit.covariance",
        "xfit.features",
        "xscan.logits",
        "xscan.probabilities",
    }
    assert np.isnan(decoded["xfit.covariance"][0, 0, 0])
    assert np.isnan(decoded["xfit.standard_errors"][0, 0])
    variance_index = FEATURE_NAMES.index("variance_weighted")
    assert np.all(decoded["xfit.features"][:, variance_index] == 0.0)
    assert np.allclose(decoded["xscan.logits"], expected_logits)

    original_pack = pipeline._pack_scientific_evidence

    def pack_with_wrong_kernel_shape(**kwargs: Any) -> Any:
        return replace(
            original_pack(**kwargs),
            xpois_kernel_shape=(5, 5),
        )

    monkeypatch.setattr(
        pipeline,
        "_pack_scientific_evidence",
        pack_with_wrong_kernel_shape,
    )
    with pytest.raises(ValueError, match="kernel_shape does not match"):
        run_device_pipeline_item(item, context)
    assert cpu_calls[-1] == (228,)
    assert len(cpu_calls) == 3
    assert len(validated_receipts) == 2


def test_pipeline_failure_completes_device_streams_and_releases_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cp, torch = _gpu_modules()
    import cuphoton.xscan.device_pipeline as pipeline
    import cuphoton.xscan.training as training
    from cuphoton.xscan.config import PerformanceConfig

    class Model(torch.nn.Module):
        input_mode = "triplet"
        xfit_feature_names = FEATURE_NAMES

        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))

        def forward(self, images: Any, *, xfit_features: Any) -> Any:
            return images.mean(dim=(1, 2, 3)) * self.weight

    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "checkpoint.pt").write_bytes(b"checkpoint")
    config = _config(checkpoint_dir)

    def load_model(
        run_dir: Path,
        *,
        device: Any,
        performance_override: PerformanceConfig,
    ) -> tuple[Any, Any, Any]:
        del run_dir
        return (
            Model().to(device),
            _checkpoint_contract(config),
            performance_override,
        )

    monkeypatch.setattr(training, "load_model_from_checkpoint", load_model)
    monkeypatch.setattr(
        pipeline,
        "_load_feature_schema_contract",
        lambda *_args, **_kwargs: _feature_schema(config),
    )
    context = DeviceWorkerContext.initialize(config)
    path = tmp_path / "image.npy"
    np.save(path, np.zeros((31, 31), dtype=np.float64))
    item = DevicePipelineItem(
        item_id="failure",
        reference=_descriptor(path),
        target=_descriptor(path),
        candidates=(
            DevicePipelineCandidate(
                candidate_id=1,
                center_x=15,
                center_y=15,
                source_index=0,
            ),
        ),
    )

    def solve(
        reference: Any, target: Any, *_args: Any, **_kwargs: Any
    ) -> Any:
        return _fake_xpois_result(cp, target - reference, chi2=0.0)

    features = _fake_features(cp, 1)

    def fail_after_handoff(**_kwargs: Any) -> Any:
        torch.ones(1, device=context.device).add_(1)
        raise RuntimeError("model failure")

    functions = _PipelineFunctions(
        gaussian_basis_component=lambda **values: values,
        solve_constant_kernel_device=solve,
        fit_dipoles_device=lambda *_args, **_kwargs: _fake_xfit_result(cp, 1),
        transform_xfit_result_features_device=(
            lambda *_args, **_kwargs: features
        ),
        cupy_to_torch=training._cupy_to_torch,
        predict_tensors=fail_after_handoff,
    )
    monkeypatch.setattr(
        pipeline, "_load_pipeline_functions", lambda: functions
    )

    with pytest.raises(RuntimeError, match="model failure"):
        run_device_pipeline_item(item, context)

    assert context.producer_stream.done
    assert context.consumer_stream.query()
    assert context._poisoned_reason is None
    assert context._item_lock.acquire(blocking=False)
    context._item_lock.release()

    def fail_on_producer(
        reference: Any, target: Any, *_args: Any, **_kwargs: Any
    ) -> Any:
        del target
        cp.add(reference, 1.0, out=reference)
        raise RuntimeError("producer failure")

    producer_failure_functions = _PipelineFunctions(
        gaussian_basis_component=lambda **values: values,
        solve_constant_kernel_device=fail_on_producer,
        fit_dipoles_device=lambda *_args, **_kwargs: _fake_xfit_result(cp, 1),
        transform_xfit_result_features_device=(
            lambda *_args, **_kwargs: features
        ),
        cupy_to_torch=training._cupy_to_torch,
        predict_tensors=fail_after_handoff,
    )
    monkeypatch.setattr(
        pipeline,
        "_load_pipeline_functions",
        lambda: producer_failure_functions,
    )
    with pytest.raises(RuntimeError, match="producer failure"):
        run_device_pipeline_item(item, context)

    assert context.producer_stream.done
    assert context.consumer_stream.query()
    assert context._poisoned_reason is None
    assert context._item_lock.acquire(blocking=False)
    context._item_lock.release()


def test_failed_cleanup_poisons_context_and_attempts_both_streams() -> None:
    import threading

    import cuphoton.xscan.device_pipeline as pipeline

    attempted = []

    def fail_producer() -> None:
        attempted.append("producer")
        raise RuntimeError("stream failed")

    context = object.__new__(DeviceWorkerContext)
    context._item_lock = threading.Lock()
    context._poisoned_reason = None
    context.producer_stream = SimpleNamespace(synchronize=fail_producer)
    context.consumer_stream = SimpleNamespace(
        synchronize=lambda: attempted.append("consumer")
    )
    original_error = ValueError("item failed")
    pipeline._complete_failed_device_work(context, original_error)
    assert attempted == ["producer", "consumer"]
    assert "stream failed" in original_error.__notes__[0]
    with pytest.raises(RuntimeError, match="unusable after failed cleanup"):
        with context._claim_item("next-item"):
            pytest.fail("poisoned context accepted work")
    assert context._item_lock.acquire(blocking=False)
    context._item_lock.release()


def test_changed_input_fails_before_device_ingress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "checkpoint.pt").write_bytes(b"checkpoint")
    path = tmp_path / "image.npy"
    np.save(path, np.zeros((31, 31), dtype=np.float64))
    descriptor = _descriptor(path)
    item = DevicePipelineItem(
        item_id="changed",
        reference=descriptor,
        target=descriptor,
        candidates=(
            DevicePipelineCandidate(
                candidate_id=0,
                center_x=15,
                center_y=15,
                source_index=0,
            ),
        ),
    )
    np.save(path, np.ones((31, 31), dtype=np.float64))

    from cuphoton.xscan.device_pipeline import _read_item_inputs

    monkeypatch.setattr(
        np,
        "load",
        lambda *_args, **_kwargs: pytest.fail("changed input was read"),
    )
    with pytest.raises(RuntimeError, match="changed before execution"):
        _read_item_inputs(item)


def test_xfit_config_rejects_values_the_solver_cannot_use() -> None:
    with pytest.raises(ValueError, match="model"):
        DeviceXFitPipelineConfig(model="stamp")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="mode"):
        DeviceXFitPipelineConfig(mode="split")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="damping_increase"):
        DeviceXFitPipelineConfig(damping_increase=0.5)
    with pytest.raises(ValueError, match="damping_decrease"):
        DeviceXFitPipelineConfig(damping_decrease=1.0)
    with pytest.raises(ValueError, match="finite_difference_step"):
        DeviceXFitPipelineConfig(finite_difference_step=0.0)


def test_loaded_model_contract_matches_checkpoint_contract() -> None:
    with pytest.raises(ValueError, match="triplet"):
        _validate_loaded_model_contract(
            SimpleNamespace(
                input_mode="pair",
                xfit_feature_names=FEATURE_NAMES,
            ),
            feature_names=FEATURE_NAMES,
        )
    with pytest.raises(ValueError, match="canonical ordered"):
        _validate_loaded_model_contract(
            SimpleNamespace(
                input_mode="triplet",
                xfit_feature_names=tuple(reversed(FEATURE_NAMES)),
            ),
            feature_names=FEATURE_NAMES,
        )
