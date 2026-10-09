# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Native execution choices retain the device pipeline's strict contracts."""

from __future__ import annotations

from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest

from cuphoton.xfit import DipoleFitUncertaintyReason, LMConfig, LMStatus
from cuphoton.xscan import device_pipeline as pipeline
from cuphoton.xscan.device_pipeline import (
    DevicePipelineConfig,
    DeviceXFitPipelineConfig,
    DeviceXPOISPipelineConfig,
)


def _config(tmp_path, xfit=None):
    return DevicePipelineConfig(
        device="cuda:0",
        checkpoint_dir=str(tmp_path),
        checkpoint_sha256="a" * 64,
        feature_schema_path=str(tmp_path / "features.json"),
        feature_schema_sha256="b" * 64,
        stamp_shape=(11, 11),
        decision_threshold=0.5,
        xpois=DeviceXPOISPipelineConfig(
            kernel_shape=(3, 3), basis_sigmas=(0.8,), basis_degrees=(0,)
        ),
        xfit=DeviceXFitPipelineConfig() if xfit is None else xfit,
    )


def test_default_execution_preserves_legacy_payload_and_hash(tmp_path):
    config = _config(tmp_path)
    legacy = config.to_payload()
    legacy_xfit = asdict(config.xfit)
    del legacy_xfit["backend"]
    del legacy_xfit["fusion"]
    legacy["xfit"] = legacy_xfit

    restored = DevicePipelineConfig.from_payload(legacy)
    assert restored.to_payload() == legacy
    assert restored.configuration_sha256 == pipeline._stable_payload_sha256(
        legacy
    )
    assert restored.xfit.execution_payload() == {}

    explicit = dict(legacy_xfit, backend="cupy")
    assert DeviceXFitPipelineConfig.from_payload(explicit) == restored.xfit


def test_native_execution_roundtrip_changes_identity_not_lm_settings(
    tmp_path,
):
    baseline = _config(tmp_path)
    native = replace(
        baseline,
        xfit=replace(baseline.xfit, backend="native"),
    )
    restored = DevicePipelineConfig.from_payload(native.to_payload())
    assert restored == native
    assert native.configuration_sha256 != baseline.configuration_sha256
    assert native.xfit.execution_payload() == {
        "backend": "native",
    }
    assert native.xfit.solver_payload() == baseline.xfit.solver_payload()
    assert LMConfig(**native.xfit.solver_payload()) == LMConfig()


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({"backend": "numpy"}, ValueError, "backend"),
        (
            {"backend": "native", "use_finite_difference": True},
            ValueError,
            "finite differences",
        ),
    ],
)
def test_invalid_execution_is_rejected_in_config(options, error, message):
    with pytest.raises(error, match=message):
        DeviceXFitPipelineConfig(**options)


@pytest.mark.parametrize("backend", ["cupy", "native"])
def test_stage_forwards_execution_and_reports_actual_backend(
    tmp_path, monkeypatch, backend
):
    pytest.importorskip("torch")
    import cuphoton.xfit as xfit
    from cuphoton.xscan import xfit_features
    from cuphoton.xscan.pipeline_benchmark import stages
    from cuphoton.xscan.xfit_features import FEATURE_NAMES

    config = _config(
        tmp_path,
        DeviceXFitPipelineConfig(backend=backend),
    )
    captured = {}
    result = SimpleNamespace(
        **{name: np.zeros(1) for name in stages.XFIT_FIELDS},
        parameter_names=("amplitude",),
        backend=backend,
        solver="levenberg-marquardt",
        model="gaussian",
        mode="difference",
        dtype="float64",
    )

    def fit(images, **kwargs):
        captured.update(kwargs)
        return result

    monkeypatch.setattr(xfit, "fit_dipoles_device", fit)
    monkeypatch.setattr(
        xfit_features,
        "transform_xfit_result_features_device",
        lambda *_, **__: SimpleNamespace(
            values=np.zeros((1, len(FEATURE_NAMES))),
            feature_names=FEATURE_NAMES,
        ),
    )
    monkeypatch.setattr(
        stages, "_read_array", lambda *args: np.zeros((1, 11, 11))
    )
    cp = SimpleNamespace(
        asarray=np.asarray,
        asnumpy=np.asarray,
        cuda=SimpleNamespace(
            Device=lambda: SimpleNamespace(synchronize=lambda: None)
        ),
    )
    _, metadata, _ = stages._run_item(
        "xfit",
        None,
        config,
        {"cp": cp, "torch": None},
        tmp_path,
        {"xpois": {"items": [{"artifacts": {"difference": {}}}]}},
        0,
    )
    assert metadata["backend"] == backend
    assert captured["config"] == LMConfig()
    if backend == "native":
        assert captured["backend"] == "native"
    else:
        assert "backend" not in captured


def _evidence_metadata(backend):
    from cuphoton.xscan.xfit_features import FEATURE_NAMES

    layout = [None] * (len(pipeline._XFIT_SCALAR_EVIDENCE_FIELDS) + 8)
    layout[1] = {
        "name": "xpois.background_coefficients",
        "offset": 1,
        "shape": [1],
        "source_dtype": "<f8",
    }
    return {
        "parameter_names": list(pipeline._GAUSSIAN_PARAMETER_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "candidate_count": 1,
        "layout": layout,
        "xfit": {
            "schema": "cuphoton.xfit.device-fit-result/v1",
            "backend": backend,
            "solver": "levenberg-marquardt",
            "model": "gaussian",
            "mode": "difference",
            "dtype": "float64",
            "status_code_names": {str(int(s)): s.name for s in LMStatus},
            "uncertainty_reason_code_names": {
                str(int(s)): s.name for s in DipoleFitUncertaintyReason
            },
        },
        "xpois": {
            "schema": "cuphoton.xpois.device-fit-result/v1",
            "backend": "cupy",
            "solver": "constant",
            "kernel_shape": [3, 3],
            "flux_conserve": True,
            "basis_terms": [
                {
                    "component_index": 0,
                    "sigma": 0.8,
                    "poly_u_degree": 0,
                    "poly_v_degree": 0,
                    "zero_sum": False,
                }
            ],
        },
    }


def test_native_evidence_preserves_layout_and_remaining_constraints(tmp_path):
    pytest.importorskip("torch")
    native = _evidence_metadata("native")
    assert pipeline._device_pipeline_evidence_layout_contract(native) == (
        pipeline._device_pipeline_evidence_layout_contract(
            _evidence_metadata("cupy")
        )
    )
    with pytest.raises(ValueError, match="backend does not match config"):
        pipeline._validate_device_pipeline_evidence_config(
            native, {}, config=_config(tmp_path)
        )
    native["xfit"]["dtype"] = "float32"
    with pytest.raises(ValueError, match="xfit.dtype"):
        pipeline._device_pipeline_evidence_layout_contract(native)
    with pytest.raises(ValueError, match="xfit.backend"):
        pipeline._device_pipeline_evidence_layout_contract(
            _evidence_metadata("numpy")
        )
