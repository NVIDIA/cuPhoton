# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

import cuphoton.xfit as xfit
import cuphoton.xscan.device_pipeline as pipeline
import cuphoton.xscan.xfit_features as features
from cuphoton.xscan.pipeline_benchmark import stages


def _config(tmp_path, *, backend="cupy"):
    return pipeline.DevicePipelineConfig(
        device="cuda:0",
        checkpoint_dir=str(tmp_path),
        checkpoint_sha256="a" * 64,
        feature_schema_path=str(tmp_path / "schema.json"),
        feature_schema_sha256="b" * 64,
        stamp_shape=(3, 3),
        decision_threshold=0.5,
        xpois=pipeline.DeviceXPOISPipelineConfig(
            kernel_shape=(3, 3), basis_sigmas=(0.8,), basis_degrees=(0,)
        ),
        xfit=pipeline.DeviceXFitPipelineConfig(backend=backend),
    )


def test_pipeline_backend_payload_preserves_legacy_default(tmp_path):
    default = _config(tmp_path)
    legacy = default.to_payload()
    assert "backend" not in legacy["xfit"]
    restored = pipeline.DevicePipelineConfig.from_payload(legacy)
    assert restored.xfit.backend == "cupy"
    assert restored.configuration_sha256 == default.configuration_sha256
    legacy["xfit"]["backend"] = "cupy"
    assert (
        pipeline.DevicePipelineConfig.from_payload(
            legacy
        ).configuration_sha256
        == default.configuration_sha256
    )
    selected = _config(tmp_path, backend="numba-cuda-mlir")
    payload = selected.to_payload()
    assert payload["xfit"]["backend"] == "numba-cuda-mlir"
    assert pipeline.DevicePipelineConfig.from_payload(payload) == selected
    assert selected.configuration_sha256 != default.configuration_sha256
    assert "backend" not in selected.xfit.solver_payload()
    assert xfit.LMConfig(**selected.xfit.solver_payload()) == xfit.LMConfig()
    with pytest.raises(ValueError, match="xfit backend"):
        pipeline.DeviceXFitPipelineConfig(backend="numba-cuda")
    with pytest.raises(ValueError, match="finite differences"):
        pipeline.DeviceXFitPipelineConfig(
            backend="numba-cuda-mlir", use_finite_difference=True
        )


@pytest.mark.parametrize("backend", ["cupy", "numba-cuda-mlir"])
def test_pipeline_and_separated_stage_forward_backend(
    tmp_path, monkeypatch, backend
):
    config = _config(tmp_path, backend=backend)
    descriptor = pipeline.NpyArrayDescriptor(
        path=str(tmp_path / "image.npy"),
        sha256="c" * 64,
        shape=(5, 5),
        dtype="float64",
    )
    item = pipeline.DevicePipelineItem(
        item_id="pair",
        reference=descriptor,
        target=descriptor,
        candidates=(pipeline.DevicePipelineCandidate("candidate", 2, 2, 0),),
    )
    difference = np.zeros((1, 3, 3))
    calls = []

    class ReachedFit(Exception):
        pass

    def fit(images, **kwargs):
        assert images is difference
        assert kwargs.get("backend", "cupy") == backend
        assert kwargs["model"] == "gaussian"
        assert kwargs["mode"] == "difference"
        assert kwargs["config"] == xfit.LMConfig()
        calls.append(kwargs)
        raise ReachedFit

    class Stream(nullcontext):
        def synchronize(self):
            pass

    cp = SimpleNamespace(
        asarray=np.asarray,
        cuda=SimpleNamespace(Device=lambda: Stream()),
    )
    context = pipeline.DeviceWorkerContext(
        config=config,
        cp=cp,
        torch=None,
        model=None,
        performance=None,
        feature_variance_present=False,
        device="cuda:0",
        producer_stream=Stream(),
        consumer_stream=Stream(),
        solver_config=xfit.LMConfig(),
        load_seconds=0,
    )
    monkeypatch.setattr(
        pipeline.DeviceWorkerContext,
        "_validate_active_device",
        lambda self: None,
    )
    monkeypatch.setattr(
        pipeline, "_validate_item_contract", lambda *a, **k: None
    )
    monkeypatch.setattr(
        pipeline,
        "_read_item_inputs",
        lambda item: {
            "reference": np.zeros((5, 5)),
            "target": np.zeros((5, 5)),
        },
    )
    monkeypatch.setattr(
        pipeline, "_read_fits_device_inputs", lambda *a, **k: ({}, ())
    )
    monkeypatch.setattr(
        pipeline, "_extract_stamps", lambda **k: (None, difference)
    )
    monkeypatch.setattr(
        pipeline,
        "_load_pipeline_functions",
        lambda: SimpleNamespace(
            gaussian_basis_component=lambda **kwargs: None,
            solve_constant_kernel_device=lambda *a, **k: SimpleNamespace(
                residual=None
            ),
            fit_dipoles_device=fit,
        ),
    )
    with pytest.raises(ReachedFit):
        pipeline.run_device_pipeline_item(item, context)

    monkeypatch.setattr(xfit, "fit_dipoles_device", fit)
    monkeypatch.setattr(stages, "_read_array", lambda *args: difference)
    priors = {"xpois": {"items": [{"artifacts": {"difference": None}}]}}
    with pytest.raises(ReachedFit):
        stages._run_item(
            "xfit",
            item,
            config,
            {"cp": cp, "torch": None},
            tmp_path,
            priors,
            0,
        )
    assert len(calls) == 2


@pytest.mark.parametrize("requested", ["cupy", "numba-cuda-mlir"])
def test_pipeline_evidence_rejects_backend_different_from_config(
    tmp_path, requested
):
    actual = "cupy" if requested == "numba-cuda-mlir" else "numba-cuda-mlir"
    result = SimpleNamespace(
        parameter_names=xfit.GaussianDipoleModel.parameter_names,
        backend=actual,
        solver="levenberg-marquardt",
        model="gaussian",
        mode="difference",
        dtype="float64",
    )
    with pytest.raises(ValueError, match="xFit evidence contract"):
        pipeline._pack_scientific_evidence(
            cp=None,
            xpois_result=None,
            xfit_result=result,
            features=SimpleNamespace(feature_names=features.FEATURE_NAMES),
            candidate_count=1,
            kernel_shape=(3, 3),
            flux_conserve=False,
            xfit_backend=requested,
        )
    with pytest.raises(
        ValueError, match="xFit backend does not match config"
    ):
        pipeline._validate_device_pipeline_evidence_config(
            {"xfit": {"backend": actual}},
            {},
            config=_config(tmp_path, backend=requested),
        )


def test_numba_pipeline_feature_source_preserves_array_checks():
    class DeviceArray(np.ndarray):
        @property
        def device(self):
            return SimpleNamespace(id=0)

    def array(shape, dtype="float64"):
        return np.zeros(shape, dtype=dtype).view(DeviceArray)

    result = SimpleNamespace(
        schema="cuphoton.xfit.device-fit-result/v1",
        backend="numba-cuda-mlir",
        solver="levenberg-marquardt",
        result_location="device",
        model="gaussian",
        mode="difference",
        dtype="float64",
        device_id=0,
        parameter_names=xfit.GaussianDipoleModel.parameter_names,
        parameters=array((1, 8)),
        standard_errors=array((1, 8)),
        covariance=array((1, 8, 8)),
        converged=array((1,), bool),
        uncertainty_valid=array((1,), bool),
        degrees_of_freedom=array((1,), "int64"),
        valid_pixel_fraction=array((1,)),
        fractional_null_improvement=array((1,)),
        delta_chi_square=array((1,)),
        reduced_chi_square=array((1,)),
    )
    cp = SimpleNamespace(ndarray=DeviceArray)
    names = features._validate_device_feature_source(
        result, cp=cp, active_device_id=0
    )
    assert names == xfit.GaussianDipoleModel.parameter_names
    with pytest.raises(ValueError, match="active device is 1"):
        features._validate_device_feature_source(
            result, cp=cp, active_device_id=1
        )
    result.parameters = np.zeros((1, 8))
    with pytest.raises(TypeError, match="parameters must be a CuPy array"):
        features._validate_device_feature_source(
            result, cp=cp, active_device_id=0
        )


def test_numba_pipeline_cli_runs_real_stages_and_matches_cupy(
    tmp_path, capsys
):
    import json
    from dataclasses import replace

    from cuphoton.core.cli import run_component
    from cuphoton.xscan.pipeline_benchmark.fixture import prepare_fixture
    from cuphoton.xscan.pipeline_benchmark.runner import read_inputs

    cp = pytest.importorskip("cupy")
    torch = pytest.importorskip("torch")
    pytest.importorskip("numba_cuda_mlir.cuda")
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as exc:
        if exc.status != 100:
            raise
        pytest.skip("CUDA device is unavailable")
    if count < 1 or not torch.cuda.is_available():
        pytest.skip("CUDA device is unavailable")

    config_path, items_path = prepare_fixture(
        tmp_path / "inputs",
        images=1,
        image_size=48,
        candidates=2,
        stamp_size=9,
        seed=2026,
        device="cuda:0",
    )
    config, items = read_inputs(config_path, items_path)
    assert "backend" not in config.to_payload()["xfit"]
    exact_fields = {
        "xfit.status_codes",
        "xfit.converged",
        "xfit.evaluations",
        "xfit.valid_pixel_count",
        "xfit.valid_pixel_fraction",
        "xfit.null_chi_square",
        "xfit.degrees_of_freedom",
        "xfit.uncertainty_valid",
        "xfit.uncertainty_reason_codes",
    }
    reference = reference_payload = None
    for backend in ("cupy", "numba-cuda-mlir"):
        selected = replace(config, xfit=replace(config.xfit, backend=backend))
        selected_path = tmp_path / f"{backend}.json"
        selected_path.write_text(json.dumps(selected.to_payload()))
        output = tmp_path / backend
        rc = run_component(
            "xscan",
            [
                "benchmark-pipeline",
                "--stage",
                "pipeline",
                "--config",
                str(selected_path),
                "--items",
                str(items_path),
                "--output",
                str(output),
                "--warmup",
                "1",
                "--repeat",
                "1",
            ],
        )
        captured = capsys.readouterr()
        assert rc == 0, captured.err
        summary = json.loads((output / "summary.json").read_text())
        assert [row["completed_items"] for row in summary["rounds"]] == [1, 1]
        previous = None
        for index in range(2):
            result = json.loads(
                (output / f"round-{index:03d}" / "item-0000.json").read_text()
            )
            assert (
                result["configuration_sha256"]
                == selected.configuration_sha256
            )
            evidence = result["scientific_evidence"]
            assert evidence["xfit"]["backend"] == backend
            arrays = pipeline.decode_device_pipeline_evidence(
                evidence, config=selected
            )
            assert set(arrays) == set(stages.SCIENCE_KEYS)
            assert len(result["predictions"]) == len(items[0].candidates) == 2
            assert result["transfers"]["pipeline_full_array_d2h_bytes"] == 0
            assert result["transfers"]["pipeline_compact_h2d_bytes"] == 0
            assert result["transfers"]["terminal_d2h_calls"] == 1
            if previous is not None:
                for name, values in arrays.items():
                    np.testing.assert_array_equal(values, previous[name])
            previous = arrays
            if reference is None:
                reference, reference_payload = arrays, result
            for key in (
                "item_id",
                "input_sha256",
                "checkpoint_sha256",
                "feature_schema_sha256",
                "xpois",
                "transfers",
            ):
                assert result[key] == reference_payload[key]
            for actual, expected in zip(
                result["predictions"],
                reference_payload["predictions"],
                strict=True,
            ):
                assert actual.keys() == expected.keys()
                for key in actual:
                    if key in {"logit", "probability"}:
                        np.testing.assert_allclose(
                            actual[key], expected[key], rtol=3e-5, atol=3e-5
                        )
                    else:
                        assert actual[key] == expected[key]
            for name, actual in arrays.items():
                expected = reference[name]
                assert (
                    actual.shape == expected.shape
                    and actual.dtype == expected.dtype
                )
                if name.startswith("xpois.") or name in exact_fields:
                    np.testing.assert_array_equal(actual, expected)
                    continue
                for mask in (np.isnan, np.isposinf, np.isneginf):
                    np.testing.assert_array_equal(
                        mask(actual), mask(expected)
                    )
                tolerance = (
                    3e-5
                    if name == "xfit.features" or name.startswith("xscan.")
                    else 5e-12
                )
                np.testing.assert_allclose(
                    actual,
                    expected,
                    rtol=tolerance,
                    atol=tolerance,
                    equal_nan=True,
                )
