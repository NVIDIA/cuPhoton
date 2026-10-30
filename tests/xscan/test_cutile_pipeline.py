# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

import cuphoton.xscan.device_pipeline as pipeline
from cuphoton.xscan.pipeline_benchmark import stages


@pytest.mark.parametrize("fusion", [False, True], ids=["unfused", "fused"])
def test_cutile_pipeline_cli_runs_real_stages_and_matches_cupy(
    tmp_path, capsys, fusion
):
    import json
    from dataclasses import replace

    from cuphoton.core.cli import run_component
    from cuphoton.xscan.pipeline_benchmark.fixture import prepare_fixture
    from cuphoton.xscan.pipeline_benchmark.runner import read_inputs

    cp = pytest.importorskip("cupy")
    torch = pytest.importorskip("torch")
    pytest.importorskip("cuda.tile")
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
    for backend in ("cupy", "cutile"):
        selected = replace(
            config,
            xfit=replace(
                config.xfit,
                backend=backend,
                fusion=fusion if backend == "cutile" else False,
            ),
        )
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
