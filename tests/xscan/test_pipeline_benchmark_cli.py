# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""CPU-only checks for the benchmark's shared CLI adapter."""

import json
from pathlib import Path

import pytest

from cuphoton.core.cli import run_component
from cuphoton.xscan.pipeline_benchmark import runner


def test_benchmark_help(capsys):
    assert run_component("xscan", ["help", "benchmark-pipeline"]) == 0
    output = capsys.readouterr().out
    assert "--output" in output
    assert "--config" in output
    assert "--stage" in output


def test_benchmark_defaults_and_path_conversion(monkeypatch):
    received = []
    monkeypatch.setattr(runner, "run_benchmark", received.append)
    assert (
        run_component("xscan", ["benchmark-pipeline", "--output", "~/bench"])
        == 0
    )
    (args,) = received
    assert vars(args) == {
        "output": Path("~/bench").expanduser(),
        "config": None,
        "items": None,
        "fits_reader": None,
        "images": 4,
        "image_size": 256,
        "candidates": 9,
        "stamp_size": 17,
        "seed": 2026,
        "device": "cuda:0",
        "warmup": 1,
        "repeat": 3,
        "timeout": 600.0,
        "order": "pipeline-first",
        "stage": "compare",
    }


def test_benchmark_child_options(monkeypatch):
    received = []
    monkeypatch.setattr(runner, "run_benchmark", received.append)
    assert (
        run_component(
            "xscan",
            [
                "benchmark-pipeline",
                "--output",
                "result",
                "--config",
                "~/config.json",
                "--items",
                "~/items.json",
                "--stage",
                "xfit",
                "--order",
                "staged-first",
                "--timeout",
                "12.5",
                "--images",
                "2",
                "--image-size",
                "1024",
                "--candidates",
                "16",
                "--stamp-size",
                "31",
                "--seed",
                "0",
                "--device",
                "cuda:1",
                "--warmup",
                "2",
                "--repeat",
                "4",
                "--fits-reader",
                "astropy",
            ],
        )
        == 0
    )
    (args,) = received
    assert args.output == Path("result")
    assert args.config == Path("~/config.json").expanduser()
    assert args.items == Path("~/items.json").expanduser()
    assert args.fits_reader == "astropy"
    assert (args.stage, args.order, args.timeout) == (
        "xfit",
        "staged-first",
        12.5,
    )
    assert (
        args.images,
        args.image_size,
        args.candidates,
        args.stamp_size,
    ) == (2, 1024, 16, 31)
    assert (args.seed, args.device, args.warmup, args.repeat) == (
        0,
        "cuda:1",
        2,
        4,
    )


def test_benchmark_config_selects_cutile_fusion(tmp_path, monkeypatch):
    from cuphoton.xscan.device_pipeline import (
        DevicePipelineCandidate,
        DevicePipelineConfig,
        DevicePipelineItem,
        DeviceXFitPipelineConfig,
        DeviceXPOISPipelineConfig,
        NpyArrayDescriptor,
    )
    from cuphoton.xscan.pipeline_benchmark import stages

    config = DevicePipelineConfig(
        device="cuda:0",
        checkpoint_dir=str(tmp_path),
        checkpoint_sha256="a" * 64,
        feature_schema_path=str(tmp_path / "schema.json"),
        feature_schema_sha256="b" * 64,
        stamp_shape=(17, 17),
        decision_threshold=0.5,
        xpois=DeviceXPOISPipelineConfig(
            kernel_shape=(3, 3), basis_sigmas=(0.8,), basis_degrees=(0,)
        ),
        xfit=DeviceXFitPipelineConfig(
            backend="cutile", fusion=True, max_evaluations=13
        ),
    )
    image = NpyArrayDescriptor(
        path=str(tmp_path / "image.npy"),
        sha256="c" * 64,
        shape=(64, 64),
        dtype="float64",
    )
    item = DevicePipelineItem(
        item_id="pair-0",
        reference=image,
        target=image,
        candidates=(DevicePipelineCandidate("candidate-0", 32, 32, 0),),
    )
    config_path, items_path = (
        tmp_path / "config.json",
        tmp_path / "items.json",
    )
    config_path.write_text(json.dumps(config.to_payload()))
    items_path.write_text(json.dumps([item.to_payload()]))
    received = []
    monkeypatch.setattr(
        stages, "run_stage", lambda *args: received.append(args)
    )
    output = tmp_path / "result"

    assert (
        run_component(
            "xscan",
            [
                "benchmark-pipeline",
                "--config",
                str(config_path),
                "--items",
                str(items_path),
                "--stage",
                "xfit",
                "--output",
                str(output),
            ],
        )
        == 0
    )
    ((stage, actual_config, actual_items, actual_output),) = received
    assert stage == "xfit"
    assert actual_config == config
    assert actual_config.xfit.backend == "cutile"
    assert actual_config.xfit.fusion is True
    assert actual_config.xfit.max_evaluations == 13
    assert actual_items == [item]
    assert actual_output == output


@pytest.mark.parametrize("timeout", ["0", "-1", "nan", "inf"])
def test_benchmark_rejects_invalid_timeout(monkeypatch, timeout):
    received = []
    monkeypatch.setattr(runner, "run_benchmark", received.append)
    assert (
        run_component(
            "xscan",
            [
                "benchmark-pipeline",
                "--output",
                "result",
                "--timeout",
                timeout,
            ],
        )
        != 0
    )
    assert not received


def test_benchmark_reports_unpaired_inputs(tmp_path, capsys):
    output = tmp_path / "result"
    assert (
        run_component(
            "xscan",
            [
                "benchmark-pipeline",
                "--output",
                str(output),
                "--config",
                str(tmp_path / "config.json"),
            ],
        )
        == 1
    )
    captured = capsys.readouterr()
    assert "provide both --config and --items" in captured.err + captured.out
    assert not output.exists()
