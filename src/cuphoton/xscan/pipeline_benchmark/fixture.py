# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Deterministic CPU inputs and an untrained benchmark fusion model."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import fftconvolve

from cuphoton.core.artifacts import file_sha256
from cuphoton.xfit import GaussianDipoleModel, LMConfig, fit_dipoles
from cuphoton.xfit.io import load_xfit_dataset, write_fit_artifacts
from cuphoton.xpois import (
    GaussianBasisComponent,
    build_gaussian_polynomial_basis,
)
from cuphoton.xscan.config import ModelConfig, PerformanceConfig
from cuphoton.xscan.device_pipeline import (
    DevicePipelineCandidate,
    DevicePipelineConfig,
    DevicePipelineItem,
    DeviceXFitPipelineConfig,
    DeviceXPOISPipelineConfig,
    NpyArrayDescriptor,
)
from cuphoton.xscan.xfit_features import (
    FEATURE_NAMES,
    build_xfit_feature_bundle,
)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _save_array(path: Path, values: np.ndarray) -> NpyArrayDescriptor:
    values = np.ascontiguousarray(values)
    np.save(path, values, allow_pickle=False)
    return NpyArrayDescriptor(
        path=str(path.resolve()),
        sha256=file_sha256(path),
        shape=tuple(values.shape),
        dtype=values.dtype.str,
    )


def _feature_schema(root: Path, stamp_size: int) -> Path:
    """Produce the canonical schema and artifacts from a real CPU fit."""

    calibration = root / "schema-calibration"
    calibration.mkdir()
    truth = np.array([[1.0, 1.6, 1.2, 0.2, 1.5, 0.6, -1.5, -0.6]])
    stamps = GaussianDipoleModel((stamp_size, stamp_size)).evaluate(truth)
    stamps += np.random.default_rng(0).normal(0.0, 0.001, stamps.shape)
    np.save(calibration / "difference.npy", stamps, allow_pickle=False)
    (calibration / "metadata.jsonl").write_text(
        '{"candidate_id":"schema-example"}\n', encoding="utf-8"
    )
    input_path = calibration / "input.npz"
    np.savez(
        input_path,
        candidate_id=np.array(["schema-example"]),
        images=stamps,
    )
    dataset = load_xfit_dataset(
        input_path, model="gaussian", mode="difference"
    )
    solver = LMConfig(max_evaluations=100)
    result = fit_dipoles(
        stamps,
        model="gaussian",
        initial=truth,
        backend="numpy",
        config=solver,
    )
    fit_root = calibration / "fit"
    write_fit_artifacts(
        fit_root,
        dataset=dataset,
        result=result,
        effective_config={
            "backend": "numpy",
            "model": "gaussian",
            "mode": "difference",
            "solver": asdict(solver),
            "purpose": "Synthetic schema example, not benchmark inference",
        },
    )
    feature_root = calibration / "features"
    build_xfit_feature_bundle(
        dataset_dir=calibration,
        xfit_run_dir=fit_root,
        output_dir=feature_root,
    )
    return feature_root / "schema.json"


def _checkpoint(
    root: Path, stamp_size: int, seed: int, schema_path: Path
) -> Path:
    import torch

    from cuphoton.xscan.model import build_model

    model_config = ModelConfig(
        input_mode="triplet",
        image_size=stamp_size,
        depths=[1],
        num_heads=[1],
        embed_dims=[8],
        decoder_embedding_dim=4,
        pos_dim=8,
        output_nc=1,
        drop_rate=0.0,
        attn_drop=0.0,
        xfit_feature_names=list(FEATURE_NAMES),
        xfit_hidden_dim=8,
    )
    # Fork only the CPU generator; construction never initializes CUDA.
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.random.default_generator.manual_seed(seed)
        model = build_model(**asdict(model_config))
        assert model.xfit_head is not None
        # The model constructor zeroes this layer. Exercise fusion here.
        with torch.no_grad():
            model.xfit_head[-1].weight.normal_(mean=0.0, std=0.1)
            model.xfit_head[-1].bias.fill_(0.03)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    checkpoint_dir = root / "checkpoint"
    checkpoint_dir.mkdir()
    torch.save(
        {
            "model_config": asdict(model_config),
            "model_state": model.state_dict(),
            "train_config": {"performance": asdict(PerformanceConfig())},
            "xfit_feature_bundle": {
                "schema_sha256": file_sha256(schema_path),
                "feature_sha256": schema["artifacts"]["features"]["sha256"],
                "source_artifacts": schema["source_artifacts"],
            },
            "benchmark_fixture": {
                "seed": seed,
                "trained": False,
                "description": "Random synthetic model; no accuracy claim",
            },
        },
        checkpoint_dir / "checkpoint.pt",
    )
    return checkpoint_dir


def prepare_fixture(
    root: Path,
    *,
    images: int,
    image_size: int,
    candidates: int,
    stamp_size: int,
    seed: int,
    device: str,
) -> tuple[Path, Path]:
    """Write config/items manifests and all inputs below a new directory.

    Each image contains an independently seeded textured reference, a matched
    target with planted dipoles, a variance plane and an exclusion mask around
    candidate stamps. Candidate order is a seeded permutation of spatial grid
    positions. Inputs and model weights repeat for the same arguments;
    provenance paths/timestamps and their artifact hashes are specific to this
    preparation. The small untrained model is a workflow fixture, not a model
    accuracy or representative-production-throughput benchmark.
    """

    for name, value in (
        ("images", images),
        ("image_size", image_size),
        ("candidates", candidates),
        ("stamp_size", stamp_size),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise ValueError(f"{name} must be a positive integer")
    if stamp_size < 9 or stamp_size % 2 == 0:
        raise ValueError("stamp_size must be odd and at least 9")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed < 2**63
    ):
        raise ValueError("seed must be an integer in [0, 2**63)")
    if not isinstance(device, str) or not device.startswith("cuda:"):
        raise ValueError("device must be an explicit cuda:N device")
    ordinal = device.removeprefix("cuda:")
    if (
        not ordinal.isascii()
        or not ordinal.isdecimal()
        or str(int(ordinal)) != ordinal
    ):
        raise ValueError("device must be an explicit cuda:N device")
    half = stamp_size // 2
    margin = half + 7
    side = math.isqrt(candidates - 1) + 1
    low, high = margin, image_size - 1 - margin
    if high - low < 2 or (
        side > 1 and (high - low) // (side - 1) < stamp_size + 2
    ):
        raise ValueError(
            "image_size is too small for nonoverlapping candidate stamps"
        )
    coordinates = (
        np.array([image_size // 2])
        if side == 1
        else np.linspace(low, high, side, dtype=int)
    )
    centers = [(int(y), int(x)) for y in coordinates for x in coordinates][
        :candidates
    ]
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    schema_path = _feature_schema(root, stamp_size)
    checkpoint_dir = _checkpoint(root, stamp_size, seed, schema_path)
    xpois_config = DeviceXPOISPipelineConfig(
        kernel_shape=(15, 15),
        basis_sigmas=(1.5, 3.0, 6.0),
        basis_degrees=(2, 1, 0),
        flux_conserve=False,
    )
    config = DevicePipelineConfig(
        device=device,
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_sha256=file_sha256(checkpoint_dir / "checkpoint.pt"),
        feature_schema_path=str(schema_path),
        feature_schema_sha256=file_sha256(schema_path),
        stamp_shape=(stamp_size, stamp_size),
        decision_threshold=0.5,
        xpois=xpois_config,
        xfit=DeviceXFitPipelineConfig(max_evaluations=100),
    )
    kernels, _ = build_gaussian_polynomial_basis(
        xpois_config.kernel_shape,
        [
            GaussianBasisComponent(sigma=sigma, degree=degree)
            for sigma, degree in zip(
                xpois_config.basis_sigmas,
                xpois_config.basis_degrees,
                strict=True,
            )
        ],
        flux_conserve=xpois_config.flux_conserve,
    )
    truth_kernel = 0.88 * kernels[0] + 0.12 * kernels[-1]
    model = GaussianDipoleModel(config.stamp_shape)
    items = []
    for image_index in range(images):
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, image_index])
        )
        reference = rng.normal(0.2, 0.05, (image_size, image_size))
        target = fftconvolve(reference, truth_kernel, mode="same") + 0.1
        target += rng.normal(0.0, 0.001, target.shape)
        fit_mask = np.ones(target.shape, dtype=bool)
        descriptors = []
        for spatial_index in rng.permutation(candidates).tolist():
            cy, cx = centers[spatial_index]
            region = (
                slice(cy - half, cy + half + 1),
                slice(cx - half, cx + half + 1),
            )
            truth = np.array(
                [
                    [
                        rng.uniform(0.5, 1.5),
                        1.7,
                        1.2,
                        rng.uniform(-0.5, 0.5),
                        1.5,
                        0.7,
                        -1.5,
                        -0.7,
                    ]
                ],
                dtype=np.float64,
            )
            target[region] += model.evaluate(truth)[0]
            fit_mask[region] = False
            descriptors.append(
                DevicePipelineCandidate(
                    candidate_id=f"image-{image_index:04d}-candidate-{spatial_index:04d}",
                    center_x=cx,
                    center_y=cy,
                    source_index=spatial_index,
                )
            )
        image_root = root / f"image-{image_index:04d}"
        image_root.mkdir()
        items.append(
            DevicePipelineItem(
                item_id=f"image-{image_index:04d}",
                reference=_save_array(
                    image_root / "reference.npy", reference
                ),
                target=_save_array(image_root / "target.npy", target),
                variance=_save_array(
                    image_root / "variance.npy", np.full(target.shape, 1.0e-6)
                ),
                fit_mask=_save_array(image_root / "fit-mask.npy", fit_mask),
                candidates=tuple(descriptors),
            )
        )
    config_path, items_path = root / "config.json", root / "items.json"
    _write_json(config_path, config.to_payload())
    _write_json(items_path, [item.to_payload() for item in items])
    return config_path, items_path
