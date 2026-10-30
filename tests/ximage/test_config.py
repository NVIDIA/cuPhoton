# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import assert_type

import pytest
import yaml
from pydantic import ValidationError

from cuphoton.ximage.config import (
    OutputConfig,
    PsfSolveConfig,
    ReconstructionConfig,
    load_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_default_output_root_uses_product_state_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    expected = (tmp_path / "cuphoton" / "ximage" / "runs").resolve()

    assert OutputConfig().root_dir == expected
    assert not expected.exists()


def test_load_psf_config(tmp_path: Path) -> None:
    path = tmp_path / "psf.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "backend": "auto",
                "method": "adam",
                "data": {"base": "data"},
                "output": {
                    "root_dir": "runs",
                    "name": "psf-test",
                    "overwrite": True,
                },
                "srf": 2,
            },
            sort_keys=False,
        )
    )
    config = load_config(path, PsfSolveConfig)
    assert_type(config, PsfSolveConfig)
    assert config.method == "adam"
    assert config.backend == "auto"
    assert config.device == "auto"
    assert config.dtype == "float64"
    assert config.srf == 2
    assert config.output.name == "psf-test"


def test_load_config_uses_context_default_without_overriding_yaml(
    tmp_path: Path,
) -> None:
    default_root = tmp_path / "context-runs"
    defaulted_path = tmp_path / "defaulted.yaml"
    defaulted_path.write_text("method: adam\n")
    explicit_path = tmp_path / "explicit.yaml"
    explicit_path.write_text(
        yaml.safe_dump(
            {
                "method": "adam",
                "output": {"root_dir": "explicit-runs"},
            }
        )
    )

    defaulted = load_config(
        defaulted_path,
        PsfSolveConfig,
        default_output_root=default_root,
    )
    explicit = load_config(
        explicit_path,
        PsfSolveConfig,
        default_output_root=default_root,
    )

    assert defaulted.output.root_dir == default_root
    assert explicit.output.root_dir == Path("explicit-runs")


@pytest.mark.parametrize(
    ("filename", "model_type"),
    [
        ("psf-adam.example.yaml", PsfSolveConfig),
        ("psf-mm.example.yaml", PsfSolveConfig),
        ("reconstruct-l2.example.yaml", ReconstructionConfig),
        ("reconstruct-rho.example.yaml", ReconstructionConfig),
    ],
)
def test_documented_example_configs_parse(filename, model_type) -> None:
    config = load_config(
        PROJECT_ROOT / "examples" / "ximage" / filename,
        model_type,
    )

    assert config.output.name is not None


def test_load_reconstruction_config_rejects_extra(tmp_path: Path) -> None:
    path = tmp_path / "reconstruct.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "algorithm": "l2",
                "backend": "torch",
                "data": {"base": "data"},
                "output": {"root_dir": "runs"},
                "unexpected": True,
            },
            sort_keys=False,
        )
    )
    with pytest.raises(ValidationError):
        load_config(path, ReconstructionConfig)


@pytest.mark.parametrize("document", ["false", "0", "[]", '""'])
def test_load_config_rejects_non_mapping_documents(
    tmp_path: Path, document: str
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(document)

    with pytest.raises(ValidationError):
        load_config(
            path,
            ReconstructionConfig,
            default_output_root=tmp_path / "runs",
        )


@pytest.mark.parametrize("document", ["", "null"])
def test_load_config_accepts_empty_documents(
    tmp_path: Path, document: str
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(document)
    output_root = tmp_path / "runs"

    config = load_config(
        path, ReconstructionConfig, default_output_root=output_root
    )

    assert config.output.root_dir == output_root


def test_load_reconstruction_config_accepts_tiling(tmp_path: Path) -> None:
    path = tmp_path / "reconstruct-tiled.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "algorithm": "l2",
                "backend": "torch",
                "data": {"base": "data"},
                "output": {"root_dir": "runs"},
                "tiling": {
                    "enabled": True,
                    "tile_height": 128,
                    "tile_width": 128,
                    "halo": 16,
                },
            },
            sort_keys=False,
        )
    )

    config = load_config(path, ReconstructionConfig)

    assert config.tiling.enabled is True
    assert config.tiling.tile_height == 128
    assert config.tiling.tile_width == 128
    assert config.tiling.halo == 16


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"g_size": 4}, "g_size must be odd"),
        ({"srf": 0}, "greater than or equal to 1"),
        ({"sigma_g": 0}, "greater than 0"),
        ({"num_steps": 0}, "greater than or equal to 1"),
        (
            {"min_iter": 2, "max_iter": 1},
            "min_iter must not exceed max_iter",
        ),
        ({"method": "mm", "loss": "L1"}, "supports only"),
    ],
)
def test_psf_config_rejects_invalid_kernel_geometry(
    payload: dict[str, object], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        PsfSolveConfig.model_validate(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"max_iter": 0},
        {"ucf": 0},
        {"learning_rate": 0},
        {"stop_tol": 0},
        {"roi": {"height": 0}},
        {"mask": {"mode": "conservative"}},
    ],
)
def test_reconstruction_config_rejects_nonpositive_controls(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        ReconstructionConfig.model_validate(payload)


@pytest.mark.parametrize(
    "loss", ["bisquare_scale", "gauss_mixture", "gauss_uniform_mixture"]
)
def test_reconstruction_config_rejects_unsupported_losses(loss: str) -> None:
    with pytest.raises(ValidationError, match="rho_loss"):
        ReconstructionConfig.model_validate({"rho_loss": loss})


@pytest.mark.parametrize(
    "name", ["", " ", ".", "..", "nested/run", "nested\\run"]
)
def test_output_run_name_requires_one_directory_component(name) -> None:
    with pytest.raises(ValidationError, match="nonempty directory name"):
        OutputConfig(name=name)


def test_output_run_name_accepts_a_descriptive_leaf() -> None:
    assert OutputConfig(name="field-a run-1").name == "field-a run-1"


@pytest.mark.parametrize("loss", ["huber", "cauchy"])
@pytest.mark.parametrize("field", ["delta_rho_start", "delta_rho_end"])
def test_robust_scale_must_remain_finite_and_positive_in_dtype(loss, field):
    for value in (float("inf"), float("nan"), 1e100, 1e-100):
        with pytest.raises(ValidationError):
            ReconstructionConfig.model_validate(
                {"rho_loss": loss, "dtype": "float32", field: value}
            )
    for value in (1e100, 1e-100):
        config = ReconstructionConfig.model_validate(
            {"rho_loss": loss, "dtype": "float64", field: value}
        )
        assert getattr(config, field) == value
