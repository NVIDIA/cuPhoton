# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Typed configuration models for xImage workflows."""

from __future__ import annotations

from pathlib import Path, PureWindowsPath
from typing import Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from cuphoton.core.cli import ApplicationContext, get_component
from cuphoton.ximage.data.bundle import STORE_PRIORITY


class Model(BaseModel):
    """Strict base model that rejects unknown configuration fields."""

    model_config = ConfigDict(extra="forbid")


class DataConfig(Model):
    """Input store and exposure-selection options."""

    base: Path = Path("data")
    prefer: tuple[str, ...] = STORE_PRIORITY
    exposure_indices: list[int] | None = None
    max_exposures: int | None = Field(default=None, ge=1)


class RoiConfig(Model):
    """Spatial region in ``(row, column)`` coordinates."""

    center: tuple[int, int] | None = None
    height: int | None = Field(default=None, ge=1)
    width: int | None = Field(default=None, ge=1)


class TilingConfig(Model):
    """Optional tiled-reconstruction geometry in pixels."""

    enabled: bool = False
    tile_height: int | None = Field(default=None, ge=1)
    tile_width: int | None = Field(default=None, ge=1)
    halo: int = Field(default=0, ge=0)


class MaskConfig(Model):
    """Mask construction policy for reconstruction inputs."""

    mode: Literal["non_conservative", "conservative", "none"] = (
        "non_conservative"
    )
    dilation_factor: int = Field(default=1, ge=1)


class OutputConfig(Model):
    """Run-directory naming and overwrite policy."""

    root_dir: Path = Field(
        default_factory=lambda: (
            ApplicationContext.for_component(get_component("ximage")).runs_dir
        )
    )
    name: str | None = None
    overwrite: bool = False

    @field_validator("name")
    @classmethod
    def require_run_name(cls, value: str | None) -> str | None:
        """Keep a run name to one directory component on all platforms."""

        if value is not None and (
            not value.strip()
            or value in {".", ".."}
            or "\x00" in value
            or Path(value).name != value
            or PureWindowsPath(value).name != value
        ):
            raise ValueError("run name must be a nonempty directory name")
        return value


class WorkflowConfig(Model):
    """Execution, input, and output settings shared by all workflows."""

    backend: Literal["auto", "torch"] = "auto"
    device: Literal["auto", "cpu", "cuda"] = "auto"
    dtype: Literal["float32", "float64"] = "float64"
    data: DataConfig = Field(default_factory=DataConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


class PsfSolveConfig(WorkflowConfig):
    """Configuration for Adam or multiplicative-update PSF solving."""

    method: Literal["adam", "mm"] | None = None
    psf_artifact: str = "psfs"
    seed: int = Field(default=0, ge=0)
    srf: int = Field(default=1, ge=1)
    sigma_g: float = Field(default=0.55, gt=0.0)
    g_size: int | None = Field(default=None, ge=1)
    padding: Literal["VALID", "SAME"] = "VALID"
    h_initialization: Literal[
        "ones", "zeros", "gaussian", "dirac", "random"
    ] = "ones"
    loss: Literal["L2", "L1"] = "L2"
    grad_reg_coeff: float = Field(default=0.0, ge=0.0)
    lr: float = Field(default=1e-3, gt=0.0)
    num_steps: int = Field(default=3100, ge=1)
    ucf: float = Field(default=2.0, ge=1.0)
    min_iter: int = Field(default=10, ge=0)
    max_iter: int = Field(default=100, ge=1)
    tol: float = Field(default=1e-6, gt=0.0)
    verbose: bool = False

    @field_validator("g_size")
    @classmethod
    def require_odd_gaussian_size(cls, value: int | None) -> int | None:
        """Reject even kernels with ambiguous centers and transpose crops."""

        if value is not None and value % 2 == 0:
            raise ValueError("g_size must be odd")
        return value

    @model_validator(mode="after")
    def validate_mm_iteration_range(self) -> PsfSolveConfig:
        """Validate method-specific PSF solver semantics."""

        if self.min_iter > self.max_iter:
            raise ValueError("min_iter must not exceed max_iter")
        if self.method == "mm" and self.loss != "L2":
            raise ValueError("method='mm' supports only loss='L2'")
        return self


class ReconstructionConfig(WorkflowConfig):
    """Configuration for L2 or robust multi-exposure reconstruction."""

    algorithm: Literal["l2", "rho"] | None = None
    roi: RoiConfig = Field(default_factory=RoiConfig)
    tiling: TilingConfig = Field(default_factory=TilingConfig)
    mask: MaskConfig = Field(default_factory=MaskConfig)
    psf_path: Path | None = None
    srf: int = Field(default=1, ge=1)
    x_initialization: Literal["median", "mean", "constant"] = "median"
    ucf: float = Field(default=2.0, ge=1.0)
    learning_rate: float = Field(default=1.0, gt=0.0)
    max_iter: int = Field(default=75, ge=1)
    stop_tol: float = Field(default=1e-6, gt=0.0)
    effective_mask_mu: float = Field(default=0.1, gt=0.0, lt=1.0)
    exposure_batch_size: int | None = Field(default=None, ge=1)
    padding: Literal["VALID", "SAME"] = "VALID"
    background_model: Literal["none", "scalar_per_exposure"] = "none"
    verbose: bool = False
    rho_loss: Literal[
        "huber",
        "cauchy",
        "bisquare",
        "damped_rl",
    ] = "huber"
    delta_rho_start: float = Field(default=2.0, gt=0.0)
    delta_rho_end: float = Field(default=1.0, gt=0.0)


def load_config[ModelT: Model](
    path: Path,
    model_type: type[ModelT],
    *,
    default_output_root: Path | None = None,
) -> ModelT:
    """Load and strictly validate a YAML workflow configuration."""

    payload = yaml.safe_load(path.read_text()) or {}
    if default_output_root is not None and isinstance(payload, dict):
        output = payload.get("output")
        if output is None:
            payload["output"] = {"root_dir": default_output_root}
        elif isinstance(output, dict):
            output.setdefault("root_dir", default_output_root)
    return model_type.model_validate(payload)
