# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""CPU backend implementation."""

from __future__ import annotations

import numpy as np

from ..geometry import ReprojectionResult, ReprojectionSpec
from ..interpolation import (
    propagate_mask_or,
    sample_bilinear_array,
    sample_bilinear_variance_array,
    sample_lanczos_array,
    sample_lanczos_variance_array,
)
from ..mapping import PreparedReprojection


class CpuBackend:
    """CPU reference backend."""

    name = "cpu"

    def is_available(self) -> bool:
        return True

    def reproject(
        self,
        source: np.ndarray,
        prepared: PreparedReprojection,
        spec: ReprojectionSpec,
        *,
        source_mask: np.ndarray | None = None,
    ) -> ReprojectionResult:
        source_arr = np.asarray(source, dtype=np.float64)
        if source_arr.ndim != 2:
            raise ValueError("source must be a 2D array")

        if spec.interpolation == "bilinear":
            sampled = sample_bilinear_array(
                source_arr,
                prepared.x_local,
                prepared.y_local,
                fill_value=spec.fill_value,
            )
        elif spec.interpolation == "lanczos3":
            sampled = sample_lanczos_array(
                source_arr,
                prepared.x_local,
                prepared.y_local,
                a=spec.lanczos_a,
                two_a_footprint=spec.two_a_footprint,
                fill_value=spec.fill_value,
            )
        else:
            raise ValueError("interpolation must be 'bilinear' or 'lanczos3'")

        if spec.area_scaling:
            sampled = sampled * prepared.relative_area
        sampled = np.where(prepared.valid, sampled, spec.fill_value)

        mask = None
        if source_mask is not None:
            mask = propagate_mask_or(
                source_mask,
                prepared.x_local,
                prepared.y_local,
                invalid_mask_value=spec.invalid_mask_value,
            )
        else:
            mask = ~prepared.valid

        return ReprojectionResult(
            image=np.asarray(sampled, dtype=source_arr.dtype),
            mask=mask,
            bbox=spec.output_bbox,
            backend=self.name,
            interpolation=spec.interpolation,
            metadata={
                "area_scaling": spec.area_scaling,
                "mapping_grid_step": spec.mapping_grid_step,
            },
        )

    def reproject_mask(
        self,
        source_mask: np.ndarray,
        prepared: PreparedReprojection,
        spec: ReprojectionSpec,
    ) -> np.ndarray:
        """Propagate every mask bit contributing to the selected kernel."""

        return propagate_mask_or(
            source_mask,
            prepared.x_local,
            prepared.y_local,
            interpolation=spec.interpolation,
            lanczos_a=spec.lanczos_a,
            two_a_footprint=spec.two_a_footprint,
            invalid_mask_value=spec.invalid_mask_value,
        )

    def reproject_variance(
        self,
        source_variance: np.ndarray,
        prepared: PreparedReprojection,
        spec: ReprojectionSpec,
        *,
        variance_fill_value: float = float("nan"),
    ) -> np.ndarray:
        """Propagate a diagonal source variance plane."""

        variance = np.asarray(source_variance, dtype=np.float64)
        if variance.ndim != 2:
            raise ValueError("variance must be a 2D array")

        if spec.interpolation == "bilinear":
            sampled = sample_bilinear_variance_array(
                variance,
                prepared.x_local,
                prepared.y_local,
                fill_value=variance_fill_value,
            )
        elif spec.interpolation == "lanczos3":
            sampled = sample_lanczos_variance_array(
                variance,
                prepared.x_local,
                prepared.y_local,
                a=spec.lanczos_a,
                two_a_footprint=spec.two_a_footprint,
                fill_value=variance_fill_value,
            )
        else:
            raise ValueError("interpolation must be 'bilinear' or 'lanczos3'")

        sampled = np.where(
            prepared.valid,
            sampled,
            variance_fill_value,
        )
        if spec.area_scaling:
            safe_area = np.where(
                prepared.valid,
                prepared.relative_area,
                1.0,
            )
            sampled = sampled * np.square(safe_area)
        return sampled
