# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""CuPy backend implementation.

Wraps the validated CuPy Lanczos kernel in ``_cupy_kernels`` behind xRep's
backend interface, consuming the same ``PreparedReprojection`` coordinate and
area arrays as the CPU and Torch backends.
"""

from __future__ import annotations

import os

import numpy as np

from ..geometry import (
    DeviceReprojectionResult,
    ReprojectionResult,
    ReprojectionSpec,
)
from ..mapping import PreparedReprojection


class CupyBackend:
    """CuPy-backed interpolation backend."""

    name = "cupy"

    def is_available(self) -> bool:
        try:
            import cupy as cp

            return cp.cuda.runtime.getDeviceCount() > 0
        except Exception:
            return False

    def reproject(
        self,
        source: np.ndarray,
        prepared: PreparedReprojection,
        spec: ReprojectionSpec,
        *,
        source_mask: np.ndarray | None = None,
    ) -> ReprojectionResult:
        import cupy as cp

        device_result = self.reproject_device(
            source,
            prepared,
            spec,
            source_mask=source_mask,
        )

        return ReprojectionResult(
            image=cp.asnumpy(device_result.image),
            mask=(
                cp.asnumpy(device_result.mask)
                if device_result.mask is not None
                else None
            ),
            bbox=device_result.bbox,
            backend=device_result.backend,
            interpolation=device_result.interpolation,
            metadata=device_result.metadata,
        )

    def reproject_device(
        self,
        source,
        prepared: PreparedReprojection,
        spec: ReprojectionSpec,
        *,
        source_mask=None,
    ) -> DeviceReprojectionResult:
        import cupy as cp

        from ._cupy_kernels import (
            propagate_mask_or_cupy_raw,
            reproject_lanczos3_cupy_raw,
            sample_bilinear_cupy,
            sample_lanczos_cupy,
        )

        if getattr(source, "ndim", None) != 2:
            raise ValueError("source must be a 2D array")

        src = cp.asarray(source, dtype=cp.float64)
        x = cp.asarray(prepared.x_local, dtype=cp.float64)
        y = cp.asarray(prepared.y_local, dtype=cp.float64)
        valid = cp.asarray(prepared.valid)
        area = cp.asarray(prepared.relative_area, dtype=cp.float64)

        if spec.interpolation == "bilinear":
            lanczos_kernel = None
            sampled_is_final = False
            sampled = sample_bilinear_cupy(
                src, x, y, fill_value=spec.fill_value
            )
        elif spec.interpolation == "lanczos3":
            lanczos_kernel = _lanczos_kernel_variant()
            if (
                lanczos_kernel == "raw"
                and spec.lanczos_a == 3
                and spec.two_a_footprint
            ):
                sampled_is_final = True
                sampled = reproject_lanczos3_cupy_raw(
                    src,
                    x,
                    y,
                    valid,
                    area,
                    area_scaling=spec.area_scaling,
                    fill_value=spec.fill_value,
                )
            else:
                if lanczos_kernel == "raw":
                    lanczos_kernel = "elementwise"
                sampled_is_final = False
                sampled = sample_lanczos_cupy(
                    src,
                    x,
                    y,
                    a=spec.lanczos_a,
                    two_a_footprint=spec.two_a_footprint,
                )
        else:
            raise ValueError("interpolation must be 'bilinear' or 'lanczos3'")

        if not sampled_is_final:
            if spec.area_scaling:
                sampled = sampled * area
            sampled = cp.where(valid, sampled, spec.fill_value)

        if source_mask is not None:
            mask = propagate_mask_or_cupy_raw(
                cp.asarray(source_mask),
                x,
                y,
                invalid_mask_value=spec.invalid_mask_value,
            )
        else:
            mask = ~valid

        device = cp.cuda.runtime.getDeviceProperties(
            cp.cuda.runtime.getDevice()
        )["name"].decode()

        return DeviceReprojectionResult(
            image=sampled,
            mask=mask,
            bbox=spec.output_bbox,
            backend=self.name,
            interpolation=spec.interpolation,
            metadata={
                "area_scaling": spec.area_scaling,
                "mapping_grid_step": spec.mapping_grid_step,
                "device": device,
                "lanczos_kernel": lanczos_kernel,
                "result_location": "device",
            },
        )

    def reproject_mask(
        self,
        source_mask: np.ndarray,
        prepared: PreparedReprojection,
        spec: ReprojectionSpec,
    ) -> np.ndarray:
        """Propagate every mask bit contributing to the selected kernel."""

        import cupy as cp

        from ._cupy_kernels import propagate_mask_or_cupy

        mask = propagate_mask_or_cupy(
            cp.asarray(source_mask),
            cp.asarray(prepared.x_local, dtype=cp.float64),
            cp.asarray(prepared.y_local, dtype=cp.float64),
            interpolation=spec.interpolation,
            lanczos_a=spec.lanczos_a,
            two_a_footprint=spec.two_a_footprint,
            invalid_mask_value=spec.invalid_mask_value,
        )
        return cp.asnumpy(mask)

    def reproject_variance(
        self,
        source_variance: np.ndarray,
        prepared: PreparedReprojection,
        spec: ReprojectionSpec,
        *,
        variance_fill_value: float = float("nan"),
    ) -> np.ndarray:
        """Propagate a diagonal source variance plane."""

        import cupy as cp

        from ._cupy_kernels import (
            sample_bilinear_variance_cupy,
            sample_lanczos_variance_cupy,
        )

        if getattr(source_variance, "ndim", None) != 2:
            raise ValueError("variance must be a 2D array")

        variance = cp.asarray(source_variance, dtype=cp.float64)
        x = cp.asarray(prepared.x_local, dtype=cp.float64)
        y = cp.asarray(prepared.y_local, dtype=cp.float64)
        valid = cp.asarray(prepared.valid)
        area = cp.asarray(prepared.relative_area, dtype=cp.float64)

        if spec.interpolation == "bilinear":
            sampled = sample_bilinear_variance_cupy(
                variance,
                x,
                y,
                fill_value=variance_fill_value,
            )
        elif spec.interpolation == "lanczos3":
            sampled = sample_lanczos_variance_cupy(
                variance,
                x,
                y,
                a=spec.lanczos_a,
                two_a_footprint=spec.two_a_footprint,
            )
        else:
            raise ValueError("interpolation must be 'bilinear' or 'lanczos3'")

        sampled = cp.where(valid, sampled, variance_fill_value)
        if spec.area_scaling:
            sampled = sampled * cp.square(cp.where(valid, area, 1.0))
        return cp.asnumpy(sampled)


def _lanczos_kernel_variant() -> str:
    value = os.environ.get(
        "CUPHOTON_XREP_CUPY_LANCZOS_KERNEL",
        "elementwise",
    ).strip()
    if value not in {"elementwise", "raw"}:
        raise ValueError(
            "CUPHOTON_XREP_CUPY_LANCZOS_KERNEL must be 'elementwise' or 'raw'"
        )
    return value
