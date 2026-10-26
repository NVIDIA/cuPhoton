# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""CuPy backend implementation.

Wraps the validated CuPy Lanczos kernel in ``_cupy_kernels`` behind xRep's
backend interface, consuming the same ``PreparedReprojection`` coordinate and
area arrays as the CPU and Torch backends.
"""

from __future__ import annotations

import functools
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
        source_variance=None,
        variance_fill_value: float = float("nan"),
    ) -> DeviceReprojectionResult:
        import cupy as cp

        from ._cupy_kernels import (
            propagate_mask_or_cupy_raw,
            reproject_lanczos3_cupy_raw,
            reproject_with_variance_cupy,
            sample_bilinear_cupy,
            sample_lanczos_cupy,
        )

        if getattr(source, "ndim", None) != 2:
            raise ValueError("source must be a 2D array")
        if spec.interpolation not in {"bilinear", "lanczos3"}:
            raise ValueError("interpolation must be 'bilinear' or 'lanczos3'")

        src = cp.asarray(source, dtype=cp.float64)
        x = cp.asarray(prepared.x_local, dtype=cp.float64)
        y = cp.asarray(prepared.y_local, dtype=cp.float64)
        valid = cp.asarray(prepared.valid)
        area = cp.asarray(prepared.relative_area, dtype=cp.float64)

        lanczos_kernel = _resolve_lanczos_kernel(spec)

        variance = None
        if source_variance is not None:
            sampled, variance = reproject_with_variance_cupy(
                src,
                cp.asarray(source_variance, dtype=cp.float64),
                x,
                y,
                valid,
                area,
                interpolation=spec.interpolation,
                lanczos_a=spec.lanczos_a,
                two_a_footprint=spec.two_a_footprint,
                lanczos3_specialized=lanczos_kernel == "raw",
                area_scaling=spec.area_scaling,
                fill_value=spec.fill_value,
                variance_fill_value=variance_fill_value,
            )
        elif spec.interpolation == "bilinear":
            sampled = sample_bilinear_cupy(
                src, x, y, fill_value=spec.fill_value
            )
        elif lanczos_kernel == "raw":
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
            sampled = sample_lanczos_cupy(
                src,
                x,
                y,
                a=spec.lanczos_a,
                two_a_footprint=spec.two_a_footprint,
            )

        if source_variance is None and lanczos_kernel != "raw":
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

        metadata = {
            "area_scaling": spec.area_scaling,
            "mapping_grid_step": spec.mapping_grid_step,
            "device": _device_name(cp.cuda.runtime.getDevice()),
            "lanczos_kernel": lanczos_kernel,
            "result_location": "device",
        }
        if variance is not None:
            metadata.update(
                {
                    "variance_propagation": "diagonal",
                    "variance_covariance_propagated": False,
                    "variance_area_scaling_power": (
                        2 if spec.area_scaling else 0
                    ),
                }
            )

        return DeviceReprojectionResult(
            image=sampled,
            mask=mask,
            bbox=spec.output_bbox,
            backend=self.name,
            interpolation=spec.interpolation,
            metadata=metadata,
            variance=variance,
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

        from ._cupy_kernels import reproject_with_variance_cupy

        if getattr(source_variance, "ndim", None) != 2:
            raise ValueError("variance must be a 2D array")

        # The image half of the fused kernel is discarded; it takes the
        # image fill so a swapped return would not pass as a variance.
        variance = cp.asarray(source_variance, dtype=cp.float64)
        _, sampled = reproject_with_variance_cupy(
            variance,
            variance,
            cp.asarray(prepared.x_local, dtype=cp.float64),
            cp.asarray(prepared.y_local, dtype=cp.float64),
            cp.asarray(prepared.valid),
            cp.asarray(prepared.relative_area, dtype=cp.float64),
            interpolation=spec.interpolation,
            lanczos_a=spec.lanczos_a,
            two_a_footprint=spec.two_a_footprint,
            lanczos3_specialized=_resolve_lanczos_kernel(spec) == "raw",
            area_scaling=spec.area_scaling,
            fill_value=spec.fill_value,
            variance_fill_value=variance_fill_value,
        )
        return cp.asnumpy(sampled)


@functools.cache
def _device_name(device_id: int) -> str:
    import cupy as cp

    properties = cp.cuda.runtime.getDeviceProperties(device_id)
    return properties["name"].decode()


def _resolve_lanczos_kernel(spec: ReprojectionSpec) -> str | None:
    """Return the Lanczos kernel variant for ``spec``, ``None`` for bilinear.

    The compile-time ``raw`` variant covers only ``lanczos_a=3`` with the
    two-radius footprint; other kernels fall back to ``elementwise``.
    """

    if spec.interpolation != "lanczos3":
        return None
    variant = _lanczos_kernel_variant()
    specialized = spec.lanczos_a == 3 and spec.two_a_footprint
    if variant == "raw" and not specialized:
        return "elementwise"
    return variant


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
