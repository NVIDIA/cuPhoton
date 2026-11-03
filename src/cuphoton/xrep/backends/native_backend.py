# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Ahead-of-time CUDA reprojection with CuPy-owned device buffers."""

from __future__ import annotations

import numpy as np

from cuphoton.core._native_imaging import asarray, extension

from ..geometry import DeviceReprojectionResult
from ..mapping import PreparedReprojection, _coarse_mapping_grid
from .cupy_backend import CupyBackend, _device_name


def prepare_native(spec, source_shape) -> PreparedReprojection:
    """Expand the CPU WCS grid and prepare pixel geometry in native CUDA."""
    import cupy as cp

    mapped, x_edges, y_edges = _coarse_mapping_grid(
        spec.mapping,
        width=spec.output_bbox.width,
        height=spec.output_bbox.height,
        interp_len=spec.mapping_grid_step,
    )
    shape = spec.output_bbox.shape
    x = cp.empty(shape, dtype=np.float64)
    y = cp.empty(shape, dtype=np.float64)
    area = cp.empty(shape, dtype=np.float64)
    valid = cp.empty(shape, dtype=np.bool_)
    extension().prepare_mapping(
        asarray(mapped),
        asarray(x_edges),
        asarray(y_edges),
        x,
        y,
        area,
        valid,
        int(source_shape[0]),
        int(source_shape[1]),
        int(spec.source_origin[0]),
        int(spec.source_origin[1]),
        cp.cuda.get_current_stream().ptr,
    )
    return PreparedReprojection(x, y, area, valid)


def mapping_edge_score(prepared, source_shape):
    """Return each mapped coordinate's distance to the nearest source edge.

    Coordinates outside the source have negative scores. NaNs propagate with
    the same left-to-right minimum order as the NumPy/CuPy expression.
    """
    import cupy as cp

    if len(source_shape) != 2 or min(source_shape) < 1:
        raise ValueError("source_shape must contain two positive dimensions")
    result = cp.empty(prepared.output_shape, dtype=np.float64)
    extension().mapping_edge_score(
        asarray(prepared.x_local, dtype=np.float64),
        asarray(prepared.y_local, dtype=np.float64),
        result,
        int(source_shape[0]),
        int(source_shape[1]),
        cp.cuda.get_current_stream().ptr,
    )
    return result


def sample_mapping(prepared, ys, xs) -> np.ndarray:
    """Gather source coordinates at host integer positions.

    Only the selected coordinates cross back to the host. Columns in the
    returned array contain source x and source y, in input position order.
    """
    import cupy as cp

    rows, columns = np.asarray(ys), np.asarray(xs)
    if rows.shape != columns.shape:
        raise ValueError("ys and xs must have matching shapes")
    if rows.dtype.kind not in {"i", "u"} or columns.dtype.kind not in {
        "i",
        "u",
    }:
        raise TypeError("ys and xs must contain integer positions")
    height, width = prepared.output_shape
    if (
        np.any(rows < 0)
        or np.any(rows >= height)
        or np.any(columns < 0)
        or np.any(columns >= width)
    ):
        raise ValueError("positions must lie inside the prepared output grid")
    indices = np.column_stack((rows.ravel(), columns.ravel())).astype(
        np.int64
    )
    result = cp.empty(indices.shape, dtype=np.float64)
    extension().sample_mapping(
        asarray(prepared.x_local, dtype=np.float64),
        asarray(prepared.y_local, dtype=np.float64),
        asarray(indices),
        result,
        cp.cuda.get_current_stream().ptr,
    )
    return cp.asnumpy(result)


class NativeBackend(CupyBackend):
    """CUDA reprojection using kernels compiled into cuPhoton."""

    name = "native"

    def is_available(self) -> bool:
        try:
            import cupy as cp

            extension()
            return cp.cuda.runtime.getDeviceCount() > 0
        except (ImportError, RuntimeError, OSError):
            return False

    def _run(
        self,
        source,
        prepared,
        spec,
        *,
        source_mask=None,
        source_variance=None,
        variance_fill_value=float("nan"),
        want_image=True,
        want_mask=True,
        mask_uses_lanczos=False,
    ):
        import cupy as cp

        if getattr(source, "ndim", None) != 2:
            raise ValueError("source must be a 2D array")
        if spec.interpolation not in {"bilinear", "lanczos3"}:
            raise ValueError("interpolation must be 'bilinear' or 'lanczos3'")
        if spec.interpolation == "lanczos3":
            largest_a = (16 - (0 if spec.two_a_footprint else 1)) // 2
            if not 1 <= spec.lanczos_a <= largest_a:
                raise ValueError(
                    f"lanczos_a must be between 1 and {largest_a}"
                )
        if prepared.output_shape != spec.output_bbox.shape:
            raise ValueError(
                "prepared output shape must match spec output bbox"
            )
        src = asarray(
            source,
            dtype=np.float64
            if want_image or source_variance is not None
            else None,
        )
        variance = None
        if source_variance is not None:
            if source_variance.shape != source.shape:
                raise ValueError(
                    "source_variance shape must match source shape"
                )
            variance = asarray(source_variance, dtype=np.float64)
        mask = None
        if source_mask is not None:
            if source_mask.shape != source.shape:
                raise ValueError("source_mask shape must match source shape")
            if np.dtype(source_mask.dtype).kind not in {"b", "i", "u"}:
                raise TypeError("mask must have an integer or boolean dtype")
            mask = asarray(source_mask)
        shape = prepared.output_shape
        image_out = cp.empty(shape, dtype=np.float64) if want_image else None
        variance_out = (
            cp.empty(shape, dtype=np.float64)
            if variance is not None
            else None
        )
        mask_out = (
            cp.empty(shape, dtype=np.bool_ if mask is None else mask.dtype)
            if want_mask
            else None
        )
        invalid_bits = 0
        if mask is not None:
            invalid_bits = int(mask.dtype.type(spec.invalid_mask_value))
            invalid_bits &= (1 << 64) - 1
        extension().reproject(
            src,
            variance,
            mask,
            asarray(prepared.x_local, dtype=np.float64),
            asarray(prepared.y_local, dtype=np.float64),
            asarray(prepared.relative_area, dtype=np.float64),
            asarray(prepared.valid, dtype=np.bool_),
            image_out,
            variance_out,
            mask_out,
            int(spec.interpolation == "lanczos3"),
            int(spec.lanczos_a),
            int(spec.two_a_footprint),
            int(spec.area_scaling),
            int(mask_uses_lanczos),
            float(spec.fill_value),
            float(variance_fill_value),
            invalid_bits,
            cp.cuda.get_current_stream().ptr,
        )
        return image_out, variance_out, mask_out

    def reproject_device(
        self,
        source,
        prepared,
        spec,
        *,
        source_mask=None,
        source_variance=None,
        variance_fill_value=float("nan"),
    ) -> DeviceReprojectionResult:
        import cupy as cp

        image, variance, mask = self._run(
            source,
            prepared,
            spec,
            source_mask=source_mask,
            source_variance=source_variance,
            variance_fill_value=variance_fill_value,
        )
        metadata = {
            "area_scaling": spec.area_scaling,
            "mapping_grid_step": spec.mapping_grid_step,
            "device": _device_name(cp.cuda.runtime.getDevice()),
            "lanczos_kernel": "native"
            if spec.interpolation == "lanczos3"
            else None,
            "result_location": "device",
        }
        if variance is not None:
            metadata.update(
                variance_propagation="diagonal",
                variance_covariance_propagated=False,
                variance_area_scaling_power=2 if spec.area_scaling else 0,
            )
        return DeviceReprojectionResult(
            image=image,
            mask=mask,
            bbox=spec.output_bbox,
            backend=self.name,
            interpolation=spec.interpolation,
            metadata=metadata,
            variance=variance,
        )

    def reproject_variance(
        self,
        source_variance,
        prepared,
        spec,
        *,
        variance_fill_value=float("nan"),
    ) -> np.ndarray:
        import cupy as cp

        _, variance, _ = self._run(
            source_variance,
            prepared,
            spec,
            source_variance=source_variance,
            variance_fill_value=variance_fill_value,
            want_image=False,
            want_mask=False,
        )
        return cp.asnumpy(variance)

    def reproject_mask(self, source_mask, prepared, spec) -> np.ndarray:
        import cupy as cp

        _, _, mask = self._run(
            source_mask,
            prepared,
            spec,
            source_mask=source_mask,
            want_image=False,
            mask_uses_lanczos=True,
        )
        return cp.asnumpy(mask)
