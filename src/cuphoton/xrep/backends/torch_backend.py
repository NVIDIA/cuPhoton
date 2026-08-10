# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Torch backend implementation."""

from __future__ import annotations

import os

import numpy as np

from ..geometry import ReprojectionResult, ReprojectionSpec
from ..mapping import PreparedReprojection


class TorchBackend:
    """Torch-backed interpolation backend."""

    name = "torch"

    def is_available(self) -> bool:
        try:
            import torch  # noqa: F401
        except Exception:
            return False
        return True

    def reproject(
        self,
        source: np.ndarray,
        prepared: PreparedReprojection,
        spec: ReprojectionSpec,
        *,
        source_mask: np.ndarray | None = None,
    ) -> ReprojectionResult:
        torch = _load_torch()
        if source.ndim != 2:
            raise ValueError("source must be a 2D array")

        device = _select_torch_device()
        source_array = _as_native(source)
        source_t = torch.as_tensor(
            source_array,
            dtype=torch.float64,
            device=device,
        )
        x_t = torch.as_tensor(
            prepared.x_local,
            dtype=torch.float64,
            device=device,
        )
        y_t = torch.as_tensor(
            prepared.y_local,
            dtype=torch.float64,
            device=device,
        )
        valid_t = torch.as_tensor(
            prepared.valid,
            dtype=torch.bool,
            device=device,
        )
        area_t = torch.as_tensor(
            prepared.relative_area,
            dtype=torch.float64,
            device=device,
        )

        with torch.inference_mode():
            if spec.interpolation == "bilinear":
                sampled = _sample_bilinear_torch(
                    source_t,
                    x_t,
                    y_t,
                    fill_value=spec.fill_value,
                )
            elif spec.interpolation == "lanczos3":
                sampled = _sample_lanczos_torch(
                    source_t,
                    x_t,
                    y_t,
                    a=spec.lanczos_a,
                    two_a_footprint=spec.two_a_footprint,
                    fill_value=spec.fill_value,
                )
            else:
                raise ValueError(
                    "interpolation must be 'bilinear' or 'lanczos3'"
                )

            if spec.area_scaling:
                sampled = sampled * area_t
            sampled = torch.where(
                valid_t,
                sampled,
                torch.full_like(sampled, float(spec.fill_value)),
            )

            if source_mask is not None:
                mask_out = _propagate_mask_or_torch(
                    torch.as_tensor(_as_native(source_mask), device=device),
                    x_t,
                    y_t,
                    interpolation="bilinear",
                    lanczos_a=spec.lanczos_a,
                    two_a_footprint=spec.two_a_footprint,
                    invalid_mask_value=spec.invalid_mask_value,
                )
                mask_out_np = mask_out.detach().cpu().numpy()
            else:
                mask_out_np = (~valid_t).detach().cpu().numpy()

        return ReprojectionResult(
            image=sampled.detach().cpu().numpy(),
            mask=mask_out_np,
            bbox=spec.output_bbox,
            backend=self.name,
            interpolation=spec.interpolation,
            metadata={
                "area_scaling": spec.area_scaling,
                "mapping_grid_step": spec.mapping_grid_step,
                "device": str(device),
            },
        )

    def reproject_mask(
        self,
        source_mask: np.ndarray,
        prepared: PreparedReprojection,
        spec: ReprojectionSpec,
    ) -> np.ndarray:
        """Propagate every mask bit contributing to the selected kernel."""

        torch = _load_torch()
        device = _select_torch_device()
        with torch.inference_mode():
            mask_out = _propagate_mask_or_torch(
                torch.as_tensor(_as_native(source_mask), device=device),
                torch.as_tensor(
                    prepared.x_local,
                    dtype=torch.float64,
                    device=device,
                ),
                torch.as_tensor(
                    prepared.y_local,
                    dtype=torch.float64,
                    device=device,
                ),
                interpolation=spec.interpolation,
                lanczos_a=spec.lanczos_a,
                two_a_footprint=spec.two_a_footprint,
                invalid_mask_value=spec.invalid_mask_value,
            )
        return mask_out.detach().cpu().numpy()

    def reproject_variance(
        self,
        source_variance: np.ndarray,
        prepared: PreparedReprojection,
        spec: ReprojectionSpec,
        *,
        variance_fill_value: float = float("nan"),
    ) -> np.ndarray:
        """Propagate a diagonal source variance plane."""

        torch = _load_torch()
        if source_variance.ndim != 2:
            raise ValueError("variance must be a 2D array")

        device = _select_torch_device()
        variance_t = torch.as_tensor(
            _as_native(source_variance),
            dtype=torch.float64,
            device=device,
        )
        x_t = torch.as_tensor(
            prepared.x_local,
            dtype=torch.float64,
            device=device,
        )
        y_t = torch.as_tensor(
            prepared.y_local,
            dtype=torch.float64,
            device=device,
        )
        valid_t = torch.as_tensor(
            prepared.valid,
            dtype=torch.bool,
            device=device,
        )
        area_t = torch.as_tensor(
            prepared.relative_area,
            dtype=torch.float64,
            device=device,
        )

        with torch.inference_mode():
            if spec.interpolation == "bilinear":
                sampled = _sample_bilinear_variance_torch(
                    variance_t,
                    x_t,
                    y_t,
                    fill_value=variance_fill_value,
                )
            elif spec.interpolation == "lanczos3":
                sampled = _sample_lanczos_torch(
                    variance_t,
                    x_t,
                    y_t,
                    a=spec.lanczos_a,
                    two_a_footprint=spec.two_a_footprint,
                    fill_value=variance_fill_value,
                    squared_weights=True,
                )
            else:
                raise ValueError(
                    "interpolation must be 'bilinear' or 'lanczos3'"
                )

            sampled = torch.where(
                valid_t,
                sampled,
                torch.full_like(
                    sampled,
                    float(variance_fill_value),
                ),
            )
            if spec.area_scaling:
                safe_area = torch.where(
                    valid_t,
                    area_t,
                    torch.ones_like(area_t),
                )
                sampled = sampled * torch.square(safe_area)
        return sampled.detach().cpu().numpy()


def _load_torch():
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            "Torch backend requested but PyTorch is unavailable; run "
            "'uv sync --extra torch' for development or install "
            "'cuphoton[torch]'"
        ) from exc
    return torch


def _as_native(array):
    """Return a native-byte-order ndarray.

    torch.as_tensor rejects non-native byte order, and FITS arrays are
    always big-endian (``>f8``), so ``reproject_fits`` and ``reproject_array``
    inputs must be byteswapped to native order before they reach torch. No-op
    for arrays already native.
    """
    arr = np.asarray(array)
    if arr.dtype.isnative:
        return arr
    return arr.astype(arr.dtype.newbyteorder("="))


class _TorchArrayModule:
    """Minimal NumPy-like array module backed by torch on a fixed device.

    Implements only the surface used by ``mapping.interpolate_mapping_grid`` /
    ``mapping.prepare_reprojection`` so the dense coordinate/area expansion
    can run on the GPU. Returned tensors live on ``device`` and the torch
    backend consumes them with zero-copy ``as_tensor`` calls.
    """

    def __init__(self, torch, device):
        self._t = torch
        self.device = device
        self.int64 = torch.int64
        self.float64 = torch.float64

    def arange(self, start, stop, dtype=None):
        return self._t.arange(start, stop, dtype=dtype, device=self.device)

    def asarray(self, x, dtype=None):
        if isinstance(x, self._t.Tensor):
            out = x.to(self.device)
        else:
            out = self._t.as_tensor(np.asarray(x), device=self.device)
        if dtype is not None:
            out = out.to(dtype)
        return out

    def searchsorted(self, sorted_seq, values, side="left"):
        return self._t.searchsorted(sorted_seq, values, side=side)

    def broadcast_to(self, x, shape):
        return self._t.broadcast_to(x, shape)

    def abs(self, x):
        return self._t.abs(x)

    def isfinite(self, x):
        return self._t.isfinite(x)

    def where(self, cond, a, b):
        if not isinstance(a, self._t.Tensor):
            a = self._t.as_tensor(
                a, dtype=self._t.float64, device=self.device
            )
        if not isinstance(b, self._t.Tensor):
            b = self._t.as_tensor(
                b, dtype=self._t.float64, device=self.device
            )
        return self._t.where(cond, a, b)


def make_torch_xp():
    """Return a torch-backed array module on the selected device."""
    torch = _load_torch()
    return _TorchArrayModule(torch, _select_torch_device())


def _sample_bilinear_torch(source, x, y, *, fill_value: float):
    torch = _load_torch()
    height, width = source.shape
    x0 = torch.floor(x).to(torch.int64)
    y0 = torch.floor(y).to(torch.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    fx = x - x0.to(x.dtype)
    fy = y - y0.to(y.dtype)
    w00 = (1.0 - fx) * (1.0 - fy)
    w10 = fx * (1.0 - fy)
    w01 = (1.0 - fx) * fy
    w11 = fx * fy

    in00 = (x0 >= 0) & (x0 < width) & (y0 >= 0) & (y0 < height)
    in10 = (x1 >= 0) & (x1 < width) & (y0 >= 0) & (y0 < height)
    in01 = (x0 >= 0) & (x0 < width) & (y1 >= 0) & (y1 < height)
    in11 = (x1 >= 0) & (x1 < width) & (y1 >= 0) & (y1 < height)

    zero = torch.zeros_like(x, dtype=source.dtype)
    w00 = torch.where(in00, w00, zero)
    w10 = torch.where(in10, w10, zero)
    w01 = torch.where(in01, w01, zero)
    w11 = torch.where(in11, w11, zero)
    norm = w00 + w10 + w01 + w11

    cx0 = torch.clamp(x0, 0, width - 1)
    cy0 = torch.clamp(y0, 0, height - 1)
    cx1 = torch.clamp(x1, 0, width - 1)
    cy1 = torch.clamp(y1, 0, height - 1)
    value = torch.zeros_like(norm, dtype=source.dtype)
    for weight, sample in (
        (w00, source[cy0, cx0]),
        (w10, source[cy0, cx1]),
        (w01, source[cy1, cx0]),
        (w11, source[cy1, cx1]),
    ):
        value = value + weight * torch.where(
            weight != 0.0,
            sample,
            torch.zeros_like(sample),
        )
    return torch.where(
        norm > 1e-12,
        value / norm,
        torch.full_like(value, float(fill_value)),
    )


def _sample_bilinear_variance_torch(
    source_variance,
    x,
    y,
    *,
    fill_value: float,
    eps: float = 1e-12,
):
    torch = _load_torch()
    height, width = source_variance.shape
    x0 = torch.floor(x).to(torch.int64)
    y0 = torch.floor(y).to(torch.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    fx = x - x0.to(x.dtype)
    fy = y - y0.to(y.dtype)
    w00 = (1.0 - fx) * (1.0 - fy)
    w10 = fx * (1.0 - fy)
    w01 = (1.0 - fx) * fy
    w11 = fx * fy

    in00 = (x0 >= 0) & (x0 < width) & (y0 >= 0) & (y0 < height)
    in10 = (x1 >= 0) & (x1 < width) & (y0 >= 0) & (y0 < height)
    in01 = (x0 >= 0) & (x0 < width) & (y1 >= 0) & (y1 < height)
    in11 = (x1 >= 0) & (x1 < width) & (y1 >= 0) & (y1 < height)

    zero = torch.zeros_like(x, dtype=source_variance.dtype)
    w00 = torch.where(in00, w00, zero)
    w10 = torch.where(in10, w10, zero)
    w01 = torch.where(in01, w01, zero)
    w11 = torch.where(in11, w11, zero)
    norm = w00 + w10 + w01 + w11

    cx0 = torch.clamp(x0, 0, width - 1)
    cy0 = torch.clamp(y0, 0, height - 1)
    cx1 = torch.clamp(x1, 0, width - 1)
    cy1 = torch.clamp(y1, 0, height - 1)
    value = torch.zeros_like(norm, dtype=source_variance.dtype)
    for weight, sample in (
        (w00, source_variance[cy0, cx0]),
        (w10, source_variance[cy0, cx1]),
        (w01, source_variance[cy1, cx0]),
        (w11, source_variance[cy1, cx1]),
    ):
        value = value + torch.square(weight) * torch.where(
            weight != 0.0,
            sample,
            torch.zeros_like(sample),
        )
    return torch.where(
        norm > eps,
        value / torch.square(norm),
        torch.full_like(value, float(fill_value)),
    )


def _sample_lanczos_torch(
    source,
    x,
    y,
    *,
    a: int,
    two_a_footprint: bool,
    fill_value: float,
    squared_weights: bool = False,
):
    torch = _load_torch()
    height, width = source.shape
    i0 = torch.floor(x).to(torch.int64)
    j0 = torch.floor(y).to(torch.int64)
    fx = x - i0.to(x.dtype)
    fy = y - j0.to(y.dtype)

    start = -(a - 1) if two_a_footprint else -a
    win = 2 * a if two_a_footprint else (2 * a + 1)
    x_weights = []
    y_weights = []
    sum_wx = torch.zeros_like(x, dtype=source.dtype)
    sum_wy = torch.zeros_like(y, dtype=source.dtype)

    for off in range(start, start + win):
        xx = i0 + off
        valid = (xx >= 0) & (xx < width)
        weight = torch.where(
            valid,
            _lanczos_weight_torch(
                torch.as_tensor(float(off), dtype=x.dtype, device=x.device)
                - fx,
                a,
            ).to(source.dtype),
            torch.zeros_like(x, dtype=source.dtype),
        )
        x_weights.append(weight)
        sum_wx = sum_wx + weight

    for off in range(start, start + win):
        yy = j0 + off
        valid = (yy >= 0) & (yy < height)
        weight = torch.where(
            valid,
            _lanczos_weight_torch(
                torch.as_tensor(float(off), dtype=y.dtype, device=y.device)
                - fy,
                a,
            ).to(source.dtype),
            torch.zeros_like(y, dtype=source.dtype),
        )
        y_weights.append(weight)
        sum_wy = sum_wy + weight

    norm = sum_wx * sum_wy
    numerator = torch.zeros_like(x, dtype=source.dtype)

    for iy, off_y in enumerate(range(start, start + win)):
        yy = j0 + off_y
        valid_y = (yy >= 0) & (yy < height)
        cy = torch.clamp(yy, 0, height - 1)
        wy = y_weights[iy]
        if not bool(torch.any(wy != 0.0)):
            continue

        for ix, off_x in enumerate(range(start, start + win)):
            xx = i0 + off_x
            valid_x = (xx >= 0) & (xx < width)
            weight = wy * x_weights[ix]
            valid = valid_y & valid_x & (weight != 0.0)
            if not bool(torch.any(valid)):
                continue
            cx = torch.clamp(xx, 0, width - 1)
            contribution_weight = (
                torch.square(weight) if squared_weights else weight
            )
            sample = torch.where(
                valid,
                source[cy, cx],
                torch.zeros_like(weight),
            )
            numerator = numerator + contribution_weight * sample

    denominator = torch.square(norm) if squared_weights else norm
    return torch.where(
        norm > 1e-12,
        numerator / denominator,
        torch.full_like(numerator, float(fill_value)),
    )


def _propagate_mask_or_torch(
    source_mask,
    x,
    y,
    *,
    interpolation: str,
    lanczos_a: int,
    two_a_footprint: bool,
    invalid_mask_value,
):
    torch = _load_torch()
    original_dtype = source_mask.dtype
    unsigned_64 = source_mask.dtype == torch.uint64
    unsigned_cast = source_mask.dtype in {
        torch.uint8,
        torch.uint16,
        torch.uint32,
    }
    if unsigned_64:
        # CUDA does not implement advanced indexing for uint64 tensors.
        # Reinterpreting the bits as int64 keeps indexing and OR exact.
        source_mask = source_mask.view(torch.int64)
        invalid_mask_value = (
            np.asarray(invalid_mask_value, dtype=np.uint64)
            .reshape(())
            .view(np.int64)
            .item()
        )
    elif unsigned_cast:
        # CUDA also lacks advanced indexing for uint16/uint32.  Their values
        # are exactly representable as int64 and convert back without loss.
        source_mask = source_mask.to(torch.int64)
        invalid_mask_value = int(invalid_mask_value)
    if interpolation not in {"bilinear", "lanczos3"}:
        raise ValueError("interpolation must be 'bilinear' or 'lanczos3'")

    height, width = source_mask.shape
    finite = torch.isfinite(x) & torch.isfinite(y)
    safe_x = torch.where(finite, x, torch.zeros_like(x))
    safe_y = torch.where(finite, y, torch.zeros_like(y))
    x0 = torch.floor(safe_x).to(torch.int64)
    y0 = torch.floor(safe_y).to(torch.int64)
    fx = safe_x - x0.to(torch.float64)
    fy = safe_y - y0.to(torch.float64)

    if interpolation == "bilinear":
        x_contributors = ((0, (1.0 - fx) != 0.0), (1, fx != 0.0))
        y_contributors = ((0, (1.0 - fy) != 0.0), (1, fy != 0.0))
    else:
        start = -(lanczos_a - 1) if two_a_footprint else -lanczos_a
        window = 2 * lanczos_a if two_a_footprint else 2 * lanczos_a + 1
        x_contributors = tuple(
            (
                offset,
                _lanczos_weight_is_nonzero_torch(
                    offset - fx,
                    lanczos_a,
                ),
            )
            for offset in range(start, start + window)
        )
        y_contributors = tuple(
            (
                offset,
                _lanczos_weight_is_nonzero_torch(
                    offset - fy,
                    lanczos_a,
                ),
            )
            for offset in range(start, start + window)
        )

    out = torch.zeros(
        x.shape,
        dtype=source_mask.dtype,
        device=source_mask.device,
    )
    for offset_y, nonzero_y in y_contributors:
        yy = y0 + offset_y
        cy = torch.clamp(yy, 0, height - 1)
        valid_y = nonzero_y & (yy >= 0) & (yy < height)
        for offset_x, nonzero_x in x_contributors:
            xx = x0 + offset_x
            cx = torch.clamp(xx, 0, width - 1)
            contributes = finite & valid_y & nonzero_x
            contributes &= (xx >= 0) & (xx < width)
            sample = source_mask[cy, cx]
            out = out | torch.where(
                contributes,
                sample,
                torch.zeros_like(sample),
            )

    invalid = ~(
        finite
        & (x >= 0.0)
        & (x <= float(width - 1))
        & (y >= 0.0)
        & (y <= float(height - 1))
    )
    invalid_value = torch.as_tensor(
        invalid_mask_value,
        dtype=source_mask.dtype,
        device=source_mask.device,
    )
    result = torch.where(invalid, out | invalid_value, out)
    if unsigned_64:
        return result.view(torch.uint64)
    if unsigned_cast:
        return result.to(original_dtype)
    return result


def _lanczos_weight_is_nonzero_torch(t, a: int):
    torch = _load_torch()
    within_support = torch.abs(t) < float(a)
    sinc_nonzero = (t == 0.0) | (t != torch.trunc(t))
    return within_support & sinc_nonzero


def _sincpi_torch(value):
    torch = _load_torch()
    out = torch.ones_like(value, dtype=value.dtype)
    integer_zero = (value != 0.0) & (value == torch.trunc(value))
    out = torch.where(integer_zero, torch.zeros_like(out), out)
    mask = (torch.abs(value) >= 1e-18) & ~integer_zero
    out = torch.where(
        mask,
        torch.sin(torch.pi * value) / (torch.pi * value),
        out,
    )
    return out


def _lanczos_weight_torch(value, a: int):
    torch = _load_torch()
    out = torch.zeros_like(value, dtype=value.dtype)
    mask = torch.abs(value) <= float(a)
    return torch.where(
        mask,
        _sincpi_torch(value) * _sincpi_torch(value / float(a)),
        out,
    )


def _select_torch_device():
    torch = _load_torch()
    requested = os.environ.get("CUPHOTON_XREP_TORCH_DEVICE")
    if requested:
        return torch.device(requested)
    if not torch.cuda.is_available():
        return torch.device("cpu")
    preferred: tuple[int, tuple[int, int, int]] | None = None
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        score = (
            int(props.major),
            int(props.minor),
            int(props.total_memory),
        )
        if preferred is None or score > preferred[1]:
            preferred = (index, score)
    assert preferred is not None
    return torch.device(f"cuda:{preferred[0]}")
