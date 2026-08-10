# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""CuPy interpolation kernels for the CuPy backend.

The validated Lanczos implementation uses a single-launch fused
``ElementwiseKernel`` that computes per-pixel weights and accumulates in
registers (no materialised footprint tensor). Bilinear sampling and mask
propagation mirror the CPU reference semantics in ``xrep.interpolation`` so
the CuPy backend is interchangeable with the CPU and Torch backends.
"""

from __future__ import annotations

import cupy as cp

_MASK_OR_RAW_SOURCE = r"""
#define DEFINE_MASK_OR_KERNEL(NAME, T)                                \
extern "C" __global__ void NAME(                                      \
    const T* __restrict__ source_mask,                                \
    long long H,                                                      \
    long long W,                                                      \
    const double* __restrict__ x_in,                                  \
    const double* __restrict__ y_in,                                  \
    T invalid_mask_value,                                             \
    T* __restrict__ out,                                              \
    long long n                                                       \
) {                                                                   \
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; \
    if (idx >= n) return;                                             \
                                                                      \
    double x = x_in[idx];                                             \
    double y = y_in[idx];                                             \
    long long x0 = (long long)floor(x);                               \
    long long y0 = (long long)floor(y);                               \
    long long x1 = x0 + 1;                                            \
    long long y1 = y0 + 1;                                            \
                                                                      \
    double fx = x - (double)x0;                                       \
    double fy = y - (double)y0;                                       \
    double w0x = 1.0 - fx;                                            \
    double w1x = fx;                                                  \
    double w0y = 1.0 - fy;                                            \
    double w1y = fy;                                                  \
                                                                      \
    bool in00 = (x0 >= 0) && (x0 < W) && (y0 >= 0) && (y0 < H);       \
    bool in10 = (x1 >= 0) && (x1 < W) && (y0 >= 0) && (y0 < H);       \
    bool in01 = (x0 >= 0) && (x0 < W) && (y1 >= 0) && (y1 < H);       \
    bool in11 = (x1 >= 0) && (x1 < W) && (y1 >= 0) && (y1 < H);       \
                                                                      \
    bool k00 = in00 && (w0x > 0.0) && (w0y > 0.0);                    \
    bool k10 = in10 && (w1x > 0.0) && (w0y > 0.0);                    \
    bool k01 = in01 && (w0x > 0.0) && (w1y > 0.0);                    \
    bool k11 = in11 && (w1x > 0.0) && (w1y > 0.0);                    \
                                                                      \
    long long cx0 = x0 < 0 ? 0 : (x0 >= W ? W - 1 : x0);              \
    long long cy0 = y0 < 0 ? 0 : (y0 >= H ? H - 1 : y0);              \
    long long cx1 = x1 < 0 ? 0 : (x1 >= W ? W - 1 : x1);              \
    long long cy1 = y1 < 0 ? 0 : (y1 >= H ? H - 1 : y1);              \
                                                                      \
    T value = (T)0;                                                   \
    if (k00) value = (T)(value | source_mask[cy0 * W + cx0]);         \
    if (k10) value = (T)(value | source_mask[cy0 * W + cx1]);         \
    if (k01) value = (T)(value | source_mask[cy1 * W + cx0]);         \
    if (k11) value = (T)(value | source_mask[cy1 * W + cx1]);         \
                                                                      \
    bool invalid = !(                                                 \
        isfinite(x) && isfinite(y) &&                                 \
        (x >= 0.0) && (x <= (double)(W - 1)) &&                       \
        (y >= 0.0) && (y <= (double)(H - 1))                          \
    );                                                                \
    out[idx] = invalid ? (T)(value | invalid_mask_value) : value;     \
}

DEFINE_MASK_OR_KERNEL(mask_or_bool, bool)
DEFINE_MASK_OR_KERNEL(mask_or_i8, signed char)
DEFINE_MASK_OR_KERNEL(mask_or_u8, unsigned char)
DEFINE_MASK_OR_KERNEL(mask_or_i16, short)
DEFINE_MASK_OR_KERNEL(mask_or_u16, unsigned short)
DEFINE_MASK_OR_KERNEL(mask_or_i32, int)
DEFINE_MASK_OR_KERNEL(mask_or_u32, unsigned int)
DEFINE_MASK_OR_KERNEL(mask_or_i64, long long)
DEFINE_MASK_OR_KERNEL(mask_or_u64, unsigned long long)
"""

_MASK_OR_RAW_KERNELS = {
    ("b", 1): cp.RawKernel(_MASK_OR_RAW_SOURCE, "mask_or_bool"),
    ("i", 1): cp.RawKernel(_MASK_OR_RAW_SOURCE, "mask_or_i8"),
    ("u", 1): cp.RawKernel(_MASK_OR_RAW_SOURCE, "mask_or_u8"),
    ("i", 2): cp.RawKernel(_MASK_OR_RAW_SOURCE, "mask_or_i16"),
    ("u", 2): cp.RawKernel(_MASK_OR_RAW_SOURCE, "mask_or_u16"),
    ("i", 4): cp.RawKernel(_MASK_OR_RAW_SOURCE, "mask_or_i32"),
    ("u", 4): cp.RawKernel(_MASK_OR_RAW_SOURCE, "mask_or_u32"),
    ("i", 8): cp.RawKernel(_MASK_OR_RAW_SOURCE, "mask_or_i64"),
    ("u", 8): cp.RawKernel(_MASK_OR_RAW_SOURCE, "mask_or_u64"),
}

# ------------------------------------------------------------------------- #
# Lanczos
# ------------------------------------------------------------------------- #
_LANCZOS_PREAMBLE = r"""
extern "C" {

__device__ __forceinline__ double sincpi(double t) {
    const double PI = 3.14159265358979323846;
    double at = fabs(t);
    if (at < 1e-18) return 1.0;
    if (t == trunc(t)) return 0.0;
    return sin(PI * t) / (PI * t);
}

__device__ __forceinline__ double lanczos_w(double t, int a) {
    // Inclusive boundary (|t| <= a) to match AFW edges
    double at = fabs(t);
    if (at <= (double)a) {
        return sincpi(t) * sincpi(t / (double)a);
    } else {
        return 0.0;
    }
}

} // extern "C"
"""

_lanczos_elem = cp.ElementwiseKernel(
    in_params=r"""
        raw float64 src,
        int64 H,
        int64 W,
        float64 xf,
        float64 yf,
        int32 a,
        int32 two_a_footprint,
        int32 squared_weights,
        float64 eps
    """,
    out_params="float64 out",
    operation=r"""
        double x = xf;
        double y = yf;

        long long i0 = (long long)floor(x);
        long long j0 = (long long)floor(y);
        double fx = x - (double)i0;
        double fy = y - (double)j0;

        int a_loc = a;
        int start = two_a_footprint ? (-(a_loc - 1)) : (-a_loc);
        int win   = two_a_footprint ? (2*a_loc)      : (2*a_loc + 1);

        const int MAXWIN = 16;
        double wx[MAXWIN];
        double wy[MAXWIN];

        double sum_wx = 0.0, sum_wy = 0.0;

        for (int k=0; k<win; ++k){
            int off = start + k;
            long long xx = i0 + (long long)off;
            double w = 0.0;
            if (xx >= 0 && xx < W){
                w = lanczos_w((double)off - fx, a_loc);
                sum_wx += w;
            }
            wx[k] = w;
        }

        for (int k=0; k<win; ++k){
            int off = start + k;
            long long yy = j0 + (long long)off;
            double w = 0.0;
            if (yy >= 0 && yy < H){
                w = lanczos_w((double)off - fy, a_loc);
                sum_wy += w;
            }
            wy[k] = w;
        }

        double ksum = sum_wx * sum_wy;

        double num = 0.0;
        for (int iy=0; iy<win; ++iy){
            double wyv = wy[iy];
            if (wyv == 0.0) continue;
            long long yy = j0 + (long long)(start + iy);
            if (yy < 0 || yy >= H) continue;
            long long base = yy * W;

            for (int ix=0; ix<win; ++ix){
                double wxv = wx[ix];
                if (wxv == 0.0) continue;
                long long xx = i0 + (long long)(start + ix);
                if (xx < 0 || xx >= W) continue;

                double weight = wyv * wxv;
                double v = src[base + xx];
                num += squared_weights
                    ? weight * weight * v
                    : weight * v;
            }
        }

        if (ksum > eps) {
            out = squared_weights
                ? num / (ksum * ksum)
                : num / ksum;
        } else {
            long long nnx = llround(x);
            long long nny = llround(y);
            if (nnx < 0) nnx = 0; else if (nnx >= W) nnx = W-1;
            if (nny < 0) nny = 0; else if (nny >= H) nny = H-1;
            out = src[nny * W + nnx];
        }
    """,
    name="lanczos_sample_elem_fp64",
    preamble=_LANCZOS_PREAMBLE,
)


_LANCZOS3_RAW_SOURCE = r"""
extern "C" {

__device__ __forceinline__ double xrep_sincpi(double t) {
    const double PI = 3.14159265358979323846;
    double at = fabs(t);
    if (at < 1e-18) return 1.0;
    if (t == trunc(t)) return 0.0;
    return sin(PI * t) / (PI * t);
}

__device__ __forceinline__ double xrep_lanczos3_w(double t) {
    double at = fabs(t);
    if (at <= 3.0) {
        return xrep_sincpi(t) * xrep_sincpi(t / 3.0);
    } else {
        return 0.0;
    }
}

__device__ __forceinline__ double xrep_lanczos3_sample_one(
    const double* __restrict__ src,
    long long H,
    long long W,
    double x,
    double y,
    double eps
) {
    long long i0 = (long long)floor(x);
    long long j0 = (long long)floor(y);
    double fx = x - (double)i0;
    double fy = y - (double)j0;

    const int start = -2;
    const int win = 6;
    double wx[6];
    double wy[6];
    double sum_wx = 0.0;
    double sum_wy = 0.0;

    #pragma unroll
    for (int k = 0; k < win; ++k) {
        int off = start + k;
        long long xx = i0 + (long long)off;
        double w = 0.0;
        if (xx >= 0 && xx < W) {
            w = xrep_lanczos3_w((double)off - fx);
            sum_wx += w;
        }
        wx[k] = w;
    }

    #pragma unroll
    for (int k = 0; k < win; ++k) {
        int off = start + k;
        long long yy = j0 + (long long)off;
        double w = 0.0;
        if (yy >= 0 && yy < H) {
            w = xrep_lanczos3_w((double)off - fy);
            sum_wy += w;
        }
        wy[k] = w;
    }

    double ksum = sum_wx * sum_wy;
    double num = 0.0;

    #pragma unroll
    for (int iy = 0; iy < win; ++iy) {
        double wyv = wy[iy];
        if (wyv == 0.0) continue;
        long long yy = j0 + (long long)(start + iy);
        if (yy < 0 || yy >= H) continue;
        long long base = yy * W;

        #pragma unroll
        for (int ix = 0; ix < win; ++ix) {
            double wxv = wx[ix];
            if (wxv == 0.0) continue;
            long long xx = i0 + (long long)(start + ix);
            if (xx < 0 || xx >= W) continue;

            double v = src[base + xx];
            num += (wyv * wxv) * v;
        }
    }

    if (ksum > eps) {
        return num / ksum;
    } else {
        long long nnx = llround(x);
        long long nny = llround(y);
        if (nnx < 0) nnx = 0; else if (nnx >= W) nnx = W - 1;
        if (nny < 0) nny = 0; else if (nny >= H) nny = H - 1;
        return src[nny * W + nnx];
    }
}

__global__ void lanczos3_sample_raw_fp64(
    const double* __restrict__ src,
    long long H,
    long long W,
    const double* __restrict__ x_in,
    const double* __restrict__ y_in,
    double eps,
    double* __restrict__ out,
    long long n
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    out[idx] = xrep_lanczos3_sample_one(
        src,
        H,
        W,
        x_in[idx],
        y_in[idx],
        eps
    );
}

__global__ void lanczos3_reproject_raw_fp64(
    const double* __restrict__ src,
    long long H,
    long long W,
    const double* __restrict__ x_in,
    const double* __restrict__ y_in,
    const bool* __restrict__ valid,
    const double* __restrict__ area,
    int area_scaling,
    double fill_value,
    double eps,
    double* __restrict__ out,
    long long n
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (!valid[idx]) {
        out[idx] = fill_value;
        return;
    }

    double value = xrep_lanczos3_sample_one(
        src,
        H,
        W,
        x_in[idx],
        y_in[idx],
        eps
    );
    if (area_scaling) {
        value *= area[idx];
    }
    out[idx] = value;
}

} // extern "C"
"""

_LANCZOS3_RAW_KERNEL = cp.RawKernel(
    _LANCZOS3_RAW_SOURCE,
    "lanczos3_sample_raw_fp64",
)

_LANCZOS3_REPROJECT_RAW_KERNEL = cp.RawKernel(
    _LANCZOS3_RAW_SOURCE,
    "lanczos3_reproject_raw_fp64",
)


def sample_lanczos_cupy(
    source: cp.ndarray,
    x: cp.ndarray,
    y: cp.ndarray,
    *,
    a: int = 3,
    two_a_footprint: bool = True,
    eps: float = 1e-12,
) -> cp.ndarray:
    """Single-launch fused Lanczos sampling (AFW-aligned numerics)."""

    if source.ndim != 2:
        raise ValueError("source must be 2D")
    height, width = (int(source.shape[0]), int(source.shape[1]))
    hout, wout = x.shape

    src64 = source.astype(cp.float64, copy=False).ravel()
    xf = x.astype(cp.float64, copy=False)
    yf = y.astype(cp.float64, copy=False)

    out64 = _lanczos_elem(
        src64,
        height,
        width,
        xf,
        yf,
        a,
        1 if two_a_footprint else 0,
        0,
        eps,
    )
    return out64.reshape(hout, wout)


def sample_lanczos_variance_cupy(
    source_variance: cp.ndarray,
    x: cp.ndarray,
    y: cp.ndarray,
    *,
    a: int = 3,
    two_a_footprint: bool = True,
    eps: float = 1e-12,
) -> cp.ndarray:
    """Propagate diagonal variance with squared normalized Lanczos weights."""

    if source_variance.ndim != 2:
        raise ValueError("source_variance must be 2D")
    height, width = (
        int(source_variance.shape[0]),
        int(source_variance.shape[1]),
    )
    output_height, output_width = x.shape

    variance64 = source_variance.astype(cp.float64, copy=False).ravel()
    x64 = x.astype(cp.float64, copy=False)
    y64 = y.astype(cp.float64, copy=False)
    output = _lanczos_elem(
        variance64,
        height,
        width,
        x64,
        y64,
        a,
        1 if two_a_footprint else 0,
        1,
        eps,
    )
    return output.reshape(output_height, output_width)


def sample_lanczos3_cupy_raw(
    source: cp.ndarray,
    x: cp.ndarray,
    y: cp.ndarray,
    *,
    eps: float = 1e-12,
) -> cp.ndarray:
    """Specialized RawKernel Lanczos3 sampler for the AFW 6-tap footprint."""

    if source.ndim != 2:
        raise ValueError("source must be 2D")
    height, width = (int(source.shape[0]), int(source.shape[1]))
    output = cp.empty(x.shape, dtype=cp.float64)
    n = int(output.size)
    if n == 0:
        return output

    src64 = cp.ascontiguousarray(
        source.astype(cp.float64, copy=False)
    ).ravel()
    x64 = cp.ascontiguousarray(x.astype(cp.float64, copy=False)).ravel()
    y64 = cp.ascontiguousarray(y.astype(cp.float64, copy=False)).ravel()
    out64 = output.ravel()

    block_size = 128
    grid_size = ((n + block_size - 1) // block_size,)
    _LANCZOS3_RAW_KERNEL(
        grid_size,
        (block_size,),
        (
            src64,
            height,
            width,
            x64,
            y64,
            float(eps),
            out64,
            n,
        ),
    )
    return output


def reproject_lanczos3_cupy_raw(
    source: cp.ndarray,
    x: cp.ndarray,
    y: cp.ndarray,
    valid: cp.ndarray,
    area: cp.ndarray,
    *,
    area_scaling: bool,
    fill_value: float,
    eps: float = 1e-12,
) -> cp.ndarray:
    """Reproject with a Lanczos3 RawKernel that fuses validity and area."""

    if source.ndim != 2:
        raise ValueError("source must be 2D")
    height, width = (int(source.shape[0]), int(source.shape[1]))
    output = cp.empty(x.shape, dtype=cp.float64)
    n = int(output.size)
    if n == 0:
        return output

    src64 = cp.ascontiguousarray(
        source.astype(cp.float64, copy=False)
    ).ravel()
    x64 = cp.ascontiguousarray(x.astype(cp.float64, copy=False)).ravel()
    y64 = cp.ascontiguousarray(y.astype(cp.float64, copy=False)).ravel()
    valid_bool = cp.ascontiguousarray(valid.astype(cp.bool_, copy=False))
    area64 = cp.ascontiguousarray(area.astype(cp.float64, copy=False)).ravel()
    out64 = output.ravel()

    block_size = 128
    grid_size = ((n + block_size - 1) // block_size,)
    _LANCZOS3_REPROJECT_RAW_KERNEL(
        grid_size,
        (block_size,),
        (
            src64,
            height,
            width,
            x64,
            y64,
            valid_bool.ravel(),
            area64,
            int(area_scaling),
            float(fill_value),
            float(eps),
            out64,
            n,
        ),
    )
    return output


# ------------------------------------------------------------------------- #
# Bilinear (mirrors xrep.interpolation.sample_bilinear_array)
# ------------------------------------------------------------------------- #
def sample_bilinear_cupy(
    source: cp.ndarray,
    x: cp.ndarray,
    y: cp.ndarray,
    *,
    fill_value: float = float("nan"),
    eps: float = 1e-12,
) -> cp.ndarray:
    height, width = source.shape
    x0 = cp.floor(x).astype(cp.int64)
    y0 = cp.floor(y).astype(cp.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    fx = x - x0
    fy = y - y0
    w00 = (1.0 - fx) * (1.0 - fy)
    w10 = fx * (1.0 - fy)
    w01 = (1.0 - fx) * fy
    w11 = fx * fy

    in00 = (x0 >= 0) & (x0 < width) & (y0 >= 0) & (y0 < height)
    in10 = (x1 >= 0) & (x1 < width) & (y0 >= 0) & (y0 < height)
    in01 = (x0 >= 0) & (x0 < width) & (y1 >= 0) & (y1 < height)
    in11 = (x1 >= 0) & (x1 < width) & (y1 >= 0) & (y1 < height)

    w00 = cp.where(in00, w00, 0.0)
    w10 = cp.where(in10, w10, 0.0)
    w01 = cp.where(in01, w01, 0.0)
    w11 = cp.where(in11, w11, 0.0)
    norm = w00 + w10 + w01 + w11

    cx0 = cp.clip(x0, 0, width - 1)
    cy0 = cp.clip(y0, 0, height - 1)
    cx1 = cp.clip(x1, 0, width - 1)
    cy1 = cp.clip(y1, 0, height - 1)

    source64 = source.astype(cp.float64, copy=False)
    value = cp.zeros_like(norm, dtype=cp.float64)
    for weight, sample in (
        (w00, source64[cy0, cx0]),
        (w10, source64[cy0, cx1]),
        (w01, source64[cy1, cx0]),
        (w11, source64[cy1, cx1]),
    ):
        value += weight * cp.where(weight != 0.0, sample, 0.0)
    out = cp.full_like(value, fill_value, dtype=cp.float64)
    safe = norm > eps
    out = cp.where(safe, value / cp.where(safe, norm, 1.0), out)
    return out


def sample_bilinear_variance_cupy(
    source_variance: cp.ndarray,
    x: cp.ndarray,
    y: cp.ndarray,
    *,
    fill_value: float = float("nan"),
    eps: float = 1e-12,
) -> cp.ndarray:
    """Propagate variance with squared normalized bilinear weights."""

    height, width = source_variance.shape
    x0 = cp.floor(x).astype(cp.int64)
    y0 = cp.floor(y).astype(cp.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    fx = x - x0
    fy = y - y0
    w00 = (1.0 - fx) * (1.0 - fy)
    w10 = fx * (1.0 - fy)
    w01 = (1.0 - fx) * fy
    w11 = fx * fy

    in00 = (x0 >= 0) & (x0 < width) & (y0 >= 0) & (y0 < height)
    in10 = (x1 >= 0) & (x1 < width) & (y0 >= 0) & (y0 < height)
    in01 = (x0 >= 0) & (x0 < width) & (y1 >= 0) & (y1 < height)
    in11 = (x1 >= 0) & (x1 < width) & (y1 >= 0) & (y1 < height)

    w00 = cp.where(in00, w00, 0.0)
    w10 = cp.where(in10, w10, 0.0)
    w01 = cp.where(in01, w01, 0.0)
    w11 = cp.where(in11, w11, 0.0)
    norm = w00 + w10 + w01 + w11

    cx0 = cp.clip(x0, 0, width - 1)
    cy0 = cp.clip(y0, 0, height - 1)
    cx1 = cp.clip(x1, 0, width - 1)
    cy1 = cp.clip(y1, 0, height - 1)

    variance64 = source_variance.astype(cp.float64, copy=False)
    value = cp.zeros_like(norm, dtype=cp.float64)
    for weight, sample in (
        (w00, variance64[cy0, cx0]),
        (w10, variance64[cy0, cx1]),
        (w01, variance64[cy1, cx0]),
        (w11, variance64[cy1, cx1]),
    ):
        value += cp.square(weight) * cp.where(
            weight != 0.0,
            sample,
            0.0,
        )
    safe = norm > eps
    return cp.where(
        safe,
        value / cp.where(safe, cp.square(norm), 1.0),
        cp.asarray(fill_value, dtype=cp.float64),
    )


# ------------------------------------------------------------------------- #
# Mask propagation (mirrors xrep.interpolation.propagate_mask_or)
# ------------------------------------------------------------------------- #
def propagate_mask_or_cupy(
    source_mask: cp.ndarray,
    x: cp.ndarray,
    y: cp.ndarray,
    *,
    interpolation: str = "bilinear",
    lanczos_a: int = 3,
    two_a_footprint: bool = True,
    invalid_mask_value: int | bool = 1,
) -> cp.ndarray:
    mask = source_mask
    if mask.ndim != 2:
        raise ValueError("source_mask must be 2D")
    if interpolation not in {"bilinear", "lanczos3"}:
        raise ValueError("interpolation must be 'bilinear' or 'lanczos3'")

    height, width = mask.shape
    finite = cp.isfinite(x) & cp.isfinite(y)
    safe_x = cp.where(finite, x, 0.0)
    safe_y = cp.where(finite, y, 0.0)
    x0 = cp.floor(safe_x).astype(cp.int64)
    y0 = cp.floor(safe_y).astype(cp.int64)
    fx = safe_x - x0
    fy = safe_y - y0

    if interpolation == "bilinear":
        x_contributors = ((0, (1.0 - fx) != 0.0), (1, fx != 0.0))
        y_contributors = ((0, (1.0 - fy) != 0.0), (1, fy != 0.0))
    else:
        start = -(lanczos_a - 1) if two_a_footprint else -lanczos_a
        window = 2 * lanczos_a if two_a_footprint else 2 * lanczos_a + 1
        x_contributors = tuple(
            (
                offset,
                _lanczos_weight_is_nonzero_cupy(
                    offset - fx,
                    lanczos_a,
                ),
            )
            for offset in range(start, start + window)
        )
        y_contributors = tuple(
            (
                offset,
                _lanczos_weight_is_nonzero_cupy(
                    offset - fy,
                    lanczos_a,
                ),
            )
            for offset in range(start, start + window)
        )

    out = cp.zeros(x.shape, dtype=mask.dtype)
    for offset_y, nonzero_y in y_contributors:
        yy = y0 + offset_y
        cy = cp.clip(yy, 0, height - 1)
        valid_y = nonzero_y & (yy >= 0) & (yy < height)
        for offset_x, nonzero_x in x_contributors:
            xx = x0 + offset_x
            cx = cp.clip(xx, 0, width - 1)
            contributes = finite & valid_y & nonzero_x
            contributes &= (xx >= 0) & (xx < width)
            out |= cp.where(
                contributes,
                mask[cy, cx],
                cp.zeros_like(mask[cy, cx]),
            )

    invalid_value = cp.asarray(invalid_mask_value, dtype=mask.dtype)

    invalid = ~(
        finite
        & (x >= 0.0)
        & (x <= float(width - 1))
        & (y >= 0.0)
        & (y <= float(height - 1))
    )
    return cp.where(invalid, out | invalid_value, out).astype(mask.dtype)


def propagate_mask_or_cupy_raw(
    source_mask: cp.ndarray,
    x: cp.ndarray,
    y: cp.ndarray,
    *,
    interpolation: str = "bilinear",
    lanczos_a: int = 3,
    two_a_footprint: bool = True,
    invalid_mask_value: int | bool = 1,
) -> cp.ndarray:
    """Single-kernel mask propagation for bool and integer mask arrays."""

    mask = cp.asarray(source_mask)
    if interpolation != "bilinear":
        return propagate_mask_or_cupy(
            mask,
            x,
            y,
            interpolation=interpolation,
            lanczos_a=lanczos_a,
            two_a_footprint=two_a_footprint,
            invalid_mask_value=invalid_mask_value,
        )
    kernel = _MASK_OR_RAW_KERNELS.get((mask.dtype.kind, mask.dtype.itemsize))
    if kernel is None:
        return propagate_mask_or_cupy(
            mask,
            x,
            y,
            invalid_mask_value=invalid_mask_value,
        )

    height, width = (int(mask.shape[0]), int(mask.shape[1]))
    output = cp.empty(x.shape, dtype=mask.dtype)
    n = int(output.size)
    if n == 0:
        return output

    mask_contig = cp.ascontiguousarray(mask).ravel()
    x64 = cp.ascontiguousarray(x.astype(cp.float64, copy=False)).ravel()
    y64 = cp.ascontiguousarray(y.astype(cp.float64, copy=False)).ravel()
    out = output.ravel()
    invalid_value = mask.dtype.type(invalid_mask_value)

    block_size = 128
    grid_size = ((n + block_size - 1) // block_size,)
    kernel(
        grid_size,
        (block_size,),
        (
            mask_contig,
            height,
            width,
            x64,
            y64,
            invalid_value,
            out,
            n,
        ),
    )
    return output


def _lanczos_weight_is_nonzero_cupy(t: cp.ndarray, a: int) -> cp.ndarray:
    """Return the mathematical nonzero support of a Lanczos weight."""

    within_support = cp.abs(t) < float(a)
    sinc_nonzero = (t == 0.0) | (t != cp.trunc(t))
    return within_support & sinc_nonzero
