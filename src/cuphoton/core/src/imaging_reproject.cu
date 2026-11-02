/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "imaging_api.h"

#include <cmath>
#include <memory>

namespace cuphoton::imaging {
namespace {

using Index = std::int64_t;

// Dense mapping previously used separate CuPy ufuncs. Preserve their rounding
// boundaries even when the surrounding extension is built with FMA enabled.
__device__ double mapped_coordinate(
    const double* mapped,
    const Index* xe,
    const Index* ye,
    Index nx,
    Index ny,
    Index x,
    Index y,
    int component) {
    Index lo = 1, hi = nx - 1;
    while (lo < hi) {
        const Index mid = lo + (hi - lo) / 2;
        if (xe[mid] < x) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    const Index j = lo;
    lo = 1;
    hi = ny - 1;
    while (lo < hi) {
        const Index mid = lo + (hi - lo) / 2;
        if (ye[mid] < y) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    const Index i = lo;
    const double tx = __ddiv_rn(
        __dsub_rn(double(x), double(xe[j - 1])),
        __dsub_rn(double(xe[j]), double(xe[j - 1])));
    const double ty = __ddiv_rn(
        __dsub_rn(double(y), double(ye[i - 1])),
        __dsub_rn(double(ye[i]), double(ye[i - 1])));
    const double left = __dadd_rn(
        __dmul_rn(
            __dsub_rn(1.0, ty), mapped[((i - 1) * nx + j - 1) * 2 + component]),
        __dmul_rn(ty, mapped[(i * nx + j - 1) * 2 + component]));
    const double right = __dadd_rn(
        __dmul_rn(
            __dsub_rn(1.0, ty), mapped[((i - 1) * nx + j) * 2 + component]),
        __dmul_rn(ty, mapped[(i * nx + j) * 2 + component]));
    return __dadd_rn(__dmul_rn(__dsub_rn(1.0, tx), left), __dmul_rn(tx, right));
}

__global__ void mapping_kernel(
    const double* mapped,
    const Index* xe,
    const Index* ye,
    Index nx,
    Index ny,
    Index width,
    Index height,
    Index source_width,
    Index source_height,
    double origin_x,
    double origin_y,
    double* x_out,
    double* y_out,
    double* area,
    bool* valid) {
    const Index k = Index(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= width * height) {
        return;
    }
    const Index x = k % width, y = k / width;
    const double sx = mapped_coordinate(mapped, xe, ye, nx, ny, x, y, 0);
    const double sy = mapped_coordinate(mapped, xe, ye, nx, ny, x, y, 1);
    const double plx =
        mapped_coordinate(mapped, xe, ye, nx, ny, x - 1, y - 1, 0);
    const double ply =
        mapped_coordinate(mapped, xe, ye, nx, ny, x - 1, y - 1, 1);
    const double pux = mapped_coordinate(mapped, xe, ye, nx, ny, x, y - 1, 0);
    const double puy = mapped_coordinate(mapped, xe, ye, nx, ny, x, y - 1, 1);
    const double ax = __dsub_rn(sx, plx), ay = __dsub_rn(sy, ply);
    const double bx = __dsub_rn(sx, pux), by = __dsub_rn(sy, puy);
    const double relative_area =
        fabs(__dsub_rn(__dmul_rn(ax, by), __dmul_rn(ay, bx)));
    const double lx = __dsub_rn(sx, origin_x), ly = __dsub_rn(sy, origin_y);
    const bool inside = isfinite(lx) && isfinite(ly) && lx >= 0.0 && ly >= 0.0
        && lx <= double(source_width - 1) && ly <= double(source_height - 1);
    x_out[k] = lx;
    y_out[k] = ly;
    valid[k] = inside;
    area[k] = inside && isfinite(relative_area) ? relative_area : 0.0;
}

__device__ double sincpi(double t) {
    constexpr double pi = 3.14159265358979323846;
    if (fabs(t) < 1e-18) {
        return 1.0;
    }
    if (t == trunc(t)) {
        return 0.0;
    }
    return sin(pi * t) / (pi * t);
}

__device__ double lanczos_weight(double t, int a) {
    return fabs(t) <= double(a) ? sincpi(t) * sincpi(t / double(a)) : 0.0;
}

__device__ void sample(
    const double* source,
    const double* variance,
    Index height,
    Index width,
    double x,
    double y,
    int interpolation,
    int a,
    bool two_a,
    double& value,
    double& var) {
    const Index x0 = Index(floor(x)), y0 = Index(floor(y));
    const double fx = x - double(x0), fy = y - double(y0);
    double num = 0.0, num_var = 0.0, norm = 0.0;
    if (interpolation == 0) {
        double weights[4] = {
            (1.0 - fx) * (1.0 - fy), fx * (1.0 - fy), (1.0 - fx) * fy, fx * fy};
        const Index xs[4] = {x0, x0 + 1, x0, x0 + 1};
        const Index ys[4] = {y0, y0, y0 + 1, y0 + 1};
        for (int k = 0; k < 4; ++k) {
            if (xs[k] < 0 || xs[k] >= width || ys[k] < 0 || ys[k] >= height) {
                weights[k] = 0.0;
            }
            norm += weights[k];
        }
        for (int k = 0; k < 4; ++k) {
            const double w = weights[k];
            if (w == 0.0) {
                continue;
            }
            const Index offset = ys[k] * width + xs[k];
            if (source) {
                num = variance ? __fma_rn(w, source[offset], num)
                               : __dadd_rn(num, __dmul_rn(w, source[offset]));
            }
            if (variance) {
                num_var = __fma_rn(w * w, variance[offset], num_var);
            }
        }
    } else {
        const int start = two_a ? -(a - 1) : -a;
        const int window = two_a ? 2 * a : 2 * a + 1;
        double wx[16], wy[16];
        double sum_x = 0.0, sum_y = 0.0;
        for (int k = 0; k < window; ++k) {
            const Index xx = x0 + start + k, yy = y0 + start + k;
            wx[k] =
                xx >= 0 && xx < width ? lanczos_weight(start + k - fx, a) : 0.0;
            wy[k] = yy >= 0 && yy < height ? lanczos_weight(start + k - fy, a)
                                           : 0.0;
            sum_x += wx[k];
            sum_y += wy[k];
        }
        norm = sum_x * sum_y;
        for (int iy = 0; iy < window; ++iy) {
            if (wy[iy] == 0.0) {
                continue;
            }
            const Index yy = y0 + start + iy;
            if (yy < 0 || yy >= height) {
                continue;
            }
            for (int ix = 0; ix < window; ++ix) {
                if (wx[ix] == 0.0) {
                    continue;
                }
                const Index xx = x0 + start + ix;
                if (xx < 0 || xx >= width) {
                    continue;
                }
                const double weight = wy[iy] * wx[ix];
                if (source) {
                    num = __fma_rn(weight, source[yy * width + xx], num);
                }
                if (variance) {
                    num_var = __fma_rn(
                        weight * weight, variance[yy * width + xx], num_var);
                }
            }
        }
    }
    if (norm > 1e-12) {
        value = num / norm;
        var = num_var / (norm * norm);
    } else if (interpolation != 0 && source) {
        Index xx = llround(x), yy = llround(y);
        xx = xx < 0 ? 0 : (xx >= width ? width - 1 : xx);
        yy = yy < 0 ? 0 : (yy >= height ? height - 1 : yy);
        value = source[yy * width + xx];
    }
}

__device__ bool mask_support(
    double phase, int offset, int interpolation, int a) {
    if (interpolation == 0) {
        return offset == 0 ? (1.0 - phase) != 0.0 : phase != 0.0;
    }
    const double t = double(offset) - phase;
    return fabs(t) < double(a) && (t == 0.0 || t != trunc(t));
}

template <class Mask>
__global__ void reproject_kernel(
    const double* source,
    const double* variance,
    const Mask* mask,
    Index height,
    Index width,
    const double* x_in,
    const double* y_in,
    const double* area,
    const bool* valid,
    double* out,
    double* out_var,
    Mask* out_mask,
    Index count,
    int interpolation,
    int a,
    bool two_a,
    bool scale_area,
    int mask_interpolation,
    double fill,
    double variance_fill,
    Mask invalid_mask) {
    const Index k = Index(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= count) {
        return;
    }
    const double x = x_in[k], y = y_in[k];
    if (out || out_var) {
        double value = fill, var = variance_fill;
        if (valid[k]) {
            sample(
                source,
                variance,
                height,
                width,
                x,
                y,
                interpolation,
                a,
                two_a,
                value,
                var);
            if (scale_area) {
                value *= area[k];
                var *= area[k] * area[k];
            }
        }
        if (out) {
            out[k] = value;
        }
        if (out_var) {
            out_var[k] = var;
        }
    }
    if (!out_mask) {
        return;
    }
    if (!mask) {
        out_mask[k] = Mask(!valid[k]);
        return;
    }
    Mask bits = Mask(0);
    const bool finite = isfinite(x) && isfinite(y);
    if (finite) {
        const Index x0 = Index(floor(x)), y0 = Index(floor(y));
        const double fx = x - double(x0), fy = y - double(y0);
        const int start = mask_interpolation == 0 ? 0 : (two_a ? -(a - 1) : -a);
        const int window =
            mask_interpolation == 0 ? 2 : (two_a ? 2 * a : 2 * a + 1);
        for (int iy = 0; iy < window; ++iy) {
            const int oy = start + iy;
            const Index yy = y0 + oy;
            if (yy < 0 || yy >= height
                || !mask_support(fy, oy, mask_interpolation, a)) {
                continue;
            }
            for (int ix = 0; ix < window; ++ix) {
                const int ox = start + ix;
                const Index xx = x0 + ox;
                if (xx < 0 || xx >= width
                    || !mask_support(fx, ox, mask_interpolation, a)) {
                    continue;
                }
                bits = Mask(bits | mask[yy * width + xx]);
            }
        }
    }
    const bool inside = finite && x >= 0.0 && y >= 0.0 && x <= double(width - 1)
        && y <= double(height - 1);
    out_mask[k] = inside ? bits : Mask(bits | invalid_mask);
}

__device__ double ordered_minimum(double a, double b) {
    // NumPy/CuPy minimum propagates NaNs and selects the second equal value.
    return isnan(a) ? a : (a < b ? a : b);
}

__global__ void edge_score_kernel(
    const double* x,
    const double* y,
    double* out,
    Index count,
    Index height,
    Index width) {
    const Index k = Index(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= count) {
        return;
    }
    double score = ordered_minimum(x[k], y[k]);
    score = ordered_minimum(score, __dsub_rn(double(width - 1), x[k]));
    out[k] = ordered_minimum(score, __dsub_rn(double(height - 1), y[k]));
}

__global__ void sample_mapping_kernel(
    const double* x,
    const double* y,
    const Index* indices,
    double* out,
    Index count,
    Index height,
    Index width) {
    const Index k = Index(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= count) {
        return;
    }
    const Index row = indices[2 * k], column = indices[2 * k + 1];
    if (row < 0 || row >= height || column < 0 || column >= width) {
        out[2 * k] = out[2 * k + 1] = nan("");
        return;
    }
    const Index index = row * width + column;
    out[2 * k] = x[index];
    out[2 * k + 1] = y[index];
}

void same_shape(const Array& a, const Array& b) {
    if (a.shape != b.shape) {
        throw std::invalid_argument("reprojection buffer shapes do not match");
    }
}

}  // namespace

PyObject* prepare_mapping(PyObject*, PyObject* args) {
    PyObject *mapped_o, *xe_o, *ye_o, *x_o, *y_o, *area_o, *valid_o;
    long long source_height, source_width, origin_x, origin_y;
    unsigned long long stream;
    if (!PyArg_ParseTuple(
            args,
            "OOOOOOOLLLLK",
            &mapped_o,
            &xe_o,
            &ye_o,
            &x_o,
            &y_o,
            &area_o,
            &valid_o,
            &source_height,
            &source_width,
            &origin_x,
            &origin_y,
            &stream)) {
        return nullptr;
    }
    try {
        Array mapped(mapped_o), xe(xe_o), ye(ye_o), x(x_o), y(y_o),
            area(area_o), valid(valid_o);
        mapped.require('f', 8, 3);
        xe.require('i', 8, 1);
        ye.require('i', 8, 1);
        x.require('f', 8, 2, true);
        y.require('f', 8, 2, true);
        area.require('f', 8, 2, true);
        valid.require('b', 1, 2, true);
        same_shape(x, y);
        same_shape(x, area);
        same_shape(x, valid);
        if (source_height < 1 || source_width < 1 || xe.size < 2 || ye.size < 2
            || mapped.shape[0] != ye.size || mapped.shape[1] != xe.size
            || mapped.shape[2] != 2) {
            throw std::invalid_argument(
                "incompatible coarse mapping or source dimensions");
        }
        StreamGuard guard(stream);
        for (const auto* a : {&mapped, &xe, &ye, &x, &y, &area, &valid}) {
            a->wait(guard.stream);
        }
        {
            Detached detached;
            if (x.size) {
                mapping_kernel<<<(x.size + 255) / 256, 256, 0, guard.stream>>>(
                    mapped.data<double>(),
                    xe.data<Index>(),
                    ye.data<Index>(),
                    xe.size,
                    ye.size,
                    x.shape[1],
                    x.shape[0],
                    source_width,
                    source_height,
                    double(origin_x),
                    double(origin_y),
                    x.data<double>(),
                    y.data<double>(),
                    area.data<double>(),
                    valid.data<bool>());
            }
            guard.finish();
        }
        Py_RETURN_NONE;
    } catch (...) {
        return translate_exception();
    }
}

PyObject* reproject(PyObject*, PyObject* args) {
    PyObject *source_o, *variance_o, *mask_o, *x_o, *y_o, *area_o, *valid_o;
    PyObject *out_o, *out_var_o, *out_mask_o;
    int interpolation, a, two_a, scale_area, mask_lanczos;
    double fill, variance_fill;
    unsigned long long invalid_mask, stream;
    if (!PyArg_ParseTuple(
            args,
            "OOOOOOOOOOiiiiiddKK",
            &source_o,
            &variance_o,
            &mask_o,
            &x_o,
            &y_o,
            &area_o,
            &valid_o,
            &out_o,
            &out_var_o,
            &out_mask_o,
            &interpolation,
            &a,
            &two_a,
            &scale_area,
            &mask_lanczos,
            &fill,
            &variance_fill,
            &invalid_mask,
            &stream)) {
        return nullptr;
    }
    try {
        Array source(source_o), x(x_o), y(y_o), area(area_o), valid(valid_o);
        if (out_o != Py_None || out_var_o != Py_None) {
            source.require('f', 8, 2);
        } else if (source.shape.size() != 2) {
            throw std::invalid_argument("source must be two-dimensional");
        }
        x.require('f', 8, 2);
        y.require('f', 8, 2);
        area.require('f', 8, 2);
        valid.require('b', 1, 2);
        same_shape(x, y);
        same_shape(x, area);
        same_shape(x, valid);
        if (source.shape[0] < 1 || source.shape[1] < 1 || interpolation < 0
            || interpolation > 1
            || (interpolation == 1
                && (a < 1 || (two_a ? 2 * a : 2 * a + 1) > 16))) {
            throw std::invalid_argument(
                "invalid source dimensions or interpolation window");
        }
        auto optional = [](PyObject* object) -> std::unique_ptr<Array> {
            return object == Py_None ? nullptr
                                     : std::make_unique<Array>(object);
        };
        auto variance = optional(variance_o), mask = optional(mask_o),
             out = optional(out_o);
        auto out_var = optional(out_var_o), out_mask = optional(out_mask_o);
        if (variance) {
            variance->require('f', 8, 2);
            same_shape(source, *variance);
        }
        if (out) {
            out->require('f', 8, 2, true);
            same_shape(x, *out);
        }
        if (out_var) {
            out_var->require('f', 8, 2, true);
            same_shape(x, *out_var);
            if (!variance) {
                throw std::invalid_argument(
                    "variance output requires a source variance");
            }
        }
        if (mask) {
            if (mask->kind != 'b' && mask->kind != 'i' && mask->kind != 'u') {
                throw std::invalid_argument(
                    "mask must have an integer or boolean dtype");
            }
            mask->require(mask->kind, mask->itemsize, 2);
            same_shape(source, *mask);
        }
        if (out_mask) {
            out_mask->require(
                mask ? mask->kind : 'b', mask ? mask->itemsize : 1, 2, true);
            same_shape(x, *out_mask);
        }
        StreamGuard guard(stream);
        for (const auto* arr : {&source,
                 &x,
                 &y,
                 &area,
                 &valid,
                 variance.get(),
                 mask.get(),
                 out.get(),
                 out_var.get(),
                 out_mask.get()}) {
            if (arr) {
                arr->wait(guard.stream);
            }
        }
        const auto launch = [&](auto mask_type) {
            using Mask = decltype(mask_type);
            if (x.size) {
                reproject_kernel<Mask>
                    <<<(x.size + 127) / 128, 128, 0, guard.stream>>>(
                        out ? source.data<double>() : nullptr,
                        variance ? variance->data<double>() : nullptr,
                        mask ? mask->data<Mask>() : nullptr,
                        source.shape[0],
                        source.shape[1],
                        x.data<double>(),
                        y.data<double>(),
                        area.data<double>(),
                        valid.data<bool>(),
                        out ? out->data<double>() : nullptr,
                        out_var ? out_var->data<double>() : nullptr,
                        out_mask ? out_mask->data<Mask>() : nullptr,
                        x.size,
                        interpolation,
                        a,
                        bool(two_a),
                        bool(scale_area),
                        mask_lanczos ? interpolation : 0,
                        fill,
                        variance_fill,
                        Mask(invalid_mask));
            }
        };
        {
            Detached detached;
            if (!mask || mask->kind == 'b') {
                launch(bool{});
            } else if (mask->itemsize == 1) {
                launch(std::uint8_t{});
            } else if (mask->itemsize == 2) {
                launch(std::uint16_t{});
            } else if (mask->itemsize == 4) {
                launch(std::uint32_t{});
            } else {
                launch(std::uint64_t{});
            }
            guard.finish();
        }
        Py_RETURN_NONE;
    } catch (...) {
        return translate_exception();
    }
}

PyObject* mapping_edge_score(PyObject*, PyObject* args) {
    PyObject *x_o, *y_o, *out_o;
    long long height, width;
    unsigned long long stream;
    if (!PyArg_ParseTuple(
            args, "OOOLLK", &x_o, &y_o, &out_o, &height, &width, &stream)) {
        return nullptr;
    }
    try {
        Array x(x_o), y(y_o), out(out_o);
        x.require('f', 8, 2);
        y.require('f', 8, 2);
        out.require('f', 8, 2, true);
        same_shape(x, y);
        same_shape(x, out);
        if (height < 1 || width < 1) {
            throw std::invalid_argument("source dimensions must be positive");
        }
        StreamGuard guard(stream);
        for (const auto* arr : {&x, &y, &out}) {
            arr->wait(guard.stream);
        }
        {
            Detached detached;
            if (x.size) {
                edge_score_kernel<<<
                    (x.size + 255) / 256,
                    256,
                    0,
                    guard.stream>>>(
                    x.data<double>(),
                    y.data<double>(),
                    out.data<double>(),
                    x.size,
                    height,
                    width);
            }
            guard.finish();
        }
        Py_RETURN_NONE;
    } catch (...) {
        return translate_exception();
    }
}

PyObject* sample_mapping(PyObject*, PyObject* args) {
    PyObject *x_o, *y_o, *indices_o, *out_o;
    unsigned long long stream;
    if (!PyArg_ParseTuple(
            args, "OOOOK", &x_o, &y_o, &indices_o, &out_o, &stream)) {
        return nullptr;
    }
    try {
        Array x(x_o), y(y_o), indices(indices_o), out(out_o);
        x.require('f', 8, 2);
        y.require('f', 8, 2);
        indices.require('i', 8, 2);
        out.require('f', 8, 2, true);
        same_shape(x, y);
        same_shape(indices, out);
        if (indices.shape[1] != 2) {
            throw std::invalid_argument(
                "mapping indices must have shape (N, 2)");
        }
        StreamGuard guard(stream);
        for (const auto* arr : {&x, &y, &indices, &out}) {
            arr->wait(guard.stream);
        }
        {
            Detached detached;
            const Index count = indices.shape[0];
            if (count) {
                sample_mapping_kernel<<<
                    (count + 255) / 256,
                    256,
                    0,
                    guard.stream>>>(
                    x.data<double>(),
                    y.data<double>(),
                    indices.data<Index>(),
                    out.data<double>(),
                    count,
                    x.shape[0],
                    x.shape[1]);
            }
            guard.finish();
        }
        Py_RETURN_NONE;
    } catch (...) {
        return translate_exception();
    }
}

}  // namespace cuphoton::imaging
