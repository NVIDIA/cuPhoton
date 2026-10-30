// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include "native_api.h"
#include "native_jacobian.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

extern "C" int xfit_library_norms(
    const void*, void*, int, int, bool, std::uintptr_t);

namespace cuphoton::xfit {

KernelSelection select_kernels(
    KernelPolicy policy, int major, int minor, int cublas_version) {
    if (policy == KernelPolicy::Portable) {
        return {};
    }
    if (policy == KernelPolicy::TestSM120
        || policy == KernelPolicy::TestSM121) {
        major = 12;
        minor = policy == KernelPolicy::TestSM121 ? 1 : 0;
    }
    return {
        major == 12 && (minor == 0 || minor == 1),
        major == 12 && minor == 0,
        major == 12 && minor == 1 && cublas_version == 130101,
    };
}

namespace {

constexpr int threads = 128;
constexpr int parameters = 8;
enum Status : std::int8_t {
    active = 0,
    f_converged = 1,
    x_converged = 2,
    g_converged = 3,
    max_evaluations = 4,
    invalid_residual = 5,
    singular = 6,
    no_progress = 7
};
using Clock = std::chrono::steady_clock;

void check(cudaError_t error) {
    if (error != cudaSuccess) {
        // This failure is consumed here; do not report it again at the next
        // successful call's launch check.
        cudaGetLastError();
        throw std::runtime_error(
            std::string("xFit CUDA: ") + cudaGetErrorString(error));
    }
}

void check(cublasStatus_t error) {
    if (error != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            "xFit cuBLAS status " + std::to_string(int(error)));
    }
}

struct DeviceGuard {
    int previous;
    explicit DeviceGuard(int device) {
        check(cudaGetDevice(&previous));
        if (previous != device) {
            check(cudaSetDevice(device));
        }
    }
    ~DeviceGuard() {
        cudaSetDevice(previous);
    }
};

template <typename T>
struct Work {
    T *jacobian, *packed_jacobian, *transposed, *compact_residual;
    T *gradient, *hessian, *step;
    T *trial_x, *trial_residual, *damping, *cost, *trial_cost, *residual_norm;
    double *system, *step_packed, *step_norm, *scale, *scaled_step, *prediction;
    double *matrix_packed, *solve_packed;
    double *wide_jacobian, *wide_transposed, *wide_residual;
    double *wide_gradient, *wide_hessian;
    int *rejected, *indices, *accepted, *pivots, *factor_info;
    double **matrix_ptrs, **rhs_ptrs;
};

template <typename T>
__device__ T maximum(T a, T b) {
    return isnan(a) || isnan(b) ? T(NAN) : (a > b ? a : b);
}

// Preserve individual CuPy arithmetic boundaries while allowing library
// transcendental implementations to use their normal internal contraction.
template <typename T>
__device__ T rn_add(T a, T b) {
    if constexpr (sizeof(T) == 8) {
        return __dadd_rn(a, b);
    } else {
        return __fadd_rn(a, b);
    }
}

template <typename T>
__device__ T rn_sub(T a, T b) {
    if constexpr (sizeof(T) == 8) {
        return __dsub_rn(a, b);
    } else {
        return __fsub_rn(a, b);
    }
}

template <typename T>
__device__ T rn_mul(T a, T b) {
    if constexpr (sizeof(T) == 8) {
        return __dmul_rn(a, b);
    } else {
        return __fmul_rn(a, b);
    }
}

template <typename T>
__device__ T rn_div(T a, T b) {
    if constexpr (sizeof(T) == 8) {
        return __ddiv_rn(a, b);
    } else {
        return __fdiv_rn(a, b);
    }
}

// Explicit conversions preserve FP32 subnormals even in the model unit's
// flush-to-zero build, matching the shared solver's linear algebra workspace.
template <typename Output, typename Input>
__device__ Output workspace_cast(Input value) {
    if constexpr (sizeof(Output) == sizeof(Input)) {
        return value;
    } else if constexpr (sizeof(Output) == 8) {
        double result;
        asm("cvt.f64.f32 %0, %1;" : "=d"(result) : "f"(value));
        return result;
    } else {
        float result;
        asm("cvt.rn.f32.f64 %0, %1;" : "=f"(result) : "d"(value));
        return result;
    }
}

template <typename Input, typename Output>
__global__ void convert_workspace(
    const Input* input, Output* output, std::size_t count) {
    std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = workspace_cast<Output>(input[index]);
    }
}

// Combine per-thread finite-value flags with the same block reduction in both paths.
template <typename T>
__device__ T sum(T value) {
    __shared__ T partial[threads / 32];
    for (int offset = 16; offset; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    if (threadIdx.x % 32 == 0) {
        partial[threadIdx.x / 32] = value;
    }
    __syncthreads();
    value = threadIdx.x < threads / 32 ? partial[threadIdx.x] : T(0);
    if (threadIdx.x < 32) {
        for (int offset = 16; offset; offset /= 2) {
            value += __shfl_down_sync(0xffffffff, value, offset);
        }
    }
    // Prevent the next reduction from overwriting shared values early.
    __syncthreads();
    return value;
}

template <typename T>
struct Gaussian {
    T amplitude, sx, sy, cosine, sine, ix, iy, xp, yp, xn, yn, cube_exponent;

    __device__ explicit Gaussian(const T* x, T exponent) {
        cube_exponent = exponent;
        amplitude = x[0];
        sx = exp(x[1]);
        sy = exp(x[2]);
        if (!isfinite(sx) || sx <= 0) {
            sx = T(NAN);
        }
        if (!isfinite(sy) || sy <= 0) {
            sy = T(NAN);
        }
        cosine = cos(x[3]);
        sine = sin(x[3]);
        ix = rn_div(T(1), rn_mul(sx, sx));
        iy = rn_div(T(1), rn_mul(sy, sy));
        xp = x[4];
        yp = x[5];
        xn = x[6];
        yn = x[7];
    }

    __device__ void star(T x, T y, T* out, bool derivatives) const {
        T xr = rn_add(rn_mul(cosine, x), rn_mul(sine, y));
        T yr = rn_sub(rn_mul(cosine, y), rn_mul(sine, x));
        T unit = exp(rn_mul(
            T(-0.5),
            rn_add(rn_mul(rn_mul(xr, xr), ix), rn_mul(rn_mul(yr, yr), iy))));
        T value = rn_mul(amplitude, unit);
        out[0] = value;
        if (!derivatives) {
            return;
        }
        out[1] = unit;
        out[2] = rn_div(rn_mul(rn_mul(value, xr), xr), pow(sx, cube_exponent));
        out[3] = rn_div(rn_mul(rn_mul(value, yr), yr), pow(sy, cube_exponent));
        out[4] = rn_mul(rn_mul(rn_mul(-value, xr), yr), rn_sub(ix, iy));
        out[5] = rn_mul(
            value,
            rn_sub(
                rn_mul(rn_mul(cosine, xr), ix), rn_mul(rn_mul(sine, yr), iy)));
        out[6] = rn_mul(
            value,
            rn_add(
                rn_mul(rn_mul(sine, xr), ix), rn_mul(rn_mul(cosine, yr), iy)));
    }

    __device__ T observation(int index, const Batch& b, T* derivative) const {
        int pixels = b.height * b.width;
        int plane = index / pixels;
        int pixel = index % pixels;
        T x = rn_sub(T(pixel % b.width), rn_div(T(b.width - 1), T(2)));
        T y = rn_sub(T(pixel / b.width), rn_div(T(b.height - 1), T(2)));
        T pos[7], neg[7];
        star(rn_sub(x, xp), rn_sub(y, yp), pos, derivative != nullptr);
        star(rn_sub(x, xn), rn_sub(y, yn), neg, derivative != nullptr);
        T value = plane == 0 ? rn_sub(pos[0], neg[0])
                             : (plane == 1 ? pos[0] : neg[0]);
        if (derivative) {
            for (int p = 0; p < 4; ++p) {
                derivative[p] = plane == 0
                    ? rn_sub(pos[p + 1], neg[p + 1])
                    : (plane == 1 ? pos[p + 1] : neg[p + 1]);
            }
            derivative[4] = plane < 2 ? pos[5] : T(0);
            derivative[5] = plane < 2 ? pos[6] : T(0);
            derivative[6] = plane == 0 ? -neg[5] : (plane == 2 ? neg[5] : T(0));
            derivative[7] = plane == 0 ? -neg[6] : (plane == 2 ? neg[6] : T(0));
            // Match the public physical-width Jacobian followed by its chain rule.
            derivative[1] = rn_mul(derivative[1], sx);
            derivative[2] = rn_mul(derivative[2], sy);
        }
        return value;
    }
};

template <typename T>
__global__ void initialize(Batch b, Settings s, Work<T> w) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= b.count) {
        return;
    }
    b.status[row] = active;
    b.evaluations[row] = 1;
    b.diagnostic_current[row] = false;
    w.damping[row] = T(s.initial_damping);
    w.rejected[row] = 0;
}

template <typename T>
__global__ void residual(Batch b, const T* x, T* output, T cube_exponent) {
    int row = blockIdx.x;
    if (b.status[row] != active) {
        return;
    }
    int observations = b.height * b.width * b.planes;
    std::size_t base = std::size_t(row) * observations;
    Gaussian<T> model(x + row * parameters, cube_exponent);
    const T* images = static_cast<const T*>(b.images);
    const T* weights = static_cast<const T*>(b.weights);
    for (std::size_t i = threadIdx.x; i < std::size_t(observations);
        i += blockDim.x) {
        T value = model.observation(i, b, nullptr);
        output[base + i] =
            rn_mul(rn_sub(value, images[base + i]), weights[base + i]);
    }
}

// Fuse trial residual evaluation and packing for the RTX small-batch path.
namespace fused_trial_residual_fp64 {

constexpr int kThreads = 32;

struct FastDiv {
    uint32_t multiplier;
    uint32_t shift;

    __host__ explicit FastDiv(uint32_t divisor) {
        uint32_t logarithm = 0;
        while ((1ull << logarithm) < divisor) {
            ++logarithm;
        }
        shift = 31 + logarithm;
        multiplier =
            static_cast<uint32_t>(((1ull << shift) + divisor - 1) / divisor);
    }

    __device__ __forceinline__ uint32_t div(uint32_t numerator) const {
        return static_cast<uint32_t>(
            (static_cast<uint64_t>(numerator) * multiplier) >> shift);
    }
};

__device__ __forceinline__ double inverse_square_width(double logarithm) {
    double width = exp(logarithm);
    width = isfinite(width) && width > 0.0
        ? width
        : __longlong_as_double(0x7ff8000000000000ll);
    return __ddiv_rn(1.0, __dmul_rn(width, width));
}

__device__ __forceinline__ double lobe_argument(
    double coord_x,
    double coord_y,
    double position_x,
    double position_y,
    double cosine,
    double sine,
    double inverse_x,
    double inverse_y) {
    const double delta_x = __dsub_rn(coord_x, position_x);
    const double delta_y = __dsub_rn(coord_y, position_y);
    const double rotated_x =
        __dadd_rn(__dmul_rn(cosine, delta_x), __dmul_rn(sine, delta_y));
    const double rotated_y =
        __dsub_rn(__dmul_rn(cosine, delta_y), __dmul_rn(sine, delta_x));
    const double square = __dadd_rn(
        __dmul_rn(__dmul_rn(rotated_x, rotated_x), inverse_x),
        __dmul_rn(__dmul_rn(rotated_y, rotated_y), inverse_y));
    return __dmul_rn(-0.5, square);
}

template <int Planes>
__global__ void __launch_bounds__(kThreads) fused_residual(
    const double* __restrict__ trial_x,
    const double* __restrict__ weights,
    const int32_t* __restrict__ indices,
    const double* __restrict__ images,
    const unsigned long long* __restrict__ current,
    double* __restrict__ trial,
    unsigned long long* __restrict__ compact_current,
    double* __restrict__ compact_trial,
    int selected,
    int height,
    int width,
    int pixels,
    FastDiv div_width) {
    const int selection = blockIdx.y + static_cast<int>(blockIdx.z) * 65535;
    if (selection >= selected) {
        return;
    }
    const int pixel = blockIdx.x * kThreads + threadIdx.x;
    const int lane = threadIdx.x & 31;
    const bool active = pixel < pixels;
    const int64_t source = __ldg(indices + selection);
    const int64_t full_offset = source * (int64_t(Planes) * pixels) + pixel;
    const int64_t compact_offset =
        int64_t(selection) * (int64_t(Planes) * pixels) + pixel;
    double data[Planes];
    double weight[Planes];
    unsigned long long previous[Planes];
#pragma unroll
    for (int plane = 0; plane < Planes; ++plane) {
        data[plane] = active
            ? __ldg(images + full_offset + int64_t(plane) * pixels)
            : 0.0;
        weight[plane] = active
            ? __ldg(weights + full_offset + int64_t(plane) * pixels)
            : 0.0;
        previous[plane] =
            active ? __ldg(current + full_offset + int64_t(plane) * pixels) : 0;
    }

    const double parameter = __ldg(trial_x + source * 8 + (lane & 7));
    const double log_width =
        __shfl_sync(0xffffffffu, parameter, 1 + (lane & 1));
    const double angle = __shfl_sync(0xffffffffu, parameter, 3);
    double inverse = 0.0;
    double sine = 0.0;
    double cosine = 0.0;
    if (lane < 2) {
        inverse = inverse_square_width(log_width);
    }
    if (lane == 2) {
        sincos(angle, &sine, &cosine);
    }
    const double amplitude = __shfl_sync(0xffffffffu, parameter, 0);
    const double positive_x = __shfl_sync(0xffffffffu, parameter, 4);
    const double positive_y = __shfl_sync(0xffffffffu, parameter, 5);
    const double negative_x = __shfl_sync(0xffffffffu, parameter, 6);
    const double negative_y = __shfl_sync(0xffffffffu, parameter, 7);
    const double inverse_x = __shfl_sync(0xffffffffu, inverse, 0);
    const double inverse_y = __shfl_sync(0xffffffffu, inverse, 1);
    sine = __shfl_sync(0xffffffffu, sine, 2);
    cosine = __shfl_sync(0xffffffffu, cosine, 2);
    if (!active) {
        return;
    }

    const uint32_t row = div_width.div(static_cast<uint32_t>(pixel));
    const uint32_t column = static_cast<uint32_t>(pixel) - row * width;
    const double coord_x =
        __dsub_rn(static_cast<double>(column), 0.5 * (width - 1));
    const double coord_y =
        __dsub_rn(static_cast<double>(row), 0.5 * (height - 1));
    const double positive_argument = lobe_argument(
        coord_x,
        coord_y,
        positive_x,
        positive_y,
        cosine,
        sine,
        inverse_x,
        inverse_y);
    const double negative_argument = lobe_argument(
        coord_x,
        coord_y,
        negative_x,
        negative_y,
        cosine,
        sine,
        inverse_x,
        inverse_y);
    const double positive = __dmul_rn(amplitude, exp(positive_argument));
    const double negative = __dmul_rn(amplitude, exp(negative_argument));
    double result[Planes];
    result[0] =
        __dmul_rn(__dsub_rn(__dsub_rn(positive, negative), data[0]), weight[0]);
    if constexpr (Planes == 3) {
        result[1] = __dmul_rn(__dsub_rn(positive, data[1]), weight[1]);
        result[2] = __dmul_rn(__dsub_rn(negative, data[2]), weight[2]);
    }
#pragma unroll
    for (int plane = 0; plane < Planes; ++plane) {
        const int64_t plane_offset = int64_t(plane) * pixels;
        trial[full_offset + plane_offset] = result[plane];
        compact_trial[compact_offset + plane_offset] = result[plane];
        compact_current[compact_offset + plane_offset] = previous[plane];
    }
}

void launch(const Batch& b, Work<double> w, int count, cudaStream_t stream) {
    const int pixels = b.height * b.width;
    const dim3 grid(
        (uint64_t(pixels) + kThreads - 1) / kThreads,
        count < 65535 ? count : 65535,
        (uint64_t(count) + 65534) / 65535);
    fused_residual<1><<<grid, kThreads, 0, stream>>>(
        w.trial_x,
        static_cast<const double*>(b.weights),
        w.indices,
        static_cast<const double*>(b.images),
        static_cast<const unsigned long long*>(b.residuals),
        w.trial_residual,
        reinterpret_cast<unsigned long long*>(w.compact_residual),
        w.transposed,
        count,
        b.height,
        b.width,
        pixels,
        FastDiv(b.width));
}

}  // namespace fused_trial_residual_fp64

template <typename T>
__global__ void initial_status(Batch b, int* any_active) {
    int row = blockIdx.x;
    int observations = b.height * b.width * b.planes;
    const T* x = static_cast<const T*>(b.x) + row * parameters;
    const T* r =
        static_cast<const T*>(b.residuals) + std::size_t(row) * observations;
    int invalid = 0;
    for (int p = threadIdx.x; p < parameters; p += blockDim.x) {
        invalid |= !isfinite(x[p]);
    }
    for (std::size_t i = threadIdx.x; i < std::size_t(observations);
        i += blockDim.x) {
        invalid |= !isfinite(r[i]);
    }
    invalid = sum(invalid);
    if (threadIdx.x == 0) {
        if (invalid) {
            b.status[row] = invalid_residual;
        } else {
            atomicExch(any_active, 1);
        }
    }
}

// Stable row order matches CuPy's boolean indexing at each solver boundary.
template <typename T>
__global__ void compact_rows(
    Batch b,
    Settings s,
    Work<T> w,
    int* count,
    bool accepted_only,
    bool enforce_budget,
    int finite_count) {
    // Fused outputs use the original selected-row order. Consume all flags
    // before the stable selection below overwrites that row list. factor_info
    // is scratch here; cuBLAS writes its integer results later in the iteration.
    const bool* finite_row = reinterpret_cast<const bool*>(w.factor_info);
    for (int slot = 0; slot < finite_count; ++slot) {
        int row = w.indices[slot];
        if (finite_row[slot]) {
            b.diagnostic_current[row] = true;
        } else {
            b.status[row] = invalid_residual;
        }
    }
    int size = 0;
    for (int row = 0; row < b.count; ++row) {
        if (enforce_budget && b.status[row] == active
            && b.evaluations[row] >= s.max_evaluations) {
            b.status[row] = max_evaluations;
        }
        if (accepted_only ? w.accepted[row] != 0 : b.status[row] == active) {
            w.indices[size++] = row;
        }
    }
    *count = size;
}

// Spatially tiled FP64 Jacobian for the RTX primary small-batch path.
// Per-row finite flags are consumed by the existing stable row selection.
namespace tiled_jacobian_fp64 {

constexpr int kHelperThreads =
    96;  // three helper warps for the per-row scalars

__device__ __forceinline__ bool nonfinite(double v) {
    return (__double2hiint(v) & 0x7ff00000) == 0x7ff00000;
}

// Byte-sized, naturally aligned relaxed atomic store at device scope. Concurrent
// clears of the same flag are morally strong (no data race) and never touch
// neighbouring bytes, so any K is valid.
__device__ __forceinline__ void atomic_clear_flag(bool* flag) {
    const unsigned short zero = 0;
    asm volatile("st.relaxed.gpu.u8 [%0], %1;" ::"l"(flag), "h"(zero)
        : "memory");
}

struct StarPart {
    double xr, yr, value, unit, angle, px, py;
};

// Everything in star() that does not need the pow() results.
__device__ __forceinline__ void star_part(
    double cx,
    double cy,
    double ox,
    double oy,
    double c,
    double s,
    double ix,
    double iy,
    double ixmiy,
    double amp,
    StarPart& o) {
    const double dx = __dsub_rn(cx, ox);
    const double dy = __dsub_rn(cy, oy);
    const double xr = __dadd_rn(__dmul_rn(c, dx), __dmul_rn(s, dy));
    const double yr = __dsub_rn(__dmul_rn(c, dy), __dmul_rn(s, dx));
    const double q = __dadd_rn(
        __dmul_rn(__dmul_rn(xr, xr), ix), __dmul_rn(__dmul_rn(yr, yr), iy));
    const double unit = exp(__dmul_rn(-0.5, q));
    const double value = __dmul_rn(amp, unit);
    o.xr = xr;
    o.yr = yr;
    o.value = value;
    o.unit = unit;
    o.angle = __dmul_rn(__dmul_rn(__dmul_rn(-value, xr), yr), ixmiy);
    o.px = __dmul_rn(
        value,
        __dsub_rn(
            __dmul_rn(__dmul_rn(c, xr), ix), __dmul_rn(__dmul_rn(s, yr), iy)));
    o.py = __dmul_rn(
        value,
        __dadd_rn(
            __dmul_rn(__dmul_rn(s, xr), ix), __dmul_rn(__dmul_rn(c, yr), iy)));
}

// Writes one plane's 8 weighted derivatives to both layouts; returns nonfinite.
__device__ __forceinline__ bool store8(
    const double (&v)[8],
    double w,
    double* __restrict__ packed,
    double* __restrict__ transposed,
    long long k,
    long long M,
    long long m) {
    double j[8];
    bool bad = false;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        j[i] = __dmul_rn(v[i], w);
        bad |= nonfinite(j[i]);
    }
    double* pk = packed + (k * 8) * M + m;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        pk[i * M] = j[i];
    }
    double2* tr = reinterpret_cast<double2*>(transposed + (k * M + m) * 8);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        tr[i] = make_double2(j[2 * i], j[2 * i + 1]);
    }
    return bad;
}

// blockDim.x = pix_threads + kHelperThreads. Threads [0, pix_threads) own one
// pixel (all planes) of a tile; the three helper warps compute the row scalars:
// warp A lanes 0/1 run exp -> validity -> 1/(s*s) and then pow(s, exponent),
// warps B/C run cos and sin, so the long scalar chains run concurrently and
// pow overlaps the pixel exponentials.
__global__ void __launch_bounds__(256 + kHelperThreads) jacobian_pack_kernel(
    const double* __restrict__ x,
    const double* __restrict__ weights,
    const int* __restrict__ indices,
    const double* __restrict__ residuals,
    double* __restrict__ packed,
    double* __restrict__ transposed,
    double* __restrict__ compact,
    bool* __restrict__ finite_row,
    long long K,
    int P,
    int H,
    int W,
    int pix_threads,
    long long tiles,
    double exponent) {
    __shared__ double sh[8];
    const int t = threadIdx.x;
    const int helper = t - pix_threads;
    const long long HW = (long long)H * W;
    const long long M = HW * P;
    const long long total = K * tiles;
    const double half_w = __dmul_rn((double)(W - 1), 0.5);
    const double half_h = __dmul_rn((double)(H - 1), 0.5);

    for (long long item = blockIdx.x; item < total; item += gridDim.x) {
        const long long k = item / tiles;
        const long long tile = item - k * tiles;
        const long long r = indices[k];
        const double* xrow = x + r * 8;

        double s_self = 0.0;
        if (helper == 0 || helper == 1) {
            double s = exp(xrow[1 + helper]);
            if (!(isfinite(s) && s > 0.0)) {
                s = __longlong_as_double(0x7ff8000000000000LL);
            }
            s_self = s;
            sh[helper] = s;
            sh[2 + helper] = __ddiv_rn(1.0, __dmul_rn(s, s));
        } else if (helper == 32) {
            sh[6] = cos(xrow[3]);
        } else if (helper == 64) {
            sh[7] = sin(xrow[3]);
        }
        __syncthreads();

        const long long pix = tile * pix_threads + t;
        const bool active = helper < 0 && pix < HW;
        StarPart a, b;
        double w0 = 0.0, w1 = 0.0, w2 = 0.0;
        if (helper == 0 || helper == 1) {
            sh[4 + helper] = pow(s_self, exponent);
        } else if (active) {
            const double c = sh[6], s = sh[7], ix = sh[2], iy = sh[3];
            const double ixmiy = __dsub_rn(ix, iy);
            const double amp = xrow[0];
            const int row = (int)(pix / W);
            const int col = (int)(pix - (long long)row * W);
            const double cx = __dsub_rn((double)col, half_w);
            const double cy = __dsub_rn((double)row, half_h);
            const double* wrow = weights + r * M + pix;
            w0 = wrow[0];
            if (P == 3) {
                w1 = wrow[HW];
                w2 = wrow[2 * HW];
            }
            star_part(cx, cy, xrow[4], xrow[5], c, s, ix, iy, ixmiy, amp, a);
            star_part(cx, cy, xrow[6], xrow[7], c, s, ix, iy, ixmiy, amp, b);
            const double* rrow = residuals + r * M + pix;
            double* crow = compact + k * M + pix;
            crow[0] = rrow[0];
            if (P == 3) {
                crow[HW] = rrow[HW];
                crow[2 * HW] = rrow[2 * HW];
            }
        }
        __syncthreads();

        bool bad = false;
        if (active) {
            const double sx = sh[0], sy = sh[1], sxc = sh[4], syc = sh[5];
            const double awx =
                __ddiv_rn(__dmul_rn(__dmul_rn(a.value, a.xr), a.xr), sxc);
            const double awy =
                __ddiv_rn(__dmul_rn(__dmul_rn(a.value, a.yr), a.yr), syc);
            const double bwx =
                __ddiv_rn(__dmul_rn(__dmul_rn(b.value, b.xr), b.xr), sxc);
            const double bwy =
                __ddiv_rn(__dmul_rn(__dmul_rn(b.value, b.yr), b.yr), syc);
            {
                double v[8];
                v[0] = __dsub_rn(a.unit, b.unit);
                v[1] = __dmul_rn(__dsub_rn(awx, bwx), sx);
                v[2] = __dmul_rn(__dsub_rn(awy, bwy), sy);
                v[3] = __dsub_rn(a.angle, b.angle);
                v[4] = a.px;
                v[5] = a.py;
                v[6] = -b.px;
                v[7] = -b.py;
                bad |= store8(v, w0, packed, transposed, k, M, pix);
            }
            if (P == 3) {
                double v[8];
                v[0] = a.unit;
                v[1] = __dmul_rn(awx, sx);
                v[2] = __dmul_rn(awy, sy);
                v[3] = a.angle;
                v[4] = a.px;
                v[5] = a.py;
                v[6] = 0.0;
                v[7] = 0.0;
                bad |= store8(v, w1, packed, transposed, k, M, HW + pix);
                v[0] = b.unit;
                v[1] = __dmul_rn(bwx, sx);
                v[2] = __dmul_rn(bwy, sy);
                v[3] = b.angle;
                v[4] = 0.0;
                v[5] = 0.0;
                v[6] = b.px;
                v[7] = b.py;
                bad |= store8(v, w2, packed, transposed, k, M, 2 * HW + pix);
            }
        }
        // finite_row[k] was preset to true in stream order before this launch.
        // One thread per CTA clears it with a byte-sized atomic store when any
        // weighted derivative in the tile is nonfinite. The barrier also protects
        // sh[] before the next work item.
        if (__syncthreads_or(bad) && t == 0) {
            atomic_clear_flag(finite_row + k);
        }
    }
}

void launch(
    const Batch& b,
    Work<double> w,
    int count,
    int multiprocessors,
    double exponent,
    cudaStream_t stream) {
    const long long HW = static_cast<long long>(b.height) * b.width;
    int pix_threads = 32;
    while (pix_threads < 256
        && count * ((HW + pix_threads - 1) / pix_threads) > multiprocessors) {
        pix_threads *= 2;
    }
    const long long tiles = (HW + pix_threads - 1) / pix_threads;
    const long long total = count * tiles;
    const long long max_grid = 1LL << 20;
    const unsigned grid =
        static_cast<unsigned>(total < max_grid ? total : max_grid);
    auto* finite_row = reinterpret_cast<bool*>(w.factor_info);
    check(cudaMemsetAsync(finite_row, 1, std::size_t(count), stream));
    jacobian_pack_kernel<<<grid, pix_threads + kHelperThreads, 0, stream>>>(
        static_cast<const double*>(b.x),
        static_cast<const double*>(b.weights),
        w.indices,
        static_cast<const double*>(b.residuals),
        w.packed_jacobian,
        w.transposed,
        w.compact_residual,
        finite_row,
        count,
        b.planes,
        b.height,
        b.width,
        pix_threads,
        tiles,
        exponent);
    check(cudaGetLastError());
}

}  // namespace tiled_jacobian_fp64

template <typename T, bool Pack = false>
__global__ void evaluate_jacobian(Batch b, Work<T> w, T cube_exponent) {
    int slot = blockIdx.x;
    int row = w.indices[slot];
    int observations = b.height * b.width * b.planes;
    std::size_t base = std::size_t(row) * observations;
    Gaussian<T> model(
        static_cast<const T*>(b.x) + row * parameters, cube_exponent);
    const T* weights = static_cast<const T*>(b.weights);
    const T* residuals = static_cast<const T*>(b.residuals);
    int invalid = 0;
    for (std::size_t i = threadIdx.x; i < std::size_t(observations);
        i += blockDim.x) {
        T derivative[parameters];
        model.observation(i, b, derivative);
        for (int p = 0; p < parameters; ++p) {
            T value = rn_mul(derivative[p], weights[base + i]);
            invalid |= !isfinite(value);
            if constexpr (Pack) {
                w.packed_jacobian
                    [(std::size_t(slot) * parameters + p) * observations + i] =
                    value;
                w.transposed
                    [(std::size_t(slot) * observations + i) * parameters + p] =
                    value;
            } else {
                w.jacobian
                    [(std::size_t(row) * parameters + p) * observations + i] =
                    value;
            }
        }
        if constexpr (Pack) {
            w.compact_residual[std::size_t(slot) * observations + i] =
                residuals[base + i];
        }
    }
    invalid = sum(invalid);
    if (threadIdx.x == 0) {
        if (invalid) {
            b.status[row] = invalid_residual;
        } else {
            b.diagnostic_current[row] = true;
        }
    }
}

template <typename T>
__global__ void pack_equations(Batch b, Work<T> w) {
    int slot = blockIdx.x;
    int row = w.indices[slot];
    int observations = b.height * b.width * b.planes;
    const T* r = static_cast<const T*>(b.residuals);
    for (std::size_t i = threadIdx.x; i < std::size_t(observations);
        i += blockDim.x) {
        w.compact_residual[std::size_t(slot) * observations + i] =
            r[std::size_t(row) * observations + i];
        for (int p = 0; p < parameters; ++p) {
            T value =
                w.jacobian
                    [(std::size_t(row) * parameters + p) * observations + i];
            w.packed_jacobian
                [(std::size_t(slot) * parameters + p) * observations + i] =
                value;
            w.transposed
                [(std::size_t(slot) * observations + i) * parameters + p] =
                value;
        }
    }
}

__device__ __forceinline__ double hessian_chain_fp64(
    const double* a, const double* b, int begin, int end, double result) {
    constexpr int prefetch = 16;
    int i = begin;
    for (; i + prefetch <= end; i += prefetch) {
        double av[prefetch], bv[prefetch];
#pragma unroll
        for (int j = 0; j < prefetch; ++j) {
            av[j] = a[i + j];
            bv[j] = b[i + j];
        }
#pragma unroll
        for (int j = 0; j < prefetch; ++j) {
            result = __fma_rn(av[j], bv[j], result);
        }
    }
    for (; i < end; ++i) {
        result = __fma_rn(a[i], b[i], result);
    }
    return result;
}

__global__ __launch_bounds__(128) void hessian_single_fp64(
    const double* __restrict__ jacobian, double* __restrict__ hessian) {
    constexpr int observations = 51 * 51;
    __shared__ double a[observations];
    __shared__ double b[observations];
    __shared__ double partials[5];
    // One block owns each upper-triangular pair. The surviving Jacobian rows
    // are finite, so swapping the two FMA multiplicands preserves the result.
    int pair = blockIdx.x;
    int row = 0;
#pragma unroll
    for (int width = parameters; width > 1; --width) {
        if (pair >= width) {
            pair -= width;
            ++row;
        } else {
            break;
        }
    }
    int column = row + pair;
    int tid = threadIdx.x;
    for (int i = tid; i < observations; i += threads) {
        a[i] = jacobian[row * observations + i];
    }
    if (row != column) {
        for (int i = tid; i < observations; i += threads) {
            b[i] = jacobian[column * observations + i];
        }
    }
    __syncthreads();
    if (tid < 5) {
        int begin = tid * 522;
        int end = tid == 4 ? observations : begin + 522;
        const double* staged_b = row == column ? a : b;
        partials[tid] = hessian_chain_fp64(a, staged_b, begin, end, 0.0);
    }
    __syncthreads();
    if (tid == 0) {
        double result = 0.0;
        // Parallelize the five chains while retaining their original order.
#pragma unroll
        for (int p = 0; p < 5; ++p) {
            result = __dadd_rn(partials[p], result);
        }
        hessian[column * parameters + row] = result;
        if (column != row) {
            hessian[row * parameters + column] = result;
        }
    }
}

__global__ void hessian_small_fp64(
    const double* __restrict__ jacobian, double* __restrict__ hessian) {
    constexpr int observations = 51 * 51;
    constexpr int tile_size = 256;
    __shared__ double staged[4][tile_size];
    // Ten upper-triangular 2x2 tiles cover the 36 independent matrix entries.
    int tile = blockIdx.x;
    int tile_row = 0;
#pragma unroll
    for (int width = parameters / 2; width > 1; --width) {
        if (tile >= width) {
            tile -= width;
            ++tile_row;
        } else {
            break;
        }
    }
    int tile_column = tile_row + tile;
    bool diagonal = tile_row == tile_column;
    int row = tile_row * 2 + (threadIdx.x & 1);
    int column = tile_column * 2 + ((threadIdx.x >> 1) & 1);
    bool owner = threadIdx.x < 4 && row <= column;
    int a_column = threadIdx.x & 1;
    int b_column = ((threadIdx.x >> 1) & 1) + (diagonal ? 0 : 2);
    const double* values = jacobian + blockIdx.y * parameters * observations;
    double result = 0.0;
    for (int begin = 0; begin < observations; begin += tile_size) {
        int size = min(tile_size, observations - begin);
        int staged_columns = diagonal ? 2 : 4;
        for (int j = threadIdx.x; j < staged_columns * tile_size;
            j += blockDim.x) {
            int c = j / tile_size;
            int i = j % tile_size;
            int input_column =
                c < 2 ? tile_row * 2 + c : tile_column * 2 + c - 2;
            if (i < size) {
                staged[c][i] = values[input_column * observations + begin + i];
            }
        }
        __syncthreads();
        if (owner) {
            // Carry the chain across tiles without restarting or regrouping it.
            result = hessian_chain_fp64(
                staged[a_column], staged[b_column], 0, size, result);
        }
        __syncthreads();
    }
    if (owner) {
        double* output = hessian + blockIdx.y * 64;
        output[column * parameters + row] = result;
        if (row != column) {
            output[row * parameters + column] = result;
        }
    }
}

template <typename T>
__global__ void prepare_system(Batch b, Settings s, Work<T> w, int count) {
    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= count) {
        return;
    }
    int row = w.indices[slot];
    double scale[parameters];
    double scaled_gradient = 0;
    for (int p = 0; p < parameters; ++p) {
        // Widen before comparison or division: FP32 curvature can be
        // subnormal even when the solution and covariance are representable.
        double diagonal =
            workspace_cast<double>(w.hessian[slot * 64 + p * parameters + p]);
        scale[p] = sqrt(diagonal > 0.0 ? diagonal : 1.0);
        w.scale[row * parameters + p] = scale[p];
        double gradient = rn_div(
            workspace_cast<double>(w.gradient[slot * parameters + p]),
            scale[p]);
        w.scaled_step[row * parameters + p] = -gradient;
        scaled_gradient = maximum(scaled_gradient, abs(gradient));
    }
    // Match the shared solver's threshold in the residual dtype, including
    // FP32 rounding before comparison with the widened scaled gradient.
    if (scaled_gradient
        <= rn_mul(T(s.g_tol), maximum(T(1), w.residual_norm[slot]))) {
        b.status[row] = g_converged;
        return;
    }
    // CuPy einsum restores its Hessian with a final transpose. The raw GEMM
    // output is therefore already column-major for the following LU copy.
    for (int column = 0; column < parameters; ++column) {
        for (int line = 0; line < parameters; ++line) {
            double normalized = rn_div(
                rn_div(
                    workspace_cast<double>(
                        w.hessian[slot * 64 + column * parameters + line]),
                    scale[line]),
                scale[column]);
            double damping = rn_mul(
                workspace_cast<double>(w.damping[row]), double(line == column));
            w.system[row * 64 + column * parameters + line] =
                rn_add(normalized, damping);
        }
    }
}

template <typename T>
__global__ void pack_systems(Work<T> w, int count) {
    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= count) {
        return;
    }
    int row = w.indices[slot];
    for (int p = 0; p < 64; ++p) {
        w.matrix_packed[slot * 64 + p] = w.system[row * 64 + p];
    }
    for (int p = 0; p < parameters; ++p) {
        w.solve_packed[slot * parameters + p] =
            w.scaled_step[row * parameters + p];
    }
    w.matrix_ptrs[slot] = w.matrix_packed + slot * 64;
    w.rhs_ptrs[slot] = w.solve_packed + slot * parameters;
}

template <typename T>
__global__ void scatter_step(Batch b, Work<T> w, int count) {
    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= count) {
        return;
    }
    int row = w.indices[slot];
    bool valid = true;
    double prediction = 0;
    const T* x = static_cast<const T*>(b.x);
    for (int p = 0; p < parameters; ++p) {
        double scaled_step = w.solve_packed[slot * parameters + p];
        T value = workspace_cast<T>(
            rn_div(scaled_step, w.scale[row * parameters + p]));
        valid &= isfinite(value);
        // scaled_step still holds the normalized right-hand side, -gradient.
        prediction = rn_add(
            prediction,
            rn_mul(
                scaled_step,
                rn_add(
                    rn_mul(workspace_cast<double>(w.damping[row]), scaled_step),
                    w.scaled_step[row * parameters + p])));
        w.scaled_step[row * parameters + p] = scaled_step;
        w.step[row * parameters + p] = value;
        w.trial_x[row * parameters + p] =
            rn_add(x[row * parameters + p], value);
    }
    w.prediction[row] = rn_mul(0.5, prediction);
    if (!valid) {
        b.status[row] = singular;
    }
}

template <typename T>
__global__ void pack_residuals(Batch b, Work<T> w) {
    int slot = blockIdx.x;
    int row = w.indices[slot];
    int observations = b.height * b.width * b.planes;
    const T* r = static_cast<const T*>(b.residuals);
    for (std::size_t i = threadIdx.x; i < std::size_t(observations);
        i += blockDim.x) {
        w.compact_residual[std::size_t(slot) * observations + i] =
            r[std::size_t(row) * observations + i];
        // The transpose scratch is no longer needed until the next iteration.
        w.transposed[std::size_t(slot) * observations + i] =
            w.trial_residual[std::size_t(row) * observations + i];
    }
}

template <typename T>
__global__ void classify_trial(Batch b, Settings s, Work<T> w) {
    int slot = blockIdx.x;
    int row = w.indices[slot];
    int observations = b.height * b.width * b.planes;
    std::size_t base = std::size_t(row) * observations;
    int invalid = 0;
    for (std::size_t i = threadIdx.x; i < std::size_t(observations);
        i += blockDim.x) {
        invalid |= !isfinite(w.trial_residual[base + i]);
    }
    invalid = sum(invalid);
    __shared__ bool accepted;
    if (threadIdx.x == 0) {
        constexpr T eps = T(sizeof(T) == 8 ? 0x1p-52 : 0x1p-23);
        ++b.evaluations[row];
        T cost = rn_mul(T(0.5), w.cost[slot]);
        T trial_cost = rn_mul(T(0.5), w.trial_cost[slot]);
        T reduction = rn_sub(cost, trial_cost);
        accepted = !invalid && reduction > T(0);
        w.accepted[row] = accepted;
        if (accepted) {
            b.diagnostic_current[row] = false;
            w.damping[row] =
                maximum(eps, rn_mul(w.damping[row], T(s.damping_decrease)));
            w.rejected[row] = 0;
            T threshold = rn_mul(T(s.f_tol), maximum(T(1), cost));
            double prediction = w.prediction[row];
            if (reduction <= threshold && isfinite(prediction)
                && prediction > T(0) && prediction <= threshold
                && rn_mul(T(0.5), reduction) <= prediction) {
                b.status[row] = f_converged;
            }
        } else {
            w.damping[row] = rn_mul(w.damping[row], T(s.damping_increase));
            ++w.rejected[row];
            if (w.damping[row] > rn_div(T(1), eps) || w.rejected[row] >= 32) {
                b.status[row] = invalid ? invalid_residual : no_progress;
            }
        }
    }
    __syncthreads();
    if (accepted) {
        T* x = static_cast<T*>(b.x);
        T* r = static_cast<T*>(b.residuals);
        for (int p = threadIdx.x; p < parameters; p += blockDim.x) {
            x[row * parameters + p] = w.trial_x[row * parameters + p];
        }
        for (std::size_t i = threadIdx.x; i < std::size_t(observations);
            i += blockDim.x) {
            r[base + i] = w.trial_residual[base + i];
        }
    }
}

template <typename T>
__global__ void pack_accepted_steps(Work<T> w, int count) {
    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= count) {
        return;
    }
    int row = w.indices[slot];
    for (int p = 0; p < parameters; ++p) {
        w.step_packed[slot * parameters + p] =
            w.scaled_step[row * parameters + p];
    }
}

template <typename T>
__global__ void finish_convergence(Batch b, Settings s, Work<T> w, int count) {
    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= count) {
        return;
    }
    int row = w.indices[slot];
    if (b.status[row] == active
        && w.step_norm[slot] <= rn_mul(s.x_tol, rn_add(s.x_tol, 1.0))) {
        b.status[row] = x_converged;
    }
}

template <typename T>
__global__ void multiply_single_observation(
    const T* left, const T* right, T* output, int m, int n, int count) {
    std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= std::size_t(count) * m * n) {
        return;
    }
    std::size_t batch = index / (m * n);
    int row = (index / n) % m;
    int column = index % n;
    output[index] = rn_mul(left[batch * m + row], right[batch * n + column]);
}

}  // namespace

struct Workspace::Impl {
    KernelPolicy kernel_policy = KernelPolicy::Automatic;
    int cublas_version = 0;
    int device = 0;
    bool fused_jacobian = false;
    bool tiled_jacobian = false;
    bool fused_trial_residual = false;
    int jacobian_sms = 0;
    bool small_hessian = false;
    cudaStream_t stream = nullptr;
    cublasHandle_t handle = nullptr;
    double cube_exponent = 3.0;
    cudaEvent_t dependency = nullptr, start = nullptr, stop = nullptr;
    void* storage = nullptr;
    std::size_t capacity = 0;
    int *device_active = nullptr, *host_active = nullptr;

    ~Impl() {
        int previous = device;
        cudaGetDevice(&previous);
        cudaSetDevice(device);
        if (stream) {
            cudaStreamSynchronize(stream);
        }
        if (handle) {
            cublasDestroy(handle);
        }
        if (storage) {
            cudaFree(storage);
        }
        if (device_active) {
            cudaFree(device_active);
        }
        if (host_active) {
            cudaFreeHost(host_active);
        }
        if (dependency) {
            cudaEventDestroy(dependency);
        }
        if (start) {
            cudaEventDestroy(start);
        }
        if (stop) {
            cudaEventDestroy(stop);
        }
        if (stream) {
            cudaStreamDestroy(stream);
        }
        cudaSetDevice(previous);
    }

    template <typename T>
    Work<T> workspace(const Batch& b) {
        std::size_t count = b.count;
        std::size_t observations = std::size_t(b.height) * b.width * b.planes;
        std::size_t row_bytes = (92 + 26 * observations) * sizeof(T)
            + 162 * sizeof(double) + 12 * sizeof(int) + 2 * sizeof(double*);
        if constexpr (sizeof(T) == 4) {
            row_bytes += (72 + 17 * observations) * sizeof(double);
        }
        // CuPy allocates separate aligned arrays. Preserve those base alignments
        // without changing per-row strides or the values sent to cuBLAS.
        constexpr std::size_t array_alignment = 256;
        constexpr std::size_t pointer_alignment = alignof(void*);
        constexpr std::size_t padding =
            26 * (array_alignment - 1) + pointer_alignment;
        if (count
            > (std::numeric_limits<std::size_t>::max() - padding) / row_bytes) {
            throw std::overflow_error("xFit native workspace is too large");
        }
        std::size_t bytes = count * row_bytes + padding;
        if (bytes > capacity) {
            void* replacement = nullptr;
            check(cudaMalloc(&replacement, bytes));
            if (storage) {
                cudaFree(storage);
            }
            storage = replacement;
            capacity = bytes;
        }
        T* cursor = static_cast<T*>(storage);
        auto take = [&cursor](std::size_t size) {
            std::uintptr_t address = reinterpret_cast<std::uintptr_t>(cursor);
            address = (address + array_alignment - 1) & ~(array_alignment - 1);
            cursor = reinterpret_cast<T*>(address);
            T* result = cursor;
            cursor += size;
            return result;
        };
        Work<T> w{};
        w.jacobian = take(count * 8 * observations);
        w.packed_jacobian = take(count * 8 * observations);
        w.transposed = take(count * 8 * observations);
        w.compact_residual = take(count * observations);
        w.gradient = take(count * 8);
        w.hessian = take(count * 64);
        w.step = take(count * 8);
        w.trial_x = take(count * 8);
        w.trial_residual = take(count * observations);
        w.damping = take(count);
        w.cost = take(count);
        w.trial_cost = take(count);
        w.residual_norm = take(count);
        auto take_double = [&cursor](std::size_t size) {
            std::uintptr_t address = reinterpret_cast<std::uintptr_t>(cursor);
            address = (address + array_alignment - 1) & ~(array_alignment - 1);
            double* result = reinterpret_cast<double*>(address);
            cursor = reinterpret_cast<T*>(result + size);
            return result;
        };
        w.system = take_double(count * 64);
        w.step_packed = take_double(count * 8);
        w.step_norm = take_double(count);
        w.scale = take_double(count * 8);
        w.scaled_step = take_double(count * 8);
        w.prediction = take_double(count);
        w.matrix_packed = take_double(count * 64);
        w.solve_packed = take_double(count * 8);
        if constexpr (sizeof(T) == 4) {
            w.wide_jacobian = take_double(count * 8 * observations);
            w.wide_transposed = take_double(count * 8 * observations);
            w.wide_residual = take_double(count * observations);
            w.wide_gradient = take_double(count * 8);
            w.wide_hessian = take_double(count * 64);
        }
        int* integers = reinterpret_cast<int*>(cursor);
        w.rejected = integers;
        integers += count;
        w.indices = integers;
        integers += count;
        w.accepted = integers;
        integers += count;
        w.pivots = integers;
        integers += count * 8;
        w.factor_info = integers;
        integers += count;
        std::uintptr_t pointer_address =
            reinterpret_cast<std::uintptr_t>(integers);
        pointer_address = (pointer_address + pointer_alignment - 1)
            & ~(pointer_alignment - 1);
        w.matrix_ptrs = reinterpret_cast<double**>(pointer_address);
        w.rhs_ptrs = w.matrix_ptrs + count;
        return w;
    }

    void synchronize(Timings& timings) {
        auto begin = Clock::now();
        cudaError_t error = cudaStreamSynchronize(stream);
        timings.synchronize_seconds +=
            std::chrono::duration<double>(Clock::now() - begin).count();
        check(error);
    }

    void read_active(Timings& timings) {
        check(cudaMemcpyAsync(
            host_active,
            device_active,
            sizeof(int),
            cudaMemcpyDeviceToHost,
            stream));
        synchronize(timings);
    }

    template <typename T>
    Work<T> initialize_iteration_state(
        const Batch& b, const Settings& s, Timings& timings) {
        Work<T> w = workspace<T>(b);
        int blocks = (b.count + threads - 1) / threads;
        initialize<<<blocks, threads, 0, stream>>>(b, s, w);
        residual<T><<<b.count, threads, 0, stream>>>(
            b,
            static_cast<const T*>(b.x),
            static_cast<T*>(b.residuals),
            T(cube_exponent));
        check(cudaMemsetAsync(device_active, 0, sizeof(int), stream));
        initial_status<T><<<b.count, threads, 0, stream>>>(b, device_active);
        check(cudaGetLastError());
        read_active(timings);
        return w;
    }

    template <typename T>
    int select_rows(
        const Batch& b,
        const Settings& s,
        Work<T> w,
        Timings& timings,
        bool accepted_only = false,
        bool enforce_budget = false,
        int finite_count = 0) {
        compact_rows<<<1, 1, 0, stream>>>(
            b,
            s,
            w,
            device_active,
            accepted_only,
            enforce_budget,
            finite_count);
        check(cudaGetLastError());
        read_active(timings);
        return *host_active;
    }

    template <typename T>
    void matmul(
        const T* left,
        const T* right,
        T* output,
        int m,
        int n,
        int k,
        int count) {
        // Einsum squeezes a singleton contraction axis, leaving an elementwise
        // product. Match that route and preserve signed zero without GEMM's sum.
        if (k == 1) {
            std::size_t size = std::size_t(count) * m * n;
            multiply_single_observation<<<
                (size + threads - 1) / threads,
                threads,
                0,
                stream>>>(left, right, output, m, n, count);
            check(cudaGetLastError());
            return;
        }
        T one = T(1), zero = T(0);
        cudaDataType type = sizeof(T) == 8 ? CUDA_R_64F : CUDA_R_32F;
        // Match CuPy 14.1.1's N,N strided-batched route, operand swap,
        // legacy cudaDataType compute overload, and default algorithm.
        check(cublasGemmStridedBatchedEx(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            n,
            m,
            k,
            &one,
            right,
            type,
            n,
            static_cast<long long>(k) * n,
            left,
            type,
            k,
            static_cast<long long>(m) * k,
            &zero,
            output,
            type,
            n,
            static_cast<long long>(m) * n,
            count,
            type,
            CUBLAS_GEMM_DEFAULT));
    }

    template <typename T>
    void norms(const T* input, T* output, int count, int observations) {
        check(
            static_cast<cudaError_t>(xfit_library_norms(
                input,
                output,
                count,
                observations,
                sizeof(T) == 8,
                reinterpret_cast<std::uintptr_t>(stream))));
    }

    template <typename T>
    void library_solve(Work<T> w, int count) {
        int info = 0;
        check(cublasDgetrfBatched(
            handle, 8, w.matrix_ptrs, 8, w.pivots, w.factor_info, count));
        check(cublasDgetrsBatched(
            handle,
            CUBLAS_OP_N,
            8,
            1,
            const_cast<const double* const*>(w.matrix_ptrs),
            8,
            w.pivots,
            w.rhs_ptrs,
            8,
            &info,
            count));
        // CuPy's default linalg error mode ignores per-matrix factor_info.
        // Nonfinite solutions are rejected by scatter_step, just as in Python.
        if (info != 0) {
            throw std::runtime_error(
                "xFit getrsBatched info " + std::to_string(info));
        }
    }

    template <typename Input, typename Output>
    void convert(const Input* input, Output* output, std::size_t count) {
        convert_workspace<<<
            (count + threads - 1) / threads,
            threads,
            0,
            stream>>>(input, output, count);
        check(cudaGetLastError());
    }

    template <typename T>
    void normal_equations(
        Batch b, Work<T> w, int count, int observations, Timings& timings) {
        if constexpr (sizeof(T) == 4) {
            // Widen before multiplying, then retain FP32 solver state.
            std::size_t values = std::size_t(count) * observations;
            convert(w.packed_jacobian, w.wide_jacobian, values * 8);
            convert(w.transposed, w.wide_transposed, values * 8);
            convert(w.compact_residual, w.wide_residual, values);
            matmul(
                w.wide_residual,
                w.wide_transposed,
                w.wide_gradient,
                1,
                8,
                observations,
                count);
            matmul(
                w.wide_jacobian,
                w.wide_transposed,
                w.wide_hessian,
                8,
                8,
                observations,
                count);
            convert(w.wide_gradient, w.gradient, std::size_t(count) * 8);
            convert(w.wide_hessian, w.hessian, std::size_t(count) * 64);
        } else {
            // Reversed binary einsum operands: r[batch,1,m] @ JT[batch,m,8].
            matmul(
                w.compact_residual,
                w.transposed,
                w.gradient,
                1,
                8,
                observations,
                count);
            if (small_hessian && b.count <= 16 && b.planes == 1
                && b.height == 51 && b.width == 51) {
                if (count == 1) {
                    hessian_single_fp64<<<36, 128, 0, stream>>>(
                        w.packed_jacobian, w.hessian);
                    ++timings.hessian_single_launches;
                } else {
                    hessian_small_fp64<<<dim3(10, count), 128, 0, stream>>>(
                        w.packed_jacobian, w.hessian);
                    ++timings.hessian_batch_launches;
                }
                check(cudaGetLastError());
            } else {
                matmul(
                    w.packed_jacobian,
                    w.transposed,
                    w.hessian,
                    8,
                    8,
                    observations,
                    count);
            }
        }
    }

    template <typename T>
    void one_iteration(
        const Batch& b, const Settings& s, Work<T> w, Timings& timings) {
        ++timings.iterations;
        int observations = b.height * b.width * b.planes;
        check(cudaMemsetAsync(
            w.accepted, 0, std::size_t(b.count) * sizeof(int), stream));
        int count = select_rows(b, s, w, timings, false, true);
        if (!count) {
            return;
        }

        bool packed = false;
        if constexpr (sizeof(T) == sizeof(double)) {
            if (fused_jacobian) {
                int selected = count;
                check(
                    detail::launch_jacobian_pack_fp64(
                        static_cast<const double*>(b.x),
                        static_cast<const double*>(b.weights),
                        w.indices,
                        static_cast<const double*>(b.residuals),
                        w.packed_jacobian,
                        w.transposed,
                        w.compact_residual,
                        reinterpret_cast<bool*>(w.factor_info),
                        b.count,
                        selected,
                        b.planes,
                        b.height,
                        b.width,
                        device,
                        stream));
                count = select_rows(b, s, w, timings, false, false, selected);
                if (!count) {
                    return;
                }
                packed = count == selected;
            } else if (tiled_jacobian && b.count <= 16 && b.planes == 1
                && b.height == 51 && b.width == 51) {
                int selected = count;
                tiled_jacobian_fp64::launch(
                    b, w, selected, jacobian_sms, cube_exponent, stream);
                ++timings.tiled_jacobian_launches;
                count = select_rows(b, s, w, timings, false, false, selected);
                if (!count) {
                    return;
                }
                packed = count == selected;
            } else if (b.planes == 1 && b.height == 51 && b.width == 51) {
                // Preserve the portable arithmetic while writing the two
                // cuBLAS layouts directly for the primary stamp shape.
                int selected = count;
                evaluate_jacobian<T, true>
                    <<<count, threads, 0, stream>>>(b, w, T(cube_exponent));
                count = select_rows(b, s, w, timings);
                if (!count) {
                    return;
                }
                packed = count == selected;
            }
        }
        if (!packed) {
            // Mixed-invalid fused batches changed the selected-row order.
            // Rebuild the survivor layouts with the original native path.
            evaluate_jacobian<<<count, threads, 0, stream>>>(
                b, w, T(cube_exponent));
            count = select_rows(b, s, w, timings);
            if (!count) {
                return;
            }
            pack_equations<<<count, threads, 0, stream>>>(b, w);
            check(cudaGetLastError());
        }
        normal_equations(b, w, count, observations, timings);
        norms(w.compact_residual, w.residual_norm, count, observations);
        prepare_system<<<(count + threads - 1) / threads, threads, 0, stream>>>(
            b, s, w, count);

        count = select_rows(b, s, w, timings);
        if (!count) {
            return;
        }
        pack_systems<<<(count + threads - 1) / threads, threads, 0, stream>>>(
            w, count);
        check(cudaGetLastError());
        library_solve(w, count);
        scatter_step<<<(count + threads - 1) / threads, threads, 0, stream>>>(
            b, w, count);

        count = select_rows(b, s, w, timings);
        if (!count) {
            return;
        }
        bool trial_packed = false;
        if constexpr (sizeof(T) == sizeof(double)) {
            if (fused_trial_residual && b.count <= 16 && b.planes == 1
                && b.height == 51 && b.width == 51) {
                fused_trial_residual_fp64::launch(b, w, count, stream);
                ++timings.fused_trial_residual_launches;
                trial_packed = true;
            }
        }
        if (!trial_packed) {
            residual<T><<<b.count, threads, 0, stream>>>(
                b, w.trial_x, w.trial_residual, T(cube_exponent));
            pack_residuals<<<count, threads, 0, stream>>>(b, w);
        }
        check(cudaGetLastError());
        matmul(
            w.compact_residual,
            w.compact_residual,
            w.cost,
            1,
            1,
            observations,
            count);
        matmul(
            w.transposed,
            w.transposed,
            w.trial_cost,
            1,
            1,
            observations,
            count);
        classify_trial<<<count, threads, 0, stream>>>(b, s, w);

        count = select_rows(b, s, w, timings, true);
        if (count) {
            pack_accepted_steps<<<
                (count + threads - 1) / threads,
                threads,
                0,
                stream>>>(w, count);
            check(cudaGetLastError());
            norms(w.step_packed, w.step_norm, count, parameters);
            finish_convergence<<<
                (count + threads - 1) / threads,
                threads,
                0,
                stream>>>(b, s, w, count);
        }
        select_rows(b, s, w, timings);
    }

    template <typename T>
    void execute(const Batch& b, const Settings& s, Timings& timings) {
        Work<T> w = initialize_iteration_state<T>(b, s, timings);
        while (*host_active) {
            one_iteration(b, s, w, timings);
        }
    }
};

Workspace::Workspace(int device, KernelPolicy policy)
    : impl_(nullptr) {
    // Validate selection before creating an owner whose destructor uses it.
    DeviceGuard guard(device);
    auto state = std::make_unique<Impl>();
    state->device = device;
    state->kernel_policy = policy;
    int major = 0, minor = 0, clusters = 0;
    check(cudaDeviceGetAttribute(
        &major, cudaDevAttrComputeCapabilityMajor, device));
    check(cudaDeviceGetAttribute(
        &minor, cudaDevAttrComputeCapabilityMinor, device));
    if (major == 10 && minor == 0) {
        check(cudaDeviceGetAttribute(
            &clusters, cudaDevAttrClusterLaunch, device));
    }
    state->fused_jacobian = policy == KernelPolicy::Automatic && major == 10
        && minor == 0 && clusters != 0;
    check(cudaStreamCreateWithFlags(&state->stream, cudaStreamNonBlocking));
    check(cublasCreate(&state->handle));
    check(cublasGetVersion(state->handle, &state->cublas_version));
    const auto selected =
        select_kernels(policy, major, minor, state->cublas_version);
    state->tiled_jacobian = selected.tiled_jacobian;
    state->fused_trial_residual = selected.fused_trial_residual;
    state->small_hessian = selected.small_hessian;
    if (state->tiled_jacobian) {
        // Test policies still use the executing GPU's actual geometry.
        check(cudaDeviceGetAttribute(
            &state->jacobian_sms, cudaDevAttrMultiProcessorCount, device));
    }
    check(cublasSetStream(state->handle, state->stream));
    check(cudaEventCreateWithFlags(&state->dependency, cudaEventDisableTiming));
    check(cudaEventCreate(&state->start));
    check(cudaEventCreate(&state->stop));
    check(cudaMalloc(&state->device_active, sizeof(int)));
    check(cudaMallocHost(&state->host_active, sizeof(int)));
    impl_ = state.release();
}

Workspace::~Workspace() {
    delete impl_;
}

Timings Workspace::run(
    const Batch& b, const Settings& s, std::uintptr_t producer_stream) {
    auto begin = Clock::now();
    DeviceGuard guard(impl_->device);
    if (b.count <= 0 || b.height <= 0 || b.width <= 0
        || (b.planes != 1 && b.planes != 3)) {
        throw std::invalid_argument("invalid xFit native batch dimensions");
    }
    if (b.count > std::numeric_limits<int>::max() / 64) {
        throw std::overflow_error(
            "xFit native batch count exceeds index capacity");
    }
    if (std::size_t(b.height) * b.width * b.planes
        > std::size_t(std::numeric_limits<int>::max())) {
        throw std::overflow_error(
            "xFit native observation count exceeds int32");
    }
    Timings timings{};
    timings.kernel_policy = impl_->kernel_policy;
    timings.cublas_version = impl_->cublas_version;
    try {
        check(cudaEventRecord(
            impl_->dependency,
            reinterpret_cast<cudaStream_t>(producer_stream)));
        check(cudaStreamWaitEvent(impl_->stream, impl_->dependency, 0));
        check(cudaEventRecord(impl_->start, impl_->stream));
        if (b.fp64) {
            impl_->execute<double>(b, s, timings);
        } else {
            impl_->execute<float>(b, s, timings);
        }
        check(cudaEventRecord(impl_->stop, impl_->stream));
        impl_->synchronize(timings);
        check(cudaEventElapsedTime(
            &timings.gpu_milliseconds, impl_->start, impl_->stop));
    } catch (...) {
        // Preserve the original error, but finish all work that may refer to
        // the caller's arrays before the binding releases their references.
        cudaStreamSynchronize(impl_->stream);
        throw;
    }
    timings.host_seconds =
        std::chrono::duration<double>(Clock::now() - begin).count()
        - timings.synchronize_seconds;
    return timings;
}

}  // namespace cuphoton::xfit
