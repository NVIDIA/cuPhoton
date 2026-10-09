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
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

extern "C" int xfit_library_norms(
    const void*, void*, int, int, bool, std::uintptr_t);

namespace cuphoton::xfit {
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
    T *gradient, *hessian, *system, *matrix_packed, *step, *step_packed;
    T *trial_x, *trial_residual, *damping, *cost, *trial_cost;
    T *residual_norm, *step_norm;
    int *rejected, *indices, *accepted, *pivots, *factor_info;
    T **matrix_ptrs, **rhs_ptrs;
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

template <typename T>
__global__ void evaluate_jacobian(Batch b, Work<T> w, T cube_exponent) {
    int row = w.indices[blockIdx.x];
    int observations = b.height * b.width * b.planes;
    std::size_t base = std::size_t(row) * observations;
    Gaussian<T> model(
        static_cast<const T*>(b.x) + row * parameters, cube_exponent);
    const T* weights = static_cast<const T*>(b.weights);
    int invalid = 0;
    for (std::size_t i = threadIdx.x; i < std::size_t(observations);
        i += blockDim.x) {
        T derivative[parameters];
        model.observation(i, b, derivative);
        for (int p = 0; p < parameters; ++p) {
            T value = rn_mul(derivative[p], weights[base + i]);
            invalid |= !isfinite(value);
            w.jacobian[(std::size_t(row) * parameters + p) * observations + i] =
                value;
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

template <typename T>
__global__ void prepare_system(Batch b, Settings s, Work<T> w, int count) {
    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= count) {
        return;
    }
    int row = w.indices[slot];
    constexpr T eps = T(sizeof(T) == 8 ? 0x1p-52 : 0x1p-23);
    T diagonal[parameters];
    T scaled_gradient = 0;
    for (int p = 0; p < parameters; ++p) {
        diagonal[p] = maximum(w.hessian[slot * 64 + p * parameters + p], eps);
        T gradient = w.gradient[slot * parameters + p];
        w.step[row * parameters + p] = -gradient;
        scaled_gradient =
            maximum(scaled_gradient, rn_div(abs(gradient), sqrt(diagonal[p])));
    }
    if (scaled_gradient
        <= rn_mul(T(s.g_tol), maximum(T(1), w.residual_norm[slot]))) {
        b.status[row] = g_converged;
        return;
    }
    // CuPy einsum restores its Hessian with a final transpose. The raw GEMM
    // output is therefore already column-major for the following LU copy.
    for (int column = 0; column < parameters; ++column) {
        for (int line = 0; line < parameters; ++line) {
            T damping = rn_mul(
                rn_mul(w.damping[row], T(line == column)), diagonal[column]);
            w.system[row * 64 + column * parameters + line] = rn_add(
                w.hessian[slot * 64 + column * parameters + line], damping);
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
        w.step_packed[slot * parameters + p] = w.step[row * parameters + p];
    }
    w.matrix_ptrs[slot] = w.matrix_packed + slot * 64;
    w.rhs_ptrs[slot] = w.step_packed + slot * parameters;
}

template <typename T>
__global__ void scatter_step(Batch b, Work<T> w, int count) {
    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= count) {
        return;
    }
    int row = w.indices[slot];
    bool valid = true;
    for (int p = 0; p < parameters; ++p) {
        valid &= isfinite(w.step_packed[slot * parameters + p]);
    }
    if (!valid) {
        b.status[row] = singular;
        return;
    }
    const T* x = static_cast<const T*>(b.x);
    for (int p = 0; p < parameters; ++p) {
        T value = w.step_packed[slot * parameters + p];
        w.step[row * parameters + p] = value;
        w.trial_x[row * parameters + p] =
            rn_add(x[row * parameters + p], value);
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
            if (reduction <= rn_mul(T(s.f_tol), maximum(T(1), cost))) {
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
        w.step_packed[slot * parameters + p] = w.step[row * parameters + p];
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
        && w.step_norm[slot] <= T(rn_mul(s.x_tol, rn_add(s.x_tol, 1.0)))) {
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
    int device = 0;
    bool fused_jacobian = false;
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
        std::size_t row_bytes = (229 + 26 * observations) * sizeof(T)
            + 12 * sizeof(int) + 2 * sizeof(T*);
        // CuPy allocates separate aligned arrays. Preserve those base alignments
        // without changing per-row strides or the values sent to cuBLAS.
        constexpr std::size_t array_alignment = 256;
        constexpr std::size_t pointer_alignment = alignof(void*);
        constexpr std::size_t padding =
            17 * (array_alignment - 1) + pointer_alignment;
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
        w.system = take(count * 64);
        w.matrix_packed = take(count * 64);
        w.step = take(count * 8);
        w.step_packed = take(count * 8);
        w.trial_x = take(count * 8);
        w.trial_residual = take(count * observations);
        w.damping = take(count);
        w.cost = take(count);
        w.trial_cost = take(count);
        w.residual_norm = take(count);
        w.step_norm = take(count);
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
        w.matrix_ptrs = reinterpret_cast<T**>(pointer_address);
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
        if constexpr (sizeof(T) == 8) {
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
        } else {
            check(cublasSgetrfBatched(
                handle, 8, w.matrix_ptrs, 8, w.pivots, w.factor_info, count));
            check(cublasSgetrsBatched(
                handle,
                CUBLAS_OP_N,
                8,
                1,
                const_cast<const float* const*>(w.matrix_ptrs),
                8,
                w.pivots,
                w.rhs_ptrs,
                8,
                &info,
                count));
        }
        // CuPy's default linalg error mode ignores per-matrix factor_info.
        // Nonfinite solutions are rejected by scatter_step, just as in Python.
        if (info != 0) {
            throw std::runtime_error(
                "xFit getrsBatched info " + std::to_string(info));
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
        // Reversed binary einsum operands: r[batch,1,m] @ JT[batch,m,8].
        matmul(
            w.compact_residual,
            w.transposed,
            w.gradient,
            1,
            8,
            observations,
            count);
        matmul(
            w.packed_jacobian,
            w.transposed,
            w.hessian,
            8,
            8,
            observations,
            count);
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
        residual<T><<<b.count, threads, 0, stream>>>(
            b, w.trial_x, w.trial_residual, T(cube_exponent));
        pack_residuals<<<count, threads, 0, stream>>>(b, w);
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

Workspace::Workspace(int device)
    : impl_(nullptr) {
    // Validate selection before creating an owner whose destructor uses it.
    DeviceGuard guard(device);
    auto state = std::make_unique<Impl>();
    state->device = device;
    int major = 0, minor = 0, clusters = 0;
    check(cudaDeviceGetAttribute(
        &major, cudaDevAttrComputeCapabilityMajor, device));
    check(cudaDeviceGetAttribute(
        &minor, cudaDevAttrComputeCapabilityMinor, device));
    if (major == 10 && minor == 0) {
        check(cudaDeviceGetAttribute(
            &clusters, cudaDevAttrClusterLaunch, device));
    }
    state->fused_jacobian = major == 10 && minor == 0 && clusters != 0;
    check(cudaStreamCreateWithFlags(&state->stream, cudaStreamNonBlocking));
    check(cublasCreate(&state->handle));
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
