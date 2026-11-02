// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

// Included by native.cu so preparation and final diagnostics share the LM
// Gaussian's explicit arithmetic boundaries and transcendental implementation.
namespace {

void check(cusolverStatus_t error) {
    if (error != CUSOLVER_STATUS_SUCCESS) {
        throw std::runtime_error(
            "xFit cuSOLVER status " + std::to_string(int(error)));
    }
}

struct PipelineBuffer {
    void* pointer = nullptr;
    std::size_t capacity = 0;
    ~PipelineBuffer() {
        cudaFree(pointer);
    }
    void reserve(std::size_t bytes) {
        if (bytes <= capacity) {
            return;
        }
        void* next = nullptr;
        check(cudaMalloc(&next, bytes));
        cudaFree(pointer);
        pointer = next;
        capacity = bytes;
    }
    template <typename T>
    T* get(std::size_t count) {
        reserve(count * sizeof(T));
        return static_cast<T*>(pointer);
    }
};

template <typename T>
__device__ T read_input(Input input, int row, int plane, int y, int x) {
    const int coordinate[4] = {row, plane, y, x};
    std::int64_t offset = 0;
    for (int axis = 0; axis < 4; ++axis) {
        offset += (input.shape[axis] == 1 ? 0 : coordinate[axis])
            * input.strides[axis];
    }
    const char* address = static_cast<const char*>(input.data) + offset;
    if (input.kind == 'f' || input.kind == 'c') {
        int size = input.kind == 'c' ? input.itemsize / 2 : input.itemsize;
        if (size == 2) {
            return T(__half2float(*reinterpret_cast<const __half*>(address)));
        }
        if (size == 4) {
            return workspace_cast<T>(*reinterpret_cast<const float*>(address));
        }
        return workspace_cast<T>(*reinterpret_cast<const double*>(address));
    }
    if (input.kind == 'u' || input.kind == 'b') {
        if (input.itemsize == 1) {
            return T(*reinterpret_cast<const std::uint8_t*>(address));
        }
        if (input.itemsize == 2) {
            return T(*reinterpret_cast<const std::uint16_t*>(address));
        }
        if (input.itemsize == 4) {
            return T(*reinterpret_cast<const std::uint32_t*>(address));
        }
        return T(*reinterpret_cast<const std::uint64_t*>(address));
    }
    if (input.itemsize == 1) {
        return T(*reinterpret_cast<const std::int8_t*>(address));
    }
    if (input.itemsize == 2) {
        return T(*reinterpret_cast<const std::int16_t*>(address));
    }
    if (input.itemsize == 4) {
        return T(*reinterpret_cast<const std::int32_t*>(address));
    }
    return T(*reinterpret_cast<const std::int64_t*>(address));
}

template <typename T>
__global__ void prepare_pixels(GaussianFit fit, int* error) {
    int m = fit.height * fit.width * fit.planes;
    std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= std::size_t(fit.count) * m) {
        return;
    }
    int row = i / m, pixel = i % m;
    int plane = pixel / (fit.height * fit.width);
    int y = (pixel / fit.width) % fit.height, x = pixel % fit.width;
    // Validate mask in its own precision before narrowing image arithmetic.
    double mask =
        fit.mask.data ? read_input<double>(fit.mask, row, plane, y, x) : 1;
    bool included = mask != 0;
    if (!isfinite(mask)) {
        atomicMin(error, 1);
    }
    T value = read_input<T>(fit.images, row, plane, y, x);
    if (included && !isfinite(value)) {
        atomicMin(error, 2);
    }
    if (!isfinite(value)) {
        value = 0;
    }
    T weight = included ? T(1) : T(0);
    if (fit.variance.data) {
        T variance = read_input<T>(fit.variance, row, plane, y, x);
        if (included && (!isfinite(variance) || variance <= 0)) {
            atomicMin(error, 3);
        }
        T safe = included ? variance : T(1);
        weight = rn_div(weight, sqrt(safe));
    }
    if (fit.planes == 3) {
        weight = rn_mul(weight, T(fit.split_weights[plane]));
    }
    static_cast<T*>(fit.outputs[1])[i] = value;
    static_cast<T*>(fit.outputs[2])[i] = weight;
    static_cast<bool*>(fit.outputs[7])[i] = included;
}

template <typename T>
__global__ void prepare_parameters(GaussianFit fit, int* error) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= fit.count) {
        return;
    }
    T values[8];
    if (fit.initial.data) {
        for (int p = 0; p < 8; ++p) {
            values[p] = read_input<T>(fit.initial, row, 0, 0, p);
        }
    } else {
        int pixels = fit.height * fit.width, m = pixels * fit.planes;
        const T* images =
            static_cast<const T*>(fit.outputs[1]) + std::size_t(row) * m;
        const T* weights =
            static_cast<const T*>(fit.outputs[2]) + std::size_t(row) * m;
        T positive_score = -INFINITY,
          negative_score = fit.planes == 1 ? INFINITY : -INFINITY;
        T positive_peak = -INFINITY,
          negative_peak = fit.planes == 1 ? INFINITY : -INFINITY;
        int positive_index = 0, negative_index = 0;
        for (int pixel = 0; pixel < pixels; ++pixel) {
            int pos = pixel + (fit.planes == 3 ? pixels : 0);
            int neg = pixel + (fit.planes == 3 ? 2 * pixels : 0);
            if (weights[pos] != 0) {
                T score = rn_mul(images[pos], weights[pos]);
                if (!isnan(positive_score)
                    && (isnan(score) || score > positive_score)) {
                    positive_score = score;
                    positive_index = pixel;
                }
                if (images[pos] > positive_peak) {
                    positive_peak = images[pos];
                }
            }
            if (weights[neg] != 0) {
                T score = rn_mul(images[neg], weights[neg]);
                if (!isnan(negative_score)
                    && (isnan(score)
                        || (fit.planes == 1 ? score < negative_score
                                            : score > negative_score))) {
                    negative_score = score;
                    negative_index = pixel;
                }
                if (fit.planes == 1 ? images[neg] < negative_peak
                                    : images[neg] > negative_peak) {
                    negative_peak = images[neg];
                }
            }
        }
        values[0] = maximum(
            positive_peak, fit.planes == 1 ? -negative_peak : negative_peak);
        if (!isfinite(values[0])) {
            values[0] = 0;
        }
        values[1] = values[2] = T(
            fmax(1.0, (fit.height < fit.width ? fit.height : fit.width) / 8.0));
        values[3] = 0;
        values[4] = T(positive_index % fit.width - (fit.width - 1) / 2.0);
        values[5] = T(positive_index / fit.width - (fit.height - 1) / 2.0);
        values[6] = T(negative_index % fit.width - (fit.width - 1) / 2.0);
        values[7] = T(negative_index / fit.width - (fit.height - 1) / 2.0);
    }
    for (int p = 0; p < 8; ++p) {
        if (!isfinite(values[p])) {
            atomicMin(error, 4);
        }
        if ((p == 1 || p == 2) && values[p] <= 0) {
            atomicMin(error, 5);
        }
        static_cast<T*>(fit.outputs[0])[row * 8 + p] =
            (p == 1 || p == 2) ? log(values[p]) : values[p];
    }
}

template <typename T>
__global__ void final_jacobians(
    GaussianFit fit, T* jacobian, T* scale, bool* eligible, T cube_exponent) {
    int row = blockIdx.x, m = fit.height * fit.width * fit.planes;
    auto status = static_cast<const std::int8_t*>(fit.outputs[4])[row];
    bool current = static_cast<const bool*>(fit.outputs[6])[row];
    bool selected = status != invalid_residual
        && (current || (status >= f_converged && status <= g_converged));
    Batch batch{};
    batch.height = fit.height;
    batch.width = fit.width;
    batch.planes = fit.planes;
    Gaussian<T> model(
        static_cast<const T*>(fit.outputs[0]) + row * 8, cube_exponent);
    const T* weights =
        static_cast<const T*>(fit.outputs[2]) + std::size_t(row) * m;
    bool finite = selected;
    for (int pixel = threadIdx.x; pixel < m; pixel += blockDim.x) {
        T derivatives[8];
        model.observation(pixel, batch, derivatives);
        for (int p = 0; p < 8; ++p) {
            T value = rn_mul(derivatives[p], weights[pixel]);
            finite = finite && isfinite(value);
            jacobian[(std::size_t(row) * 8 + p) * m + pixel] = value;
        }
    }
    __shared__ int valid;
    if (threadIdx.x == 0) {
        valid = 1;
    }
    __syncthreads();
    if (!finite) {
        atomicAnd(&valid, 0);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        eligible[row] = valid;
    }
    if (threadIdx.x < 8) {
        T largest = 0;
        for (int pixel = 0; pixel < m; ++pixel) {
            largest = maximum(
                largest,
                fabs(
                    jacobian
                        [(std::size_t(row) * 8 + threadIdx.x) * m + pixel]));
        }
        scale[row * 8 + threadIdx.x] = largest > 0 ? largest : T(1);
    }
}

template <typename T>
__global__ void pack_gradient_operands(
    const T* jacobian,
    const T* residual,
    const int* indices,
    T* transposed,
    T* compact_residual,
    int count,
    int m) {
    std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= std::size_t(count) * 8 * m) {
        return;
    }
    int slot = i / (8 * m), pixel = (i / 8) % m, p = i % 8;
    int row = indices[slot];
    transposed[i] = jacobian[(std::size_t(row) * 8 + p) * m + pixel];
    if (p == 0) {
        compact_residual[std::size_t(slot) * m + pixel] =
            residual[std::size_t(row) * m + pixel];
    }
}

template <typename T>
__global__ void publish_gradients(
    const T* compact, const int* indices, T* gradient, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count * 8) {
        gradient[indices[i / 8] * 8 + i % 8] = compact[i];
    }
}

template <typename T>
__global__ void normalize_jacobians(
    const T* jacobian,
    const T* scale,
    const int* indices,
    T* normalized,
    int count,
    int m,
    bool transpose) {
    std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= std::size_t(count) * 8 * m) {
        return;
    }
    int slot = i / (8 * m), p = (i / m) % 8, pixel = i % m;
    int row = indices[slot];
    T value = rn_div(
        jacobian[(std::size_t(row) * 8 + p) * m + pixel], scale[row * 8 + p]);
    normalized
        [std::size_t(slot) * 8 * m
            + (transpose ? pixel * 8 + p : p * m + pixel)] = value;
}

template <typename T>
__global__ void extract_reduced(const T* qr, T* reduced, int count, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count * 64) {
        return;
    }
    int slot = i / 64, row = i % 8, column = (i / 8) % 8;
    // R.T, column-major, has the original normalized J's left vectors.
    reduced[i] =
        column <= row ? qr[std::size_t(slot) * 8 * m + row * m + column] : T(0);
}

template <typename T>
__global__ void inverse_factors(
    const T* vectors,
    const T* singular_values,
    const T* scale,
    const int* indices,
    const int* info,
    T* factors,
    int* rank,
    int count,
    int m,
    bool transposed_vectors) {
    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= count) {
        return;
    }
    int row = indices[slot], n = min(8, m), rank_value = 0;
    T largest = 0;
    for (int p = 0; p < n; ++p) {
        largest = maximum(largest, singular_values[slot * n + p]);
    }
    constexpr T epsilon = sizeof(T) == 8 ? T(0x1p-52) : T(0x1p-23);
    T threshold = rn_mul(T(8 > m ? 8 : m) * epsilon, largest);
    if (info[slot] == 0) {
        for (int p = 0; p < n; ++p) {
            rank_value += singular_values[slot * n + p] > threshold;
        }
    }
    rank[row] = info[slot] == 0 ? rank_value : -1;
    for (int p = 0; p < 8; ++p) {
        for (int q = 0; q < 8; ++q) {
            T value = T(NAN);
            if (rank_value == 8 && info[slot] == 0) {
                T u = vectors
                    [slot * 64 + (transposed_vectors ? p * 8 + q : q * 8 + p)];
                value = rn_div(
                    rn_div(u, singular_values[slot * 8 + q]),
                    scale[row * 8 + p]);
            }
            factors[slot * 64 + p * 8 + q] = value;
        }
    }
}

template <typename T>
__global__ void publish_factors(
    GaussianFit fit,
    const T* factors,
    const T* covariance,
    const int* indices,
    int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count * 64) {
        return;
    }
    int row = indices[i / 64];
    static_cast<T*>(fit.outputs[21])[row * 64 + i % 64] = covariance[i];
    static_cast<T*>(fit.outputs[22])[row * 64 + i % 64] = factors[i];
}

template <typename T>
__global__ void initialize_diagnostics(
    GaussianFit fit, int* rank, bool initialize_normal = true) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= fit.count) {
        return;
    }
    rank[row] = -1;
    if (initialize_normal) {
        for (int p = 0; p < 8; ++p) {
            static_cast<T*>(fit.outputs[26])[row * 8 + p] = T(NAN);
        }
    }
    for (int i = 0; i < 64; ++i) {
        static_cast<T*>(fit.outputs[21])[row * 64 + i] = T(NAN);
        static_cast<T*>(fit.outputs[22])[row * 64 + i] = T(NAN);
        if (initialize_normal) {
            static_cast<T*>(fit.outputs[27])[row * 64 + i] = T(NAN);
        }
    }
}

template <typename T>
__global__ void mask_normal_diagnostics(GaussianFit fit, const bool* eligible) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= fit.count || eligible[row]) {
        return;
    }
    for (int p = 0; p < 8; ++p) {
        static_cast<T*>(fit.outputs[26])[row * 8 + p] = T(NAN);
    }
    for (int i = 0; i < 64; ++i) {
        static_cast<T*>(fit.outputs[27])[row * 64 + i] = T(NAN);
    }
}

template <typename T>
__global__ void physical_parameters(GaussianFit fit) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= fit.count) {
        return;
    }
    const T* x = static_cast<const T*>(fit.outputs[0]) + row * 8;
    T* out = static_cast<T*>(fit.outputs[8]) + row * 8;
    for (int p = 0; p < 8; ++p) {
        out[p] = p == 1 || p == 2 ? exp(x[p]) : x[p];
    }
    T half = T(1.57079632679489661923132169163975144),
      period = rn_mul(T(2), half);
    T shifted = rn_add(out[3], half);
    T angle = fmod(shifted, period);
    if (angle < 0) {
        angle = rn_add(angle, period);
    }
    out[3] = rn_sub(angle, half);
}

template <typename T>
__global__ void public_residuals(GaussianFit fit, T* squared, T cube_exponent) {
    int row = blockIdx.x, m = fit.height * fit.width * fit.planes;
    T parameters[8];
    for (int p = 0; p < 8; ++p) {
        parameters[p] = static_cast<const T*>(fit.outputs[0])[row * 8 + p];
    }
    parameters[3] = static_cast<const T*>(fit.outputs[8])[row * 8 + 3];
    Gaussian<T> model(parameters, cube_exponent);
    Batch batch{};
    batch.height = fit.height;
    batch.width = fit.width;
    batch.planes = fit.planes;
    for (int pixel = threadIdx.x; pixel < m; pixel += blockDim.x) {
        std::size_t i = std::size_t(row) * m + pixel;
        T residual = rn_sub(
            model.observation(pixel, batch, nullptr),
            static_cast<const T*>(fit.outputs[1])[i]);
        static_cast<T*>(fit.outputs[9])[i] = residual;
        T weight = static_cast<const T*>(fit.outputs[2])[i];
        T weighted = rn_mul(residual, weight);
        T null_value = rn_mul(static_cast<const T*>(fit.outputs[1])[i], weight);
        squared[i] = rn_mul(weighted, weighted);
        squared[std::size_t(fit.count) * m + i] =
            rn_mul(null_value, null_value);
    }
}

template <typename T>
__global__ void public_statistics(
    GaussianFit fit, const int* ranks, const bool* available) {
    int row = blockIdx.x;
    int m = fit.height * fit.width * fit.planes;
    std::int64_t included = 0, effective = 0;
    for (int pixel = threadIdx.x; pixel < m; pixel += blockDim.x) {
        std::size_t i = std::size_t(row) * m + pixel;
        T weight = static_cast<const T*>(fit.outputs[2])[i];
        included += static_cast<const bool*>(fit.outputs[7])[i];
        effective += weight != 0;
    }
    included = sum(included);
    effective = sum(effective);
    if (threadIdx.x != 0) {
        return;
    }
    T chi = static_cast<const T*>(fit.outputs[13])[row];
    T null_chi = static_cast<const T*>(fit.outputs[16])[row];
    auto status = static_cast<const std::int8_t*>(fit.outputs[4])[row];
    bool converged = status >= f_converged && status <= g_converged;
    std::int64_t dof = effective - 8;
    double reduced = dof > 0 ? double(chi) / double(dof) : NAN;
    static_cast<bool*>(fit.outputs[10])[row] = converged;
    static_cast<std::int64_t*>(fit.outputs[11])[row] =
        static_cast<const std::int32_t*>(fit.outputs[5])[row];
    static_cast<T*>(fit.outputs[12])[row] = sqrt(chi);
    static_cast<T*>(fit.outputs[13])[row] = chi;
    static_cast<std::int64_t*>(fit.outputs[14])[row] = included;
    static_cast<double*>(fit.outputs[15])[row] = double(included) / m;
    static_cast<T*>(fit.outputs[16])[row] = null_chi;
    bool finite = isfinite(chi) && isfinite(null_chi);
    static_cast<T*>(fit.outputs[17])[row] =
        finite ? rn_sub(null_chi, chi) : T(NAN);
    static_cast<double*>(fit.outputs[18])[row] = finite && null_chi > 0
        ? double(rn_sub(T(1), rn_div(chi, null_chi)))
        : NAN;
    static_cast<std::int64_t*>(fit.outputs[19])[row] = dof;
    static_cast<double*>(fit.outputs[20])[row] = reduced;
    T* covariance = static_cast<T*>(fit.outputs[21]) + row * 64;
    T* factor = static_cast<T*>(fit.outputs[22]) + row * 64;
    const T* parameters = static_cast<const T*>(fit.outputs[8]) + row * 8;
    bool base_finite = true, diagonal_valid = true;
    for (int p = 0; p < 8; ++p) {
        for (int q = 0; q < 8; ++q) {
            T pscale = p == 1 || p == 2 ? parameters[p] : T(1);
            T qscale = q == 1 || q == 2 ? parameters[q] : T(1);
            T base = rn_mul(rn_mul(covariance[p * 8 + q], pscale), qscale);
            base_finite = base_finite && isfinite(base);
            covariance[p * 8 + q] = fit.variance.data
                ? base
                : workspace_cast<T>(double(base) * reduced);
            T transformed = rn_mul(factor[p * 8 + q], pscale);
            factor[p * 8 + q] = fit.variance.data
                ? transformed
                : workspace_cast<T>(double(transformed) * sqrt(reduced));
            if (p == q) {
                diagonal_valid = diagonal_valid
                    && isfinite(covariance[p * 8 + q])
                    && covariance[p * 8 + q] >= 0;
            }
        }
    }
    std::int8_t reason = !converged ? 1
        : dof <= 0                  ? 2
        : !available[row]           ? 3
        : ranks[row] < 8            ? 4
        : !base_finite              ? 5
        : !diagonal_valid           ? 6
                                    : 0;
    static_cast<bool*>(fit.outputs[24])[row] = reason == 0;
    static_cast<std::int8_t*>(fit.outputs[25])[row] = reason;
    for (int p = 0; p < 8; ++p) {
        static_cast<T*>(fit.outputs[23])[row * 8 + p] =
            reason == 0 ? sqrt(covariance[p * 8 + p]) : T(NAN);
    }
    if (reason != 0) {
        for (int i = 0; i < 64; ++i) {
            covariance[i] = factor[i] = T(NAN);
        }
    }
}

}  // namespace

struct GaussianPipeline::Impl {
    int device;
    cusolverDnHandle_t solver = nullptr;
    cublasHandle_t blas = nullptr;
    gesvdjInfo_t svd_info = nullptr;
    PipelineBuffer errors, jacobian, scales, eligible, indices, normalized;
    PipelineBuffer reduced, vectors, right_vectors, singular_values, tau;
    PipelineBuffer workspace, info, factors, covariance;
    PipelineBuffer squared_residuals, reduction_workspace;
    PipelineBuffer gradient_workspace;
    explicit Impl(int id)
        : device(id) {}
    ~Impl() {
        DeviceGuard guard(device);
        if (svd_info) {
            cusolverDnDestroyGesvdjInfo(svd_info);
        }
        if (solver) {
            cusolverDnDestroy(solver);
        }
        if (blas) {
            cublasDestroy(blas);
        }
    }

    template <typename T>
    void diagnostics(const GaussianFit& fit, cudaStream_t stream) {
        int count = fit.count, m = fit.height * fit.width * fit.planes;
        T* j = jacobian.get<T>(std::size_t(count) * 8 * m);
        T* scale = scales.get<T>(count * 8);
        bool* valid = eligible.get<bool>(count);
        int* ranks = static_cast<int*>(fit.outputs[28]);
        initialize_diagnostics<T>
            <<<(count + 127) / 128, 128, 0, stream>>>(fit, ranks);
        final_jacobians<T>
            <<<count, threads, 0, stream>>>(fit, j, scale, valid, T(3));
        check(cudaGetLastError());
        std::vector<unsigned char> valid_host(count);
        check(cudaMemcpyAsync(
            valid_host.data(), valid, count, cudaMemcpyDeviceToHost, stream));
        check(cudaStreamSynchronize(stream));
        std::vector<int> rows;
        for (int row = 0; row < count; ++row) {
            if (valid_host[row]) {
                rows.push_back(row);
            }
        }
        int selected = int(rows.size());
        try {
            int* index = indices.get<int>(selected);
            if (selected) {
                check(cudaMemcpyAsync(
                    index,
                    rows.data(),
                    selected * sizeof(int),
                    cudaMemcpyHostToDevice,
                    stream));
            }
            // Retain the final gradient and Gauss--Newton Hessian diagnostics
            // in solver coordinates, independently of covariance validity.
            T alpha = 1, beta = 0;
            const T* residual = static_cast<const T*>(fit.outputs[3]);
            T* gradient = static_cast<T*>(fit.outputs[26]);
            if (m == 1) {
                multiply_single_observation<<<
                    (count * 8 + threads - 1) / threads,
                    threads,
                    0,
                    stream>>>(j, residual, gradient, 8, 1, count);
            } else if (selected == 1) {
                // A singleton einsum squeezes its batch axis and uses dot,
                // which retains the parameter-major Jacobian's transpose.
                int row = rows[0];
                cudaDataType type = sizeof(T) == 8 ? CUDA_R_64F : CUDA_R_32F;
                check(cublasGemmEx(
                    blas,
                    CUBLAS_OP_T,
                    CUBLAS_OP_N,
                    8,
                    1,
                    m,
                    &alpha,
                    j + std::size_t(row) * 8 * m,
                    type,
                    m,
                    residual + std::size_t(row) * m,
                    type,
                    m,
                    &beta,
                    gradient + row * 8,
                    type,
                    8,
                    type,
                    CUBLAS_GEMM_DEFAULT));
            } else if (selected > 1) {
                // Batched einsum reverses its operands, then matmul makes
                // the transposed Jacobian C-contiguous. Preserve that layout
                // and cuBLAS reduction order for cancelling gradients.
                T* transposed =
                    normalized.get<T>(std::size_t(selected) * 8 * m);
                constexpr std::size_t alignment = 256 / sizeof(T);
                std::size_t residual_values =
                    (std::size_t(selected) * m + alignment - 1) / alignment
                    * alignment;
                T* compact_residual = gradient_workspace.get<T>(
                    residual_values + std::size_t(selected) * 8);
                T* compact_gradient = compact_residual + residual_values;
                pack_gradient_operands<<<
                    (std::size_t(selected) * 8 * m + 255) / 256,
                    256,
                    0,
                    stream>>>(
                    j,
                    residual,
                    index,
                    transposed,
                    compact_residual,
                    selected,
                    m);
                cudaDataType type = sizeof(T) == 8 ? CUDA_R_64F : CUDA_R_32F;
                check(cublasGemmStridedBatchedEx(
                    blas,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    8,
                    1,
                    m,
                    &alpha,
                    transposed,
                    type,
                    8,
                    8LL * m,
                    compact_residual,
                    type,
                    m,
                    m,
                    &beta,
                    compact_gradient,
                    type,
                    8,
                    8,
                    selected,
                    type,
                    CUBLAS_GEMM_DEFAULT));
                publish_gradients<<<
                    (selected * 8 + 255) / 256,
                    256,
                    0,
                    stream>>>(compact_gradient, index, gradient, selected);
            }
            if constexpr (sizeof(T) == 8) {
                check(cublasDgemmStridedBatched(
                    blas,
                    CUBLAS_OP_T,
                    CUBLAS_OP_N,
                    8,
                    8,
                    m,
                    &alpha,
                    j,
                    m,
                    8LL * m,
                    j,
                    m,
                    8LL * m,
                    &beta,
                    static_cast<T*>(fit.outputs[27]),
                    8,
                    64,
                    count));
            } else {
                check(cublasSgemmStridedBatched(
                    blas,
                    CUBLAS_OP_T,
                    CUBLAS_OP_N,
                    8,
                    8,
                    m,
                    &alpha,
                    j,
                    m,
                    8LL * m,
                    j,
                    m,
                    8LL * m,
                    &beta,
                    static_cast<T*>(fit.outputs[27]),
                    8,
                    64,
                    count));
            }
            mask_normal_diagnostics<T>
                <<<(count + 127) / 128, 128, 0, stream>>>(fit, valid);
            if (selected) {
                bool qr = selected > 1 && m > 16;
                bool jacobi = qr || m <= 32;
                T* a = normalized.get<T>(std::size_t(selected) * 8 * m);
                normalize_jacobians<<<
                    (std::size_t(selected) * 8 * m + 255) / 256,
                    256,
                    0,
                    stream>>>(j, scale, index, a, selected, m, !qr && jacobi);
                int* dev_info = info.get<int>(selected);
                T* u = vectors.get<T>(selected * 64);
                T* v = right_vectors.get<T>(
                    std::size_t(selected) * (jacobi && !qr ? m * m : 64));
                T* s = singular_values.get<T>(selected * std::min(8, m));
                int lwork = 0;
                if (qr) {
                    T* taus = tau.get<T>(8 * selected);
                    if constexpr (sizeof(T) == 8) {
                        check(cusolverDnDgeqrf_bufferSize(
                            solver, m, 8, a, m, &lwork));
                    } else {
                        check(cusolverDnSgeqrf_bufferSize(
                            solver, m, 8, a, m, &lwork));
                    }
                    T* work = workspace.get<T>(lwork);
                    for (int row = 0; row < selected; ++row) {
                        if constexpr (sizeof(T) == 8) {
                            check(cusolverDnDgeqrf(
                                solver,
                                m,
                                8,
                                a + std::size_t(row) * 8 * m,
                                m,
                                taus + row * 8,
                                work,
                                lwork,
                                dev_info + row));
                        } else {
                            check(cusolverDnSgeqrf(
                                solver,
                                m,
                                8,
                                a + std::size_t(row) * 8 * m,
                                m,
                                taus + row * 8,
                                work,
                                lwork,
                                dev_info + row));
                        }
                    }
                    T* r = reduced.get<T>(selected * 64);
                    extract_reduced<<<
                        (selected * 64 + 255) / 256,
                        256,
                        0,
                        stream>>>(a, r, selected, m);
                    a = r;
                }
                if (jacobi) {
                    int columns = qr ? 8 : m;
                    if constexpr (sizeof(T) == 8) {
                        check(cusolverDnDgesvdjBatched_bufferSize(
                            solver,
                            CUSOLVER_EIG_MODE_VECTOR,
                            8,
                            columns,
                            a,
                            8,
                            s,
                            u,
                            8,
                            v,
                            columns,
                            &lwork,
                            svd_info,
                            selected));
                    } else {
                        check(cusolverDnSgesvdjBatched_bufferSize(
                            solver,
                            CUSOLVER_EIG_MODE_VECTOR,
                            8,
                            columns,
                            a,
                            8,
                            s,
                            u,
                            8,
                            v,
                            columns,
                            &lwork,
                            svd_info,
                            selected));
                    }
                    T* work = workspace.get<T>(lwork);
                    if constexpr (sizeof(T) == 8) {
                        check(cusolverDnDgesvdjBatched(
                            solver,
                            CUSOLVER_EIG_MODE_VECTOR,
                            8,
                            columns,
                            a,
                            8,
                            s,
                            u,
                            8,
                            v,
                            columns,
                            work,
                            lwork,
                            dev_info,
                            svd_info,
                            selected));
                    } else {
                        check(cusolverDnSgesvdjBatched(
                            solver,
                            CUSOLVER_EIG_MODE_VECTOR,
                            8,
                            columns,
                            a,
                            8,
                            s,
                            u,
                            8,
                            v,
                            columns,
                            work,
                            lwork,
                            dev_info,
                            svd_info,
                            selected));
                    }
                } else {
                    if constexpr (sizeof(T) == 8) {
                        check(
                            cusolverDnDgesvd_bufferSize(solver, m, 8, &lwork));
                    } else {
                        check(
                            cusolverDnSgesvd_bufferSize(solver, m, 8, &lwork));
                    }
                    T* work = workspace.get<T>(lwork);
                    // For the sole wide Jacobian, factor J.T and retain V.T.
                    if constexpr (sizeof(T) == 8) {
                        check(cusolverDnDgesvd(
                            solver,
                            'O',
                            'S',
                            m,
                            8,
                            a,
                            m,
                            s,
                            a,
                            m,
                            u,
                            8,
                            work,
                            lwork,
                            nullptr,
                            dev_info));
                    } else {
                        check(cusolverDnSgesvd(
                            solver,
                            'O',
                            'S',
                            m,
                            8,
                            a,
                            m,
                            s,
                            a,
                            m,
                            u,
                            8,
                            work,
                            lwork,
                            nullptr,
                            dev_info));
                    }
                }
                T* f = factors.get<T>(selected * 64);
                T* cov = covariance.get<T>(selected * 64);
                inverse_factors<<<(selected + 127) / 128, 128, 0, stream>>>(
                    u,
                    s,
                    scale,
                    index,
                    dev_info,
                    f,
                    ranks,
                    selected,
                    m,
                    !jacobi);
                T alpha = 1, beta = 0;
                if constexpr (sizeof(T) == 8) {
                    check(cublasDgemmStridedBatched(
                        blas,
                        CUBLAS_OP_T,
                        CUBLAS_OP_N,
                        8,
                        8,
                        8,
                        &alpha,
                        f,
                        8,
                        64,
                        f,
                        8,
                        64,
                        &beta,
                        cov,
                        8,
                        64,
                        selected));
                } else {
                    check(cublasSgemmStridedBatched(
                        blas,
                        CUBLAS_OP_T,
                        CUBLAS_OP_N,
                        8,
                        8,
                        8,
                        &alpha,
                        f,
                        8,
                        64,
                        f,
                        8,
                        64,
                        &beta,
                        cov,
                        8,
                        64,
                        selected));
                }
                publish_factors<<<
                    (selected * 64 + 255) / 256,
                    256,
                    0,
                    stream>>>(fit, f, cov, index, selected);
            }
        } catch (const std::exception&) {
            // Match the public solver's row-level diagnostic failure isolation.
            // A batched library failure must not discard a different fit's
            // covariance. A failed single row retains its NaN covariance and
            // unavailable rank while public status/quality remain meaningful.
            check(cudaStreamSynchronize(stream));
            if (count > 1) {
                for (int row = 0; row < count; ++row) {
                    GaussianFit single = fit;
                    single.count = 1;
                    for (int field = 0; field < 29; ++field) {
                        std::size_t elements = 1;
                        if (field == 0 || field == 8 || field == 23
                            || field == 26) {
                            elements = 8;
                        }
                        if (field == 1 || field == 2 || field == 3 || field == 7
                            || field == 9) {
                            elements = m;
                        }
                        if (field == 21 || field == 22 || field == 27) {
                            elements = 64;
                        }
                        std::size_t bytes = sizeof(T);
                        if (field == 4 || field == 6 || field == 7
                            || field == 10 || field == 24 || field == 25) {
                            bytes = 1;
                        }
                        if (field == 5 || field == 28) {
                            bytes = 4;
                        }
                        if (field == 11 || field == 14 || field == 15
                            || field == 18 || field == 19 || field == 20) {
                            bytes = 8;
                        }
                        single.outputs[field] =
                            static_cast<char*>(fit.outputs[field])
                            + row * elements * bytes;
                    }
                    diagnostics<T>(single, stream);
                }
                return;
            }
            initialize_diagnostics<T><<<1, 1, 0, stream>>>(fit, ranks, false);
        }
        physical_parameters<T><<<(count + 127) / 128, 128, 0, stream>>>(fit);
        T* squared = squared_residuals.get<T>(std::size_t(count) * m * 2);
        public_residuals<T><<<count, threads, 0, stream>>>(fit, squared, T(3));
        std::size_t reduction_bytes = 0;
        check(
            static_cast<cudaError_t>(xfit_library_sums(
                squared,
                fit.outputs[13],
                count,
                m,
                sizeof(T) == sizeof(double),
                nullptr,
                &reduction_bytes,
                reinterpret_cast<std::uintptr_t>(stream))));
        void* storage = reduction_workspace.get<char>(reduction_bytes);
        for (int component = 0; component < 2; ++component) {
            check(
                static_cast<cudaError_t>(xfit_library_sums(
                    squared + std::size_t(component) * count * m,
                    fit.outputs[component ? 16 : 13],
                    count,
                    m,
                    sizeof(T) == sizeof(double),
                    storage,
                    &reduction_bytes,
                    reinterpret_cast<std::uintptr_t>(stream))));
        }
        public_statistics<T><<<count, threads, 0, stream>>>(fit, ranks, valid);
        check(cudaGetLastError());
    }

    template <typename T>
    Timings execute(
        Workspace& lm,
        const GaussianFit& fit,
        const Settings& settings,
        cudaStream_t stream) {
        int *error = errors.get<int>(1), host_error = 6;
        check(cudaMemcpyAsync(
            error, &host_error, sizeof(int), cudaMemcpyHostToDevice, stream));
        std::size_t size =
            std::size_t(fit.count) * fit.height * fit.width * fit.planes;
        prepare_pixels<T><<<(size + 255) / 256, 256, 0, stream>>>(fit, error);
        prepare_parameters<T>
            <<<(fit.count + 127) / 128, 128, 0, stream>>>(fit, error);
        check(cudaGetLastError());
        check(cudaMemcpyAsync(
            &host_error, error, sizeof(int), cudaMemcpyDeviceToHost, stream));
        check(cudaStreamSynchronize(stream));
        const char* messages[] = {"",
            "mask must contain only finite values",
            "images must be finite on included pixels",
            "variance must be finite and positive on included pixels",
            "initial parameters must contain only finite values",
            "initial sigma_x and sigma_y values must be positive"};
        if (host_error < 6) {
            throw std::invalid_argument(messages[host_error]);
        }
        Batch batch{fit.outputs[0],
            fit.outputs[1],
            fit.outputs[2],
            fit.outputs[3],
            static_cast<std::int8_t*>(fit.outputs[4]),
            static_cast<std::int32_t*>(fit.outputs[5]),
            static_cast<bool*>(fit.outputs[6]),
            fit.count,
            fit.height,
            fit.width,
            fit.planes,
            fit.fp64};
        Timings timings =
            lm.run(batch, settings, reinterpret_cast<std::uintptr_t>(stream));
        diagnostics<T>(fit, stream);
        return timings;
    }
};

GaussianPipeline::GaussianPipeline(int device)
    : impl_(nullptr) {
    DeviceGuard guard(device);
    auto state = std::make_unique<Impl>(device);
    check(cusolverDnCreate(&state->solver));
    check(cublasCreate(&state->blas));
    check(cusolverDnCreateGesvdjInfo(&state->svd_info));
    impl_ = state.release();
}

GaussianPipeline::~GaussianPipeline() {
    DeviceGuard guard(impl_->device);
    delete impl_;
}

Timings GaussianPipeline::run(
    Workspace& lm,
    const GaussianFit& fit,
    const Settings& settings,
    std::uintptr_t producer_stream) {
    DeviceGuard guard(impl_->device);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(producer_stream);
    check(cusolverDnSetStream(impl_->solver, stream));
    check(cublasSetStream(impl_->blas, stream));
    try {
        Timings timings = fit.fp64
            ? impl_->execute<double>(lm, fit, settings, stream)
            : impl_->execute<float>(lm, fit, settings, stream);
        check(cudaStreamSynchronize(stream));
        return timings;
    } catch (...) {
        cudaStreamSynchronize(stream);
        throw;
    }
}
