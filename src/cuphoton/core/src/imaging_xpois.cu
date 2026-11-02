/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "imaging_api.h"

#include <cublas_v2.h>
#include <cufft.h>
#include <cusolverDn.h>
#include <cub/cub.cuh>
#include <thrust/iterator/counting_iterator.h>

#include <algorithm>
#include <cmath>
#include <climits>
#include <memory>

namespace cuphoton::imaging {
namespace {

constexpr int threads = 256;
constexpr int fit_chunk_size = 65536;

int blocks(std::int64_t count) {
    return static_cast<int>((count + threads - 1) / threads);
}

void blas_check(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            "native xPois cuBLAS failure: " + std::to_string(status));
    }
}

void solver_check(cusolverStatus_t status) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        throw std::runtime_error(
            "native xPois cuSOLVER failure: " + std::to_string(status));
    }
}

void fft_check(cufftResult status) {
    if (status != CUFFT_SUCCESS) {
        throw std::runtime_error(
            "native xPois cuFFT failure: " + std::to_string(status));
    }
}

struct Libraries {
    cublasHandle_t blas = nullptr;
    cusolverDnHandle_t solver = nullptr;
    explicit Libraries(cudaStream_t stream) {
        blas_check(cublasCreate(&blas));
        try {
            blas_check(cublasSetStream(blas, stream));
            solver_check(cusolverDnCreate(&solver));
            solver_check(cusolverDnSetStream(solver, stream));
        } catch (...) {
            if (solver) {
                cusolverDnDestroy(solver);
            }
            cublasDestroy(blas);
            throw;
        }
    }
    ~Libraries() {
        if (solver) {
            cusolverDnDestroy(solver);
        }
        if (blas) {
            cublasDestroy(blas);
        }
    }
};

struct FftPlan {
    cufftHandle handle = 0;
    FftPlan(int height, int width, cufftType type, cudaStream_t stream) {
        fft_check(cufftCreate(&handle));
        try {
            std::size_t workspace = 0;
            fft_check(cufftMakePlan2d(handle, height, width, type, &workspace));
            fft_check(cufftSetStream(handle, stream));
        } catch (...) {
            cufftDestroy(handle);
            throw;
        }
    }
    ~FftPlan() {
        cufftDestroy(handle);
    }
};

// Match the FP64 cuBLAS GEMM path used by the CuPy matrix operations, including
// its explicit zero-beta products. Chunk accumulation is a separate operation.
void gemm(
    cublasHandle_t handle,
    cublasOperation_t left_op,
    cublasOperation_t right_op,
    int m,
    int n,
    int k,
    const double* left,
    int left_stride,
    const double* right,
    int right_stride,
    double* output,
    int output_stride) {
    const double one = 1.0;
    const double zero = 0.0;
    blas_check(cublasGemmEx(
        handle,
        left_op,
        right_op,
        m,
        n,
        k,
        &one,
        left,
        CUDA_R_64F,
        left_stride,
        right,
        CUDA_R_64F,
        right_stride,
        &zero,
        output,
        CUDA_R_64F,
        output_stride,
        CUBLAS_COMPUTE_64F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

__global__ void clean_inputs(
    const double* reference,
    const double* target,
    const double* variance,
    const void* mask,
    bool boolean_mask,
    double* reference_clean,
    double* target_clean,
    unsigned char* invalid,
    int* counters,
    int height,
    int width,
    int margin_y,
    int margin_x) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = height * width;
    if (index >= count) {
        return;
    }
    const double r = reference[index];
    const double t = target[index];
    reference_clean[index] = isfinite(r) ? r : 0.0;
    target_clean[index] = isfinite(t) ? t : 0.0;
    invalid[index] = !isfinite(r);
    if (variance && isfinite(variance[index]) && variance[index] <= 0.0) {
        atomicExch(counters, 1);
    }
    double selected = 1.0;
    if (mask) {
        selected = boolean_mask ? static_cast<const bool*>(mask)[index]
                                : static_cast<const double*>(mask)[index];
        if (!isfinite(selected)) {
            atomicExch(counters + 1, 1);
        }
        if (selected != 0.0 && selected != 1.0) {
            atomicExch(counters + 2, 1);
        }
    }
    const int y = index / width;
    const int x = index % width;
    if (selected != 0.0 && y >= margin_y && y < height - margin_y
        && x >= margin_x && x < width - margin_x) {
        atomicAdd(counters + 3, 1);
    }
}

__global__ void dilate_invalid_x(
    const unsigned char* input,
    unsigned char* output,
    int height,
    int width,
    int radius) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= height * width) {
        return;
    }
    const int x = index % width;
    const int first = max(0, x - radius);
    const int last = min(width - 1, x + radius);
    unsigned char invalid = 0;
    for (int column = first; column <= last; ++column) {
        invalid |= input[index - x + column];
    }
    output[index] = invalid;
}

__global__ void fit_masks(
    const unsigned char* invalid_x,
    const double* target,
    const double* variance,
    const void* input_mask,
    bool boolean_mask,
    bool* fit_mask,
    unsigned char* output_invalid,
    int height,
    int width,
    int margin_y,
    int margin_x) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= height * width) {
        return;
    }
    const int y = index / width;
    const int x = index % width;
    bool invalid =
        !isfinite(target[index]) || (variance && !isfinite(variance[index]));
    const int first = max(0, y - margin_y);
    const int last = min(height - 1, y + margin_y);
    for (int row = first; row <= last; ++row) {
        invalid |= invalid_x[row * width + x] != 0;
    }
    const bool interior = y >= margin_y && y < height - margin_y
        && x >= margin_x && x < width - margin_x;
    const bool selected = !input_mask
        || (boolean_mask
                ? static_cast<const bool*>(input_mask)[index]
                : static_cast<const double*>(input_mask)[index] != 0.0);
    fit_mask[index] = interior && !invalid && selected;
    output_invalid[index] = invalid || !interior;
}

__global__ void reverse_basis(
    const double* basis, double* reversed, int count, int pixels) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count * pixels) {
        reversed[index] =
            basis[(index / pixels) * pixels + pixels - 1 - index % pixels];
    }
}

__device__ double coordinate(int index, int length) {
    if (length == 1) {
        return -1.0;
    }
    if (index == length - 1) {
        return 1.0;
    }
    return fma(static_cast<double>(index), 2.0 / (length - 1), -1.0);
}

__global__ void background_basis(
    double* terms, int height, int width, int degree, int count) {
    const std::int64_t index =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t pixels = static_cast<std::int64_t>(height) * width;
    if (index >= pixels * count) {
        return;
    }
    const int term = index / pixels;
    int remaining = term;
    int deg_x = 0;
    while (remaining > degree - deg_x) {
        remaining -= degree - deg_x + 1;
        ++deg_x;
    }
    const int deg_y = remaining;
    const int pixel = index % pixels;
    const double x = coordinate(pixel % width, width);
    const double y = coordinate(pixel / width, height);
    terms[index] =
        pow(x, static_cast<double>(deg_x)) * pow(y, static_cast<double>(deg_y));
}

__global__ void gather_patches(
    const double* reference,
    const int* selected,
    double* patches,
    int rows,
    int width,
    int kernel_height,
    int kernel_width) {
    const int patch_pixels = kernel_height * kernel_width;
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows * patch_pixels) {
        return;
    }
    const int center = selected[index / patch_pixels];
    const int offset = index % patch_pixels;
    patches[index] = reference
        [center + (offset / kernel_width - kernel_height / 2) * width
            + offset % kernel_width - kernel_width / 2];
}

__global__ void weighted_design(
    const double* projected,
    const double* backgrounds,
    const double* target,
    const double* variance,
    const int* selected,
    double* design,
    double* weighted_target,
    int rows,
    int basis_count,
    int background_count,
    int image_pixels) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    const int pixel = selected[row];
    const double weight =
        variance ? __ddiv_rn(1.0, __dsqrt_rn(variance[pixel])) : 1.0;
    const int columns = basis_count + background_count;
    for (int column = 0; column < basis_count; ++column) {
        design[row * columns + column] =
            projected[row * basis_count + column] * weight;
    }
    for (int column = 0; column < background_count; ++column) {
        design[row * columns + basis_count + column] =
            backgrounds
                [static_cast<std::int64_t>(column) * image_pixels + pixel]
            * weight;
    }
    weighted_target[row] = target[pixel] * weight;
}

__global__ void accumulate(
    double* destination, const double* source, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        destination[index] += source[index];
    }
}

__global__ void transpose_square(
    const double* source, double* destination, int size) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size * size) {
        destination[index] = source[(index % size) * size + index / size];
    }
}

__global__ void pad_fft(
    const double* input,
    double* output,
    int height,
    int width,
    int fft_height,
    int fft_width) {
    const std::int64_t index =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<std::int64_t>(fft_height) * fft_width) {
        return;
    }
    const int y = index / fft_width;
    const int x = index % fft_width;
    output[index] = y < height && x < width ? input[y * width + x] : 0.0;
}

__global__ void multiply_spectra(
    cufftDoubleComplex* left,
    const cufftDoubleComplex* right,
    std::int64_t count,
    bool pre_blackwell) {
    const std::int64_t index =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const cufftDoubleComplex a = left[index];
    const cufftDoubleComplex b = right[index];
    // Explicit fused products match the complex multiplication used by CuPy's
    // default NVRTC build even when the surrounding scalar kernels disable FMA.
    // Select using the executing device, including when it loads older PTX.
    const double imaginary =
        pre_blackwell ? fma(a.x, b.y, a.y * b.x) : fma(a.y, b.x, a.x * b.y);
    left[index] = {fma(a.x, b.x, -a.y * b.y), imaginary};
}

__global__ void image_outputs(
    const double* convolution,
    const double* background,
    const double* target,
    const unsigned char* invalid,
    double* matched,
    double* residual,
    int height,
    int width,
    int fft_height,
    int fft_width,
    int margin_y,
    int margin_x) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= height * width) {
        return;
    }
    const int y = index / width;
    const int x = index % width;
    const double convolved =
        convolution
            [static_cast<std::int64_t>(y + margin_y) * fft_width + x + margin_x]
        / (static_cast<double>(fft_height) * fft_width);
    const double model = convolved + background[index];
    matched[index] = invalid[index] ? nan("") : model;
    residual[index] = invalid[index] ? nan("") : target[index] - model;
}

__global__ void residual_squares(
    const double* residual,
    const double* variance,
    const int* selected,
    double* values,
    int count,
    double exponent) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const int pixel = selected[index];
    const double value = residual[pixel];
    // CuPy's **2 uses its floating-point power ufunc with a scalar argument.
    // Keep that libdevice path instead of replacing it with multiplication.
    values[index] = pow(value, exponent) / (variance ? variance[pixel] : 1.0);
}

void require_shape(const Array& array, const std::vector<std::int64_t>& shape) {
    if (array.shape != shape) {
        throw std::invalid_argument("native xPois buffer shapes differ");
    }
}

}  // namespace

PyObject* solve_constant_kernel(PyObject*, PyObject* arguments) {
    PyObject* inputs;
    PyObject* outputs;
    int degree, fft_height, fft_width;
    unsigned long long stream_value;
    if (!PyArg_ParseTuple(
            arguments,
            "OOiiiK",
            &inputs,
            &outputs,
            &degree,
            &fft_height,
            &fft_width,
            &stream_value)) {
        return nullptr;
    }
    try {
        if (!PyTuple_Check(inputs) || PyTuple_Size(inputs) != 5
            || !PyTuple_Check(outputs) || PyTuple_Size(outputs) != 6) {
            throw std::invalid_argument(
                "native xPois expects five inputs and six outputs");
        }
        Array reference(PyTuple_GetItem(inputs, 0));
        Array target(PyTuple_GetItem(inputs, 1));
        Array basis(PyTuple_GetItem(inputs, 4));
        std::unique_ptr<Array> variance;
        std::unique_ptr<Array> input_mask;
        if (PyTuple_GetItem(inputs, 2) != Py_None) {
            variance = std::make_unique<Array>(PyTuple_GetItem(inputs, 2));
        }
        if (PyTuple_GetItem(inputs, 3) != Py_None) {
            input_mask = std::make_unique<Array>(PyTuple_GetItem(inputs, 3));
        }
        Array kernel(PyTuple_GetItem(outputs, 0));
        Array matched(PyTuple_GetItem(outputs, 1));
        Array residual(PyTuple_GetItem(outputs, 2));
        Array fit_mask(PyTuple_GetItem(outputs, 3));
        Array background(PyTuple_GetItem(outputs, 4));
        Array coefficients(PyTuple_GetItem(outputs, 5));
        reference.require('f', 8, 2);
        target.require('f', 8, 2);
        basis.require('f', 8, 3);
        kernel.require('f', 8, 2, true);
        matched.require('f', 8, 2, true);
        residual.require('f', 8, 2, true);
        fit_mask.require('b', 1, 2, true);
        background.require('f', 8, 2, true);
        coefficients.require('f', 8, 1, true);
        for (const Array* array :
            {&target, &matched, &residual, &fit_mask, &background}) {
            require_shape(*array, reference.shape);
        }
        if (variance) {
            variance->require('f', 8, 2);
            require_shape(*variance, reference.shape);
        }
        if (input_mask) {
            input_mask->require(
                input_mask->kind == 'b' ? 'b' : 'f',
                input_mask->kind == 'b' ? 1 : 8,
                2);
            require_shape(*input_mask, reference.shape);
        }
        if (reference.size > INT_MAX || basis.size > INT_MAX || degree < 0
            || degree > 31) {
            throw std::invalid_argument(
                "native xPois dimensions exceed the supported range");
        }
        const int height = static_cast<int>(reference.shape[0]);
        const int width = static_cast<int>(reference.shape[1]);
        const int pixels = static_cast<int>(reference.size);
        const int basis_count = static_cast<int>(basis.shape[0]);
        const int kernel_height = static_cast<int>(basis.shape[1]);
        const int kernel_width = static_cast<int>(basis.shape[2]);
        const int kernel_pixels = kernel_height * kernel_width;
        const int background_count = (degree + 1) * (degree + 2) / 2;
        const int columns = basis_count + background_count;
        if (kernel_height == 1 || kernel_width == 1) {
            throw std::invalid_argument(
                "native xPois requires both kernel dimensions greater than one");
        }
        if (basis_count < 1 || kernel_height < 1 || kernel_width < 1
            || !(kernel_height % 2) || !(kernel_width % 2)
            || kernel_height > height || kernel_width > width) {
            throw std::invalid_argument(
                "kernel_shape leaves no valid interior pixels to fit");
        }
        require_shape(kernel, {kernel_height, kernel_width});
        require_shape(coefficients, {columns});
        if (fft_height < height + kernel_height - 1
            || fft_width < width + kernel_width - 1
            || static_cast<std::int64_t>(kernel_pixels) * fit_chunk_size
                > INT_MAX
            || static_cast<std::int64_t>(columns) * fit_chunk_size > INT_MAX) {
            throw std::invalid_argument(
                "native xPois FFT or design dimensions are invalid");
        }
        int row_count = 0;
        double chi2 = 0.0;
        double condition_number = 0.0;
        {
            Detached detached;
            StreamGuard stream(stream_value);
            int device = 0;
            int compute_major = 0;
            check(cudaGetDevice(&device));
            check(cudaDeviceGetAttribute(
                &compute_major, cudaDevAttrComputeCapabilityMajor, device));
            for (const Array* array : {&reference,
                     &target,
                     &basis,
                     &kernel,
                     &matched,
                     &residual,
                     &fit_mask,
                     &background,
                     &coefficients}) {
                array->wait(stream.stream);
            }
            if (variance) {
                variance->wait(stream.stream);
            }
            if (input_mask) {
                input_mask->wait(stream.stream);
            }
            const double* variance_data =
                variance ? variance->data<double>() : nullptr;
            const void* mask_data = input_mask ? input_mask->pointer : nullptr;
            const bool boolean_mask = input_mask && input_mask->kind == 'b';
            DeviceBuffer<double> reference_clean(pixels), target_clean(pixels);
            DeviceBuffer<unsigned char> invalid(pixels), invalid_x(pixels),
                output_invalid(pixels);
            DeviceBuffer<int> counters(4), selected(pixels), selected_count(1);
            check(cudaMemsetAsync(
                counters.get(), 0, 4 * sizeof(int), stream.stream));
            clean_inputs<<<blocks(pixels), threads, 0, stream.stream>>>(
                reference.data<double>(),
                target.data<double>(),
                variance_data,
                mask_data,
                boolean_mask,
                reference_clean.get(),
                target_clean.get(),
                invalid.get(),
                counters.get(),
                height,
                width,
                kernel_height / 2,
                kernel_width / 2);
            dilate_invalid_x<<<blocks(pixels), threads, 0, stream.stream>>>(
                invalid.get(),
                invalid_x.get(),
                height,
                width,
                kernel_width / 2);
            fit_masks<<<blocks(pixels), threads, 0, stream.stream>>>(
                invalid_x.get(),
                target.data<double>(),
                variance_data,
                mask_data,
                boolean_mask,
                fit_mask.data<bool>(),
                output_invalid.get(),
                height,
                width,
                kernel_height / 2,
                kernel_width / 2);
            std::size_t select_bytes = 0;
            auto indices = thrust::make_counting_iterator<int>(0);
            check(
                cub::DeviceSelect::Flagged(
                    nullptr,
                    select_bytes,
                    indices,
                    fit_mask.data<bool>(),
                    selected.get(),
                    selected_count.get(),
                    pixels,
                    stream.stream));
            DeviceBuffer<unsigned char> select_workspace(select_bytes);
            check(
                cub::DeviceSelect::Flagged(
                    select_workspace.get(),
                    select_bytes,
                    indices,
                    fit_mask.data<bool>(),
                    selected.get(),
                    selected_count.get(),
                    pixels,
                    stream.stream));
            int host_counters[4];
            check(cudaMemcpyAsync(
                host_counters,
                counters.get(),
                sizeof(host_counters),
                cudaMemcpyDeviceToHost,
                stream.stream));
            check(cudaMemcpyAsync(
                &row_count,
                selected_count.get(),
                sizeof(row_count),
                cudaMemcpyDeviceToHost,
                stream.stream));
            stream.finish();
            if (host_counters[0]) {
                throw std::invalid_argument(
                    "variance must be strictly positive");
            }
            if (host_counters[1]) {
                throw std::invalid_argument(
                    "fit_mask array must not contain NaN or inf values");
            }
            if (host_counters[2]) {
                throw std::invalid_argument(
                    "fit_mask array must contain only 0/1 values");
            }
            if (!host_counters[3]) {
                throw std::invalid_argument(
                    "fit_mask must select at least one interior pixel");
            }
            if (!row_count) {
                throw std::invalid_argument(
                    "no finite pixels remain within the fit region");
            }
            if (row_count < columns) {
                throw std::invalid_argument(
                    "fit region is underdetermined for the requested basis: "
                    + std::to_string(row_count) + " equations for "
                    + std::to_string(columns) + " coefficients");
            }
            Libraries libraries(stream.stream);
            DeviceBuffer<double> reversed(basis.size);
            DeviceBuffer<double> backgrounds(
                static_cast<std::size_t>(pixels) * background_count);
            reverse_basis<<<blocks(basis.size), threads, 0, stream.stream>>>(
                basis.data<double>(),
                reversed.get(),
                basis_count,
                kernel_pixels);
            background_basis<<<
                blocks(backgrounds.count),
                threads,
                0,
                stream.stream>>>(
                backgrounds.get(), height, width, degree, background_count);
            const int chunk_rows = std::min(row_count, fit_chunk_size);
            DeviceBuffer<double> patches(
                static_cast<std::size_t>(chunk_rows) * kernel_pixels);
            DeviceBuffer<double> projected(
                static_cast<std::size_t>(chunk_rows) * basis_count);
            DeviceBuffer<double> design(
                static_cast<std::size_t>(chunk_rows) * columns);
            DeviceBuffer<double> weighted_target(chunk_rows);
            DeviceBuffer<double> gram(
                static_cast<std::size_t>(columns) * columns);
            DeviceBuffer<double> partial_gram(gram.count), partial_rhs(columns);
            check(cudaMemsetAsync(
                gram.get(), 0, gram.count * sizeof(double), stream.stream));
            check(cudaMemsetAsync(
                coefficients.pointer,
                0,
                columns * sizeof(double),
                stream.stream));
            for (int start = 0; start < row_count; start += fit_chunk_size) {
                const int rows = std::min(fit_chunk_size, row_count - start);
                gather_patches<<<
                    blocks(static_cast<std::int64_t>(rows) * kernel_pixels),
                    threads,
                    0,
                    stream.stream>>>(
                    reference_clean.get(),
                    selected.get() + start,
                    patches.get(),
                    rows,
                    width,
                    kernel_height,
                    kernel_width);
                gemm(
                    libraries.blas,
                    CUBLAS_OP_T,
                    CUBLAS_OP_N,
                    basis_count,
                    rows,
                    kernel_pixels,
                    reversed.get(),
                    kernel_pixels,
                    patches.get(),
                    kernel_pixels,
                    projected.get(),
                    basis_count);
                weighted_design<<<blocks(rows), threads, 0, stream.stream>>>(
                    projected.get(),
                    backgrounds.get(),
                    target_clean.get(),
                    variance_data,
                    selected.get() + start,
                    design.get(),
                    weighted_target.get(),
                    rows,
                    basis_count,
                    background_count,
                    pixels);
                gemm(
                    libraries.blas,
                    CUBLAS_OP_N,
                    CUBLAS_OP_T,
                    columns,
                    columns,
                    rows,
                    design.get(),
                    columns,
                    design.get(),
                    columns,
                    partial_gram.get(),
                    columns);
                gemm(
                    libraries.blas,
                    CUBLAS_OP_T,
                    CUBLAS_OP_T,
                    1,
                    columns,
                    rows,
                    weighted_target.get(),
                    rows,
                    design.get(),
                    columns,
                    partial_rhs.get(),
                    1);
                accumulate<<<blocks(gram.count), threads, 0, stream.stream>>>(
                    gram.get(), partial_gram.get(), gram.count);
                accumulate<<<blocks(columns), threads, 0, stream.stream>>>(
                    coefficients.data<double>(), partial_rhs.get(), columns);
            }
            DeviceBuffer<double> matrix(gram.count), singular_values(columns);
            DeviceBuffer<int> info(1), pivots(columns);
            check(cudaMemcpyAsync(
                matrix.get(),
                gram.get(),
                gram.count * sizeof(double),
                cudaMemcpyDeviceToDevice,
                stream.stream));
            int svd_workspace_size = 0;
            solver_check(cusolverDnDgesvd_bufferSize(
                libraries.solver, columns, columns, &svd_workspace_size));
            DeviceBuffer<double> svd_workspace(svd_workspace_size);
            solver_check(cusolverDnDgesvd(
                libraries.solver,
                'N',
                'N',
                columns,
                columns,
                matrix.get(),
                columns,
                singular_values.get(),
                nullptr,
                columns,
                nullptr,
                columns,
                svd_workspace.get(),
                svd_workspace_size,
                nullptr,
                info.get()));
            int host_info = 0;
            double extremes[2];
            check(cudaMemcpyAsync(
                &host_info,
                info.get(),
                sizeof(int),
                cudaMemcpyDeviceToHost,
                stream.stream));
            check(cudaMemcpyAsync(
                extremes,
                singular_values.get(),
                sizeof(double),
                cudaMemcpyDeviceToHost,
                stream.stream));
            check(cudaMemcpyAsync(
                extremes + 1,
                singular_values.get() + columns - 1,
                sizeof(double),
                cudaMemcpyDeviceToHost,
                stream.stream));
            stream.finish();
            if (host_info) {
                throw std::invalid_argument(
                    "fit normal equations are singular for the requested basis");
            }
            condition_number = extremes[0] / extremes[1];
            if (!std::isfinite(condition_number) || condition_number > 1e12) {
                throw std::invalid_argument(
                    "fit normal equations are ill-conditioned for the requested basis");
            }
            transpose_square<<<blocks(gram.count), threads, 0, stream.stream>>>(
                gram.get(), matrix.get(), columns);
            int lu_workspace_size = 0;
            solver_check(cusolverDnDgetrf_bufferSize(
                libraries.solver,
                columns,
                columns,
                matrix.get(),
                columns,
                &lu_workspace_size));
            DeviceBuffer<double> lu_workspace(lu_workspace_size);
            solver_check(cusolverDnDgetrf(
                libraries.solver,
                columns,
                columns,
                matrix.get(),
                columns,
                lu_workspace.get(),
                pivots.get(),
                info.get()));
            check(cudaMemcpyAsync(
                &host_info,
                info.get(),
                sizeof(int),
                cudaMemcpyDeviceToHost,
                stream.stream));
            stream.finish();
            if (host_info) {
                throw std::invalid_argument(
                    "fit normal equations could not be solved");
            }
            solver_check(cusolverDnDgetrs(
                libraries.solver,
                CUBLAS_OP_N,
                columns,
                1,
                matrix.get(),
                columns,
                pivots.get(),
                coefficients.data<double>(),
                columns,
                info.get()));
            check(cudaMemcpyAsync(
                &host_info,
                info.get(),
                sizeof(int),
                cudaMemcpyDeviceToHost,
                stream.stream));
            stream.finish();
            if (host_info) {
                throw std::invalid_argument(
                    "fit normal equations could not be solved");
            }
            gemm(
                libraries.blas,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                kernel_pixels,
                1,
                basis_count,
                basis.data<double>(),
                kernel_pixels,
                coefficients.data<double>(),
                basis_count,
                kernel.data<double>(),
                kernel_pixels);
            gemm(
                libraries.blas,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                pixels,
                1,
                background_count,
                backgrounds.get(),
                pixels,
                coefficients.data<double>() + basis_count,
                background_count,
                background.data<double>(),
                pixels);
            const std::size_t fft_pixels =
                static_cast<std::size_t>(fft_height) * fft_width;
            const std::size_t spectrum_pixels =
                static_cast<std::size_t>(fft_height) * (fft_width / 2 + 1);
            FftPlan forward(fft_height, fft_width, CUFFT_D2Z, stream.stream);
            FftPlan inverse(fft_height, fft_width, CUFFT_Z2D, stream.stream);
            DeviceBuffer<double> padded(fft_pixels), convolution(fft_pixels);
            DeviceBuffer<cufftDoubleComplex> reference_spectrum(
                spectrum_pixels),
                kernel_spectrum(spectrum_pixels);
            pad_fft<<<blocks(fft_pixels), threads, 0, stream.stream>>>(
                reference_clean.get(),
                padded.get(),
                height,
                width,
                fft_height,
                fft_width);
            fft_check(cufftExecD2Z(
                forward.handle, padded.get(), reference_spectrum.get()));
            pad_fft<<<blocks(fft_pixels), threads, 0, stream.stream>>>(
                kernel.data<double>(),
                padded.get(),
                kernel_height,
                kernel_width,
                fft_height,
                fft_width);
            fft_check(cufftExecD2Z(
                forward.handle, padded.get(), kernel_spectrum.get()));
            multiply_spectra<<<
                blocks(spectrum_pixels),
                threads,
                0,
                stream.stream>>>(
                reference_spectrum.get(),
                kernel_spectrum.get(),
                spectrum_pixels,
                compute_major < 10);
            fft_check(cufftExecZ2D(
                inverse.handle, reference_spectrum.get(), convolution.get()));
            image_outputs<<<blocks(pixels), threads, 0, stream.stream>>>(
                convolution.get(),
                background.data<double>(),
                target_clean.get(),
                output_invalid.get(),
                matched.data<double>(),
                residual.data<double>(),
                height,
                width,
                fft_height,
                fft_width,
                kernel_height / 2,
                kernel_width / 2);
            DeviceBuffer<double> chi2_values(row_count), chi2_sum(1);
            residual_squares<<<blocks(row_count), threads, 0, stream.stream>>>(
                residual.data<double>(),
                variance_data,
                selected.get(),
                chi2_values.get(),
                row_count,
                2.0);
            std::size_t reduction_bytes = 0;
            check(
                cub::DeviceReduce::Sum(
                    nullptr,
                    reduction_bytes,
                    chi2_values.get(),
                    chi2_sum.get(),
                    row_count,
                    stream.stream));
            DeviceBuffer<unsigned char> reduction_workspace(reduction_bytes);
            check(
                cub::DeviceReduce::Sum(
                    reduction_workspace.get(),
                    reduction_bytes,
                    chi2_values.get(),
                    chi2_sum.get(),
                    row_count,
                    stream.stream));
            check(cudaMemcpyAsync(
                &chi2,
                chi2_sum.get(),
                sizeof(chi2),
                cudaMemcpyDeviceToHost,
                stream.stream));
            stream.finish();
        }
        return Py_BuildValue(
            "{s:i,s:d,s:d}",
            "fit_pixel_count",
            row_count,
            "chi2",
            chi2,
            "condition_number",
            condition_number);
    } catch (...) {
        return translate_exception();
    }
}

}  // namespace cuphoton::imaging
