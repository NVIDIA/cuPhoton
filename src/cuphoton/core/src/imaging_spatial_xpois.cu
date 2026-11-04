/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "imaging_api.h"

#include <cublas_v2.h>
#include <cusolverDn.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <map>
#include <memory>

namespace cuphoton::imaging {
namespace {

constexpr int threads = 256;

int blocks(std::int64_t count) {
    return static_cast<int>((count + threads - 1) / threads);
}

void blas_check(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            "native spatial xPois cuBLAS failure: " + std::to_string(status));
    }
}

void solver_check(cusolverStatus_t status) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        throw std::runtime_error(
            "native spatial xPois cuSOLVER failure: " + std::to_string(status));
    }
}

struct Libraries {
    cublasHandle_t blas = nullptr;
    cusolverDnHandle_t solver = nullptr;
    explicit Libraries(cudaStream_t stream) {
        blas_check(cublasCreate(&blas));
        try {
            blas_check(cublasSetStream(blas, stream));
        } catch (...) {
            cublasDestroy(blas);
            throw;
        }
    }

    void prepare_solver(cudaStream_t stream) {
        if (!solver) {
            cusolverDnHandle_t next = nullptr;
            solver_check(cusolverDnCreate(&next));
            try {
                solver_check(cusolverDnSetStream(next, stream));
            } catch (...) {
                cusolverDnDestroy(next);
                throw;
            }
            solver = next;
        }
    }

    void set_stream(cudaStream_t stream) {
        blas_check(cublasSetStream(blas, stream));
        if (solver) {
            solver_check(cusolverDnSetStream(solver, stream));
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

template <class T>
struct Scratch {
    std::unique_ptr<DeviceBuffer<T>> allocation;

    Scratch& resize(std::size_t size) {
        if (!allocation || allocation->count < size) {
            allocation = std::make_unique<DeviceBuffer<T>>(size);
        }
        return *this;
    }

    T* get() const {
        return allocation->get();
    }
};

struct SpatialWorkspace {
    struct State {
        Libraries libraries;
        Scratch<double> reversed, patches, projected, kernels;
        Scratch<double> design, weighted_target, tau, factorization;
        Scratch<int> invalid, info;

        explicit State(cudaStream_t stream)
            : libraries(stream) {}
    };

    int device;
    std::unique_ptr<State> state;
    SpatialWorkspace(int owning_device, cudaStream_t stream)
        : device(owning_device),
          state(std::make_unique<State>(stream)) {}

    ~SpatialWorkspace() {
        // Thread exit may select another device or follow CUDA teardown.
        // Match the other imaging workspaces' non-throwing owned cleanup.
        int previous = device;
        cudaGetDevice(&previous);
        cudaSetDevice(device);
        state.reset();
        cudaSetDevice(previous);
    }
};

SpatialWorkspace::State& workspace_for(int device, cudaStream_t stream) {
    // StreamGuard drains each call on success and failure before these
    // buffers and handles can be reused on another stream in this thread.
    thread_local std::map<int, std::unique_ptr<SpatialWorkspace>> workspaces;
    auto& workspace = workspaces[device];
    if (!workspace) {
        workspace = std::make_unique<SpatialWorkspace>(device, stream);
    }
    workspace->state->libraries.set_stream(stream);
    return *workspace->state;
}

int term_count(int degree) {
    if (degree < 0 || degree > 65533) {
        throw std::invalid_argument(
            "spatial degree is outside the supported range");
    }
    return static_cast<int>(
        (static_cast<std::int64_t>(degree) + 1) * (degree + 2) / 2);
}

struct Model {
    int height, width, kernel_height, kernel_width, area, bases;
    int shape_degree, photometric_degree, background_degree;
    int shape_terms, photometric_terms, background_terms, columns;
    double origin_y, origin_x, low_y, low_x, extent_y, extent_x;
};

void require_shape(const Array& array, const std::vector<std::int64_t>& shape) {
    if (array.shape != shape) {
        throw std::invalid_argument(
            "native spatial xPois buffer shapes differ");
    }
}

void require_tuple(PyObject* value, Py_ssize_t length) {
    if (!PyTuple_Check(value) || PyTuple_Size(value) != length) {
        throw std::invalid_argument(
            "native spatial xPois argument tuple has wrong length");
    }
}

Model resolve_model(
    const Array& source,
    const Array& basis,
    PyObject* domain,
    int shape_degree,
    int photometric_degree,
    int background_degree,
    int chunk_size) {
    source.require('f', 8, 2);
    basis.require('f', 8, 3);
    long long oy, ox, y0, y1, x0, x1;
    if (!PyArg_ParseTuple(domain, "LLLLLL", &oy, &ox, &y0, &y1, &x0, &x1)) {
        throw std::invalid_argument(
            "spatial domain must contain six integer coordinates");
    }
    const long double ey = static_cast<long double>(y1) - y0;
    const long double ex = static_cast<long double>(x1) - x0;
    if (ey <= 0 || ex <= 0 || oy < y0 || ox < x0
        || static_cast<long double>(oy) + source.shape[0] > y1
        || static_cast<long double>(ox) + source.shape[1] > x1) {
        throw std::invalid_argument(
            "source array lies outside the spatial domain");
    }
    if (source.size > INT_MAX || basis.size > INT_MAX) {
        throw std::invalid_argument(
            "native spatial xPois source and basis buffers are limited to "
            "2,147,483,647 elements each (32-bit indexing)");
    }
    if (source.shape[0] < 1 || source.shape[1] < 1 || basis.shape[0] < 1
        || basis.shape[1] < 1 || basis.shape[2] < 1 || !(basis.shape[1] % 2)
        || !(basis.shape[2] % 2) || basis.shape[1] > source.shape[0]
        || basis.shape[2] > source.shape[1]) {
        throw std::invalid_argument(
            "native spatial xPois image or kernel dimensions are invalid");
    }
    Model model{};
    model.height = static_cast<int>(source.shape[0]);
    model.width = static_cast<int>(source.shape[1]);
    model.kernel_height = static_cast<int>(basis.shape[1]);
    model.kernel_width = static_cast<int>(basis.shape[2]);
    model.area = model.kernel_height * model.kernel_width;
    model.bases = static_cast<int>(basis.shape[0]);
    model.shape_degree = shape_degree;
    model.photometric_degree = photometric_degree;
    model.background_degree = background_degree;
    model.shape_terms = term_count(shape_degree);
    model.photometric_terms = term_count(photometric_degree);
    model.background_terms = term_count(background_degree);
    const std::int64_t columns =
        static_cast<std::int64_t>(model.bases - 1) * model.shape_terms
        + model.photometric_terms + model.background_terms;
    if (columns > INT_MAX || chunk_size < 1 || chunk_size > INT_MAX - columns
        || static_cast<std::int64_t>(chunk_size) * model.area > INT_MAX
        || static_cast<std::int64_t>(chunk_size) * model.bases > INT_MAX
        || columns * (columns + chunk_size) > INT_MAX) {
        throw std::invalid_argument(
            "native spatial xPois design dimensions exceed the supported range");
    }
    model.columns = static_cast<int>(columns);
    const int max_degree = std::max(
        photometric_degree,
        std::max(background_degree, model.bases > 1 ? shape_degree : 0));
    if (max_degree > 0 && (ey == 1 || ex == 1)) {
        throw std::invalid_argument(
            "spatial degrees must be zero on singleton domain axes");
    }
    model.origin_y = static_cast<double>(oy);
    model.origin_x = static_cast<double>(ox);
    model.low_y = static_cast<double>(y0);
    model.low_x = static_cast<double>(x0);
    model.extent_y = static_cast<double>(ey - 1);
    model.extent_x = static_cast<double>(ex - 1);
    return model;
}

__device__ double chebyshev(double x, int degree) {
    double prior = 1.0;
    if (degree == 0) {
        return prior;
    }
    double current = x;
    for (int i = 2; i <= degree; ++i) {
        const double next = __dadd_rn(__dmul_rn(2.0 * x, current), -prior);
        prior = current;
        current = next;
    }
    return current;
}

__device__ double term(double x, double y, int degree, int index) {
    int dx = 0;
    while (index >= degree + 1 - dx) {
        index -= degree + 1 - dx;
        ++dx;
    }
    return chebyshev(x, dx) * chebyshev(y, index);
}

__device__ void coordinates(
    std::int64_t pixel, const Model& model, double& x, double& y) {
    x = model.extent_x == 0.0
        ? 0.0
        : __dadd_rn(
              __ddiv_rn(
                  2.0 * ((pixel % model.width + model.origin_x) - model.low_x),
                  model.extent_x),
              -1.0);
    y = model.extent_y == 0.0
        ? 0.0
        : __dadd_rn(
              __ddiv_rn(
                  2.0 * ((pixel / model.width + model.origin_y) - model.low_y),
                  model.extent_y),
              -1.0);
}

__global__ void validate_indices(
    const std::int64_t* indices, int count, Model model, int* invalid) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }
    const auto pixel = indices[i];
    if (pixel < 0
        || pixel >= static_cast<std::int64_t>(model.height) * model.width) {
        atomicExch(invalid, 1);
        return;
    }
    const int y = static_cast<int>(pixel / model.width);
    const int x = static_cast<int>(pixel % model.width);
    if (y < model.kernel_height / 2
        || y >= model.height - model.kernel_height / 2
        || x < model.kernel_width / 2
        || x >= model.width - model.kernel_width / 2) {
        atomicExch(invalid, 1);
    }
}

void check_indices(
    const Array& indices, Model model, cudaStream_t stream, int* invalid) {
    indices.require('i', 8, 1);
    if (indices.size > INT_MAX) {
        throw std::invalid_argument(
            "native spatial xPois has too many selected rows");
    }
    if (!indices.size) {
        return;
    }
    check(cudaMemsetAsync(invalid, 0, sizeof(int), stream));
    validate_indices<<<blocks(indices.size), threads, 0, stream>>>(
        indices.data<std::int64_t>(),
        static_cast<int>(indices.size),
        model,
        invalid);
    int host_invalid = 0;
    check(cudaMemcpyAsync(
        &host_invalid, invalid, sizeof(int), cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream));
    if (host_invalid) {
        throw std::invalid_argument(
            "selected spatial fit or output rows lie outside the kernel interior");
    }
}

__global__ void reverse_basis(
    const double* basis, double* reversed, int count, int area) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        reversed[i] = basis[(i / area) * area + area - 1 - i % area];
    }
}

__global__ void gather_patches(
    const double* source,
    const std::int64_t* indices,
    int rows,
    Model model,
    double* patches) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * model.area) {
        return;
    }
    const auto center = indices[i / model.area];
    const int k = i % model.area;
    const auto pixel = center
        + (k / model.kernel_width - model.kernel_height / 2) * model.width
        + k % model.kernel_width - model.kernel_width / 2;
    patches[i] = source[pixel];
}

void project_basis(
    cublasHandle_t blas,
    const double* reversed,
    const double* patches,
    double* projected,
    int rows,
    Model model) {
    const double one = 1.0, zero = 0.0;
    blas_check(cublasDgemm(
        blas,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        model.bases,
        rows,
        model.area,
        &one,
        reversed,
        model.area,
        patches,
        model.area,
        &zero,
        projected,
        model.bases));
}

__global__ void weighted_design(
    const double* projected,
    const double* target,
    const double* variance,
    const std::int64_t* indices,
    const double* precision,
    const double* upper,
    const double* rhs,
    int prior_rows,
    int rows,
    Model model,
    double* design,
    double* weighted_target,
    int* invalid) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int leading = prior_rows + rows;
    if (i >= leading * model.columns) {
        return;
    }
    const int column = i / leading;
    const int row = i % leading;
    if (row < prior_rows) {
        design[i] = upper[row * model.columns + column];
        if (column == 0) {
            weighted_target[row] = rhs[row];
        }
        return;
    }
    const int sample = row - prior_rows;
    const auto pixel = indices[sample];
    const double scale = __ddiv_rn(
        precision ? __dsqrt_rn(precision[sample]) : 1.0,
        __dsqrt_rn(variance[pixel]));
    if (!isfinite(scale) || scale <= 0.0) {
        atomicExch(invalid, 1);
    }
    double x, y;
    coordinates(pixel, model, x, y);
    double value;
    const int shape_end =
        model.photometric_terms + (model.bases - 1) * model.shape_terms;
    if (column < model.photometric_terms) {
        value = projected[sample * model.bases]
            * term(x, y, model.photometric_degree, column);
    } else if (column < shape_end) {
        const int offset = column - model.photometric_terms;
        value = projected[sample * model.bases + 1 + offset / model.shape_terms]
            * term(x, y, model.shape_degree, offset % model.shape_terms);
    } else {
        value = term(x, y, model.background_degree, column - shape_end);
    }
    design[i] = value * scale;
    if (column == 0) {
        weighted_target[row] = target[pixel] * scale;
    }
}

__global__ void extract_qr(
    const double* design,
    const double* transformed,
    int leading,
    int retained,
    int columns,
    double* upper,
    double* rhs) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < columns * columns) {
        const int row = i / columns;
        const int col = i % columns;
        upper[i] =
            row < retained && row <= col ? design[col * leading + row] : 0.0;
        if (col == 0) {
            rhs[row] = row < retained ? transformed[row] : 0.0;
        }
    }
}

__global__ void fill_nan(double* values, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        values[i] = CUDART_NAN;
    }
}

__device__ double field(
    const double* coefficients, double x, double y, int degree, int count) {
    double value = 0.0;
    for (int i = 0; i < count; ++i) {
        value += coefficients[i] * term(x, y, degree, i);
    }
    return value;
}

__global__ void reconstruct(
    const double* projected,
    const double* coefficients,
    const std::int64_t* indices,
    int rows,
    Model model,
    double* matched) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    double x, y;
    coordinates(indices[row], model, x, y);
    double value = projected[row * model.bases]
        * field(
            coefficients,
            x,
            y,
            model.photometric_degree,
            model.photometric_terms);
    for (int basis = 1; basis < model.bases; ++basis) {
        value += projected[row * model.bases + basis]
            * field(
                coefficients + model.photometric_terms
                    + (basis - 1) * model.shape_terms,
                x,
                y,
                model.shape_degree,
                model.shape_terms);
    }
    value += field(
        coefficients + model.photometric_terms
            + (model.bases - 1) * model.shape_terms,
        x,
        y,
        model.background_degree,
        model.background_terms);
    matched[indices[row]] = value;
}

__global__ void kernel_weights(
    const double* coefficients,
    const std::int64_t* indices,
    int rows,
    Model model,
    double* weights) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * model.bases) {
        return;
    }
    const int row = i / model.bases;
    const int basis = i % model.bases;
    double x, y;
    coordinates(indices[row], model, x, y);
    weights[i] = basis == 0 ? field(
                                  coefficients,
                                  x,
                                  y,
                                  model.photometric_degree,
                                  model.photometric_terms)
                            : field(
                                  coefficients + model.photometric_terms
                                      + (basis - 1) * model.shape_terms,
                                  x,
                                  y,
                                  model.shape_degree,
                                  model.shape_terms);
}

__global__ void propagate_variance(
    const double* patches,
    const double* kernels,
    const std::int64_t* indices,
    int rows,
    Model model,
    double* propagated) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    double value = 0.0;
    for (int k = 0; k < model.area; ++k) {
        const double kernel = kernels[row * model.area + k];
        value += patches[row * model.area + k] * (kernel * kernel);
    }
    propagated[indices[row]] = value;
}

}  // namespace

PyObject* accumulate_spatial_qr(PyObject*, PyObject* arguments) {
    PyObject *inputs, *outputs, *domain;
    int shape_degree, photometric_degree, background_degree, chunk_size;
    unsigned long long stream_value;
    if (!PyArg_ParseTuple(
            arguments,
            "OOOiiiiK",
            &inputs,
            &outputs,
            &domain,
            &shape_degree,
            &photometric_degree,
            &background_degree,
            &chunk_size,
            &stream_value)) {
        return nullptr;
    }
    try {
        require_tuple(inputs, 6);
        require_tuple(outputs, 2);
        Array source(PyTuple_GetItem(inputs, 0));
        Array target(PyTuple_GetItem(inputs, 1));
        Array variance(PyTuple_GetItem(inputs, 2));
        Array basis(PyTuple_GetItem(inputs, 3));
        Array indices(PyTuple_GetItem(inputs, 4));
        std::unique_ptr<Array> precision;
        if (PyTuple_GetItem(inputs, 5) != Py_None) {
            precision = std::make_unique<Array>(PyTuple_GetItem(inputs, 5));
            precision->require('f', 8, 1);
            require_shape(*precision, indices.shape);
        }
        Array upper(PyTuple_GetItem(outputs, 0));
        Array rhs(PyTuple_GetItem(outputs, 1));
        const Model model = resolve_model(
            source,
            basis,
            domain,
            shape_degree,
            photometric_degree,
            background_degree,
            chunk_size);
        target.require('f', 8, 2);
        variance.require('f', 8, 2);
        indices.require('i', 8, 1);
        upper.require('f', 8, 2, true);
        rhs.require('f', 8, 1, true);
        require_shape(target, source.shape);
        require_shape(variance, source.shape);
        require_shape(upper, {model.columns, model.columns});
        require_shape(rhs, {model.columns});
        if (indices.size < model.columns) {
            throw std::invalid_argument(
                "fit region is underdetermined for the requested spatial model");
        }
        {
            Detached detached;
            StreamGuard stream(stream_value);
            int device = 0;
            check(cudaGetDevice(&device));
            for (const Array* array :
                {&source, &target, &variance, &basis, &indices, &upper, &rhs}) {
                array->wait(stream.stream);
            }
            if (precision) {
                precision->wait(stream.stream);
            }
            auto& scratch = workspace_for(device, stream.stream);
            auto& invalid = scratch.invalid.resize(1);
            check_indices(indices, model, stream.stream, invalid.get());
            auto& libraries = scratch.libraries;
            libraries.prepare_solver(stream.stream);
            const int capacity = chunk_size + model.columns;
            auto& reversed = scratch.reversed.resize(basis.size);
            auto& patches = scratch.patches.resize(
                static_cast<std::size_t>(chunk_size) * model.area);
            auto& projected = scratch.projected.resize(
                static_cast<std::size_t>(chunk_size) * model.bases);
            auto& design = scratch.design.resize(
                static_cast<std::size_t>(capacity) * model.columns);
            auto& weighted_target = scratch.weighted_target.resize(capacity);
            auto& tau = scratch.tau.resize(model.columns);
            auto& info = scratch.info.resize(3);
            int qr_workspace, apply_workspace;
            solver_check(cusolverDnDgeqrf_bufferSize(
                libraries.solver,
                capacity,
                model.columns,
                design.get(),
                capacity,
                &qr_workspace));
            solver_check(cusolverDnDormqr_bufferSize(
                libraries.solver,
                CUBLAS_SIDE_LEFT,
                CUBLAS_OP_T,
                capacity,
                1,
                model.columns,
                design.get(),
                capacity,
                tau.get(),
                weighted_target.get(),
                capacity,
                &apply_workspace));
            const int workspace_size = std::max(qr_workspace, apply_workspace);
            auto& workspace = scratch.factorization.resize(workspace_size);
            reverse_basis<<<blocks(basis.size), threads, 0, stream.stream>>>(
                basis.data<double>(),
                reversed.get(),
                static_cast<int>(basis.size),
                model.area);
            int prior_rows = 0;
            for (int start = 0; start < indices.size;) {
                const int rows = static_cast<int>(
                    std::min<std::int64_t>(chunk_size, indices.size - start));
                const int leading = prior_rows + rows;
                const int retained = std::min(leading, model.columns);
                const auto* selected = indices.data<std::int64_t>() + start;
                gather_patches<<<
                    blocks(static_cast<std::int64_t>(rows) * model.area),
                    threads,
                    0,
                    stream.stream>>>(
                    source.data<double>(),
                    selected,
                    rows,
                    model,
                    patches.get());
                project_basis(
                    libraries.blas,
                    reversed.get(),
                    patches.get(),
                    projected.get(),
                    rows,
                    model);
                check(cudaMemsetAsync(
                    info.get(), 0, 3 * sizeof(int), stream.stream));
                weighted_design<<<
                    blocks(static_cast<std::int64_t>(leading) * model.columns),
                    threads,
                    0,
                    stream.stream>>>(
                    projected.get(),
                    target.data<double>(),
                    variance.data<double>(),
                    selected,
                    precision ? precision->data<double>() + start : nullptr,
                    upper.data<double>(),
                    rhs.data<double>(),
                    prior_rows,
                    rows,
                    model,
                    design.get(),
                    weighted_target.get(),
                    info.get() + 2);
                solver_check(cusolverDnDgeqrf(
                    libraries.solver,
                    leading,
                    model.columns,
                    design.get(),
                    leading,
                    tau.get(),
                    workspace.get(),
                    workspace_size,
                    info.get()));
                solver_check(cusolverDnDormqr(
                    libraries.solver,
                    CUBLAS_SIDE_LEFT,
                    CUBLAS_OP_T,
                    leading,
                    1,
                    retained,
                    design.get(),
                    leading,
                    tau.get(),
                    weighted_target.get(),
                    leading,
                    workspace.get(),
                    workspace_size,
                    info.get() + 1));
                int host_info[3] = {};
                check(cudaMemcpyAsync(
                    host_info,
                    info.get(),
                    sizeof(host_info),
                    cudaMemcpyDeviceToHost,
                    stream.stream));
                check(cudaStreamSynchronize(stream.stream));
                if (host_info[2]) {
                    throw std::invalid_argument(
                        "combined fit row scales must be finite and positive");
                }
                if (host_info[0] || host_info[1]) {
                    throw std::runtime_error(
                        "native spatial xPois QR factorization failed");
                }
                extract_qr<<<
                    blocks(
                        static_cast<std::int64_t>(model.columns)
                        * model.columns),
                    threads,
                    0,
                    stream.stream>>>(
                    design.get(),
                    weighted_target.get(),
                    leading,
                    retained,
                    model.columns,
                    upper.data<double>(),
                    rhs.data<double>());
                prior_rows = retained;
                start += rows;
            }
            stream.finish();
        }
        Py_RETURN_NONE;
    } catch (...) {
        return translate_exception();
    }
}

PyObject* evaluate_spatial_kernel(PyObject*, PyObject* arguments) {
    PyObject *inputs, *outputs, *domain;
    int shape_degree, photometric_degree, background_degree, chunk_size;
    unsigned long long stream_value;
    if (!PyArg_ParseTuple(
            arguments,
            "OOOiiiiK",
            &inputs,
            &outputs,
            &domain,
            &shape_degree,
            &photometric_degree,
            &background_degree,
            &chunk_size,
            &stream_value)) {
        return nullptr;
    }
    try {
        require_tuple(inputs, 6);
        require_tuple(outputs, 2);
        Array source(PyTuple_GetItem(inputs, 0));
        Array basis(PyTuple_GetItem(inputs, 1));
        Array coefficients(PyTuple_GetItem(inputs, 2));
        Array indices(PyTuple_GetItem(inputs, 3));
        Array matched(PyTuple_GetItem(outputs, 0));
        std::unique_ptr<Array> variance, diagnostic_indices, propagated;
        const bool diagnostics = PyTuple_GetItem(inputs, 4) != Py_None;
        if (diagnostics != (PyTuple_GetItem(inputs, 5) != Py_None)
            || diagnostics != (PyTuple_GetItem(outputs, 1) != Py_None)) {
            throw std::invalid_argument(
                "source variance diagnostics require all three buffers");
        }
        if (diagnostics) {
            variance = std::make_unique<Array>(PyTuple_GetItem(inputs, 4));
            diagnostic_indices =
                std::make_unique<Array>(PyTuple_GetItem(inputs, 5));
            propagated = std::make_unique<Array>(PyTuple_GetItem(outputs, 1));
            variance->require('f', 8, 2);
            diagnostic_indices->require('i', 8, 1);
            propagated->require('f', 8, 2, true);
            require_shape(*variance, source.shape);
            require_shape(*propagated, source.shape);
        }
        const Model model = resolve_model(
            source,
            basis,
            domain,
            shape_degree,
            photometric_degree,
            background_degree,
            chunk_size);
        coefficients.require('f', 8, 1);
        indices.require('i', 8, 1);
        matched.require('f', 8, 2, true);
        require_shape(coefficients, {model.columns});
        require_shape(matched, source.shape);
        {
            Detached detached;
            StreamGuard stream(stream_value);
            int device = 0;
            check(cudaGetDevice(&device));
            for (const Array* array :
                {&source, &basis, &coefficients, &indices, &matched}) {
                array->wait(stream.stream);
            }
            if (diagnostics) {
                variance->wait(stream.stream);
                diagnostic_indices->wait(stream.stream);
                propagated->wait(stream.stream);
            }
            auto& scratch = workspace_for(device, stream.stream);
            auto& invalid = scratch.invalid.resize(1);
            check_indices(indices, model, stream.stream, invalid.get());
            if (diagnostics) {
                check_indices(
                    *diagnostic_indices, model, stream.stream, invalid.get());
            }
            auto& libraries = scratch.libraries;
            auto& reversed = scratch.reversed.resize(basis.size);
            auto& patches = scratch.patches.resize(
                static_cast<std::size_t>(chunk_size) * model.area);
            auto& projected = scratch.projected.resize(
                static_cast<std::size_t>(chunk_size) * model.bases);
            reverse_basis<<<blocks(basis.size), threads, 0, stream.stream>>>(
                basis.data<double>(),
                reversed.get(),
                static_cast<int>(basis.size),
                model.area);
            fill_nan<<<blocks(source.size), threads, 0, stream.stream>>>(
                matched.data<double>(), static_cast<int>(source.size));
            for (int start = 0; start < indices.size;) {
                const int rows = static_cast<int>(
                    std::min<std::int64_t>(chunk_size, indices.size - start));
                const auto* selected = indices.data<std::int64_t>() + start;
                gather_patches<<<
                    blocks(static_cast<std::int64_t>(rows) * model.area),
                    threads,
                    0,
                    stream.stream>>>(
                    source.data<double>(),
                    selected,
                    rows,
                    model,
                    patches.get());
                project_basis(
                    libraries.blas,
                    reversed.get(),
                    patches.get(),
                    projected.get(),
                    rows,
                    model);
                reconstruct<<<blocks(rows), threads, 0, stream.stream>>>(
                    projected.get(),
                    coefficients.data<double>(),
                    selected,
                    rows,
                    model,
                    matched.data<double>());
                start += rows;
            }
            if (diagnostics) {
                auto& kernels = scratch.kernels.resize(
                    static_cast<std::size_t>(chunk_size) * model.area);
                check(cudaMemsetAsync(
                    propagated->pointer,
                    0,
                    propagated->size * sizeof(double),
                    stream.stream));
                const double one = 1.0, zero = 0.0;
                for (int start = 0; start < diagnostic_indices->size;) {
                    const int rows = static_cast<int>(std::min<std::int64_t>(
                        chunk_size, diagnostic_indices->size - start));
                    const auto* selected =
                        diagnostic_indices->data<std::int64_t>() + start;
                    gather_patches<<<
                        blocks(static_cast<std::int64_t>(rows) * model.area),
                        threads,
                        0,
                        stream.stream>>>(
                        variance->data<double>(),
                        selected,
                        rows,
                        model,
                        patches.get());
                    kernel_weights<<<
                        blocks(static_cast<std::int64_t>(rows) * model.bases),
                        threads,
                        0,
                        stream.stream>>>(
                        coefficients.data<double>(),
                        selected,
                        rows,
                        model,
                        projected.get());
                    blas_check(cublasDgemm(
                        libraries.blas,
                        CUBLAS_OP_N,
                        CUBLAS_OP_N,
                        model.area,
                        rows,
                        model.bases,
                        &one,
                        reversed.get(),
                        model.area,
                        projected.get(),
                        model.bases,
                        &zero,
                        kernels.get(),
                        model.area));
                    propagate_variance<<<
                        blocks(rows),
                        threads,
                        0,
                        stream.stream>>>(
                        patches.get(),
                        kernels.get(),
                        selected,
                        rows,
                        model,
                        propagated->data<double>());
                    start += rows;
                }
            }
            stream.finish();
        }
        Py_RETURN_NONE;
    } catch (...) {
        return translate_exception();
    }
}

}  // namespace cuphoton::imaging
