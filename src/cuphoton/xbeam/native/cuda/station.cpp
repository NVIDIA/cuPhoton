// SPDX-License-Identifier: Apache-2.0
#include "xbeam/cuda.hpp"
#include "xbeam/radio.hpp"
#include <cmath>
#include <cuda_runtime.h>
#include <cufft.h>
#include <limits>
namespace xbeam {
void launch_front(
    const StationLayout&,
    const void*,
    const void*,
    const void*,
    void*,
    void*,
    void*,
    cudaStream_t);
void launch_channel(
    const StationLayout&, const void*, const void*, void*, cudaStream_t);
namespace {
void check(cudaError_t e) {
    if (e != cudaSuccess) {
        throw RadioError(cudaGetErrorString(e));
    }
}
void check_fft(cufftResult e) {
    if (e != CUFFT_SUCCESS) {
        throw RadioError("cuFFT error " + std::to_string(int(e)));
    }
}
struct Memory {
    void* pointer = nullptr;
    void allocate(std::size_t bytes) {
        if (bytes) {
            check(cudaMalloc(&pointer, bytes));
        }
    }
    ~Memory() {
        if (pointer) {
            cudaFree(pointer);
        }
    }
};
struct Stream {
    cudaStream_t value = nullptr;
    Stream() {
        check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking));
    }
    ~Stream() {
        if (value) {
            cudaStreamSynchronize(value);
            cudaStreamDestroy(value);
        }
    }
};
struct FFT {
    cufftHandle value = 0;
    FFT() {
        check_fft(cufftCreate(&value));
    }
    ~FFT() {
        if (value) {
            cufftDestroy(value);
        }
    }
};
std::size_t size(int rows, int columns, std::size_t item) {
    if (rows <= 0 || columns <= 0
        || std::size_t(rows)
            > std::numeric_limits<std::size_t>::max() / columns / item) {
        throw RadioError("invalid native CUDA shape");
    }
    return std::size_t(rows) * columns * item;
}
}  // namespace
struct CudaStation::Impl {
    StationLayout g;
    // Stream destroyed before buffers: all queued consumers must finish first.
    Memory rotated, output, weight, counts, workspace;
    FFT plan;
    Stream work;
    std::size_t output_bytes, weight_bytes;
    explicit Impl(const StationLayout& layout)
        : g(layout) {
        if (g.channels <= 0 || g.channels > g.fft_samples || g.mapping < 0
            || g.mapping > 3 || (g.mapping >= 2 && g.channels != g.fft_samples)
            || !std::isfinite(g.sample_rate) || g.sample_rate <= 0
            || !std::isfinite(g.reference_hz) || !std::isfinite(g.lo_hz)
            || !std::isfinite(g.wall_lo_cycles) || g.validity_stride < 1
            || g.samples_per_frame < 1 || g.payload_bytes < 1
            || g.channel_count < 1 || g.channel_index < 0
            || g.channel_index >= g.channel_count || g.source_samples < 1) {
            throw RadioError("invalid native CUDA station geometry");
        }
        const auto bits = std::int64_t(g.samples_per_frame) * g.channel_count
            * (g.complex_input ? 4 : 2);
        if (bits != std::int64_t(g.payload_bytes) * 8
            || g.source_samples % g.samples_per_frame) {
            throw RadioError("inconsistent native packed sample geometry");
        }
        output_bytes = size(g.rows, g.channels, 8);
        weight_bytes = size(g.rows, 1, 8);
        rotated.allocate(size(g.rows, g.fft_samples, 8));
        output.allocate(output_bytes);
        weight.allocate(weight_bytes);
        if (!g.complex_input && g.fft_samples == 262144 && g.rows >= 8
            && g.rows <= 1600) {
            counts.allocate(size(g.rows, 64, 4));
        }
        check_fft(cufftSetAutoAllocation(plan.value, 0));
        std::size_t bytes = 0;
        check_fft(cufftMakePlan1d(
            plan.value, g.fft_samples, CUFFT_C2C, g.rows, &bytes));
        workspace.allocate(bytes);
        check_fft(cufftSetWorkArea(plan.value, workspace.pointer));
        check_fft(cufftSetStream(plan.value, work.value));
    }
};
CudaStation::CudaStation(const StationLayout& g)
    : impl(std::make_unique<Impl>(g)) {}
CudaStation::~CudaStation() = default;
void CudaStation::submit(
    std::uintptr_t source, std::uintptr_t validity, std::uintptr_t parameters) {
    if (!source || !validity || !parameters) {
        throw RadioError("missing native CUDA input address");
    }
    auto& p = *impl;
    launch_front(
        p.g,
        reinterpret_cast<void*>(source),
        reinterpret_cast<void*>(validity),
        reinterpret_cast<void*>(parameters),
        p.rotated.pointer,
        p.weight.pointer,
        p.counts.pointer,
        p.work.value);
    check(cudaGetLastError());
    check_fft(cufftExecC2C(
        p.plan.value,
        static_cast<cufftComplex*>(p.rotated.pointer),
        static_cast<cufftComplex*>(p.rotated.pointer),
        CUFFT_FORWARD));
    launch_channel(
        p.g,
        p.rotated.pointer,
        reinterpret_cast<void*>(parameters),
        p.output.pointer,
        p.work.value);
    check(cudaGetLastError());
}
void CudaStation::download(float* spectra, double* weights) {
    if (!spectra || !weights) {
        throw RadioError("missing native output buffers");
    }
    check(cudaMemcpyAsync(
        spectra,
        impl->output.pointer,
        impl->output_bytes,
        cudaMemcpyDeviceToHost,
        impl->work.value));
    check(cudaMemcpyAsync(
        weights,
        impl->weight.pointer,
        impl->weight_bytes,
        cudaMemcpyDeviceToHost,
        impl->work.value));
    check(cudaStreamSynchronize(impl->work.value));
}
std::uintptr_t CudaStation::stream() const {
    return reinterpret_cast<std::uintptr_t>(impl->work.value);
}
std::uintptr_t CudaStation::spectra() const {
    return reinterpret_cast<std::uintptr_t>(impl->output.pointer);
}
std::uintptr_t CudaStation::weights() const {
    return reinterpret_cast<std::uintptr_t>(impl->weight.pointer);
}
}  // namespace xbeam

namespace xbeam {
void CudaStation::source_window(std::int64_t first, std::int64_t count) {
    if (count < 1 || count % impl->g.samples_per_frame) {
        throw RadioError("invalid packed source window");
    }
    impl->g.source_start = first;
    impl->g.source_samples = count;
}
}  // namespace xbeam
