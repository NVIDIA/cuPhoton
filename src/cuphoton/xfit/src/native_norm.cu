// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

// Row norms matching the CuPy 14.1.1 CUB and singleton reduction order.
// Algorithm reference: cupy/cupy v14.1.1 cupy/_core/_cub_reduction.pyx.
// Singleton fallback reference: cupy/_core/_reduction.pyx at the same tag.
#include <cmath>
#include <cstdint>
#include <cub/block/block_reduce.cuh>
#include <cuda_runtime.h>

template <typename T>
struct Add {
    __device__ __forceinline__ T operator()(T a, T b) const {
        return a + b;
    }
};

template <typename T, int block_size>
__global__ void norm_rows(const T* input, T* output, int observations) {
    using Reduce = cub::BlockReduce<T, block_size>;
    __shared__ typename Reduce::TempStorage storage;
    const T* row = input + std::size_t(blockIdx.x) * observations;
    T aggregate = 0;
    for (int tile = 0; tile < observations; tile += block_size * 4) {
        T values[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            int index = tile + int(threadIdx.x) * 4 + j;
            T value = index < observations ? row[index] : T(0);
            values[j] = value * value;
        }
        aggregate = Add<T>()(aggregate, Reduce(storage).Reduce(values, Add<T>()));
        __syncthreads();
    }
    if (threadIdx.x == 0)
        output[blockIdx.x] = sqrt(aggregate);
}

template <typename T>
__global__ void norm_singleton(const T* input, T* output, int observations, int stride) {
    __shared__ T sums[512];
    int tid = threadIdx.x;
    int output_index = tid % stride;
    int first = output_index + tid / stride;
    T value = 0;
    for (int index = first; index < observations; index += 512 / stride)
        value += input[index] * input[index];
    sums[tid] = value;
    __syncthreads();
    for (int offset = 256; offset >= stride; offset /= 2) {
        if (tid < offset)
            sums[tid] = sums[tid] + sums[tid + offset];
        __syncthreads();
    }
    if (tid == 0)
        output[0] = sqrt(sums[0]);
}

template <typename T>
void launch(const T* input, T* output, int count, int observations, cudaStream_t stream) {
    // (1,m) is both C/F-contiguous. CuPy tests F order first and falls back
    // to its generic axis-one reduction for this shape.
    if (count == 1) {
        int ratio = 512 / observations;
        int stride = 1;
        while (stride <= ratio / 2)
            stride *= 2;
        norm_singleton<<<1, 512, 0, stream>>>(input, output, observations, stride);
        return;
    }
    int blocks = 32;
    while (blocks < 512 && blocks < (observations + 3) / 4)
        blocks *= 2;
    switch (blocks) {
#define LAUNCH(B)                                                                                  \
    case B:                                                                                        \
        norm_rows<T, B><<<count, B, 0, stream>>>(input, output, observations);                     \
        break
        LAUNCH(32);
        LAUNCH(64);
        LAUNCH(128);
        LAUNCH(256);
        LAUNCH(512);
#undef LAUNCH
    }
}

extern "C" int xfit_library_norms(
    const void* input,
    void* output,
    int count,
    int observations,
    bool fp64,
    std::uintptr_t stream_ptr) {
    auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    if (fp64)
        launch(
            static_cast<const double*>(input),
            static_cast<double*>(output),
            count,
            observations,
            stream);
    else
        launch(
            static_cast<const float*>(input),
            static_cast<float*>(output),
            count,
            observations,
            stream);
    // The owning Workspace drains this stream on both success and failure.
    return cudaGetLastError();
}
