// SPDX-License-Identifier: Apache-2.0
#include "xbeam/cuda.hpp"
#include "xbeam/radio.hpp"
#include <algorithm>
#include <cuda_runtime.h>
namespace xbeam {
__global__ void unpack_frame_payload(
    const unsigned char* raw,
    unsigned char* payload,
    long long total,
    long long framed,
    long long bytes,
    long long prefix) {
    for (long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        index < total;
        index += (long long)gridDim.x * blockDim.x) {
        payload[index] =
            raw[prefix + (index / bytes) * framed + 32 + index % bytes];
    }
}
template <int PayloadBytes>
__global__ void unpack_frame_payload_chunk(
    const unsigned char* raw,
    unsigned char* payload,
    long long first,
    long long destination_first,
    long long total,
    long long framed,
    long long bytes,
    long long file_offset,
    long long read_bytes) {
    const long long width = PayloadBytes ? PayloadBytes : bytes;
    for (long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        index < total;
        index += (long long)gridDim.x * blockDim.x) {
        auto source = (first + index / width) * framed + 32 + index % width;
        if (source >= file_offset && source - file_offset < read_bytes) {
            payload[(first - destination_first) * width + index] =
                raw[source - file_offset];
        }
    }
}

void unpack_frame_chunk(
    const void* raw,
    void* payload,
    std::int64_t destination_first,
    std::int64_t frames,
    std::int64_t framed,
    std::int64_t bytes,
    std::int64_t offset,
    std::int64_t read_bytes,
    std::uintptr_t stream) {
    if (!raw || !payload || destination_first < 0 || frames <= 0 || framed <= 32
        || bytes != framed - 32 || offset < 0 || read_bytes <= 0
        || offset > INT64_MAX - read_bytes
        || destination_first > INT64_MAX - frames
        || destination_first + frames > INT64_MAX / framed) {
        throw RadioError("invalid native chunk extraction scope");
    }
    auto first = std::max(destination_first, offset / framed);
    auto stop = std::min(
        destination_first + frames, 1 + (offset + read_bytes - 1) / framed);
    if (stop <= first) {
        return;
    }
    if (stop - first > INT64_MAX / bytes) {
        throw RadioError("native chunk extraction exceeds int64");
    }
    auto total = (stop - first) * bytes;
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(std::min<std::int64_t>((total - 1) / 256 + 1, 65535));
    config.blockDim = dim3(256);
    config.stream = reinterpret_cast<cudaStream_t>(stream);
    auto launch = [&](auto kernel) {
        return cudaLaunchKernelEx(
            &config,
            kernel,
            static_cast<const unsigned char*>(raw),
            static_cast<unsigned char*>(payload),
            first,
            destination_first,
            total,
            framed,
            bytes,
            offset,
            read_bytes);
    };
    auto status = bytes == 8192 ? launch(unpack_frame_payload_chunk<8192>)
        : bytes == 8000         ? launch(unpack_frame_payload_chunk<8000>)
                                : launch(unpack_frame_payload_chunk<0>);
    if (status != cudaSuccess) {
        throw RadioError(cudaGetErrorString(status));
    }
}
void unpack_frames(
    const void* raw,
    void* payload,
    std::int64_t frames,
    std::int64_t framed,
    std::int64_t bytes,
    std::int64_t prefix,
    std::uintptr_t stream) {
    if (!raw || !payload || frames <= 0 || framed <= 32 || bytes != framed - 32
        || prefix < 0 || frames > INT64_MAX / bytes) {
        throw RadioError("invalid native framed payload extraction");
    }
    auto total = frames * bytes;
    unpack_frame_payload<<<
        static_cast<unsigned int>(
            std::min<std::int64_t>((total + 255) / 256, 65535)),
        256,
        0,
        reinterpret_cast<cudaStream_t>(stream)>>>(
        static_cast<const unsigned char*>(raw),
        static_cast<unsigned char*>(payload),
        total,
        framed,
        bytes,
        prefix);
    auto status = cudaGetLastError();
    if (status != cudaSuccess) {
        throw RadioError(cudaGetErrorString(status));
    }
}
}  // namespace xbeam
