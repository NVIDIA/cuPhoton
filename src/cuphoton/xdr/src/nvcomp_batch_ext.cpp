/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Batched DEFLATE decompression — pybind11 helper that bypasses the
// per-tile Python `nvcomp.as_array` loop.
//
// Takes device pointers + offset/length arrays and calls
// `nvcompBatchedDeflateDecompressAsync` directly. The whole call releases the
// GIL so a Python prefetch thread can run during decode.

#include "nvcomp_batch_ext.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cuda_runtime.h>
#include <nvcomp/deflate.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace xdr_gpu {

inline void check_nvcomp(nvcompStatus_t s, const char* ctx) {
    if (s != nvcompSuccess) {
        throw std::runtime_error(
            std::string("nvcomp error in ") + ctx + ": status=" + std::to_string(static_cast<int>(s)));
    }
}

py::object batch_deflate_decompress_impl(
    std::uintptr_t d_concat_ptr,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> rel_offsets,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> lengths,
    std::uintptr_t d_out_ptr,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> out_offsets,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> out_sizes,
    std::uintptr_t stream_ptr,
    bool use_native_pool);

// Batched DEFLATE decompress across N tiles packed inside one device buffer.
//
// Parameters:
//   d_concat_ptr : uintptr_t
//       Device base address of the concatenated compressed bytes.
//   rel_offsets  : int64[n]
//       Per-tile offset (in bytes) from `d_concat_ptr` to the start of each
//       tile's RAW-DEFLATE payload (caller already stripped the gzip wrapper).
//   lengths      : int64[n]
//       Per-tile compressed length in bytes.
//   d_out_ptr    : uintptr_t
//       Device base address of the concatenated output buffer.
//   out_offsets  : int64[n]
//       Per-tile offset inside `d_out_ptr`.
//   out_sizes    : int64[n]
//       Per-tile uncompressed size (must match actual decompressed size).
//   stream_ptr   : uintptr_t
//       CUDA stream. 0 = default stream.
//
// Raises on nvcomp / CUDA failure. Returns None.
void batch_deflate_decompress(
    std::uintptr_t d_concat_ptr,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> rel_offsets,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> lengths,
    std::uintptr_t d_out_ptr,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> out_offsets,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> out_sizes,
    std::uintptr_t stream_ptr) {
    batch_deflate_decompress_impl(
        d_concat_ptr,
        rel_offsets,
        lengths,
        d_out_ptr,
        out_offsets,
        out_sizes,
        stream_ptr,
        false);
}

py::object batch_deflate_decompress_pooled(
    std::uintptr_t d_concat_ptr,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> rel_offsets,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> lengths,
    std::uintptr_t d_out_ptr,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> out_offsets,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> out_sizes,
    std::uintptr_t stream_ptr) {
    return batch_deflate_decompress_impl(
        d_concat_ptr,
        rel_offsets,
        lengths,
        d_out_ptr,
        out_offsets,
        out_sizes,
        stream_ptr,
        true);
}

py::object batch_deflate_decompress_impl(
    std::uintptr_t d_concat_ptr,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> rel_offsets,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> lengths,
    std::uintptr_t d_out_ptr,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> out_offsets,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> out_sizes,
    std::uintptr_t stream_ptr,
    bool use_native_pool) {
    const std::size_t n = static_cast<std::size_t>(rel_offsets.size());
    if (lengths.size() != static_cast<py::ssize_t>(n)
        || out_offsets.size() != static_cast<py::ssize_t>(n)
        || out_sizes.size() != static_cast<py::ssize_t>(n)) {
        throw std::invalid_argument("all per-tile arrays must have the same length");
    }
    if (n == 0) {
        return py::none();
    }

    auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);

    // Build host-side pointer/size arrays.
    std::vector<const void*> h_comp_ptrs(n);
    std::vector<void*>       h_out_ptrs(n);
    std::vector<std::size_t> h_comp_sizes(n);
    std::vector<std::size_t> h_out_sizes(n);

    auto ro = rel_offsets.unchecked<1>();
    auto ln = lengths.unchecked<1>();
    auto oo = out_offsets.unchecked<1>();
    auto os = out_sizes.unchecked<1>();

    std::size_t max_uncomp = 0;
    std::size_t total_uncomp = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (ro(i) < 0 || ln(i) < 0 || oo(i) < 0 || os(i) < 0) {
            throw std::invalid_argument(
                "per-tile offsets and sizes must be non-negative");
        }
        h_comp_ptrs[i]  = reinterpret_cast<const void*>(d_concat_ptr + static_cast<std::size_t>(ro(i)));
        h_out_ptrs[i]   = reinterpret_cast<void*>(d_out_ptr + static_cast<std::size_t>(oo(i)));
        h_comp_sizes[i] = static_cast<std::size_t>(ln(i));
        h_out_sizes[i]  = static_cast<std::size_t>(os(i));
        max_uncomp = std::max(max_uncomp, h_out_sizes[i]);
        total_uncomp += h_out_sizes[i];
    }

    // Allocate device-side mirror arrays + temp in one stream-ordered batch.
    void* d_comp_ptrs  = nullptr;
    void* d_out_ptrs   = nullptr;
    void* d_comp_sizes = nullptr;
    void* d_out_sizes  = nullptr;
    void* d_temp       = nullptr;

    auto opts = nvcompBatchedDeflateDecompressDefaultOpts;
    // The hardware backend requires non-stream-ordered scratch plus explicit
    // actual-size and status buffers. Keep this path on CUDA until those
    // ownership and reporting semantics are implemented together.
    opts.backend = NVCOMP_DECOMPRESS_BACKEND_CUDA;
    std::size_t temp_bytes = 0;
    check_nvcomp(nvcompBatchedDeflateDecompressGetTempSizeAsync(
        n, max_uncomp, opts, &temp_bytes, total_uncomp), "get temp size");

    std::vector<std::shared_ptr<void>> pooled_keepalive;
    if (use_native_pool) {
        int device_id = -1;
        check_cuda(cudaGetDevice(&device_id), "cudaGetDevice native nvcomp scratch");
        auto comp_ptrs = acquire_native_device_allocation(device_id, n * sizeof(void*));
        auto out_ptrs = acquire_native_device_allocation(device_id, n * sizeof(void*));
        auto comp_sizes = acquire_native_device_allocation(device_id, n * sizeof(std::size_t));
        auto out_sizes_buf = acquire_native_device_allocation(device_id, n * sizeof(std::size_t));
        d_comp_ptrs = comp_ptrs.data;
        d_out_ptrs = out_ptrs.data;
        d_comp_sizes = comp_sizes.data;
        d_out_sizes = out_sizes_buf.data;
        pooled_keepalive.push_back(std::move(comp_ptrs.owner));
        pooled_keepalive.push_back(std::move(out_ptrs.owner));
        pooled_keepalive.push_back(std::move(comp_sizes.owner));
        pooled_keepalive.push_back(std::move(out_sizes_buf.owner));
        if (temp_bytes > 0) {
            auto temp = acquire_native_device_allocation(device_id, temp_bytes);
            d_temp = temp.data;
            pooled_keepalive.push_back(std::move(temp.owner));
        }
    } else {
        check_cuda(cudaMallocAsync(&d_comp_ptrs,  n * sizeof(void*),       stream), "alloc comp_ptrs");
        check_cuda(cudaMallocAsync(&d_out_ptrs,   n * sizeof(void*),       stream), "alloc out_ptrs");
        check_cuda(cudaMallocAsync(&d_comp_sizes, n * sizeof(std::size_t), stream), "alloc comp_sizes");
        check_cuda(cudaMallocAsync(&d_out_sizes,  n * sizeof(std::size_t), stream), "alloc out_sizes");
        if (temp_bytes > 0) {
            check_cuda(cudaMallocAsync(&d_temp, temp_bytes, stream), "alloc temp");
        }
    }

    try {
        check_cuda(cudaMemcpyAsync(d_comp_ptrs,  h_comp_ptrs.data(),  n * sizeof(void*),
                                   cudaMemcpyHostToDevice, stream), "copy comp_ptrs");
        check_cuda(cudaMemcpyAsync(d_out_ptrs,   h_out_ptrs.data(),   n * sizeof(void*),
                                   cudaMemcpyHostToDevice, stream), "copy out_ptrs");
        check_cuda(cudaMemcpyAsync(d_comp_sizes, h_comp_sizes.data(), n * sizeof(std::size_t),
                                   cudaMemcpyHostToDevice, stream), "copy comp_sizes");
        check_cuda(cudaMemcpyAsync(d_out_sizes,  h_out_sizes.data(),  n * sizeof(std::size_t),
                                   cudaMemcpyHostToDevice, stream), "copy out_sizes");

        // Release the GIL around the nvcomp call so Python threads (prefetcher)
        // can progress while the async kernel launches + returns.
        {
            py::gil_scoped_release release;
            check_nvcomp(nvcompBatchedDeflateDecompressAsync(
                static_cast<const void* const*>(d_comp_ptrs),
                static_cast<const std::size_t*>(d_comp_sizes),
                static_cast<const std::size_t*>(d_out_sizes),
                nullptr,                     // device_uncompressed_chunk_bytes — optional for CUDA backend
                n,
                d_temp,
                temp_bytes,
                static_cast<void* const*>(d_out_ptrs),
                opts,
                nullptr,                     // device_statuses — optional
                stream
            ), "nvcompBatchedDeflateDecompressAsync");
        }
    } catch (...) {
        if (use_native_pool) {
            cudaStreamSynchronize(stream);
        }
        throw;
    }

    if (use_native_pool) {
        auto holder = new std::vector<std::shared_ptr<void>>(
            std::move(pooled_keepalive));
        return py::capsule(holder, [](void* p) {
            delete reinterpret_cast<std::vector<std::shared_ptr<void>>*>(p);
        });
    }

    // Stream-ordered frees — nvcomp consumes its inputs asynchronously and
    // these frees will be sequenced after the kernel completes.
    if (d_temp)   cudaFreeAsync(d_temp,       stream);
    cudaFreeAsync(d_comp_ptrs,  stream);
    cudaFreeAsync(d_out_ptrs,   stream);
    cudaFreeAsync(d_comp_sizes, stream);
    cudaFreeAsync(d_out_sizes,  stream);
    return py::none();
}



}  // namespace xdr_gpu

PYBIND11_MODULE(_nvcomp_batch_ext, m) {
    m.doc() = "Batched DEFLATE decompression — device-pointer interface to nvcomp.";

    xdr_gpu::bind_io(m);
    xdr_gpu::bind_memory_manager(m);

    m.def("batch_deflate_decompress", &xdr_gpu::batch_deflate_decompress,
          py::arg("d_concat_ptr"),
          py::arg("rel_offsets"),
          py::arg("lengths"),
          py::arg("d_out_ptr"),
          py::arg("out_offsets"),
          py::arg("out_sizes"),
          py::arg("stream_ptr") = 0,
          "Batched DEFLATE decompress across N tiles packed in one device buffer.\n"
          "Caller must have already stripped the RFC-1952 gzip wrapper from each\n"
          "tile (advance rel_offsets past the header, shorten lengths by header+8)."
    );
    m.def("batch_deflate_decompress_pooled", &xdr_gpu::batch_deflate_decompress_pooled,
          py::arg("d_concat_ptr"),
          py::arg("rel_offsets"),
          py::arg("lengths"),
          py::arg("d_out_ptr"),
          py::arg("out_offsets"),
          py::arg("out_sizes"),
          py::arg("stream_ptr") = 0,
          "Batched DEFLATE decompress using native pooled scratch buffers.\n"
          "Returns an owner capsule that must stay alive until stream work completes."
    );
}
