/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Batched Gzip/DEFLATE decompression — pybind11 helper that bypasses the
// per-tile Python `nvcomp.as_array` loop.
//
// Takes device pointers + offset/length arrays and calls
// the corresponding nvCOMP batched API directly. The decode launch releases the
// GIL so a Python prefetch thread can run during decode.

#include "nvcomp_batch_ext.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cuda_runtime.h>
#include <nvcomp/deflate.h>
#include <nvcomp/gzip.h>

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
            std::string("nvcomp error in ") + ctx
            + ": status=" + std::to_string(static_cast<int>(s)));
    }
}

py::object batch_decompress_impl(
    std::uintptr_t d_concat_ptr,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> rel_offsets,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> lengths,
    std::uintptr_t d_out_ptr,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> out_offsets,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> out_sizes,
    std::uintptr_t stream_ptr,
    bool use_native_pool,
    bool gzip_wrapped);

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
    batch_decompress_impl(
        d_concat_ptr,
        rel_offsets,
        lengths,
        d_out_ptr,
        out_offsets,
        out_sizes,
        stream_ptr,
        false,
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
    return batch_decompress_impl(
        d_concat_ptr,
        rel_offsets,
        lengths,
        d_out_ptr,
        out_offsets,
        out_sizes,
        stream_ptr,
        true,
        false);
}

py::object batch_gzip_decompress(
    std::uintptr_t d_concat_ptr,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> rel_offsets,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> lengths,
    std::uintptr_t d_out_ptr,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> out_offsets,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> out_sizes,
    std::uintptr_t stream_ptr,
    bool use_native_pool) {
    return batch_decompress_impl(
        d_concat_ptr,
        rel_offsets,
        lengths,
        d_out_ptr,
        out_offsets,
        out_sizes,
        stream_ptr,
        use_native_pool,
        true);
}

py::object batch_decompress_impl(
    std::uintptr_t d_concat_ptr,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> rel_offsets,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> lengths,
    std::uintptr_t d_out_ptr,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> out_offsets,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> out_sizes,
    std::uintptr_t stream_ptr,
    bool use_native_pool,
    bool gzip_wrapped) {
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
    std::vector<void*> h_out_ptrs(n);
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
            throw std::invalid_argument("per-tile offsets and sizes must be non-negative");
        }
        h_comp_ptrs[i] =
            reinterpret_cast<const void*>(d_concat_ptr + static_cast<std::size_t>(ro(i)));
        if (!gzip_wrapped && reinterpret_cast<std::uintptr_t>(h_comp_ptrs[i]) % 4 != 0) {
            throw std::invalid_argument("raw DEFLATE inputs must be 4-byte aligned");
        }
        h_out_ptrs[i] = reinterpret_cast<void*>(d_out_ptr + static_cast<std::size_t>(oo(i)));
        h_comp_sizes[i] = static_cast<std::size_t>(ln(i));
        h_out_sizes[i] = static_cast<std::size_t>(os(i));
        max_uncomp = std::max(max_uncomp, h_out_sizes[i]);
        total_uncomp += h_out_sizes[i];
    }

    // Allocate device-side mirror arrays + temp in one stream-ordered batch.
    void* d_comp_ptrs = nullptr;
    void* d_out_ptrs = nullptr;
    void* d_comp_sizes = nullptr;
    void* d_out_sizes = nullptr;
    void* d_temp = nullptr;
    void* d_statuses = nullptr;
    void* d_actual_sizes = nullptr;

    auto opts = nvcompBatchedDeflateDecompressDefaultOpts;
    // The hardware backend requires non-stream-ordered scratch allocations.
    // Keep this path on CUDA until that ownership contract is supported.
    opts.backend = NVCOMP_DECOMPRESS_BACKEND_CUDA;
    auto gzip_opts = nvcompBatchedGzipDecompressDefaultOpts;
    // Let nvCOMP use compatible hardware engines and fall back to CUDA.
    gzip_opts.backend = NVCOMP_DECOMPRESS_BACKEND_DEFAULT;
    // NAIVE accepts byte-aligned FITS gzip tile starts. LOOKAHEAD requires
    // additional input alignment and is intended for much larger chunks.
    gzip_opts.algorithm = NVCOMP_GZIP_DECOMPRESS_ALGORITHM_NAIVE;
    std::size_t temp_bytes = 0;
    check_nvcomp(
        gzip_wrapped ? nvcompBatchedGzipDecompressGetTempSizeAsync(
                           n, max_uncomp, gzip_opts, &temp_bytes, total_uncomp)
                     : nvcompBatchedDeflateDecompressGetTempSizeAsync(
                           n, max_uncomp, opts, &temp_bytes, total_uncomp),
        "get temp size");

    std::vector<std::shared_ptr<void>> pooled_keepalive;
    auto release_async_scratch = [&]() {
        for (void* ptr : {d_temp,
                 d_comp_ptrs,
                 d_out_ptrs,
                 d_comp_sizes,
                 d_out_sizes,
                 d_statuses,
                 d_actual_sizes}) {
            if (ptr != nullptr) {
                cudaFreeAsync(ptr, stream);
            }
        }
    };

    try {
        if (use_native_pool) {
            int device_id = -1;
            check_cuda(cudaGetDevice(&device_id), "cudaGetDevice native nvcomp scratch");
            auto comp_ptrs = acquire_native_device_allocation(device_id, n * sizeof(void*));
            auto out_ptrs = acquire_native_device_allocation(device_id, n * sizeof(void*));
            auto comp_sizes = acquire_native_device_allocation(device_id, n * sizeof(std::size_t));
            auto out_sizes_buf =
                acquire_native_device_allocation(device_id, n * sizeof(std::size_t));
            auto statuses = acquire_native_device_allocation(device_id, n * sizeof(nvcompStatus_t));
            auto actual_sizes =
                acquire_native_device_allocation(device_id, n * sizeof(std::size_t));
            d_comp_ptrs = comp_ptrs.data;
            d_out_ptrs = out_ptrs.data;
            d_comp_sizes = comp_sizes.data;
            d_out_sizes = out_sizes_buf.data;
            d_statuses = statuses.data;
            d_actual_sizes = actual_sizes.data;
            pooled_keepalive.push_back(std::move(comp_ptrs.owner));
            pooled_keepalive.push_back(std::move(out_ptrs.owner));
            pooled_keepalive.push_back(std::move(comp_sizes.owner));
            pooled_keepalive.push_back(std::move(out_sizes_buf.owner));
            pooled_keepalive.push_back(std::move(statuses.owner));
            pooled_keepalive.push_back(std::move(actual_sizes.owner));
            if (temp_bytes > 0) {
                auto temp = acquire_native_device_allocation(device_id, temp_bytes);
                d_temp = temp.data;
                pooled_keepalive.push_back(std::move(temp.owner));
            }
        } else {
            check_cuda(cudaMallocAsync(&d_comp_ptrs, n * sizeof(void*), stream), "alloc comp_ptrs");
            check_cuda(cudaMallocAsync(&d_out_ptrs, n * sizeof(void*), stream), "alloc out_ptrs");
            check_cuda(
                cudaMallocAsync(&d_comp_sizes, n * sizeof(std::size_t), stream),
                "alloc comp_sizes");
            check_cuda(
                cudaMallocAsync(&d_out_sizes, n * sizeof(std::size_t), stream), "alloc out_sizes");
            check_cuda(
                cudaMallocAsync(&d_statuses, n * sizeof(nvcompStatus_t), stream), "alloc statuses");
            check_cuda(
                cudaMallocAsync(&d_actual_sizes, n * sizeof(std::size_t), stream),
                "alloc actual sizes");
            if (temp_bytes > 0) {
                check_cuda(cudaMallocAsync(&d_temp, temp_bytes, stream), "alloc temp");
            }
        }

        check_cuda(
            cudaMemcpyAsync(
                d_comp_ptrs, h_comp_ptrs.data(), n * sizeof(void*), cudaMemcpyHostToDevice, stream),
            "copy comp_ptrs");
        check_cuda(
            cudaMemcpyAsync(
                d_out_ptrs, h_out_ptrs.data(), n * sizeof(void*), cudaMemcpyHostToDevice, stream),
            "copy out_ptrs");
        check_cuda(
            cudaMemcpyAsync(
                d_comp_sizes,
                h_comp_sizes.data(),
                n * sizeof(std::size_t),
                cudaMemcpyHostToDevice,
                stream),
            "copy comp_sizes");
        check_cuda(
            cudaMemcpyAsync(
                d_out_sizes,
                h_out_sizes.data(),
                n * sizeof(std::size_t),
                cudaMemcpyHostToDevice,
                stream),
            "copy out_sizes");

        // Release the GIL around the nvcomp call so Python threads (prefetcher)
        // can progress while the async kernel launches + returns.
        {
            py::gil_scoped_release release;
            // Pass actual-size and status buffers together: nvCOMP 5.2
            // rejects a mixed null/non-null pair on the CUDA backend.
            check_nvcomp(
                gzip_wrapped ? nvcompBatchedGzipDecompressAsync(
                                   static_cast<const void* const*>(d_comp_ptrs),
                                   static_cast<const std::size_t*>(d_comp_sizes),
                                   static_cast<const std::size_t*>(d_out_sizes),
                                   static_cast<std::size_t*>(d_actual_sizes),
                                   n,
                                   d_temp,
                                   temp_bytes,
                                   static_cast<void* const*>(d_out_ptrs),
                                   gzip_opts,
                                   static_cast<nvcompStatus_t*>(d_statuses),
                                   stream)
                             : nvcompBatchedDeflateDecompressAsync(
                                   static_cast<const void* const*>(d_comp_ptrs),
                                   static_cast<const std::size_t*>(d_comp_sizes),
                                   static_cast<const std::size_t*>(d_out_sizes),
                                   static_cast<std::size_t*>(d_actual_sizes),
                                   n,
                                   d_temp,
                                   temp_bytes,
                                   static_cast<void* const*>(d_out_ptrs),
                                   opts,
                                   static_cast<nvcompStatus_t*>(d_statuses),
                                   stream),
                gzip_wrapped ? "batched gzip decompression" : "batched deflate decompression");
        }

        if (use_native_pool) {
            auto holder = std::make_unique<std::vector<std::shared_ptr<void>>>();
            py::capsule owner(holder.get(), [](void* p) {
                delete reinterpret_cast<std::vector<std::shared_ptr<void>>*>(p);
            });
            holder->swap(pooled_keepalive);
            holder.release();
            return owner;
        }
    } catch (...) {
        if (use_native_pool) {
            cudaStreamSynchronize(stream);
        } else {
            release_async_scratch();
        }
        throw;
    }

    // Stream-ordered frees — nvcomp consumes its inputs asynchronously and
    // these frees will be sequenced after the kernel completes.
    release_async_scratch();
    return py::none();
}

}  // namespace xdr_gpu

PYBIND11_MODULE(_nvcomp_batch_ext, m) {
    m.doc() = "Batched Gzip/DEFLATE decompression — device-pointer interface to nvcomp.";

    xdr_gpu::bind_io(m);
    xdr_gpu::bind_memory_manager(m);

    m.def(
        "batch_deflate_decompress",
        &xdr_gpu::batch_deflate_decompress,
        py::arg("d_concat_ptr"),
        py::arg("rel_offsets"),
        py::arg("lengths"),
        py::arg("d_out_ptr"),
        py::arg("out_offsets"),
        py::arg("out_sizes"),
        py::arg("stream_ptr") = 0,
        "Batched DEFLATE decompress across N tiles packed in one device buffer.\n"
        "Caller must have already stripped the RFC-1952 gzip wrapper from each\n"
        "tile (advance rel_offsets past the header, shorten lengths by header+8).");
    m.def(
        "batch_deflate_decompress_pooled",
        &xdr_gpu::batch_deflate_decompress_pooled,
        py::arg("d_concat_ptr"),
        py::arg("rel_offsets"),
        py::arg("lengths"),
        py::arg("d_out_ptr"),
        py::arg("out_offsets"),
        py::arg("out_sizes"),
        py::arg("stream_ptr") = 0,
        "Batched DEFLATE decompress using native pooled scratch buffers.\n"
        "Returns an owner capsule that must stay alive until stream work completes.");
    m.def(
        "batch_gzip_decompress",
        &xdr_gpu::batch_gzip_decompress,
        py::arg("d_concat_ptr"),
        py::arg("rel_offsets"),
        py::arg("lengths"),
        py::arg("d_out_ptr"),
        py::arg("out_offsets"),
        py::arg("out_sizes"),
        py::arg("stream_ptr") = 0,
        py::arg("use_native_pool") = false,
        "Batched Gzip decompress including RFC-1952 headers and trailers.\n"
        "Pooled calls return an owner capsule retained until stream completion.");
}
