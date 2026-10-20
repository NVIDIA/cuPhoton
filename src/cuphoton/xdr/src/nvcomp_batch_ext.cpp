/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Batched Gzip/DEFLATE decompression — pybind11 helper that bypasses the
// per-tile Python `nvcomp.as_array` loop.
//
// Takes device pointers + offset/length arrays and calls
// the corresponding nvCOMP batched API directly. Decode completion is checked
// with the GIL released so a Python prefetch thread can progress meanwhile.

#include "nvcomp_batch_ext.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cuda_runtime.h>
#include <nvcomp/deflate.h>
#include <nvcomp/gzip.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
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

// Own every metadata transfer until decode completion. Downstream kernels may
// use the device status/size arrays, so callers retain this object through their
// final stream event, not merely until check() returns.
struct DecodeCompletion {
    struct Resources {
        std::vector<std::shared_ptr<void>> pooled_keepalive;
        std::vector<void*> async_scratch;
        NativePinnedAllocation host;
    };
    std::unique_ptr<Resources> resources = std::make_unique<Resources>();
    cudaStream_t stream = nullptr;
    cudaEvent_t event = nullptr;
    bool submitted = false;
    bool event_recorded = false;
    std::size_t count = 0;
    const void** comp_ptrs = nullptr;
    void** out_ptrs = nullptr;
    std::size_t* comp_sizes = nullptr;
    std::size_t* expected_sizes = nullptr;
    std::size_t* actual_sizes = nullptr;
    nvcompStatus_t* statuses = nullptr;
    void* d_statuses = nullptr;
    void* d_actual_sizes = nullptr;
    void* d_expected_sizes = nullptr;

    DecodeCompletion(std::size_t n, cudaStream_t decode_stream)
        : stream(decode_stream),
          count(n) {
        if (n == 0) {
            return;
        }
        int device_id = -1;
        check_cuda(
            cudaGetDevice(&device_id), "cudaGetDevice decode completion");
        resources->host = acquire_native_pinned_allocation(
            device_id,
            n
                * (2 * sizeof(void*) + 3 * sizeof(std::size_t)
                    + sizeof(nvcompStatus_t)));
        auto* cursor = resources->host.data;
        comp_ptrs = reinterpret_cast<const void**>(cursor);
        cursor += n * sizeof(void*);
        out_ptrs = reinterpret_cast<void**>(cursor);
        cursor += n * sizeof(void*);
        comp_sizes = reinterpret_cast<std::size_t*>(cursor);
        cursor += n * sizeof(std::size_t);
        expected_sizes = reinterpret_cast<std::size_t*>(cursor);
        cursor += n * sizeof(std::size_t);
        actual_sizes = reinterpret_cast<std::size_t*>(cursor);
        cursor += n * sizeof(std::size_t);
        statuses = reinterpret_cast<nvcompStatus_t*>(cursor);
        check_cuda(
            cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
            "create decode completion event");
    }

    ~DecodeCompletion() {
        // Also covers submission/copy failures before an event was recorded.
        // The host destinations must not return to their pool while DMA runs.
        cudaError_t result = cudaSuccess;
        if (event_recorded) {
            result = cudaEventSynchronize(event);
        } else if (submitted) {
            result = cudaStreamSynchronize(stream);
        }
        if (result != cudaSuccess) {
            // A fatal CUDA failure gives no safe reuse boundary. Quarantine
            // these allocations for the remaining process lifetime.
            (void)resources.release();
        } else {
            for (void* ptr : resources->async_scratch) {
                cudaFree(ptr);
            }
        }
        if (event != nullptr) {
            cudaEventDestroy(event);
        }
    }

    void check() const {
        if (event_recorded) {
            check_cuda(
                cudaEventSynchronize(event), "wait for decode completion");
        }
        for (std::size_t i = 0; i < count; ++i) {
            if (statuses[i] != nvcompSuccess) {
                throw std::runtime_error(
                    "nvcomp decompression failed for tile " + std::to_string(i)
                    + ": status="
                    + std::to_string(static_cast<int>(statuses[i])));
            }
            if (actual_sizes[i] != expected_sizes[i]) {
                throw std::runtime_error(
                    "nvcomp decoded size differs for tile " + std::to_string(i)
                    + ": expected " + std::to_string(expected_sizes[i])
                    + ", got " + std::to_string(actual_sizes[i]));
            }
        }
    }
};

py::object batch_decompress_impl(
    std::uintptr_t d_concat_ptr,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>
        rel_offsets,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>
        lengths,
    std::uintptr_t d_out_ptr,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>
        out_offsets,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>
        out_sizes,
    std::uintptr_t stream_ptr,
    bool use_native_pool,
    bool gzip_wrapped,
    const std::string& decompression_backend,
    bool deferred = false);

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
//   decompression_backend : string
//       Validated as "auto" or "cuda". Raw DEFLATE always uses CUDA.
//
// Waits for the decode stream and checks each tile's status and actual size.
// Raises on nvcomp / CUDA failure or a decoded-size mismatch. Returns None.
void batch_deflate_decompress(
    std::uintptr_t d_concat_ptr,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>
        rel_offsets,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>
        lengths,
    std::uintptr_t d_out_ptr,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>
        out_offsets,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>
        out_sizes,
    std::uintptr_t stream_ptr,
    const std::string& decompression_backend) {
    batch_decompress_impl(
        d_concat_ptr,
        rel_offsets,
        lengths,
        d_out_ptr,
        out_offsets,
        out_sizes,
        stream_ptr,
        false,
        false,
        decompression_backend);
}

py::object batch_deflate_decompress_pooled(
    std::uintptr_t d_concat_ptr,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>
        rel_offsets,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>
        lengths,
    std::uintptr_t d_out_ptr,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>
        out_offsets,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>
        out_sizes,
    std::uintptr_t stream_ptr,
    const std::string& decompression_backend) {
    return batch_decompress_impl(
        d_concat_ptr,
        rel_offsets,
        lengths,
        d_out_ptr,
        out_offsets,
        out_sizes,
        stream_ptr,
        true,
        false,
        decompression_backend);
}

// Gzip uses the same pointer/size arrays, with offsets and lengths covering
// complete RFC-1952 streams. "auto" permits compatible hardware with CUDA
// fallback; "cuda" selects CUDA. A pooled owner must survive stream completion.
py::object batch_gzip_decompress(
    std::uintptr_t d_concat_ptr,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>
        rel_offsets,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>
        lengths,
    std::uintptr_t d_out_ptr,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>
        out_offsets,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>
        out_sizes,
    std::uintptr_t stream_ptr,
    bool use_native_pool,
    const std::string& decompression_backend) {
    return batch_decompress_impl(
        d_concat_ptr,
        rel_offsets,
        lengths,
        d_out_ptr,
        out_offsets,
        out_sizes,
        stream_ptr,
        use_native_pool,
        true,
        decompression_backend);
}

py::object batch_decompress_impl(
    std::uintptr_t d_concat_ptr,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>
        rel_offsets,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>
        lengths,
    std::uintptr_t d_out_ptr,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>
        out_offsets,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>
        out_sizes,
    std::uintptr_t stream_ptr,
    bool use_native_pool,
    bool gzip_wrapped,
    const std::string& decompression_backend,
    bool deferred) {
    nvcompDecompressBackend_t backend;
    if (decompression_backend == "auto") {
        backend = NVCOMP_DECOMPRESS_BACKEND_DEFAULT;
    } else if (decompression_backend == "cuda") {
        backend = NVCOMP_DECOMPRESS_BACKEND_CUDA;
    } else {
        throw std::invalid_argument(
            "decompression_backend must be 'auto' or 'cuda'");
    }
    const std::size_t n = static_cast<std::size_t>(rel_offsets.size());
    if (lengths.size() != static_cast<py::ssize_t>(n)
        || out_offsets.size() != static_cast<py::ssize_t>(n)
        || out_sizes.size() != static_cast<py::ssize_t>(n)) {
        throw std::invalid_argument(
            "all per-tile arrays must have the same length");
    }
    if (n == 0) {
        return deferred
            ? py::cast(std::make_shared<DecodeCompletion>(0, nullptr))
            : py::none();
    }

    auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    auto completion =
        deferred ? std::make_shared<DecodeCompletion>(n, stream) : nullptr;

    // Build host-side pointer/size arrays.
    std::vector<const void*> h_comp_ptrs(n);
    std::vector<void*> h_out_ptrs(n);
    std::vector<std::size_t> h_comp_sizes(n);
    std::vector<std::size_t> h_out_sizes(n);
    std::vector<nvcompStatus_t> h_statuses(n);
    std::vector<std::size_t> h_actual_sizes(n);

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
        h_comp_ptrs[i] = reinterpret_cast<const void*>(
            d_concat_ptr + static_cast<std::size_t>(ro(i)));
        if (!gzip_wrapped
            && reinterpret_cast<std::uintptr_t>(h_comp_ptrs[i]) % 4 != 0) {
            throw std::invalid_argument(
                "raw DEFLATE inputs must be 4-byte aligned");
        }
        h_out_ptrs[i] = reinterpret_cast<void*>(
            d_out_ptr + static_cast<std::size_t>(oo(i)));
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
    // Raw DEFLATE retains its CUDA path in either mode.
    opts.backend = NVCOMP_DECOMPRESS_BACKEND_CUDA;
    auto gzip_opts = nvcompBatchedGzipDecompressDefaultOpts;
    // Non-pooled scratch uses cudaMallocAsync, whose allocations are not
    // hardware-decompression capable. Select CUDA directly to avoid a failed
    // hardware launch followed by fallback on every call. Pooled calls retain
    // nvCOMP's automatic selection when requested.
    gzip_opts.backend =
        use_native_pool ? backend : NVCOMP_DECOMPRESS_BACKEND_CUDA;
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

    std::vector<std::shared_ptr<void>> synchronous_keepalive;
    auto& pooled_keepalive = deferred ? completion->resources->pooled_keepalive
                                      : synchronous_keepalive;
    pooled_keepalive.reserve(7);
    if (deferred) {
        completion->resources->async_scratch.reserve(7);
    }
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
    auto allocate_async_scratch =
        [&](void** ptr, std::size_t bytes, const char* context) {
            if (deferred) {
                completion->submitted = true;
            }
            check_cuda(cudaMallocAsync(ptr, bytes, stream), context);
            if (deferred) {
                completion->resources->async_scratch.push_back(*ptr);
            }
        };

    try {
        if (use_native_pool) {
            int device_id = -1;
            check_cuda(
                cudaGetDevice(&device_id),
                "cudaGetDevice native nvcomp scratch");
            auto comp_ptrs =
                acquire_native_device_allocation(device_id, n * sizeof(void*));
            auto out_ptrs =
                acquire_native_device_allocation(device_id, n * sizeof(void*));
            auto comp_sizes = acquire_native_device_allocation(
                device_id, n * sizeof(std::size_t));
            auto out_sizes_buf = acquire_native_device_allocation(
                device_id, n * sizeof(std::size_t));
            auto statuses = acquire_native_device_allocation(
                device_id, n * sizeof(nvcompStatus_t));
            auto actual_sizes = acquire_native_device_allocation(
                device_id, n * sizeof(std::size_t));
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
                auto temp =
                    acquire_native_device_allocation(device_id, temp_bytes);
                d_temp = temp.data;
                pooled_keepalive.push_back(std::move(temp.owner));
            }
        } else {
            allocate_async_scratch(
                &d_comp_ptrs, n * sizeof(void*), "alloc comp_ptrs");
            allocate_async_scratch(
                &d_out_ptrs, n * sizeof(void*), "alloc out_ptrs");
            allocate_async_scratch(
                &d_comp_sizes, n * sizeof(std::size_t), "alloc comp_sizes");
            allocate_async_scratch(
                &d_out_sizes, n * sizeof(std::size_t), "alloc out_sizes");
            allocate_async_scratch(
                &d_statuses, n * sizeof(nvcompStatus_t), "alloc statuses");
            allocate_async_scratch(
                &d_actual_sizes, n * sizeof(std::size_t), "alloc actual sizes");
            if (temp_bytes > 0) {
                allocate_async_scratch(&d_temp, temp_bytes, "alloc temp");
            }
        }

        if (deferred) {
            std::memcpy(
                completion->comp_ptrs, h_comp_ptrs.data(), n * sizeof(void*));
            std::memcpy(
                completion->out_ptrs, h_out_ptrs.data(), n * sizeof(void*));
            std::memcpy(
                completion->comp_sizes,
                h_comp_sizes.data(),
                n * sizeof(std::size_t));
            std::memcpy(
                completion->expected_sizes,
                h_out_sizes.data(),
                n * sizeof(std::size_t));
            completion->d_statuses = d_statuses;
            completion->d_actual_sizes = d_actual_sizes;
            completion->d_expected_sizes = d_out_sizes;
            completion->submitted = true;
        }

        check_cuda(
            cudaMemcpyAsync(
                d_comp_ptrs,
                deferred ? completion->comp_ptrs : h_comp_ptrs.data(),
                n * sizeof(void*),
                cudaMemcpyHostToDevice,
                stream),
            "copy comp_ptrs");
        check_cuda(
            cudaMemcpyAsync(
                d_out_ptrs,
                deferred ? completion->out_ptrs : h_out_ptrs.data(),
                n * sizeof(void*),
                cudaMemcpyHostToDevice,
                stream),
            "copy out_ptrs");
        check_cuda(
            cudaMemcpyAsync(
                d_comp_sizes,
                deferred ? completion->comp_sizes : h_comp_sizes.data(),
                n * sizeof(std::size_t),
                cudaMemcpyHostToDevice,
                stream),
            "copy comp_sizes");
        check_cuda(
            cudaMemcpyAsync(
                d_out_sizes,
                deferred ? completion->expected_sizes : h_out_sizes.data(),
                n * sizeof(std::size_t),
                cudaMemcpyHostToDevice,
                stream),
            "copy out_sizes");

        // The API return reports submission, not per-tile success. Synchronous
        // callers check before accepting pixels; deferred callers check at
        // their batch boundary and guard downstream kernels with device status.
        // nvCOMP still requires valid input: these completion checks are not
        // validation of compressed payloads.
        {
            py::gil_scoped_release release;
            // Pass actual-size and status buffers together: nvCOMP 5.2
            // rejects a mixed null/non-null pair on the CUDA backend.
            check_nvcomp(
                gzip_wrapped
                    ? nvcompBatchedGzipDecompressAsync(
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
                gzip_wrapped ? "batched gzip decompression"
                             : "batched deflate decompression");
            check_cuda(
                cudaMemcpyAsync(
                    deferred ? completion->statuses : h_statuses.data(),
                    d_statuses,
                    n * sizeof(nvcompStatus_t),
                    cudaMemcpyDeviceToHost,
                    stream),
                "copy decode statuses");
            check_cuda(
                cudaMemcpyAsync(
                    deferred ? completion->actual_sizes : h_actual_sizes.data(),
                    d_actual_sizes,
                    n * sizeof(std::size_t),
                    cudaMemcpyDeviceToHost,
                    stream),
                "copy actual decoded sizes");
            if (deferred) {
                check_cuda(
                    cudaEventRecord(completion->event, stream),
                    "record decode completion event");
                completion->event_recorded = true;
            } else {
                check_cuda(
                    cudaStreamSynchronize(stream),
                    "wait for decode completion");
                for (std::size_t i = 0; i < n; ++i) {
                    if (h_statuses[i] != nvcompSuccess) {
                        throw std::runtime_error(
                            "nvcomp decompression failed for tile "
                            + std::to_string(i) + ": status="
                            + std::to_string(static_cast<int>(h_statuses[i])));
                    }
                    if (h_actual_sizes[i] != h_out_sizes[i]) {
                        throw std::runtime_error(
                            "nvcomp decoded size differs for tile "
                            + std::to_string(i) + ": expected "
                            + std::to_string(h_out_sizes[i]) + ", got "
                            + std::to_string(h_actual_sizes[i]));
                    }
                }
            }
        }

        if (deferred) {
            return py::cast(completion);
        }
        if (use_native_pool) {
            auto holder =
                std::make_unique<std::vector<std::shared_ptr<void>>>();
            py::capsule owner(holder.get(), [](void* p) {
                delete reinterpret_cast<std::vector<std::shared_ptr<void>>*>(p);
            });
            holder->swap(pooled_keepalive);
            holder.release();
            return owner;
        }
    } catch (...) {
        if (deferred) {
            py::gil_scoped_release release;
            completion.reset();
            throw;
        }
        {
            // Keep host copy destinations and pooled owners alive until
            // outstanding stream work has finished, including copy failures.
            py::gil_scoped_release release;
            cudaStreamSynchronize(stream);
        }
        if (!use_native_pool) {
            release_async_scratch();
        }
        throw;
    }

    // Decode has completed; preserve stream ordering for scratch releases.
    release_async_scratch();
    return py::none();
}

}  // namespace xdr_gpu

PYBIND11_MODULE(_nvcomp_batch_ext, m) {
    m.doc() =
        "Batched Gzip/DEFLATE decompression — device-pointer interface to nvcomp.";

    m.attr("supports_decompression_backend") = true;

    xdr_gpu::bind_io(m);
    xdr_gpu::bind_memory_manager(m);

    py::class_<
        xdr_gpu::DecodeCompletion,
        std::shared_ptr<xdr_gpu::DecodeCompletion>>(m, "DecodeCompletion")
        .def(
            "check",
            &xdr_gpu::DecodeCompletion::check,
            py::call_guard<py::gil_scoped_release>(),
            "Wait for decode completion and validate per-tile status and size.")
        .def_property_readonly(
            "status_ptr",
            [](const xdr_gpu::DecodeCompletion& c) {
                return reinterpret_cast<std::uintptr_t>(c.d_statuses);
            })
        .def_property_readonly(
            "actual_sizes_ptr",
            [](const xdr_gpu::DecodeCompletion& c) {
                return reinterpret_cast<std::uintptr_t>(c.d_actual_sizes);
            })
        .def_property_readonly(
            "expected_sizes_ptr", [](const xdr_gpu::DecodeCompletion& c) {
                return reinterpret_cast<std::uintptr_t>(c.d_expected_sizes);
            });

    auto bind_deferred = [&](const char* name, bool gzip_wrapped) {
        m.def(
            name,
            [gzip_wrapped](
                std::uintptr_t d_concat_ptr,
                py::array_t<
                    std::int64_t,
                    py::array::c_style | py::array::forcecast> rel_offsets,
                py::array_t<
                    std::int64_t,
                    py::array::c_style | py::array::forcecast> lengths,
                std::uintptr_t d_out_ptr,
                py::array_t<
                    std::int64_t,
                    py::array::c_style | py::array::forcecast> out_offsets,
                py::array_t<
                    std::int64_t,
                    py::array::c_style | py::array::forcecast> out_sizes,
                std::uintptr_t stream_ptr,
                bool use_native_pool,
                const std::string& decompression_backend) {
                return xdr_gpu::batch_decompress_impl(
                    d_concat_ptr,
                    rel_offsets,
                    lengths,
                    d_out_ptr,
                    out_offsets,
                    out_sizes,
                    stream_ptr,
                    use_native_pool,
                    gzip_wrapped,
                    decompression_backend,
                    true);
            },
            py::arg("d_concat_ptr"),
            py::arg("rel_offsets"),
            py::arg("lengths"),
            py::arg("d_out_ptr"),
            py::arg("out_offsets"),
            py::arg("out_sizes"),
            py::arg("stream_ptr") = 0,
            py::arg("use_native_pool") = true,
            py::arg("decompression_backend") = "auto",
            "Submit decode and return a completion owner without a host wait.\n"
            "Keep inputs, output and completion alive through downstream work.\n"
            "Guard downstream reads with device status and decoded sizes, and\n"
            "call completion.check() before publishing the batch result.");
    };
    bind_deferred("batch_gzip_decompress_deferred", true);
    bind_deferred("batch_deflate_decompress_deferred", false);

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
        py::arg("decompression_backend") = "auto",
        "Batched DEFLATE decompress across N tiles packed in one device buffer.\n"
        "Waits for decode completion and validates per-tile status and size.\n"
        "Caller must have already stripped the RFC-1952 gzip wrapper from each\n"
        "tile (advance rel_offsets past the header, shorten lengths by header+8).\n"
        "decompression_backend accepts auto/cuda; raw DEFLATE always uses CUDA.");
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
        py::arg("decompression_backend") = "auto",
        "Batched DEFLATE decompress using native pooled scratch buffers.\n"
        "Waits for decode completion and validates per-tile status and size.\n"
        "Returns an owner capsule for pooled scratch buffers.\n"
        "decompression_backend accepts auto/cuda; raw DEFLATE always uses CUDA.");
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
        py::arg("decompression_backend") = "auto",
        "Batched Gzip decompress including RFC-1952 headers and trailers.\n"
        "Waits for decode completion and validates per-tile status and size.\n"
        "Pooled calls return an owner capsule for scratch buffers.\n"
        "decompression_backend=auto permits compatible hardware with CUDA fallback;\n"
        "cuda selects CUDA explicitly.");
}
