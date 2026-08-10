/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nvcomp_batch_ext.h"

#include <pybind11/pybind11.h>

#include <cuda_runtime.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unistd.h>
#include <vector>

namespace xdr_gpu {

struct PinnedHostBuffer {
    std::uint8_t* data = nullptr;
    std::size_t size = 0;
    std::size_t capacity = 0;
    int device_id = -1;

    PinnedHostBuffer(int device, std::size_t nbytes)
        : size(nbytes),
          capacity(nbytes),
          device_id(device) {
        if (capacity == 0) {
            return;
        }
        void* ptr = nullptr;
        check_cuda(cudaHostAlloc(&ptr, capacity, cudaHostAllocDefault), "cudaHostAlloc native batch buffer");
        data = static_cast<std::uint8_t*>(ptr);
    }

    PinnedHostBuffer(const PinnedHostBuffer&) = delete;
    PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;

    ~PinnedHostBuffer() {
        if (data != nullptr) {
            cudaFreeHost(data);
            data = nullptr;
        }
    }
};

class PinnedHostBufferPool;
PinnedHostBufferPool& native_pinned_pool();

class PinnedHostBufferPool {
public:
    std::shared_ptr<PinnedHostBuffer> acquire(int device_id, std::size_t nbytes) {
        if (nbytes == 0) {
            auto* empty = new PinnedHostBuffer(device_id, 0);
            return std::shared_ptr<PinnedHostBuffer>(
                empty,
                [](PinnedHostBuffer* ptr) { delete ptr; });
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (auto buffer = take_idle_locked(device_id, nbytes)) {
                return make_shared_from_raw(std::move(buffer), nbytes);
            }
        }

        // cudaHostAlloc/cudaFreeHost are very expensive under concurrent use.
        // Serialize cold allocations while still allowing cached buffers to be
        // acquired through the fast path above.
        std::lock_guard<std::mutex> alloc_lock(cuda_host_alloc_mutex_);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (auto buffer = take_idle_locked(device_id, nbytes)) {
                return make_shared_from_raw(std::move(buffer), nbytes);
            }
        }

        auto buffer = std::make_unique<PinnedHostBuffer>(device_id, nbytes);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto& cache = device_cache_locked(device_id);
            ++cache.allocations;
            cache.checked_out_bytes += buffer->capacity;
        }
        return make_shared_from_raw(std::move(buffer), nbytes);
    }

    void release(std::unique_ptr<PinnedHostBuffer> buffer) {
        if (!buffer || buffer->capacity == 0) {
            return;
        }

        std::vector<std::unique_ptr<PinnedHostBuffer>> to_free;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto& cache = device_cache_locked(buffer->device_id);
            ++cache.releases;
            if (cache.checked_out_bytes >= buffer->capacity) {
                cache.checked_out_bytes -= buffer->capacity;
            } else {
                cache.checked_out_bytes = 0;
            }

            const std::size_t max_idle = max_idle_bytes_;
            if (max_idle == 0 || buffer->capacity > max_idle) {
                ++cache.frees;
                to_free.push_back(std::move(buffer));
            } else {
                while (cache.idle_bytes + buffer->capacity > max_idle && !cache.idle.empty()) {
                    auto it = cache.idle.begin();
                    cache.idle_bytes -= it->second->capacity;
                    ++cache.frees;
                    to_free.push_back(std::move(it->second));
                    cache.idle.erase(it);
                }
                if (cache.idle_bytes + buffer->capacity <= max_idle) {
                    buffer->size = 0;
                    cache.idle_bytes += buffer->capacity;
                    cache.idle.emplace(buffer->capacity, std::move(buffer));
                } else {
                    ++cache.frees;
                    to_free.push_back(std::move(buffer));
                }
            }
        }

        free_buffers(std::move(to_free));
    }

    py::dict stats(int device_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (device_id >= 0) {
            return stats_for_locked(device_id);
        }

        PoolStats total;
        for (const auto& item : caches_) {
            const auto& cache = item.second;
            total.allocations += cache.allocations;
            total.reuses += cache.reuses;
            total.releases += cache.releases;
            total.frees += cache.frees;
            total.idle_bytes += cache.idle_bytes;
            total.checked_out_bytes += cache.checked_out_bytes;
            total.idle_buffers += cache.idle.size();
        }
        return stats_dict(total, max_idle_bytes_);
    }

    void clear(int device_id) {
        std::vector<std::unique_ptr<PinnedHostBuffer>> to_free;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (device_id >= 0) {
                clear_locked(device_id, to_free);
            } else {
                for (auto& item : caches_) {
                    clear_locked(item.first, to_free);
                }
            }
        }
        free_buffers(std::move(to_free));
    }

private:
    struct DeviceCache {
        std::multimap<std::size_t, std::unique_ptr<PinnedHostBuffer>> idle;
        std::size_t idle_bytes = 0;
        std::size_t checked_out_bytes = 0;
        std::uint64_t allocations = 0;
        std::uint64_t reuses = 0;
        std::uint64_t releases = 0;
        std::uint64_t frees = 0;
    };

    struct PoolStats {
        std::size_t idle_buffers = 0;
        std::size_t idle_bytes = 0;
        std::size_t checked_out_bytes = 0;
        std::uint64_t allocations = 0;
        std::uint64_t reuses = 0;
        std::uint64_t releases = 0;
        std::uint64_t frees = 0;
    };

    DeviceCache& device_cache_locked(int device_id) {
        return caches_[device_id];
    }

    std::unique_ptr<PinnedHostBuffer> take_idle_locked(int device_id, std::size_t nbytes) {
        auto& cache = device_cache_locked(device_id);
        auto it = cache.idle.lower_bound(nbytes);
        if (it == cache.idle.end()) {
            return nullptr;
        }
        auto buffer = std::move(it->second);
        cache.idle_bytes -= buffer->capacity;
        cache.idle.erase(it);
        buffer->size = nbytes;
        cache.checked_out_bytes += buffer->capacity;
        ++cache.reuses;
        return buffer;
    }

    std::shared_ptr<PinnedHostBuffer> make_shared_from_raw(
        std::unique_ptr<PinnedHostBuffer> buffer,
        std::size_t logical_size) {
        buffer->size = logical_size;
        PinnedHostBuffer* raw = buffer.release();
        return std::shared_ptr<PinnedHostBuffer>(
            raw,
            [](PinnedHostBuffer* ptr) {
                native_pinned_pool().release(std::unique_ptr<PinnedHostBuffer>(ptr));
            });
    }

    py::dict stats_for_locked(int device_id) {
        PoolStats stats;
        auto it = caches_.find(device_id);
        if (it != caches_.end()) {
            const auto& cache = it->second;
            stats.allocations = cache.allocations;
            stats.reuses = cache.reuses;
            stats.releases = cache.releases;
            stats.frees = cache.frees;
            stats.idle_bytes = cache.idle_bytes;
            stats.checked_out_bytes = cache.checked_out_bytes;
            stats.idle_buffers = cache.idle.size();
        }
        return stats_dict(stats, max_idle_bytes_);
    }

    static py::dict stats_dict(const PoolStats& stats, std::size_t max_idle_bytes) {
        py::dict out;
        out["allocations"] = stats.allocations;
        out["reuses"] = stats.reuses;
        out["releases"] = stats.releases;
        out["frees"] = stats.frees;
        out["idle_buffers"] = stats.idle_buffers;
        out["idle_bytes"] = stats.idle_bytes;
        out["checked_out_bytes"] = stats.checked_out_bytes;
        out["max_idle_bytes"] = max_idle_bytes;
        return out;
    }

    void clear_locked(
        int device_id,
        std::vector<std::unique_ptr<PinnedHostBuffer>>& to_free) {
        auto it = caches_.find(device_id);
        if (it == caches_.end()) {
            return;
        }
        auto& cache = it->second;
        for (auto& entry : cache.idle) {
            ++cache.frees;
            to_free.push_back(std::move(entry.second));
        }
        cache.idle.clear();
        cache.idle_bytes = 0;
    }

    static void free_buffers(std::vector<std::unique_ptr<PinnedHostBuffer>> buffers) {
        if (buffers.empty()) {
            return;
        }
        std::lock_guard<std::mutex> alloc_lock(cuda_host_alloc_mutex_);
        buffers.clear();
    }

    static std::size_t default_max_idle_bytes() {
        constexpr std::size_t gib = static_cast<std::size_t>(1024) * 1024 * 1024;
        std::size_t default_cap = 8 * gib;
        const long pages = ::sysconf(_SC_PHYS_PAGES);
        const long page_size = ::sysconf(_SC_PAGE_SIZE);
        if (pages > 0 && page_size > 0) {
            const auto ram_bytes = static_cast<unsigned long long>(pages)
                                   * static_cast<unsigned long long>(page_size);
            const auto quarter = ram_bytes / 4;
            if (quarter < static_cast<unsigned long long>(default_cap)) {
                default_cap = static_cast<std::size_t>(quarter);
            }
        }
        return default_cap;
    }

    static std::size_t configured_max_idle_bytes() {
        const char* env = std::getenv("CUPHOTON_XDR_PINNED_POOL_MAX_BYTES");
        if (env == nullptr || *env == '\0') {
            return default_max_idle_bytes();
        }

        errno = 0;
        char* end = nullptr;
        const unsigned long long value = std::strtoull(env, &end, 10);
        if (errno != 0 || end == env || *end != '\0') {
            return default_max_idle_bytes();
        }
        if (value > static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max())) {
            return std::numeric_limits<std::size_t>::max();
        }
        return static_cast<std::size_t>(value);
    }

    static std::mutex cuda_host_alloc_mutex_;

    std::mutex mutex_;
    std::map<int, DeviceCache> caches_;
    const std::size_t max_idle_bytes_ = configured_max_idle_bytes();
};

std::mutex PinnedHostBufferPool::cuda_host_alloc_mutex_;

PinnedHostBufferPool& native_pinned_pool() {
    static auto* pool = new PinnedHostBufferPool();
    return *pool;
}

struct DeviceBuffer {
    std::uint8_t* data = nullptr;
    std::size_t size = 0;
    std::size_t capacity = 0;
    int device_id = -1;

    DeviceBuffer(int device, std::size_t nbytes)
        : size(nbytes),
          capacity(nbytes),
          device_id(device) {
        if (capacity == 0) {
            return;
        }
        if (device_id >= 0) {
            check_cuda(cudaSetDevice(device_id), "cudaSetDevice native device buffer");
        }
        void* ptr = nullptr;
        check_cuda(cudaMalloc(&ptr, capacity), "cudaMalloc native batch device buffer");
        data = static_cast<std::uint8_t*>(ptr);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        if (data != nullptr) {
            if (device_id >= 0) {
                cudaSetDevice(device_id);
            }
            cudaFree(data);
            data = nullptr;
        }
    }
};

class DeviceBufferPool;
DeviceBufferPool& native_device_pool();

class DeviceBufferPool {
public:
    std::shared_ptr<DeviceBuffer> acquire(int device_id, std::size_t nbytes) {
        if (nbytes == 0) {
            auto* empty = new DeviceBuffer(device_id, 0);
            return std::shared_ptr<DeviceBuffer>(
                empty,
                [](DeviceBuffer* ptr) { delete ptr; });
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (auto buffer = take_idle_locked(device_id, nbytes)) {
                return make_shared_from_raw(std::move(buffer), nbytes);
            }
        }

        std::lock_guard<std::mutex> alloc_lock(cuda_device_alloc_mutex_);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (auto buffer = take_idle_locked(device_id, nbytes)) {
                return make_shared_from_raw(std::move(buffer), nbytes);
            }
        }

        auto buffer = std::make_unique<DeviceBuffer>(device_id, nbytes);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto& cache = device_cache_locked(device_id);
            ++cache.allocations;
            cache.checked_out_bytes += buffer->capacity;
        }
        return make_shared_from_raw(std::move(buffer), nbytes);
    }

    void release(std::unique_ptr<DeviceBuffer> buffer) {
        if (!buffer || buffer->capacity == 0) {
            return;
        }

        std::vector<std::unique_ptr<DeviceBuffer>> to_free;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto& cache = device_cache_locked(buffer->device_id);
            ++cache.releases;
            if (cache.checked_out_bytes >= buffer->capacity) {
                cache.checked_out_bytes -= buffer->capacity;
            } else {
                cache.checked_out_bytes = 0;
            }

            const std::size_t max_idle = max_idle_bytes_;
            if (max_idle == 0 || buffer->capacity > max_idle) {
                ++cache.frees;
                to_free.push_back(std::move(buffer));
            } else {
                while (cache.idle_bytes + buffer->capacity > max_idle && !cache.idle.empty()) {
                    auto it = cache.idle.begin();
                    cache.idle_bytes -= it->second->capacity;
                    ++cache.frees;
                    to_free.push_back(std::move(it->second));
                    cache.idle.erase(it);
                }
                if (cache.idle_bytes + buffer->capacity <= max_idle) {
                    buffer->size = 0;
                    cache.idle_bytes += buffer->capacity;
                    cache.idle.emplace(buffer->capacity, std::move(buffer));
                } else {
                    ++cache.frees;
                    to_free.push_back(std::move(buffer));
                }
            }
        }

        free_buffers(std::move(to_free));
    }

    py::dict stats(int device_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (device_id >= 0) {
            return stats_for_locked(device_id);
        }

        PoolStats total;
        for (const auto& item : caches_) {
            const auto& cache = item.second;
            total.allocations += cache.allocations;
            total.reuses += cache.reuses;
            total.releases += cache.releases;
            total.frees += cache.frees;
            total.idle_bytes += cache.idle_bytes;
            total.checked_out_bytes += cache.checked_out_bytes;
            total.idle_buffers += cache.idle.size();
        }
        return stats_dict(total, max_idle_bytes_);
    }

    void clear(int device_id) {
        std::vector<std::unique_ptr<DeviceBuffer>> to_free;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (device_id >= 0) {
                clear_locked(device_id, to_free);
            } else {
                for (auto& item : caches_) {
                    clear_locked(item.first, to_free);
                }
            }
        }
        free_buffers(std::move(to_free));
    }

private:
    struct DeviceCache {
        std::multimap<std::size_t, std::unique_ptr<DeviceBuffer>> idle;
        std::size_t idle_bytes = 0;
        std::size_t checked_out_bytes = 0;
        std::uint64_t allocations = 0;
        std::uint64_t reuses = 0;
        std::uint64_t releases = 0;
        std::uint64_t frees = 0;
    };

    struct PoolStats {
        std::size_t idle_buffers = 0;
        std::size_t idle_bytes = 0;
        std::size_t checked_out_bytes = 0;
        std::uint64_t allocations = 0;
        std::uint64_t reuses = 0;
        std::uint64_t releases = 0;
        std::uint64_t frees = 0;
    };

    DeviceCache& device_cache_locked(int device_id) {
        return caches_[device_id];
    }

    std::unique_ptr<DeviceBuffer> take_idle_locked(int device_id, std::size_t nbytes) {
        auto& cache = device_cache_locked(device_id);
        auto it = cache.idle.lower_bound(nbytes);
        if (it == cache.idle.end()) {
            return nullptr;
        }
        auto buffer = std::move(it->second);
        cache.idle_bytes -= buffer->capacity;
        cache.idle.erase(it);
        buffer->size = nbytes;
        cache.checked_out_bytes += buffer->capacity;
        ++cache.reuses;
        return buffer;
    }

    std::shared_ptr<DeviceBuffer> make_shared_from_raw(
        std::unique_ptr<DeviceBuffer> buffer,
        std::size_t logical_size) {
        buffer->size = logical_size;
        DeviceBuffer* raw = buffer.release();
        return std::shared_ptr<DeviceBuffer>(
            raw,
            [](DeviceBuffer* ptr) {
                native_device_pool().release(std::unique_ptr<DeviceBuffer>(ptr));
            });
    }

    py::dict stats_for_locked(int device_id) {
        PoolStats stats;
        auto it = caches_.find(device_id);
        if (it != caches_.end()) {
            const auto& cache = it->second;
            stats.allocations = cache.allocations;
            stats.reuses = cache.reuses;
            stats.releases = cache.releases;
            stats.frees = cache.frees;
            stats.idle_bytes = cache.idle_bytes;
            stats.checked_out_bytes = cache.checked_out_bytes;
            stats.idle_buffers = cache.idle.size();
        }
        return stats_dict(stats, max_idle_bytes_);
    }

    static py::dict stats_dict(const PoolStats& stats, std::size_t max_idle_bytes) {
        py::dict out;
        out["allocations"] = stats.allocations;
        out["reuses"] = stats.reuses;
        out["releases"] = stats.releases;
        out["frees"] = stats.frees;
        out["idle_buffers"] = stats.idle_buffers;
        out["idle_bytes"] = stats.idle_bytes;
        out["checked_out_bytes"] = stats.checked_out_bytes;
        out["max_idle_bytes"] = max_idle_bytes;
        return out;
    }

    void clear_locked(
        int device_id,
        std::vector<std::unique_ptr<DeviceBuffer>>& to_free) {
        auto it = caches_.find(device_id);
        if (it == caches_.end()) {
            return;
        }
        auto& cache = it->second;
        for (auto& entry : cache.idle) {
            ++cache.frees;
            to_free.push_back(std::move(entry.second));
        }
        cache.idle.clear();
        cache.idle_bytes = 0;
    }

    static void free_buffers(std::vector<std::unique_ptr<DeviceBuffer>> buffers) {
        if (buffers.empty()) {
            return;
        }
        std::lock_guard<std::mutex> alloc_lock(cuda_device_alloc_mutex_);
        buffers.clear();
    }

    static std::size_t default_max_idle_bytes() {
        constexpr std::size_t gib = static_cast<std::size_t>(1024) * 1024 * 1024;
        return 8 * gib;
    }

    static std::size_t configured_max_idle_bytes() {
        const char* env = std::getenv("CUPHOTON_XDR_DEVICE_POOL_MAX_BYTES");
        if (env == nullptr || *env == '\0') {
            return default_max_idle_bytes();
        }

        errno = 0;
        char* end = nullptr;
        const unsigned long long value = std::strtoull(env, &end, 10);
        if (errno != 0 || end == env || *end != '\0') {
            return default_max_idle_bytes();
        }
        if (value > static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max())) {
            return std::numeric_limits<std::size_t>::max();
        }
        return static_cast<std::size_t>(value);
    }

    static std::mutex cuda_device_alloc_mutex_;

    std::mutex mutex_;
    std::map<int, DeviceCache> caches_;
    const std::size_t max_idle_bytes_ = configured_max_idle_bytes();
};

std::mutex DeviceBufferPool::cuda_device_alloc_mutex_;

DeviceBufferPool& native_device_pool() {
    static auto* pool = new DeviceBufferPool();
    return *pool;
}



NativeDeviceAllocation acquire_native_device_allocation(int device_id, std::size_t nbytes) {
    auto buffer = native_device_pool().acquire(device_id, nbytes);
    NativeDeviceAllocation allocation;
    allocation.data = buffer->data;
    allocation.size = buffer->size;
    allocation.owner = std::static_pointer_cast<void>(std::move(buffer));
    return allocation;
}

py::dict native_pinned_pool_stats_dict(int device_id) {
    return native_pinned_pool().stats(device_id);
}

void clear_native_pinned_pool_impl(int device_id) {
    native_pinned_pool().clear(device_id);
}

py::dict native_device_pool_stats_dict(int device_id) {
    return native_device_pool().stats(device_id);
}

void clear_native_device_pool_impl(int device_id) {
    native_device_pool().clear(device_id);
}

py::tuple acquire_native_device_buffer(std::size_t nbytes, int device_id) {
    if (device_id < 0) {
        check_cuda(cudaGetDevice(&device_id), "cudaGetDevice native device buffer acquire");
    }
    NativeDeviceAllocation allocation;
    {
        py::gil_scoped_release release;
        allocation = acquire_native_device_allocation(device_id, nbytes);
    }
    auto holder = new std::shared_ptr<void>(std::move(allocation.owner));
    py::capsule owner(holder, [](void* p) {
        delete reinterpret_cast<std::shared_ptr<void>*>(p);
    });
    return py::make_tuple(owner, reinterpret_cast<std::uintptr_t>(allocation.data), allocation.size);
}

void bind_memory_manager(py::module_& m) {
    m.def("native_pinned_pool_stats",
          [](int device_id) { return native_pinned_pool_stats_dict(device_id); },
          py::arg("device_id") = -1,
          "Return legacy process-wide pinned host buffer pool statistics. device_id=-1 sums all devices.");
    m.def("clear_native_pinned_pool",
          [](int device_id) { clear_native_pinned_pool_impl(device_id); },
          py::arg("device_id") = -1,
          "Free idle process-wide pinned host buffers. device_id=-1 clears all devices.");
    m.def("native_device_pool_stats",
          [](int device_id) { return native_device_pool_stats_dict(device_id); },
          py::arg("device_id") = -1,
          "Return process-wide native device buffer pool statistics. device_id=-1 sums all devices.");
    m.def("clear_native_device_pool",
          [](int device_id) { clear_native_device_pool_impl(device_id); },
          py::arg("device_id") = -1,
          "Free idle process-wide native device buffers. device_id=-1 clears all devices.");
    m.def("acquire_native_device_buffer", &acquire_native_device_buffer,
          py::arg("nbytes"),
          py::arg("device_id") = -1,
          "Acquire a uint8-sized native pooled device buffer as (owner, ptr, nbytes).");
}


}  // namespace xdr_gpu
