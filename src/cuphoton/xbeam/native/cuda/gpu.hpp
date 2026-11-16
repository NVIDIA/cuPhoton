// SPDX-License-Identifier: Apache-2.0
// Small RAII wrappers over the CUDA runtime, shared by the native execution code.
#pragma once
#include "xbeam/radio.hpp"
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <vector>
namespace xbeam::gpu {

inline void check(cudaError_t status) {
    if (status != cudaSuccess) {
        throw RadioError(cudaGetErrorString(status));
    }
}

// Grow-only device buffer. `alignment` > 1 over-allocates so `p` is aligned
// (NVFS needs 64-KiB aligned device pages); the owning pointer is retained.
struct Memory {
    void* p = nullptr;
    std::size_t capacity = 0;
    explicit Memory(std::size_t alignment = 1)
        : alignment_(alignment) {}
    Memory(const Memory&) = delete;
    Memory& operator=(const Memory&) = delete;
    ~Memory() {
        if (allocation_) {
            cudaFree(allocation_);
        }
    }
    void reserve(std::size_t bytes) {
        bytes = (bytes + alignment_ - 1) / alignment_ * alignment_;
        if (bytes <= capacity) {
            return;
        }
        if (allocation_) {
            check(cudaFree(allocation_));
            allocation_ = p = nullptr;
            capacity = 0;
        }
        check(cudaMalloc(&allocation_, bytes + alignment_ - 1));
        p = reinterpret_cast<void*>(
            (reinterpret_cast<std::uintptr_t>(allocation_) + alignment_ - 1)
            / alignment_ * alignment_);
        capacity = bytes;
    }
    template <class T>
    void upload(const std::vector<T>& data) {
        reserve(data.size() * sizeof(T));
        check(cudaMemcpy(
            p, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::uintptr_t address() const {
        return reinterpret_cast<std::uintptr_t>(p);
    }

private:
    std::size_t alignment_;
    void* allocation_ = nullptr;
};

// Grow-only page-locked host buffer.
struct Pinned {
    void* p = nullptr;
    std::size_t capacity = 0;
    Pinned() = default;
    Pinned(const Pinned&) = delete;
    Pinned& operator=(const Pinned&) = delete;
    ~Pinned() {
        if (p) {
            cudaFreeHost(p);
        }
    }
    void reserve(std::size_t bytes) {
        if (bytes <= capacity) {
            return;
        }
        if (p) {
            check(cudaFreeHost(p));
            p = nullptr;
            capacity = 0;
        }
        check(cudaHostAlloc(&p, bytes, cudaHostAllocDefault));
        capacity = bytes;
    }
    template <class T>
    T* as() const {
        return static_cast<T*>(p);
    }
};

struct Stream {
    cudaStream_t value = nullptr;
    Stream() {
        check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking));
    }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    ~Stream() {
        if (value) {
            cudaStreamSynchronize(value);
            cudaStreamDestroy(value);
        }
    }
};

struct Event {
    cudaEvent_t value = nullptr;
    Event() {
        check(cudaEventCreateWithFlags(&value, cudaEventDisableTiming));
    }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    ~Event() {
        if (value) {
            cudaEventDestroy(value);
        }
    }
};

}  // namespace xbeam::gpu
