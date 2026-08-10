/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <pybind11/pybind11.h>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace xdr_gpu {

inline void check_cuda(cudaError_t err, const char* ctx) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA error in ") + ctx + ": " + cudaGetErrorString(err));
    }
}

struct NativeDeviceAllocation {
    std::shared_ptr<void> owner;
    std::uint8_t* data = nullptr;
    std::size_t size = 0;
};

NativeDeviceAllocation acquire_native_device_allocation(int device_id, std::size_t nbytes);
py::dict native_pinned_pool_stats_dict(int device_id);
void clear_native_pinned_pool_impl(int device_id);
py::dict native_device_pool_stats_dict(int device_id);
void clear_native_device_pool_impl(int device_id);
void bind_memory_manager(py::module_& m);
void bind_io(py::module_& m);

}  // namespace xdr_gpu
