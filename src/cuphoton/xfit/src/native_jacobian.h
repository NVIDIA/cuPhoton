// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime_api.h>

namespace cuphoton::xfit::detail {

// Outputs retain the input selection order. Consume finite_row before changing
// indices; packed outputs for surviving rows may need to be regenerated.
// The caller selects this FP64 path only on a compatible SM100 device and
// supplies distinct, aligned workspace buffers on its private CUDA stream.
cudaError_t launch_jacobian_pack_fp64(
    const double* x,
    const double* weights,
    const int* indices,
    const double* residuals,
    double* packed,
    double* transposed,
    double* compact,
    bool* finite_row,
    long long n,
    long long k,
    int planes,
    int height,
    int width,
    int device,
    cudaStream_t stream);

}  // namespace cuphoton::xfit::detail
