// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace cuphoton::xray {

std::int32_t solve_roots(
    const double* descending,
    double* roots,
    std::int32_t* stats,
    char* errors,
    std::int32_t degree,
    std::int32_t batch,
    std::int32_t max_sweeps,
    std::int32_t workers) noexcept;

std::int32_t refine_selected(
    const double* descending,
    const double* seeds,
    const double* separation,
    double* output,
    double* history,
    std::int32_t* status_steps,
    std::int32_t degree,
    std::int32_t count);

}  // namespace cuphoton::xray
