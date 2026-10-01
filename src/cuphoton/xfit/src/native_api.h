// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

namespace cuphoton::xfit {

struct Settings {
    double f_tol, x_tol, g_tol;
    double initial_damping, damping_increase, damping_decrease;
    int max_evaluations;
};

// All device buffers are contiguous and owned by the caller through completion.
// x uses log widths and is updated in place. diagnostic_current indicates that
// the last valid Jacobian describes x (including a final rejected step).
struct Batch {
    void* x;
    const void* images;
    const void* weights;
    void* residuals;
    std::int8_t* status;
    std::int32_t* evaluations;
    bool* diagnostic_current;
    int count, height, width, planes;
    bool fp64;
};

struct Timings {
    double host_seconds;
    double synchronize_seconds;
    float gpu_milliseconds;
    int iterations;
};

class Workspace {
public:
    explicit Workspace(int device);
    ~Workspace();
    Workspace(const Workspace&) = delete;
    Workspace& operator=(const Workspace&) = delete;
    // producer_stream is borrowed only for recording a dependency event.
    // Successful calls and exceptions both drain all submitted work before
    // returning, so caller-owned buffers may then be released safely.
    Timings run(
        const Batch& batch,
        const Settings& settings,
        std::uintptr_t producer_stream);

private:
    struct Impl;
    Impl* impl_;
};

}  // namespace cuphoton::xfit
