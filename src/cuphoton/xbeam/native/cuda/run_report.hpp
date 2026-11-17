// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "xbeam/radio.hpp"
#include <filesystem>
#include <fstream>
#include <string>
namespace xbeam {

// A completion marker becomes visible only after its entire contents close.
inline void publish_run_report(
    const std::filesystem::path& directory, const std::string& contents) {
    const auto target = directory / "native-run.json";
    const auto temporary = directory / "native-run.json.tmp";
    try {
        std::ofstream report(temporary, std::ios::binary | std::ios::out);
        if (!report) {
            throw RadioError("cannot create native completion report");
        }
        report << contents << '\n';
        report.close();
        if (!report) {
            throw RadioError("native completion report write failed");
        }
        std::error_code error;
        std::filesystem::rename(temporary, target, error);
        if (error) {
            throw RadioError(
                "cannot publish native completion report: " + error.message());
        }
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw;
    }
}
}  // namespace xbeam
