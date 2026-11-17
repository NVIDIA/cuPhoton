// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0
#include "run_report.hpp"
#include <csignal>
#include <fstream>
#include <iterator>
#include <sys/resource.h>

int main() {
    namespace fs = std::filesystem;
    const fs::path success("report-success"), failure("report-failure");
    fs::create_directory(success);
    fs::create_directory(failure);
    const std::string contents = "{\"records\": 123}";
    xbeam::publish_run_report(success, contents);
    std::ifstream report(success / "native-run.json");
    const std::string actual{std::istreambuf_iterator<char>(report),
        std::istreambuf_iterator<char>()};
    if (actual != contents + '\n'
        || fs::exists(success / "native-run.json.tmp")) {
        return 1;
    }

    // Exercise a real buffered write/close failure without filling a disk.
    rlimit original{};
    if (getrlimit(RLIMIT_FSIZE, &original)) {
        return 2;
    }
    auto limited = original;
    limited.rlim_cur = 8;
    const auto old_signal = std::signal(SIGXFSZ, SIG_IGN);
    if (setrlimit(RLIMIT_FSIZE, &limited)) {
        return 3;
    }
    bool rejected = false;
    try {
        xbeam::publish_run_report(failure, contents);
    } catch (const xbeam::RadioError&) {
        rejected = true;
    }
    if (setrlimit(RLIMIT_FSIZE, &original)) {
        return 4;
    }
    std::signal(SIGXFSZ, old_signal);
    if (!rejected || fs::exists(failure / "native-run.json")
        || fs::exists(failure / "native-run.json.tmp")) {
        return 5;
    }
    return 0;
}
