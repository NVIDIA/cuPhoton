// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0
#include "radio_model.hpp"
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 3) {
        return 2;
    }
    try {
        std::cout << xbeam::open_index(argv[1], argv[2])->metadata_json();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
