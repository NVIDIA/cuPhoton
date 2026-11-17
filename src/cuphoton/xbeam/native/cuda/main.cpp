// SPDX-License-Identifier: Apache-2.0
#include "xbeam/execution.hpp"
#include <iostream>
int main(int argc, char** argv) {
    try {
        std::vector<std::string> arguments;
        for (int i = 0; i < argc; ++i) {
            arguments.emplace_back(argv[i]);
        }
        std::cout << xbeam::correlate_radio(arguments) << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
}
