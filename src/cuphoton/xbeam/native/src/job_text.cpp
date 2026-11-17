// SPDX-License-Identifier: Apache-2.0
#include "xbeam/radio.hpp"
#include <sstream>
namespace xbeam {
namespace {
std::string trim(const std::string& s) {
    const auto start = s.find_first_not_of(" \t\r\n\v\f");
    if (start == std::string::npos) {
        return {};
    }
    const auto end = s.find_last_not_of(" \t\r\n\v\f");
    return s.substr(start, end - start + 1);
}
}  // namespace
std::vector<TextEntry> parse_job_text(const std::string& text) {
    std::vector<TextEntry> entries;
    std::istringstream stream(text);
    std::string line;
    std::size_t number = 0;
    while (std::getline(stream, line)) {
        ++number;
        const auto stripped = trim(line);
        if (stripped.empty() || stripped[0] == '#' || stripped[0] == '@') {
            continue;
        }
        const auto colon = line.find(':');
        if (colon == std::string::npos) {
            throw RadioError(
                "job line " + std::to_string(number) + ": expected KEY: value");
        }
        auto key = trim(line.substr(0, colon));
        if (key.empty()) {
            throw RadioError(
                "job line " + std::to_string(number) + ": empty field name");
        }
        entries.push_back(
            {std::move(key), trim(line.substr(colon + 1)), number});
    }
    return entries;
}
}  // namespace xbeam
