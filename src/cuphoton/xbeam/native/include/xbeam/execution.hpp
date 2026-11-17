// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <string>
#include <vector>
namespace xbeam {
// The execution library owns all planning, IO, CUDA work, and output.
std::string correlate_radio(const std::vector<std::string>& arguments);
// Optional Cython build bridge; CPU-only builds fail explicitly.
std::string correlate_native(const std::vector<std::string>& arguments);
bool native_cuda_available();
}  // namespace xbeam
