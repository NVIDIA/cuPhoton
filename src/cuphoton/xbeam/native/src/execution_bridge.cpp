// SPDX-License-Identifier: Apache-2.0
#include "xbeam/execution.hpp"
#include "xbeam/radio.hpp"
namespace xbeam {
bool native_cuda_available() {
#ifdef XBEAM_BIND_CUDA
    return true;
#else
    return false;
#endif
}
std::string correlate_native(const std::vector<std::string>& arguments) {
#ifdef XBEAM_BIND_CUDA
    return correlate_radio(arguments);
#else
    (void)arguments;
    throw RadioError(
        "native CUDA execution was not built; enable "
        "CUPHOTON_XBEAM_NATIVE_CUDA=1 during the opt-in extension build");
#endif
}
}  // namespace xbeam
