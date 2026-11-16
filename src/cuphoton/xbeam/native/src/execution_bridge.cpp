// SPDX-License-Identifier: Apache-2.0
#include "xbeam/execution.hpp"
#include "xbeam/radio.hpp"
#include "xbeam/series.hpp"
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
NativeSeries* create_native_series(const std::string& configuration) {
#ifdef XBEAM_BIND_CUDA
    return new NativeSeries(configuration);
#else
    (void)configuration;
    throw RadioError(
        "native CUDA execution was not built; enable CUPHOTON_XBEAM_NATIVE_CUDA=1");
#endif
}
void destroy_native_series(NativeSeries* series) {
#ifdef XBEAM_BIND_CUDA
    delete series;
#else
    (void)series;
#endif
}
void run_native_series(
    NativeSeries* series,
    std::int64_t first,
    int windows,
    void* spectrum,
    std::size_t spectrum_bytes,
    void* lag,
    std::size_t lag_bytes,
    std::int64_t* statistics,
    std::size_t statistics_count,
    std::int64_t* coverage,
    std::size_t coverage_count) {
#ifdef XBEAM_BIND_CUDA
    series->run(
        first,
        windows,
        spectrum,
        spectrum_bytes,
        lag,
        lag_bytes,
        statistics,
        statistics_count,
        coverage,
        coverage_count);
#else
    (void)series;
    (void)first;
    (void)windows;
    (void)spectrum;
    (void)spectrum_bytes;
    (void)lag;
    (void)lag_bytes;
    (void)statistics;
    (void)statistics_count;
    (void)coverage;
    (void)coverage_count;
    throw RadioError("native CUDA execution was not built");
#endif
}
}  // namespace xbeam
