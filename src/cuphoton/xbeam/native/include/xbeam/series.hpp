// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
namespace xbeam {
// HDF5 metadata is validated by the adapter; native code owns raw delivery and DSP.
class NativeSeries {
public:
    explicit NativeSeries(const std::string& configuration);
    ~NativeSeries();
    NativeSeries(const NativeSeries&) = delete;
    NativeSeries& operator=(const NativeSeries&) = delete;
    void run(
        std::int64_t first,
        int windows,
        void* spectrum,
        std::size_t spectrum_bytes,
        void* lag,
        std::size_t lag_bytes,
        std::int64_t* statistics,
        std::size_t statistics_count,
        std::int64_t* coverage,
        std::size_t coverage_count);

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
// Optional-build bridge: CPU-only extensions fail before opening the input.
NativeSeries* create_native_series(const std::string&);
void destroy_native_series(NativeSeries*);
void run_native_series(
    NativeSeries*,
    std::int64_t first,
    int windows,
    void* spectrum,
    std::size_t spectrum_bytes,
    void* lag,
    std::size_t lag_bytes,
    std::int64_t* statistics,
    std::size_t statistics_count,
    std::int64_t* coverage,
    std::size_t coverage_count);
// Shared cuFile driver owner, also used by recording delivery.
void ensure_cufile_driver();
// Uses the same generated XMAC kernels as radio execution.
void sum_series_spectra(
    bool wide,
    int rows,
    int channels,
    const void* a,
    const void* b,
    const double* weights,
    double scale,
    void* output,
    double* exposure,
    std::uintptr_t stream);
void pack_series_input(
    const void* raw,
    const unsigned char* mask,
    int dtype,
    bool wide,
    int rows,
    int samples,
    int fft_samples,
    std::int64_t hop,
    void* output,
    unsigned char* validity,
    void* mask_spectrum,
    int* error,
    std::uintptr_t stream);
void series_statistics(
    const unsigned char* a,
    const unsigned char* b,
    int rows,
    int samples,
    std::int64_t* statistics,
    std::uintptr_t stream);
void add_series_spectrum(
    bool wide,
    int channels,
    const void* input,
    void* output,
    std::uintptr_t stream);
void gather_series_lag(
    bool wide,
    int pairs,
    int fft_samples,
    int lag_first,
    int lag_count,
    const void* inverse,
    void* output,
    const void* mask_inverse,
    std::int64_t* coverage,
    std::uintptr_t stream);
}  // namespace xbeam
