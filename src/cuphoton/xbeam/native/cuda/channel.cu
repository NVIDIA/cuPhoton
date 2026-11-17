// SPDX-License-Identifier: Apache-2.0
#include "channel.inc"
#include "xbeam/cuda.hpp"
namespace xbeam {
void launch_channel(
    const StationLayout& g,
    const void* transformed,
    const void* parameters,
    void* output,
    cudaStream_t stream) {
    const int offset = g.mapping == 0 ? 0
        : g.mapping == 3              ? -(g.fft_samples / 2)
                                      : -g.channels;
    const auto* input = static_cast<const float2*>(transformed);
    const auto* p = static_cast<const ::BlockParameters*>(parameters);
    auto* result = static_cast<float2*>(output);
    const double spacing = g.sample_rate / g.fft_samples;
    switch (g.mapping) {
    case 0:
        channelize_residual_identity<<<g.rows, 256, 0, stream>>>(
            input, p, g.fft_samples, g.channels, offset, spacing, result);
        break;
    case 1:
        channelize_residual_real_lsb<<<g.rows, 256, 0, stream>>>(
            input, p, g.fft_samples, g.channels, offset, spacing, result);
        break;
    case 2:
        channelize_residual_complex_lsb<<<g.rows, 256, 0, stream>>>(
            input, p, g.fft_samples, g.channels, offset, spacing, result);
        break;
    case 3:
        channelize_residual_complex_dsb<<<g.rows, 256, 0, stream>>>(
            input, p, g.fft_samples, g.channels, offset, spacing, result);
        break;
    }
}
}  // namespace xbeam
