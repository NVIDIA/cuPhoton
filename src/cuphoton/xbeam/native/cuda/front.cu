// SPDX-License-Identifier: Apache-2.0
#include "front.inc"
#include "xbeam/cuda.hpp"
namespace xbeam {
void launch_front(
    const StationLayout& g,
    const void* source,
    const void* validity,
    const void* parameters,
    void* rotated,
    void* weights,
    void* counts,
    cudaStream_t stream) {
    const bool chunked = !g.complex_input && g.fft_samples == 262144
        && g.rows >= 8 && g.rows <= 1600;
    if (chunked) {
        front_chunked_packed_real<<<dim3(g.rows, 1, 64), 256, 0, stream>>>(
            static_cast<const unsigned char*>(source),
            static_cast<const unsigned char*>(validity),
            g.validity_stride,
            g.authoritative_validity,
            g.source_samples,
            g.samples_per_frame,
            g.payload_bytes,
            g.channel_count,
            g.channel_index,
            g.source_start,
            static_cast<const ::BlockParameters*>(parameters),
            g.fft_samples,
            1. / g.sample_rate,
            g.reference_hz,
            g.lo_hz,
            g.wall_lo_cycles,
            static_cast<float2*>(rotated),
            static_cast<unsigned int*>(counts));
        finish_front_weights<<<g.rows, 64, 0, stream>>>(
            static_cast<unsigned int*>(counts),
            64,
            g.fft_samples,
            static_cast<double*>(weights));
    } else if (g.complex_input) {
        front_packed_complex<<<g.rows, 256, 0, stream>>>(
            static_cast<const unsigned char*>(source),
            static_cast<const unsigned char*>(validity),
            g.validity_stride,
            g.authoritative_validity,
            g.source_samples,
            g.samples_per_frame,
            g.payload_bytes,
            g.channel_count,
            g.channel_index,
            g.source_start,
            static_cast<const ::BlockParameters*>(parameters),
            g.fft_samples,
            1. / g.sample_rate,
            g.reference_hz,
            g.lo_hz,
            g.wall_lo_cycles,
            static_cast<float2*>(rotated),
            static_cast<double*>(weights));
    } else {
        front_packed_real<<<g.rows, 256, 0, stream>>>(
            static_cast<const unsigned char*>(source),
            static_cast<const unsigned char*>(validity),
            g.validity_stride,
            g.authoritative_validity,
            g.source_samples,
            g.samples_per_frame,
            g.payload_bytes,
            g.channel_count,
            g.channel_index,
            g.source_start,
            static_cast<const ::BlockParameters*>(parameters),
            g.fft_samples,
            1. / g.sample_rate,
            g.reference_hz,
            g.lo_hz,
            g.wall_lo_cycles,
            static_cast<float2*>(rotated),
            static_cast<double*>(weights));
    }
}
}  // namespace xbeam
