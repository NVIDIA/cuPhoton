# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Cuda channelize implementation."""

from __future__ import annotations

_RESIDUAL_SOURCE = r"""
#include <cuComplex.h>

struct __align__(8) BlockParameters {
    long long output_index;
    long long coarse_sample;
    double block_start;
    double delay_zero;
    double delay_step;
    double delay_quadratic;
    double residual;
    long long source_offset;
    long long source_cutover;
    long long valid_start;
    long long valid_stop;
};

extern "C" __global__ void apply_residual_double(
    cuDoubleComplex* spectrum,
    const BlockParameters* parameters,
    const double* frequencies,
    long long element_count,
    int channel_count
) {
    const long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= element_count) {
        return;
    }
    const int channel = index % channel_count;
    const long long block = index / channel_count;
    double cycles = parameters[block].residual * frequencies[channel];
    cycles -= floor(cycles + 0.5);
    double sine;
    double cosine;
    sincos(6.283185307179586476925286766559 * cycles, &sine, &cosine);
    const cuDoubleComplex value = spectrum[index];
    spectrum[index] = make_cuDoubleComplex(
        cuCreal(value) * cosine - cuCimag(value) * sine,
        cuCreal(value) * sine + cuCimag(value) * cosine
    );
}

extern "C" __global__ void apply_residual_float(
    float2* spectrum,
    const BlockParameters* parameters,
    const double* frequencies,
    long long element_count,
    int channel_count
) {
    const long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= element_count) {
        return;
    }
    const int channel = index % channel_count;
    const long long block = index / channel_count;
    double cycles = parameters[block].residual * frequencies[channel];
    cycles -= floor(cycles + 0.5);
    float sine;
    float cosine;
    __sincosf(
        (float)(6.283185307179586476925286766559 * cycles),
        &sine,
        &cosine
    );
    const float2 value = spectrum[index];
    spectrum[index] = make_float2(
        value.x * cosine - value.y * sine,
        value.x * sine + value.y * cosine
    );
}
"""


_CHANNELIZE_RESIDUAL_SOURCE = r"""
struct __align__(8) BlockParameters {
    long long output_index;
    long long coarse_sample;
    double block_start;
    double delay_zero;
    double delay_step;
    double delay_quadratic;
    double residual;
    long long source_offset;
    long long source_cutover;
    long long valid_start;
    long long valid_stop;
};

enum ChannelMapping {
    IDENTITY = 0,
    REAL_LSB = 1,
    COMPLEX_LSB = 2,
    COMPLEX_DSB = 3
};

__device__ __forceinline__ float2 complex_lsb_value(
    const float2* row, int count, int channel) {
    if (channel == 0) {
        return row[0];
    }
    const float2 value = row[count - channel];
    return make_float2(value.x, -value.y);
}

template <int MAPPING>
__device__ __forceinline__ void channelize_residual_impl(
    const float2* transformed,
    const BlockParameters* parameters,
    int fft_length,
    int channel_count,
    int baseband_bin_offset,
    double channel_spacing_hz,
    float2* output
) {
    const int block = blockIdx.x;
    int channel = threadIdx.x;
    const double residual = parameters[block].residual;
    const double frequency_hz = (double)(baseband_bin_offset + channel)
        * channel_spacing_hz;
    double cycles = residual * frequency_hz;
    cycles -= floor(cycles + 0.5);
    float sine;
    float cosine;
    // The recurrence carries this seed across many channel strides.  The
    // approximate intrinsic's seed error grows beyond the release tolerance
    // for long spectra, so retain correctly-rounded sincosf here.
    sincosf(
        (float)(6.283185307179586476925286766559 * cycles),
        &sine,
        &cosine
    );
    __shared__ float delta_sine;
    __shared__ float delta_cosine;
    if (threadIdx.x == 0) {
        double delta_cycles = residual * channel_spacing_hz * blockDim.x;
        delta_cycles -= floor(delta_cycles + 0.5);
        sincosf(
            (float)(6.283185307179586476925286766559 * delta_cycles),
            &delta_sine,
            &delta_cosine
        );
    }
    __syncthreads();

    int recurrence_steps = 0;
    for (; channel < channel_count; channel += blockDim.x) {
        int source_channel = channel;
        if (MAPPING == REAL_LSB) {
            source_channel = fft_length - channel_count + channel;
        } else if (MAPPING == COMPLEX_DSB) {
            source_channel = ((fft_length + 1) / 2 + channel) % fft_length;
        }

        const float2* row = transformed + (long long)block * fft_length;
        const float2 value = MAPPING == COMPLEX_LSB
            ? complex_lsb_value(row, fft_length, channel)
            : row[source_channel];
        output[(long long)block * channel_count + channel] = make_float2(
            value.x * cosine - value.y * sine,
            value.x * sine + value.y * cosine
        );
        const float next_cosine =
            cosine * delta_cosine - sine * delta_sine;
        sine = sine * delta_cosine + cosine * delta_sine;
        cosine = next_cosine;
        ++recurrence_steps;
        if (
            (recurrence_steps & 31) == 0
            && channel + blockDim.x < channel_count
        ) {
            const float inverse = rsqrtf(cosine * cosine + sine * sine);
            cosine *= inverse;
            sine *= inverse;
        }
    }
}

extern "C" __global__ void channelize_residual_identity(
    const float2* transformed,
    const BlockParameters* parameters,
    int fft_length,
    int channel_count,
    int baseband_bin_offset,
    double channel_spacing_hz,
    float2* output
) {
    channelize_residual_impl<IDENTITY>(
        transformed, parameters, fft_length, channel_count,
        baseband_bin_offset, channel_spacing_hz, output
    );
}

extern "C" __global__ void channelize_residual_real_lsb(
    const float2* transformed,
    const BlockParameters* parameters,
    int fft_length,
    int channel_count,
    int baseband_bin_offset,
    double channel_spacing_hz,
    float2* output
) {
    channelize_residual_impl<REAL_LSB>(
        transformed, parameters, fft_length, channel_count,
        baseband_bin_offset, channel_spacing_hz, output
    );
}

extern "C" __global__ void channelize_residual_complex_lsb(
    const float2* transformed,
    const BlockParameters* parameters,
    int fft_length,
    int channel_count,
    int baseband_bin_offset,
    double channel_spacing_hz,
    float2* output
) {
    channelize_residual_impl<COMPLEX_LSB>(
        transformed, parameters, fft_length, channel_count,
        baseband_bin_offset, channel_spacing_hz, output
    );
}

extern "C" __global__ void channelize_residual_complex_dsb(
    const float2* transformed,
    const BlockParameters* parameters,
    int fft_length,
    int channel_count,
    int baseband_bin_offset,
    double channel_spacing_hz,
    float2* output
) {
    channelize_residual_impl<COMPLEX_DSB>(
        transformed, parameters, fft_length, channel_count,
        baseband_bin_offset, channel_spacing_hz, output
    );
}
"""


_CLEAR_REJECTED_FFT_SOURCE = r"""
extern "C" __global__ void clear_rejected_fft_rows(
    float2* spectrum, double* weight, double* reported_weight,
    const long long* rows, long long row_stride, int channels,
    bool has_reported_weight
) {
    const long long row = rows[blockIdx.x];
    for (int channel = threadIdx.x; channel < channels;
        channel += blockDim.x) {
        spectrum[row * row_stride + channel] = make_float2(0.0f, 0.0f);
    }
    if (threadIdx.x == 0) {
        weight[row] = 0.0;
        if (has_reported_weight) reported_weight[row] = 0.0;
    }
}
"""
