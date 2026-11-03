# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Cuda front implementation."""

from __future__ import annotations

_FRONT_SOURCE = r"""
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

struct __align__(8) FrontBandParameters {
    long long source_start_sample;
    int channel_index;
    int padding;
    double fringe_reference_hz;
    double lo_offset_hz;
    double wall_lo_cycles;
};

__device__ __forceinline__ double reduce_cycles(double value) {
    return value - floor(value + 0.5);
}

__device__ __forceinline__ float decode_2bit_level(unsigned char code) {
    return code == 0 ? -3.3359f : (code == 1 ? -1.0f : (code == 2 ?
        1.0f : 3.3359f));
}

template <bool COMPLEX_INPUT, bool PACKED_INPUT>
__device__ __forceinline__ float2 load_sample(
    const void* source,
    long long index,
    long long packed_frame,
    long long packed_within_frame,
    int samples_per_frame,
    int payload_bytes,
    int channel_count,
    int channel_index
) {
    if (PACKED_INPUT) {
        const long long payload_start = packed_frame * payload_bytes;
        if (COMPLEX_INPUT) {
            const long long first_component =
                (packed_within_frame * channel_count + channel_index) * 2;
            const long long second_component = first_component + 1;
            const unsigned char packed_i =
                static_cast<const unsigned char*>(source)[
                payload_start + first_component / 4
            ];
            const unsigned char packed_q =
                static_cast<const unsigned char*>(source)[
                payload_start + second_component / 4
            ];
            const unsigned char code_i =
                (packed_i >> ((first_component & 3) * 2)) & 3;
            const unsigned char code_q =
                (packed_q >> ((second_component & 3) * 2)) & 3;
            return make_float2(
                decode_2bit_level(code_i),
                decode_2bit_level(code_q)
            );
        }
        const long long component =
            packed_within_frame * channel_count + channel_index;
        const unsigned char packed =
            static_cast<const unsigned char*>(source)[
            payload_start + component / 4
        ];
        const unsigned char code = (packed >> ((component & 3) * 2)) & 3;
        return make_float2(decode_2bit_level(code), 0.0f);
    }
    if (COMPLEX_INPUT) {
        const float2 value = static_cast<const float2*>(source)[index];
        return value;
    }
    const float value = static_cast<const float*>(source)[index];
    return make_float2(value, 0.0f);
}

template <bool COMPLEX_INPUT, bool PACKED_INPUT,
          bool CHUNKED = false, typename Weight = double>
__device__ __forceinline__ void front_impl(
    const void* source,
    const unsigned char* source_validity,
    int validity_stride,
    int authoritative_validity,
    long long source_size,
    int samples_per_frame,
    int payload_bytes,
    int channel_count,
    int channel_index,
    long long source_start_sample,
    const BlockParameters* parameters,
    int fft_length,
    double inverse_sample_rate,
    double fringe_reference_hz,
    double lo_offset_hz,
    double wall_lo_cycles,
    float2* rotated,
    Weight* weights
) {
    const int block = blockIdx.x;
    // Match the existing 16-iteration recurrence reseed interval exactly.
    const int sample_begin = CHUNKED ? (int)blockIdx.z * 4096 : 0;
    const int sample_limit = sample_begin + 4096;
    const int sample_end = CHUNKED && sample_limit < fft_length
        ? sample_limit : fft_length;
    const long long weight_index = CHUNKED
        ? (long long)block * gridDim.z + blockIdx.z : block;
    const BlockParameters parameter = parameters[block];
    if (parameter.valid_stop <= parameter.valid_start) {
        for (int i = sample_begin + threadIdx.x;
             i < sample_end; i += blockDim.x) {
            rotated[(long long)block * fft_length + i] =
                make_float2(0.0f, 0.0f);
        }
        if (threadIdx.x == 0) {
            weights[weight_index] = 0;
        }
        return;
    }
    unsigned int valid_count = 0;
    int sample = sample_begin + threadIdx.x;
    const double stride = (double)blockDim.x;
    const double phase_constant =
        -fringe_reference_hz * parameter.delay_zero + wall_lo_cycles
        + lo_offset_hz * parameter.block_start;
    const double phase_linear = -fringe_reference_hz * parameter.delay_step
        + lo_offset_hz * inverse_sample_rate;
    const double phase_quadratic =
        -fringe_reference_hz * parameter.delay_quadratic;
    double cycles_value = phase_constant + phase_linear * sample
        + phase_quadratic * sample * sample;
    double delta_cycles_value = phase_linear * stride + phase_quadratic
        * (2.0 * sample * stride + stride * stride);
    const double second_delta_cycles =
        2.0 * phase_quadratic * stride * stride;
    cycles_value = reduce_cycles(cycles_value);
    delta_cycles_value = reduce_cycles(delta_cycles_value);
    const double reduced_second_delta = reduce_cycles(second_delta_cycles);
    float sine;
    float cosine;
    float delta_sine;
    float delta_cosine;
    float second_sine;
    float second_cosine;
    __sincosf((float)(6.283185307179586476925286766559 * cycles_value),
        &sine, &cosine);
    __sincosf(
        (float)(6.283185307179586476925286766559 * delta_cycles_value),
        &delta_sine,
        &delta_cosine
    );
    __sincosf(
        (float)(6.283185307179586476925286766559 * reduced_second_delta),
        &second_sine,
        &second_cosine
    );
    int recurrence_steps = 0;
    const long long source_base = parameter.output_index
        - parameter.coarse_sample - source_start_sample;
    long long packed_frame = 0;
    long long packed_within_frame = 0;
    bool packed_position_initialized = false;
    for (; sample < sample_end; sample += blockDim.x) {
        const long long source_index = source_base + sample
            + (sample >= parameter.source_cutover ?
                parameter.source_offset : 0);
        const bool in_range = source_index >= 0 && source_index < source_size;
        if (PACKED_INPUT && parameter.source_offset != 0 &&
            source_index >= 0) {
            packed_frame = source_index / samples_per_frame;
            packed_within_frame =
                source_index - packed_frame * samples_per_frame;
            packed_position_initialized = true;
        } else if (PACKED_INPUT && !packed_position_initialized &&
            source_index >= 0) {
            packed_frame = source_index / samples_per_frame;
            packed_within_frame = source_index
                - packed_frame * samples_per_frame;
            packed_position_initialized = true;
        }
        bool valid = false;
        if (in_range && sample >= parameter.valid_start
                && sample < parameter.valid_stop) {
            valid = authoritative_validity != 0 || (PACKED_INPUT
                ? source_validity[packed_frame] != 0
                : source_validity[source_index / validity_stride] != 0);
        }
        float2 value = make_float2(0.0f, 0.0f);
        if (valid) {
            value = load_sample<COMPLEX_INPUT, PACKED_INPUT>(
                source,
                source_index,
                packed_frame,
                packed_within_frame,
                samples_per_frame,
                payload_bytes,
                channel_count,
                channel_index
            );
            ++valid_count;
        }

        rotated[(long long)block * fft_length + sample] = make_float2(
            value.x * cosine - value.y * sine,
            value.x * sine + value.y * cosine
        );
        const float next_cosine = cosine * delta_cosine - sine * delta_sine;
        sine = sine * delta_cosine + cosine * delta_sine;
        cosine = next_cosine;
        const float next_delta_cosine =
            delta_cosine * second_cosine - delta_sine * second_sine;
        delta_sine =
            delta_sine * second_cosine + delta_cosine * second_sine;
        delta_cosine = next_delta_cosine;
        ++recurrence_steps;
        if ((recurrence_steps & 15) == 0) {
            // Normalizing magnitude does not correct accumulated phase error.
            // Restart both recurrences from the binary64 phase polynomial.
            const double next_sample = (double)sample + stride;
            const double next_cycles = reduce_cycles(
                phase_constant + phase_linear * next_sample
                + phase_quadratic * next_sample * next_sample
            );
            const double next_delta = reduce_cycles(
                phase_linear * stride + phase_quadratic
                * (2.0 * next_sample * stride + stride * stride)
            );
            __sincosf(
                (float)(6.283185307179586476925286766559 * next_cycles),
                &sine, &cosine
            );
            __sincosf(
                (float)(6.283185307179586476925286766559 * next_delta),
                &delta_sine, &delta_cosine
            );
        }
        if (PACKED_INPUT && packed_position_initialized &&
            parameter.source_offset == 0) {
            packed_within_frame += blockDim.x;
            while (packed_within_frame >= samples_per_frame) {
                packed_within_frame -= samples_per_frame;
                ++packed_frame;
            }
        }
    }

    __shared__ unsigned int counts[256];
    counts[threadIdx.x] = valid_count;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) {
            counts[threadIdx.x] += counts[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        if (CHUNKED) {
            weights[weight_index] = counts[0];
        } else {
            weights[weight_index] = (double)counts[0] / (double)fft_length;
        }
    }
}

extern "C" __global__ void front_real(
    const float* source,
    const unsigned char* source_validity,
    int validity_stride,
    int authoritative_validity,
    long long source_size,
    int samples_per_frame,
    int payload_bytes,
    int channel_count,
    int channel_index,
    long long source_start_sample,
    const BlockParameters* parameters,
    int fft_length,
    double inverse_sample_rate,
    double fringe_reference_hz,
    double lo_offset_hz,
    double wall_lo_cycles,
    float2* rotated,
    double* weights
) {
    front_impl<false, false>(
        source, source_validity, validity_stride, authoritative_validity,
        source_size,
        samples_per_frame, payload_bytes, channel_count, channel_index,
        source_start_sample,
        parameters,
        fft_length, inverse_sample_rate, fringe_reference_hz, lo_offset_hz,
        wall_lo_cycles, rotated, weights
    );
}

extern "C" __global__ void front_complex(
    const float2* source,
    const unsigned char* source_validity,
    int validity_stride,
    int authoritative_validity,
    long long source_size,
    int samples_per_frame,
    int payload_bytes,
    int channel_count,
    int channel_index,
    long long source_start_sample,
    const BlockParameters* parameters,
    int fft_length,
    double inverse_sample_rate,
    double fringe_reference_hz,
    double lo_offset_hz,
    double wall_lo_cycles,
    float2* rotated,
    double* weights
) {
    front_impl<true, false>(
        source, source_validity, validity_stride, authoritative_validity,
        source_size,
        samples_per_frame, payload_bytes, channel_count, channel_index,
        source_start_sample,
        parameters,
        fft_length, inverse_sample_rate, fringe_reference_hz, lo_offset_hz,
        wall_lo_cycles, rotated, weights
    );
}

extern "C" __global__ void front_packed_real(
    const unsigned char* source,
    const unsigned char* source_validity,
    int validity_stride,
    int authoritative_validity,
    long long source_size,
    int samples_per_frame,
    int payload_bytes,
    int channel_count,
    int channel_index,
    long long source_start_sample,
    const BlockParameters* parameters,
    int fft_length,
    double inverse_sample_rate,
    double fringe_reference_hz,
    double lo_offset_hz,
    double wall_lo_cycles,
    float2* rotated,
    double* weights
) {
    front_impl<false, true>(
        source, source_validity, validity_stride, authoritative_validity,
        source_size,
        samples_per_frame, payload_bytes, channel_count, channel_index,
        source_start_sample,
        parameters,
        fft_length, inverse_sample_rate, fringe_reference_hz, lo_offset_hz,
        wall_lo_cycles, rotated, weights
    );
}

extern "C" __global__ void front_packed_real_bands(
    const unsigned char* source,
    const unsigned char* source_validity,
    int validity_stride,
    long long source_size,
    int samples_per_frame,
    int payload_bytes,
    int channel_count,
    const FrontBandParameters* band_parameters,
    const BlockParameters* parameters,
    int block_count,
    int fft_length,
    double inverse_sample_rate,
    float2* rotated,
    double* weights
) {
    const int band = (int)blockIdx.y;
    const FrontBandParameters metadata = band_parameters[band];
    const long long block_elements = (long long)block_count * fft_length;
    front_impl<false, true>(
        source, source_validity, validity_stride, 0, source_size,
        samples_per_frame, payload_bytes, channel_count,
        metadata.channel_index,
        metadata.source_start_sample,
        parameters + (long long)band * block_count,
        fft_length, inverse_sample_rate, metadata.fringe_reference_hz,
        metadata.lo_offset_hz, metadata.wall_lo_cycles,
        rotated + (long long)band * block_elements,
        weights + (long long)band * block_count
    );
}

extern "C" __global__ void front_packed_complex(
    const unsigned char* source,
    const unsigned char* source_validity,
    int validity_stride,
    int authoritative_validity,
    long long source_size,
    int samples_per_frame,
    int payload_bytes,
    int channel_count,
    int channel_index,
    long long source_start_sample,
    const BlockParameters* parameters,
    int fft_length,
    double inverse_sample_rate,
    double fringe_reference_hz,
    double lo_offset_hz,
    double wall_lo_cycles,
    float2* rotated,
    double* weights
) {
    front_impl<true, true>(
        source, source_validity, validity_stride, authoritative_validity,
        source_size,
        samples_per_frame, payload_bytes, channel_count, channel_index,
        source_start_sample,
        parameters,
        fft_length, inverse_sample_rate, fringe_reference_hz, lo_offset_hz,
        wall_lo_cycles, rotated, weights
    );
}

extern "C" __global__ void front_chunked_real(
    const float* source,
    const unsigned char* source_validity,
    int validity_stride,
    int authoritative_validity,
    long long source_size,
    int samples_per_frame,
    int payload_bytes,
    int channel_count,
    int channel_index,
    long long source_start_sample,
    const BlockParameters* parameters,
    int fft_length,
    double inverse_sample_rate,
    double fringe_reference_hz,
    double lo_offset_hz,
    double wall_lo_cycles,
    float2* rotated,
    unsigned int* weights
) {
    front_impl<false, false, true>(
        source, source_validity, validity_stride, authoritative_validity,
        source_size,
        samples_per_frame, payload_bytes, channel_count, channel_index,
        source_start_sample,
        parameters,
        fft_length, inverse_sample_rate, fringe_reference_hz, lo_offset_hz,
        wall_lo_cycles, rotated, weights
    );
}

extern "C" __global__ void front_chunked_packed_real(
    const unsigned char* source,
    const unsigned char* source_validity,
    int validity_stride,
    int authoritative_validity,
    long long source_size,
    int samples_per_frame,
    int payload_bytes,
    int channel_count,
    int channel_index,
    long long source_start_sample,
    const BlockParameters* parameters,
    int fft_length,
    double inverse_sample_rate,
    double fringe_reference_hz,
    double lo_offset_hz,
    double wall_lo_cycles,
    float2* rotated,
    unsigned int* weights
) {
    front_impl<false, true, true>(
        source, source_validity, validity_stride, authoritative_validity,
        source_size,
        samples_per_frame, payload_bytes, channel_count, channel_index,
        source_start_sample,
        parameters,
        fft_length, inverse_sample_rate, fringe_reference_hz, lo_offset_hz,
        wall_lo_cycles, rotated, weights
    );
}

extern "C" __global__ void front_chunked_packed_real_bands(
    const unsigned char* source,
    const unsigned char* source_validity,
    int validity_stride,
    long long source_size,
    int samples_per_frame,
    int payload_bytes,
    int channel_count,
    const FrontBandParameters* band_parameters,
    const BlockParameters* parameters,
    int block_count,
    int fft_length,
    double inverse_sample_rate,
    float2* rotated,
    unsigned int* weights
) {
    const int band = (int)blockIdx.y;
    const FrontBandParameters metadata = band_parameters[band];
    const long long block_elements = (long long)block_count * fft_length;
    front_impl<false, true, true>(
        source, source_validity, validity_stride, 0, source_size,
        samples_per_frame, payload_bytes, channel_count,
        metadata.channel_index,
        metadata.source_start_sample,
        parameters + (long long)band * block_count,
        fft_length, inverse_sample_rate, metadata.fringe_reference_hz,
        metadata.lo_offset_hz, metadata.wall_lo_cycles,
        rotated + (long long)band * block_elements,
        weights + (long long)band * block_count * gridDim.z
    );
}

extern "C" __global__ void finish_front_weights(
    const unsigned int* counts, int chunks, int fft_length, double* weights
) {
    const int row = blockIdx.x;
    unsigned int total = 0;
    for (int chunk = threadIdx.x; chunk < chunks; chunk += blockDim.x) {
        total += counts[(long long)row * chunks + chunk];
    }
    for (int offset = 16; offset; offset >>= 1) {
        total += __shfl_down_sync(0xffffffffu, total, offset);
    }
    __shared__ unsigned int partial[2];
    if ((threadIdx.x & 31) == 0) partial[threadIdx.x >> 5] = total;
    __syncthreads();
    if (threadIdx.x == 0) {
        weights[row] = (double)(partial[0] + partial[1]) / (double)fft_length;
    }
}
"""
