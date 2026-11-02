# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Cuda visibility implementation."""

from __future__ import annotations

_VISIBILITY_SOURCE = r"""
#include <cuComplex.h>

template <typename OutputType>
__device__ __forceinline__ OutputType make_visibility_output(
    double real_value,
    double imag_value
);

template <>
__device__ __forceinline__ float2 make_visibility_output<float2>(
    double real_value,
    double imag_value
) {
    return make_float2((float)real_value, (float)imag_value);
}

template <>
__device__ __forceinline__ double2 make_visibility_output<double2>(
    double real_value,
    double imag_value
) {
    return make_double2(real_value, imag_value);
}

__device__ __forceinline__ void reduce_visibility_weights(
    const double* weight_a,
    const double* weight_b,
    bool multiply_weights,
    int block_count,
    double* weight_parts,
    double* weight_sum
) {
    double local_weight = 0.0;
    for (int block = threadIdx.x; block < block_count; block += blockDim.x) {
        const double first = weight_a[block];
        local_weight += multiply_weights ? first * weight_b[block] : first;
    }
    weight_parts[threadIdx.x] = local_weight;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) {
            weight_parts[threadIdx.x] += weight_parts[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        weight_sum[0] = weight_parts[0];
    }
}

template <
    typename ComplexType,
    typename OutputType,
    typename AccumulatorType,
    int CHANNELS_TO_AVERAGE
>
__device__ __forceinline__ void visibility_tiled_impl(
    const ComplexType* spectrum_a,
    const ComplexType* spectrum_b,
    const double* weight_a,
    const double* weight_b,
    bool multiply_weights,
    int block_count,
    int input_channel_count,
    double inverse_spectral_normalization,
    OutputType* output,
    double* weight_sum
) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int channel = (int)blockIdx.x * 32 + lane;
    AccumulatorType real_sum = (AccumulatorType)0;
    AccumulatorType imag_sum = (AccumulatorType)0;
    if (channel < input_channel_count) {
        for (int block = warp; block < block_count; block += 8) {
            const long long row = (long long)block * input_channel_count;
            const ComplexType a = spectrum_a[row + channel];
            const ComplexType b = spectrum_b[row + channel];
            real_sum += (AccumulatorType)a.x * (AccumulatorType)b.x
                + (AccumulatorType)a.y * (AccumulatorType)b.y;
            imag_sum += (AccumulatorType)a.y * (AccumulatorType)b.x
                - (AccumulatorType)a.x * (AccumulatorType)b.y;
        }
    }

    __shared__ AccumulatorType real_parts[8][32];
    __shared__ AccumulatorType imag_parts[8][32];
    real_parts[warp][lane] = real_sum;
    imag_parts[warp][lane] = imag_sum;
    __syncthreads();
    if (warp == 0) {
        real_sum = (AccumulatorType)0;
        imag_sum = (AccumulatorType)0;
        #pragma unroll
        for (int partial = 0; partial < 8; ++partial) {
            real_sum += real_parts[partial][lane];
            imag_sum += imag_parts[partial][lane];
        }
        if (CHANNELS_TO_AVERAGE == 32) {
            real_sum += __shfl_down_sync(0xffffffff, real_sum, 16);
            imag_sum += __shfl_down_sync(0xffffffff, imag_sum, 16);
            real_sum += __shfl_down_sync(0xffffffff, real_sum, 8);
            imag_sum += __shfl_down_sync(0xffffffff, imag_sum, 8);
            real_sum += __shfl_down_sync(0xffffffff, real_sum, 4);
            imag_sum += __shfl_down_sync(0xffffffff, imag_sum, 4);
        }
        if (CHANNELS_TO_AVERAGE >= 4) {
            real_sum += __shfl_down_sync(0xffffffff, real_sum, 1);
            imag_sum += __shfl_down_sync(0xffffffff, imag_sum, 1);
            real_sum += __shfl_down_sync(0xffffffff, real_sum, 2);
            imag_sum += __shfl_down_sync(0xffffffff, imag_sum, 2);
        }
        if (
            channel < input_channel_count
            && (
                CHANNELS_TO_AVERAGE == 1
                || (lane & (CHANNELS_TO_AVERAGE - 1)) == 0
            )
        ) {
            const int output_channel = channel / CHANNELS_TO_AVERAGE;
            const double scale = inverse_spectral_normalization
                / CHANNELS_TO_AVERAGE;
            output[output_channel] = make_visibility_output<OutputType>(
                real_sum * scale, imag_sum * scale
            );
        }
    }

    if (blockIdx.x == 0) {
        __shared__ double weight_parts[256];
        reduce_visibility_weights(
            weight_a, weight_b, multiply_weights, block_count,
            weight_parts, weight_sum
        );
    }
}

template <typename ComplexType, typename OutputType, typename AccumulatorType>
__device__ __forceinline__ void visibility_generic_impl(
    const ComplexType* spectrum_a,
    const ComplexType* spectrum_b,
    const double* weight_a,
    const double* weight_b,
    bool multiply_weights,
    int block_count,
    int input_channel_count,
    int channels_to_average,
    double inverse_spectral_normalization,
    OutputType* output,
    double* weight_sum
) {
    const int output_channel = (int)blockIdx.x * blockDim.x + threadIdx.x;
    const int output_channel_count =
        input_channel_count / channels_to_average;
    if (output_channel < output_channel_count) {
        const int first_channel = output_channel * channels_to_average;
        AccumulatorType real_sum = (AccumulatorType)0;
        AccumulatorType imag_sum = (AccumulatorType)0;
        for (int block = 0; block < block_count; ++block) {
            const long long row = (long long)block * input_channel_count;
            for (int offset = 0; offset < channels_to_average; ++offset) {
                const int channel = first_channel + offset;
                const ComplexType a = spectrum_a[row + channel];
                const ComplexType b = spectrum_b[row + channel];
                real_sum += (AccumulatorType)a.x * (AccumulatorType)b.x
                    + (AccumulatorType)a.y * (AccumulatorType)b.y;
                imag_sum += (AccumulatorType)a.y * (AccumulatorType)b.x
                    - (AccumulatorType)a.x * (AccumulatorType)b.y;
            }
        }
        const double scale = inverse_spectral_normalization
            / channels_to_average;
        output[output_channel] = make_visibility_output<OutputType>(
            real_sum * scale, imag_sum * scale
        );
    }
    if (blockIdx.x == 0) {
        __shared__ double weight_parts[256];
        reduce_visibility_weights(
            weight_a, weight_b, multiply_weights, block_count,
            weight_parts, weight_sum
        );
    }
}

extern "C" __global__ void visibility_product_float_avg1(
    const float2* spectrum_a,
    const float2* spectrum_b,
    const double* weight_a,
    const double* weight_b,
    bool multiply_weights,
    int block_count,
    int input_channel_count,
    double inverse_spectral_normalization,
    float2* output,
    double* weight_sum
) {
    visibility_tiled_impl<float2, float2, float, 1>(
        spectrum_a, spectrum_b, weight_a, weight_b, multiply_weights,
        block_count, input_channel_count,
        inverse_spectral_normalization, output, weight_sum
    );
}

extern "C" __global__ void visibility_product_float_avg4(
    const float2* spectrum_a,
    const float2* spectrum_b,
    const double* weight_a,
    const double* weight_b,
    bool multiply_weights,
    int block_count,
    int input_channel_count,
    double inverse_spectral_normalization,
    float2* output,
    double* weight_sum
) {
    visibility_tiled_impl<float2, float2, float, 4>(
        spectrum_a, spectrum_b, weight_a, weight_b, multiply_weights,
        block_count, input_channel_count,
        inverse_spectral_normalization, output, weight_sum
    );
}

extern "C" __global__ void visibility_product_float_avg32(
    const float2* spectrum_a,
    const float2* spectrum_b,
    const double* weight_a,
    const double* weight_b,
    bool multiply_weights,
    int block_count,
    int input_channel_count,
    double inverse_spectral_normalization,
    float2* output,
    double* weight_sum
) {
    visibility_tiled_impl<float2, float2, float, 32>(
        spectrum_a, spectrum_b, weight_a, weight_b, multiply_weights,
        block_count, input_channel_count,
        inverse_spectral_normalization, output, weight_sum
    );
}

extern "C" __global__ void visibility_product_float_generic(
    const float2* spectrum_a,
    const float2* spectrum_b,
    const double* weight_a,
    const double* weight_b,
    bool multiply_weights,
    int block_count,
    int input_channel_count,
    int channels_to_average,
    double inverse_spectral_normalization,
    float2* output,
    double* weight_sum
) {
    visibility_generic_impl<float2, float2, float>(
        spectrum_a, spectrum_b, weight_a, weight_b, multiply_weights,
        block_count, input_channel_count, channels_to_average,
        inverse_spectral_normalization, output, weight_sum
    );
}

extern "C" __global__ void visibility_product_double_avg1(
    const double2* spectrum_a,
    const double2* spectrum_b,
    const double* weight_a,
    const double* weight_b,
    bool multiply_weights,
    int block_count,
    int input_channel_count,
    double inverse_spectral_normalization,
    double2* output,
    double* weight_sum
) {
    visibility_tiled_impl<double2, double2, double, 1>(
        spectrum_a, spectrum_b, weight_a, weight_b, multiply_weights,
        block_count, input_channel_count,
        inverse_spectral_normalization, output, weight_sum
    );
}

extern "C" __global__ void visibility_product_double_avg4(
    const double2* spectrum_a,
    const double2* spectrum_b,
    const double* weight_a,
    const double* weight_b,
    bool multiply_weights,
    int block_count,
    int input_channel_count,
    double inverse_spectral_normalization,
    double2* output,
    double* weight_sum
) {
    visibility_tiled_impl<double2, double2, double, 4>(
        spectrum_a, spectrum_b, weight_a, weight_b, multiply_weights,
        block_count, input_channel_count,
        inverse_spectral_normalization, output, weight_sum
    );
}

extern "C" __global__ void visibility_product_double_avg32(
    const double2* spectrum_a,
    const double2* spectrum_b,
    const double* weight_a,
    const double* weight_b,
    bool multiply_weights,
    int block_count,
    int input_channel_count,
    double inverse_spectral_normalization,
    double2* output,
    double* weight_sum
) {
    visibility_tiled_impl<double2, double2, double, 32>(
        spectrum_a, spectrum_b, weight_a, weight_b, multiply_weights,
        block_count, input_channel_count,
        inverse_spectral_normalization, output, weight_sum
    );
}

extern "C" __global__ void visibility_product_double_generic(
    const double2* spectrum_a,
    const double2* spectrum_b,
    const double* weight_a,
    const double* weight_b,
    bool multiply_weights,
    int block_count,
    int input_channel_count,
    int channels_to_average,
    double inverse_spectral_normalization,
    double2* output,
    double* weight_sum
) {
    visibility_generic_impl<double2, double2, double>(
        spectrum_a, spectrum_b, weight_a, weight_b, multiply_weights,
        block_count, input_channel_count, channels_to_average,
        inverse_spectral_normalization, output, weight_sum
    );
}

#ifdef BATCHED_DOUBLE
using BatchedComplex = double2;
using BatchedScalar = double;
#define BATCHED_VISIBILITY_ENTRY(suffix) \
    batched_visibility_product_double_##suffix
#else
using BatchedComplex = float2;
using BatchedScalar = float;
#define BATCHED_VISIBILITY_ENTRY(suffix) \
    batched_visibility_product_float_##suffix
#endif

__device__ __forceinline__ void store_batched_visibility(
    BatchedComplex* output, long long index, double real, double imag
) {
    BatchedComplex value = make_visibility_output<BatchedComplex>(real, imag);
#ifdef BATCHED_ACCUMULATE
    value.x += output[index].x;
    value.y += output[index].y;
#endif
    output[index] = value;
}

template <int CHANNELS_TO_AVERAGE>
__device__ __forceinline__ void batched_visibility_tiled_impl(
    const BatchedComplex* spectrum_a,
    const BatchedComplex* spectrum_b,
    long long spectrum_a_stride,
    long long spectrum_b_stride,
    const double* weight_a,
    const double* weight_b,
    bool multiply_weights,
    const int* group_starts,
    const int* group_lengths,
    const long long* output_offsets,
    const long long* weight_offsets,
    int input_channel_count,
    double inverse_spectral_normalization,
    BatchedComplex* output,
    double* weight_sum
) {
    const int group = (int)blockIdx.y;
    const int tile = (int)blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int channel = tile * 32 + lane;
    const int group_start = group_starts[group];
    const int group_length = group_lengths[group];
    BatchedScalar real_sum = 0.0f;
    BatchedScalar imag_sum = 0.0f;
    if (channel < input_channel_count) {
        const int group_stop = group_start + group_length;
        for (int block = group_start + warp; block < group_stop; block += 8) {
            const BatchedComplex a = spectrum_a[
                (long long)block * spectrum_a_stride + channel
            ];
            const BatchedComplex b = spectrum_b[
                (long long)block * spectrum_b_stride + channel
            ];
            real_sum += a.x * b.x + a.y * b.y;
            imag_sum += a.y * b.x - a.x * b.y;
        }
    }

    __shared__ BatchedScalar real_parts[8][32];
    __shared__ BatchedScalar imag_parts[8][32];
    real_parts[warp][lane] = real_sum;
    imag_parts[warp][lane] = imag_sum;
    __syncthreads();
    if (warp == 0) {
        real_sum = 0.0f;
        imag_sum = 0.0f;
        #pragma unroll
        for (int partial = 0; partial < 8; ++partial) {
            real_sum += real_parts[partial][lane];
            imag_sum += imag_parts[partial][lane];
        }
        if (CHANNELS_TO_AVERAGE == 32) {
            real_sum += __shfl_down_sync(0xffffffff, real_sum, 16);
            imag_sum += __shfl_down_sync(0xffffffff, imag_sum, 16);
            real_sum += __shfl_down_sync(0xffffffff, real_sum, 8);
            imag_sum += __shfl_down_sync(0xffffffff, imag_sum, 8);
            real_sum += __shfl_down_sync(0xffffffff, real_sum, 4);
            imag_sum += __shfl_down_sync(0xffffffff, imag_sum, 4);
        }
        if (CHANNELS_TO_AVERAGE >= 4) {
            real_sum += __shfl_down_sync(0xffffffff, real_sum, 1);
            imag_sum += __shfl_down_sync(0xffffffff, imag_sum, 1);
            real_sum += __shfl_down_sync(0xffffffff, real_sum, 2);
            imag_sum += __shfl_down_sync(0xffffffff, imag_sum, 2);
        }
        if (
            channel < input_channel_count
            && (
                CHANNELS_TO_AVERAGE == 1
                || (lane & (CHANNELS_TO_AVERAGE - 1)) == 0
            )
        ) {
            const long long output_channel = output_offsets[group]
                + channel / CHANNELS_TO_AVERAGE;
            const double scale = inverse_spectral_normalization
                / CHANNELS_TO_AVERAGE;
            store_batched_visibility(
                output, output_channel, real_sum * scale, imag_sum * scale
            );
        }
    }

    if (tile == 0) {
        __shared__ double weight_parts[256];
        reduce_visibility_weights(
            weight_a + group_start,
            weight_b + group_start,
            multiply_weights,
            group_length,
            weight_parts,
            weight_sum + weight_offsets[group]
        );
    }
}

__device__ __forceinline__ void batched_visibility_generic_impl(
    const BatchedComplex* spectrum_a,
    const BatchedComplex* spectrum_b,
    long long spectrum_a_stride,
    long long spectrum_b_stride,
    const double* weight_a,
    const double* weight_b,
    bool multiply_weights,
    const int* group_starts,
    const int* group_lengths,
    const long long* output_offsets,
    const long long* weight_offsets,
    int input_channel_count,
    int channels_to_average,
    double inverse_spectral_normalization,
    BatchedComplex* output,
    double* weight_sum
) {
    const int group = (int)blockIdx.y;
    const int output_channel = (int)blockIdx.x * blockDim.x + threadIdx.x;
    const int output_channel_count =
        input_channel_count / channels_to_average;
    const int group_start = group_starts[group];
    const int group_length = group_lengths[group];
    if (output_channel < output_channel_count) {
        const int first_channel = output_channel * channels_to_average;
        BatchedScalar real_sum = 0.0f;
        BatchedScalar imag_sum = 0.0f;
        for (int relative_block = 0; relative_block < group_length;
            ++relative_block) {
            const long long block = group_start + relative_block;
            for (int offset = 0; offset < channels_to_average; ++offset) {
                const int channel = first_channel + offset;
                const BatchedComplex a = spectrum_a[
                    block * spectrum_a_stride + channel
                ];
                const BatchedComplex b = spectrum_b[
                    block * spectrum_b_stride + channel
                ];
                real_sum += a.x * b.x + a.y * b.y;
                imag_sum += a.y * b.x - a.x * b.y;
            }
        }
        const double scale = inverse_spectral_normalization
            / channels_to_average;
        store_batched_visibility(
            output, output_offsets[group] + output_channel,
            real_sum * scale, imag_sum * scale
        );
    }
    if (blockIdx.x == 0) {
        __shared__ double weight_parts[256];
        reduce_visibility_weights(
            weight_a + group_start,
            weight_b + group_start,
            multiply_weights,
            group_length,
            weight_parts,
            weight_sum + weight_offsets[group]
        );
    }
}

extern "C" __global__ void BATCHED_VISIBILITY_ENTRY(avg1)(
    const BatchedComplex* spectrum_a,
    const BatchedComplex* spectrum_b,
    long long spectrum_a_stride,
    long long spectrum_b_stride,
    const double* weight_a,
    const double* weight_b,
    bool multiply_weights,
    const int* group_starts,
    const int* group_lengths,
    const long long* output_offsets,
    const long long* weight_offsets,
    int input_channel_count,
    double inverse_spectral_normalization,
    BatchedComplex* output,
    double* weight_sum
) {
    batched_visibility_tiled_impl<1>(
        spectrum_a, spectrum_b, spectrum_a_stride, spectrum_b_stride,
        weight_a, weight_b, multiply_weights,
        group_starts, group_lengths, output_offsets, weight_offsets,
        input_channel_count, inverse_spectral_normalization, output,
            weight_sum
    );
}

extern "C" __global__ void BATCHED_VISIBILITY_ENTRY(avg4)(
    const BatchedComplex* spectrum_a,
    const BatchedComplex* spectrum_b,
    long long spectrum_a_stride,
    long long spectrum_b_stride,
    const double* weight_a,
    const double* weight_b,
    bool multiply_weights,
    const int* group_starts,
    const int* group_lengths,
    const long long* output_offsets,
    const long long* weight_offsets,
    int input_channel_count,
    double inverse_spectral_normalization,
    BatchedComplex* output,
    double* weight_sum
) {
    batched_visibility_tiled_impl<4>(
        spectrum_a, spectrum_b, spectrum_a_stride, spectrum_b_stride,
        weight_a, weight_b, multiply_weights,
        group_starts, group_lengths, output_offsets, weight_offsets,
        input_channel_count, inverse_spectral_normalization, output,
            weight_sum
    );
}

extern "C" __global__ void BATCHED_VISIBILITY_ENTRY(avg32)(
    const BatchedComplex* spectrum_a,
    const BatchedComplex* spectrum_b,
    long long spectrum_a_stride,
    long long spectrum_b_stride,
    const double* weight_a,
    const double* weight_b,
    bool multiply_weights,
    const int* group_starts,
    const int* group_lengths,
    const long long* output_offsets,
    const long long* weight_offsets,
    int input_channel_count,
    double inverse_spectral_normalization,
    BatchedComplex* output,
    double* weight_sum
) {
    batched_visibility_tiled_impl<32>(
        spectrum_a, spectrum_b, spectrum_a_stride, spectrum_b_stride,
        weight_a, weight_b, multiply_weights,
        group_starts, group_lengths, output_offsets, weight_offsets,
        input_channel_count, inverse_spectral_normalization, output,
            weight_sum
    );
}

extern "C" __global__ void BATCHED_VISIBILITY_ENTRY(generic)(
    const BatchedComplex* spectrum_a,
    const BatchedComplex* spectrum_b,
    long long spectrum_a_stride,
    long long spectrum_b_stride,
    const double* weight_a,
    const double* weight_b,
    bool multiply_weights,
    const int* group_starts,
    const int* group_lengths,
    const long long* output_offsets,
    const long long* weight_offsets,
    int input_channel_count,
    int channels_to_average,
    double inverse_spectral_normalization,
    BatchedComplex* output,
    double* weight_sum
) {
    batched_visibility_generic_impl(
        spectrum_a, spectrum_b, spectrum_a_stride, spectrum_b_stride,
        weight_a, weight_b, multiply_weights,
        group_starts, group_lengths, output_offsets, weight_offsets,
        input_channel_count, channels_to_average,
        inverse_spectral_normalization, output, weight_sum
    );
}
"""


_MULTI_VISIBILITY_SOURCE = r"""
#include <cuComplex.h>

enum { MAX_BANDS = 8, MAX_PRODUCTS = 8, WARPS_PER_BLOCK = 8 };
#ifndef BAND_LIMIT
#define BAND_LIMIT MAX_BANDS
#endif
#ifndef PRODUCT_LIMIT
#define PRODUCT_LIMIT MAX_PRODUCTS
#endif

extern "C" __global__ void batched_visibility_products_float(
    const unsigned long long* spectrum_pointers,
    const long long* spectrum_strides,
    const int* product_left,
    const int* product_right,
    const int* band_counts,
    const int* product_counts,
    const int* product_rows,
    const unsigned long long* weight_left_pointers,
    const unsigned long long* weight_right_pointers,
    const unsigned char* multiply_weights,
    const int* group_starts,
    const int* group_lengths,
    const long long* output_offsets,
    const long long* weight_offsets,
    const double* inverse_normalizations,
    int group_count,
    int input_channel_count,
    int channels_to_average,
    int pack_offset,
    float2* output,
    double* weight_sum
) {
    const int pack = pack_offset + (int)blockIdx.z;
    const int group = (int)blockIdx.y;
    const int tile = (int)blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int channel = tile * 32 + lane;
    const int group_start = group_starts[group];
    const int group_stop = group_start + group_lengths[group];
    const int band_count = band_counts[pack];
    const int product_count = product_counts[pack];
    const int band_base = pack * MAX_BANDS;
    const int product_base = pack * MAX_PRODUCTS;
    float real_acc[PRODUCT_LIMIT] = {0.0f};
    float imag_acc[PRODUCT_LIMIT] = {0.0f};

    if (channel < input_channel_count) {
        for (int block = group_start + warp; block < group_stop;
             block += WARPS_PER_BLOCK) {
            float2 values[BAND_LIMIT];
            #pragma unroll
            for (int band = 0; band < BAND_LIMIT; ++band) {
                if (band < band_count) {
                    const float2* spectrum = reinterpret_cast<const float2*>(
                        spectrum_pointers[band_base + band]
                    );
                    const long long element =
                        (long long)block *
                            spectrum_strides[band_base + band] + channel;
                    values[band] = spectrum[element];
                }
            }
            #pragma unroll
            for (int product = 0; product < PRODUCT_LIMIT; ++product) {
                if (product < product_count) {
                    const float2 a =
                        values[product_left[product_base + product]];
                    const float2 b =
                        values[product_right[product_base + product]];
                    real_acc[product] += a.x * b.x + a.y * b.y;
                    imag_acc[product] += a.y * b.x - a.x * b.y;
                }
            }
        }
    }

    __shared__ float real_parts[PRODUCT_LIMIT][WARPS_PER_BLOCK][32];
    __shared__ float imag_parts[PRODUCT_LIMIT][WARPS_PER_BLOCK][32];
    #pragma unroll
    for (int product = 0; product < PRODUCT_LIMIT; ++product) {
        if (product < product_count) {
            real_parts[product][warp][lane] = real_acc[product];
            imag_parts[product][warp][lane] = imag_acc[product];
        }
    }
    __syncthreads();

    if (warp == 0) {
        #pragma unroll
        for (int product = 0; product < PRODUCT_LIMIT; ++product) {
            if (product < product_count) {
                float real_total = 0.0f;
                float imag_total = 0.0f;
                #pragma unroll
                for (int partial = 0; partial < WARPS_PER_BLOCK; ++partial) {
                    real_total += real_parts[product][partial][lane];
                    imag_total += imag_parts[product][partial][lane];
                }
                if (channels_to_average == 32) {
                    real_total += __shfl_down_sync(
                        0xffffffff, real_total, 16
                    );
                    imag_total += __shfl_down_sync(
                        0xffffffff, imag_total, 16
                    );
                    real_total += __shfl_down_sync(
                        0xffffffff, real_total, 8
                    );
                    imag_total += __shfl_down_sync(
                        0xffffffff, imag_total, 8
                    );
                    real_total += __shfl_down_sync(
                        0xffffffff, real_total, 4
                    );
                    imag_total += __shfl_down_sync(
                        0xffffffff, imag_total, 4
                    );
                }
                if (channels_to_average >= 4) {
                    real_total += __shfl_down_sync(0xffffffff, real_total, 1);
                    imag_total += __shfl_down_sync(0xffffffff, imag_total, 1);
                    real_total += __shfl_down_sync(0xffffffff, real_total, 2);
                    imag_total += __shfl_down_sync(0xffffffff, imag_total, 2);
                }
                if (
                    channel < input_channel_count
                    && (
                        channels_to_average == 1
                        || (lane & (channels_to_average - 1)) == 0
                    )
                ) {
                    const long long offset =
                        output_offsets[
                            (long long)product_rows[
                                product_base + product
                            ] * group_count + group
                        ]
                        + channel / channels_to_average;
                    const double scale = inverse_normalizations[
                        product_base + product
                    ]
                        / channels_to_average;
                    output[offset] = make_float2(
                        (float)(real_total * scale),
                        (float)(imag_total * scale)
                    );
                }
            }
        }
    }

    if (tile == 0) {
        __shared__ double weight_parts[PRODUCT_LIMIT][256];
        #pragma unroll
        for (int product = 0; product < PRODUCT_LIMIT; ++product) {
            if (product < product_count) {
                const double* left = reinterpret_cast<const double*>(
                    weight_left_pointers[product_base + product]
                );
                const double* right = reinterpret_cast<const double*>(
                    weight_right_pointers[product_base + product]
                );
                double subtotal = 0.0;
                for (int block = group_start + threadIdx.x;
                    block < group_stop;
                     block += blockDim.x) {
                    subtotal += multiply_weights[product_base + product]
                        ? left[block] * right[block]
                        : left[block];
                }
                weight_parts[product][threadIdx.x] = subtotal;
            }
        }
        __syncthreads();
        for (int offset = 128; offset > 0; offset >>= 1) {
            if (threadIdx.x < offset) {
                #pragma unroll
                for (int product = 0; product < PRODUCT_LIMIT; ++product) {
                    if (product < product_count) {
                        weight_parts[product][threadIdx.x] +=
                            weight_parts[product][threadIdx.x + offset];
                    }
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            #pragma unroll
            for (int product = 0; product < PRODUCT_LIMIT; ++product) {
                if (product < product_count) {
                    const long long offset =
                        weight_offsets[
                            (long long)product_rows[
                                product_base + product
                            ] * group_count + group
                        ];
                    weight_sum[offset] = weight_parts[product][0];
                }
            }
        }
    }
}
"""
