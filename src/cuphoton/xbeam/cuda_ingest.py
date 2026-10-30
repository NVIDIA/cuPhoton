# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Cuda ingest implementation."""

from __future__ import annotations

_DECODE_SOURCE = r"""
extern "C" __global__ void decode_real_band(
    const unsigned char* payloads,
    long long sample_count,
    int samples_per_frame,
    int payload_bytes,
    int channel_count,
    int channel_index,
    float* output
) {
    const long long sample = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (sample >= sample_count) {
        return;
    }
    const long long frame = sample / samples_per_frame;
    const long long within_frame = sample - frame * samples_per_frame;
    const long long component = within_frame * channel_count + channel_index;
    const unsigned char packed = payloads[frame * payload_bytes +
        component / 4];
    const unsigned char code = (packed >> ((component & 3) * 2)) & 3;
    const float levels[4] = {-3.3359f, -1.0f, 1.0f, 3.3359f};
    output[sample] = levels[code];
}

extern "C" __global__ void decode_complex_band(
    const unsigned char* payloads,
    long long sample_count,
    int samples_per_frame,
    int payload_bytes,
    int channel_count,
    int channel_index,
    float2* output
) {
    const long long sample = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (sample >= sample_count) {
        return;
    }
    const long long frame = sample / samples_per_frame;
    const long long within_frame = sample - frame * samples_per_frame;
    const long long first_component =
        (within_frame * channel_count + channel_index) * 2;
    const long long payload_start = frame * payload_bytes;
    const unsigned char packed_i = payloads[
        payload_start + first_component / 4
    ];
    const long long q_component = first_component + 1;
    const unsigned char packed_q = payloads[
        payload_start + q_component / 4
    ];
    const unsigned char code_i =
        (packed_i >> ((first_component & 3) * 2)) & 3;
    const unsigned char code_q =
        (packed_q >> ((q_component & 3) * 2)) & 3;
    const float levels[4] = {-3.3359f, -1.0f, 1.0f, 3.3359f};
    output[sample] = make_float2(levels[code_i], levels[code_q]);
}
"""


_VDIF_INGEST_SOURCE = r"""

extern "C" __global__ void strip_scatter_vdif_payload(
    const unsigned long long* raw_frames,
    const int* destination_slots,
    int frame_count,
    int frame_words,
    int payload_words,
    unsigned long long* output
) {
    const long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long word_count = (long long)frame_count * payload_words;
    if (index >= word_count) {
        return;
    }
    const int frame = (int)(index / payload_words);
    const int payload_word = (int)(index - (long long)frame * payload_words);
    const int destination = destination_slots[frame];
    if (destination < 0) {
        return;
    }
    output[(long long)destination * payload_words + payload_word] =
        raw_frames[
        (long long)frame * frame_words + 4 + payload_word
    ];
}

extern "C" __global__ void strip_scatter_sparse_vdif_payload(
    const unsigned long long* raw_frames,
    const int* source_frames,
    const int* destination_slots,
    int mapping_count,
    int source_frame_count,
    int destination_count,
    int frame_words,
    int payload_words,
    unsigned long long* output
) {
    const long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long word_count = (long long)mapping_count * payload_words;
    if (index >= word_count) {
        return;
    }
    const int mapping = (int)(index / payload_words);
    const int payload_word =
        (int)(index - (long long)mapping * payload_words);
    const int source = source_frames[mapping];
    const int destination = destination_slots[mapping];
    if (
        source < 0 || source >= source_frame_count
        || destination < 0 || destination >= destination_count
    ) {
        return;
    }
    output[(long long)destination * payload_words + payload_word] =
        raw_frames[
        (long long)source * frame_words + 4 + payload_word
    ];
}

extern "C" __global__ void strip_scatter_vdif_payload_validity(
    const unsigned long long* raw_frames,
    const int* destination_slots,
    int frame_count,
    int frame_words,
    int payload_words,
    unsigned long long* output,
    unsigned char* output_validity
) {
    const long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long word_count = (long long)frame_count * payload_words;
    if (index >= word_count) {
        return;
    }
    const int frame = (int)(index / payload_words);
    const int payload_word = (int)(index - (long long)frame * payload_words);
    const int destination = destination_slots[frame];
    if (destination < 0) {
        return;
    }
    const long long input_start = (long long)frame * frame_words;
    output[(long long)destination * payload_words + payload_word] =
        raw_frames[
        input_start + 4 + payload_word
    ];
    if (payload_word == 0) {
        output_validity[destination] =
            (raw_frames[input_start] & 0x80000000ULL) == 0;
    }
}
"""
