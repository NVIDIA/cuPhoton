// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "xbeam/radio.hpp"
#include "xbeam/timeline.hpp"
namespace xbeam {
struct RawRequestState {
    std::vector<std::int64_t> anchors;
    std::vector<unsigned char> events;
    std::int64_t allocation_frames, samples_per_frame, timestamp_origin;
};
struct RequestFact {
    std::int64_t first_fft, desired_sample;
    std::int64_t active_count = 0;
    double endpoint_drift;
    std::vector<std::int64_t> ordinary_samples;
};
struct JobRequestInputs {
    std::int64_t first_fft, stop_fft, timeline_fft_count, timestamp_origin;
    double sample_rate;
    std::vector<RequestFact> requests;
    std::vector<std::vector<std::int64_t>> output_samples;
    std::vector<std::vector<double>> centre_delays, read_average;
    std::vector<double> left_delays;
};
struct DiscardBatch {
    std::vector<std::int64_t> allocations;
    std::vector<RequestCursor> cursors;
    std::vector<unsigned char> valid, events;
    std::vector<std::vector<unsigned char>> eligible;
    std::vector<FrameInterval> excluded;
};
RawRequestState raw_request_state(
    const PhysicalTimeline&,
    std::int64_t samples_per_frame,
    std::int64_t timestamp_origin,
    bool clock_frame_owner);
DiscardBatch compile_discards(
    const RawRequestState&,
    const std::vector<RequestFact>&,
    const RequestGeometry&,
    double sample_rate,
    std::int64_t frame_bytes,
    std::int64_t control_word_bits);
struct WindowBounds {
    std::int64_t nearest, valid_first, valid_stop;
    bool before, after;
};
WindowBounds window_bounds(
    std::int64_t output,
    std::int64_t head,
    double delay_seconds,
    std::int64_t sample_rate,
    std::int64_t framed_extent,
    std::int64_t decoded_extent,
    std::int64_t fft_samples);
}  // namespace xbeam
