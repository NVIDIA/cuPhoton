// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "xbeam/replay.hpp"
namespace xbeam {
struct WindowGeometry {
    std::int64_t frames_per_fft, ffts_per_request, buffered_ffts, ring_segments;
    std::int64_t allocation_frames, request_frames, read_buffer_ratio;
};
// The subintegration is supplied as its exact decimal-derived rational.
WindowGeometry size_windows(
    std::int64_t samples_per_frame,
    std::int64_t fft_samples,
    std::int64_t sample_rate,
    std::int64_t subint_num,
    std::int64_t subint_den,
    std::int64_t buffer_factor,
    std::int64_t data_segments,
    std::int64_t buffered_ffts,
    double reference_rate_allowance);
struct NormalExtent {
    std::int64_t frame_bytes, payload_bytes, request_frames, read_buffer_ratio;
    std::int64_t ratio_num, ratio_den, samples_per_frame;
    std::int64_t framed_admission_samples, decoded_samples, allocation_bytes;
};
NormalExtent normal_extent(
    std::int64_t frame_bytes,
    std::int64_t payload_bytes,
    std::int64_t request_frames,
    std::int64_t read_buffer_ratio,
    std::int64_t ratio_num,
    std::int64_t ratio_den);
std::int64_t leading_head(
    std::int64_t output,
    double endpoint_delay,
    std::int64_t sample_rate,
    std::int64_t samples_per_frame,
    std::int64_t timestamp_quantum_ns);
struct ReturnedWindow {
    std::int64_t fft, output, logical_head, physical_head;
    double centre_delay;
    std::int64_t framed_extent, decoded_extent;
    bool published, clamp_negative;
};
// A bounded normal-return contract. Initial returns and EOF are explicit inputs.
std::vector<ReturnedWindow> returned_windows(
    const PhysicalTimeline&,
    const RawRequestState&,
    const JobRequestInputs&,
    const RequestGeometry&,
    const NormalExtent&,
    std::int64_t control_bits,
    std::int64_t timestamp_quantum_ns,
    std::int64_t known_source_stop);
}  // namespace xbeam
namespace xbeam {
struct ReplayRow {
    std::int64_t fft, physical_start, timing_displacement, valid_first,
        valid_stop;
    bool accepted;
};
// Produces source coordinates and decoder masks; frame payload validity is
// still supplied independently to the GPU frontend.
std::vector<ReplayRow> compile_replay_rows(
    const std::vector<ReturnedWindow>&,
    std::int64_t fft_samples,
    std::int64_t sample_rate,
    std::int64_t known_source_stop);
}  // namespace xbeam

namespace xbeam {
std::int64_t packed_frame_slot(std::uint64_t timestamp, std::int64_t fps);
std::int64_t reference_epoch_seconds(int epoch);
}  // namespace xbeam
