// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "xbeam/window.hpp"
#include <memory>
namespace xbeam {
struct RecordingGeometry {
    std::int64_t sample_rate, frames_per_second, samples_per_frame, frame_count;
    std::int64_t frame_bytes, payload_bytes, job_slot;
    int channel_count, thread_id;
    WindowGeometry windows;
    RequestGeometry request;
    NormalExtent extent;
};
struct RadioReplayBatch {
    std::vector<SourceBlock> blocks;
    std::vector<ReplayRow> replay;
    int channel_index;
};
class RadioRecording {
public:
    RadioRecording(const RadioJob&, int stream, const std::string& index);
    // Adopts an already validated sidecar instead of validating it again.
    RadioRecording(const RadioJob&, int stream, IndexReader&& index);
    ~RadioRecording();
    const RecordingGeometry& geometry() const;
    std::vector<unsigned char> frame_validity(
        std::int64_t first, std::int64_t count) const;
    RadioReplayBatch plan(
        const std::string& scan,
        int band,
        std::int64_t first,
        std::int64_t stop,
        std::int64_t timeline_count) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
}  // namespace xbeam
