// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <vector>
namespace xbeam {
struct PhysicalRun {
    std::int64_t first, stop, timestamp;
};
struct SampleSpan {
    std::int64_t request_first, request_stop, physical_first;
};
struct AnchorGap {
    std::int64_t allocation, physical_boundary, first, stop;
};
struct FrameInterval {
    std::int64_t first, stop;
};
class PhysicalTimeline {
public:
    PhysicalTimeline(
        const std::vector<PhysicalRun>&, std::int64_t allocation_frames);
    const std::vector<std::int64_t>& allocation_starts() const {
        return allocation_starts_;
    }
    const std::vector<std::int64_t>& anchors() const {
        return anchors_;
    }
    std::vector<FrameInterval> uncovered_heads() const;
    std::vector<AnchorGap> deferred_gaps() const;
    std::vector<SampleSpan> source_spans(
        std::int64_t first,
        std::int64_t count,
        std::int64_t samples_per_frame) const;
    std::vector<SampleSpan> latched_spans(
        std::int64_t allocation,
        std::int64_t first,
        std::int64_t count,
        std::int64_t samples_per_frame) const;

    const std::vector<PhysicalRun>& physical_runs() const {
        return runs;
    }
    std::int64_t allocation_frames() const {
        return capacity;
    }

private:
    std::vector<PhysicalRun> runs;
    std::vector<std::int64_t> physical_stops, timestamp_stops,
        allocation_starts_, anchors_;
    std::int64_t capacity;
};
}  // namespace xbeam
