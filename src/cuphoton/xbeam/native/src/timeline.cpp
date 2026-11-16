// SPDX-License-Identifier: Apache-2.0
#include "xbeam/timeline.hpp"
#include "xbeam/radio.hpp"
#include <algorithm>
#include <limits>
namespace xbeam {
namespace {
using Wide = __int128;
std::int64_t checked(Wide x) {
    if (x < INT64_MIN || x > INT64_MAX) {
        throw RadioError("timeline coordinate exceeds int64");
    }
    return static_cast<std::int64_t>(x);
}
std::int64_t ceil_div(std::int64_t x, std::int64_t y) {
    return x / y + (x % y > 0);
}
Wide floor_div(Wide x, Wide y) {
    return x / y - (x % y < 0);
}
void bounds(std::int64_t first, std::int64_t count, std::int64_t samples) {
    if (first < 0 || count <= 0 || samples <= 0) {
        throw RadioError("invalid timeline sample request");
    }
}
}  // namespace
PhysicalTimeline::PhysicalTimeline(
    const std::vector<PhysicalRun>& input, std::int64_t allocation_frames)
    : runs(input),
      capacity(allocation_frames) {
    if (capacity <= 0 || runs.empty() || runs.front().first != 0) {
        throw RadioError(
            "timeline requires positive allocation and zero-based runs");
    }
    for (std::size_t i = 0; i < runs.size(); ++i) {
        const auto& r = runs[i];
        if (r.first < 0 || r.timestamp < 0 || r.stop <= r.first
            || (i && r.first < runs[i - 1].stop)) {
            throw RadioError("invalid physical timeline runs");
        }
        auto stop = checked(Wide(r.timestamp) + r.stop - r.first);
        if (i
            && Wide(r.timestamp) - timestamp_stops.back()
                    - (Wide(r.first) - runs[i - 1].stop)
                < 0) {
            throw RadioError("backwards timestamp residual");
        }
        physical_stops.push_back(r.stop);
        timestamp_stops.push_back(stop);
    }
    const auto allocation_count = ceil_div(runs.back().stop, capacity);
    // Both vectors together have a bounded 256 MiB admission budget.
    if (allocation_count > 16777216) {
        throw RadioError("timeline allocation table exceeds memory budget");
    }
    allocation_starts_.reserve(allocation_count);
    anchors_.reserve(allocation_count);
    for (std::int64_t i = 0; i < allocation_count; ++i) {
        auto first = checked(Wide(i) * capacity);
        auto pos = std::upper_bound(
                       physical_stops.begin(), physical_stops.end(), first)
            - physical_stops.begin();
        if (std::size_t(pos) >= runs.size()) {
            pos = runs.size() - 1;
        }
        allocation_starts_.push_back(first);
        anchors_.push_back(
            checked(Wide(runs[pos].timestamp) + first - runs[pos].first));
    }
}
std::vector<FrameInterval> PhysicalTimeline::uncovered_heads() const {
    std::vector<FrameInterval> out;
    for (std::size_t i = 1; i < runs.size(); ++i) {
        if (runs[i].first > runs[i - 1].stop) {
            auto first = ceil_div(runs[i - 1].stop, capacity),
                 stop = ceil_div(runs[i].first, capacity);
            if (first >= stop) {
                continue;
            }
            if (!out.empty() && first <= out.back().stop) {
                out.back().stop = std::max(stop, out.back().stop);
            } else {
                out.push_back({first, stop});
            }
        }
    }
    return out;
}
std::vector<AnchorGap> PhysicalTimeline::deferred_gaps() const {
    std::vector<AnchorGap> out;
    for (std::size_t i = 1; i < anchors_.size(); ++i) {
        auto expected = checked(
            Wide(anchors_[i - 1]) + allocation_starts_[i]
            - allocation_starts_[i - 1]);
        if (anchors_[i] > expected) {
            out.push_back(
                {static_cast<std::int64_t>(i),
                    allocation_starts_[i],
                    expected,
                    anchors_[i]});
        }
    }
    return out;
}
std::vector<SampleSpan> PhysicalTimeline::source_spans(
    std::int64_t first, std::int64_t count, std::int64_t samples) const {
    bounds(first, count, samples);
    Wide end = Wide(first) + count;
    std::vector<SampleSpan> out;
    auto pos =
        std::upper_bound(
            timestamp_stops.begin(), timestamp_stops.end(), first / samples)
        - timestamp_stops.begin();
    for (std::size_t i = pos; i < runs.size(); ++i) {
        Wide a = Wide(runs[i].timestamp) * samples,
             b = Wide(timestamp_stops[i]) * samples;
        if (a >= end) {
            break;
        }
        Wide begin = std::max(Wide(first), a), stop = std::min(end, b);
        if (begin < stop) {
            out.push_back(
                {checked(begin - first),
                    checked(stop - first),
                    checked(Wide(runs[i].first) * samples + begin - a)});
        }
    }
    return out;
}
std::vector<SampleSpan> PhysicalTimeline::latched_spans(
    std::int64_t allocation,
    std::int64_t first,
    std::int64_t count,
    std::int64_t samples) const {
    bounds(first, count, samples);
    if (allocation < 0 || std::size_t(allocation) >= anchors_.size()) {
        throw RadioError("allocation exceeds timeline");
    }
    const Wide physical = Wide(allocation_starts_[allocation]) * samples + first
        - Wide(anchors_[allocation]) * samples;
    const Wide end = physical + count;
    std::vector<SampleSpan> out;
    auto origin = floor_div(physical, samples);
    auto pos = std::upper_bound(
                   physical_stops.begin(),
                   physical_stops.end(),
                   origin,
                   [](Wide x, std::int64_t y) { return x < y; })
        - physical_stops.begin();
    for (std::size_t i = pos; i < runs.size(); ++i) {
        Wide a = Wide(runs[i].first) * samples,
             b = Wide(runs[i].stop) * samples;
        if (a >= end) {
            break;
        }
        Wide begin = std::max(physical, a), stop = std::min(end, b);
        if (begin >= stop) {
            continue;
        }
        SampleSpan span{checked(begin - physical),
            checked(stop - physical),
            checked(begin)};
        if (!out.empty() && out.back().request_stop == span.request_first
            && Wide(out.back().physical_first) + out.back().request_stop
                    - out.back().request_first
                == span.physical_first) {
            out.back().request_stop = span.request_stop;
        } else {
            out.push_back(span);
        }
    }
    return out;
}
}  // namespace xbeam
