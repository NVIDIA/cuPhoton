// SPDX-License-Identifier: Apache-2.0
#include "xbeam/replay.hpp"
#include <algorithm>
#include <cmath>
namespace xbeam {
namespace {
using Wide = __int128;
std::int64_t narrow(Wide x) {
    if (x < INT64_MIN || x > INT64_MAX) {
        throw RadioError("replay coordinate exceeds int64");
    }
    return static_cast<std::int64_t>(x);
}
}  // namespace
RawRequestState raw_request_state(
    const PhysicalTimeline& timeline,
    std::int64_t samples,
    std::int64_t origin,
    bool clock_owner) {
    if (samples <= 0) {
        throw RadioError("samples per frame must be positive");
    }
    RawRequestState state;
    state.allocation_frames = timeline.allocation_frames();
    state.samples_per_frame = samples;
    state.timestamp_origin = origin;
    for (auto anchor : timeline.anchors()) {
        state.anchors.push_back(narrow((Wide(anchor) - origin) * samples));
    }
    state.events.resize(state.anchors.size(), 0);
    const auto& runs = timeline.physical_runs();
    for (std::size_t i = 1; i < runs.size(); ++i) {
        Wide residual = Wide(runs[i].timestamp) - runs[i - 1].timestamp
            - (runs[i - 1].stop - runs[i - 1].first)
            - (runs[i].first - runs[i - 1].stop);
        if (residual > 0) {
            auto preceding = clock_owner ? runs[i - 1].stop : runs[i].first;
            auto event = (preceding - 1) / state.allocation_frames;
            if (event < 0 || std::size_t(event) >= state.events.size()) {
                throw RadioError("event allocation exceeds timeline");
            }
            state.events[event] = 1;
        }
    }
    return state;
}
DiscardBatch compile_discards(
    const RawRequestState& state,
    const std::vector<RequestFact>& facts,
    const RequestGeometry& g,
    double rate,
    std::int64_t frame_bytes,
    std::int64_t control_bits) {
    if (frame_bytes <= 0 || control_bits <= 0 || state.allocation_frames <= 0
        || state.anchors.empty()
        || state.anchors.size() != state.events.size()) {
        throw RadioError("invalid framed replay geometry");
    }
    const auto extent = narrow(Wide(state.allocation_frames) * frame_bytes);
    request_cursor(
        0, 0, g, rate);  // Geometry is validated even for an empty batch.
    const auto bytes =
        narrow(Wide(g.fft_samples) * g.ratio_denominator / g.ratio_numerator);
    DiscardBatch out;
    std::vector<std::int64_t> excluded;
    for (const auto& fact : facts) {
        const auto width = fact.ordinary_samples.size();
        const auto active =
            fact.active_count ? std::size_t(fact.active_count) : width;
        if (active < 1 || active > width) {
            throw RadioError("invalid request count");
        }
        if (fact.first_fft < 0 || width == 0
            || width > std::size_t(g.ffts_per_request)) {
            throw RadioError("invalid replay request width/start");
        }
        narrow(Wide(fact.first_fft) + width - 1);
        auto selected =
            std::upper_bound(
                state.anchors.begin(), state.anchors.end(), fact.desired_sample)
            - state.anchors.begin() - 1;
        bool allocated =
            selected >= 0 && std::size_t(selected) < state.anchors.size();
        auto local = allocated
            ? narrow(Wide(fact.desired_sample) - state.anchors[selected])
            : 0;
        auto cursor = request_cursor(local, fact.endpoint_drift, g, rate);
        bool valid = allocated && cursor.valid;
        if (!valid) {
            cursor = {};
        }
        bool event = valid && state.events[selected];
        std::vector<unsigned char> mask(width, valid ? 1 : 0);
        if (event) {
            bool any = false;
            for (std::size_t j = 0; j < width; ++j) {
                mask[j] =
                    narrow(Wide(cursor.origin_bytes) + Wide(j) * bytes) < extent
                    && std::int64_t(j) >= cursor.skipped_ffts && j < active;
                if (j < std::size_t(control_bits) && mask[j]) {
                    any = true;
                }
            }
            if (!any) {
                std::fill(mask.begin(), mask.end(), 0);
            }
            if (std::size_t(selected + 1) < state.anchors.size()) {
                for (std::size_t j = 0; j < width; ++j) {
                    if (fact.ordinary_samples[j]
                        >= state.anchors[selected + 1]) {
                        mask[j] = 1;
                    }
                }
            }
        }
        for (std::size_t j = 0; j < width; ++j) {
            if (std::int64_t(j) < cursor.skipped_ffts || j >= active) {
                mask[j] = 0;
            }
            if (valid && j < active && !mask[j]) {
                excluded.push_back(narrow(Wide(fact.first_fft) + j));
            }
        }
        out.allocations.push_back(selected);
        out.cursors.push_back(cursor);
        out.valid.push_back(valid);
        out.events.push_back(event);
        out.eligible.push_back(std::move(mask));
    }
    std::sort(excluded.begin(), excluded.end());
    excluded.erase(
        std::unique(excluded.begin(), excluded.end()), excluded.end());
    for (auto i : excluded) {
        auto stop = narrow(Wide(i) + 1);
        if (!out.excluded.empty() && out.excluded.back().stop == i) {
            out.excluded.back().stop = stop;
        } else {
            out.excluded.push_back({i, stop});
        }
    }
    return out;
}
WindowBounds window_bounds(
    std::int64_t output,
    std::int64_t head,
    double delay,
    std::int64_t rate,
    std::int64_t extent,
    std::int64_t decoded,
    std::int64_t fft) {
    if (rate <= 0 || extent < 0 || decoded < 0 || fft <= 0
        || !std::isfinite(delay)) {
        throw RadioError("invalid returned-window clock geometry");
    }
    const double samples = delay * double(rate);
    if (!std::isfinite(samples) || std::abs(samples) >= std::ldexp(1., 53)) {
        throw RadioError("window delay exceeds exact sample-phase range");
    }
    const auto whole = static_cast<std::int64_t>(std::floor(samples));
    Wide nearest = Wide(output) - head - whole;
    bool integral = samples == double(whole);
    const double midpoint = double(whole) + .5;
    if (nearest < 0 && (integral || samples < midpoint)) {
        ++nearest;
    }
    if (nearest > 0 && !integral && samples > midpoint) {
        --nearest;
    }
    auto value = narrow(nearest);
    auto first = nearest <= -Wide(fft) ? fft
        : nearest < 0                  ? narrow(-nearest)
                                       : 0;
    auto stop = nearest >= decoded      ? 0
        : nearest > Wide(decoded) - fft ? narrow(Wide(decoded) - nearest)
                                        : fft;
    stop = std::max(stop, first);
    return {value, first, stop, value<-1, nearest> Wide(extent) - fft};
}
}  // namespace xbeam
