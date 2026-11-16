// SPDX-License-Identifier: Apache-2.0
#include "xbeam/window.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
namespace xbeam {
namespace {
using I = __int128;
std::int64_t checked(I v) {
    if (v < INT64_MIN || v > INT64_MAX) {
        throw RadioError("returned window coordinate exceeds int64");
    }
    return static_cast<std::int64_t>(v);
}
std::int64_t positive(I v) {
    if (v <= 0) {
        throw RadioError("window extent must be positive");
    }
    return checked(v);
}
I floor_div(I a, I b) {
    I q = a / b;
    return q - (a % b < 0);
}
}  // namespace
WindowGeometry size_windows(
    std::int64_t spf,
    std::int64_t fft,
    std::int64_t rate,
    std::int64_t num,
    std::int64_t den,
    std::int64_t factor,
    std::int64_t segments,
    std::int64_t buffered,
    double allowance) {
    if (spf <= 0 || fft <= 0 || rate <= 0 || num <= 0 || den <= 0 || factor <= 0
        || segments <= 0 || buffered <= 0 || fft % spf || factor % segments
        || !std::isfinite(allowance) || allowance < 0) {
        throw RadioError("invalid framed buffer geometry");
    }
    const I product = I(num) * rate, divisor = I(den) * fft;
    if (product % divisor) {
        throw RadioError("request duration is not FFT aligned");
    }
    auto width = positive(product / divisor), frames = fft / spf;
    I nominal = I(width) * frames;
    if (nominal >= (I(1) << 53)) {
        throw RadioError("nominal frames exceed exact binary64 range");
    }
    double increment = double(nominal) * allowance;
    double expanded = double(nominal) + increment;
    if (!std::isfinite(expanded) || expanded >= std::ldexp(1., 53) - 2.) {
        throw RadioError("expanded frames exceed exact binary64 range");
    }
    auto request = checked(I(static_cast<std::int64_t>(expanded)) + 2);
    return {frames,
        width,
        buffered,
        segments,
        positive(I(factor / segments) * request),
        request,
        factor / segments};
}
NormalExtent normal_extent(
    std::int64_t frame,
    std::int64_t payload,
    std::int64_t requests,
    std::int64_t ratio,
    std::int64_t num,
    std::int64_t den) {
    if (frame <= 0 || payload <= 0 || payload > frame || requests <= 0
        || ratio <= 0 || num <= 0 || den <= 0) {
        throw RadioError("invalid normal returned extent");
    }
    I spf = I(payload) * num, admission = I(requests) * frame * num;
    if (spf % den || admission % den) {
        throw RadioError("normal return requires integral samples");
    }
    positive(I(requests) * ratio);
    return {frame,
        payload,
        requests,
        ratio,
        num,
        den,
        positive(spf / den),
        positive(admission / den),
        positive(I(requests) * (spf / den)),
        positive(I(requests) * ratio * frame)};
}
std::int64_t leading_head(
    std::int64_t output,
    double endpoint,
    std::int64_t rate,
    std::int64_t spf,
    std::int64_t quantum) {
    if (rate <= 0 || spf <= 0 || quantum <= 0 || !std::isfinite(endpoint)) {
        throw RadioError("invalid leading head clock");
    }
    double units = std::trunc(endpoint * (1000000000. / double(quantum)));
    if (!std::isfinite(units) || units < -std::ldexp(1., 63)
        || units >= std::ldexp(1., 63)) {
        throw RadioError("endpoint nanoseconds exceed int64");
    }
    I ns = checked(I(static_cast<std::int64_t>(units)) * quantum);
    // ns * rate can span up to 126 bits; the subsequent subtraction stays
    // within signed 128 for validated int64 operands and billion-scale term.
    I position = I(output) * 1000000000 - ns * rate;
    return checked(floor_div(position, I(1000000000) * spf) * spf);
}
}  // namespace xbeam
namespace xbeam {
std::vector<ReturnedWindow> returned_windows(
    const PhysicalTimeline& timeline,
    const RawRequestState& state,
    const JobRequestInputs& inputs,
    const RequestGeometry& g,
    const NormalExtent& extent,
    std::int64_t control,
    std::int64_t quantum,
    std::int64_t source_stop) {
    if (inputs.sample_rate != std::trunc(inputs.sample_rate)
        || inputs.sample_rate <= 0 || inputs.sample_rate >= std::ldexp(1., 63)
        || control <= 0 || quantum <= 0
        || state.timestamp_origin != inputs.timestamp_origin
        || g.ratio_numerator != extent.ratio_num
        || g.ratio_denominator != extent.ratio_den
        || state.samples_per_frame != extent.samples_per_frame
        || I(extent.request_frames) * extent.read_buffer_ratio
            != state.allocation_frames) {
        throw RadioError("returned contract and raw geometry differ");
    }
    const auto rate = static_cast<std::int64_t>(inputs.sample_rate);
    const auto& runs = timeline.physical_runs();
    if (runs.empty()
        || source_stop
            < checked(I(runs.back().stop) * state.samples_per_frame)) {
        throw RadioError("known source end excludes physical history");
    }
    const auto batch = compile_discards(
        state,
        inputs.requests,
        g,
        inputs.sample_rate,
        extent.frame_bytes,
        control);
    std::vector<ReturnedWindow> result;
    if (inputs.stop_fft - inputs.first_fft > 16777216) {
        throw RadioError("returned scope exceeds memory budget");
    }
    result.reserve(inputs.stop_fft - inputs.first_fft);
    const auto uncovered = timeline.uncovered_heads();
    auto in_hole = [&](std::int64_t owner) {
        auto it = std::upper_bound(
            uncovered.begin(),
            uncovered.end(),
            owner,
            [](auto v, const auto& r) { return v < r.stop; });
        return it != uncovered.end() && owner >= it->first;
    };
    auto next_run = [&](std::int64_t frame) {
        return std::upper_bound(
            runs.begin(), runs.end(), frame, [](auto v, const auto& r) {
                return v < r.stop;
            });
    };
    const auto spf = state.samples_per_frame,
               capacity = state.allocation_frames;
    const auto header_prefix = checked(
        (I(capacity) * (extent.frame_bytes - extent.payload_bytes)
                * extent.ratio_num
            + extent.ratio_den - 1)
        / extent.ratio_den);
    bool seen_normal = false;
    for (std::size_t row = 0; row < inputs.requests.size(); ++row) {
        const auto& fact = inputs.requests[row];
        const auto active = fact.active_count
            ? fact.active_count
            : std::int64_t(fact.ordinary_samples.size());
        const auto width = fact.ordinary_samples.size();
        if (inputs.output_samples.at(row).size() != width
            || inputs.centre_delays.at(row).size() != width) {
            throw RadioError(
                "returned clock matrices have inconsistent shapes");
        }
        const bool initial = fact.desired_sample < state.anchors.front();
        if (initial && seen_normal) {
            throw RadioError("initial requests must form a prefix");
        }
        seen_normal |= !initial;
        std::int64_t selected = batch.allocations[row], allocation = selected,
                     cursor = 0, head = 0, original_head = 0;
        std::vector<unsigned char> eligible(width, 0);
        bool event = false, transfer = false;
        if (initial) {
            if (state.anchors.front() != 0
                || timeline.allocation_starts().front() != 0) {
                throw RadioError("initial source must start at job origin");
            }
            std::int64_t end = 0;
            for (const auto& run : runs) {
                if (run.first != end
                    || I(run.timestamp) - state.timestamp_origin != run.first) {
                    break;
                }
                end = run.stop;
                if (end >= capacity) {
                    break;
                }
            }
            if (end < capacity) {
                throw RadioError("initial allocation clock is not contiguous");
            }
            auto c = request_cursor(
                fact.desired_sample,
                fact.endpoint_drift,
                g,
                inputs.sample_rate);
            for (std::size_t j = 0; j < width; ++j) {
                eligible[j] = c.valid && std::int64_t(j) < active
                    && std::int64_t(j) >= c.skipped_ffts;
            }
            allocation = 0;
        } else {
            if (!batch.valid[row]) {
                throw RadioError("returned aligned head is unresolved");
            }
            event = batch.events[row];
            cursor = batch.cursors[row].origin_bytes;
            if (cursor % g.ratio_denominator) {
                throw RadioError("aligned cursor does not name a sample");
            }
            original_head = head = checked(
                I(state.anchors[selected])
                + I(cursor / g.ratio_denominator) * g.ratio_numerator);
            bool control_present = false;
            for (std::size_t j = 0; j < width; ++j) {
                bool valid = std::int64_t(j) < active
                    && std::int64_t(j) >= batch.cursors[row].skipped_ffts;
                if (event) {
                    valid &= I(cursor)
                            + I(j) * g.fft_samples * g.ratio_denominator
                                / g.ratio_numerator
                        < extent.allocation_bytes;
                }
                eligible[j] = valid;
                if (j < std::size_t(control) && valid) {
                    control_present = true;
                }
            }
            if (event && !control_present) {
                std::fill(eligible.begin(), eligible.end(), 0);
            }
            transfer = event
                && std::none_of(
                    eligible.begin(), eligible.end(), [](auto x) { return x; });
            if (transfer) {
                ++allocation;
                if (std::size_t(allocation) >= state.anchors.size()) {
                    throw RadioError("successor window lies outside history");
                }
                cursor = 0;
                head = state.anchors[allocation];
                auto c = request_cursor(
                    checked(I(fact.desired_sample) - head),
                    fact.endpoint_drift,
                    g,
                    inputs.sample_rate);
                for (std::size_t j = 0; j < width; ++j) {
                    eligible[j] = c.valid && std::int64_t(j) < active
                        && std::int64_t(j) >= c.skipped_ffts;
                }
            }
        }
        const auto physical_head = initial
            ? 0
            : checked(
                  I(timeline.allocation_starts().at(allocation)) * spf
                  + I(cursor / g.ratio_denominator) * g.ratio_numerator);
        for (std::int64_t j = 0; j < active; ++j) {
            auto fft = checked(I(fact.first_fft) + j);
            if (fft < inputs.first_fft || fft >= inputs.stop_fft) {
                continue;
            }
            ReturnedWindow window{fft,
                inputs.output_samples[row][j],
                head,
                physical_head,
                inputs.centre_delays[row][j],
                extent.framed_admission_samples,
                extent.decoded_samples,
                bool(eligible[j]),
                initial};
            if (!initial && !event && j == 0 && head == original_head) {
                auto replacement = leading_head(
                    window.output,
                    inputs.left_delays.at(row),
                    rate,
                    spf,
                    quantum);
                window.physical_head =
                    checked(I(physical_head) + replacement - head);
                window.logical_head = replacement;
                window.clamp_negative = true;
            }
            auto bounds = window_bounds(
                window.output,
                window.logical_head,
                window.centre_delay,
                rate,
                window.framed_extent,
                window.decoded_extent,
                g.fft_samples);
            auto physical = checked(
                I(window.physical_head)
                + std::max<std::int64_t>(bounds.nearest, 0));
            const bool unchanged =
                !initial && window.logical_head == original_head;
            if (!initial && event && unchanged && !bounds.before
                && !bounds.after) {
                auto framehead = timeline.allocation_starts().at(selected);
                auto r = next_run(framehead);
                bool covered = r != runs.end() && r->first <= framehead;
                auto boundary = checked((I(framehead) + capacity) * spf);
                bool withheld = (j == 0 || !covered) && physical >= boundary;
                auto framed_samples = checked(
                    (I(extent.allocation_bytes) * g.ratio_numerator
                        + g.ratio_denominator - 1)
                    / g.ratio_denominator);
                auto allowance = checked(I(framed_samples) - I(capacity) * spf);
                if (!withheld && window.published
                    && physical >= checked(I(boundary) + allowance)
                    && std::size_t(selected + 1) < state.anchors.size()) {
                    withheld = I(state.anchors[selected + 1])
                        > I(state.anchors[selected]) + I(capacity) * spf;
                }
                window.published &= !withheld;
            }
            // Initial prefix publication is a separate contract and does not
            // inherit the normal-return allocation-head gate.
            if (!initial) {
                bool published =
                    window.published && !bounds.before && !bounds.after;
                const auto end = checked(I(physical) + g.fft_samples);
                auto r = next_run(physical / spf);
                bool covered = r != runs.end() && I(r->first) * spf < end;
                bool empty = published && !covered;
                double units = std::trunc(
                    window.centre_delay * (1000000000. / double(quantum)));
                auto ownbounds = window_bounds(
                    window.output,
                    window.logical_head,
                    units * (double(quantum) / 1000000000.),
                    rate,
                    window.framed_extent,
                    window.decoded_extent,
                    g.fft_samples);
                auto ownership = checked(
                    I(window.physical_head)
                    + std::max<std::int64_t>(ownbounds.nearest, 0));
                auto owner = ownership / spf / capacity;
                bool withheld = in_hole(owner);
                if (withheld && published) {
                    auto hole_head = checked(I(owner) * capacity),
                         head_sample = checked(I(hole_head) * spf);
                    auto resume = next_run(hole_head);
                    if (resume == runs.end() || resume == runs.begin()) {
                        throw RadioError(
                            "uncovered allocation has no trustworthy successor");
                    }
                    const auto& previous = *(resume - 1);
                    I residual = I(resume->timestamp) - previous.timestamp
                        - (I(resume->first) - previous.first);
                    auto decoded_stop = checked(
                        I(window.physical_head) + window.decoded_extent);
                    bool retained = residual == 0
                        && !in_hole(window.physical_head / spf / capacity)
                        && window.physical_head < head_sample
                        && physical < checked(I(head_sample) + header_prefix)
                        && std::max(physical, checked(I(resume->first) * spf))
                            < std::min(end, decoded_stop);
                    if (retained) {
                        withheld = false;
                    }
                }
                window.published &= !withheld && !empty;
            }
            if (window.physical_head < 0) {
                throw RadioError("negative retained physical head");
            }
            window.decoded_extent = std::min(
                window.decoded_extent,
                std::max<std::int64_t>(
                    checked(I(source_stop) - window.physical_head), 0));
            result.push_back(window);
        }
    }
    if (result.size() != std::size_t(inputs.stop_fft - inputs.first_fft)) {
        throw RadioError("returned requests do not cover selected scope");
    }
    for (std::size_t k = 0; k < result.size(); ++k) {
        if (result[k].fft != inputs.first_fft + std::int64_t(k)) {
            throw RadioError("returned requests are not contiguous");
        }
    }
    return result;
}
}  // namespace xbeam
namespace xbeam {
std::vector<ReplayRow> compile_replay_rows(
    const std::vector<ReturnedWindow>& windows,
    std::int64_t fft,
    std::int64_t rate,
    std::int64_t stop) {
    if (fft <= 0 || rate <= 0 || stop < 0) {
        throw RadioError("invalid replay source extent");
    }
    std::vector<ReplayRow> rows;
    rows.reserve(windows.size());
    for (const auto& w : windows) {
        auto bounds = window_bounds(
            w.output,
            w.logical_head,
            w.centre_delay,
            rate,
            w.framed_extent,
            w.decoded_extent,
            fft);
        auto physical = checked(
            I(w.physical_head) + std::max<std::int64_t>(bounds.nearest, 0));
        if (w.physical_head < 0
            || (I(w.physical_head) + w.decoded_extent > stop
                && w.decoded_extent > 0)) {
            throw RadioError("retained window exceeds declared source end");
        }
        bool accepted =
            w.published && !bounds.before && !bounds.after && physical < stop;
        if (w.clamp_negative && bounds.nearest < 0) {
            bounds = window_bounds(
                w.logical_head,
                w.logical_head,
                0,
                rate,
                w.framed_extent,
                w.decoded_extent,
                fft);
        }
        const auto logical = checked(
            I(w.logical_head)
            + std::max<std::int64_t>(
                window_bounds(
                    w.output,
                    w.logical_head,
                    w.centre_delay,
                    rate,
                    w.framed_extent,
                    w.decoded_extent,
                    fft)
                    .nearest,
                0));
        double coarse = std::nearbyint(w.centre_delay * double(rate));
        if (!std::isfinite(coarse) || coarse < -std::ldexp(1., 63)
            || coarse >= std::ldexp(1., 63)) {
            throw RadioError("replay centre delay exceeds int64 samples");
        }
        auto displacement =
            checked(I(logical) - w.output + static_cast<std::int64_t>(coarse));
        rows.push_back(
            {w.fft,
                physical,
                displacement,
                accepted ? bounds.valid_first : 0,
                accepted ? bounds.valid_stop : 0,
                accepted});
    }
    return rows;
}
}  // namespace xbeam

namespace xbeam {
std::int64_t reference_epoch_seconds(int epoch) {
    if (epoch < 0 || epoch > 63) {
        throw RadioError("VDIF reference epoch is outside its six-bit range");
    }
    const int year = 2000 + epoch / 2;
    std::int64_t days = 0;
    auto leap = [](int y) {
        return y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
    };
    for (int y = 2000; y < year; ++y) {
        days += 365 + leap(y);
    }
    if (epoch % 2) {
        days += 181 + leap(year);
    }
    // UTC leap transitions are part of the shared VDIF time convention.
    const int transitions[] = {12, 18, 25, 31, 34};
    int leaps = 0;
    for (int t : transitions) {
        leaps += epoch >= t;
    }
    return days * 86400 + leaps;
}
std::int64_t packed_frame_slot(std::uint64_t stamp, std::int64_t fps) {
    if (stamp >= (std::uint64_t(1) << 60) || fps <= 0 || fps > (1 << 24)) {
        throw RadioError("invalid packed VDIF timestamp");
    }
    auto frame = stamp & 0xffffff;
    if (frame >= std::uint64_t(fps)) {
        throw RadioError("VDIF frame number exceeds rate");
    }
    auto seconds = (stamp >> 24) & 0x3fffffff;
    return checked(
        (I(reference_epoch_seconds(stamp >> 54)) + seconds) * fps + frame);
}
}  // namespace xbeam
