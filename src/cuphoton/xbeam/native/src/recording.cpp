// SPDX-License-Identifier: Apache-2.0
#include "xbeam/recording.hpp"
#include <algorithm>
#include <cmath>
#include <nlohmann/json.hpp>
namespace xbeam {
namespace {
using J = nlohmann::json;
J recording_document(const RadioJob& job) {
    auto document = J::parse(job.normalized_json());
    if (document.at("buffered_reader").is_null()) {
        throw RadioError(
            "native replay requires DATA BUFFER FACTOR, NUM DATA SEGMENTS "
            "and NUM BUFFERED FFTS in the job metadata");
    }
    return document;
}
const J& by_id(const J& array, int id) {
    for (const auto& row : array) {
        if (row.at("identifier") == id) {
            return row;
        }
    }
    throw RadioError("unknown native recording identifier");
}
std::int64_t checked(__int128 value) {
    if (value < INT64_MIN || value > INT64_MAX) {
        throw RadioError("recording coordinate exceeds int64");
    }
    return static_cast<std::int64_t>(value);
}
std::string source_path(const J& stream) {
    if (stream.at("source_files").size() != 1) {
        throw RadioError("raw replay requires one physical source per stream");
    }
    return stream.at("source_files").at(0);
}
}  // namespace
struct RadioRecording::Impl {
    const RadioJob& job;
    int stream_id;
    J document, stream, metadata;
    IndexReader index;
    RecordingGeometry g{};
    std::unique_ptr<PhysicalTimeline> timeline;
    RawRequestState state;
    Impl(
        const RadioJob& input,
        int id,
        const std::string& sidecar,
        IndexReader* validated)
        : job(input),
          stream_id(id),
          document(recording_document(input)),
          stream(by_id(document.at("datastreams"), id)),
          index(
              validated ? std::move(*validated)
                        : IndexReader(source_path(stream), sidecar)) {
        metadata = J::parse(index.metadata_json());
        if (metadata.at("threads").size() != 1
            || stream.at("sampling") != "real") {
            throw RadioError(
                "buffered raw replay requires one real-sampled physical thread");
        }
        const auto& thread = metadata.at("threads").at(0);
        if (thread.at("complex_data").get<bool>()
            || thread.at("bits_per_sample") != stream.at("quantization_bits")
            || thread.at("frame_size_bytes") != stream.at("frame_size_bytes")) {
            throw RadioError("sidecar and recording sample geometry differ");
        }
        g.thread_id = thread.at("thread_id");
        g.channel_count = thread.at("channel_count");
        g.samples_per_frame = thread.at("samples_per_frame");
        g.frame_bytes = stream.at("frame_size_bytes");
        g.payload_bytes = g.frame_bytes - 32;
        g.frame_count = metadata.at("frame_count");
        double rate = stream.at("sample_rate_hz");
        if (!std::isfinite(rate) || rate <= 0 || rate >= std::ldexp(1., 63)
            || rate != std::trunc(rate)) {
            throw RadioError(
                "raw replay requires an integral positive sample rate");
        }
        g.sample_rate = static_cast<std::int64_t>(rate);
        if (g.samples_per_frame <= 0 || g.sample_rate % g.samples_per_frame
            || g.channel_count <= 0 || stream.at("quantization_bits") != 2) {
            throw RadioError("inconsistent native recording geometry");
        }
        g.frames_per_second = g.sample_rate / g.samples_per_frame;
        auto packed = index.physical_runs();
        if (packed.empty()) {
            throw RadioError("physical timestamp history is empty");
        }
        g.job_slot = job.frame_slot(
            g.frames_per_second, packed.front().packed_timestamp >> 54);
        const auto& bands = stream.at("recorded_bands");
        if (bands.empty()) {
            throw RadioError("recording declares no bands");
        }
        int logical = bands.at(0).at("thread_id");
        for (const auto& band : bands) {
            if (band.at("thread_id") != logical) {
                throw RadioError(
                    "single physical recording cannot alias multiple "
                    "requested threads");
            }
        }
        if (bands.size() != std::size_t(g.channel_count)) {
            throw RadioError(
                "recorded band count differs from packed channels");
        }
        auto rational = job.subintegration_ratio();
        auto fft = checked(
            __int128(by_id(
                document.at("frequencies"), bands.at(0).at("frequency_id"))
                    .at("channel_count")
                    .get<std::int64_t>())
            * 2);
        const auto& reader = document.at("buffered_reader");
        g.windows = size_windows(
            g.samples_per_frame,
            fft,
            g.sample_rate,
            rational.first,
            rational.second,
            reader.at("buffer_factor"),
            reader.at("data_segments"),
            reader.at("buffered_ffts"),
            0);
        auto den = checked(__int128(2) * g.channel_count);
        // Reduce the packing ratio before cursor divisibility checks.
        std::int64_t num = 8;
        auto a = num, b = den;
        while (b) {
            auto r = a % b;
            a = b;
            b = r;
        }
        num /= a;
        den /= a;
        g.request = {num, den, fft, g.windows.ffts_per_request, 8192, 4};
        g.extent = normal_extent(
            g.frame_bytes,
            g.payload_bytes,
            g.windows.request_frames,
            g.windows.read_buffer_ratio,
            num,
            den);
        std::vector<PhysicalRun> runs;
        for (const auto& run : packed) {
            runs.push_back(
                {checked(run.first),
                    checked(run.stop),
                    packed_frame_slot(
                        run.packed_timestamp, g.frames_per_second)});
        }
        timeline = std::make_unique<PhysicalTimeline>(
            runs, g.windows.allocation_frames);
        state =
            raw_request_state(*timeline, g.samples_per_frame, g.job_slot, true);
    }
};
RadioRecording::RadioRecording(
    const RadioJob& job, int stream, const std::string& index)
    : impl(std::make_unique<Impl>(job, stream, index, nullptr)) {}
RadioRecording::RadioRecording(
    const RadioJob& job, int stream, IndexReader&& index)
    : impl(std::make_unique<Impl>(job, stream, std::string(), &index)) {}
RadioRecording::~RadioRecording() = default;
const RecordingGeometry& RadioRecording::geometry() const {
    return impl->g;
}
RadioReplayBatch RadioRecording::plan(
    const std::string& scan,
    int band,
    std::int64_t first,
    std::int64_t stop,
    std::int64_t count) const {
    const auto& g = impl->g;
    auto requests = impl->job.requests(
        scan,
        impl->stream_id,
        band,
        g.request,
        first,
        stop,
        count,
        g.job_slot,
        g.job_slot,
        false,
        256);
    auto windows = returned_windows(
        *impl->timeline,
        impl->state,
        requests,
        g.request,
        g.extent,
        30,
        1,
        checked(__int128(g.frame_count) * g.samples_per_frame));
    auto rows = compile_replay_rows(
        windows,
        g.request.fft_samples,
        g.sample_rate,
        checked(__int128(g.frame_count) * g.samples_per_frame));
    const J* selected_scan = nullptr;
    for (const auto& s : impl->document.at("scans")) {
        if (s.at("identifier") == scan) {
            selected_scan = &s;
        }
    }
    if (!selected_scan) {
        throw RadioError("unknown recording scan");
    }
    double origin = std::max(
        0., std::stod(selected_scan->at("start_offset_s").get<std::string>()));
    std::vector<double> starts;
    for (const auto& row : rows) {
        starts.push_back(
            origin
            + double(row.fft)
                * (double(g.request.fft_samples) / double(g.sample_rate)));
    }
    auto blocks = impl->job.source_blocks(
        scan, impl->stream_id, band, starts.data(), starts.size());
    int channel = -1;
    const auto& bands = impl->stream.at("recorded_bands");
    for (std::size_t k = 0; k < bands.size(); ++k) {
        if (bands[k].at("identifier") == band) {
            channel = k;
        }
    }
    if (channel < 0) {
        throw RadioError("unknown packed channel");
    }
    return {std::move(blocks), std::move(rows), channel};
}

}  // namespace xbeam

namespace xbeam {
std::vector<unsigned char> RadioRecording::frame_validity(
    std::int64_t first, std::int64_t count) const {
    if (first < 0 || count < 0) {
        throw RadioError("negative physical validity scope");
    }
    return impl->index.frame_validity(first, count);
}
}  // namespace xbeam
