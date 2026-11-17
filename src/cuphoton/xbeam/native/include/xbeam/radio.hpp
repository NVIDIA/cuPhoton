// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
namespace xbeam {
class RadioError : public std::runtime_error {
    using std::runtime_error::runtime_error;
};
struct VdifHeader {
    bool invalid, complex_data;
    std::uint32_t seconds_from_epoch, reference_epoch, frame_number;
    std::uint32_t version, log2_channels, frame_length_bytes;
    std::uint32_t bits_per_sample, thread_id, station_id, extended_data_version;
    std::array<std::uint32_t, 4> extended_words;
    std::uint64_t samples_per_frame;
};
VdifHeader parse_vdif_header(
    const unsigned char*,
    std::size_t,
    bool strict = true,
    bool require_two_bit = true);
struct RecordingSummary {
    std::uint64_t frames = 0, invalid_frames = 0, source_bytes = 0;
    std::uint32_t frame_bytes = 0;
    std::array<std::uint64_t, 1024> thread_frames{};
};
RecordingSummary inspect_vdif(const std::string&, bool require_two_bit = true);
std::string inspect_vdif_json(const std::string&);
}  // namespace xbeam

namespace xbeam {
struct SwinMetadata {
    std::int32_t baseline, configuration_id, source_id, frequency_id;
    std::uint32_t mjd, pulsar_bin;
    double seconds, weight, u, v, w;
    char polarization_a, polarization_b;
};
std::string encode_swin(
    const SwinMetadata&, const float* interleaved, std::size_t channels);
void write_swin_file(const std::string& path, const std::string& records);
}  // namespace xbeam

namespace xbeam {
struct RequestGeometry {
    std::int64_t ratio_numerator, ratio_denominator, fft_samples;
    std::int64_t ffts_per_request, alignment_bytes, word_bytes;
};
struct RequestCursor {
    bool valid = false;
    std::int64_t origin_bytes = 0, skipped_ffts = 0, decoder_delta_samples = 0;
};
RequestCursor request_cursor(
    std::int64_t local_samples,
    double drift_seconds,
    const RequestGeometry&,
    double sample_rate);
}  // namespace xbeam

#include <vector>
namespace xbeam {
struct TextEntry {
    std::string key, value;
    std::size_t line_number;
};
std::vector<TextEntry> parse_job_text(const std::string& text);
}  // namespace xbeam
namespace xbeam {
std::string load_interop_job_json(const std::string& path);
}
#include <memory>
#include <utility>
namespace xbeam {
struct IndexFrame {
    std::uint64_t timestamp, physical_frame;
    std::uint16_t thread;
    std::uint8_t state;
};
struct PhysicalIndexRun {
    std::uint64_t first, stop, packed_timestamp;
};
class IndexReader {
public:
    IndexReader(const std::string& source, const std::string& index);
    ~IndexReader();
    IndexReader(const IndexReader&) = delete;
    IndexReader& operator=(const IndexReader&) = delete;
    IndexReader(IndexReader&&) noexcept;
    std::string metadata_json() const;
    std::vector<PhysicalIndexRun> physical_runs() const;
    std::vector<unsigned char> frame_validity(
        std::uint64_t first, std::uint64_t count) const;
    std::vector<IndexFrame> query(
        std::uint16_t thread, std::uint64_t first, std::uint64_t stop) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
}  // namespace xbeam
namespace xbeam {
struct SourceBlock {
    std::int64_t fft_samples, output_sample, coarse_sample;
    double centre_delay;
    double block_start = 0, delay_zero = 0, delay_step = 0, delay_quadratic = 0,
           residual = 0;
};
struct JobRequestInputs;
class RadioJob {
public:
    explicit RadioJob(const std::string& input);
    ~RadioJob();
    RadioJob(const RadioJob&) = delete;
    RadioJob& operator=(const RadioJob&) = delete;
    std::string normalized_json() const;
    std::int64_t frame_slot(
        std::int64_t frames_per_second, int reference_epoch) const;
    std::pair<std::int64_t, std::int64_t> subintegration_ratio() const;
    SwinMetadata record_metadata(
        const std::string& scan,
        std::int64_t integration,
        int station_a,
        int station_b,
        int frequency,
        char pol_a,
        char pol_b) const;
    std::vector<SourceBlock> source_blocks(
        const std::string& scan,
        int stream,
        int band,
        const double* starts,
        std::size_t count,
        bool literal_endpoint = false,
        bool phase_parameters = true) const;
    JobRequestInputs requests(
        const std::string& scan,
        int stream,
        int band,
        const RequestGeometry&,
        std::int64_t first,
        std::int64_t stop,
        std::int64_t timeline_count,
        std::int64_t job_start_slot,
        std::int64_t timestamp_origin,
        bool endpoint_mean,
        std::int64_t maximum_requests) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
}  // namespace xbeam
