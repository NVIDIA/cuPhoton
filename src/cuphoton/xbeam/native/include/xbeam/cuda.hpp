// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <memory>
namespace xbeam {
struct StationLayout {
    int rows, fft_samples, channels, mapping;
    int validity_stride, samples_per_frame, payload_bytes, channel_count,
        channel_index;
    bool complex_input = false, authoritative_validity = false;
    std::int64_t source_samples = 0, source_start = 0;
    double sample_rate = 0, reference_hz = 0, lo_hz = 0, wall_lo_cycles = 0;
};
// Binary-compatible with the existing frontend's per-FFT metadata.
struct BlockParameters {
    std::int64_t output_index, coarse_sample;
    double block_start, delay_zero, delay_step, delay_quadratic, residual;
    std::int64_t source_offset, source_cutover, valid_start, valid_stop;
};
static_assert(sizeof(BlockParameters) == 88, "block parameter ABI changed");
class CudaStation {
public:
    explicit CudaStation(const StationLayout&);
    ~CudaStation();
    CudaStation(const CudaStation&) = delete;
    CudaStation& operator=(const CudaStation&) = delete;
    // Arguments are device addresses; caller retains them until stream completion.
    void submit(
        std::uintptr_t source,
        std::uintptr_t validity,
        std::uintptr_t parameters);
    void download(float* interleaved, double* weights);
    void source_window(std::int64_t first, std::int64_t count);
    std::uintptr_t stream() const;
    std::uintptr_t spectra() const;
    std::uintptr_t weights() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
}  // namespace xbeam
namespace xbeam {
struct ProductLayout {
    int channels, averaging, groups;
    std::int64_t stride_a, stride_b;
    double inverse_normalization;
    bool multiply_weights;
};
void submit_product(
    const ProductLayout&,
    std::uintptr_t spectrum_a,
    std::uintptr_t spectrum_b,
    std::uintptr_t weight_a,
    std::uintptr_t weight_b,
    std::uintptr_t group_starts,
    std::uintptr_t group_lengths,
    std::uintptr_t output_offsets,
    std::uintptr_t weight_offsets,
    std::uintptr_t output,
    std::uintptr_t weight_sum,
    std::uintptr_t stream);
}  // namespace xbeam

#include <string>
#include <vector>
namespace xbeam {
struct NativeReadChunk {
    std::uint64_t generation;
    std::size_t index, offset, size;
};
// How a recording is read. Buffered and Direct bounce through pinned host memory
// (Direct opens with O_DIRECT, bypassing the page cache); GpuDirect uses cuFile.
enum class ReadMode { Buffered, Direct, GpuDirect };
class CudaRecordingInput {
public:
    CudaRecordingInput(
        const std::string& source,
        std::int64_t frame_bytes,
        std::int64_t payload_bytes,
        ReadMode mode,
        std::size_t chunk_bytes = 16 * 1024 * 1024,
        bool shared_staging = false);
    ~CudaRecordingInput();
    std::vector<NativeReadChunk> begin(
        std::int64_t first,
        std::int64_t stop,
        const std::vector<unsigned char>& validity);
    void read_chunk(const NativeReadChunk&);
    void finish();
    void read(
        std::int64_t first,
        std::int64_t stop,
        const std::vector<unsigned char>& validity);
    std::uintptr_t payload() const;
    std::uintptr_t validity() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
void unpack_frames(
    const void* raw,
    void* payload,
    std::int64_t frames,
    std::int64_t frame_bytes,
    std::int64_t payload_bytes,
    std::int64_t prefix,
    std::uintptr_t stream);
void unpack_frame_chunk(
    const void* raw,
    void* payload,
    std::int64_t first_frame,
    std::int64_t frames,
    std::int64_t frame_bytes,
    std::int64_t payload_bytes,
    std::int64_t file_offset,
    std::int64_t read_bytes,
    std::uintptr_t stream);
}  // namespace xbeam

namespace xbeam {
struct VisibilityProduct {
    std::uintptr_t spectrum_a, spectrum_b, weight_a, weight_b;
    std::int64_t stride_a, stride_b;
    int channels, averaging;
    double inverse_normalization;
    bool multiply_weights;
};
class CudaProducts {
public:
    explicit CudaProducts(int products_per_pack = 4, bool reuse_sources = true);
    ~CudaProducts();
    void configure(const std::vector<VisibilityProduct>&);
    void submit(
        int groups,
        std::uintptr_t group_starts,
        std::uintptr_t group_lengths,
        std::uintptr_t output_offsets,
        std::uintptr_t weight_offsets,
        std::uintptr_t output,
        std::uintptr_t weights,
        std::uintptr_t stream);

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
}  // namespace xbeam
