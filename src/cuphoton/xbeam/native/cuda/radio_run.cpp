// SPDX-License-Identifier: Apache-2.0
// Native radio execution: no Python or CuPy runtime calls.
#include "gpu.hpp"
#include "radio_model.hpp"
#include "radio_options.hpp"
#include "xbeam/cuda.hpp"
#include "xbeam/execution.hpp"
#include "xbeam/recording.hpp"
#include "xbeam/work_queue.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <map>
#include <set>
#include <system_error>
#include <vector>
namespace xbeam {
namespace {
using gpu::check;
using Clock = std::chrono::steady_clock;
using Extent =
    std::pair<std::int64_t, std::int64_t>;  // frame range [first, stop)

int select_device(int device) {
    check(cudaSetDevice(device));
    return device;
}
double since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

// One station pipeline: a recorded band of one datastream.
struct Parent {
    RadioReplayBatch plan;
    StationLayout layout{};
    std::unique_ptr<CudaStation> executor;
    gpu::Memory parameters;
    gpu::Pinned parameters_host;
    gpu::Event done;
};
// One physical recording, double-buffered so the next tile reads during compute.
struct Input {
    std::unique_ptr<RadioRecording> recording;
    std::unique_ptr<CudaRecordingInput> delivery[2];
};
struct InputTiming {
    double index = 0, recording = 0, delivery = 0;
};

// A tile of FFT blocks whose reads are in flight.
struct Batch {
    std::int64_t first = 0, stop = 0;
    int slot = 0;
    std::map<Key, RadioReplayBatch> plans;
    std::map<int, Extent> extents;
    std::vector<std::future<void>> deliveries;
    // Streams without an accepted block still get a one-frame read.
    Extent extent(int stream) const {
        auto found = extents.find(stream);
        return found == extents.end() ? Extent{0, 1} : found->second;
    }
};

// Guided self-scheduling: large chunks first, small chunks at the tail.
std::vector<Extent> guided_chunks(
    std::int64_t first,
    std::int64_t count,
    int workers,
    std::int64_t min_chunk) {
    std::vector<Extent> table;
    for (std::int64_t position = first, remaining = count; remaining > 0;) {
        const auto size = std::min(
            remaining,
            std::max(min_chunk, (remaining + 2 * workers - 1) / (2 * workers)));
        table.emplace_back(position, size);
        position += size;
        remaining -= size;
    }
    return table;
}

// Rethrow the first failure only after every task finished.
template <class Futures>
void join_all(Futures& futures) {
    std::exception_ptr failure;
    for (auto& future : futures) {
        try {
            future.get();
        } catch (...) {
            if (!failure) {
                failure = std::current_exception();
            }
        }
    }
    if (failure) {
        std::rethrow_exception(failure);
    }
}

class Run {
public:
    explicit Run(Options options);
    Json execute();

private:
    void load_scope();
    void open_inputs();
    void derive_time_grid();
    void allocate_device_state();

    Batch prepare(std::int64_t first, std::int64_t stop, int slot);
    void await_delivery(Batch& batch);
    void submit_station(
        const Key& key, Parent& parent, const Batch& batch, int rows);
    void accumulate_products(
        int rows, std::vector<float>& sums, std::vector<double>& weights);
    void integrate(std::int64_t index);
    void write_records(
        std::int64_t index,
        std::vector<float>& sums,
        const std::vector<double>& weights);
    void run_range(std::int64_t first, std::int64_t count);
    Json run_chunks();
    Json report(Json chunks) const;

    const Options opts_;
    // The clock includes CUDA context creation, as in every earlier report.
    const Clock::time_point started_ = Clock::now();
    // Must precede every GPU resource below, which binds to the current device.
    const int device_ = select_device(opts_.device);
    RadioJob native_;
    Json job_;

    // Scan geometry.
    double integration_ = 0, origin_ = 0, end_ = 0, fft_seconds_ = 0;
    std::int64_t width_ = 0, per_integration_ = 0, timeline_ = 0;
    std::vector<Product> specs_;
    std::set<Key> keys_;
    std::size_t elements_ = 0;  // output channels across all products

    // Declared before the workers and GPU state, so they are destroyed last.
    std::map<int, Input> inputs_;
    std::map<Key, Parent> parents_;
    std::map<int, InputTiming> input_timing_;

    std::ofstream file_;
    gpu::Memory output_, weights_, starts_, lengths_, offsets_, weight_offsets_;
    WorkQueue workers_;
    // Declared before the products so it outlives their destructor.
    gpu::Stream xmac_stream_;
    CudaProducts products_;
    gpu::Pinned host_output_, host_exposure_;
    int uploaded_rows_ = -1;

    std::uint64_t records_ = 0;
    double job_seconds_ = 0, setup_seconds_ = 0;
    double plan_seconds_ = 0, read_seconds_ = 0, front_seconds_ = 0,
           xmac_seconds_ = 0;
};

Run::Run(Options options)
    : opts_(std::move(options)),
      native_(opts_.job),
      job_(Json::parse(native_.normalized_json())),
      workers_(opts_.upload_workers),
      products_(opts_.xmac_pack_products, opts_.xmac_reuse_sources) {
    load_scope();
    job_seconds_ = since(started_);
    open_inputs();
    derive_time_grid();
    allocate_device_state();
    setup_seconds_ = since(started_);
}

void Run::load_scope() {
    const Json* scan = nullptr;
    for (const auto& candidate : job_.at("scans")) {
        if (candidate.at("identifier") == opts_.scan) {
            scan = &candidate;
        }
    }
    if (!scan) {
        throw RadioError("unknown execution scan");
    }
    auto seconds = [](const Json& value) {
        return std::stod(value.get<std::string>());
    };
    integration_ = seconds(job_.at("integration_duration_s"));
    const double subintegration = seconds(job_.at("subintegration_duration_s")),
                 start_offset = seconds(scan->at("start_offset_s")),
                 execution = seconds(job_.at("execution_duration_s")),
                 fractional_start =
                     seconds(job_.at("start_time").at("seconds_of_day"));
    origin_ = std::max(0., start_offset);
    end_ = std::min(execution, start_offset + seconds(scan->at("duration_s")));
    end_ = std::min(
        end_, execution - (fractional_start - std::floor(fractional_start)));
    if (opts_.first_integration < 0 || opts_.integration_count <= 0
        || opts_.fft_tile < 8 || opts_.fft_tile > 1600 || end_ <= origin_
        || integration_ <= 0 || subintegration <= 0) {
        throw RadioError("invalid native correlation scope");
    }
    specs_ = products(job_);
    if (specs_.empty()) {
        throw RadioError("job has no visibility products");
    }
    for (const auto& p : specs_) {
        keys_.insert(p.a.parent);
        keys_.insert(p.b.parent);
        elements_ += p.channels / p.average;
    }
}

// Each recording's sidecar validation, timeline construction and cuFile
// registration is independent. Build them concurrently, one task per file.
void Run::open_inputs() {
    std::vector<int> streams;
    for (const auto& key : keys_) {
        if (streams.empty() || streams.back() != key.first) {
            streams.push_back(key.first);
        }
    }
    for (int stream : streams) {
        inputs_[stream];
        input_timing_[stream];
    }
    std::vector<std::future<void>> tasks;
    for (int stream : streams) {
        tasks.push_back(std::async(std::launch::async, [this, stream] {
            check(cudaSetDevice(device_));
            auto& input = inputs_.at(stream);
            auto& timing = input_timing_.at(stream);
            const auto& datastream = by_id(job_.at("datastreams"), stream);
            if (datastream.at("source_files").size() != 1) {
                throw RadioError(
                    "native replay requires one physical recording");
            }
            const std::string source = datastream.at("source_files").at(0);
            auto phase = Clock::now();
            auto index = open_index(source, opts_.index_dir);
            timing.index = since(phase);
            phase = Clock::now();
            input.recording = std::make_unique<RadioRecording>(
                native_, stream, std::move(*index));
            timing.recording = since(phase);
            phase = Clock::now();
            const auto& geometry = input.recording->geometry();
            for (int slot = 0; slot < (opts_.prefetch ? 2 : 1); ++slot) {
                input.delivery[slot] = std::make_unique<CudaRecordingInput>(
                    source,
                    geometry.frame_bytes,
                    geometry.payload_bytes,
                    opts_.read_mode,
                    opts_.chunk_bytes,
                    opts_.shared_staging);
            }
            timing.delivery = since(phase);
        }));
    }
    join_all(tasks);
}

void Run::derive_time_grid() {
    for (const auto& key : keys_) {
        const auto& g = inputs_.at(key.first).recording->geometry();
        const double duration =
            double(g.request.fft_samples) / double(g.sample_rate);
        if (fft_seconds_ && duration != fft_seconds_) {
            throw RadioError(
                "native execution requires a shared FFT time grid");
        }
        fft_seconds_ = duration;
        width_ = g.request.ffts_per_request;
        parents_.try_emplace(key);
    }
    if (opts_.fft_tile % width_) {
        throw RadioError("FFT tile must contain whole subintegrations");
    }
    per_integration_ = std::llround(integration_ / fft_seconds_);
    if (std::abs(per_integration_ * fft_seconds_ - integration_) > 1e-12
        || per_integration_ % width_) {
        throw RadioError(
            "native integration must contain whole reader requests");
    }
    const double nominal = std::ceil((end_ - origin_) / fft_seconds_);
    if (!std::isfinite(nominal) || nominal < 1
        || nominal >= std::ldexp(1., 63)) {
        throw RadioError("native FFT timeline exceeds int64");
    }
    timeline_ = static_cast<std::int64_t>(nominal);
    if ((__int128(opts_.first_integration) + opts_.integration_count)
            * per_integration_
        > timeline_) {
        throw RadioError("selected integrations exceed scan");
    }
}

void Run::allocate_device_state() {
    if (!std::filesystem::create_directory(opts_.output)) {
        throw RadioError("native output directory must be new");
    }
    if (!opts_.dynamic()) {
        file_.open(
            std::filesystem::path(opts_.output) / "DIFX_native.s0000.b0000",
            std::ios::binary | std::ios::out);
        if (!file_) {
            throw RadioError("cannot create native output");
        }
    }
    output_.reserve(elements_ * 8);
    weights_.reserve(specs_.size() * 8);
    starts_.upload(std::vector<int>{0});
    std::vector<long long> offsets, weight_offsets;
    for (std::size_t i = 0; i < specs_.size(); ++i) {
        offsets.push_back(specs_[i].offset);
        weight_offsets.push_back(i);
    }
    offsets_.upload(offsets);
    weight_offsets_.upload(weight_offsets);
    host_output_.reserve(elements_ * 8);
    host_exposure_.reserve(specs_.size() * 8);
}

// Plan one tile and start reading its frames into delivery slot `slot`.
Batch Run::prepare(std::int64_t first, std::int64_t stop, int slot) {
    auto phase = Clock::now();
    Batch batch;
    batch.first = first;
    batch.stop = stop;
    batch.slot = slot;
    for (const auto& entry : parents_) {
        const auto key = entry.first;
        auto& recording = *inputs_.at(key.first).recording;
        auto plan =
            recording.plan(opts_.scan, key.second, first, stop, timeline_);
        const auto& g = recording.geometry();
        for (const auto& r : plan.replay) {
            if (r.accepted) {
                const auto left = r.physical_start / g.samples_per_frame;
                const auto right = std::min(
                    g.frame_count,
                    (r.physical_start + g.request.fft_samples
                        + g.samples_per_frame - 1)
                        / g.samples_per_frame);
                auto [it, inserted] =
                    batch.extents.try_emplace(key.first, Extent{left, right});
                if (!inserted) {
                    it->second.first = std::min(it->second.first, left);
                    it->second.second = std::max(it->second.second, right);
                }
            }
        }
        batch.plans.emplace(key, std::move(plan));
    }
    plan_seconds_ += since(phase);
    for (auto& entry : inputs_) {
        auto* input = &entry.second;
        const auto range = batch.extent(entry.first);
        const auto validity = input->recording->frame_validity(
            range.first, range.second - range.first);
        for (const auto& chunk :
            input->delivery[slot]->begin(range.first, range.second, validity)) {
            batch.deliveries.push_back(
                workers_.submit([input, slot, chunk, device = device_] {
                    check(cudaSetDevice(device));
                    input->delivery[slot]->read_chunk(chunk);
                }));
        }
    }
    return batch;
}

void Run::await_delivery(Batch& batch) {
    const auto phase = Clock::now();
    join_all(batch.deliveries);
    for (auto& entry : inputs_) {
        entry.second.delivery[batch.slot]->finish();
        entry.second.recording->frame_validity(
            batch.extent(entry.first).first, 0);
    }
    read_seconds_ += since(phase);
}

// Build the block parameters, (re)create the station executor and enqueue the
// station's FFT pipeline on its own stream.
void Run::submit_station(
    const Key& key, Parent& parent, const Batch& batch, int rows) {
    auto& input = inputs_.at(key.first);
    const auto& g = input.recording->geometry();
    const auto range = batch.extent(key.first);
    const auto &datastream = by_id(job_.at("datastreams"), key.first),
               &band = by_id(datastream.at("recorded_bands"), key.second),
               &frequency =
                   by_id(job_.at("frequencies"), band.at("frequency_id"));
    std::vector<BlockParameters> parameters;
    for (std::size_t k = 0; k < parent.plan.blocks.size(); ++k) {
        const auto& b = parent.plan.blocks[k];
        const auto& r = parent.plan.replay[k];
        parameters.push_back(
            {b.output_sample,
                b.coarse_sample,
                b.block_start,
                b.delay_zero,
                b.delay_step,
                b.delay_quadratic,
                (double(b.coarse_sample - r.timing_displacement)
                    / g.sample_rate)
                    - b.centre_delay,
                r.accepted
                    ? r.physical_start - (b.output_sample - b.coarse_sample)
                    : 0,
                0,
                r.valid_first,
                r.valid_stop});
    }
    auto& layout = parent.layout;
    if (!parent.executor || layout.rows != rows) {
        parent.executor.reset();
        layout = {};
        layout.rows = rows;
        layout.fft_samples = g.request.fft_samples;
        layout.channels = frequency.at("channel_count");
        layout.mapping = frequency.at("sideband") == "USB" ? 0 : 1;
        layout.validity_stride = g.samples_per_frame;
        layout.samples_per_frame = g.samples_per_frame;
        layout.payload_bytes = g.payload_bytes;
        layout.channel_count = g.channel_count;
        layout.channel_index = parent.plan.channel_index;
        layout.sample_rate = g.sample_rate;
        layout.reference_hz = frequency.at("band_edge_hz");
        layout.lo_hz = band.at("lo_offset_hz");
        // The declared LO offset uses job-relative block time; there is no
        // additional absolute wall-clock phase.
        layout.wall_lo_cycles = 0;
        layout.source_start = range.first * g.samples_per_frame;
        layout.source_samples =
            (range.second - range.first) * g.samples_per_frame;
        parent.executor = std::make_unique<CudaStation>(layout);
    }
    parent.executor->source_window(
        range.first * g.samples_per_frame,
        (range.second - range.first) * g.samples_per_frame);
    // Stream-ordered upload from pinned memory. The host buffer and device
    // parameters are reused only after the tile's stream synchronization.
    const auto bytes = parameters.size() * sizeof(BlockParameters);
    parent.parameters.reserve(bytes);
    parent.parameters_host.reserve(bytes);
    std::memcpy(parent.parameters_host.p, parameters.data(), bytes);
    check(cudaMemcpyAsync(
        parent.parameters.p,
        parent.parameters_host.p,
        bytes,
        cudaMemcpyHostToDevice,
        reinterpret_cast<cudaStream_t>(parent.executor->stream())));
    parent.executor->submit(
        input.delivery[batch.slot]->payload(),
        input.delivery[batch.slot]->validity(),
        parent.parameters.address());
}

// Cross-multiply the station spectra, download the tile's partial integration
// and add it into `sums` / `weights`.
void Run::accumulate_products(
    int rows, std::vector<float>& sums, std::vector<double>& weights) {
    if (rows != uploaded_rows_) {
        // The previous tile's consumers completed before this point.
        lengths_.upload(std::vector<int>{rows});
        uploaded_rows_ = rows;
    }
    const auto phase = Clock::now();
    std::vector<VisibilityProduct> visibility;
    for (const auto& p : specs_) {
        auto &a = parents_.at(p.a.parent), &b = parents_.at(p.b.parent);
        const bool parallel =
            p.automatic && p.a.parent == p.b.parent && p.a.start == p.b.start;
        const int channels = parallel ? p.channels / p.average : p.channels,
                  average = parallel ? 1 : p.average;
        visibility.push_back(
            {a.executor->spectra() + p.a.start * 8,
                b.executor->spectra() + p.b.start * 8,
                a.executor->weights(),
                b.executor->weights(),
                a.layout.channels,
                b.layout.channels,
                channels,
                average,
                1.
                    / std::sqrt(
                        double(a.layout.fft_samples) * b.layout.fft_samples),
                !parallel});
    }
    products_.configure(visibility);
    products_.submit(
        1,
        starts_.address(),
        lengths_.address(),
        offsets_.address(),
        weight_offsets_.address(),
        output_.address(),
        weights_.address(),
        reinterpret_cast<std::uintptr_t>(xmac_stream_.value));
    check(cudaMemcpyAsync(
        host_output_.p,
        output_.p,
        elements_ * 8,
        cudaMemcpyDeviceToHost,
        xmac_stream_.value));
    check(cudaMemcpyAsync(
        host_exposure_.p,
        weights_.p,
        specs_.size() * 8,
        cudaMemcpyDeviceToHost,
        xmac_stream_.value));
    check(cudaStreamSynchronize(xmac_stream_.value));
    xmac_seconds_ += since(phase);
    const float* host = host_output_.as<float>();
    const double* host_weights = host_exposure_.as<double>();
    for (std::size_t k = 0; k < elements_ * 2; ++k) {
        sums[k] += host[k];
    }
    for (std::size_t k = 0; k < weights.size(); ++k) {
        weights[k] += host_weights[k];
    }
}

// One integration: tiles are read ahead into the second delivery slot while the
// current tile computes.
void Run::integrate(std::int64_t index) {
    std::vector<float> sums(elements_ * 2, 0);
    std::vector<double> weights(specs_.size(), 0);
    const auto first = index * per_integration_,
               stop = first + per_integration_;
    const auto tile = opts_.fft_tile;
    auto current = prepare(first, std::min(stop, first + tile), 0);
    while (current.first < stop) {
        await_delivery(current);
        const int rows = current.stop - current.first;
        std::unique_ptr<Batch> next;
        if (opts_.prefetch && current.stop < stop) {
            next = std::make_unique<Batch>(prepare(
                current.stop,
                std::min(stop, current.stop + tile),
                1 - current.slot));
        }
        for (auto& entry : parents_) {
            entry.second.plan = std::move(current.plans.at(entry.first));
        }
        auto phase = Clock::now();
        for (auto& entry : parents_) {
            submit_station(entry.first, entry.second, current, rows);
        }
        // Order the product stream after every station stream on the GPU, with
        // one host wait after the products and output download. A device-wide
        // wait would also stall the next-batch delivery streams.
        for (auto& entry : parents_) {
            auto station =
                reinterpret_cast<cudaStream_t>(entry.second.executor->stream());
            check(cudaEventRecord(entry.second.done.value, station));
            check(cudaStreamWaitEvent(
                xmac_stream_.value, entry.second.done.value, 0));
        }
        front_seconds_ += since(phase);
        accumulate_products(rows, sums, weights);
        if (current.stop == stop) {
            break;
        }
        current = next
            ? std::move(*next)
            : prepare(current.stop, std::min(stop, current.stop + tile), 0);
    }
    write_records(index, sums, weights);
    std::cerr << "integration " << index << " records=" << records_ << '\n';
}

void Run::write_records(
    std::int64_t index,
    std::vector<float>& sums,
    const std::vector<double>& weights) {
    std::map<Key, SwinMetadata> metadata_cache;
    for (std::size_t k = 0; k < specs_.size(); ++k) {
        if (weights[k] <= 0) {
            continue;
        }
        const auto& p = specs_[k];
        const int channels = p.channels / p.average;
        for (int j = 0; j < 2 * channels; ++j) {
            sums[p.offset * 2 + j] =
                float(double(sums[p.offset * 2 + j]) / weights[k]);
        }
        const Key stations{p.a.station, p.b.station};
        auto found = metadata_cache.find(stations);
        if (found == metadata_cache.end()) {
            found = metadata_cache
                        .emplace(
                            stations,
                            native_.record_metadata(
                                opts_.scan,
                                index,
                                p.a.station,
                                p.b.station,
                                p.frequency,
                                p.a.polarization,
                                p.b.polarization))
                        .first;
        }
        auto metadata = found->second;
        metadata.frequency_id = p.frequency;
        metadata.polarization_a = p.a.polarization;
        metadata.polarization_b = p.b.polarization;
        metadata.weight = weights[k] * fft_seconds_ / integration_;
        const auto encoded =
            encode_swin(metadata, sums.data() + p.offset * 2, channels);
        file_.write(encoded.data(), encoded.size());
        if (!file_) {
            throw RadioError("native output write failed");
        }
        ++records_;
    }
}

void Run::run_range(std::int64_t first, std::int64_t count) {
    for (auto index = first; index < first + count; ++index) {
        integrate(index);
    }
}

// Dynamic mode: independent worker processes (possibly on several nodes) claim
// chunks of a shared, deterministic table with atomic directory creation, one
// output file per chunk.
Json Run::run_chunks() {
    const std::filesystem::path claims(opts_.claim_dir), output(opts_.output);
    if (!std::filesystem::is_directory(claims)) {
        throw RadioError("claim directory must already exist");
    }
    const auto table = guided_chunks(
        opts_.first_integration,
        opts_.integration_count,
        opts_.workers_total,
        opts_.min_chunk);
    Json reports = Json::array();
    for (std::size_t c = 0; c < table.size(); ++c) {
        char claim[64], name[64];
        std::snprintf(claim, sizeof claim, "chunk-%06zu", c);
        std::error_code ec;
        if (!std::filesystem::create_directory(claims / claim, ec)) {
            if (ec) {
                throw RadioError("cannot claim chunk: " + ec.message());
            }
            continue;  // another worker owns it
        }
        std::snprintf(name, sizeof name, "DIFX_native.chunk-%06zu", c);
        const auto began = Clock::now();
        const auto before = records_;
        file_.open(output / name, std::ios::binary | std::ios::out);
        if (!file_) {
            throw RadioError("cannot create native chunk output");
        }
        run_range(table[c].first, table[c].second);
        file_.close();
        if (!file_) {
            throw RadioError("native chunk close failed");
        }
        reports.push_back(
            {{"index", c},
                {"first_integration", table[c].first},
                {"integration_count", table[c].second},
                {"records", records_ - before},
                {"seconds", since(began)},
                {"file", std::string(name)}});
    }
    return reports;
}

Json Run::execute() {
    Json chunks = Json::array();
    if (opts_.dynamic()) {
        chunks = run_chunks();
    } else {
        run_range(opts_.first_integration, opts_.integration_count);
        file_.close();
        if (!file_) {
            throw RadioError("native output close failed");
        }
    }
    auto result = report(std::move(chunks));
    std::ofstream(std::filesystem::path(opts_.output) / "native-run.json")
        << result.dump(2) << '\n';
    return result;
}

Json Run::report(Json chunks) const {
    Json result = {{"backend", "native-cuda"},
        {"profile", opts_.profile},
        {"records", records_},
        {"seconds", since(started_)},
        {"first_integration", opts_.first_integration},
        {"integration_count", opts_.integration_count},
        {"fft_tile", opts_.fft_tile},
        {"io", io_name(opts_.read_mode)}};
    result["setup_seconds"] = setup_seconds_;
    result["dynamic"] = opts_.dynamic();
    result["chunks"] = std::move(chunks);
    result["upload_workers"] = opts_.upload_workers;
    result["read_chunk_bytes"] = opts_.chunk_bytes;
    result["read_staging"] = opts_.shared_staging ? "shared" : "per-input";
    result["xmac_pack_products"] = opts_.xmac_pack_products;
    result["xmac_reuse_sources"] = opts_.xmac_reuse_sources;
    result["prefetch"] = opts_.prefetch;
    result["read_measurement"] =
        "exposed delivery wait; prefetched IO can overlap compute";
    result["setup_job_seconds"] = job_seconds_;
    Json setup_inputs = Json::object();
    for (const auto& [stream, timing] : input_timing_) {
        setup_inputs[std::to_string(stream)] = {
            {"index_validation_seconds", timing.index},
            {"recording_seconds", timing.recording},
            {"delivery_open_seconds", timing.delivery}};
    }
    result["setup_inputs"] = setup_inputs;
    result["timing_note"] =
        "front_seconds is host submission; xmac_seconds includes the GPU wait";
    result["plan_seconds"] = plan_seconds_;
    result["read_seconds"] = read_seconds_;
    result["front_seconds"] = front_seconds_;
    result["xmac_seconds"] = xmac_seconds_;
    return result;
}
}  // namespace

std::string correlate_radio(const std::vector<std::string>& arguments) {
    for (const auto& argument : arguments) {
        if (argument.find('\0') != std::string::npos) {
            throw RadioError("native execution argument contains NUL");
        }
    }
    return Run(Options::parse(arguments)).execute().dump();
}
}  // namespace xbeam
