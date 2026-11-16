// SPDX-License-Identifier: Apache-2.0
// Bounded real-recording frontend check. This is not a correlator.
#include "xbeam/cuda.hpp"
#include "xbeam/radio.hpp"
#include <cmath>
#include <ctime>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <vector>
namespace {
using J = nlohmann::json;
void check(cudaError_t c) {
    if (c != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(c));
    }
}
struct Buffer {
    void* p = nullptr;
    explicit Buffer(std::size_t n) {
        check(cudaMalloc(&p, n));
    }
    ~Buffer() {
        if (p) {
            cudaFree(p);
        }
    }
};
const J& id(const J& a, int i) {
    for (const auto& x : a) {
        if (x.at("identifier") == i) {
            return x;
        }
    }
    throw std::runtime_error("unknown identifier");
}
}  // namespace
int main(int argc, char** argv) {
    try {
        if (argc != 5) {
            throw std::runtime_error(
                "usage: xbeam-native-station-file-test JOB.input "
                "STREAM BAND OUTPUT_PREFIX");
        }
        xbeam::RadioJob native(argv[1]);
        J job = J::parse(native.normalized_json());
        int stream_id = std::stoi(argv[2]), band_id = std::stoi(argv[3]);
        const auto& stream = id(job["datastreams"], stream_id);
        const auto& band = id(stream["recorded_bands"], band_id);
        const auto& frequency = id(job["frequencies"], band["frequency_id"]);
        std::string source = stream.at("source_files").at(0);
        std::ifstream in(source, std::ios::binary);
        if (!in) {
            throw std::runtime_error("cannot read recording");
        }
        std::vector<unsigned char> frame(
            stream.at("frame_size_bytes").get<int>());
        xbeam::VdifHeader first{};
        std::map<std::int64_t, std::vector<unsigned char>> payloads;
        std::map<std::int64_t, bool> validity;
        double rate = stream.at("sample_rate_hz");
        std::int64_t maximum = 0;
        for (int n = 0; n < 4096; ++n) {
            if (!in.read(reinterpret_cast<char*>(frame.data()), frame.size())) {
                throw std::runtime_error("short bounded VDIF input");
            }
            auto h = xbeam::parse_vdif_header(frame.data(), frame.size());
            if (n == 0) {
                first = h;
            }
            if (h.frame_length_bytes != frame.size()
                || h.thread_id != first.thread_id
                || h.samples_per_frame != first.samples_per_frame
                || h.reference_epoch != first.reference_epoch) {
                throw std::runtime_error(
                    "bounded frontend test requires consistent frame/thread geometry");
            }
            const auto fps = std::llround(rate / first.samples_per_frame);
            if (double(fps) * first.samples_per_frame != rate) {
                throw std::runtime_error(
                    "nonintegral frames-per-second in bounded fixture");
            }
            std::int64_t slot =
                (std::int64_t(h.seconds_from_epoch) - first.seconds_from_epoch)
                    * fps
                + std::int64_t(h.frame_number) - first.frame_number;
            if (slot < 0 || slot > 16384) {
                throw std::runtime_error(
                    "bounded fixture timeline exceeds limit");
            }
            if (!payloads.count(slot) || (!validity[slot] && !h.invalid)) {
                payloads[slot] =
                    std::vector<unsigned char>(frame.begin() + 32, frame.end());
                validity[slot] = !h.invalid;
            }
            maximum = std::max(maximum, slot);
        }
        const int samples_per_frame = first.samples_per_frame,
                  payload_bytes = first.frame_length_bytes - 32;
        std::vector<unsigned char> payload((maximum + 1) * payload_bytes, 0),
            valid(maximum + 1, 0);
        for (const auto& row : payloads) {
            std::copy(
                row.second.begin(),
                row.second.end(),
                payload.begin() + row.first * payload_bytes);
            valid[row.first] = validity[row.first];
        }
        std::tm epoch{};
        epoch.tm_year = 100 + first.reference_epoch / 2;
        epoch.tm_mon = (first.reference_epoch % 2) * 6;
        epoch.tm_mday = 1;
        const auto unix_epoch = ::timegm(&epoch);
        const auto job_day =
            (job.at("start_time").at("mjd").get<std::int64_t>() - 40587)
            * 86400;
        double source_seconds =
            double(
                std::int64_t(unix_epoch) + first.seconds_from_epoch - job_day)
            - std::stod(
                job.at("start_time").at("seconds_of_day").get<std::string>());
        auto start_sample = std::llround(source_seconds * rate)
            + std::int64_t(first.frame_number) * samples_per_frame;
        const int rows = 8,
                  fft = frequency.at("channel_count").get<int>()
            * (stream.at("sampling") == "real" ? 2 : 1);
        std::vector<double> starts(rows);
        double origin = std::max(
            0.,
            std::stod(
                job.at("scans").at(0).at("start_offset_s").get<std::string>()));
        // Select a useful bounded test interval from the source extent and model,
        // rather than accepting spectra from an all-invalid leading window.
        const double source_midpoint =
            (double(start_sample)
                + double(maximum + 1) * samples_per_frame / 2.)
            / rate;
        double trial = std::max(origin, source_midpoint);
        for (int iteration = 0; iteration < 4; ++iteration) {
            auto pivot = native.source_blocks(
                job.at("scans").at(0).at("identifier"),
                stream_id,
                band_id,
                &trial,
                1);
            trial = source_midpoint + pivot[0].centre_delay;
        }
        for (int i = 0; i < rows; ++i) {
            starts[i] = trial + double(i) * fft / rate;
        }
        auto blocks = native.source_blocks(
            job.at("scans").at(0).at("identifier"),
            stream_id,
            band_id,
            starts.data(),
            rows);
        std::vector<xbeam::BlockParameters> parameters(rows);
        for (int i = 0; i < rows; ++i) {
            auto& p = parameters[i];
            auto& b = blocks[i];
            p.output_index = b.output_sample;
            p.coarse_sample = b.coarse_sample;
            p.block_start = b.block_start;
            p.delay_zero = b.delay_zero;
            p.delay_step = b.delay_step;
            p.delay_quadratic = b.delay_quadratic;
            p.residual = b.residual;
            p.valid_stop = fft;
        }
        xbeam::StationLayout layout{};
        layout.rows = rows;
        layout.fft_samples = fft;
        layout.channels = frequency.at("channel_count");
        layout.complex_input = stream.at("sampling") != "real";
        layout.mapping = frequency.at("sideband") == "USB" ? 0
            : layout.complex_input                         ? 2
                                                           : 1;
        if (stream.at("sampling") == "complex_dsb") {
            layout.mapping = 3;
        }
        layout.validity_stride = samples_per_frame;
        layout.samples_per_frame = samples_per_frame;
        layout.payload_bytes = payload_bytes;
        layout.channel_count = 1 << first.log2_channels;
        layout.channel_index = band_id % layout.channel_count;
        layout.source_samples = (maximum + 1) * samples_per_frame;
        layout.source_start = start_sample;
        layout.sample_rate = rate;
        double sign = stream.at("sampling") == "complex"
                && frequency.at("sideband") == "LSB"
            ? -1.
            : 1.;
        layout.reference_hz = sign
            * (frequency.at("band_edge_hz").get<double>()
                + (stream.at("sampling") == "complex_dsb"
                        ? .5 * frequency.at("bandwidth_hz").get<double>()
                        : 0.));
        layout.lo_hz = sign * band.at("lo_offset_hz").get<double>();
        Buffer ds(payload.size()), dv(valid.size()),
            dp(parameters.size() * sizeof(parameters[0]));
        check(cudaMemcpy(
            ds.p, payload.data(), payload.size(), cudaMemcpyHostToDevice));
        check(cudaMemcpy(
            dv.p, valid.data(), valid.size(), cudaMemcpyHostToDevice));
        check(cudaMemcpy(
            dp.p,
            parameters.data(),
            parameters.size() * sizeof(parameters[0]),
            cudaMemcpyHostToDevice));
        xbeam::CudaStation station(layout);
        station.submit(
            reinterpret_cast<std::uintptr_t>(ds.p),
            reinterpret_cast<std::uintptr_t>(dv.p),
            reinterpret_cast<std::uintptr_t>(dp.p));
        std::vector<float> spectra(std::size_t(rows) * layout.channels * 2);
        std::vector<double> weights(rows);
        station.download(spectra.data(), weights.data());
        std::string prefix = argv[4];
        std::ofstream out(prefix + ".spectra", std::ios::binary);
        out.write(reinterpret_cast<char*>(spectra.data()), spectra.size() * 4);
        out.close();
        if (!out) {
            throw std::runtime_error("spectrum write failed");
        }
        J metadata = {{"source", source},
            {"stream", stream_id},
            {"band", band_id},
            {"rows", rows},
            {"channels", layout.channels},
            {"starts", starts},
            {"weights", weights},
            {"source_start_sample", start_sample},
            {"source_samples", layout.source_samples},
            {"channel_index", layout.channel_index},
            {"frames", maximum + 1}};
        std::ofstream meta(prefix + ".json");
        meta << metadata.dump(2);
        meta.close();
        if (!meta) {
            throw std::runtime_error("metadata write failed");
        }
        std::ofstream bytes(prefix + ".payload", std::ios::binary);
        bytes.write(reinterpret_cast<char*>(payload.data()), payload.size());
        bytes.close();
        std::ofstream validity_out(prefix + ".validity", std::ios::binary);
        validity_out.write(reinterpret_cast<char*>(valid.data()), valid.size());
        validity_out.close();
        std::cout << metadata.dump() << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
}
