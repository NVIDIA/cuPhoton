// SPDX-License-Identifier: Apache-2.0
#include "xbeam/cuda.hpp"
#include <cmath>
#include <complex>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace {
void check(cudaError_t code) {
    if (code != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(code));
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
}  // namespace
int main() {
    try {
        for (int n : {8000, 262144}) {
            for (int pattern : {0x55, 0xe4}) {
                const int rows = 8, frame = n == 8000 ? 8000 : 32768;
                std::vector<unsigned char> source(
                    std::size_t(rows) * n / 4, pattern),
                    valid(std::size_t(rows) * n / frame, 1);
                std::vector<xbeam::BlockParameters> parameters(rows);
                for (int row = 0; row < rows; ++row) {
                    parameters[row].output_index = std::int64_t(row) * n;
                    parameters[row].valid_stop = n;
                }
                Buffer device_source(source.size()), device_valid(valid.size()),
                    device_parameters(
                        parameters.size() * sizeof(parameters[0]));
                check(cudaMemcpy(
                    device_source.p,
                    source.data(),
                    source.size(),
                    cudaMemcpyHostToDevice));
                check(cudaMemcpy(
                    device_valid.p,
                    valid.data(),
                    valid.size(),
                    cudaMemcpyHostToDevice));
                check(cudaMemcpy(
                    device_parameters.p,
                    parameters.data(),
                    parameters.size() * sizeof(parameters[0]),
                    cudaMemcpyHostToDevice));
                xbeam::StationLayout layout{};
                layout.rows = rows;
                layout.fft_samples = n;
                layout.channels = n / 2;
                layout.mapping = 0;
                layout.validity_stride = frame;
                layout.samples_per_frame = frame;
                layout.payload_bytes = frame / 4;
                layout.channel_count = 1;
                layout.channel_index = 0;
                layout.source_samples = std::int64_t(rows) * n;
                layout.sample_rate = n == 8000 ? 125000000. : 4096000000.;
                xbeam::CudaStation station(layout);
                station.submit(
                    reinterpret_cast<std::uintptr_t>(device_source.p),
                    reinterpret_cast<std::uintptr_t>(device_valid.p),
                    reinterpret_cast<std::uintptr_t>(device_parameters.p));
                std::vector<float> output(
                    std::size_t(rows) * layout.channels * 2);
                std::vector<double> weights(rows);
                station.download(output.data(), weights.data());
                const float levels[] = {-3.3359f, -1.f, 1.f, 3.3359f};
                double maximum = 0;
                for (int row = 0; row < rows; ++row) {
                    if (weights[row] != 1.) {
                        throw std::runtime_error(
                            "native frontend weight mismatch");
                    }
                    for (int channel = 0; channel < layout.channels;
                        ++channel) {
                        std::complex<double> expected = 0.;
                        if (channel % (n / 4) == 0) {
                            for (int sample = 0; sample < 4; ++sample) {
                                double phase = -2. * std::acos(-1.)
                                    * (channel / (n / 4)) * sample / 4.;
                                expected +=
                                    double(
                                        levels[(pattern >> (sample * 2)) & 3])
                                    * std::complex<double>(
                                        std::cos(phase), std::sin(phase));
                            }
                            expected *= double(n) / 4.;
                        }
                        auto index =
                            (std::size_t(row) * layout.channels + channel) * 2;
                        double error = std::abs(
                            std::complex<double>(
                                output[index], output[index + 1])
                            - expected);
                        if (!std::isfinite(error)) {
                            throw std::runtime_error(
                                "nonfinite native frontend output");
                        }
                        maximum = std::max(maximum, error);
                        if (error > 1e-3 + 2e-6 * std::abs(expected)) {
                            throw std::runtime_error(
                                "native frontend spectrum mismatch");
                        }
                    }
                }
                std::cout << "fft=" << n << " pattern=" << pattern
                          << " maximum_error=" << maximum << " PASS\n";
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
}
