// SPDX-License-Identifier: Apache-2.0
#include "xbeam/cuda.hpp"
#include "xbeam/work_queue.hpp"
#include <algorithm>
#include <cuda_runtime.h>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <unistd.h>
#include <vector>
namespace {
void check(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}
struct Temporary {
    std::string path;
    explicit Temporary(const std::string& directory) {
        auto name =
            (std::filesystem::path(directory) / "xbeam-native-input-XXXXXX")
                .string();
        std::vector<char> buffer(name.begin(), name.end());
        buffer.push_back(0);
        int fd = ::mkstemp(buffer.data());
        if (fd < 0) {
            throw std::runtime_error("cannot create native input fixture");
        }
        path = buffer.data();
        ::close(fd);
    }
    ~Temporary() {
        ::unlink(path.c_str());
    }
};
unsigned char code(int frame, int byte) {
    return (frame * 17 + byte * 7) % 251;
}
}  // namespace
int main(int argc, char** argv) {
    try {
        check(cudaSetDevice(0));
        const std::string io = argc > 2 ? argv[2] : "posix";
        const auto mode = io == "cufile" ? xbeam::ReadMode::GpuDirect
            : io == "posix-direct"       ? xbeam::ReadMode::Direct
                                         : xbeam::ReadMode::Buffered;
        const bool shared = argc > 4 && std::string(argv[4]) == "shared";
        const std::size_t chunk =
            argc > 3 ? std::stoull(argv[3]) : 16 * 1024 * 1024;
        if (!chunk || chunk % 65536 || chunk > 1073741824) {
            throw std::runtime_error("invalid test chunk size");
        }
        for (int payload : {8000, 8192}) {
            const int framed = payload + 32,
                      frames =
                          std::max<std::size_t>(4097, chunk / payload + 128);
            Temporary temporary(argc > 1 ? argv[1] : "/tmp");
            FILE* file = ::fopen(temporary.path.c_str(), "wb");
            if (!file) {
                throw std::runtime_error("cannot write native input fixture");
            }
            for (int frame = 0; frame < frames; ++frame) {
                std::vector<unsigned char> data(framed, 0xAA);
                for (int byte = 0; byte < payload; ++byte) {
                    data[32 + byte] = code(frame, byte);
                }
                if (::fwrite(data.data(), 1, data.size(), file)
                    != data.size()) {
                    ::fclose(file);
                    throw std::runtime_error(
                        "short native input fixture write");
                }
            }
            if (::fclose(file)) {
                throw std::runtime_error("native input fixture close failed");
            }
            xbeam::CudaRecordingInput a(
                temporary.path, framed, payload, mode, chunk, shared),
                b(temporary.path, framed, payload, mode, chunk, shared);
            xbeam::WorkQueue queue(2);
            for (auto range : std::vector<std::pair<int, int>>{{1, 4},
                     {0, 32},
                     {31, frames},
                     {frames - 17, frames},
                     {20, 25}}) {
                std::vector<unsigned char> validity(
                    range.second - range.first, 1);
                validity[0] = 0;
                std::vector<std::future<void>> pending;
                for (auto* input : {&a, &b}) {
                    auto chunks =
                        input->begin(range.first, range.second, validity);
                    bool refused = false;
                    try {
                        input->finish();
                    } catch (const std::exception&) {
                        refused = true;
                    }
                    if (!refused) {
                        throw std::runtime_error(
                            "input finished before its chunks");
                    }
                    if (range.second == frames && range.first == 31
                        && chunks.size() < 2) {
                        throw std::runtime_error(
                            "fixture did not cross read chunks");
                    }
                    for (const auto& chunk : chunks) {
                        pending.push_back(queue.submit([input, chunk] {
                            check(cudaSetDevice(0));
                            input->read_chunk(chunk);
                        }));
                    }
                }
                for (auto& future : pending) {
                    future.wait();
                }
                for (auto& future : pending) {
                    future.get();
                }
                a.finish();
                b.finish();
                for (auto* input : {&a, &b}) {
                    std::vector<unsigned char> actual(
                        validity.size() * payload),
                        mask(validity.size());
                    check(cudaMemcpy(
                        actual.data(),
                        reinterpret_cast<void*>(input->payload()),
                        actual.size(),
                        cudaMemcpyDeviceToHost));
                    check(cudaMemcpy(
                        mask.data(),
                        reinterpret_cast<void*>(input->validity()),
                        mask.size(),
                        cudaMemcpyDeviceToHost));
                    if (mask != validity) {
                        throw std::runtime_error(
                            "native validity upload mismatch");
                    }
                    for (int frame = range.first; frame < range.second;
                        ++frame) {
                        for (int byte = 0; byte < payload; ++byte) {
                            if (actual[(frame - range.first) * payload + byte]
                                != code(frame, byte)) {
                                throw std::runtime_error(
                                    "native framed payload mismatch");
                            }
                        }
                    }
                }
            }
            // A failed read must never publish the partially populated buffer,
            // even after all chunk tasks have returned.
            xbeam::CudaRecordingInput broken(
                temporary.path, framed, payload, mode, chunk, shared);
            auto chunks =
                broken.begin(0, 32, std::vector<unsigned char>(32, 1));
            if (::truncate(temporary.path.c_str(), 0)) {
                throw std::runtime_error("cannot truncate failed-read fixture");
            }
            bool read_failed = false, publication_failed = false;
            try {
                broken.read_chunk(chunks.at(0));
            } catch (const std::exception&) {
                read_failed = true;
            }
            try {
                broken.finish();
            } catch (const std::exception&) {
                publication_failed = true;
            }
            if (!read_failed || !publication_failed) {
                throw std::runtime_error("failed native read was published");
            }
            std::cout << "payload=" << payload << " mode=" << io
                      << " chunk=" << chunk << " shared=" << shared
                      << " concurrent resize/unaligned/EOF delivery PASS\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
