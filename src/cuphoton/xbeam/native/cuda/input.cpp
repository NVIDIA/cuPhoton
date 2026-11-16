// SPDX-License-Identifier: Apache-2.0
#include "gpu.hpp"
#include "xbeam/cuda.hpp"
#include "xbeam/radio.hpp"
#include <cuda_runtime.h>
#ifdef XBEAM_HAVE_CUFILE
#include <cufile.h>
#endif
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
namespace xbeam {
namespace {
using gpu::check;
// NVFS and cuFile require 64-KiB aligned device pages.
constexpr std::size_t kDevicePage = 65536;
#ifdef XBEAM_HAVE_CUFILE
struct Driver {
    Driver() {
        auto result = cuFileDriverOpen();
        if (result.err != CU_FILE_SUCCESS) {
            throw RadioError(
                "cuFile driver open failed: " + std::to_string(result.err));
        }
    }
    ~Driver() {
        cuFileDriverClose();
    }
};
struct ThreadStaging {
    gpu::Memory raw{kDevicePage};
    cudaStream_t stream = nullptr;
    bool registered = false;
    int device = 0;
    ThreadStaging() {
        check(cudaGetDevice(&device));
        int low, high;
        check(cudaDeviceGetStreamPriorityRange(&low, &high));
        check(
            cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, high));
    }
    ~ThreadStaging() {
        cudaSetDevice(device);
        if (stream) {
            cudaStreamSynchronize(stream);
        }
        if (registered) {
            cuFileBufDeregister(raw.p);
        }
        if (stream) {
            cudaStreamDestroy(stream);
        }
    }
    void reserve(std::size_t size) {
        if (registered && size <= raw.capacity) {
            return;
        }
        if (registered) {
            auto status = cuFileBufDeregister(raw.p);
            if (status.err != CU_FILE_SUCCESS) {
                throw RadioError("cuFile staging deregistration failed");
            }
            registered = false;
        }
        raw.reserve(size);
        auto status = cuFileBufRegister(raw.p, raw.capacity, 0);
        if (status.err != CU_FILE_SUCCESS) {
            throw RadioError(
                "cuFile staging registration failed: "
                + std::to_string(status.err));
        }
        registered = true;
    }
};
ThreadStaging& worker_staging(std::size_t bytes) {
    thread_local std::unique_ptr<ThreadStaging> staging;
    int device;
    check(cudaGetDevice(&device));
    if (staging && staging->device != device) {
        staging.reset();
        check(cudaSetDevice(device));
    }
    if (!staging) {
        staging = std::make_unique<ThreadStaging>();
    }
    staging->reserve(bytes);
    return *staging;
}
#endif
}  // namespace
struct CudaRecordingInput::Impl {
    int fd = -1;
    const ReadMode mode;
    const bool shared_staging;
    bool gds() const {
        return mode == ReadMode::GpuDirect;
    }
    // Direct and cuFile reads bypass the page cache and need aligned extents.
    bool aligned() const {
        return mode != ReadMode::Buffered;
    }
    std::size_t chunk_bytes;
    std::int64_t framed, bytes, file_size;
    gpu::Memory raw{kDevicePage}, packed{kDevicePage}, valid{kDevicePage};
    gpu::Pinned host;
    cudaStream_t stream = nullptr;
    std::atomic<bool> active{false};
    std::atomic<bool> failed{false};
    std::atomic<std::size_t> remaining{0};
    std::uint64_t generation = 0;
    std::int64_t read_begin = 0, read_prefix = 0, read_frames = 0;
    std::vector<unsigned char> pending_validity;
    std::vector<NativeReadChunk> chunks;
    std::unique_ptr<std::atomic<unsigned char>[]> claimed;
#ifdef XBEAM_HAVE_CUFILE
    CUfileHandle_t handle = nullptr;
    std::vector<void*> registered_regions;
#endif
    Impl(
        const std::string& source,
        std::int64_t frame,
        std::int64_t payload,
        ReadMode read_mode,
        std::size_t chunk,
        bool shared)
        : mode(read_mode),
          shared_staging(shared),
          chunk_bytes(chunk),
          framed(frame),
          bytes(payload) {
        if (frame <= 32 || payload != frame - 32) {
            throw RadioError("invalid recording frame geometry");
        }
        if (!chunk_bytes || chunk_bytes % 65536 || chunk_bytes > 1073741824) {
            throw RadioError(
                "native read chunks must be 64-KiB multiples up to 1 GiB");
        }
        if (shared_staging && !gds()) {
            throw RadioError("shared read staging requires cuFile");
        }
#ifndef XBEAM_HAVE_CUFILE
        if (gds()) {
            throw RadioError("native CUDA build does not include cuFile");
        }
#endif
        fd = ::open(
            source.c_str(), O_RDONLY | O_CLOEXEC | (aligned() ? O_DIRECT : 0));
        if (fd < 0) {
            throw RadioError("cannot open native recording input");
        }
        try {
            struct stat st{};
            if (::fstat(fd, &st) || !S_ISREG(st.st_mode) || st.st_size <= 0
                || st.st_size % frame) {
                throw RadioError("invalid native input file extent");
            }
            file_size = st.st_size;
            check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
#ifdef XBEAM_HAVE_CUFILE
            if (gds()) {
                static Driver driver;
                CUfileDescr_t descriptor{};
                descriptor.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
                descriptor.handle.fd = fd;
                auto status = cuFileHandleRegister(&handle, &descriptor);
                if (status.err != CU_FILE_SUCCESS) {
                    throw RadioError(
                        "cuFile handle registration failed: "
                        + std::to_string(status.err));
                }
            }
#endif
        } catch (...) {
            if (stream) {
                cudaStreamDestroy(stream);
            }
            ::close(fd);
            fd = -1;
            throw;
        }
    }
    ~Impl() {
        if (stream) {
            cudaStreamSynchronize(stream);
            cudaStreamDestroy(stream);
        }
#ifdef XBEAM_HAVE_CUFILE
        for (auto region : registered_regions) {
            cuFileBufDeregister(region);
        }
        if (handle) {
            cuFileHandleDeregister(handle);
        }
#endif
        if (fd >= 0) {
            ::close(fd);
        }
    }
};
CudaRecordingInput::CudaRecordingInput(
    const std::string& source,
    std::int64_t frame,
    std::int64_t payload,
    ReadMode mode,
    std::size_t chunk_bytes,
    bool shared_staging)
    : impl(
          std::make_unique<Impl>(
              source, frame, payload, mode, chunk_bytes, shared_staging)) {}
CudaRecordingInput::~CudaRecordingInput() = default;
std::vector<NativeReadChunk> CudaRecordingInput::begin(
    std::int64_t first,
    std::int64_t stop,
    const std::vector<unsigned char>& validity) {
    auto& state = *impl;
    if (state.active.exchange(true)) {
        throw RadioError("native input already has an active transfer");
    }
    try {
        auto& p = *impl;
        if (first < 0 || stop <= first || stop > p.file_size / p.framed
            || std::size_t(stop - first) != validity.size()
            || __int128(stop - first) * p.framed > 1073741824) {
            throw RadioError("native input read exceeds bounded extent");
        }
        const auto offset = first * p.framed,
                   begin = p.aligned() ? offset / 65536 * 65536 : offset;
        const auto end = p.aligned()
            ? std::min(p.file_size, (stop * p.framed + 65535) / 65536 * 65536)
            : stop * p.framed;
        const auto count = std::size_t(end - begin);
#ifdef XBEAM_HAVE_CUFILE
        if (p.gds() && !p.shared_staging && count > p.raw.capacity) {
            while (!p.registered_regions.empty()) {
                auto status = cuFileBufDeregister(p.registered_regions.back());
                if (status.err != CU_FILE_SUCCESS) {
                    throw RadioError("cuFile buffer deregistration failed");
                }
                p.registered_regions.pop_back();
            }
        }
#endif
        if (!p.shared_staging) {
            p.raw.reserve((count + 65535) / 65536 * 65536);
        }
#ifdef XBEAM_HAVE_CUFILE
        if (p.gds() && !p.shared_staging && p.registered_regions.empty()) {
            p.registered_regions.reserve(
                (p.raw.capacity + p.chunk_bytes - 1) / p.chunk_bytes);
            try {
                for (std::size_t offset = 0; offset < p.raw.capacity;
                    offset += p.chunk_bytes) {
                    auto address =
                        static_cast<unsigned char*>(p.raw.p) + offset;
                    auto size =
                        std::min(p.chunk_bytes, p.raw.capacity - offset);
                    auto status = cuFileBufRegister(address, size, 0);
                    if (status.err != CU_FILE_SUCCESS) {
                        throw RadioError(
                            "cuFile buffer registration failed: "
                            + std::to_string(status.err));
                    }
                    p.registered_regions.push_back(address);
                }
            } catch (...) {
                for (auto region : p.registered_regions) {
                    cuFileBufDeregister(region);
                }
                p.registered_regions.clear();
                throw;
            }
        }
#endif
        p.packed.reserve(std::size_t(stop - first) * p.bytes);
        p.valid.reserve(validity.size());

        if (!p.gds()) {
            p.host.reserve(
                p.mode == ReadMode::Direct ? (count + 65535) / 65536 * 65536
                                           : count);
        }
        if (p.generation == UINT64_MAX) {
            throw RadioError("native input generation overflow");
        }
        ++p.generation;
        p.read_begin = begin;
        p.read_prefix = offset - begin;
        p.read_frames = stop - first;
        p.pending_validity = validity;
        p.chunks.clear();
        for (std::size_t position = 0; position < count;
            position += p.chunk_bytes) {
            p.chunks.push_back(
                {p.generation,
                    p.chunks.size(),
                    position,
                    std::min(p.chunk_bytes, count - position)});
        }
        p.claimed =
            std::make_unique<std::atomic<unsigned char>[]>(p.chunks.size());
        for (std::size_t k = 0; k < p.chunks.size(); ++k) {
            p.claimed[k].store(0);
        }
        p.remaining.store(p.chunks.size());
        p.failed.store(false);
        return p.chunks;
    } catch (...) {
        state.active.store(false);
        throw;
    }
}
void CudaRecordingInput::read_chunk(const NativeReadChunk& chunk) {
    auto& p = *impl;
    if (!p.active.load() || chunk.generation != p.generation
        || chunk.index >= p.chunks.size()) {
        throw RadioError("stale native input chunk");
    }
    const auto& expected = p.chunks[chunk.index];
    if (chunk.offset != expected.offset || chunk.size != expected.size
        || p.claimed[chunk.index].exchange(1)) {
        throw RadioError("invalid or duplicate native input chunk");
    }
    struct Completion {
        std::atomic<std::size_t>& remaining;
        std::atomic<bool>& failed;
        bool success = false;
        ~Completion() {
            if (!success) {
                failed.store(true);
            }
            remaining.fetch_sub(1);
        }
    } complete{p.remaining, p.failed};
    if (p.gds()) {
#ifdef XBEAM_HAVE_CUFILE
        auto* staging =
            p.shared_staging ? &worker_staging(p.chunk_bytes) : nullptr;
        auto pointer = staging
            ? staging->raw.p
            : static_cast<unsigned char*>(p.raw.p) + chunk.offset;
        auto n = cuFileRead(
            p.handle, pointer, chunk.size, p.read_begin + chunk.offset, 0);
        if (n < 0 || std::size_t(n) != chunk.size) {
            throw RadioError(
                "cuFile native chunk read was short or failed: "
                + std::to_string(n));
        }
        if (staging) {
            auto first = (p.read_begin + p.read_prefix) / p.framed;
            unpack_frame_chunk(
                pointer,
                p.packed.p,
                first,
                p.read_frames,
                p.framed,
                p.bytes,
                p.read_begin + chunk.offset,
                chunk.size,
                reinterpret_cast<std::uintptr_t>(staging->stream));
            // Future completion includes extraction. High priority keeps a
            // short staging operation from waiting behind a long compute grid.
            check(cudaStreamSynchronize(staging->stream));
        }
#endif
    } else {
        std::size_t done = 0;
        // O_DIRECT needs block-multiple lengths; only the last chunk can be ragged
        // and the kernel returns the bytes up to EOF for it.
        const std::size_t want = p.mode == ReadMode::Direct
            ? (chunk.size + 4095) / 4096 * 4096
            : chunk.size;
        while (done < chunk.size) {
            auto n = ::pread(
                p.fd,
                static_cast<unsigned char*>(p.host.p) + chunk.offset + done,
                want - done,
                p.read_begin + chunk.offset + done);
            if (n < 0 && errno == EINTR) {
                continue;
            }
            if (n <= 0) {
                throw RadioError("short native POSIX chunk read");
            }
            done += n;
        }
        check(cudaMemcpyAsync(
            static_cast<unsigned char*>(p.raw.p) + chunk.offset,
            static_cast<unsigned char*>(p.host.p) + chunk.offset,
            chunk.size,
            cudaMemcpyHostToDevice,
            p.stream));
    }
    complete.success = true;
}
void CudaRecordingInput::finish() {
    auto& p = *impl;
    if (!p.active.load() || p.remaining.load() != 0) {
        throw RadioError("unfinished native input chunks");
    }
    if (p.failed.load()) {
        throw RadioError("failed native input transfer cannot be published");
    }
    check(cudaMemcpyAsync(
        p.valid.p,
        p.pending_validity.data(),
        p.pending_validity.size(),
        cudaMemcpyHostToDevice,
        p.stream));
    if (!p.shared_staging) {
        unpack_frames(
            p.raw.p,
            p.packed.p,
            p.read_frames,
            p.framed,
            p.bytes,
            p.read_prefix,
            reinterpret_cast<std::uintptr_t>(p.stream));
    }
    check(cudaStreamSynchronize(p.stream));
    p.active.store(false);
}
void CudaRecordingInput::read(
    std::int64_t first,
    std::int64_t stop,
    const std::vector<unsigned char>& validity) {
    for (const auto& chunk : begin(first, stop, validity)) {
        read_chunk(chunk);
    }
    finish();
}
std::uintptr_t CudaRecordingInput::payload() const {
    return reinterpret_cast<std::uintptr_t>(impl->packed.p);
}
std::uintptr_t CudaRecordingInput::validity() const {
    return reinterpret_cast<std::uintptr_t>(impl->valid.p);
}
}  // namespace xbeam
