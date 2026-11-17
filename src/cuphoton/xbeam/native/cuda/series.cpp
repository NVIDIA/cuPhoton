// SPDX-License-Identifier: Apache-2.0
#include "xbeam/series.hpp"
#include "xbeam/work_queue.hpp"
#include "gpu.hpp"
#include <nlohmann/json.hpp>
#include <cufft.h>
#ifdef XBEAM_HAVE_CUFILE
#include <cufile.h>
#endif
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <fcntl.h>
#include <future>
#include <limits>
#include <map>
#include <mutex>
#include <sys/stat.h>
#include <unistd.h>
using xbeam::gpu::check;
namespace xbeam {
namespace {
void fft_check(cufftResult result) {
    if (result != CUFFT_SUCCESS) {
        throw RadioError(
            "native series cuFFT failure: " + std::to_string(result));
    }
}
std::size_t bounded_bytes(std::int64_t count, std::size_t width) {
    if (count < 0
        || static_cast<unsigned long long>(count) > (1ULL << 30) / width) {
        throw RadioError(
            "native series buffer exceeds 1 GiB; reduce batch_windows");
    }
    return std::size_t(count) * width;
}
struct File {
    int fd = -1;
    struct stat snapshot{};
    bool direct = false;
#ifdef XBEAM_HAVE_CUFILE
    CUfileHandle_t handle = nullptr;
#endif
    ~File() {
#ifdef XBEAM_HAVE_CUFILE
        if (handle) {
            cuFileHandleDeregister(handle);
        }
#endif
        if (fd >= 0) {
            ::close(fd);
        }
    }
    void open(const nlohmann::json& config) {
        const std::string mode = config.at("io");
        if (mode != "posix" && mode != "cufile") {
            throw RadioError("unknown native series I/O mode");
        }
        direct = mode == "cufile";
        const std::string path = config.at("path");
        if (path.find('\0') != std::string::npos) {
            throw RadioError("NUL in native series path");
        }
        fd = ::open(
            path.c_str(), O_RDONLY | O_CLOEXEC | (direct ? O_DIRECT : 0));
        if (fd < 0 || ::fstat(fd, &snapshot) || !S_ISREG(snapshot.st_mode)) {
            throw RadioError("cannot open native series input");
        }
        if (std::uint64_t(snapshot.st_dev)
                != config.at("device").get<std::uint64_t>()
            || std::uint64_t(snapshot.st_ino)
                != config.at("inode").get<std::uint64_t>()) {
            throw RadioError("HDF5 source changed before native ingest");
        }
        if (direct) {
#ifdef XBEAM_HAVE_CUFILE
            ensure_cufile_driver();
            CUfileDescr_t descriptor{};
            descriptor.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
            descriptor.handle.fd = fd;
            if (cuFileHandleRegister(&handle, &descriptor).err
                != CU_FILE_SUCCESS) {
                throw RadioError(
                    "native series cuFile handle registration failed");
            }
#else
            throw RadioError("native CUDA build does not include cuFile");
#endif
        }
    }
    void unchanged() const {
        struct stat now{};
        if (::fstat(fd, &now) || now.st_size != snapshot.st_size
            || now.st_mtim.tv_sec != snapshot.st_mtim.tv_sec
            || now.st_mtim.tv_nsec != snapshot.st_mtim.tv_nsec) {
            throw RadioError("native series input changed during ingest");
        }
    }
};
struct Raw {
    gpu::Memory device{65536};
    gpu::Pinned host;
    std::size_t prefix = 0, size = 0;
#ifdef XBEAM_HAVE_CUFILE
    void* registered = nullptr;
#endif
    ~Raw() {
#ifdef XBEAM_HAVE_CUFILE
        if (registered) {
            cuFileBufDeregister(registered);
        }
#endif
    }
    void read(const File& file, std::int64_t start, std::size_t bytes) {
        if (start < 0 || bytes > std::size_t(file.snapshot.st_size)
            || std::uint64_t(start)
                > std::uint64_t(file.snapshot.st_size) - bytes) {
            throw RadioError("native series read outside file extent");
        }
        const auto begin = file.direct ? start / 65536 * 65536 : start;
        prefix = start - begin;
        size = bytes;
        const auto count = file.direct
            ? std::min<std::int64_t>(
                  file.snapshot.st_size - begin,
                  (prefix + bytes + 65535) / 65536 * 65536)
            : bytes;
        if (count <= 0 || count > (1LL << 30)) {
            throw RadioError("native series read exceeds bounded extent");
        }
#ifdef XBEAM_HAVE_CUFILE
        if (file.direct && count > device.capacity && registered) {
            if (cuFileBufDeregister(registered).err != CU_FILE_SUCCESS) {
                throw RadioError("native series cuFile deregistration failed");
            }
            registered = nullptr;
        }
#endif
        device.reserve(count);
        if (file.direct) {
#ifdef XBEAM_HAVE_CUFILE
            if (!registered) {
                if (cuFileBufRegister(device.p, device.capacity, 0).err
                    != CU_FILE_SUCCESS) {
                    throw RadioError(
                        "native series cuFile buffer registration failed");
                }
                registered = device.p;
            }
            const auto got = cuFileRead(file.handle, device.p, count, begin, 0);
            if (got < 0 || std::size_t(got) < prefix + bytes) {
                throw RadioError("short native series cuFile read");
            }
#endif
        } else {
            host.reserve(count);
            std::size_t done = 0;
            while (done < std::size_t(count)) {
                const auto got = ::pread(
                    file.fd,
                    static_cast<unsigned char*>(host.p) + done,
                    count - done,
                    begin + done);
                if (got < 0 && errno == EINTR) {
                    continue;
                }
                if (got <= 0) {
                    throw RadioError("short native series POSIX read");
                }
                done += got;
            }
        }
    }
    const void* upload(const File& file, cudaStream_t stream) {
        if (!file.direct) {
            check(cudaMemcpyAsync(
                device.p, host.p, size, cudaMemcpyHostToDevice, stream));
        }
        return static_cast<unsigned char*>(device.p) + prefix;
    }
};
struct Input {
    std::int64_t offset, mask_offset;
    int dtype, itemsize;
    Raw raw[2], masks[2];
    gpu::Memory spectrum, mask_spectrum, validity;
};
struct DeviceSelection {
    int device, previous;
    explicit DeviceSelection(const std::string& text) {
        check(cudaGetDevice(&previous));
        device = nlohmann::json::parse(text).at("gpu").get<int>();
        if (device == -1) {
            device = previous;
        }
        check(cudaSetDevice(device));
    }
    ~DeviceSelection() {
        cudaSetDevice(previous);
    }
};
struct DeviceScope {
    int previous;
    explicit DeviceScope(int device) {
        check(cudaGetDevice(&previous));
        check(cudaSetDevice(device));
    }
    ~DeviceScope() {
        cudaSetDevice(previous);
    }
};
struct Plans {
    cufftHandle samples = 0, masks = 0;
    ~Plans() {
        if (samples) {
            cufftDestroy(samples);
        }
        if (masks) {
            cufftDestroy(masks);
        }
    }
};
}  // namespace
struct NativeSeries::Impl {
    DeviceSelection selected;
    gpu::Stream stream;
    File file;
    int device, n, fft_n, group_n, batch_n, pair_count, lag_count, lag_first;
    std::int64_t sample_count, hop;
    bool wide, want_lag, want_spectrum, prefetch, masks = false;
    std::size_t complex_bytes;
    std::vector<std::unique_ptr<Input>> inputs;
    std::vector<std::pair<int, int>> pairs;
    std::map<int, std::unique_ptr<Plans>> plans;
    std::unique_ptr<WorkQueue> workers;
    gpu::Event consumed[2];
    bool recorded[2]{false, false};
    gpu::Memory product, total, mask_product, mask_total, exposure,
        dummy_weights, inverse, mask_inverse, lag, coverage, statistics, error;
    Plans inverse_plans;
    std::mutex active;
    explicit Impl(const std::string& text)
        : selected(text) {
        const auto c = nlohmann::json::parse(text);
        device = selected.device;
        check(cudaSetDevice(device));
        n = c.at("window_samples");
        fft_n = c.at("fft_samples");
        group_n = c.at("windows_per_result");
        batch_n = c.at("batch_windows");
        sample_count = c.at("sample_count");
        hop = c.at("hop_samples");
        wide = c.at("wide");
        prefetch = c.at("prefetch");
        want_lag = c.at("want_lag");
        want_spectrum = c.at("want_spectrum");
        const bool linear = c.at("linear");
        if (n <= 0 || n > (1 << 26) || fft_n < n || fft_n > (1 << 27)
            || group_n <= 0 || batch_n <= 0 || batch_n > 65535 || hop <= 0
            || sample_count < 0 || (!want_lag && !want_spectrum)
            || (linear && std::int64_t(fft_n) < 2LL * n - 1)
            || (!linear && fft_n != n)) {
            throw RadioError("invalid native series geometry");
        }
        batch_n = std::min(batch_n, group_n);
        const std::int64_t read_limit = c.at("read_buffer_bytes");
        if (read_limit < 4096 || read_limit > (1LL << 30)) {
            throw RadioError("invalid native read buffer size");
        }
        for (const auto& input : c.at("inputs")) {
            const int dtype = input.at("dtype");
            const int itemsize = dtype == 0 ? 4 : dtype == 3 ? 16 : 8;
            const auto maximum = read_limit / itemsize;
            if (maximum < n) {
                throw RadioError("native window exceeds read_buffer_bytes");
            }
            batch_n =
                int(std::min<std::int64_t>(batch_n, 1 + (maximum - n) / hop));
        }
        lag_first = linear ? -(n - 1) : -(n / 2);
        lag_count = linear ? 2 * n - 1 : n;
        complex_bytes = wide ? 16 : 8;
        const auto sample_bytes =
            bounded_bytes(std::int64_t(batch_n) * fft_n, complex_bytes);
        file.open(c);
        for (const auto& entry : c.at("inputs")) {
            auto input = std::make_unique<Input>();
            input->offset = entry.at("offset");
            input->mask_offset = entry.at("mask_offset");
            input->dtype = entry.at("dtype");
            if (input->dtype < 0 || input->dtype > 3
                || input->mask_offset < -1) {
                throw RadioError(
                    "invalid native series dtype or validity extent");
            }
            input->itemsize = input->dtype == 0 ? 4
                : input->dtype == 3             ? 16
                                                : 8;
            for (auto range : {std::make_pair(input->offset, input->itemsize),
                     std::make_pair(input->mask_offset, 1)}) {
                if (range.first == -1) {
                    continue;
                }
                if (range.first < 0
                    || __int128(range.first)
                            + __int128(sample_count) * range.second
                        > file.snapshot.st_size) {
                    throw RadioError(
                        "native series dataset exceeds input file");
                }
            }
            input->spectrum.reserve(sample_bytes);
            input->validity.reserve(
                bounded_bytes(std::int64_t(batch_n) * n, 1));
            masks |= input->mask_offset >= 0;
            inputs.push_back(std::move(input));
        }
        if (inputs.empty()) {
            throw RadioError("native series needs input signals");
        }
        for (const auto& pair : c.at("pairs")) {
            if (pair.size() != 2) {
                throw RadioError("invalid native series pair");
            }
            const int a = pair.at(0), b = pair.at(1);
            if (a < 0 || b < 0 || a >= int(inputs.size())
                || b >= int(inputs.size())) {
                throw RadioError("unknown native series pair input");
            }
            pairs.emplace_back(a, b);
        }
        if (pairs.empty() || pairs.size() > 65535) {
            throw RadioError("invalid native series pair count");
        }
        pair_count = pairs.size();
        const auto total_bytes =
            bounded_bytes(std::int64_t(pair_count) * fft_n, complex_bytes);
        total.reserve(total_bytes);
        product.reserve(std::size_t(fft_n) * complex_bytes);
        exposure.reserve(8);
        error.reserve(4);
        statistics.reserve(pair_count * 2 * 8);
        dummy_weights.upload(std::vector<double>(batch_n, 1));
        const int threads = c.at("upload_workers");
        if (threads < 1 || threads > 64) {
            throw RadioError("native series upload_workers must be in [1,64]");
        }
        workers = std::make_unique<WorkQueue>(threads);
        if (want_lag) {
            inverse.reserve(total_bytes);
            lag.reserve(bounded_bytes(
                std::int64_t(pair_count) * lag_count, complex_bytes));
            coverage.reserve(
                bounded_bytes(std::int64_t(pair_count) * lag_count, 8));
            fft_check(cufftPlan1d(
                &inverse_plans.samples,
                fft_n,
                wide ? CUFFT_Z2Z : CUFFT_C2C,
                pair_count));
            fft_check(cufftSetStream(inverse_plans.samples, stream.value));
            if (masks) {
                const auto mask_bytes =
                    bounded_bytes(std::int64_t(pair_count) * fft_n, 16);
                mask_total.reserve(mask_bytes);
                mask_inverse.reserve(mask_bytes);
                mask_product.reserve(std::size_t(fft_n) * 16);
                for (auto& input : inputs) {
                    input->mask_spectrum.reserve(
                        bounded_bytes(std::int64_t(batch_n) * fft_n, 16));
                }
                fft_check(cufftPlan1d(
                    &inverse_plans.masks, fft_n, CUFFT_Z2Z, pair_count));
                fft_check(cufftSetStream(inverse_plans.masks, stream.value));
            }
        }
        check(cudaSetDevice(selected.previous));
    }
    ~Impl() {
        cudaGetDevice(&selected.previous);
        cudaSetDevice(device);
        cudaStreamSynchronize(stream.value);
    }
    Plans& plan(int rows) {
        auto& p = plans[rows];
        if (!p) {
            p = std::make_unique<Plans>();
            fft_check(cufftPlan1d(
                &p->samples, fft_n, wide ? CUFFT_Z2Z : CUFFT_C2C, rows));
            fft_check(cufftSetStream(p->samples, stream.value));
            if (want_lag && masks) {
                fft_check(cufftPlan1d(&p->masks, fft_n, CUFFT_Z2Z, rows));
                fft_check(cufftSetStream(p->masks, stream.value));
            }
        }
        return *p;
    }
    void load(int slot, std::int64_t first, int rows) {
        check(cudaSetDevice(device));
        if (recorded[slot]) {
            check(cudaEventSynchronize(consumed[slot].value));
        }
        const std::int64_t count = std::int64_t(rows - 1) * hop + n;
        std::vector<std::future<void>> pending;
        for (auto& owner : inputs) {
            auto* input = owner.get();
            pending.push_back(
                workers->submit([this, input, slot, first, count] {
                    check(cudaSetDevice(device));
                    input->raw[slot].read(
                        file,
                        input->offset + first * input->itemsize,
                        bounded_bytes(count, input->itemsize));
                    if (input->mask_offset >= 0) {
                        input->masks[slot].read(
                            file,
                            input->mask_offset + first,
                            bounded_bytes(count, 1));
                    }
                }));
        }
        std::exception_ptr failure;
        for (auto& task : pending) {
            try {
                task.get();
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
    void compute(int slot, int rows) {
        auto& p = plan(rows);
        const auto work = reinterpret_cast<std::uintptr_t>(stream.value);
        for (auto& input : inputs) {
            const auto* raw = input->raw[slot].upload(file, stream.value);
            const auto* mask = input->mask_offset < 0
                ? nullptr
                : static_cast<const unsigned char*>(
                      input->masks[slot].upload(file, stream.value));
            pack_series_input(
                raw,
                mask,
                input->dtype,
                wide,
                rows,
                n,
                fft_n,
                hop,
                input->spectrum.p,
                static_cast<unsigned char*>(input->validity.p),
                want_lag && masks ? input->mask_spectrum.p : nullptr,
                static_cast<int*>(error.p),
                work);
            if (wide) {
                fft_check(cufftExecZ2Z(
                    p.samples,
                    static_cast<cufftDoubleComplex*>(input->spectrum.p),
                    static_cast<cufftDoubleComplex*>(input->spectrum.p),
                    CUFFT_FORWARD));
            } else {
                fft_check(cufftExecC2C(
                    p.samples,
                    static_cast<cufftComplex*>(input->spectrum.p),
                    static_cast<cufftComplex*>(input->spectrum.p),
                    CUFFT_FORWARD));
            }
            if (want_lag && masks) {
                fft_check(cufftExecZ2Z(
                    p.masks,
                    static_cast<cufftDoubleComplex*>(input->mask_spectrum.p),
                    static_cast<cufftDoubleComplex*>(input->mask_spectrum.p),
                    CUFFT_FORWARD));
            }
        }
        for (int k = 0; k < pair_count; ++k) {
            auto& a = *inputs[pairs[k].first];
            auto& b = *inputs[pairs[k].second];
            sum_series_spectra(
                wide,
                rows,
                fft_n,
                a.spectrum.p,
                b.spectrum.p,
                static_cast<const double*>(dummy_weights.p),
                1. / group_n,
                product.p,
                static_cast<double*>(exposure.p),
                work);
            add_series_spectrum(
                wide,
                fft_n,
                product.p,
                static_cast<char*>(total.p)
                    + std::size_t(k) * fft_n * complex_bytes,
                work);
            series_statistics(
                static_cast<const unsigned char*>(a.validity.p),
                static_cast<const unsigned char*>(b.validity.p),
                rows,
                n,
                static_cast<std::int64_t*>(statistics.p) + 2 * k,
                work);
            if (want_lag && masks) {
                sum_series_spectra(
                    true,
                    rows,
                    fft_n,
                    a.mask_spectrum.p,
                    b.mask_spectrum.p,
                    static_cast<const double*>(dummy_weights.p),
                    1,
                    mask_product.p,
                    static_cast<double*>(exposure.p),
                    work);
                add_series_spectrum(
                    true,
                    fft_n,
                    mask_product.p,
                    static_cast<char*>(mask_total.p)
                        + std::size_t(k) * fft_n * 16,
                    work);
            }
        }
        check(cudaEventRecord(consumed[slot].value, stream.value));
        recorded[slot] = true;
    }
};
NativeSeries::NativeSeries(const std::string& c)
    : impl(std::make_unique<Impl>(c)) {}
NativeSeries::~NativeSeries() = default;
void NativeSeries::run(
    std::int64_t first,
    int windows,
    void* spectrum,
    std::size_t spectrum_bytes,
    void* lag,
    std::size_t lag_bytes,
    std::int64_t* statistics,
    std::size_t statistics_count,
    std::int64_t* coverage,
    std::size_t coverage_count) {
    auto& p = *impl;
    std::unique_lock<std::mutex> lock(p.active, std::try_to_lock);
    if (!lock.owns_lock()) {
        throw RadioError("native series executor is already active");
    }
    DeviceScope selected(p.device);
    if (windows != p.group_n || first < 0
        || __int128(first) + __int128(windows - 1) * p.hop + p.n
            > p.sample_count
        || spectrum_bytes
            != (p.want_spectrum
                    ? std::size_t(p.pair_count) * p.fft_n * p.complex_bytes
                    : 0)
        || lag_bytes
            != (p.want_lag
                    ? std::size_t(p.pair_count) * p.lag_count * p.complex_bytes
                    : 0)
        || statistics_count != std::size_t(p.pair_count) * 2
        || coverage_count
            != (p.want_lag ? std::size_t(p.pair_count) * p.lag_count : 0)
        || !statistics || (spectrum_bytes && !spectrum) || (lag_bytes && !lag)
        || (coverage_count && !coverage)) {
        throw RadioError(
            "invalid native series result scope or output buffers");
    }
    p.file.unchanged();
    check(cudaMemsetAsync(
        p.total.p,
        0,
        std::size_t(p.pair_count) * p.fft_n * p.complex_bytes,
        p.stream.value));
    check(cudaMemsetAsync(
        p.statistics.p, 0, p.pair_count * 2 * 8, p.stream.value));
    check(cudaMemsetAsync(p.error.p, 0, 4, p.stream.value));
    if (p.want_lag && p.masks) {
        check(cudaMemsetAsync(
            p.mask_total.p,
            0,
            std::size_t(p.pair_count) * p.fft_n * 16,
            p.stream.value));
    }
    int offset = 0, slot = 0, rows = std::min(p.batch_n, windows);
    p.load(slot, first, rows);
    while (offset < windows) {
        p.compute(slot, rows);
        if (!p.prefetch) {
            check(cudaStreamSynchronize(p.stream.value));
        }
        offset += rows;
        if (offset == windows) {
            break;
        }
        slot = 1 - slot;
        rows = std::min(p.batch_n, windows - offset);
        // The transfer threads can read the next slot while queued GPU work runs.
        p.load(slot, first + std::int64_t(offset) * p.hop, rows);
    }
    int error = 0;
    check(cudaMemcpyAsync(
        &error, p.error.p, 4, cudaMemcpyDeviceToHost, p.stream.value));
    check(cudaStreamSynchronize(p.stream.value));
    if (error) {
        throw RadioError(
            error & 2 ? "invalid Boolean native series validity"
                      : "nonfinite valid native series sample");
    }
    if (spectrum_bytes) {
        check(cudaMemcpy(
            spectrum, p.total.p, spectrum_bytes, cudaMemcpyDeviceToHost));
    }
    check(cudaMemcpy(
        statistics,
        p.statistics.p,
        statistics_count * 8,
        cudaMemcpyDeviceToHost));
    if (p.want_lag) {
        if (p.wide) {
            fft_check(cufftExecZ2Z(
                p.inverse_plans.samples,
                static_cast<cufftDoubleComplex*>(p.total.p),
                static_cast<cufftDoubleComplex*>(p.inverse.p),
                CUFFT_INVERSE));
        } else {
            fft_check(cufftExecC2C(
                p.inverse_plans.samples,
                static_cast<cufftComplex*>(p.total.p),
                static_cast<cufftComplex*>(p.inverse.p),
                CUFFT_INVERSE));
        }
        if (p.masks) {
            fft_check(cufftExecZ2Z(
                p.inverse_plans.masks,
                static_cast<cufftDoubleComplex*>(p.mask_total.p),
                static_cast<cufftDoubleComplex*>(p.mask_inverse.p),
                CUFFT_INVERSE));
        }
        gather_series_lag(
            p.wide,
            p.pair_count,
            p.fft_n,
            p.lag_first,
            p.lag_count,
            p.inverse.p,
            p.lag.p,
            p.masks ? p.mask_inverse.p : nullptr,
            static_cast<std::int64_t*>(p.coverage.p),
            reinterpret_cast<std::uintptr_t>(p.stream.value));
        check(cudaMemcpyAsync(
            lag, p.lag.p, lag_bytes, cudaMemcpyDeviceToHost, p.stream.value));
        if (p.masks) {
            check(cudaMemcpyAsync(
                coverage,
                p.coverage.p,
                coverage_count * 8,
                cudaMemcpyDeviceToHost,
                p.stream.value));
        }
        check(cudaStreamSynchronize(p.stream.value));
        if (!p.masks) {
            for (int k = 0; k < p.pair_count; ++k) {
                for (int j = 0; j < p.lag_count; ++j) {
                    coverage[std::size_t(k) * p.lag_count + j] =
                        std::int64_t(windows)
                        * (p.lag_count == p.n
                                ? p.n
                                : p.n - std::abs(p.lag_first + j));
                }
            }
        }
    }
    p.file.unchanged();
}
}  // namespace xbeam
