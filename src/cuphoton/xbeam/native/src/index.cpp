// SPDX-License-Identifier: Apache-2.0
#include "xbeam/radio.hpp"
#include <array>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <limits>
#include <nlohmann/json.hpp>
#include <openssl/evp.h>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>
namespace xbeam {
namespace {
std::uint64_t u64(const unsigned char* p) {
    std::uint64_t v = 0;
    for (int i = 7; i >= 0; --i) {
        v = (v << 8) | p[i];
    }
    return v;
}
std::uint32_t u32(const unsigned char* p) {
    return std::uint32_t(u64(p) & 0xffffffff);
}
std::uint64_t align(std::uint64_t n) {
    if (n > UINT64_MAX - 4095) {
        throw RadioError("index extent overflow");
    }
    return (n + 4095) / 4096 * 4096;
}
struct File {
    int fd = -1;
    struct stat identity{};
    std::string path;
    explicit File(const std::string& p)
        : path(p) {
        if (p.find('\0') != std::string::npos) {
            throw RadioError("path contains NUL");
        }
        fd = ::open(p.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) {
            throw RadioError("cannot open index/source: " + p);
        }
        if (::fstat(fd, &identity) || !S_ISREG(identity.st_mode)) {
            ::close(fd);
            fd = -1;
            throw RadioError("index/source is not a regular file");
        }
    }
    ~File() {
        if (fd >= 0) {
            ::close(fd);
        }
    }
    void unchanged() const {
        struct stat after{}, named{};
        if (::stat(path.c_str(), &named) || named.st_dev != identity.st_dev
            || named.st_ino != identity.st_ino) {
            throw RadioError("index/source path was replaced");
        }
        if (::fstat(fd, &after) || after.st_size != identity.st_size
            || after.st_mtim.tv_sec != identity.st_mtim.tv_sec
            || after.st_mtim.tv_nsec != identity.st_mtim.tv_nsec
            || after.st_ctim.tv_sec != identity.st_ctim.tv_sec
            || after.st_ctim.tv_nsec != identity.st_ctim.tv_nsec) {
            throw RadioError("index/source changed during use");
        }
    }
    void read(
        unsigned char* dst, std::size_t count, std::uint64_t offset) const {
        std::size_t n = 0;
        while (n < count) {
            auto got = ::pread(fd, dst + n, count - n, offset + n);
            if (got < 0 && errno == EINTR) {
                continue;
            }
            if (got <= 0) {
                throw RadioError("truncated index/source read");
            }
            n += got;
        }
    }
};
// Own a bounded read window so concurrent file changes cannot invalidate
// memory being inspected. Each query owns its cache and remains independent.
struct ReadCache {
    const File& file;
    std::array<unsigned char, 65536> data;
    std::uint64_t first = 0;
    std::size_t cached = 0;
    explicit ReadCache(const File& source)
        : file(source) {}
    const unsigned char* read(std::uint64_t offset, std::size_t count) {
        const auto size = std::uint64_t(file.identity.st_size);
        if (count > data.size() || offset > size || count > size - offset) {
            throw RadioError("index read exceeds sidecar extent");
        }
        if (offset < first || offset - first > cached
            || count > cached - (offset - first)) {
            first = offset;
            cached = std::min<std::uint64_t>(data.size(), size - first);
            file.read(data.data(), cached, first);
        }
        return data.data() + (offset - first);
    }
};
std::string fingerprint(const File& f, std::uint64_t frame) {
    auto n = std::min<std::uint64_t>(frame, 4096);
    std::vector<unsigned char> data(n * 2);
    f.read(data.data(), n, 0);
    f.read(data.data() + n, n, f.identity.st_size - frame);
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int length = 0;
    if (!EVP_Digest(
            data.data(), data.size(), digest, &length, EVP_sha256(), nullptr)) {
        throw RadioError("source fingerprint failed");
    }
    std::ostringstream out;
    for (unsigned int i = 0; i < length; ++i) {
        out << std::hex << std::setw(2) << std::setfill('0')
            << unsigned(digest[i]);
    }
    return out.str();
}
}  // namespace
struct IndexReader::Impl {
    File source, index;
    std::size_t size;
    std::uint64_t count, frame_count;
    int version, stride;
    nlohmann::json metadata;
    std::vector<unsigned char> validity;
    std::vector<PhysicalIndexRun> physical;
    Impl(const std::string& s, const std::string& p)
        : source(s),
          index(p),
          size(index.identity.st_size) {
        if (size < 65536) {
            throw RadioError("truncated sidecar header");
        }
        unsigned char prefix[16];
        index.read(prefix, 16, 0);
        if (std::memcmp(prefix, "XCVDIX1\n", 8)) {
            throw RadioError("invalid sidecar magic");
        }
        // u32 requires eight readable bytes; prefix has both words available.
        version = u32(prefix + 8);
        auto bytes = std::uint32_t(prefix[12])
            | (std::uint32_t(prefix[13]) << 8)
            | (std::uint32_t(prefix[14]) << 16)
            | (std::uint32_t(prefix[15]) << 24);
        if (version < 2 || version > 4 || bytes == 0 || bytes > 65536 - 16) {
            throw RadioError("invalid sidecar version/metadata length");
        }
        std::string text(bytes, '\0');
        index.read(reinterpret_cast<unsigned char*>(text.data()), bytes, 16);
        metadata = nlohmann::json::parse(text);
        if (metadata.at("schema_version") != version) {
            throw RadioError("sidecar version fields disagree");
        }
        count = metadata.at("entry_count").get<std::uint64_t>();
        frame_count = metadata.at("frame_count").get<std::uint64_t>();
        auto frame = metadata.at("frame_size_bytes").get<std::uint64_t>();
        if (frame <= 32 || count != frame_count || count > UINT64_MAX / frame
            || count * frame != std::uint64_t(source.identity.st_size)
            || metadata.at("source_size_bytes") != source.identity.st_size) {
            throw RadioError("sidecar source geometry is stale");
        }
        const auto& provenance = metadata.at("source_provenance");
        auto mtime = std::int64_t(source.identity.st_mtim.tv_sec) * 1000000000
            + source.identity.st_mtim.tv_nsec;
        auto ctime = std::int64_t(source.identity.st_ctim.tv_sec) * 1000000000
            + source.identity.st_ctim.tv_nsec;
        if (provenance.at("version") != 1
            || provenance.at("size_bytes") != source.identity.st_size
            || provenance.at("mtime_ns") != mtime
            || provenance.at("ctime_ns") != ctime) {
            throw RadioError("sidecar source provenance is stale");
        }
        stride = version == 2 ? 16 : 24;
        if (count > (UINT64_MAX - 65536) / stride) {
            throw RadioError("sidecar extent overflow");
        }
        auto stop = 65536 + count * stride;
        auto run_offset = version == 4 ? align(stop) : stop;
        auto runs =
            metadata.value("physical_timestamp_run_count", std::uint64_t(0));
        if (runs > (UINT64_MAX - run_offset) / 24
            || size != align(run_offset + runs * 24)) {
            throw RadioError("invalid sidecar extent");
        }
        if (version < 4 && runs) {
            throw RadioError("legacy sidecar declares physical runs");
        }
        if (metadata.value("rejected_header_frame_count", 0) != 0
            || metadata.value("rejected_format_frame_count", 0) != 0) {
            throw RadioError("sidecar contains rejected malformed frames");
        }
        if (metadata.at("sample_fingerprint_sha256")
            != fingerprint(source, frame)) {
            throw RadioError("sidecar source fingerprint is stale");
        }
        if (!metadata.at("threads").is_array()
            || metadata.at("threads").empty()) {
            throw RadioError("missing sidecar thread inventory");
        }
        for (auto& thread : metadata.at("threads")) {
            if (thread.at("frame_size_bytes") != frame) {
                throw RadioError("invalid thread geometry");
            }
        }
        {
            ReadCache rows(index);
            std::uint64_t prev_stop = 0;
            if (version == 4) {
                if (metadata.at("physical_timestamp_run_layout")
                        != "physical-start-u64,physical-stop-u64,packed-timestamp-start-"
                           "u64"
                    || metadata.at("physical_timestamp_run_offset_bytes")
                        != run_offset) {
                    throw RadioError("invalid physical run layout");
                }
                for (std::uint64_t i = 0; i < runs; ++i) {
                    auto p = rows.read(run_offset + i * 24, 24);
                    auto first = u64(p), last = u64(p + 8), time = u64(p + 16);
                    if (first >= last || last > count || first < prev_stop
                        || time >= (std::uint64_t(1) << 60)
                        || (time & 0xffffff) + (last - first)
                            > (std::uint64_t(1) << 24)) {
                        throw RadioError("invalid physical timestamp run");
                    }
                    physical.push_back({first, last, time});
                    prev_stop = last;
                }
            }
            // Validate the index once before binary searches can trust its order.
            std::vector<unsigned char> seen(count / 8 + (count % 8 != 0), 0);
            std::uint64_t previous = 0;
            validity.resize(count);
            for (std::uint64_t i = 0; i < count; ++i) {
                auto entry = rows.read(65536 + i * stride, stride);
                auto timestamp = u64(entry), location = u64(entry + 8);
                auto ordinal = location & ((std::uint64_t(1) << 53) - 1);
                auto state = version == 2 ? (((location >> 53) & 1) ? 8 : 0)
                                          : u64(entry + 16);
                if (timestamp >= (std::uint64_t(1) << 60)
                    || (i && timestamp < previous) || ordinal >= count) {
                    throw RadioError("invalid sidecar entry ordering/ordinal");
                }
                if (seen[ordinal / 8] & (1u << (ordinal % 8))) {
                    throw RadioError("duplicate physical sidecar ordinal");
                }
                seen[ordinal / 8] |= 1u << (ordinal % 8);
                if (version >= 3
                    && ((state & ~std::uint64_t(23))
                        || bool(state & 7) != bool((location >> 53) & 1)
                        || (state & 6))) {
                    throw RadioError("invalid/rejected sidecar frame state");
                }
                validity[ordinal] = state == 0;
                previous = timestamp;
            }
            source.unchanged();
            index.unchanged();
        }
    }
};
IndexReader::IndexReader(const std::string& source, const std::string& index)
    : impl(std::make_unique<Impl>(source, index)) {}
IndexReader::~IndexReader() = default;
IndexReader::IndexReader(IndexReader&&) noexcept = default;
std::string IndexReader::metadata_json() const {
    impl->source.unchanged();
    impl->index.unchanged();
    return impl->metadata.dump();
}
std::vector<IndexFrame> IndexReader::query(
    std::uint16_t thread, std::uint64_t first, std::uint64_t stop) const {
    if (thread > 1023 || first > stop) {
        throw RadioError("invalid sidecar query");
    }
    impl->source.unchanged();
    impl->index.unchanged();
    ReadCache rows(impl->index);
    auto row = [&](std::uint64_t i) {
        return rows.read(65536 + i * impl->stride, impl->stride);
    };
    auto lower = [&](std::uint64_t value) {
        std::uint64_t lo = 0, hi = impl->count;
        while (lo < hi) {
            auto mid = lo + (hi - lo) / 2;
            if (u64(row(mid)) < value) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return lo;
    };
    std::vector<IndexFrame> result;
    for (auto i = lower(first), end = lower(stop); i < end; ++i) {
        auto entry = row(i);
        auto loc = u64(entry + 8);
        if ((loc >> 54) != thread) {
            continue;
        }
        auto physical = loc & ((std::uint64_t(1) << 53) - 1);
        auto state =
            impl->version == 2 ? ((loc >> 53) & 1 ? 8 : 0) : u64(entry + 16);
        if (physical >= impl->frame_count
            || (impl->version >= 3
                && ((state & ~std::uint64_t(23))
                    || bool(state & 7) != bool((loc >> 53) & 1)))) {
            throw RadioError("invalid sidecar frame entry");
        }
        result.push_back(
            {u64(entry), physical, thread, static_cast<std::uint8_t>(state)});
    }
    impl->source.unchanged();
    impl->index.unchanged();
    return result;
}
}  // namespace xbeam

namespace xbeam {
std::vector<PhysicalIndexRun> IndexReader::physical_runs() const {
    impl->source.unchanged();
    impl->index.unchanged();
    if (impl->version != 4) {
        throw RadioError("physical replay requires schema-v4 sidecar");
    }
    return impl->physical;
}
std::vector<unsigned char> IndexReader::frame_validity(
    std::uint64_t first, std::uint64_t count) const {
    impl->source.unchanged();
    impl->index.unchanged();
    if (first > impl->count || count > impl->count - first) {
        throw RadioError("physical validity query exceeds file");
    }
    if (impl->metadata.at("threads").size() != 1) {
        throw RadioError("physical replay requires a single recorded thread");
    }
    return {
        impl->validity.begin() + first, impl->validity.begin() + first + count};
}
}  // namespace xbeam
