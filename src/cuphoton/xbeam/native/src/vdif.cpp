// SPDX-License-Identifier: Apache-2.0
#include "xbeam/radio.hpp"
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>
namespace xbeam {
namespace {
std::uint32_t word(const unsigned char* p) {
    return std::uint32_t(p[0]) | (std::uint32_t(p[1]) << 8)
        | (std::uint32_t(p[2]) << 16) | (std::uint32_t(p[3]) << 24);
}
class File {
public:
    explicit File(const std::string& path)
        : fd(::open(path.c_str(), O_RDONLY | O_CLOEXEC)) {
        if (fd < 0) {
            throw RadioError(
                "cannot open VDIF file: " + std::string(std::strerror(errno)));
        }
    }
    ~File() {
        ::close(fd);
    }
    File(const File&) = delete;
    File& operator=(const File&) = delete;
    int fd;
};
}  // namespace
VdifHeader parse_vdif_header(
    const unsigned char* p,
    std::size_t size,
    bool strict,
    bool require_two_bit) {
    if (size < 16) {
        throw RadioError("truncated VDIF header (requires at least 16 bytes)");
    }
    const auto w0 = word(p), w1 = word(p + 4), w2 = word(p + 8),
               w3 = word(p + 12);
    if ((w0 >> 30) & 1) {
        throw RadioError("legacy 16-byte VDIF headers are unsupported");
    }
    if (size < 32) {
        throw RadioError("truncated standard VDIF header (requires 32 bytes)");
    }
    VdifHeader h{};
    h.invalid = w0 >> 31;
    h.seconds_from_epoch = w0 & 0x3fffffff;
    h.reference_epoch = (w1 >> 24) & 63;
    h.frame_number = w1 & 0xffffff;
    h.version = w2 >> 29;
    h.log2_channels = (w2 >> 24) & 31;
    h.frame_length_bytes = (w2 & 0xffffff) * 8;
    h.complex_data = w3 >> 31;
    h.bits_per_sample = ((w3 >> 26) & 31) + 1;
    h.thread_id = (w3 >> 16) & 1023;
    h.station_id = w3 & 65535;
    for (unsigned i = 0; i < 4; ++i) {
        h.extended_words[i] = word(p + 16 + 4 * i);
    }
    h.extended_data_version = h.extended_words[0] >> 24;
    if (strict && w1 >> 30) {
        throw RadioError("VDIF word 1 reserved bits must be zero");
    }
    if (h.version) {
        throw RadioError("unsupported VDIF version");
    }
    if (h.frame_length_bytes < 32) {
        throw RadioError("VDIF frame is smaller than its header");
    }
    if (require_two_bit && h.bits_per_sample != 2) {
        throw RadioError("unsupported VDIF quantization (expected 2 bits)");
    }
    const std::uint64_t sample_bits = (std::uint64_t(1) << h.log2_channels)
        * (h.complex_data ? 2 : 1) * h.bits_per_sample;
    const std::uint64_t payload_bits =
        std::uint64_t(h.frame_length_bytes - 32) * 8;
    if (payload_bits % sample_bits) {
        throw RadioError("VDIF payload has incomplete samples");
    }
    h.samples_per_frame = payload_bits / sample_bits;
    return h;
}
RecordingSummary inspect_vdif(const std::string& path, bool require_two_bit) {
    File file(path);
    struct stat before{}, after{};
    if (::fstat(file.fd, &before)) {
        throw RadioError("cannot stat VDIF file");
    }
    if (!S_ISREG(before.st_mode)) {
        throw RadioError("VDIF input must be a regular file");
    }
    RecordingSummary out;
    out.source_bytes = before.st_size;
    std::uint64_t offset = 0;
    while (offset < out.source_bytes) {
        unsigned char buffer[32];
        std::size_t n = 0;
        while (n < 32) {
            auto got = ::pread(
                file.fd, buffer + n, 32 - n, static_cast<off_t>(offset + n));
            if (got < 0 && errno == EINTR) {
                continue;
            }
            if (got < 0) {
                throw RadioError("VDIF header read failed");
            }
            if (!got) {
                throw RadioError(
                    "truncated VDIF header at byte " + std::to_string(offset));
            }
            n += static_cast<std::size_t>(got);
        }
        const auto h = parse_vdif_header(buffer, 32, true, require_two_bit);
        if (h.frame_length_bytes > out.source_bytes - offset) {
            throw RadioError(
                "truncated VDIF frame at byte " + std::to_string(offset));
        }
        if (!out.frames) {
            out.frame_bytes = h.frame_length_bytes;
        } else if (out.frame_bytes != h.frame_length_bytes) {
            out.frame_bytes = 0;
        }
        ++out.frames;
        out.invalid_frames += h.invalid;
        ++out.thread_frames[h.thread_id];
        offset += h.frame_length_bytes;
    }
    if (::fstat(file.fd, &after) || before.st_size != after.st_size
        || before.st_mtim.tv_sec != after.st_mtim.tv_sec
        || before.st_mtim.tv_nsec != after.st_mtim.tv_nsec
        || before.st_ctim.tv_sec != after.st_ctim.tv_sec
        || before.st_ctim.tv_nsec != after.st_ctim.tv_nsec) {
        throw RadioError("VDIF source changed during inspection");
    }
    return out;
}
std::string inspect_vdif_json(const std::string& path) {
    const auto s = inspect_vdif(path, false);
    std::ostringstream out;
    out << "{\"frames\":" << s.frames
        << ",\"invalid_frames\":" << s.invalid_frames
        << ",\"source_bytes\":" << s.source_bytes
        << ",\"frame_bytes\":" << s.frame_bytes << ",\"threads\":[";
    bool first = true;
    for (std::size_t i = 0; i < s.thread_frames.size(); ++i) {
        if (s.thread_frames[i]) {
            if (!first) {
                out << ',';
            }
            first = false;
            out << "{\"thread_id\":" << i
                << ",\"frames\":" << s.thread_frames[i] << '}';
        }
    }
    out << "]}";
    return out.str();
}
}  // namespace xbeam
