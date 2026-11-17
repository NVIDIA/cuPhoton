// SPDX-License-Identifier: Apache-2.0
#include "xbeam/radio.hpp"
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <type_traits>
#include <unistd.h>
namespace xbeam {
namespace {
template <class T>
void append(std::string& output, const T value) {
    static_assert(std::is_trivially_copyable<T>::value, "scalar required");
    unsigned char bytes[sizeof(T)];
    std::memcpy(bytes, &value, sizeof(T));
    const std::uint16_t marker = 1;
    const bool little = *reinterpret_cast<const unsigned char*>(&marker) == 1;
    for (std::size_t i = 0; i < sizeof(T); ++i) {
        output.push_back(
            static_cast<char>(bytes[little ? i : sizeof(T) - 1 - i]));
    }
}
}  // namespace
std::string encode_swin(
    const SwinMetadata& m, const float* samples, std::size_t channels) {
    static_assert(
        sizeof(float) == 4 && sizeof(double) == 8, "IEEE sizes required");
    static_assert(
        std::numeric_limits<float>::is_iec559
            && std::numeric_limits<double>::is_iec559,
        "IEEE floats required");
    const int a = m.baseline / 256, b = m.baseline % 256;
    if (a < 1 || a > 255 || b < 1 || b > 255) {
        throw RadioError("invalid SWIN baseline");
    }
    if (static_cast<unsigned char>(m.polarization_a) > 127
        || static_cast<unsigned char>(m.polarization_b) > 127) {
        throw RadioError("SWIN polarization must be ASCII");
    }
    if (channels > (std::numeric_limits<std::size_t>::max() - 74) / 8) {
        throw RadioError("SWIN channel count overflow");
    }
    if (channels && !samples) {
        throw RadioError("missing SWIN spectrum");
    }
    std::string out;
    out.reserve(74 + 8 * channels);
    append(out, std::uint32_t(0xff00ff00));
    append(out, std::uint32_t(1));
    append(out, m.baseline);
    append(out, m.mjd);
    append(out, m.seconds);
    append(out, m.configuration_id);
    append(out, m.source_id);
    append(out, m.frequency_id);
    out.push_back(m.polarization_a);
    out.push_back(m.polarization_b);
    append(out, m.pulsar_bin);
    append(out, m.weight);
    append(out, m.u);
    append(out, m.v);
    append(out, m.w);
    for (std::size_t i = 0; i < 2 * channels; ++i) {
        append(out, samples[i]);
    }
    return out;
}
void write_swin_file(const std::string& path, const std::string& records) {
    if (path.find('\0') != std::string::npos) {
        throw RadioError("path contains NUL");
    }
    int fd =
        ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0666);
    if (fd < 0) {
        throw RadioError(
            "cannot create SWIN output: " + std::string(std::strerror(errno)));
    }
    std::size_t written = 0;
    while (written < records.size()) {
        const auto n =
            ::write(fd, records.data() + written, records.size() - written);
        if (n < 0 && errno == EINTR) {
            continue;
        }
        if (n <= 0) {
            const int error = errno;
            ::close(fd);
            throw RadioError(
                "SWIN write failed: " + std::string(std::strerror(error)));
        }
        written += static_cast<std::size_t>(n);
    }
    if (::close(fd)) {
        throw RadioError("SWIN close failed");
    }
}
}  // namespace xbeam
