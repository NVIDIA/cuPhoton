// SPDX-License-Identifier: Apache-2.0
#include "radio_options.hpp"
#include "xbeam/radio.hpp"
#include <map>
#include <set>
namespace xbeam {
namespace {
using Raw = std::map<std::string, std::string>;

const std::string& required(const Raw& raw, const char* key) {
    auto found = raw.find(key);
    if (found == raw.end()) {
        throw RadioError(std::string("missing execution option ") + key);
    }
    return found->second;
}
std::string text(const Raw& raw, const char* key, const char* fallback) {
    auto found = raw.find(key);
    return found == raw.end() ? fallback : found->second;
}
std::int64_t integer(const Raw& raw, const char* key, std::int64_t fallback) {
    auto found = raw.find(key);
    return found == raw.end() ? fallback : std::stoll(found->second);
}
bool toggle(const Raw& raw, const char* key, const char* what) {
    const auto value = text(raw, key, "on");
    if (value != "on" && value != "off") {
        throw RadioError(std::string("native ") + what + " must be on or off");
    }
    return value == "on";
}
}  // namespace

const char* io_name(ReadMode mode) {
    switch (mode) {
    case ReadMode::Buffered:
        return "posix";
    case ReadMode::Direct:
        return "posix-direct";
    case ReadMode::GpuDirect:
        return "cufile";
    }
    return "?";
}

Options Options::parse(const std::vector<std::string>& argv) {
    if (argv.size() < 3) {
        throw RadioError(
            "usage: xbeam-native-correlate JOB.input --output DIR --index-dir DIR "
            "--scan ID --first-integration N --integration-count N --fft-tile N "
            "--device N --profile interop-real-usb-buffered-window-replay-v1");
    }
    Raw raw;
    for (std::size_t i = 2; i < argv.size(); i += 2) {
        if (i + 1 >= argv.size() || !raw.emplace(argv[i], argv[i + 1]).second) {
            throw RadioError("missing or repeated execution option");
        }
    }
    static const std::set<std::string> allowed = {"--output",
        "--index-dir",
        "--scan",
        "--first-integration",
        "--integration-count",
        "--fft-tile",
        "--device",
        "--profile",
        "--vdif-io",
        "--upload-workers",
        "--read-chunk-bytes",
        "--read-staging",
        "--xmac-pack-products",
        "--xmac-reuse-sources",
        "--prefetch",
        "--claim-dir",
        "--workers-total",
        "--min-chunk"};
    for (const auto& option : raw) {
        if (!allowed.count(option.first)) {
            throw RadioError(
                "unsupported native execution option: " + option.first);
        }
    }

    Options o;
    o.job = argv[1];
    o.profile = required(raw, "--profile");
    if (o.profile != "interop-real-usb-buffered-window-replay-v1") {
        throw RadioError(
            "this native runner requires the buffered-window replay contract");
    }
    o.output = required(raw, "--output");
    o.index_dir = required(raw, "--index-dir");
    o.scan = required(raw, "--scan");
    o.device = static_cast<int>(integer(raw, "--device", 0));

    const auto io = text(raw, "--vdif-io", "posix");
    if (io == "posix") {
        o.read_mode = ReadMode::Buffered;
    } else if (io == "posix-direct") {
        o.read_mode = ReadMode::Direct;
    } else if (io == "cufile") {
        o.read_mode = ReadMode::GpuDirect;
    } else {
        throw RadioError(
            "native VDIF IO must be posix, posix-direct or cufile");
    }
    o.prefetch = toggle(raw, "--prefetch", "prefetch");
    o.chunk_bytes = integer(raw, "--read-chunk-bytes", o.chunk_bytes);
    if (o.chunk_bytes <= 0 || o.chunk_bytes % 65536
        || o.chunk_bytes > 1073741824) {
        throw RadioError(
            "native read chunks must be 64-KiB multiples up to 1 GiB");
    }
    const auto staging = text(raw, "--read-staging", "per-input");
    o.shared_staging = staging == "shared";
    if ((staging != "per-input" && !o.shared_staging)
        || (o.shared_staging && o.read_mode != ReadMode::GpuDirect)) {
        throw RadioError(
            "native read staging must be per-input or shared cuFile");
    }

    o.first_integration = integer(raw, "--first-integration", 0);
    o.integration_count = std::stoll(required(raw, "--integration-count"));
    o.fft_tile = integer(raw, "--fft-tile", o.fft_tile);
    o.upload_workers =
        static_cast<int>(integer(raw, "--upload-workers", o.upload_workers));
    if (o.upload_workers < 1 || o.upload_workers > 64) {
        throw RadioError("native upload workers must be between 1 and 64");
    }
    o.xmac_pack_products = static_cast<int>(
        integer(raw, "--xmac-pack-products", o.xmac_pack_products));
    o.xmac_reuse_sources =
        toggle(raw, "--xmac-reuse-sources", "XMAC source reuse");

    o.claim_dir = text(raw, "--claim-dir", "");
    if (!o.dynamic()
        && (raw.count("--workers-total") || raw.count("--min-chunk"))) {
        throw RadioError("chunk options require --claim-dir");
    }
    o.workers_total = static_cast<int>(integer(raw, "--workers-total", 1));
    o.min_chunk = integer(raw, "--min-chunk", 4);
    if (o.dynamic()
        && (o.workers_total < 1 || o.workers_total > 4096 || o.min_chunk < 1)) {
        throw RadioError("invalid dynamic chunk scheduling options");
    }
    return o;
}
}  // namespace xbeam
