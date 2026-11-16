// SPDX-License-Identifier: Apache-2.0
#include "xbeam/radio.hpp"
#include <cmath>
#include <limits>
namespace xbeam {
namespace {
using Wide = __int128;
Wide mod(Wide x, Wide y) {
    auto r = x % y;
    return r < 0 ? r + y : r;
}
std::int64_t narrow(Wide x) {
    if (x < std::numeric_limits<std::int64_t>::min()
        || x > std::numeric_limits<std::int64_t>::max()) {
        throw RadioError("buffered request coordinate exceeds int64");
    }
    return static_cast<std::int64_t>(x);
}
Wide round_sample(double x) {
    x = std::trunc(x + 0.5);
    if (!std::isfinite(x) || x < -std::ldexp(1.0, 126)
        || x >= std::ldexp(1.0, 126)) {
        throw RadioError("buffered delay drift exceeds supported coordinates");
    }
    return narrow(static_cast<Wide>(x));
}
}  // namespace
RequestCursor request_cursor(
    std::int64_t local,
    double drift_seconds,
    const RequestGeometry& g,
    double rate) {
    if (!std::isfinite(drift_seconds)) {
        throw RadioError("delay drift must be finite");
    }
    if (!std::isfinite(rate) || rate <= 0) {
        throw RadioError("sample rate must be finite and positive");
    }
    if (g.ratio_numerator <= 0 || g.ratio_denominator <= 0 || g.fft_samples <= 0
        || g.ffts_per_request <= 0 || g.alignment_bytes <= 0
        || g.word_bytes <= 0) {
        throw RadioError("buffered request geometry must be positive");
    }
    Wide num = g.ratio_numerator, den = g.ratio_denominator;
    if ((Wide(g.fft_samples) * den) % num || g.alignment_bytes % g.word_bytes
        || (Wide(g.alignment_bytes) * num) % den) {
        throw RadioError(
            "buffered geometry must cover complete bytes, words and samples");
    }
    const auto payload = narrow(Wide(g.fft_samples) * den / num);
    Wide origin = Wide(local) * den / num;
    origin -= mod(origin, g.word_bytes);
    Wide skipped = 0;
    if (origin < 0) {
        skipped = (-origin + payload - 1) / payload;
        if (skipped >= g.ffts_per_request) {
            return {};
        }
        origin += skipped * payload;
    }
    Wide drift = 0;
    if (skipped) {
        drift = round_sample(
            static_cast<double>(skipped) * drift_seconds * rate
            / g.ffts_per_request);
        const auto correction = drift * den / num;
        if (origin - correction < 0) {
            ++skipped;
            origin += payload;
            if (skipped == g.ffts_per_request) {
                return {};
            }
            drift = round_sample(
                static_cast<double>(skipped) * drift_seconds * rate
                / g.ffts_per_request);
            origin -= drift * den / num;
        }
    }
    origin -= mod(origin, g.word_bytes);
    origin -= mod(origin, g.alignment_bytes);
    const auto origin64 = narrow(origin);
    const Wide product = Wide(origin64) * num;
    Wide presented = product / den;
    if (product < 0 && product % den) {
        --presented;
    }
    const Wide delta =
        Wide(local) + skipped * g.fft_samples - drift - presented;
    return {true, origin64, narrow(skipped), narrow(delta)};
}
}  // namespace xbeam
