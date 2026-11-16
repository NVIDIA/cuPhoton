// SPDX-License-Identifier: Apache-2.0
#include "xbeam/radio.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>
#include <regex>
#include <set>
#include <sstream>
#include <tuple>
namespace xbeam {
namespace {
using J = nlohmann::json;
using I = __int128;
I mul(I a, I b) {
    I out;
    if (__builtin_mul_overflow(a, b, &out)) {
        throw RadioError("decimal overflow");
    }
    return out;
}
I sum(I a, I b) {
    I out;
    if (__builtin_add_overflow(a, b, &out)) {
        throw RadioError("decimal overflow");
    }
    return out;
}
I power(int n) {
    if (n < 0 || n > 37) {
        throw RadioError("decimal scale out of range");
    }
    I v = 1;
    while (n--) {
        v = mul(v, 10);
    }
    return v;
}
struct D {
    I value = 0;
    int scale = 0;
    explicit D(const std::string& text) {
        std::size_t pos = 0;
        bool negative = false, dot = false;
        int digits = 0;
        if (pos < text.size() && (text[pos] == '-' || text[pos] == '+')) {
            negative = text[pos++] == '-';
        }
        for (; pos < text.size() && text[pos] != 'e' && text[pos] != 'E';
            ++pos) {
            if (text[pos] == '.' && !dot) {
                dot = true;
                continue;
            }
            if (text[pos] < '0' || text[pos] > '9') {
                throw RadioError("invalid decimal: " + text);
            }
            value = sum(mul(value, 10), text[pos] - '0');
            ++digits;
            if (dot) {
                ++scale;
            }
        }
        if (!digits) {
            throw RadioError("empty decimal");
        }
        if (pos < text.size()) {
            auto e = std::stoi(text.substr(pos + 1));
            if (e > 37 || e < -37) {
                throw RadioError("decimal exponent out of range");
            }
            scale -= e;
        }
        if (scale < 0) {
            value = mul(value, power(-scale));
            scale = 0;
        }
        if (scale > 37) {
            throw RadioError("decimal precision out of range");
        }
        if (negative) {
            value = -value;
        }
    }
    D(I v, int s)
        : value(v),
          scale(s) {}
    D operator+(const D& b) const {
        int s = std::max(scale, b.scale);
        return D(
            sum(mul(value, power(s - scale)), mul(b.value, power(s - b.scale))),
            s);
    }
    D operator-(const D& b) const {
        return *this + D(-b.value, b.scale);
    }
    D times(I x) const {
        return D(mul(value, x), scale);
    }
    std::string str() const {
        I x = value;
        bool neg = x < 0;
        if (neg) {
            x = -x;
        }
        std::string s;
        do {
            s.push_back(char('0' + x % 10));
            x /= 10;
        } while (x);
        std::reverse(s.begin(), s.end());
        if (scale) {
            if (s.size() <= std::size_t(scale)) {
                s = std::string(scale + 1 - s.size(), '0') + s;
            }
            s.insert(s.size() - scale, ".");
        }
        return (neg ? "-" : "") + s;
    }
};
struct Doc {
    std::vector<TextEntry> entries;
    std::vector<std::string> all(const std::string& key) const {
        std::vector<std::string> a;
        for (auto& e : entries) {
            if (e.key == key) {
                a.push_back(e.value);
            }
        }
        return a;
    }
    std::string one(const std::string& key) const {
        auto a = all(key);
        if (a.size() != 1) {
            throw RadioError("missing or repeated field: " + key);
        }
        return a[0];
    }
    int integer(const std::string& key) const {
        auto s = one(key);
        std::size_t n;
        int v = std::stoi(s, &n);
        if (n != s.size()) {
            throw RadioError("invalid integer: " + key);
        }
        return v;
    }
    double real(const std::string& key) const {
        auto s = one(key);
        std::size_t n;
        double v = std::stod(s, &n);
        if (n != s.size() || !std::isfinite(v)) {
            throw RadioError("invalid number: " + key);
        }
        return v;
    }
    Doc section(std::size_t start, std::size_t end) const {
        return {{entries.begin() + start, entries.begin() + end}};
    }
};
std::string read(const std::filesystem::path& p) {
    std::ifstream f(p);
    if (!f) {
        throw RadioError("cannot open job metadata: " + p.string());
    }
    std::ostringstream s;
    s << f.rdbuf();
    if (f.bad()) {
        throw RadioError("job read failed");
    }
    return s.str();
}
Doc document(const std::filesystem::path& p) {
    return {parse_job_text(read(p))};
}
std::string upper(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return char(std::toupper(c));
    });
    return s;
}
bool boolean(std::string s) {
    s = upper(s);
    if (s == "TRUE" || s == "1" || s == "YES") {
        return true;
    }
    if (s == "FALSE" || s == "0" || s == "NO") {
        return false;
    }
    throw RadioError("invalid boolean");
}
std::vector<Doc> sections(
    const Doc& doc, const std::vector<std::string>& keys) {
    std::vector<std::size_t> starts;
    for (auto& key : keys) {
        std::size_t i = 0;
        while (i < doc.entries.size() && doc.entries[i].key != key) {
            ++i;
        }
        if (i == doc.entries.size()
            || (!starts.empty() && i <= starts.back())) {
            throw RadioError("missing or unordered section: " + key);
        }
        starts.push_back(i);
    }
    std::vector<Doc> out;
    for (std::size_t i = 0; i < starts.size(); ++i) {
        out.push_back(doc.section(
            starts[i],
            i + 1 < starts.size() ? starts[i + 1] : doc.entries.size()));
    }
    return out;
}
J coeff(const Doc& d, const std::string& k, int occurrence, double scale) {
    auto rows = d.all(k);
    if (occurrence < 0 || std::size_t(occurrence) >= rows.size()) {
        throw RadioError("missing polynomial: " + k);
    }
    std::istringstream in(rows[occurrence]);
    J out = J::array();
    double x;
    while (in >> x) {
        x *= scale;
        if (!std::isfinite(x)) {
            throw RadioError("nonfinite model");
        }
        out.push_back(x);
    }
    if (!in.eof() || out.empty()) {
        throw RadioError("invalid model polynomial");
    }
    return out;
}
double low(const J& f) {
    return f.at("band_edge_hz").get<double>()
        - (f.at("sideband") == "LSB" ? f.at("bandwidth_hz").get<double>() : 0.);
}
bool same_frequency(double a, double b) {
    return std::abs(a - b)
        <= std::max(1e-6, 1e-12 * std::max(std::abs(a), std::abs(b)));
}
struct Grid {
    int channels;
    double low, spacing;
};
Grid grid(const J& frequency) {
    const int channels = frequency.at("channel_count");
    return {channels,
        low(frequency),
        frequency.at("bandwidth_hz").get<double>() / channels};
}
bool same_grid(const Grid& a, const Grid& b) {
    return a.channels == b.channels && same_frequency(a.low, b.low)
        && same_frequency(a.spacing, b.spacing);
}
void validate_spectral_geometry(const J& job) {
    const auto& frequencies = job.at("frequencies");
    const auto& streams = job.at("datastreams");
    auto band_grid = [&](int stream_id, int band_id) {
        for (const auto& stream : streams) {
            if (stream.at("identifier") != stream_id) {
                continue;
            }
            for (const auto& band : stream.at("recorded_bands")) {
                if (band.at("identifier") == band_id) {
                    return grid(
                        frequencies.at(band.at("frequency_id").get<int>()));
                }
            }
            for (const auto& zoom : stream.at("zoom_bands")) {
                if (zoom.at("identifier") == band_id) {
                    return grid(frequencies.at(
                        zoom.at("output_frequency_id").get<int>()));
                }
            }
        }
        throw RadioError("unknown visibility band");
    };
    for (const auto& stream : streams) {
        const auto& bands = stream.at("recorded_bands");
        if (bands.empty()) {
            throw RadioError("recording declares no bands");
        }
        const int channels =
            frequencies.at(bands.at(0).at("frequency_id").get<int>())
                .at("channel_count");
        for (const auto& band : bands) {
            const auto& frequency =
                frequencies.at(band.at("frequency_id").get<int>());
            const double rate = frequency.at("bandwidth_hz").get<double>()
                * (stream.at("sampling") == "real" ? 2. : 1.);
            if (!same_frequency(rate, stream.at("sample_rate_hz"))
                || frequency.at("channel_count") != channels) {
                throw RadioError(
                    "native recorded bands must share one FFT grid");
            }
        }
        for (const auto& zoom : stream.at("zoom_bands")) {
            const auto parent =
                band_grid(stream.at("identifier"), zoom.at("parent_band_id"));
            const auto output =
                grid(frequencies.at(zoom.at("output_frequency_id").get<int>()));
            const int start = zoom.at("start_channel");
            if (start < 0 || output.channels > parent.channels - start
                || !same_grid(
                    output,
                    {output.channels,
                        parent.low + start * parent.spacing,
                        parent.spacing})) {
                throw RadioError(
                    "zoom output must describe its channel-aligned parent slice");
            }
        }
    }
    for (const auto& baseline : job.at("baselines")) {
        for (const auto& product : baseline.at("products")) {
            const auto a = band_grid(
                baseline.at("datastream_a_id"), product.at("band_a_id"));
            const auto b = band_grid(
                baseline.at("datastream_b_id"), product.at("band_b_id"));
            const auto output = grid(
                frequencies.at(product.at("output_frequency_id").get<int>()));
            if (!same_grid(a, b) || !same_grid(a, output)) {
                throw RadioError(
                    "visibility bands require unsupported spectral regridding");
            }
        }
    }
}
void validate_fringe_order(const J& job) {
    const int order = job.at("fringe_approximation_order");
    if (order < 0 || order > 2) {
        throw RadioError("unsupported fringe approximation order");
    }
    if (order != 2) {
        return;
    }
    auto check = [](const J& coefficients) {
        for (std::size_t i = 3; i < coefficients.size(); ++i) {
            if (coefficients[i].get<double>() != 0.) {
                throw RadioError(
                    "fringe order 2 does not support cubic or higher delay terms");
            }
        }
    };
    for (const auto& station : job.at("stations")) {
        check(station.at("clock").at("coefficients_s"));
    }
    for (const auto& scan : job.at("scans")) {
        for (const auto& model : scan.at("station_models")) {
            check(model.at("delay_coefficients_s"));
        }
    }
}
}  // namespace
std::string load_interop_job_json(const std::string& name) {
    namespace fs = std::filesystem;
    auto path = fs::weakly_canonical(fs::absolute(name));
    auto input = document(path);
    auto calcpath =
        fs::weakly_canonical(path.parent_path() / input.one("CALC FILENAME"));
    auto calc = document(calcpath);
    auto im = document(
        fs::weakly_canonical(calcpath.parent_path() / calc.one("IM FILENAME")));
    if (upper(input.one("OUTPUT FORMAT")) != "SWIN") {
        throw RadioError("unsupported output format");
    }
    const auto scans = calc.integer("NUM SCANS"),
               stations = input.integer("TELESCOPE ENTRIES");
    if (scans < 1 || stations < 1 || im.integer("NUM SCANS") != scans) {
        throw RadioError("invalid scan/station counts");
    }
    const int mjd = input.integer("START MJD");
    const D seconds(input.one("START SECONDS"));
    J job = {{"schema_version", 1},
        {"start_time", {{"mjd", mjd}, {"seconds_of_day", seconds.str()}}},
        {"execution_duration_s", D(input.one("EXECUTE TIME (SEC)")).str()},
        {"integration_duration_s", D(input.one("INT TIME (SEC)")).str()},
        {"fringe_approximation_order", input.integer("FRINGE ROTN ORDER")},
        {"write_autocorrelations", boolean(input.one("WRITE AUTOCORRS"))},
        {"output", {{"format", "swin"}, {"spectral_placement", 0}}}};
    auto sub = D(input.one("SUBINT NANOSECONDS"));
    sub.scale += 9;
    job["subintegration_duration_s"] = sub.str();
    job["buffered_reader"] = nullptr;
    const std::vector<std::pair<std::string, std::string>> geometry = {
        {"buffer_factor", "DATA BUFFER FACTOR"},
        {"data_segments", "NUM DATA SEGMENTS"},
        {"buffered_ffts", "NUM BUFFERED FFTS"}};
    for (auto& kv : geometry) {
        if (!input.all(kv.second).empty()) {
            job["buffered_reader"] = J::object();
            break;
        }
    }
    if (!job["buffered_reader"].is_null()) {
        for (auto& kv : geometry) {
            int v = input.integer(kv.second);
            if (v < 1) {
                throw RadioError("invalid buffer geometry");
            }
            job["buffered_reader"][kv.first] = v;
        }
    }
    job["unsupported_features"] = {
        {"pulsar_binning", boolean(input.one("PULSAR BINNING"))},
        {"phased_array_output", boolean(input.one("PHASED ARRAY"))}};
    for (const auto* key : {"multiple_phase_centres",
             "phase_calibration",
             "system_temperature_calibration",
             "switched_power",
             "polarization_conversion",
             "output_band_stitching",
             "spectral_kurtosis"}) {
        job["unsupported_features"][key] = false;
    }
    if (job["unsupported_features"]["pulsar_binning"].get<bool>()
        || job["unsupported_features"]["phased_array_output"].get<bool>()) {
        throw RadioError("unsupported pulsar/phased-array output");
    }
    job["stations"] = J::array();
    for (int i = 0; i < stations; ++i) {
        auto id = std::to_string(i);
        D epoch(input.one("CLOCK REF MJD " + id));
        I day = epoch.value / power(epoch.scale);
        D sod = (epoch - D(day, 0)).times(86400);
        J clock = J::array();
        int order = input.integer("CLOCK POLY ORDER " + id);
        if (order < 0 || order > 100) {
            throw RadioError("invalid clock order");
        }
        for (int k = 0; k <= order; ++k) {
            clock.push_back(
                input.real("CLOCK COEFF " + id + "/" + std::to_string(k))
                * 1e-6);
        }
        job["stations"].push_back(
            {{"identifier", i},
                {"name", input.one("TELESCOPE NAME " + id)},
                {"clock",
                    {{"epoch",
                         {{"mjd", static_cast<int>(day)},
                             {"seconds_of_day", sod.str()}}},
                        {"coefficients_s", clock}}}});
    }
    auto& frequencies = job["frequencies"] = J::array();
    for (int i = 0; i < input.integer("FREQ ENTRIES"); ++i) {
        auto id = std::to_string(i);
        auto side = upper(input.one("SIDEBAND " + id));
        if (side != "U" && side != "L") {
            throw RadioError("invalid sideband");
        }
        frequencies.push_back(
            {{"identifier", i},
                {"band_edge_hz", input.real("FREQ (MHZ) " + id) * 1e6},
                {"bandwidth_hz", input.real("BW (MHZ) " + id) * 1e6},
                {"sideband", side == "U" ? "USB" : "LSB"},
                {"channel_count", input.integer("NUM CHANNELS " + id)},
                {"channels_to_average", input.integer("CHANS TO AVG " + id)},
                {"oversampling", input.integer("OVERSAMPLE FAC. " + id)},
                {"decimation", input.integer("DECIMATION FAC. " + id)}});
        const auto& frequency = frequencies.back();
        const int channels = frequency.at("channel_count"),
                  average = frequency.at("channels_to_average");
        if (frequency.at("bandwidth_hz").get<double>() <= 0
            || frequency.at("band_edge_hz").get<double>() < 0 || channels <= 0
            || average <= 0 || channels % average) {
            throw RadioError("invalid native frequency geometry");
        }
        if (frequency.at("oversampling") != 1
            || frequency.at("decimation") != 1) {
            throw RadioError(
                "native oversampling and decimation factors must be 1");
        }
    }
    std::vector<std::size_t> starts;
    for (std::size_t i = 0; i < input.entries.size(); ++i) {
        if (input.entries[i].key == "TELESCOPE INDEX") {
            starts.push_back(i);
        }
    }
    if (starts.size() != std::size_t(input.integer("DATASTREAM ENTRIES"))) {
        throw RadioError("datastream section count mismatch");
    }
    J streams = J::array();
    for (std::size_t i = 0; i < starts.size(); ++i) {
        auto d = input.section(
            starts[i],
            i + 1 < starts.size() ? starts[i + 1] : input.entries.size());
        auto id = std::to_string(i);
        J bands = J::array(), zoom = J::array(), files = J::array();
        std::vector<int> recfreq;
        int count = 0;
        for (int f = 0; f < d.integer("NUM RECORDED FREQS"); ++f) {
            recfreq.push_back(d.integer("REC FREQ INDEX " + std::to_string(f)));
            count += d.integer("NUM REC POLS " + std::to_string(f));
        }
        auto format = upper(d.one("DATA FORMAT"));
        int frame = d.integer("DATA FRAME SIZE");
        std::vector<int> threads(count, 0);
        if (format == "VDIF") {
            if (frame == 4032 && count > 1) {
                for (int k = 0; k < count; ++k) {
                    threads[k] = k;
                }
            }
        } else {
            std::smatch match;
            if (!std::regex_match(
                    format,
                    match,
                    std::regex(
                        "INTERLACEDVDIF/([0-9]+(?::[0-9]+)*)/[0-9]+/[0-9]+"))) {
                throw RadioError("unsupported VDIF format");
            }
            std::stringstream t(match[1].str());
            std::string v;
            threads.clear();
            while (std::getline(t, v, ':')) {
                threads.push_back(std::stoi(v));
            }
            if (threads.size() != std::size_t(count)) {
                throw RadioError("thread count mismatch");
            }
        }
        for (int b = 0; b < count; ++b) {
            auto k = std::to_string(b);
            auto slot = d.integer("REC BAND " + k + " INDEX");
            if (slot < 0 || std::size_t(slot) >= recfreq.size()) {
                throw RadioError("recorded frequency slot out of range");
            }
            bands.push_back(
                {{"identifier", b},
                    {"frequency_id", recfreq[slot]},
                    {"polarization", d.one("REC BAND " + k + " POL")},
                    {"thread_id", threads[b]},
                    {"clock_offset_s",
                        d.real("CLK OFFSET " + std::to_string(slot) + " (us)")
                            * 1e-6},
                    {"lo_offset_hz",
                        d.real(
                            "FREQ OFFSET " + std::to_string(slot) + " (Hz)")}});
        }
        int zcount = 0;
        std::vector<int> zfreq;
        for (int f = 0; f < d.integer("NUM ZOOM FREQS"); ++f) {
            zfreq.push_back(d.integer("ZOOM FREQ INDEX " + std::to_string(f)));
            zcount += d.integer("NUM ZOOM POLS " + std::to_string(f));
        }
        for (int z = 0; z < zcount; ++z) {
            auto k = std::to_string(z);
            int slot = d.integer("ZOOM BAND " + k + " INDEX");
            if (slot < 0 || std::size_t(slot) >= zfreq.size()) {
                throw RadioError("zoom slot out of range");
            }
            int output = zfreq[slot];
            auto& f = frequencies.at(output);
            auto pol = d.one("ZOOM BAND " + k + " POL");
            int parent = -1, start = 0;
            for (auto& b : bands) {
                auto& p = frequencies.at(b.at("frequency_id").get<int>());
                double width = p.at("bandwidth_hz").get<double>(),
                       outwidth = f.at("bandwidth_hz").get<double>(),
                       tol = std::max(width, outwidth) * 1e-12;
                double raw = (low(f) - low(p))
                    / (width / p.at("channel_count").get<int>());
                if (b.at("polarization") == pol && low(f) >= low(p) - tol
                    && low(f) + outwidth <= low(p) + width + tol
                    && std::abs(raw - std::round(raw)) <= 1e-7) {
                    if (parent >= 0) {
                        throw RadioError("ambiguous zoom parent");
                    }
                    parent = b.at("identifier");
                    start = int(std::round(raw));
                }
            }
            if (parent < 0) {
                throw RadioError("missing zoom parent");
            }
            zoom.push_back(
                {{"identifier", count + z},
                    {"parent_band_id", parent},
                    {"output_frequency_id", output},
                    {"start_channel", start},
                    {"channel_count", f.at("channel_count")},
                    {"polarization", pol}});
        }
        const int source_count = input.integer("D/STREAM " + id + " FILES");
        if (source_count < 0) {
            throw RadioError("recording file count must be nonnegative");
        }
        for (int f = 0; f < source_count; ++f) {
            files.push_back(
                fs::weakly_canonical(
                    path.parent_path()
                    / input.one("FILE " + id + "/" + std::to_string(f)))
                    .string());
        }
        auto sampling = upper(d.one("DATA SAMPLING"));
        if (sampling != "REAL" && sampling != "COMPLEX"
            && sampling != "COMPLEX_DSB") {
            throw RadioError("unsupported sampling");
        }
        std::string kind = sampling == "REAL" ? "real"
            : sampling == "COMPLEX"           ? "complex"
                                              : "complex_dsb";
        if (recfreq.empty()) {
            throw RadioError("missing recorded frequency");
        }
        streams.push_back(
            {{"identifier", i},
                {"station_id", d.integer("TELESCOPE INDEX")},
                {"sampling", kind},
                {"quantization_bits", d.integer("QUANTISATION BITS")},
                {"frame_size_bytes", frame},
                {"source_files", files},
                {"recorded_bands", bands},
                {"zoom_bands", zoom},
                {"sample_rate_hz",
                    frequencies.at(recfreq[0]).at("bandwidth_hz").get<double>()
                        * (sampling == "REAL" ? 2. : 1.)},
                {"data_format", "VDIF"},
                {"data_source", "file"}});
    }
    job["baselines"] = J::array();
    std::vector<std::string> keys;
    for (int b = 0; b < input.integer("BASELINE ENTRIES"); ++b) {
        keys.push_back("D/STREAM A INDEX " + std::to_string(b));
    }
    auto bs = sections(input, keys);
    for (std::size_t b = 0; b < bs.size(); ++b) {
        auto id = std::to_string(b);
        auto& d = bs[b];
        int a = d.integer("D/STREAM A INDEX " + id),
            c = d.integer("D/STREAM B INDEX " + id);
        auto& sa = streams.at(a);
        streams.at(c);
        J products = J::array();
        std::set<std::tuple<int, int, int>> product_keys;
        std::map<int, int> occurrences;
        for (int f = 0; f < d.integer("NUM FREQS " + id); ++f) {
            auto fid = id + "/" + std::to_string(f);
            auto targets = d.all("TARGET FREQ " + fid);
            if (targets.size() > 1) {
                throw RadioError("repeated target frequency");
            }
            for (int p = 0; p < d.integer("POL PRODUCTS " + fid); ++p) {
                auto left = d.all("D/STREAM A BAND " + std::to_string(p)),
                     right = d.all("D/STREAM B BAND " + std::to_string(p));
                auto n = occurrences[p]++;
                int ba = std::stoi(left.at(n)), bb = std::stoi(right.at(n)),
                    output = -1;
                if (!targets.empty()) {
                    output = std::stoi(targets[0]);
                } else {
                    for (auto& band : sa["recorded_bands"]) {
                        if (band.at("identifier") == ba) {
                            output = band.at("frequency_id");
                        }
                    }
                    for (auto& band : sa["zoom_bands"]) {
                        if (band.at("identifier") == ba) {
                            output = band.at("output_frequency_id");
                        }
                    }
                }
                if (output < 0) {
                    throw RadioError("invalid product band");
                }
                if (!product_keys.emplace(ba, bb, output).second) {
                    throw RadioError("duplicate baseline product");
                }
                products.push_back(
                    {{"band_a_id", ba},
                        {"band_b_id", bb},
                        {"output_frequency_id", output}});
            }
        }
        if (!streams.at(a).at("source_files").empty()
            && !streams.at(c).at("source_files").empty()) {
            job["baselines"].push_back(
                {{"identifier", b},
                    {"datastream_a_id", a},
                    {"datastream_b_id", c},
                    {"products", products}});
        }
    }
    job["datastreams"] = J::array();
    for (auto& s : streams) {
        if (!s.at("source_files").empty()) {
            job["datastreams"].push_back(s);
        }
    }
    keys.clear();
    for (int s = 0; s < scans; ++s) {
        keys.push_back("SCAN " + std::to_string(s) + " POINTING SRC");
    }
    auto ims = sections(im, keys);
    job["scans"] = J::array();
    std::set<std::string> scan_names;
    D fraction(seconds.value % power(seconds.scale), seconds.scale);
    for (int s = 0; s < scans; ++s) {
        auto sid = std::to_string(s), prefix = "SCAN " + sid;
        const auto name = calc.one(prefix + " IDENTIFIER");
        if (!scan_names.insert(name).second) {
            throw RadioError("duplicate scan identifier: " + name);
        }
        const D start = D(calc.one(prefix + " START (S)")) - fraction;
        const D effective_start = start.value < 0 ? D(0, 0) : start;
        auto& d = ims[s];
        if (calc.integer(prefix + " NUM PHS CTRS") != 1) {
            throw RadioError("multiple phase centres unsupported");
        }
        J models = J::array();
        int polys = d.integer(prefix + " NUM POLY");
        if (polys < 1) {
            throw RadioError("missing scan polynomials");
        }
        D first_epoch(0, 0);
        std::set<double> runtime_epochs;
        for (int p = 0; p < polys; ++p) {
            auto poly = prefix + " POLY " + std::to_string(p);
            D epoch =
                D(std::to_string((d.integer(poly + " MJD") - mjd) * 86400))
                + D(d.one(poly + " SEC")) - seconds;
            const auto runtime_epoch = std::stod(epoch.str());
            if (!std::isfinite(runtime_epoch)
                || !runtime_epochs.insert(runtime_epoch).second) {
                throw RadioError(
                    "duplicate/nonfinite model epoch at runtime precision");
            }
            if (p == 0 || (epoch - first_epoch).value < 0) {
                first_epoch = epoch;
            }
            for (int a = 0; a < stations; ++a) {
                auto key = "SRC 0 ANT " + std::to_string(a);
                models.push_back(
                    {{"station_id", a},
                        {"epoch_offset_s", epoch.str()},
                        {"delay_coefficients_s",
                            coeff(d, key + " DELAY (us)", p, 1e-6)},
                        {"u_coefficients_m", coeff(d, key + " U (m)", p, 1.)},
                        {"v_coefficients_m", coeff(d, key + " V (m)", p, 1.)},
                        {"w_coefficients_m", coeff(d, key + " W (m)", p, 1.)}});
            }
        }
        if ((first_epoch - effective_start).value > 0) {
            throw RadioError(
                "first model starts after the scan execution interval");
        }
        job["scans"].push_back(
            {{"identifier", name},
                {"start_offset_s", start.str()},
                {"duration_s", D(calc.one(prefix + " DUR (S)")).str()},
                {"source_id", calc.one(prefix + " POINTING SRC")},
                {"configuration_id", "0"},
                {"station_models", models}});
    }
    validate_spectral_geometry(job);
    validate_fringe_order(job);
    return job.dump();
}
}  // namespace xbeam

namespace xbeam {
namespace {
const J& by_id(const J& rows, int id) {
    for (const auto& row : rows) {
        if (row.at("identifier") == id) {
            return row;
        }
    }
    throw RadioError("unknown native job identifier");
}
double polynomial(const J& coefficients, double x) {
    double result = 0;
    for (auto it = coefficients.rbegin(); it != coefficients.rend(); ++it) {
        result = result * x + it->get<double>();
    }
    return result;
}
std::int64_t nearest(double x) {
    if (!std::isfinite(x) || x < -std::ldexp(1., 63)
        || x >= std::ldexp(1., 63)) {
        throw RadioError("sample coordinate exceeds int64");
    }
    return static_cast<std::int64_t>(std::nearbyint(x));
}
D absolute_time(const J& time) {
    return D(time.at("seconds_of_day").get<std::string>())
        + D(std::to_string(time.at("mjd").get<int>())).times(86400);
}
}  // namespace
struct RadioJob::Impl {
    struct Clock {
        double epoch;
        std::vector<double> coefficients;
    };
    struct Model {
        double epoch;
        std::vector<double> coefficients;
    };
    struct Band {
        int station;
        std::int64_t fft;
        double rate, offset;
    };
    J job;
    std::map<int, Clock> clocks;
    std::map<std::pair<int, int>, Band> bands;
    std::map<std::pair<std::string, int>, std::vector<Model>> models;
    int fringe_order;
    explicit Impl(const std::string& path)
        : job(J::parse(load_interop_job_json(path))) {
        fringe_order = job.at("fringe_approximation_order");
        if (fringe_order < 0 || fringe_order > 2) {
            throw RadioError("unsupported fringe approximation order");
        }
        auto coefficients = [](const J& array) {
            auto values = array.get<std::vector<double>>();
            for (double value : values) {
                if (!std::isfinite(value)) {
                    throw RadioError("nonfinite native model coefficient");
                }
            }
            return values;
        };
        for (const auto& station : job.at("stations")) {
            double epoch = std::stod(
                (absolute_time(job.at("start_time"))
                    - absolute_time(station.at("clock").at("epoch")))
                    .str());
            clocks.emplace(
                station.at("identifier"),
                Clock{epoch,
                    coefficients(station.at("clock").at("coefficients_s"))});
        }
        for (const auto& stream : job.at("datastreams")) {
            for (const auto& band : stream.at("recorded_bands")) {
                const auto& frequency =
                    by_id(job.at("frequencies"), band.at("frequency_id"));
                auto channels =
                    frequency.at("channel_count").get<std::int64_t>();
                int components = stream.at("sampling") == "real" ? 2 : 1;
                if (channels <= 0 || channels > INT64_MAX / components) {
                    throw RadioError("invalid native FFT geometry");
                }
                bands.emplace(
                    std::make_pair(
                        stream.at("identifier").get<int>(),
                        band.at("identifier").get<int>()),
                    Band{stream.at("station_id"),
                        channels * components,
                        stream.at("sample_rate_hz"),
                        band.at("clock_offset_s")});
            }
        }
        for (const auto& scan : job.at("scans")) {
            for (const auto& model : scan.at("station_models")) {
                models[{scan.at("identifier"), model.at("station_id")}]
                    .push_back(
                        {std::stod(
                             model.at("epoch_offset_s").get<std::string>()),
                            coefficients(model.at("delay_coefficients_s"))});
            }
        }
        for (auto& entry : models) {
            std::stable_sort(
                entry.second.begin(),
                entry.second.end(),
                [](const Model& a, const Model& b) {
                    return a.epoch < b.epoch;
                });
        }
    }
};
RadioJob::RadioJob(const std::string& input)
    : impl(std::make_unique<Impl>(input)) {}
RadioJob::~RadioJob() = default;
std::string RadioJob::normalized_json() const {
    return impl->job.dump();
}
std::vector<SourceBlock> RadioJob::source_blocks(
    const std::string& scan_name,
    int stream_id,
    int band_id,
    const double* starts,
    std::size_t count,
    bool literal_endpoint,
    bool phase_parameters) const {
    auto band_it = impl->bands.find({stream_id, band_id});
    if (band_it == impl->bands.end()) {
        throw RadioError("unknown native band");
    }
    const auto& band = band_it->second;
    auto model_it = impl->models.find({scan_name, band.station});
    if (model_it == impl->models.end() || model_it->second.empty()) {
        throw RadioError("missing station model");
    }
    const auto& models = model_it->second;
    const auto& clock = impl->clocks.at(band.station);
    const auto fft = band.fft;
    const double rate = band.rate, clock_epoch = clock.epoch,
                 band_offset = band.offset;
    if (!std::isfinite(rate) || rate <= 0) {
        throw RadioError("invalid native sample rate");
    }
    auto evaluate = [](const std::vector<double>& coefficients, double x) {
        double value = 0;
        for (auto it = coefficients.rbegin(); it != coefficients.rend(); ++it) {
            value = value * x + *it;
        }
        return value;
    };
    std::vector<SourceBlock> out;
    out.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        if (!std::isfinite(starts[i]) || (i && starts[i] < starts[i - 1])) {
            throw RadioError("block starts must be finite and ordered");
        }
        const double centre = literal_endpoint
            ? starts[i]
            : starts[i] + double(fft - 1) / (2. * rate);
        const Impl::Model* selected = nullptr;
        if (models.size() == 1) {
            selected = &models[0];
        } else {
            auto position = std::upper_bound(
                models.begin(),
                models.end(),
                centre,
                [](double t, const Impl::Model& m) { return t < m.epoch; });
            if (position != models.begin()) {
                selected = &*(position - 1);
            }
        }
        if (!selected) {
            throw RadioError("station model does not cover block");
        }
        double delay =
            evaluate(selected->coefficients, centre - selected->epoch);
        delay += evaluate(clock.coefficients, centre + clock_epoch);
        delay += band_offset;
        out.push_back(
            {fft, nearest(starts[i] * rate), nearest(delay * rate), delay});
        auto& block = out.back();
        block.block_start = starts[i];
        if (!phase_parameters) {
            continue;
        }
        auto value_at = [&](double t) {
            double value =
                evaluate(selected->coefficients, t - selected->epoch);
            value += evaluate(clock.coefficients, t + clock_epoch);
            return value + band_offset;
        };
        const int order = impl->fringe_order;
        if (order == 0 || fft == 1) {
            block.delay_zero = delay;
        } else {
            block.delay_zero = value_at(starts[i]);
            double right = value_at(starts[i] + double(fft - 1) / rate);
            if (order == 1) {
                block.delay_step = (right - block.delay_zero) / double(fft - 1);
            } else {
                const double last = double(fft - 1), middle_index = last / 2.;
                const double middle = value_at(starts[i] + middle_index / rate);
                block.delay_quadratic = 2.
                    * (right + block.delay_zero - 2. * middle) / (last * last);
                block.delay_step =
                    (middle - block.delay_zero
                        - block.delay_quadratic * middle_index * middle_index)
                    / middle_index;
            }
        }
        block.residual = double(block.coarse_sample) / rate - delay;
    }
    return out;
}
}  // namespace xbeam

#include "xbeam/replay.hpp"
namespace xbeam {
JobRequestInputs RadioJob::requests(
    const std::string& scan_name,
    int stream_id,
    int band_id,
    const RequestGeometry& g,
    std::int64_t first,
    std::int64_t stop,
    std::int64_t timeline_count,
    std::int64_t job_slot,
    std::int64_t origin_slot,
    bool mean,
    std::int64_t limit) const {
    if (first < 0 || first >= stop || stop > timeline_count || limit < 1
        || origin_slot != job_slot) {
        throw RadioError("invalid request scope or timestamp origin");
    }
    const auto& stream = by_id(impl->job.at("datastreams"), stream_id);
    const double rate = stream.at("sample_rate_hz");
    request_cursor(0, 0, g, rate);
    const auto width = g.ffts_per_request;
    if (timeline_count > INT64_MAX - width) {
        throw RadioError("FFT timeline exceeds int64");
    }
    const auto start = first / width * width;
    const auto rows = (stop - start) / width + ((stop - start) % width != 0);
    if (rows > limit || I(rows) * width > 16777216) {
        throw RadioError("request scope exceeds memory budget");
    }
    const J* scan = nullptr;
    for (const auto& s : impl->job.at("scans")) {
        if (s.at("identifier") == scan_name) {
            scan = &s;
        }
    }
    if (!scan) {
        throw RadioError("unknown request scan");
    }
    const double scan_origin =
        std::max(0., std::stod(scan->at("start_offset_s").get<std::string>()));
    const auto total = std::size_t(rows) * width;
    std::vector<double> times(total), left_times(rows), right_times(rows),
        literal_right;
    const double stride = double(g.fft_samples) / rate,
                 centre_offset = double(g.fft_samples - 1) / (2. * rate);
    for (std::int64_t row = 0; row < rows; ++row) {
        auto index = start + row * width;
        double t = scan_origin + double(index) * stride;
        left_times[row] = t - centre_offset;
        right_times[row] =
            t + double(width) * double(g.fft_samples) / rate - centre_offset;
        for (std::int64_t column = 0; column < width; ++column) {
            times[std::size_t(row) * width + column] =
                scan_origin + double(index + column) * stride;
        }
    }
    const auto ordinary = source_blocks(
        scan_name, stream_id, band_id, times.data(), total, false, false);
    const auto left = source_blocks(
        scan_name, stream_id, band_id, left_times.data(), rows, false, false);
    const auto right = source_blocks(
        scan_name, stream_id, band_id, right_times.data(), rows, false, false);
    std::vector<SourceBlock> literal_left, literal_stop;
    if (mean) {
        literal_right.resize(total);
        for (std::size_t i = 0; i < total; ++i) {
            literal_right[i] = times[i] + stride;
        }
        literal_left = source_blocks(
            scan_name, stream_id, band_id, times.data(), total, true, false);
        literal_stop = source_blocks(
            scan_name,
            stream_id,
            band_id,
            literal_right.data(),
            total,
            true,
            false);
    }
    JobRequestInputs out{
        first, stop, timeline_count, origin_slot, rate, {}, {}, {}, {}, {}};
    auto checked = [](I value) {
        if (value < INT64_MIN || value > INT64_MAX) {
            throw RadioError("request sample coordinate exceeds int64");
        }
        return static_cast<std::int64_t>(value);
    };
    auto offset = [&](double value) {
        value = std::trunc(value + .5);
        if (!std::isfinite(value) || value < -std::ldexp(1., 63)
            || value >= std::ldexp(1., 63)) {
            throw RadioError("request clock exceeds int64");
        }
        return static_cast<std::int64_t>(value);
    };
    for (std::int64_t row = 0; row < rows; ++row) {
        RequestFact fact;
        fact.first_fft = start + row * width;
        fact.active_count = std::min(width, timeline_count - fact.first_fft);
        const double t = scan_origin + double(fact.first_fft) * stride;
        fact.desired_sample = checked(
            I(offset(t * rate)) - offset(left[row].centre_delay * rate));
        fact.endpoint_drift = right[row].centre_delay - left[row].centre_delay;
        std::vector<std::int64_t> outputs;
        std::vector<double> centres, averages;
        for (std::int64_t column = 0; column < width; ++column) {
            auto i = std::size_t(row) * width + column;
            const auto& block = ordinary[i];
            if (block.fft_samples != g.fft_samples) {
                throw RadioError("job and request FFT lengths differ");
            }
            fact.ordinary_samples.push_back(
                checked(I(block.output_sample) - block.coarse_sample));
            outputs.push_back(block.output_sample);
            centres.push_back(block.centre_delay);
            if (mean) {
                double value = literal_left[i].centre_delay * .5
                    + literal_stop[i].centre_delay * .5;
                if (!std::isfinite(value)) {
                    throw RadioError("nonfinite read average");
                }
                averages.push_back(value);
            }
        }
        out.requests.push_back(std::move(fact));
        out.output_samples.push_back(std::move(outputs));
        out.centre_delays.push_back(std::move(centres));
        out.read_average.push_back(std::move(averages));
        out.left_delays.push_back(left[row].centre_delay);
    }
    return out;
}
}  // namespace xbeam

#include "xbeam/window.hpp"
namespace xbeam {
std::int64_t RadioJob::frame_slot(std::int64_t fps, int epoch) const {
    if (fps <= 0 || fps > (1 << 24)) {
        throw RadioError("invalid VDIF frame rate");
    }
    const auto& time = impl->job.at("start_time");
    D seconds = D(I(time.at("mjd").get<std::int64_t>()) * 86400, 0)
        + D(time.at("seconds_of_day"));
    auto value = seconds.times(fps);
    if (value.value % power(value.scale)) {
        throw RadioError("job start must be VDIF-frame aligned");
    }
    // The epoch's leap contribution is distinct from elapsed header seconds.
    int leaps = 0;
    for (int t : {12, 18, 25, 31, 34}) {
        leaps += epoch >= t;
    }
    reference_epoch_seconds(epoch);
    I slot =
        value.value / power(value.scale) - (I(51544) * 86400 - leaps) * fps;
    if (slot < INT64_MIN || slot > INT64_MAX) {
        throw RadioError("job frame slot exceeds int64");
    }
    return static_cast<std::int64_t>(slot);
}
std::pair<std::int64_t, std::int64_t> RadioJob::subintegration_ratio() const {
    D value(impl->job.at("subintegration_duration_s"));
    while (value.scale && value.value % 10 == 0) {
        value.value /= 10;
        --value.scale;
    }
    if (value.value <= 0 || value.value > INT64_MAX
        || power(value.scale) > INT64_MAX) {
        throw RadioError("subintegration rational exceeds int64");
    }
    return {static_cast<std::int64_t>(value.value),
        static_cast<std::int64_t>(power(value.scale))};
}
}  // namespace xbeam
namespace xbeam {
SwinMetadata RadioJob::record_metadata(
    const std::string& scan_name,
    std::int64_t integration,
    int a,
    int b,
    int frequency,
    char pa,
    char pb) const {
    if (integration < 0) {
        throw RadioError("negative output integration");
    }
    const J* scan = nullptr;
    for (const auto& s : impl->job.at("scans")) {
        if (s.at("identifier") == scan_name) {
            scan = &s;
        }
    }
    if (!scan) {
        throw RadioError("unknown output scan");
    }
    D origin(scan->at("start_offset_s"));
    if (origin.value < 0) {
        origin = D(0, 0);
    }
    D duration(impl->job.at("integration_duration_s"));
    D midpoint = duration.times(I(integration) * 2 + 1).times(5);
    midpoint.scale += 1;
    D time = origin + midpoint;
    D seconds = D(impl->job.at("start_time").at("seconds_of_day")) + time;
    I denominator = power(seconds.scale),
      days = seconds.value / (denominator * 86400);
    if (seconds.value % (denominator * 86400) < 0) {
        --days;
    }
    D within = seconds - D(days * 86400, 0);
    I day = impl->job.at("start_time").at("mjd").get<std::int64_t>() + days;
    if (day < 0 || day > UINT32_MAX) {
        throw RadioError("SWIN timestamp exceeds day range");
    }
    int ai = -1, bi = -1;
    const auto& stations = impl->job.at("stations");
    for (std::size_t k = 0; k < stations.size(); ++k) {
        if (stations[k].at("identifier") == a) {
            ai = k + 1;
        }
        if (stations[k].at("identifier") == b) {
            bi = k + 1;
        }
    }
    if (ai < 1 || bi < 1 || ai > 255 || bi > 255) {
        throw RadioError("SWIN station index is unsupported");
    }
    SwinMetadata m{};
    m.baseline = ai * 256 + bi;
    m.mjd = static_cast<std::uint32_t>(day);
    m.seconds = std::stod(within.str());
    m.frequency_id = frequency;
    m.polarization_a = pa;
    m.polarization_b = pb;
    m.configuration_id =
        std::stoi(scan->at("configuration_id").get<std::string>());
    m.source_id = std::stoi(scan->at("source_id").get<std::string>());
    const double t = std::stod(time.str());
    auto coordinates = [&](int station, const char* field) {
        const J* selected = nullptr;
        double epoch = -std::numeric_limits<double>::infinity();
        for (const auto& model : scan->at("station_models")) {
            if (model.at("station_id") == station) {
                double current =
                    std::stod(model.at("epoch_offset_s").get<std::string>());
                if (current <= t && current >= epoch) {
                    selected = &model;
                    epoch = current;
                }
            }
        }
        if (!selected) {
            throw RadioError("no coordinate model covers integration centre");
        }
        return polynomial(selected->at(field), t - epoch);
    };
    m.u =
        coordinates(a, "u_coefficients_m") - coordinates(b, "u_coefficients_m");
    m.v =
        coordinates(a, "v_coefficients_m") - coordinates(b, "v_coefficients_m");
    m.w =
        coordinates(a, "w_coefficients_m") - coordinates(b, "w_coefficients_m");
    return m;
}
}  // namespace xbeam
