// SPDX-License-Identifier: Apache-2.0
#include "radio_model.hpp"
#include <algorithm>
#include <map>
#include <set>
namespace xbeam {

const Json& by_id(const Json& rows, int identifier) {
    for (const auto& row : rows) {
        if (row.at("identifier") == identifier) {
            return row;
        }
    }
    throw RadioError("unknown execution identifier");
}

std::unique_ptr<IndexReader> open_index(
    const std::string& source, const std::filesystem::path& directory) {
    std::vector<std::filesystem::path> candidates;
    const auto base = std::filesystem::path(source).filename().string();
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        const auto name = entry.path().filename().string();
        if (name.rfind(base + ".", 0) == 0
            && entry.path().extension() == ".vdix") {
            candidates.push_back(entry.path());
        }
    }
    std::sort(candidates.begin(), candidates.end());
    std::string error;
    for (const auto& path : candidates) {
        try {
            auto index = std::make_unique<IndexReader>(source, path.string());
            if (Json::parse(index->metadata_json()).at("schema_version") != 4) {
                throw RadioError(
                    "native execution requires a schema-v4 sidecar");
            }
            return index;
        } catch (const std::exception& e) {
            error = e.what();
        }
    }
    throw RadioError("no valid sidecar for " + source + ": " + error);
}

View view(const Json& job, int stream, int band) {
    const auto& stream_row = by_id(job.at("datastreams"), stream);
    const int station = stream_row.at("station_id");
    for (const auto& recorded : stream_row.at("recorded_bands")) {
        if (recorded.at("identifier") == band) {
            return {{stream, band},
                0,
                by_id(job.at("frequencies"), recorded.at("frequency_id"))
                    .at("channel_count"),
                station,
                recorded.at("polarization").get<std::string>().at(0)};
        }
    }
    for (const auto& zoom : stream_row.at("zoom_bands")) {
        if (zoom.at("identifier") == band) {
            return {{stream, zoom.at("parent_band_id")},
                zoom.at("start_channel"),
                zoom.at("channel_count"),
                station,
                zoom.at("polarization").get<std::string>().at(0)};
        }
    }
    throw RadioError("unknown execution band");
}

std::vector<Product> products(const Json& job) {
    std::vector<Product> result;
    std::map<Key, std::set<int>> autos;
    auto append = [&](int stream_a,
                      int band_a,
                      int stream_b,
                      int band_b,
                      int frequency,
                      bool automatic) {
        const auto a = view(job, stream_a, band_a),
                   b = view(job, stream_b, band_b);
        const auto& row = by_id(job.at("frequencies"), frequency);
        const int channels = row.at("channel_count"),
                  average = row.at("channels_to_average");
        if (a.channels != channels || b.channels != channels || channels <= 0
            || average <= 0 || channels % average) {
            throw RadioError("visibility channel geometries differ");
        }
        result.push_back({a, b, frequency, channels, average, automatic, 0});
    };
    for (const auto& baseline : job.at("baselines")) {
        for (const auto& p : baseline.at("products")) {
            const int a = baseline.at("datastream_a_id"),
                      b = baseline.at("datastream_b_id"),
                      frequency = p.at("output_frequency_id"),
                      band_a = p.at("band_a_id"), band_b = p.at("band_b_id");
            append(a, band_a, b, band_b, frequency, false);
            autos[{a, frequency}].insert(band_a);
            autos[{b, frequency}].insert(band_b);
        }
    }
    if (job.at("write_autocorrelations").get<bool>()) {
        for (const auto& [owner, bands] : autos) {
            for (int a : bands) {
                for (int b : bands) {
                    append(owner.first, a, owner.first, b, owner.second, true);
                }
            }
        }
    }
    std::size_t offset = 0;
    for (auto& p : result) {
        p.offset = offset;
        offset += p.channels / p.average;
    }
    return result;
}
}  // namespace xbeam
