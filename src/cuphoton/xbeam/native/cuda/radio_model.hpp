// SPDX-License-Identifier: Apache-2.0
// Views of the normalized job: bands, visibility products, and sidecar lookup.
#pragma once
#include "xbeam/radio.hpp"
#include <filesystem>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <utility>
#include <vector>
namespace xbeam {

using Json = nlohmann::json;
// (datastream id, recorded band id): one station FFT pipeline.
using Key = std::pair<int, int>;

// Row of `rows` whose "identifier" equals `identifier`.
const Json& by_id(const Json& rows, int identifier);

// A recorded band or a zoom window into one.
struct View {
    Key parent;
    int start, channels, station;
    char polarization;
};
View view(const Json& job, int stream, int band);

struct Product {
    View a, b;
    int frequency, channels, average;
    bool automatic;
    std::size_t offset;  // first output channel of this product
};
// Cross products of every baseline plus, when requested, the autocorrelations.
std::vector<Product> products(const Json& job);

// First valid sidecar `<source filename>.*.vdix` in `directory`.
std::unique_ptr<IndexReader> open_index(
    const std::string& source, const std::filesystem::path& directory);

}  // namespace xbeam
