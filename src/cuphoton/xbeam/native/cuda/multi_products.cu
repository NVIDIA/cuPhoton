// SPDX-License-Identifier: Apache-2.0
#include "multi_products.inc"
#undef BAND_LIMIT
#undef PRODUCT_LIMIT
namespace narrow_products {
#define BAND_LIMIT 4
#define PRODUCT_LIMIT 4
#define batched_visibility_products_float batched_visibility_products_float_4x4
#include "multi_products.inc"
#undef batched_visibility_products_float
#undef PRODUCT_LIMIT
#undef BAND_LIMIT
}  // namespace narrow_products
namespace wide_products {
#define BAND_LIMIT 8
#define PRODUCT_LIMIT 4
#define batched_visibility_products_float batched_visibility_products_float_8x4
#include "multi_products.inc"
#undef batched_visibility_products_float
#undef PRODUCT_LIMIT
#undef BAND_LIMIT
}  // namespace wide_products
#include "xbeam/cuda.hpp"
#include "xbeam/radio.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <tuple>
namespace xbeam {
namespace {
void check(cudaError_t status) {
    if (status != cudaSuccess) {
        throw RadioError(cudaGetErrorString(status));
    }
}
struct Memory {
    void* p = nullptr;
    ~Memory() {
        if (p) {
            cudaFree(p);
        }
    }
    template <class T>
    void upload(const std::vector<T>& values) {
        check(cudaMalloc(&p, values.size() * sizeof(T)));
        check(cudaMemcpy(
            p,
            values.data(),
            values.size() * sizeof(T),
            cudaMemcpyHostToDevice));
    }
};
struct Group {
    int channels, averaging, packs, band_limit = 0;
    Memory spectra, strides, left, right, band_counts, product_counts, rows, wa,
        wb, multiply, normalization;
};
bool same(const VisibilityProduct& a, const VisibilityProduct& b) {
    return a.spectrum_a == b.spectrum_a && a.spectrum_b == b.spectrum_b
        && a.weight_a == b.weight_a && a.weight_b == b.weight_b
        && a.stride_a == b.stride_a && a.stride_b == b.stride_b
        && a.channels == b.channels && a.averaging == b.averaging
        && a.inverse_normalization == b.inverse_normalization
        && a.multiply_weights == b.multiply_weights;
}
}  // namespace
struct CudaProducts::Impl {
    std::vector<VisibilityProduct> previous;
    std::vector<std::unique_ptr<Group>> groups;
    std::vector<int> singles;
    std::uintptr_t stream = 0;
    int product_limit;
    bool reuse_sources;
};
CudaProducts::CudaProducts(int products_per_pack, bool reuse_sources)
    : impl(std::make_unique<Impl>()) {
    if (products_per_pack != 4 && products_per_pack != 8) {
        throw RadioError(
            "native XMAC packs must contain four or eight products");
    }
    impl->product_limit = products_per_pack;
    impl->reuse_sources = reuse_sources;
}
CudaProducts::~CudaProducts() {
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(impl->stream));
}
void CudaProducts::configure(const std::vector<VisibilityProduct>& products) {
    auto& state = *impl;
    if (products.size() == state.previous.size()
        && std::equal(
            products.begin(), products.end(), state.previous.begin(), same)) {
        return;
    }
    check(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(state.stream)));
    state.groups.clear();
    state.singles.clear();
    state.previous = products;
    std::map<std::pair<int, int>, std::vector<int>> compatible;
    for (std::size_t k = 0; k < products.size(); ++k) {
        const auto& p = products[k];
        if (!p.spectrum_a || !p.spectrum_b || !p.weight_a || !p.weight_b
            || p.channels <= 0 || p.averaging <= 0 || p.channels % p.averaging
            || p.stride_a < p.channels || p.stride_b < p.channels
            || !std::isfinite(p.inverse_normalization)
            || p.inverse_normalization <= 0) {
            throw RadioError("invalid prepared visibility product");
        }
        if (p.averaging == 1 || p.averaging == 4 || p.averaging == 32) {
            compatible[{p.channels, p.averaging}].push_back(k);
        } else {
            state.singles.push_back(k);
        }
    }
    for (const auto& entry : compatible) {
        auto group = std::make_unique<Group>();
        group->channels = entry.first.first;
        group->averaging = entry.first.second;
        group->packs = 0;
        std::vector<unsigned long long> spectra, wa, wb;
        std::vector<long long> strides;
        std::vector<int> left, right, band_counts, product_counts, rows;
        std::vector<unsigned char> multiply;
        std::vector<double> normalizations;
        using Band = std::pair<std::uintptr_t, std::int64_t>;
        std::set<int> remaining(entry.second.begin(), entry.second.end());
        std::map<Band, std::vector<int>> incident;
        if (state.reuse_sources) {
            for (int index : entry.second) {
                const auto& p = products[index];
                incident[{p.spectrum_a, p.stride_a}].push_back(index);
                incident[{p.spectrum_b, p.stride_b}].push_back(index);
            }
        }
        while (!remaining.empty()) {
            std::vector<Band> bands;
            std::vector<int> items, ls, rs;
            std::set<int> nearby;
            auto insert_band = [&](Band band) {
                auto it = std::find(bands.begin(), bands.end(), band);
                if (it != bands.end()) {
                    return int(it - bands.begin());
                }
                bands.push_back(band);
                if (state.reuse_sources) {
                    for (int index : incident.at(band)) {
                        if (remaining.count(index)) {
                            nearby.insert(index);
                        }
                    }
                }
                return int(bands.size() - 1);
            };
            auto added_bands = [&](int index) {
                const auto& p = products[index];
                Band a{p.spectrum_a, p.stride_a}, b{p.spectrum_b, p.stride_b};
                return int(std::find(bands.begin(), bands.end(), a)
                           == bands.end())
                    + int(
                        b != a
                        && std::find(bands.begin(), bands.end(), b)
                            == bands.end());
            };
            while (!remaining.empty()
                && items.size() < std::size_t(state.product_limit)) {
                int id = *remaining.begin(), extra = added_bands(id);
                if (state.reuse_sources) {
                    // Search only the local source graph, independent of the
                    // number of unrelated frequencies or baseline products.
                    int best = 3;
                    for (int candidate : nearby) {
                        int added = added_bands(candidate);
                        if (bands.size() + added <= 8 && added < best) {
                            id = candidate;
                            extra = best = added;
                            if (!added) {
                                break;
                            }
                        }
                    }
                }
                const auto& p = products[id];
                Band a{p.spectrum_a, p.stride_a}, b{p.spectrum_b, p.stride_b};
                if (bands.size() + extra > 8) {
                    break;
                }
                ls.push_back(insert_band(a));
                rs.push_back(insert_band(b));
                items.push_back(id);
                remaining.erase(id);
                nearby.erase(id);
            }
            if (items.empty()) {
                throw RadioError("visibility pack cannot fit its source bands");
            }
            ++group->packs;
            group->band_limit = std::max(group->band_limit, int(bands.size()));
            band_counts.push_back(bands.size());
            product_counts.push_back(items.size());
            for (std::size_t k = 0; k < 8; ++k) {
                spectra.push_back(k < bands.size() ? bands[k].first : 0);
                strides.push_back(k < bands.size() ? bands[k].second : 0);
                if (k < items.size()) {
                    const auto& p = products[items[k]];
                    left.push_back(ls[k]);
                    right.push_back(rs[k]);
                    rows.push_back(items[k]);
                    wa.push_back(p.weight_a);
                    wb.push_back(p.weight_b);
                    multiply.push_back(p.multiply_weights);
                    normalizations.push_back(p.inverse_normalization);
                } else {
                    left.push_back(0);
                    right.push_back(0);
                    rows.push_back(0);
                    wa.push_back(0);
                    wb.push_back(0);
                    multiply.push_back(0);
                    normalizations.push_back(1);
                }
            }
        }
        group->spectra.upload(spectra);
        group->strides.upload(strides);
        group->left.upload(left);
        group->right.upload(right);
        group->band_counts.upload(band_counts);
        group->product_counts.upload(product_counts);
        group->rows.upload(rows);
        group->wa.upload(wa);
        group->wb.upload(wb);
        group->multiply.upload(multiply);
        group->normalization.upload(normalizations);
        state.groups.push_back(std::move(group));
    }
}
void CudaProducts::submit(
    int groups,
    std::uintptr_t starts,
    std::uintptr_t lengths,
    std::uintptr_t offsets,
    std::uintptr_t weight_offsets,
    std::uintptr_t output,
    std::uintptr_t weights,
    std::uintptr_t stream) {
    if (groups <= 0 || groups > 65535 || !starts || !lengths || !offsets
        || !weight_offsets || !output || !weights) {
        throw RadioError("invalid prepared XMAC group submission");
    }
    impl->stream = stream;
    auto work = reinterpret_cast<cudaStream_t>(stream);
    for (const auto& pointer : impl->groups) {
        const auto& g = *pointer;
        for (int begin = 0; begin < g.packs; begin += 65535) {
            int packs = std::min(g.packs - begin, 65535);
            cudaLaunchConfig_t config{};
            config.gridDim = dim3((g.channels + 31) / 32, groups, packs);
            config.blockDim = dim3(256);
            config.stream = work;
            auto launch = [&](auto kernel) {
                return cudaLaunchKernelEx(
                    &config,
                    kernel,
                    static_cast<unsigned long long*>(g.spectra.p),
                    static_cast<long long*>(g.strides.p),
                    static_cast<int*>(g.left.p),
                    static_cast<int*>(g.right.p),
                    static_cast<int*>(g.band_counts.p),
                    static_cast<int*>(g.product_counts.p),
                    static_cast<int*>(g.rows.p),
                    static_cast<unsigned long long*>(g.wa.p),
                    static_cast<unsigned long long*>(g.wb.p),
                    static_cast<unsigned char*>(g.multiply.p),
                    reinterpret_cast<int*>(starts),
                    reinterpret_cast<int*>(lengths),
                    reinterpret_cast<long long*>(offsets),
                    reinterpret_cast<long long*>(weight_offsets),
                    static_cast<double*>(g.normalization.p),
                    groups,
                    g.channels,
                    g.averaging,
                    begin,
                    reinterpret_cast<float2*>(output),
                    reinterpret_cast<double*>(weights));
            };
            if (impl->product_limit == 8) {
                check(launch(batched_visibility_products_float));
            } else if (g.band_limit <= 4) {
                check(launch(
                    narrow_products::batched_visibility_products_float_4x4));
            } else {
                check(launch(
                    wide_products::batched_visibility_products_float_8x4));
            }
        }
    }
    for (int index : impl->singles) {
        const auto& p = impl->previous[index];
        ProductLayout layout{p.channels,
            p.averaging,
            groups,
            p.stride_a,
            p.stride_b,
            p.inverse_normalization,
            p.multiply_weights};
        submit_product(
            layout,
            p.spectrum_a,
            p.spectrum_b,
            p.weight_a,
            p.weight_b,
            starts,
            lengths,
            offsets + std::size_t(index) * groups * 8,
            weight_offsets + std::size_t(index) * groups * 8,
            output,
            weights,
            stream);
    }
}
}  // namespace xbeam
