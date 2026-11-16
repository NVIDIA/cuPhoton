// SPDX-License-Identifier: Apache-2.0
#include "xbeam/cuda.hpp"
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace {
void check(cudaError_t c) {
    if (c != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(c));
    }
}
struct Buffer {
    void* p = nullptr;
    explicit Buffer(std::size_t n) {
        check(cudaMalloc(&p, n));
    }
    ~Buffer() {
        if (p) {
            cudaFree(p);
        }
    }
    std::uintptr_t address() {
        return reinterpret_cast<std::uintptr_t>(p);
    }
};
}  // namespace
int main() {
    try {
        for (bool reuse : {false, true}) {
            for (bool narrow : {false, true}) {
                for (int pack_products : {8, 4}) {
                    for (int avg : {1, 4, 32, 7}) {
                        const int rows = 8, channels = 224, sa = 229, sb = 231,
                                  out = channels / avg;
                        std::vector<float> a(rows * sa * 2, 0),
                            b(rows * sb * 2, 0);
                        for (int row = 0; row < rows; ++row) {
                            for (int c = 0; c < sa; ++c) {
                                a[(row * sa + c) * 2] = 2;
                                a[(row * sa + c) * 2 + 1] = 3;
                            }
                        }
                        for (int row = 0; row < rows; ++row) {
                            for (int c = 0; c < sb; ++c) {
                                b[(row * sb + c) * 2] = 4;
                                b[(row * sb + c) * 2 + 1] = -1;
                            }
                        }
                        std::vector<double> wa(rows, .5), wb(rows, .25);
                        int starts[] = {0, 4}, lengths[] = {4, 4};
                        long long offsets[] = {7, 7 + out},
                                  weight_offsets[] = {2, 3};
                        Buffer da(a.size() * 4), db(b.size() * 4),
                            dwa(rows * 8), dwb(rows * 8), ds(sizeof(starts)),
                            dl(sizeof(lengths)), doo(sizeof(offsets)),
                            dwo(sizeof(weight_offsets)),
                            result((7 + out * 2) * 8), exposure(4 * 8);
                        check(cudaMemcpy(
                            da.p,
                            a.data(),
                            a.size() * 4,
                            cudaMemcpyHostToDevice));
                        check(cudaMemcpy(
                            db.p,
                            b.data(),
                            b.size() * 4,
                            cudaMemcpyHostToDevice));
                        check(cudaMemcpy(
                            dwa.p,
                            wa.data(),
                            rows * 8,
                            cudaMemcpyHostToDevice));
                        check(cudaMemcpy(
                            dwb.p,
                            wb.data(),
                            rows * 8,
                            cudaMemcpyHostToDevice));
                        check(cudaMemcpy(
                            ds.p,
                            starts,
                            sizeof(starts),
                            cudaMemcpyHostToDevice));
                        check(cudaMemcpy(
                            dl.p,
                            lengths,
                            sizeof(lengths),
                            cudaMemcpyHostToDevice));
                        check(cudaMemcpy(
                            doo.p,
                            offsets,
                            sizeof(offsets),
                            cudaMemcpyHostToDevice));
                        check(cudaMemcpy(
                            dwo.p,
                            weight_offsets,
                            sizeof(weight_offsets),
                            cudaMemcpyHostToDevice));
                        xbeam::ProductLayout g{
                            channels, avg, 2, sa, sb, .5, true};
                        xbeam::submit_product(
                            g,
                            da.address(),
                            db.address(),
                            dwa.address(),
                            dwb.address(),
                            ds.address(),
                            dl.address(),
                            doo.address(),
                            dwo.address(),
                            result.address(),
                            exposure.address(),
                            0);
                        std::vector<float> host((7 + out * 2) * 2);
                        double weights[4];
                        check(cudaMemcpy(
                            host.data(),
                            result.p,
                            host.size() * 4,
                            cudaMemcpyDeviceToHost));
                        check(cudaMemcpy(
                            weights,
                            exposure.p,
                            sizeof(weights),
                            cudaMemcpyDeviceToHost));
                        for (int group = 0; group < 2; ++group) {
                            if (weights[2 + group] != .5) {
                                throw std::runtime_error(
                                    "XMAC weight mismatch");
                            }
                            for (int c = 0; c < out; ++c) {
                                auto index = (7 + group * out + c) * 2;
                                if (host[index] != 10
                                    || host[index + 1] != 28) {
                                    throw std::runtime_error(
                                        "XMAC spectrum mismatch");
                                }
                            }
                        }
                        // Distinct rows/channels make incorrect source reuse and
                        // output permutations visible to the independent CPU sum.
                        for (int row = 0; row < rows; ++row) {
                            for (int c = 0; c < sa; ++c) {
                                a[(row * sa + c) * 2] =
                                    2 + row / 32.f + c / 256.f;
                                a[(row * sa + c) * 2 + 1] =
                                    3 + c / 128.f - row / 64.f;
                            }
                            for (int c = 0; c < sb; ++c) {
                                b[(row * sb + c) * 2] =
                                    4 - row / 64.f + c / 512.f;
                                b[(row * sb + c) * 2 + 1] =
                                    -1 + row / 128.f - c / 1024.f;
                            }
                        }
                        check(cudaMemcpy(
                            da.p,
                            a.data(),
                            a.size() * 4,
                            cudaMemcpyHostToDevice));
                        check(cudaMemcpy(
                            db.p,
                            b.data(),
                            b.size() * 4,
                            cudaMemcpyHostToDevice));
                        const int product_count = 17;
                        std::vector<xbeam::VisibilityProduct> products;
                        std::vector<long long> packed_offsets,
                            packed_weight_offsets;
                        for (int product = 0; product < product_count;
                            ++product) {
                            const auto aa = da.address()
                                + (narrow ? 0 : product % 5) * 8,
                                       bb = db.address()
                                + (narrow ? 0 : product % 7) * 8;
                            bool swap = product % 2;
                            products.push_back(
                                {swap ? bb : aa,
                                    swap ? aa : bb,
                                    swap ? dwb.address() : dwa.address(),
                                    swap ? dwa.address() : dwb.address(),
                                    swap ? sb : sa,
                                    swap ? sa : sb,
                                    channels,
                                    avg,
                                    product % 3 ? .5 : .25,
                                    !swap});
                            for (int group = 0; group < 2; ++group) {
                                packed_offsets.push_back(
                                    7 + (product * 2 + group) * out);
                                packed_weight_offsets.push_back(
                                    2 + product * 2 + group);
                            }
                        }
                        Buffer po(packed_offsets.size() * 8),
                            pw(packed_weight_offsets.size() * 8),
                            packed_result((7 + product_count * 2 * out) * 8),
                            packed_weights((2 + product_count * 2) * 8);
                        check(cudaMemcpy(
                            po.p,
                            packed_offsets.data(),
                            packed_offsets.size() * 8,
                            cudaMemcpyHostToDevice));
                        check(cudaMemcpy(
                            pw.p,
                            packed_weight_offsets.data(),
                            packed_weight_offsets.size() * 8,
                            cudaMemcpyHostToDevice));
                        auto expected = [&](int product,
                                            int group,
                                            int channel) {
                            double real = 0, imag = 0;
                            for (int row = starts[group];
                                row < starts[group] + lengths[group];
                                ++row) {
                                for (int c = channel * avg;
                                    c < (channel + 1) * avg;
                                    ++c) {
                                    auto ai =
                                        (row * sa + (narrow ? 0 : product % 5)
                                            + c)
                                        * 2;
                                    auto bi =
                                        (row * sb + (narrow ? 0 : product % 7)
                                            + c)
                                        * 2;
                                    double ar = a[ai], av = a[ai + 1],
                                           br = b[bi], bv = b[bi + 1];
                                    real += ar * br + av * bv;
                                    imag += av * br - ar * bv;
                                }
                            }
                            double scale =
                                products[product].inverse_normalization / avg;
                            return std::make_pair(
                                real * scale,
                                imag * scale * (product % 2 ? -1 : 1));
                        };
                        xbeam::CudaProducts prepared(pack_products, reuse);
                        prepared.configure(products);
                        prepared.configure(products);
                        prepared.submit(
                            2,
                            ds.address(),
                            dl.address(),
                            po.address(),
                            pw.address(),
                            packed_result.address(),
                            packed_weights.address(),
                            0);
                        std::vector<float> packed_host(
                            (7 + product_count * 2 * out) * 2);
                        std::vector<double> packed_exposure(
                            2 + product_count * 2);
                        check(cudaMemcpy(
                            packed_host.data(),
                            packed_result.p,
                            packed_host.size() * 4,
                            cudaMemcpyDeviceToHost));
                        check(cudaMemcpy(
                            packed_exposure.data(),
                            packed_weights.p,
                            packed_exposure.size() * 8,
                            cudaMemcpyDeviceToHost));
                        for (int product = 0; product < product_count;
                            ++product) {
                            for (int group = 0; group < 2; ++group) {
                                double expected_weight = product % 2 ? 1. : .5;
                                if (packed_exposure[2 + product * 2 + group]
                                    != expected_weight) {
                                    throw std::runtime_error(
                                        "prepared XMAC weight mismatch");
                                }
                                for (int c = 0; c < out; ++c) {
                                    auto index =
                                        (7 + (product * 2 + group) * out + c)
                                        * 2;
                                    auto truth = expected(product, group, c);
                                    if (std::abs(
                                            packed_host[index] - truth.first)
                                            > 5e-4
                                        || std::abs(
                                               packed_host[index + 1]
                                               - truth.second)
                                            > 5e-4) {
                                        throw std::runtime_error(
                                            "prepared XMAC spectrum mismatch");
                                    }
                                }
                            }
                        }
                        products[0].inverse_normalization = .5;
                        prepared.configure(products);
                        prepared.submit(
                            2,
                            ds.address(),
                            dl.address(),
                            po.address(),
                            pw.address(),
                            packed_result.address(),
                            packed_weights.address(),
                            0);
                        check(cudaMemcpy(
                            packed_host.data(),
                            packed_result.p,
                            packed_host.size() * 4,
                            cudaMemcpyDeviceToHost));
                        auto refreshed = expected(0, 0, 0);
                        if (std::abs(packed_host[14] - refreshed.first) > 5e-4
                            || std::abs(packed_host[15] - refreshed.second)
                                > 5e-4) {
                            throw std::runtime_error(
                                "prepared XMAC descriptor refresh mismatch");
                        }
                        std::cout
                            << "averaging=" << avg
                            << " products/pack=" << pack_products
                            << " narrow=" << narrow << " reuse=" << reuse
                            << " strided grouped and prepared XMAC PASS\n";
                    }
                }
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
}
