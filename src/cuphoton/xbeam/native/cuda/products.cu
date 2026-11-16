// SPDX-License-Identifier: Apache-2.0
#include "products.inc"
#include "xbeam/cuda.hpp"
#include "xbeam/radio.hpp"
#include <cmath>
namespace xbeam {
void submit_product(
    const ProductLayout& g,
    std::uintptr_t aa,
    std::uintptr_t bb,
    std::uintptr_t wa,
    std::uintptr_t wb,
    std::uintptr_t starts,
    std::uintptr_t lengths,
    std::uintptr_t offsets,
    std::uintptr_t weight_offsets,
    std::uintptr_t output,
    std::uintptr_t weights,
    std::uintptr_t stream) {
    if (g.channels <= 0 || g.averaging <= 0 || g.channels % g.averaging
        || g.groups <= 0 || g.groups > 65535 || g.stride_a < g.channels
        || g.stride_b < g.channels || !std::isfinite(g.inverse_normalization)
        || g.inverse_normalization <= 0 || !aa || !bb || !wa || !wb || !starts
        || !lengths || !offsets || !weight_offsets || !output || !weights) {
        throw RadioError("invalid native XMAC submission");
    }
    const auto* a = reinterpret_cast<const float2*>(aa);
    const auto* b = reinterpret_cast<const float2*>(bb);
    const auto* weight_a = reinterpret_cast<const double*>(wa);
    const auto* weight_b = reinterpret_cast<const double*>(wb);
    const auto* first = reinterpret_cast<const int*>(starts);
    const auto* count = reinterpret_cast<const int*>(lengths);
    const auto* position = reinterpret_cast<const long long*>(offsets);
    const auto* wp = reinterpret_cast<const long long*>(weight_offsets);
    auto* result = reinterpret_cast<float2*>(output);
    auto* exposure = reinterpret_cast<double*>(weights);
    const auto work = reinterpret_cast<cudaStream_t>(stream);
    const dim3 grid((g.channels + 31) / 32, g.groups);
    switch (g.averaging) {
    case 1:
        batched_visibility_product_float_avg1<<<grid, 256, 0, work>>>(
            a,
            b,
            g.stride_a,
            g.stride_b,
            weight_a,
            weight_b,
            g.multiply_weights,
            first,
            count,
            position,
            wp,
            g.channels,
            g.inverse_normalization,
            result,
            exposure);
        break;
    case 4:
        batched_visibility_product_float_avg4<<<grid, 256, 0, work>>>(
            a,
            b,
            g.stride_a,
            g.stride_b,
            weight_a,
            weight_b,
            g.multiply_weights,
            first,
            count,
            position,
            wp,
            g.channels,
            g.inverse_normalization,
            result,
            exposure);
        break;
    case 32:
        batched_visibility_product_float_avg32<<<grid, 256, 0, work>>>(
            a,
            b,
            g.stride_a,
            g.stride_b,
            weight_a,
            weight_b,
            g.multiply_weights,
            first,
            count,
            position,
            wp,
            g.channels,
            g.inverse_normalization,
            result,
            exposure);
        break;
    default:
        batched_visibility_product_float_generic<<<
            dim3((g.channels / g.averaging + 255) / 256, g.groups),
            256,
            0,
            work>>>(
            a,
            b,
            g.stride_a,
            g.stride_b,
            weight_a,
            weight_b,
            g.multiply_weights,
            first,
            count,
            position,
            wp,
            g.channels,
            g.averaging,
            g.inverse_normalization,
            result,
            exposure);
    }
    auto status = cudaGetLastError();
    if (status != cudaSuccess) {
        throw RadioError(cudaGetErrorString(status));
    }
}
}  // namespace xbeam
