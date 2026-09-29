# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""CuPy kernels for Gaussian residuals and analytic normal equations."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._types import BackendArray, FitMode

_BLOCK_SIZE = 128
_NORMAL_COUNT = 44  # Eight gradient and 36 upper-triangular Hessian entries.

# Keep arithmetic in the input dtype and preserve the model's operation order,
# including physical-sigma derivatives followed by the log-sigma chain rule.
# Disable FMA to avoid contracting separate CuPy operations.
_CUDA_SOURCE = r"""
typedef SCALAR_TYPE T;
const int block_size = 128;
const int normal_count = 44;

struct Gaussian {
    T amplitude, sigma_x, sigma_y, inverse_x2, inverse_y2, cosine, sine;

    __device__ Gaussian(const T* parameters) {
        amplitude = parameters[0];
        sigma_x = exp(parameters[1]);
        sigma_y = exp(parameters[2]);
        const T valid_x = isfinite(sigma_x) && sigma_x > T(0)
            ? sigma_x : T(nan(""));
        const T valid_y = isfinite(sigma_y) && sigma_y > T(0)
            ? sigma_y : T(nan(""));
        inverse_x2 = T(1) / (valid_x * valid_x);
        inverse_y2 = T(1) / (valid_y * valid_y);
        cosine = cos(parameters[3]);
        sine = sin(parameters[3]);
    }

    template<bool derivatives>
    __device__ T star(T x, T y, T* derivative) const {
        const T rotated_x = cosine * x + sine * y;
        const T rotated_y = cosine * y - sine * x;
        const T exponent = T(-0.5) * (
            rotated_x * rotated_x * inverse_x2
            + rotated_y * rotated_y * inverse_y2);
        const T unit = exp(exponent);
        const T value = amplitude * unit;
        if (derivatives) {
            derivative[0] = unit;
            derivative[1] = value * rotated_x * rotated_x
                / pow(sigma_x, T(3));
            derivative[2] = value * rotated_y * rotated_y
                / pow(sigma_y, T(3));
            derivative[3] = -value * rotated_x * rotated_y
                * (inverse_x2 - inverse_y2);
            derivative[4] = value * (
                cosine * rotated_x * inverse_x2
                - sine * rotated_y * inverse_y2);
            derivative[5] = value * (
                sine * rotated_x * inverse_x2
                + cosine * rotated_y * inverse_y2);
        }
        return value;
    }
};

extern "C" __global__ void residual(
    const T* parameters, const long long* indices,
    const T* images, const T* weights, T* output,
    long long batch, long long height, long long width,
    long long observations) {
    const long long offset = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (offset >= batch * observations) return;
    const long long fit = offset / observations;
    const long long observation = offset % observations;
    const long long pixels = height * width;
    const int plane = observation / pixels;
    const long long pixel = observation % pixels;
    const T x = T(pixel % width) - T(width - 1) / T(2);
    const T y = T(pixel / width) - T(height - 1) / T(2);
    const T* p = parameters + fit * 8;
    const Gaussian model(p);
    const T positive = model.star<false>(x - p[4], y - p[5], nullptr);
    const T negative = model.star<false>(x - p[6], y - p[7], nullptr);
    const T prediction = plane == 0 ? positive - negative
        : (plane == 1 ? positive : negative);
    const long long source = indices[fit] * observations + observation;
    output[offset] = (prediction - images[source]) * weights[source];
}

__device__ T warp_sum(T value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        value += __shfl_down_sync(0xffffffff, value, offset);
    return value;
}

extern "C" __global__ void normal_partials(
    const T* parameters, const long long* indices,
    const T* residuals, const T* weights, T* partials,
    long long height, long long width, long long observations,
    long long tile_count) {
    const long long fit = blockIdx.x / tile_count;
    const long long tile = blockIdx.x % tile_count;
    const long long observation = tile * block_size + threadIdx.x;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    __shared__ T warp_partials[block_size / 32][normal_count];
    T derivative[8] = {};
    T residual_value = T(0);

    if (observation < observations) {
        const long long pixels = height * width;
        const int plane = observation / pixels;
        const long long pixel = observation % pixels;
        const T x = T(pixel % width) - T(width - 1) / T(2);
        const T y = T(pixel / width) - T(height - 1) / T(2);
        const T* p = parameters + fit * 8;
        const Gaussian model(p);
        T positive[6], negative[6];
        model.star<true>(x - p[4], y - p[5], positive);
        model.star<true>(x - p[6], y - p[7], negative);
        #pragma unroll
        for (int parameter = 0; parameter < 4; ++parameter)
            derivative[parameter] = plane == 0
                ? positive[parameter] - negative[parameter]
                : (plane == 1 ? positive[parameter] : negative[parameter]);
        derivative[4] = plane < 2 ? positive[4] : T(0);
        derivative[5] = plane < 2 ? positive[5] : T(0);
        derivative[6] = plane == 0 ? -negative[4]
            : (plane == 2 ? negative[4] : T(0));
        derivative[7] = plane == 0 ? -negative[5]
            : (plane == 2 ? negative[5] : T(0));
        derivative[1] *= model.sigma_x;
        derivative[2] *= model.sigma_y;
        const T weight = weights[indices[fit] * observations + observation];
        #pragma unroll
        for (int parameter = 0; parameter < 8; ++parameter)
            derivative[parameter] *= weight;
        residual_value = residuals[fit * observations + observation];
    }

    #pragma unroll
    for (int parameter = 0; parameter < 8; ++parameter) {
        const T total = warp_sum(derivative[parameter] * residual_value);
        if (lane == 0) warp_partials[warp][parameter] = total;
    }
    int component = 8;
    #pragma unroll
    for (int row = 0; row < 8; ++row) {
        #pragma unroll
        for (int column = row; column < 8; ++column) {
            const T total = warp_sum(derivative[row] * derivative[column]);
            if (lane == 0) warp_partials[warp][component] = total;
            ++component;
        }
    }
    __syncthreads();
    if (threadIdx.x < normal_count) {
        T total = warp_partials[0][threadIdx.x];
        #pragma unroll
        for (int source_warp = 1; source_warp < block_size / 32;
             ++source_warp)
            total += warp_partials[source_warp][threadIdx.x];
        partials[(fit * tile_count + tile) * normal_count + threadIdx.x]
            = total;
    }
}

extern "C" __global__ void normal_finish(
    const T* partials, T* gradients, T* hessians, long long tile_count) {
    const long long fit = blockIdx.x;
    const int component = threadIdx.x;
    if (component >= normal_count) return;
    T total = T(0);
    for (long long tile = 0; tile < tile_count; ++tile)
        total += partials[
            (fit * tile_count + tile) * normal_count + component];
    if (component < 8) {
        gradients[fit * 8 + component] = total;
    } else {
        int column = component - 8;
        int row = 0;
        while (column >= 8 - row) {
            column -= 8 - row;
            ++row;
        }
        column += row;
        hessians[fit * 64 + row * 8 + column] = total;
        hessians[fit * 64 + column * 8 + row] = total;
    }
}
"""


class GaussianFusedProblem:
    """Own CuPy Gaussian callbacks and input storage for a single fit call.

    The caller supplies sanitized images and precomputed multiplicative
    weights. Parameters use log sigma, and indices refer to original input
    rows. Callbacks return fresh arrays because the LM solver retains earlier
    residuals while evaluating trial steps. CuPy is supplied by the resolved
    backend; importing this module does not import CuPy or initialize CUDA.
    """

    def __init__(
        self,
        cp: Any,
        images: BackendArray,
        weights: BackendArray,
        image_shape: tuple[int, int],
        mode: FitMode,
    ) -> None:
        if mode not in ("difference", "split"):
            raise ValueError("mode must be 'difference' or 'split'")
        height, width = image_shape
        if height < 1 or width < 1:
            raise ValueError("image_shape dimensions must be positive")
        self._dtype = np.dtype(images.dtype)
        if self._dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise TypeError(
                "fused Gaussian kernels require float32 or float64"
            )
        expected_shape = (
            (height, width) if mode == "difference" else (3, height, width)
        )
        if (
            images.shape[1:] != expected_shape
            or weights.shape != images.shape
        ):
            raise ValueError(
                "images and weights must match the Gaussian mode"
            )
        self._cp = cp
        self._height = np.int64(height)
        self._width = np.int64(width)
        self._observations = int(np.prod(expected_shape))
        self._tiles = (self._observations + _BLOCK_SIZE - 1) // _BLOCK_SIZE
        self._partial_capacity = 0
        self._partials: Any = None
        self._images = cp.ascontiguousarray(images)
        self._weights = cp.ascontiguousarray(weights, dtype=self._dtype)
        scalar_type = (
            "float" if self._dtype == np.dtype(np.float32) else "double"
        )
        self._module = cp.RawModule(
            code=_CUDA_SOURCE.replace("SCALAR_TYPE", scalar_type),
            options=("--std=c++11", "--fmad=false"),
        )
        self._residual_kernel = self._module.get_function("residual")
        self._normal_kernel = self._module.get_function("normal_partials")
        self._finish_kernel = self._module.get_function("normal_finish")

    def _inputs(
        self, parameters: BackendArray, indices: BackendArray
    ) -> tuple[Any, Any]:
        batch = parameters.shape[0]
        if parameters.shape != (batch, 8):
            raise ValueError("Gaussian parameters must have shape (batch, 8)")
        if indices.shape != (batch,):
            raise ValueError("indices must contain one original row per fit")
        return (
            self._cp.ascontiguousarray(parameters, dtype=self._dtype),
            self._cp.ascontiguousarray(indices, dtype=np.int64),
        )

    def residual(
        self, parameters: BackendArray, *, indices: BackendArray
    ) -> BackendArray:
        """Evaluate weighted residuals with one kernel launch."""

        parameters, indices = self._inputs(parameters, indices)
        batch = parameters.shape[0]
        output = self._cp.empty(
            (batch, self._observations), dtype=self._dtype
        )
        if batch:
            blocks = (
                batch * self._observations + _BLOCK_SIZE - 1
            ) // _BLOCK_SIZE
            self._residual_kernel(
                (blocks,),
                (_BLOCK_SIZE,),
                (
                    parameters,
                    indices,
                    self._images,
                    self._weights,
                    output,
                    np.int64(batch),
                    self._height,
                    self._width,
                    np.int64(self._observations),
                ),
            )
        return output

    def normal_equations(
        self,
        parameters: BackendArray,
        residuals: BackendArray,
        *,
        indices: BackendArray,
    ) -> tuple[BackendArray, BackendArray]:
        """Accumulate gradient and Hessian without a global Jacobian array."""

        parameters, indices = self._inputs(parameters, indices)
        batch = parameters.shape[0]
        if residuals.shape != (batch, self._observations):
            raise ValueError("residuals must match the active observations")
        residuals = self._cp.ascontiguousarray(residuals, dtype=self._dtype)
        gradient = self._cp.empty((batch, 8), dtype=self._dtype)
        hessian = self._cp.empty((batch, 8, 8), dtype=self._dtype)
        if batch:
            tiles = self._tiles
            if batch > self._partial_capacity:
                self._partials = self._cp.empty(
                    (batch, tiles, _NORMAL_COUNT), dtype=self._dtype
                )
                self._partial_capacity = batch
            # Calls belong to one fit and execute on its current stream. The
            # previous reduction finishes before this scratch is overwritten.
            partials = self._partials
            self._normal_kernel(
                (batch * tiles,),
                (_BLOCK_SIZE,),
                (
                    parameters,
                    indices,
                    residuals,
                    self._weights,
                    partials,
                    self._height,
                    self._width,
                    np.int64(self._observations),
                    np.int64(tiles),
                ),
            )
            self._finish_kernel(
                (batch,),
                (64,),
                (partials, gradient, hessian, np.int64(tiles)),
            )
        return gradient, hessian
