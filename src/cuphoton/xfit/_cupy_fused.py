# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""CuPy kernels for Gaussian residuals and analytic Jacobians."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._types import BackendArray, FitMode

_BLOCK_SIZE = 128

# Match CuPy's separate ufunc rounding, including the physical-sigma
# derivatives before the log-sigma chain rule. Explicit RN intrinsics preserve
# these boundaries while allowing normal FMA use inside libdevice functions:
# disabling FMA globally can change pow results and ill-conditioned fits.
_CUDA_SOURCE = r"""
typedef SCALAR_TYPE T;

__device__ float rn_add(float a, float b) { return __fadd_rn(a, b); }
__device__ double rn_add(double a, double b) { return __dadd_rn(a, b); }
__device__ float rn_sub(float a, float b) { return __fsub_rn(a, b); }
__device__ double rn_sub(double a, double b) { return __dsub_rn(a, b); }
__device__ float rn_mul(float a, float b) { return __fmul_rn(a, b); }
__device__ double rn_mul(double a, double b) { return __dmul_rn(a, b); }
__device__ float rn_div(float a, float b) { return __fdiv_rn(a, b); }
__device__ double rn_div(double a, double b) { return __ddiv_rn(a, b); }


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
        inverse_x2 = rn_div(T(1), rn_mul(valid_x, valid_x));
        inverse_y2 = rn_div(T(1), rn_mul(valid_y, valid_y));
        cosine = cos(parameters[3]);
        sine = sin(parameters[3]);
    }

    template<bool derivatives>
    __device__ T star(T x, T y, T* derivative) const {
        const T rotated_x = rn_add(rn_mul(cosine, x), rn_mul(sine, y));
        const T rotated_y = rn_sub(rn_mul(cosine, y), rn_mul(sine, x));
        const T exponent = rn_mul(T(-0.5), rn_add(
            rn_mul(rn_mul(rotated_x, rotated_x), inverse_x2),
            rn_mul(rn_mul(rotated_y, rotated_y), inverse_y2)));
        const T unit = exp(exponent);
        const T value = rn_mul(amplitude, unit);
        if (derivatives) {
            derivative[0] = unit;
            derivative[1] = rn_div(
                rn_mul(rn_mul(value, rotated_x), rotated_x),
                pow(sigma_x, T(3)));
            derivative[2] = rn_div(
                rn_mul(rn_mul(value, rotated_y), rotated_y),
                pow(sigma_y, T(3)));
            derivative[3] = rn_mul(
                rn_mul(rn_mul(-value, rotated_x), rotated_y),
                rn_sub(inverse_x2, inverse_y2));
            derivative[4] = rn_mul(value, rn_sub(
                rn_mul(rn_mul(cosine, rotated_x), inverse_x2),
                rn_mul(rn_mul(sine, rotated_y), inverse_y2)));
            derivative[5] = rn_mul(value, rn_add(
                rn_mul(rn_mul(sine, rotated_x), inverse_x2),
                rn_mul(rn_mul(cosine, rotated_y), inverse_y2)));
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
    if (offset >= batch * observations) {
        return;
    }
    const long long fit = offset / observations;
    const long long observation = offset % observations;
    const long long pixels = height * width;
    const int plane = observation / pixels;
    const long long pixel = observation % pixels;
    const T x = rn_sub(T(pixel % width), rn_div(T(width - 1), T(2)));
    const T y = rn_sub(T(pixel / width), rn_div(T(height - 1), T(2)));
    const T* p = parameters + fit * 8;
    const Gaussian model(p);
    const T positive = model.star<false>(
        rn_sub(x, p[4]), rn_sub(y, p[5]), nullptr);
    const T negative = model.star<false>(
        rn_sub(x, p[6]), rn_sub(y, p[7]), nullptr);
    const T prediction = plane == 0 ? rn_sub(positive, negative)
        : (plane == 1 ? positive : negative);
    const long long source = indices[fit] * observations + observation;
    output[offset] = rn_mul(
        rn_sub(prediction, images[source]), weights[source]);
}


extern "C" __global__ void materialized_jacobian(
    const T* parameters, const long long* indices,
    const T* weights, T* output,
    long long batch, long long height, long long width,
    long long observations) {
    const long long offset = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (offset >= batch * observations) {
        return;
    }
    const long long fit = offset / observations;
    const long long observation = offset % observations;
    const long long pixels = height * width;
    const int plane = observation / pixels;
    const long long pixel = observation % pixels;
    const T x = rn_sub(T(pixel % width), rn_div(T(width - 1), T(2)));
    const T y = rn_sub(T(pixel / width), rn_div(T(height - 1), T(2)));
    const T* p = parameters + fit * 8;
    const Gaussian model(p);
    T positive[6], negative[6], derivative[8];
    model.star<true>(rn_sub(x, p[4]), rn_sub(y, p[5]), positive);
    model.star<true>(rn_sub(x, p[6]), rn_sub(y, p[7]), negative);
    #pragma unroll
    for (int parameter = 0; parameter < 4; ++parameter) {
        derivative[parameter] = plane == 0
            ? rn_sub(positive[parameter], negative[parameter])
            : (plane == 1 ? positive[parameter] : negative[parameter]);
    }
    derivative[4] = plane < 2 ? positive[4] : T(0);
    derivative[5] = plane < 2 ? positive[5] : T(0);
    derivative[6] = plane == 0 ? -negative[4]
        : (plane == 2 ? negative[4] : T(0));
    derivative[7] = plane == 0 ? -negative[5]
        : (plane == 2 ? negative[5] : T(0));
    derivative[1] = rn_mul(derivative[1], model.sigma_x);
    derivative[2] = rn_mul(derivative[2], model.sigma_y);
    const T weight = weights[indices[fit] * observations + observation];
    #pragma unroll
    for (int parameter = 0; parameter < 8; ++parameter) {
        derivative[parameter] = rn_mul(derivative[parameter], weight);
        output[(fit * 8 + parameter) * observations + observation]
            = derivative[parameter];
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
        self._images = cp.ascontiguousarray(images)
        self._weights = cp.ascontiguousarray(weights, dtype=self._dtype)
        scalar_type = (
            "float" if self._dtype == np.dtype(np.float32) else "double"
        )
        self._module = cp.RawModule(
            code=_CUDA_SOURCE.replace("SCALAR_TYPE", scalar_type),
            options=("--std=c++11", "--fmad=true"),
        )
        self._residual_kernel = self._module.get_function("residual")
        self._jacobian_kernel = self._module.get_function(
            "materialized_jacobian"
        )

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

    def weighted_log_jacobian(
        self, parameters: BackendArray, *, indices: BackendArray
    ) -> BackendArray:
        """Materialize a weighted log-sigma Jacobian in baseline layout."""

        parameters, indices = self._inputs(parameters, indices)
        batch = parameters.shape[0]
        jacobian = self._cp.empty(
            (batch, 8, self._observations), dtype=self._dtype, order="C"
        )
        if batch:
            blocks = (
                batch * self._observations + _BLOCK_SIZE - 1
            ) // _BLOCK_SIZE
            self._jacobian_kernel(
                (blocks,),
                (_BLOCK_SIZE,),
                (
                    parameters,
                    indices,
                    self._weights,
                    jacobian,
                    np.int64(batch),
                    self._height,
                    self._width,
                    np.int64(self._observations),
                ),
            )
        return jacobian

    def normal_equations(
        self,
        parameters: BackendArray,
        residuals: BackendArray,
        *,
        indices: BackendArray,
    ) -> tuple[BackendArray, BackendArray]:
        """Form normal equations with the baseline CuPy contractions."""

        batch = parameters.shape[0]
        if residuals.shape != (batch, self._observations):
            raise ValueError("residuals must match the active observations")
        jacobian = self.weighted_log_jacobian(parameters, indices=indices)
        residuals = self._cp.ascontiguousarray(residuals, dtype=self._dtype)
        if not batch:
            return (
                self._cp.empty((0, 8), dtype=self._dtype),
                self._cp.empty((0, 8, 8), dtype=self._dtype),
            )
        # Keep the solver's contraction order for ill-conditioned fits.
        gradient = self._cp.einsum("knm,km->kn", jacobian, residuals)
        hessian = self._cp.einsum("knm,kpm->knp", jacobian, jacobian)
        return gradient, hessian
