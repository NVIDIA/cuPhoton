# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Optional ``cuda.tile`` kernels for xFit."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

from ._types import BackendArray, FitMode

_PARAMETER_COUNT = 8
_NORMAL_WIDTH = 16
_OBSERVATIONS_PER_TILE = 32
ct: Any | None = None


def _load_cutile() -> tuple[Any, Any]:
    global ct
    try:
        import cupy as cp

        if ct is None:
            import cuda.tile as cutile

            ct = cutile
    except Exception as exc:
        raise ImportError(
            "backend='cutile' requires CuPy and cuda-tile; install "
            "'cuphoton[cutile]'"
        ) from exc
    return cp, ct


@lru_cache(maxsize=None)
def _gaussian_normal_equations_kernel() -> Any:
    _, ct = _load_cutile()

    @ct.kernel
    def kernel(
        parameters,
        residuals,
        weights,
        gradients,
        hessians,
        height: ct.Constant[int],
        width: ct.Constant[int],
        plane_count: ct.Constant[int],
        observation_tile_count: ct.Constant[int],
    ):
        block = ct.bid(0)
        parameter_tile = ct.load(
            parameters,
            index=(block, 0),
            shape=(1, _PARAMETER_COUNT),
        )
        amplitude = ct.extract(parameter_tile, (0, 0), (1, 1))
        sigma_x = ct.exp(ct.extract(parameter_tile, (0, 1), (1, 1)))
        sigma_y = ct.exp(ct.extract(parameter_tile, (0, 2), (1, 1)))
        theta = ct.extract(parameter_tile, (0, 3), (1, 1))
        x_positive = ct.extract(parameter_tile, (0, 4), (1, 1))
        y_positive = ct.extract(parameter_tile, (0, 5), (1, 1))
        x_negative = ct.extract(parameter_tile, (0, 6), (1, 1))
        y_negative = ct.extract(parameter_tile, (0, 7), (1, 1))
        cosine = ct.cos(theta)
        sine = ct.sin(theta)
        inverse_x_squared = 1.0 / (sigma_x * sigma_x)
        inverse_y_squared = 1.0 / (sigma_y * sigma_y)
        pixel_count = height * width
        normal = ct.zeros(
            (_NORMAL_WIDTH, _NORMAL_WIDTH), dtype=parameters.dtype
        )

        for observation_tile in range(observation_tile_count):
            observation = (
                observation_tile * _OBSERVATIONS_PER_TILE
                + ct.arange(_OBSERVATIONS_PER_TILE, dtype=ct.int32)
            )
            plane = observation // pixel_count
            pixel = observation - plane * pixel_count
            y = pixel // width
            x = pixel - y * width
            x_coordinate = x.astype(parameters.dtype) - (width - 1) / 2
            y_coordinate = y.astype(parameters.dtype) - (height - 1) / 2

            x_hat_positive = x_coordinate - x_positive
            y_hat_positive = y_coordinate - y_positive
            x_rotated_positive = (
                cosine * x_hat_positive + sine * y_hat_positive
            )
            y_rotated_positive = (
                cosine * y_hat_positive - sine * x_hat_positive
            )
            unit_positive = ct.exp(
                -0.5
                * (
                    x_rotated_positive
                    * x_rotated_positive
                    * inverse_x_squared
                    + y_rotated_positive
                    * y_rotated_positive
                    * inverse_y_squared
                )
            )
            star_positive = amplitude * unit_positive
            derivatives_positive = (
                unit_positive,
                star_positive
                * x_rotated_positive
                * x_rotated_positive
                * inverse_x_squared,
                star_positive
                * y_rotated_positive
                * y_rotated_positive
                * inverse_y_squared,
                -star_positive
                * x_rotated_positive
                * y_rotated_positive
                * (inverse_x_squared - inverse_y_squared),
                star_positive
                * (
                    cosine * x_rotated_positive * inverse_x_squared
                    - sine * y_rotated_positive * inverse_y_squared
                ),
                star_positive
                * (
                    sine * x_rotated_positive * inverse_x_squared
                    + cosine * y_rotated_positive * inverse_y_squared
                ),
            )

            x_hat_negative = x_coordinate - x_negative
            y_hat_negative = y_coordinate - y_negative
            x_rotated_negative = (
                cosine * x_hat_negative + sine * y_hat_negative
            )
            y_rotated_negative = (
                cosine * y_hat_negative - sine * x_hat_negative
            )
            unit_negative = ct.exp(
                -0.5
                * (
                    x_rotated_negative
                    * x_rotated_negative
                    * inverse_x_squared
                    + y_rotated_negative
                    * y_rotated_negative
                    * inverse_y_squared
                )
            )
            star_negative = amplitude * unit_negative
            derivatives_negative = (
                unit_negative,
                star_negative
                * x_rotated_negative
                * x_rotated_negative
                * inverse_x_squared,
                star_negative
                * y_rotated_negative
                * y_rotated_negative
                * inverse_y_squared,
                -star_negative
                * x_rotated_negative
                * y_rotated_negative
                * (inverse_x_squared - inverse_y_squared),
                star_negative
                * (
                    cosine * x_rotated_negative * inverse_x_squared
                    - sine * y_rotated_negative * inverse_y_squared
                ),
                star_negative
                * (
                    sine * x_rotated_negative * inverse_x_squared
                    + cosine * y_rotated_negative * inverse_y_squared
                ),
            )

            if plane_count == 1:
                derivative_0 = (
                    derivatives_positive[0] - derivatives_negative[0]
                )
                derivative_1 = (
                    derivatives_positive[1] - derivatives_negative[1]
                )
                derivative_2 = (
                    derivatives_positive[2] - derivatives_negative[2]
                )
                derivative_3 = (
                    derivatives_positive[3] - derivatives_negative[3]
                )
                derivative_4 = derivatives_positive[4]
                derivative_5 = derivatives_positive[5]
                derivative_6 = -derivatives_negative[4]
                derivative_7 = -derivatives_negative[5]
            else:
                zero = ct.zeros(
                    (_OBSERVATIONS_PER_TILE,), dtype=parameters.dtype
                )
                derivative_0 = ct.where(
                    plane == 0,
                    derivatives_positive[0] - derivatives_negative[0],
                    ct.where(
                        plane == 1,
                        derivatives_positive[0],
                        derivatives_negative[0],
                    ),
                )
                derivative_1 = ct.where(
                    plane == 0,
                    derivatives_positive[1] - derivatives_negative[1],
                    ct.where(
                        plane == 1,
                        derivatives_positive[1],
                        derivatives_negative[1],
                    ),
                )
                derivative_2 = ct.where(
                    plane == 0,
                    derivatives_positive[2] - derivatives_negative[2],
                    ct.where(
                        plane == 1,
                        derivatives_positive[2],
                        derivatives_negative[2],
                    ),
                )
                derivative_3 = ct.where(
                    plane == 0,
                    derivatives_positive[3] - derivatives_negative[3],
                    ct.where(
                        plane == 1,
                        derivatives_positive[3],
                        derivatives_negative[3],
                    ),
                )
                derivative_4 = ct.where(
                    plane < 2, derivatives_positive[4], zero
                )
                derivative_5 = ct.where(
                    plane < 2, derivatives_positive[5], zero
                )
                derivative_6 = ct.where(
                    plane == 0,
                    -derivatives_negative[4],
                    ct.where(plane == 2, derivatives_negative[4], zero),
                )
                derivative_7 = ct.where(
                    plane == 0,
                    -derivatives_negative[5],
                    ct.where(plane == 2, derivatives_negative[5], zero),
                )

            weight = ct.load(
                weights,
                index=(block, observation_tile),
                shape=(1, _OBSERVATIONS_PER_TILE),
                padding_mode=ct.PaddingMode.ZERO,
            )
            jacobian_01 = ct.cat(
                (
                    ct.reshape(derivative_0, (1, _OBSERVATIONS_PER_TILE)),
                    ct.reshape(derivative_1, (1, _OBSERVATIONS_PER_TILE)),
                ),
                axis=0,
            )
            jacobian_23 = ct.cat(
                (
                    ct.reshape(derivative_2, (1, _OBSERVATIONS_PER_TILE)),
                    ct.reshape(derivative_3, (1, _OBSERVATIONS_PER_TILE)),
                ),
                axis=0,
            )
            jacobian_45 = ct.cat(
                (
                    ct.reshape(derivative_4, (1, _OBSERVATIONS_PER_TILE)),
                    ct.reshape(derivative_5, (1, _OBSERVATIONS_PER_TILE)),
                ),
                axis=0,
            )
            jacobian_67 = ct.cat(
                (
                    ct.reshape(derivative_6, (1, _OBSERVATIONS_PER_TILE)),
                    ct.reshape(derivative_7, (1, _OBSERVATIONS_PER_TILE)),
                ),
                axis=0,
            )
            jacobian_03 = ct.cat((jacobian_01, jacobian_23), axis=0)
            jacobian_47 = ct.cat((jacobian_45, jacobian_67), axis=0)
            jacobian = ct.cat((jacobian_03, jacobian_47), axis=0)
            jacobian *= weight
            residual = ct.load(
                residuals,
                index=(block, observation_tile),
                shape=(1, _OBSERVATIONS_PER_TILE),
                padding_mode=ct.PaddingMode.ZERO,
            )
            zero_column = ct.zeros(
                (_OBSERVATIONS_PER_TILE, 1), dtype=parameters.dtype
            )
            residual_and_zero = ct.cat(
                (
                    ct.transpose(residual),
                    zero_column,
                ),
                axis=1,
            )
            zero_pair = ct.cat((zero_column, zero_column), axis=1)
            residual_quartet = ct.cat((residual_and_zero, zero_pair), axis=1)
            zero_quartet = ct.cat((zero_pair, zero_pair), axis=1)
            residual_octet = ct.cat((residual_quartet, zero_quartet), axis=1)
            augmented = ct.cat(
                (
                    ct.transpose(jacobian),
                    residual_octet,
                ),
                axis=1,
            )
            normal = ct.mma(ct.transpose(augmented), augmented, normal)

        gradient = ct.reshape(
            ct.extract(normal, (0, _PARAMETER_COUNT), (8, 1)), (1, 8)
        )
        hessian = ct.extract(normal, (0, 0), (8, 8))
        ct.store(gradients, (block, 0), gradient)
        ct.store(hessians, (block, 0), hessian)

    return kernel


def gaussian_normal_equations(
    parameters: BackendArray,
    residuals: BackendArray,
    weights: BackendArray,
    *,
    image_shape: tuple[int, int],
    mode: FitMode,
) -> tuple[BackendArray, BackendArray]:
    """Form Gaussian normal equations with one ``cuda.tile`` CTA per fit."""

    cp, ct = _load_cutile()
    batch, parameter_count = parameters.shape
    if parameter_count != _PARAMETER_COUNT:
        raise ValueError("Gaussian cuda.tile fits require eight parameters")
    if residuals.ndim != 2 or residuals.shape[0] != batch:
        raise ValueError("residuals must have shape (batch, observations)")
    if weights.shape != residuals.shape:
        raise ValueError("weights must match residuals")
    height, width = image_shape
    plane_count = 1 if mode == "difference" else 3
    observation_count = plane_count * height * width
    if residuals.shape[1] != observation_count:
        raise ValueError("residual width does not match the image shape")
    observation_tile_count = math.ceil(
        observation_count / _OBSERVATIONS_PER_TILE
    )
    parameters = cp.ascontiguousarray(parameters)
    residuals = cp.ascontiguousarray(residuals)
    weights = cp.ascontiguousarray(weights)
    gradients = cp.empty((batch, _PARAMETER_COUNT), dtype=parameters.dtype)
    hessian_storage = cp.empty(
        (batch * _PARAMETER_COUNT, _PARAMETER_COUNT),
        dtype=parameters.dtype,
    )
    try:
        ct.launch(
            cp.cuda.get_current_stream(),
            (batch, 1, 1),
            _gaussian_normal_equations_kernel(),
            (
                parameters,
                residuals,
                weights,
                gradients,
                hessian_storage,
                int(height),
                int(width),
                int(plane_count),
                int(observation_tile_count),
            ),
        )
    except Exception as exc:
        raise RuntimeError(
            "the xFit cuda.tile normal-equation kernel failed to compile or "
            "launch; check that cuda-tile matches the CUDA runtime"
        ) from exc
    return gradients, hessian_storage.reshape(
        batch, _PARAMETER_COUNT, _PARAMETER_COUNT
    )


__all__ = ["gaussian_normal_equations"]
