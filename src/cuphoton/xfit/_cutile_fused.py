# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Fused Gaussian evaluation using public ``cuda.tile`` operations."""

import math
from functools import cache
from typing import TYPE_CHECKING, Any

from ._cutile import _load_cutile
from ._types import BackendArray, FitMode

_PARAMETER_COUNT = 8
_EVALUATION_TILE = 128


@cache
def _evaluation_kernel() -> Any:
    if TYPE_CHECKING:
        import cuda.tile as ct
    else:
        _, ct = _load_cutile()

    @ct.kernel
    def kernel(
        parameters,
        width_cubes,
        images,
        weights,
        indices,
        output,
        height: ct.Constant[int],
        width: ct.Constant[int],
        plane_count: ct.Constant[int],
        compute_jacobian: ct.Constant[bool],
    ):
        block = ct.bid(0)
        observation_tile = ct.bid(1)
        weight_row = ct.reshape(ct.load(indices, (block,), (1,)), ())
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
        # Match invalid physical widths after log-sigma conversion.
        sigma_x = ct.where(
            (sigma_x > 0) & (sigma_x < math.inf), sigma_x, math.nan
        )
        sigma_y = ct.where(
            (sigma_y > 0) & (sigma_y < math.inf), sigma_y, math.nan
        )
        if compute_jacobian:
            cubes = ct.load(width_cubes, (block, 0), (1, 2))
            sigma_x_cube = ct.extract(cubes, (0, 0), (1, 1))
            sigma_y_cube = ct.extract(cubes, (0, 1), (1, 1))
        else:
            sigma_x_cube = 1.0
            sigma_y_cube = 1.0
        cosine = ct.cos(theta)
        sine = ct.sin(theta)
        inverse_x_squared = 1.0 / (sigma_x * sigma_x)
        inverse_y_squared = 1.0 / (sigma_y * sigma_y)
        pixel_count = height * width

        # Two-term reductions preserve the separately rounded products in
        # the CuPy model. Direct multiply/add expressions may contract into
        # an FMA and change the trajectory of poorly conditioned fits.
        observation = observation_tile * _EVALUATION_TILE + ct.arange(
            _EVALUATION_TILE, dtype=ct.int32
        )
        plane = observation // pixel_count
        pixel = observation - plane * pixel_count
        y = pixel // width
        x = pixel - y * width
        x_coordinate = x.astype(parameters.dtype) - (width - 1) / 2
        y_coordinate = y.astype(parameters.dtype) - (height - 1) / 2

        x_hat_positive = x_coordinate - x_positive
        y_hat_positive = y_coordinate - y_positive
        x_rotated_positive = ct.sum(
            ct.cat((cosine * x_hat_positive, sine * y_hat_positive), axis=0),
            axis=0,
            keepdims=True,
        )
        y_rotated_positive = ct.sum(
            ct.cat(
                (cosine * y_hat_positive, -(sine * x_hat_positive)), axis=0
            ),
            axis=0,
            keepdims=True,
        )
        unit_positive = ct.exp(
            -0.5
            * (
                ct.sum(
                    ct.cat(
                        (
                            x_rotated_positive
                            * x_rotated_positive
                            * inverse_x_squared,
                            y_rotated_positive
                            * y_rotated_positive
                            * inverse_y_squared,
                        ),
                        axis=0,
                    ),
                    axis=0,
                    keepdims=True,
                )
            )
        )
        star_positive = amplitude * unit_positive
        width_x_positive = (
            star_positive
            * x_rotated_positive
            * x_rotated_positive
            / sigma_x_cube
        )
        width_y_positive = (
            star_positive
            * y_rotated_positive
            * y_rotated_positive
            / sigma_y_cube
        )
        derivatives_positive = (
            unit_positive,
            width_x_positive,
            width_y_positive,
            -star_positive
            * x_rotated_positive
            * y_rotated_positive
            * (inverse_x_squared - inverse_y_squared),
            star_positive
            * (
                ct.sum(
                    ct.cat(
                        (
                            cosine * x_rotated_positive * inverse_x_squared,
                            -(sine * y_rotated_positive * inverse_y_squared),
                        ),
                        axis=0,
                    ),
                    axis=0,
                    keepdims=True,
                )
            ),
            star_positive
            * (
                ct.sum(
                    ct.cat(
                        (
                            sine * x_rotated_positive * inverse_x_squared,
                            cosine * y_rotated_positive * inverse_y_squared,
                        ),
                        axis=0,
                    ),
                    axis=0,
                    keepdims=True,
                )
            ),
        )

        x_hat_negative = x_coordinate - x_negative
        y_hat_negative = y_coordinate - y_negative
        x_rotated_negative = ct.sum(
            ct.cat((cosine * x_hat_negative, sine * y_hat_negative), axis=0),
            axis=0,
            keepdims=True,
        )
        y_rotated_negative = ct.sum(
            ct.cat(
                (cosine * y_hat_negative, -(sine * x_hat_negative)), axis=0
            ),
            axis=0,
            keepdims=True,
        )
        unit_negative = ct.exp(
            -0.5
            * (
                ct.sum(
                    ct.cat(
                        (
                            x_rotated_negative
                            * x_rotated_negative
                            * inverse_x_squared,
                            y_rotated_negative
                            * y_rotated_negative
                            * inverse_y_squared,
                        ),
                        axis=0,
                    ),
                    axis=0,
                    keepdims=True,
                )
            )
        )
        star_negative = amplitude * unit_negative
        width_x_negative = (
            star_negative
            * x_rotated_negative
            * x_rotated_negative
            / sigma_x_cube
        )
        width_y_negative = (
            star_negative
            * y_rotated_negative
            * y_rotated_negative
            / sigma_y_cube
        )
        derivatives_negative = (
            unit_negative,
            width_x_negative,
            width_y_negative,
            -star_negative
            * x_rotated_negative
            * y_rotated_negative
            * (inverse_x_squared - inverse_y_squared),
            star_negative
            * (
                ct.sum(
                    ct.cat(
                        (
                            cosine * x_rotated_negative * inverse_x_squared,
                            -(sine * y_rotated_negative * inverse_y_squared),
                        ),
                        axis=0,
                    ),
                    axis=0,
                    keepdims=True,
                )
            ),
            star_negative
            * (
                ct.sum(
                    ct.cat(
                        (
                            sine * x_rotated_negative * inverse_x_squared,
                            cosine * y_rotated_negative * inverse_y_squared,
                        ),
                        axis=0,
                    ),
                    axis=0,
                    keepdims=True,
                )
            ),
        )

        if plane_count == 1:
            derivative_0 = derivatives_positive[0] - derivatives_negative[0]
            derivative_1 = derivatives_positive[1] - derivatives_negative[1]
            derivative_2 = derivatives_positive[2] - derivatives_negative[2]
            derivative_3 = ct.sum(
                ct.cat(
                    (derivatives_positive[3], -(derivatives_negative[3])),
                    axis=0,
                ),
                axis=0,
                keepdims=True,
            )
            derivative_4 = derivatives_positive[4]
            derivative_5 = derivatives_positive[5]
            derivative_6 = -derivatives_negative[4]
            derivative_7 = -derivatives_negative[5]
        else:
            zero = ct.zeros((_EVALUATION_TILE,), dtype=parameters.dtype)
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
                ct.sum(
                    ct.cat(
                        (derivatives_positive[3], -(derivatives_negative[3])),
                        axis=0,
                    ),
                    axis=0,
                    keepdims=True,
                ),
                ct.where(
                    plane == 1,
                    derivatives_positive[3],
                    derivatives_negative[3],
                ),
            )
            derivative_4 = ct.where(plane < 2, derivatives_positive[4], zero)
            derivative_5 = ct.where(plane < 2, derivatives_positive[5], zero)
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

        # Preserve the model's physical-width derivative and chain
        # rule order, including overflow/underflow of sigma**3.
        derivative_1 *= sigma_x
        derivative_2 *= sigma_y

        weight = ct.load(
            weights,
            index=(weight_row, observation_tile),
            shape=(1, _EVALUATION_TILE),
            padding_mode=ct.PaddingMode.ZERO,
        )
        jacobian_01 = ct.cat(
            (
                ct.reshape(derivative_0, (1, _EVALUATION_TILE)),
                ct.reshape(derivative_1, (1, _EVALUATION_TILE)),
            ),
            axis=0,
        )
        jacobian_23 = ct.cat(
            (
                ct.reshape(derivative_2, (1, _EVALUATION_TILE)),
                ct.reshape(derivative_3, (1, _EVALUATION_TILE)),
            ),
            axis=0,
        )
        jacobian_45 = ct.cat(
            (
                ct.reshape(derivative_4, (1, _EVALUATION_TILE)),
                ct.reshape(derivative_5, (1, _EVALUATION_TILE)),
            ),
            axis=0,
        )
        jacobian_67 = ct.cat(
            (
                ct.reshape(derivative_6, (1, _EVALUATION_TILE)),
                ct.reshape(derivative_7, (1, _EVALUATION_TILE)),
            ),
            axis=0,
        )
        jacobian_03 = ct.cat((jacobian_01, jacobian_23), axis=0)
        jacobian_47 = ct.cat((jacobian_45, jacobian_67), axis=0)
        jacobian = ct.cat((jacobian_03, jacobian_47), axis=0)
        jacobian *= weight

        if compute_jacobian:
            ct.store(output, (block, observation_tile), jacobian)
        else:
            prediction = ct.sum(
                ct.cat((star_positive, -star_negative), axis=0),
                axis=0,
                keepdims=True,
            )
            if plane_count == 3:
                prediction = ct.where(
                    plane == 0,
                    prediction,
                    ct.where(plane == 1, star_positive, star_negative),
                )
            image = ct.load(
                images,
                (weight_row, observation_tile),
                (1, _EVALUATION_TILE),
                padding_mode=ct.PaddingMode.ZERO,
            )
            ct.store(
                output,
                (block, observation_tile),
                (prediction - image) * weight,
            )

    return kernel


class GaussianWorkspace:
    """Buffers owned by one fit invocation and its current CuPy stream.

    Jacobian outputs are borrowed until the next Jacobian call. The solver
    copies them into its own storage before reuse. Residuals are separately
    owned because accepted residuals must survive trial evaluations.
    """

    def __init__(
        self,
        images: BackendArray,
        weights: BackendArray,
        *,
        image_shape: tuple[int, int],
        mode: FitMode,
    ) -> None:
        self._cp, self._ct = _load_cutile()
        self._stream = self._cp.cuda.get_current_stream()
        self._height, self._width = image_shape
        self._planes = 1 if mode == "difference" else 3
        batch = int(images.shape[0])
        self._observations = self._planes * self._height * self._width
        self._images = self._cp.ascontiguousarray(images).reshape(
            batch, self._observations
        )
        self._weights = self._cp.ascontiguousarray(weights).reshape(
            batch, self._observations
        )
        self._jacobians = self._cp.empty(
            (batch * 8, self._observations), dtype=images.dtype
        )
        self._width_cubes = self._cp.empty((batch, 2), dtype=images.dtype)
        self._kernel = _evaluation_kernel()

    def _evaluate(
        self,
        parameters: BackendArray,
        indices: BackendArray,
        output: BackendArray,
        *,
        compute_jacobian: bool,
    ) -> None:
        width_cubes = self._width_cubes[: parameters.shape[0]]
        if compute_jacobian:
            # Match the model's power rounding. A one-ULP difference here
            # can alter the trajectory of an ill-conditioned fit.
            self._cp.exp(parameters[:, 1:3], out=width_cubes)
            self._cp.power(width_cubes, 3, out=width_cubes)
        try:
            self._ct.launch(
                self._stream,
                (
                    parameters.shape[0],
                    math.ceil(self._observations / _EVALUATION_TILE),
                    1,
                ),
                self._kernel,
                (
                    parameters,
                    width_cubes,
                    self._images,
                    self._weights,
                    indices,
                    output,
                    self._height,
                    self._width,
                    self._planes,
                    compute_jacobian,
                ),
            )
        except Exception as exc:
            raise RuntimeError(
                "the xFit fused cuda.tile evaluation kernel failed to "
                "compile or launch; check that cuda-tile matches the CUDA "
                "runtime"
            ) from exc

    def residual(
        self, parameters: BackendArray, *, indices: BackendArray
    ) -> BackendArray:
        output = self._cp.empty(
            (parameters.shape[0], self._observations), dtype=parameters.dtype
        )
        self._evaluate(parameters, indices, output, compute_jacobian=False)
        return output

    def jacobian(
        self, parameters: BackendArray, *, indices: BackendArray
    ) -> BackendArray:
        output = self._jacobians[: parameters.shape[0] * 8]
        self._evaluate(parameters, indices, output, compute_jacobian=True)
        return output.reshape(parameters.shape[0], 8, self._observations)
