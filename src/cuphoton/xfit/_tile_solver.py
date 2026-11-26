# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Fused Gaussian fitting with cuTile and the public nvmath device solver."""

import math
from functools import cache
from typing import TYPE_CHECKING

import numpy as np

from .solver import _MAX_REJECTED_STEPS, LMStatus

# Eight Gaussian parameters, followed by the residual and seven zero columns
# in the augmented normal-equation tile. Each tile covers 32 observations.
_PARAMETER_COUNT = 8
_NORMAL_WIDTH = 16
_OBSERVATIONS_PER_TILE = 32

# The reference-specific reduction schedules below apply to one 51x51 plane.
# Their partition widths preserve FP64 association; they are not tuning knobs.
_SPECIALIZED_IMAGE_SHAPE = (51, 51)
_SPECIALIZED_OBSERVATIONS = 2601  # 51 * 51
_SM100_MMA_PARTITION_LENGTH = 218
_SM100_MMA_LAST_PARTITION_LENGTH = 203  # 2601 - 11 * 218
_SM100_GEMMSN_PARTITION_LENGTH = 1301  # ceil(2601 / 2)
_SINGLETON_HESSIAN_PARTITION_LENGTH = 522

# Qualified reference arithmetic-family boundaries, not hardware limits.
_SM100_MMA_MAX_ELIGIBLE = 12
_GROUP4_GRADIENT_MIN_ELIGIBLE = 39

# Avoid capture for short fits; bound replay work between host status checks.
# The batch cap records the qualified graph scope rather than a device limit.
_GRAPH_MIN_HOST_TRANSITIONS = 16
_GRAPH_TRANSITIONS_PER_REPLAY = 16
_GRAPH_MAX_BATCH = 256

# The kernel's extent argument encodes execution modes, not a tile dimension.
# Zero refreshes normals only; current host callers use the two modes below.
_ONE_TRANSITION = 1
_PERSISTENT_TRANSITIONS = 2

# Tile needs plain integer constants; use the shared solver's status codes.
_ACTIVE = int(LMStatus.ACTIVE)
_CONVERGED_F_TOL = int(LMStatus.CONVERGED_F_TOL)
_CONVERGED_X_TOL = int(LMStatus.CONVERGED_X_TOL)
_CONVERGED_G_TOL = int(LMStatus.CONVERGED_G_TOL)
_MAX_EVALUATIONS = int(LMStatus.MAX_EVALUATIONS)
_INVALID_RESIDUAL = int(LMStatus.INVALID_RESIDUAL)
_SINGULAR = int(LMStatus.SINGULAR)
_NO_PROGRESS = int(LMStatus.NO_PROGRESS)


def _lm_control_bits(settings, dtype):
    # Python float constants enter the Tile frontend as FP32. Transport the
    # selected compute dtype's IEEE bits through exact integer constants.
    values = np.asarray(
        (
            settings.initial_damping,
            settings.damping_increase,
            settings.damping_decrease,
            np.finfo(dtype).eps,
            settings.f_tol,
            settings.x_tol,
            settings.g_tol,
        ),
        dtype=dtype,
    )
    unsigned = np.uint64 if values.dtype == np.dtype("float64") else np.uint32
    return tuple(values.view(unsigned).tolist())


def _runtime():
    try:
        import cuda.tile as ct
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError(
            "Tile/SIMT fitting requires the tile-simt extra and CUDA 13.5"
        ) from exc
    return cp, ct


@cache
def _kernels(
    singleton_normals: bool = False,
    group4_gradient: bool = False,
    normal_recipe: str = "tile",
    known_non_singleton_cost: bool = False,
    cache_normals: bool = False,
):
    if normal_recipe not in {"tile", "sm100-gemmsn", "sm100-mma"}:
        raise ValueError("unsupported normal accumulation recipe")
    sm100_normals = normal_recipe != "tile"
    if cache_normals and not (singleton_normals and sm100_normals):
        raise ValueError("normal caching requires the SM100 graph schedule")
    cp, runtime_ct = _runtime()
    try:
        import nvmath.tile.linalg as mlinalg
    except ImportError as exc:
        raise RuntimeError(
            "Tile/SIMT fitting requires the public nvmath.tile runtime"
        ) from exc
    if TYPE_CHECKING:
        import cuda.tile as ct
    else:
        ct = runtime_ct

    def all_finite(value):
        flat = ct.reshape(value, (-1,))
        return ct.min((ct.abs(flat) < float("inf")).astype(ct.int32)) != 0

    def scaled_norm(value):
        flat = ct.reshape(value, (-1,))
        magnitude = ct.max(ct.abs(flat), propagate_nan=True)
        divisor = ct.where(magnitude == 0, 1.0, magnitude)
        normalized = flat / divisor
        return magnitude * ct.sqrt(ct.sum(normalized * normalized))

    def sm100_mma_hessian(jacobian, row, partitioned: ct.Constant[bool]):
        """Accumulate the reference's residue-first FP64 K4 products."""
        # Four worker warps need a full 8x8 FP64 atom per warp. Padding
        # prevents a scalar-FMA fallback that changes the K4 association.
        parameters = ct.reshape(ct.arange(16, dtype=ct.int32), (16, 1))
        lanes = ct.reshape(ct.arange(4, dtype=ct.int32), (1, 4))
        hessian = ct.zeros((16, 16), ct.float64)
        # Singleton: eleven 218-observation partitions and a 203-observation
        # tail (11 * 218 + 203 = 2601). Batched: one full-image partition.
        # Each partition starts with its short residue, then uses tiles of
        # 16 observations, each accumulated as four ordered K=4 MMA products.
        partitions = 12 if partitioned else 1
        for partition in range(partitions):
            begin = (
                _SM100_MMA_PARTITION_LENGTH * partition if partitioned else 0
            )
            extent = (
                ct.where(
                    partition == 11,
                    _SM100_MMA_LAST_PARTITION_LENGTH,
                    _SM100_MMA_PARTITION_LENGTH,
                )
                if partitioned
                else _SPECIALIZED_OBSERVATIONS
            )
            residue = extent % 16
            partial = ct.zeros((16, 16), ct.float64)
            for tile in range((extent + 15) // 16):
                base = begin + ct.where(
                    tile == 0, 0, residue + 16 * (tile - 1)
                )
                for group in range(4):
                    slot = 4 * group + lanes
                    jac = ct.gather(
                        jacobian,
                        (row, parameters, base + slot),
                        mask=(parameters < 8)
                        & ((tile != 0) | (slot < residue)),
                    )
                    partial = ct.mma(jac, ct.transpose(jac), partial)
            if partitioned:
                hessian = ct.add(
                    hessian, partial, rounding_mode=ct.RoundingMode.RN
                )
            else:
                hessian = partial
        return ct.extract(hessian, (0, 0), (8, 8))

    def sm100_gemmsn_hessian(jacobian, row):
        """Preserve the two K ranges and zero-seeded 256-term FMA tiles."""
        # The two ranges contain 1301 and 1300 observations. Each 256-product
        # chunk starts from zero; chunk totals and then range totals are added
        # with explicit round-to-nearest (RN), retaining the reference order.
        parameters = ct.reshape(ct.arange(8, dtype=ct.int32), (8, 1))
        halves = ct.zeros((2, 8, 8), ct.float64)
        for half in range(2):
            begin = half * _SM100_GEMMSN_PARTITION_LENGTH
            end = ct.minimum(
                begin + _SM100_GEMMSN_PARTITION_LENGTH,
                _SPECIALIZED_OBSERVATIONS,
            )
            partial = ct.zeros((8, 8), ct.float64)
            for tile_begin in range(begin, end, 256):
                tile = ct.zeros((8, 8), ct.float64)
                for observation in range(
                    tile_begin, ct.minimum(tile_begin + 256, end)
                ):
                    jac = ct.gather(jacobian, (row, parameters, observation))
                    tile = jac * ct.transpose(jac) + tile
                partial = ct.add(
                    partial, tile, rounding_mode=ct.RoundingMode.RN
                )
            halves = ct.insert(
                halves, (half, 0, 0), ct.reshape(partial, (1, 8, 8))
            )
        return ct.add(
            ct.reshape(ct.extract(halves, (0, 0, 0), (1, 8, 8)), (8, 8)),
            ct.reshape(ct.extract(halves, (1, 0, 0), (1, 8, 8)), (8, 8)),
            rounding_mode=ct.RoundingMode.RN,
        )

    def sm100_singleton_cost(residuals, row):
        """Match the singleton dot's chunks and serial partition fold."""
        # 32 contiguous ranges of ceil(2601 / 32) = 82 observations. Each
        # range uses ceil(82 / 16) = 6 zero-seeded chunks of 16 products;
        # masks exclude both range padding and observations past the image.
        partitions = ct.arange(32, dtype=ct.int32)
        partials = ct.zeros((32,), ct.float64)
        for chunk in range(6):
            values = ct.zeros((32,), ct.float64)
            for offset in range(16):
                position = 16 * chunk + offset
                observation = 82 * partitions + position
                valid = (position < 82) & (
                    observation < _SPECIALIZED_OBSERVATIONS
                )
                residual = ct.gather(
                    residuals, (row, observation), mask=valid
                )
                values = ct.where(valid, residual * residual + values, values)
            partials = ct.add(
                partials, values, rounding_mode=ct.RoundingMode.RN
            )
        # Fold range totals serially; LM uses half the squared residual norm.
        cost = ct.extract(partials, (0,), (1,))
        for partition in range(1, 32):
            cost = ct.add(
                cost,
                ct.extract(partials, (partition,), (1,)),
                rounding_mode=ct.RoundingMode.RN,
            )
        return 0.5 * ct.reshape(cost, ())

    def singleton_normal(jacobian, residuals, row):
        """Preserve the FP64 singleton accumulation order."""
        # Each partition has 128 lanes. SM100 uses 10 partitions (stride
        # 1280, up to three visits per lane); the other schedule uses 16
        # (stride 2048, up to two visits). Storage pads both to 16 partitions.
        parameters = ct.reshape(ct.arange(8, dtype=ct.int32), (8, 1))
        lanes = ct.reshape(ct.arange(128, dtype=ct.int32), (1, 128))
        partials = ct.zeros((8, 16), ct.float64)
        for partition in range(10 if sm100_normals else 16):
            observations = 128 * partition + lanes
            jac = ct.gather(jacobian, (row, parameters, observations))
            res = ct.gather(residuals, (row, observations))
            # The frontend contracts each multiply/add to a single FP64 FMA.
            values = jac * res + ct.zeros((8, 128), ct.float64)
            for offset in range(1, 3 if sm100_normals else 2):
                observations = (
                    128 * partition
                    + lanes
                    + offset * (1280 if sm100_normals else 2048)
                )
                valid = observations < _SPECIALIZED_OBSERVATIONS
                jac = ct.gather(
                    jacobian, (row, parameters, observations), mask=valid
                )
                res = ct.gather(residuals, (row, observations), mask=valid)
                values = ct.where(valid, jac * res + values, values)
            values = ct.add(values, 0.0, rounding_mode=ct.RoundingMode.RN)
            # Reduce each parameter's 128 lane partials through a fixed tree:
            # 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1. extract's (0, 1)
            # selects the upper half-subtile at each stage, not element one.
            # RN additions of lower and upper halves retain FP64 association.
            half = ct.add(
                ct.extract(values, (0, 0), (8, 64)),
                ct.extract(values, (0, 1), (8, 64)),
                rounding_mode=ct.RoundingMode.RN,
            )
            half = ct.add(
                ct.extract(half, (0, 0), (8, 32)),
                ct.extract(half, (0, 1), (8, 32)),
                rounding_mode=ct.RoundingMode.RN,
            )
            half = ct.add(
                ct.extract(half, (0, 0), (8, 16)),
                ct.extract(half, (0, 1), (8, 16)),
                rounding_mode=ct.RoundingMode.RN,
            )
            half = ct.add(
                ct.extract(half, (0, 0), (8, 8)),
                ct.extract(half, (0, 1), (8, 8)),
                rounding_mode=ct.RoundingMode.RN,
            )
            half = ct.add(
                ct.extract(half, (0, 0), (8, 4)),
                ct.extract(half, (0, 1), (8, 4)),
                rounding_mode=ct.RoundingMode.RN,
            )
            half = ct.add(
                ct.extract(half, (0, 0), (8, 2)),
                ct.extract(half, (0, 1), (8, 2)),
                rounding_mode=ct.RoundingMode.RN,
            )
            half = ct.add(
                ct.extract(half, (0, 0), (8, 1)),
                ct.extract(half, (0, 1), (8, 1)),
                rounding_mode=ct.RoundingMode.RN,
            )
            partials = ct.insert(partials, (0, partition), half)
        gradient = ct.zeros((8, 1), ct.float64)
        # Fold partition columns g, g+4, g+8, g+12 for each of four groups.
        # The subsequent zero additions and identity multiply/add operations
        # retain the reference's padded reduction and signed-zero behavior.
        for group_index in range(4):
            group = ct.zeros((8, 1), ct.float64)
            for offset in range(4):
                value = ct.extract(
                    partials, (0, group_index + 4 * offset), (8, 1)
                )
                group = ct.add(value, group, rounding_mode=ct.RoundingMode.RN)
            for _offset in range(4):
                group = ct.add(0.0, group, rounding_mode=ct.RoundingMode.RN)
            gradient = ct.add(
                gradient, group, rounding_mode=ct.RoundingMode.RN
            )
        gradient = gradient * 1.0 + ct.zeros((8, 1), ct.float64)
        gradient = ct.add(gradient, 0.0, rounding_mode=ct.RoundingMode.RN)

        if sm100_normals:
            hessian = sm100_mma_hessian(jacobian, row, True)
        else:
            # Parallelize independent partitions, retaining each partition's
            # sequential FP64 FMA recurrence and the final serial RN fold.
            # Five partitions cover four * 522 + 513 = 2601 observations;
            # the vector layout pads five partitions to eight slots.
            partitions = ct.reshape(ct.arange(8, dtype=ct.int32), (8, 1, 1))
            matrix_parameters = ct.reshape(
                ct.arange(8, dtype=ct.int32), (1, 8, 1)
            )
            hessian_partials = ct.zeros((8, 8, 8), ct.float64)
            for offset in range(_SINGLETON_HESSIAN_PARTITION_LENGTH):
                observations = (
                    _SINGLETON_HESSIAN_PARTITION_LENGTH * partitions + offset
                )
                valid = (partitions < 5) & (
                    observations < _SPECIALIZED_OBSERVATIONS
                )
                jac = ct.gather(
                    jacobian,
                    (row, matrix_parameters, observations),
                    mask=valid,
                )
                hessian_partials = ct.where(
                    valid,
                    jac * ct.transpose(jac, 1, 2) + hessian_partials,
                    hessian_partials,
                )
            hessian = ct.zeros((8, 8), ct.float64)
            for partition in range(5):
                partial = ct.reshape(
                    ct.extract(
                        hessian_partials, (partition, 0, 0), (1, 8, 8)
                    ),
                    (8, 8),
                )
                hessian = ct.add(
                    partial, hessian, rounding_mode=ct.RoundingMode.RN
                )
        return ct.reshape(gradient, (1, 8)), hessian

    def solve(a, b):
        rhs = ct.reshape(b, (8, 1))
        lu, pivots, info = mlinalg.lu_factor(a)
        if info == 0:
            solution = mlinalg.lu_solve_using_factor(lu, rhs, pivots)
        else:
            solution = ct.full((8, 1), float("nan"), a.dtype)
        return ct.reshape(solution, (1, 8)), info

    def normal(
        parameter_tile,
        images,
        weights,
        residuals,
        jacobian_scratch,
        singleton_flags,
        block,
        height: ct.Constant[int],
        width: ct.Constant[int],
        plane_count: ct.Constant[int],
        observation_tile_count: ct.Constant[int],
        need_normal: ct.Constant[bool] = True,
    ):
        use_singleton = False
        if singleton_normals:
            use_singleton = ct.reshape(
                ct.load(singleton_flags, (block,), (1,)), ()
            )
        # Parameter columns: amplitude, log(sigma_x), log(sigma_y), angle,
        # positive-lobe x/y, then negative-lobe x/y.
        amplitude = ct.extract(parameter_tile, (0, 0), (1, 1))
        sigma_x = ct.exp(ct.extract(parameter_tile, (0, 1), (1, 1)))
        sigma_y = ct.exp(ct.extract(parameter_tile, (0, 2), (1, 1)))
        sigma_x = ct.where(
            (sigma_x > 0) & (ct.abs(sigma_x) < float("inf")),
            sigma_x,
            float("nan"),
        )
        sigma_y = ct.where(
            (sigma_y > 0) & (ct.abs(sigma_y) < float("inf")),
            sigma_y,
            float("nan"),
        )
        theta = ct.extract(parameter_tile, (0, 3), (1, 1))
        x_positive = ct.extract(parameter_tile, (0, 4), (1, 1))
        y_positive = ct.extract(parameter_tile, (0, 5), (1, 1))
        x_negative = ct.extract(parameter_tile, (0, 6), (1, 1))
        y_negative = ct.extract(parameter_tile, (0, 7), (1, 1))
        cosine = ct.cos(theta)
        sine = ct.sin(theta)
        inverse_x_squared = 1.0 / (sigma_x * sigma_x)
        inverse_y_squared = 1.0 / (sigma_y * sigma_y)
        sigma_x_cube = sigma_x**3.0
        sigma_y_cube = sigma_y**3.0
        # Split-plane indices are difference, positive lobe, negative lobe.
        pixel_count = height * width
        finite_residual = ct.full((), True, ct.bool_)
        finite_jacobian = ct.full((), True, ct.bool_)
        cost = ct.full((), 0.0, parameter_tile.dtype)
        if singleton_normals and not sm100_normals:
            cost_lanes = ct.zeros((32,), dtype=parameter_tile.dtype)
        normal = ct.zeros(
            (_NORMAL_WIDTH, _NORMAL_WIDTH), dtype=parameter_tile.dtype
        )

        gradient_lanes = ct.zeros((8, 32), dtype=parameter_tile.dtype)

        for observation_tile in range(observation_tile_count):
            observation = (
                observation_tile * _OBSERVATIONS_PER_TILE
                + ct.arange(_OBSERVATIONS_PER_TILE, dtype=ct.int32)
            )
            plane = observation // pixel_count
            pixel = observation - plane * pixel_count
            y = pixel // width
            x = pixel - y * width
            x_coordinate = x.astype(parameter_tile.dtype) - (width - 1) / 2
            y_coordinate = y.astype(parameter_tile.dtype) - (height - 1) / 2

            x_hat_positive = x_coordinate - x_positive
            y_hat_positive = y_coordinate - y_positive
            x_rotated_positive = ct.sum(
                ct.cat(
                    (
                        cosine * x_hat_positive + 0.0,
                        sine * y_hat_positive + 0.0,
                    ),
                    axis=0,
                ),
                axis=0,
                keepdims=True,
                rounding_mode=ct.RoundingMode.RN,
            )
            y_rotated_positive = ct.sum(
                ct.cat(
                    (
                        cosine * y_hat_positive + 0.0,
                        -(sine * x_hat_positive + 0.0),
                    ),
                    axis=0,
                ),
                axis=0,
                keepdims=True,
                rounding_mode=ct.RoundingMode.RN,
            )
            # Adding +0.0 to each product makes the frontend emit
            # fma(a, b, +0.0), which rounds exactly like the separate
            # product and leaves the two-term sum without a multiply
            # operand that native code generation could contract.
            unit_positive = ct.exp(
                -0.5
                * ct.sum(
                    ct.cat(
                        (
                            x_rotated_positive
                            * x_rotated_positive
                            * inverse_x_squared
                            + 0.0,
                            y_rotated_positive
                            * y_rotated_positive
                            * inverse_y_squared
                            + 0.0,
                        ),
                        axis=0,
                    ),
                    axis=0,
                    keepdims=True,
                    rounding_mode=ct.RoundingMode.RN,
                )
            )
            star_positive = amplitude * unit_positive
            # Match CuPy's floating power for physical sigma derivatives.
            # An integer exponent lowers to FPowI with different rounding.
            derivatives_positive = (
                unit_positive,
                star_positive
                * x_rotated_positive
                * x_rotated_positive
                / sigma_x_cube,
                star_positive
                * y_rotated_positive
                * y_rotated_positive
                / sigma_y_cube,
                -star_positive
                * x_rotated_positive
                * y_rotated_positive
                * (inverse_x_squared - inverse_y_squared),
                star_positive
                * ct.sum(
                    ct.cat(
                        (
                            cosine * x_rotated_positive * inverse_x_squared
                            + 0.0,
                            -(
                                sine * y_rotated_positive * inverse_y_squared
                                + 0.0
                            ),
                        ),
                        axis=0,
                    ),
                    axis=0,
                    keepdims=True,
                    rounding_mode=ct.RoundingMode.RN,
                ),
                star_positive
                * ct.sum(
                    ct.cat(
                        (
                            sine * x_rotated_positive * inverse_x_squared
                            + 0.0,
                            cosine * y_rotated_positive * inverse_y_squared
                            + 0.0,
                        ),
                        axis=0,
                    ),
                    axis=0,
                    keepdims=True,
                    rounding_mode=ct.RoundingMode.RN,
                ),
            )

            x_hat_negative = x_coordinate - x_negative
            y_hat_negative = y_coordinate - y_negative
            x_rotated_negative = ct.sum(
                ct.cat(
                    (
                        cosine * x_hat_negative + 0.0,
                        sine * y_hat_negative + 0.0,
                    ),
                    axis=0,
                ),
                axis=0,
                keepdims=True,
                rounding_mode=ct.RoundingMode.RN,
            )
            y_rotated_negative = ct.sum(
                ct.cat(
                    (
                        cosine * y_hat_negative + 0.0,
                        -(sine * x_hat_negative + 0.0),
                    ),
                    axis=0,
                ),
                axis=0,
                keepdims=True,
                rounding_mode=ct.RoundingMode.RN,
            )
            unit_negative = ct.exp(
                -0.5
                * ct.sum(
                    ct.cat(
                        (
                            x_rotated_negative
                            * x_rotated_negative
                            * inverse_x_squared
                            + 0.0,
                            y_rotated_negative
                            * y_rotated_negative
                            * inverse_y_squared
                            + 0.0,
                        ),
                        axis=0,
                    ),
                    axis=0,
                    keepdims=True,
                    rounding_mode=ct.RoundingMode.RN,
                )
            )
            star_negative = amplitude * unit_negative
            derivatives_negative = (
                unit_negative,
                star_negative
                * x_rotated_negative
                * x_rotated_negative
                / sigma_x_cube,
                star_negative
                * y_rotated_negative
                * y_rotated_negative
                / sigma_y_cube,
                -star_negative
                * x_rotated_negative
                * y_rotated_negative
                * (inverse_x_squared - inverse_y_squared),
                star_negative
                * ct.sum(
                    ct.cat(
                        (
                            cosine * x_rotated_negative * inverse_x_squared
                            + 0.0,
                            -(
                                sine * y_rotated_negative * inverse_y_squared
                                + 0.0
                            ),
                        ),
                        axis=0,
                    ),
                    axis=0,
                    keepdims=True,
                    rounding_mode=ct.RoundingMode.RN,
                ),
                star_negative
                * ct.sum(
                    ct.cat(
                        (
                            sine * x_rotated_negative * inverse_x_squared
                            + 0.0,
                            cosine * y_rotated_negative * inverse_y_squared
                            + 0.0,
                        ),
                        axis=0,
                    ),
                    axis=0,
                    keepdims=True,
                    rounding_mode=ct.RoundingMode.RN,
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
                derivative_3 = ct.sum(
                    ct.cat(
                        (
                            derivatives_positive[3] + 0.0,
                            -(derivatives_negative[3] + 0.0),
                        ),
                        axis=0,
                    ),
                    axis=0,
                    keepdims=True,
                    rounding_mode=ct.RoundingMode.RN,
                )
                derivative_4 = derivatives_positive[4]
                derivative_5 = derivatives_positive[5]
                derivative_6 = -derivatives_negative[4]
                derivative_7 = -derivatives_negative[5]
            else:
                zero = ct.zeros(
                    (_OBSERVATIONS_PER_TILE,), dtype=parameter_tile.dtype
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
                    ct.sum(
                        ct.cat(
                            (
                                derivatives_positive[3] + 0.0,
                                -(derivatives_negative[3] + 0.0),
                            ),
                            axis=0,
                        ),
                        axis=0,
                        keepdims=True,
                        rounding_mode=ct.RoundingMode.RN,
                    ),
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

            # Match the physical derivative, plane combination, then log-width
            # chain order of GaussianDipoleModel and the public fitting API.
            derivative_1 *= sigma_x
            derivative_2 *= sigma_y
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
            jacobian = ct.where(
                ct.reshape(observation < plane_count * pixel_count, (1, 32)),
                jacobian,
                0,
            )
            # Keep the separately rounded star products before subtraction.
            # A multiply/subtract rewrite can otherwise leave a residual for
            # coincident stars at large finite amplitude.
            difference = ct.sum(
                ct.cat((star_positive + 0.0, -(star_negative + 0.0)), axis=0),
                axis=0,
            )
            prediction = ct.where(
                plane == 0,
                difference,
                ct.where(plane == 1, star_positive, star_negative),
            )
            observed = ct.load(
                images,
                index=(block, observation_tile),
                shape=(1, _OBSERVATIONS_PER_TILE),
                padding_mode=ct.PaddingMode.ZERO,
            )
            residual = (ct.reshape(prediction, (1, 32)) - observed) * weight
            residual = ct.where(
                ct.reshape(observation < plane_count * pixel_count, (1, 32)),
                residual,
                0,
            )
            ct.store(residuals, (block, observation_tile), residual)
            finite_residual = finite_residual & all_finite(residual)
            if not (singleton_normals and sm100_normals):
                if singleton_normals:
                    # Accumulate four adjacent observations per lane.
                    quarter = observation_tile % 4
                    cost_group = ct.extract(cost_lanes, (quarter,), (8,))
                    residual_groups = ct.reshape(residual, (8, 4))
                    residual_value = ct.reshape(
                        ct.extract(residual_groups, (0, 0), (8, 1)), (8,)
                    )
                    cost_group = cost_group + residual_value * residual_value
                    residual_value = ct.reshape(
                        ct.extract(residual_groups, (0, 1), (8, 1)), (8,)
                    )
                    cost_group = cost_group + residual_value * residual_value
                    residual_value = ct.reshape(
                        ct.extract(residual_groups, (0, 2), (8, 1)), (8,)
                    )
                    cost_group = cost_group + residual_value * residual_value
                    residual_value = ct.reshape(
                        ct.extract(residual_groups, (0, 3), (8, 1)), (8,)
                    )
                    cost_group = cost_group + residual_value * residual_value
                    cost_lanes = ct.insert(cost_lanes, (quarter,), cost_group)
                else:
                    cost += ct.sum(ct.reshape(residual * residual, (-1,)))
            zero_column = ct.zeros(
                (_OBSERVATIONS_PER_TILE, 1), dtype=parameter_tile.dtype
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
            if need_normal or sm100_normals:
                finite_jacobian = finite_jacobian & all_finite(jacobian)
            # Graph trials already form derivatives for validity checks.
            # Reuse this allocation only while its separate flag is true.
            if (
                need_normal and (sm100_normals or use_singleton)
            ) or cache_normals:
                ct.store(
                    jacobian_scratch,
                    (block, 0, observation_tile),
                    ct.reshape(jacobian, (1, 8, _OBSERVATIONS_PER_TILE)),
                )
            if need_normal:
                if not sm100_normals:
                    normal = ct.mma(
                        ct.transpose(augmented), augmented, normal
                    )
                if group4_gradient:
                    # Match the large-batch reduction: four observations,
                    # 32 lanes and dependent FP64 product accumulation.
                    # Four groups of eight lanes receive successive tiles;
                    # each lane adds its four adjacent observations in order.
                    quarter = observation_tile % 4
                    gradient_group = ct.extract(
                        gradient_lanes, (0, quarter), (8, 8)
                    )
                    jacobian_groups = ct.reshape(jacobian, (8, 8, 4))
                    residual_groups = ct.reshape(residual, (1, 8, 4))
                    gradient_group = gradient_group + ct.reshape(
                        ct.extract(jacobian_groups, (0, 0, 0), (8, 8, 1)),
                        (8, 8),
                    ) * ct.reshape(
                        ct.extract(residual_groups, (0, 0, 0), (1, 8, 1)),
                        (1, 8),
                    )
                    gradient_group = gradient_group + ct.reshape(
                        ct.extract(jacobian_groups, (0, 0, 1), (8, 8, 1)),
                        (8, 8),
                    ) * ct.reshape(
                        ct.extract(residual_groups, (0, 0, 1), (1, 8, 1)),
                        (1, 8),
                    )
                    gradient_group = gradient_group + ct.reshape(
                        ct.extract(jacobian_groups, (0, 0, 2), (8, 8, 1)),
                        (8, 8),
                    ) * ct.reshape(
                        ct.extract(residual_groups, (0, 0, 2), (1, 8, 1)),
                        (1, 8),
                    )
                    gradient_group = gradient_group + ct.reshape(
                        ct.extract(jacobian_groups, (0, 0, 3), (8, 8, 1)),
                        (8, 8),
                    ) * ct.reshape(
                        ct.extract(residual_groups, (0, 0, 3), (1, 8, 1)),
                        (1, 8),
                    )
                    gradient_lanes = ct.insert(
                        gradient_lanes, (0, quarter), gradient_group
                    )
                else:
                    gradient_lanes = gradient_lanes + jacobian * residual

        if singleton_normals and sm100_normals:
            # Keep the cost recurrence out of the model/Jacobian loop.
            # Read the stored, masked residuals in the same observation order.
            cost_lanes = ct.zeros((32,), dtype=parameter_tile.dtype)
            # Singleton cost uses four groups of eight lanes, adding four
            # adjacent residual squares per lane; other rows use 32 lanes.
            if known_non_singleton_cost or not use_singleton:
                for cost_tile in range(observation_tile_count):
                    cost_values = ct.reshape(
                        ct.load(
                            residuals,
                            (block, cost_tile),
                            (1, _OBSERVATIONS_PER_TILE),
                            padding_mode=ct.PaddingMode.ZERO,
                        ),
                        (32,),
                    )
                    cost_lanes = cost_lanes + cost_values * cost_values
            else:
                for cost_tile in range(observation_tile_count):
                    cost_residual = ct.load(
                        residuals,
                        (block, cost_tile),
                        (1, _OBSERVATIONS_PER_TILE),
                        padding_mode=ct.PaddingMode.ZERO,
                    )
                    quarter = cost_tile % 4
                    cost_group = ct.extract(cost_lanes, (quarter,), (8,))
                    residual_groups = ct.reshape(cost_residual, (8, 4))
                    residual_value = ct.reshape(
                        ct.extract(residual_groups, (0, 0), (8, 1)), (8,)
                    )
                    cost_group = cost_group + residual_value * residual_value
                    residual_value = ct.reshape(
                        ct.extract(residual_groups, (0, 1), (8, 1)), (8,)
                    )
                    cost_group = cost_group + residual_value * residual_value
                    residual_value = ct.reshape(
                        ct.extract(residual_groups, (0, 2), (8, 1)), (8,)
                    )
                    cost_group = cost_group + residual_value * residual_value
                    residual_value = ct.reshape(
                        ct.extract(residual_groups, (0, 3), (8, 1)), (8,)
                    )
                    cost_group = cost_group + residual_value * residual_value
                    cost_lanes = ct.insert(cost_lanes, (quarter,), cost_group)

        if singleton_normals:
            # The same lower-half + upper-half tree used above, now reducing
            # 32 cost lanes to one: 32 -> 16 -> 8 -> 4 -> 2 -> 1.
            cost_half = ct.extract(cost_lanes, (0,), (16,)) + ct.extract(
                cost_lanes, (1,), (16,)
            )
            cost_half = ct.extract(cost_half, (0,), (8,)) + ct.extract(
                cost_half, (1,), (8,)
            )
            cost_half = ct.extract(cost_half, (0,), (4,)) + ct.extract(
                cost_half, (1,), (4,)
            )
            cost_half = ct.extract(cost_half, (0,), (2,)) + ct.extract(
                cost_half, (1,), (2,)
            )
            cost_half = ct.extract(cost_half, (0,), (1,)) + ct.extract(
                cost_half, (1,), (1,)
            )
            cost = ct.reshape(cost_half, ())
        if sm100_normals or group4_gradient:
            # Reduce the 32 lanes of each of eight parameter gradients in the
            # same fixed tree. An unconstrained sum can change association.
            gradient_half = ct.extract(
                gradient_lanes, (0, 0), (8, 16)
            ) + ct.extract(gradient_lanes, (0, 1), (8, 16))
            gradient_half = ct.extract(
                gradient_half, (0, 0), (8, 8)
            ) + ct.extract(gradient_half, (0, 1), (8, 8))
            gradient_half = ct.extract(
                gradient_half, (0, 0), (8, 4)
            ) + ct.extract(gradient_half, (0, 1), (8, 4))
            gradient_half = ct.extract(
                gradient_half, (0, 0), (8, 2)
            ) + ct.extract(gradient_half, (0, 1), (8, 2))
            gradient_half = ct.extract(
                gradient_half, (0, 0), (8, 1)
            ) + ct.extract(gradient_half, (0, 1), (8, 1))
            gradient = ct.reshape(gradient_half, (1, 8))
        else:
            gradient = ct.reshape(ct.sum(gradient_lanes, axis=1), (1, 8))
        hessian = ct.extract(normal, (0, 0), (8, 8))
        return gradient, hessian, 0.5 * cost, finite_residual, finite_jacobian

    def saved_normal(
        jacobian_scratch,
        residuals,
        singleton_flags,
        block,
        observation_tile_count: ct.Constant[int],
    ):
        """Fold accepted trial values using the original stopping recipe."""
        use_singleton = ct.reshape(
            ct.load(singleton_flags, (block,), (1,)), ()
        )
        gradient_lanes = ct.zeros((8, 32), dtype=residuals.dtype)
        for observation_tile in range(observation_tile_count):
            jacobian = ct.reshape(
                ct.load(
                    jacobian_scratch,
                    (block, 0, observation_tile),
                    (1, 8, _OBSERVATIONS_PER_TILE),
                    padding_mode=ct.PaddingMode.ZERO,
                ),
                (8, 32),
            )
            residual = ct.load(
                residuals,
                (block, observation_tile),
                (1, _OBSERVATIONS_PER_TILE),
                padding_mode=ct.PaddingMode.ZERO,
            )
            gradient_lanes = gradient_lanes + jacobian * residual
        # Keep the cost recurrence out of the model/Jacobian loop.
        # Read the stored, masked residuals in the same observation order.
        cost_lanes = ct.zeros((32,), dtype=residuals.dtype)
        # Singleton cost uses four groups of eight lanes, adding four
        # adjacent residual squares per lane; other rows use 32 lanes.
        if known_non_singleton_cost or not use_singleton:
            for cost_tile in range(observation_tile_count):
                cost_values = ct.reshape(
                    ct.load(
                        residuals,
                        (block, cost_tile),
                        (1, _OBSERVATIONS_PER_TILE),
                        padding_mode=ct.PaddingMode.ZERO,
                    ),
                    (32,),
                )
                cost_lanes = cost_lanes + cost_values * cost_values
        else:
            for cost_tile in range(observation_tile_count):
                cost_residual = ct.load(
                    residuals,
                    (block, cost_tile),
                    (1, _OBSERVATIONS_PER_TILE),
                    padding_mode=ct.PaddingMode.ZERO,
                )
                quarter = cost_tile % 4
                cost_group = ct.extract(cost_lanes, (quarter,), (8,))
                residual_groups = ct.reshape(cost_residual, (8, 4))
                residual_value = ct.reshape(
                    ct.extract(residual_groups, (0, 0), (8, 1)), (8,)
                )
                cost_group = cost_group + residual_value * residual_value
                residual_value = ct.reshape(
                    ct.extract(residual_groups, (0, 1), (8, 1)), (8,)
                )
                cost_group = cost_group + residual_value * residual_value
                residual_value = ct.reshape(
                    ct.extract(residual_groups, (0, 2), (8, 1)), (8,)
                )
                cost_group = cost_group + residual_value * residual_value
                residual_value = ct.reshape(
                    ct.extract(residual_groups, (0, 3), (8, 1)), (8,)
                )
                cost_group = cost_group + residual_value * residual_value
                cost_lanes = ct.insert(cost_lanes, (quarter,), cost_group)
        if singleton_normals:
            # The same lower-half + upper-half tree used above, now reducing
            # 32 cost lanes to one: 32 -> 16 -> 8 -> 4 -> 2 -> 1.
            cost_half = ct.extract(cost_lanes, (0,), (16,)) + ct.extract(
                cost_lanes, (1,), (16,)
            )
            cost_half = ct.extract(cost_half, (0,), (8,)) + ct.extract(
                cost_half, (1,), (8,)
            )
            cost_half = ct.extract(cost_half, (0,), (4,)) + ct.extract(
                cost_half, (1,), (4,)
            )
            cost_half = ct.extract(cost_half, (0,), (2,)) + ct.extract(
                cost_half, (1,), (2,)
            )
            cost_half = ct.extract(cost_half, (0,), (1,)) + ct.extract(
                cost_half, (1,), (1,)
            )
            cost = ct.reshape(cost_half, ())
        if sm100_normals or group4_gradient:
            # Reduce the 32 lanes of each of eight parameter gradients in the
            # same fixed tree. An unconstrained sum can change association.
            gradient_half = ct.extract(
                gradient_lanes, (0, 0), (8, 16)
            ) + ct.extract(gradient_lanes, (0, 1), (8, 16))
            gradient_half = ct.extract(
                gradient_half, (0, 0), (8, 8)
            ) + ct.extract(gradient_half, (0, 1), (8, 8))
            gradient_half = ct.extract(
                gradient_half, (0, 0), (8, 4)
            ) + ct.extract(gradient_half, (0, 1), (8, 4))
            gradient_half = ct.extract(
                gradient_half, (0, 0), (8, 2)
            ) + ct.extract(gradient_half, (0, 1), (8, 2))
            gradient_half = ct.extract(
                gradient_half, (0, 0), (8, 1)
            ) + ct.extract(gradient_half, (0, 1), (8, 1))
            gradient = ct.reshape(gradient_half, (1, 8))
        else:
            gradient = ct.reshape(ct.sum(gradient_lanes, axis=1), (1, 8))
        return (
            gradient,
            ct.zeros((8, 8), residuals.dtype),
            0.5 * cost,
            ct.full((), True, ct.bool_),
            ct.full((), True, ct.bool_),
        )

    def control_scalar(bits: ct.Constant[int], parameters):
        if parameters.dtype == ct.float64:
            word = ct.full((), bits, ct.uint64)
        else:
            word = ct.full((), bits, ct.uint32)
        return ct.bitcast(word, parameters.dtype)

    def transition(
        x,
        g,
        h,
        step,
        info,
        current,
        cost,
        damping,
        count,
        rejected,
        status,
        parameters,
        images,
        weights,
        residuals,
        trial_residuals,
        jacobian_scratch,
        singleton_flags,
        finite_jacobian_output,
        accepted_jacobian_flags,
        block,
        height: ct.Constant[int],
        width: ct.Constant[int],
        planes: ct.Constant[int],
        tiles: ct.Constant[int],
        extent: ct.Constant[int],
        max_evaluations: ct.Constant[int],
        increase,
        decrease,
        eps,
        f_tol,
        x_tol,
        g_tol,
    ):
        if count >= max_evaluations:
            status = ct.full((), _MAX_EVALUATIONS, ct.int32)
        else:
            # Rejected trials leave x and its normal equations unchanged.
            # Host resumes invalidate active diagnostics when the contraction
            # schedule can change; accepted steps invalidate them below.
            finite_jacobian = ct.full((), True, ct.bool_)
            if ~current:
                reuse_jacobian = False
                if cache_normals:
                    reuse_jacobian = ct.reshape(
                        ct.load(accepted_jacobian_flags, (block,), (1,)), ()
                    )
                if reuse_jacobian:
                    g, h, cost, valid, finite_jacobian = saved_normal(
                        jacobian_scratch,
                        residuals,
                        singleton_flags,
                        block,
                        tiles,
                    )
                else:
                    g, h, cost, valid, finite_jacobian = normal(
                        x,
                        images,
                        weights,
                        residuals,
                        jacobian_scratch,
                        singleton_flags,
                        block,
                        height,
                        width,
                        planes,
                        tiles,
                        need_normal=True,
                    )
                if sm100_normals:
                    ct.store(
                        finite_jacobian_output,
                        (block,),
                        ct.reshape(finite_jacobian, (1,)),
                    )
                if singleton_normals:
                    if ct.reshape(
                        ct.load(singleton_flags, (block,), (1,)), ()
                    ):
                        g, h = singleton_normal(
                            jacobian_scratch, residuals, block
                        )
                    elif normal_recipe == "sm100-mma":
                        h = sm100_mma_hessian(jacobian_scratch, block, False)
                    elif normal_recipe == "sm100-gemmsn":
                        h = sm100_gemmsn_hessian(jacobian_scratch, block)
            if ~finite_jacobian:
                if cache_normals:
                    ct.store(
                        accepted_jacobian_flags,
                        (block,),
                        ct.full((1,), False, ct.bool_),
                    )
                status = ct.full((), _INVALID_RESIDUAL, ct.int32)
            else:
                current = ct.full((), True, ct.bool_)
                index = ct.arange(8, dtype=ct.int32)
                identity = ct.reshape(index, (8, 1)) == ct.reshape(
                    index, (1, 8)
                )
                diagonal = ct.sum(ct.where(identity, h, 0), axis=0)
                scale = ct.sqrt(ct.where(diagonal > 0, diagonal, 1.0))
                scale_row = ct.reshape(scale, (1, 8))
                scaled_gradient = g / scale_row
                scaled = ct.max(
                    ct.reshape(ct.abs(scaled_gradient), (-1,)),
                    propagate_nan=True,
                )
                # Zero requests normal equations without a trial step.
                if extent != 0:
                    residual_norm = ct.sqrt(2.0 * cost)
                    if ~all_finite(residual_norm):
                        # Squared cost can overflow for finite residuals.
                        # Keep overflow out of the gradient stopping test.
                        magnitude = ct.full((), 0.0, residuals.dtype)
                        for tile in range(tiles):
                            values = ct.load(
                                residuals,
                                (block, tile),
                                (1, 32),
                                padding_mode=ct.PaddingMode.ZERO,
                            )
                            magnitude = ct.maximum(
                                magnitude,
                                ct.max(ct.abs(values)),
                                propagate_nan=True,
                            )
                        divisor = ct.where(magnitude == 0, 1.0, magnitude)
                        sum_squares = ct.full((), 0.0, residuals.dtype)
                        for tile in range(tiles):
                            values = (
                                ct.load(
                                    residuals,
                                    (block, tile),
                                    (1, 32),
                                    padding_mode=ct.PaddingMode.ZERO,
                                )
                                / divisor
                            )
                            sum_squares = sum_squares + ct.sum(
                                values * values
                            )
                        residual_norm = magnitude * ct.sqrt(sum_squares)
                    if scaled <= g_tol * ct.maximum(
                        1.0, residual_norm, propagate_nan=True
                    ):
                        status = ct.full((), _CONVERGED_G_TOL, ct.int32)
                    else:
                        # Solve in Jacobian-scaled coordinates. An absolute
                        # diagonal floor changes the problem with parameter
                        # units and can falsely satisfy the gradient test.
                        scaled_hessian = (
                            h / ct.reshape(scale, (8, 1)) / scale_row
                        )
                        system = ct.sum(
                            ct.cat(
                                (
                                    ct.reshape(scaled_hessian, (1, 8, 8)),
                                    ct.reshape(
                                        damping * identity.astype(h.dtype),
                                        (1, 8, 8),
                                    ),
                                ),
                                axis=0,
                            ),
                            axis=0,
                            rounding_mode=ct.RoundingMode.RN,
                        )
                        scaled_step, info = solve(system, -scaled_gradient)
                        step = scaled_step / scale_row
                        if (info != 0) | ~all_finite(step):
                            status = ct.full((), _SINGULAR, ct.int32)
                        else:
                            trial_x = x + step
                            if cache_normals:
                                # A rejected trial overwrites J, but leaves
                                # the independent normal-equation cache valid.
                                ct.store(
                                    accepted_jacobian_flags,
                                    (block,),
                                    ct.full((1,), False, ct.bool_),
                                )
                            (
                                _,
                                _,
                                trial_cost,
                                trial_valid,
                                trial_finite_jacobian,
                            ) = normal(
                                trial_x,
                                images,
                                weights,
                                trial_residuals,
                                jacobian_scratch,
                                singleton_flags,
                                block,
                                height,
                                width,
                                planes,
                                tiles,
                                need_normal=False,
                            )
                            count += 1
                            acceptance_cost = cost
                            if (
                                sm100_normals
                                and not known_non_singleton_cost
                                and ct.reshape(
                                    ct.load(singleton_flags, (block,), (1,)),
                                    (),
                                )
                            ):
                                # Acceptance uses the singleton dot schedule;
                                # the earlier gradient stopping norm is kept.
                                acceptance_cost = sm100_singleton_cost(
                                    residuals, block
                                )
                                trial_cost = sm100_singleton_cost(
                                    trial_residuals, block
                                )
                            reduction = acceptance_cost - trial_cost
                            if trial_valid & (reduction > 0):
                                # Quadratic-model reduction using the
                                # damping that produced this step.
                                prediction = 0.5 * ct.sum(
                                    ct.reshape(
                                        scaled_step
                                        * (
                                            damping * scaled_step
                                            - scaled_gradient
                                        ),
                                        (-1,),
                                    )
                                )
                                x = trial_x
                                if cache_normals:
                                    ct.store(
                                        accepted_jacobian_flags,
                                        (block,),
                                        ct.reshape(
                                            trial_finite_jacobian, (1,)
                                        ),
                                    )
                                if sm100_normals:
                                    ct.store(
                                        finite_jacobian_output,
                                        (block,),
                                        ct.reshape(
                                            trial_finite_jacobian, (1,)
                                        ),
                                    )
                                old_cost = acceptance_cost
                                cost = trial_cost
                                current = ct.full((), False, ct.bool_)
                                damping = ct.maximum(
                                    eps,
                                    damping * decrease,
                                    propagate_nan=True,
                                )
                                rejected = ct.full((), 0, ct.int32)
                                for tile in range(tiles):
                                    r = ct.load(
                                        trial_residuals,
                                        (block, tile),
                                        (1, 32),
                                        padding_mode=ct.PaddingMode.ZERO,
                                    )
                                    ct.store(residuals, (block, tile), r)
                                f_threshold = f_tol * ct.maximum(
                                    1.0, old_cost, propagate_nan=True
                                )
                                if (
                                    (reduction <= f_threshold)
                                    & all_finite(prediction)
                                    & (prediction > 0)
                                    & (prediction <= f_threshold)
                                    & (0.5 * reduction <= prediction)
                                ):
                                    status = ct.full(
                                        (), _CONVERGED_F_TOL, ct.int32
                                    )
                                elif scaled_norm(scaled_step) <= x_tol * (
                                    x_tol + 1.0
                                ):
                                    status = ct.full(
                                        (), _CONVERGED_X_TOL, ct.int32
                                    )
                            else:
                                damping *= increase
                                rejected += 1
                                # Bound damping growth and rejection streaks.
                                if (damping > 1.0 / eps) | (
                                    rejected >= _MAX_REJECTED_STEPS
                                ):
                                    status = ct.where(
                                        trial_valid,
                                        _NO_PROGRESS,
                                        _INVALID_RESIDUAL,
                                    )
        return (
            x,
            g,
            h,
            step,
            info,
            current,
            cost,
            damping,
            count,
            rejected,
            status,
        )

    @ct.kernel(num_worker_warps=4)
    def iteration_kernel(
        parameters,
        images,
        weights,
        residuals,
        trial_residuals,
        gradients,
        hessians,
        steps,
        infos,
        statuses,
        evaluations,
        diagnostics_current,
        damping_output,
        rejected_output,
        jacobian_scratch,
        singleton_flags,
        finite_jacobian_output,
        height: ct.Constant[int],
        width: ct.Constant[int],
        planes: ct.Constant[int],
        tiles: ct.Constant[int],
        extent: ct.Constant[int],
        max_evaluations: ct.Constant[int],
        initial_damping_bits: ct.Constant[int],
        increase_bits: ct.Constant[int],
        decrease_bits: ct.Constant[int],
        eps_bits: ct.Constant[int],
        f_tol_bits: ct.Constant[int],
        x_tol_bits: ct.Constant[int],
        g_tol_bits: ct.Constant[int],
        accepted_jacobian_flags,
        normal_cost_output,
        resume: ct.Constant[bool],
        initialize_only: ct.Constant[bool],
    ):
        initial_damping = control_scalar(initial_damping_bits, parameters)
        increase = control_scalar(increase_bits, parameters)
        decrease = control_scalar(decrease_bits, parameters)
        eps = control_scalar(eps_bits, parameters)
        f_tol = control_scalar(f_tol_bits, parameters)
        x_tol = control_scalar(x_tol_bits, parameters)
        g_tol = control_scalar(g_tol_bits, parameters)
        block = ct.bid(0)
        x = ct.load(parameters, (block, 0), (1, 8))
        if resume:
            damping = ct.reshape(ct.load(damping_output, (block,), (1,)), ())
            count = ct.reshape(ct.load(evaluations, (block,), (1,)), ())
            rejected = ct.reshape(
                ct.load(rejected_output, (block,), (1,)), ()
            )
            status = ct.reshape(ct.load(statuses, (block,), (1,)), ()).astype(
                ct.int32
            )
            info = ct.reshape(ct.load(infos, (block,), (1,)), ())
            current = ct.reshape(
                ct.load(diagnostics_current, (block,), (1,)), ()
            )
            # Ordinary resumes may change the contraction schedule. The
            # fixed graph invalidates caches at entry and singleton changes.
            # Preserve terminal and exhausted rows' diagnostic-current flags.
            if not cache_normals and (status == _ACTIVE) & (
                count < max_evaluations
            ):
                current = ct.full((), False, ct.bool_)
            step = ct.load(steps, (block, 0), (1, 8))
            g = ct.load(gradients, (block, 0), (1, 8))
            h = ct.reshape(
                ct.load(hessians, (block, 0, 0), (1, 8, 8)), (8, 8)
            )
            # A graph cache becomes readable only after a valid normal step.
            # Its cost is the gradient-stopping cost, not acceptance_cost.
            cost = ct.full((), 0, parameters.dtype)
            if (
                cache_normals
                and (status == _ACTIVE) & (count < max_evaluations) & current
            ):
                cost = ct.reshape(
                    ct.load(normal_cost_output, (block,), (1,)), ()
                )
        else:
            damping = ct.full((), initial_damping, parameters.dtype)
            # The initial residual evaluation consumes the first budget slot.
            count = ct.full((), 1, ct.int32)
            rejected = ct.full((), 0, ct.int32)
            status = ct.full((), _ACTIVE, ct.int32)
            info = ct.full((), 0, ct.int32)
            current = ct.full((), False, ct.bool_)
            step = ct.zeros((1, 8), parameters.dtype)
            g, h, cost, valid, finite_jacobian = normal(
                x,
                images,
                weights,
                residuals,
                jacobian_scratch,
                singleton_flags,
                block,
                height,
                width,
                planes,
                tiles,
                need_normal=False,
            )
            if sm100_normals:
                ct.store(
                    finite_jacobian_output,
                    (block,),
                    ct.reshape(finite_jacobian, (1,)),
                )
            if ~all_finite(x) | ~valid:
                status = ct.full((), _INVALID_RESIDUAL, ct.int32)
        if extent == _PERSISTENT_TRANSITIONS and not initialize_only:
            while status == _ACTIVE:
                (
                    x,
                    g,
                    h,
                    step,
                    info,
                    current,
                    cost,
                    damping,
                    count,
                    rejected,
                    status,
                ) = transition(
                    x,
                    g,
                    h,
                    step,
                    info,
                    current,
                    cost,
                    damping,
                    count,
                    rejected,
                    status,
                    parameters,
                    images,
                    weights,
                    residuals,
                    trial_residuals,
                    jacobian_scratch,
                    singleton_flags,
                    finite_jacobian_output,
                    accepted_jacobian_flags,
                    block,
                    height,
                    width,
                    planes,
                    tiles,
                    extent,
                    max_evaluations,
                    increase,
                    decrease,
                    eps,
                    f_tol,
                    x_tol,
                    g_tol,
                )
        elif not initialize_only:
            if status == _ACTIVE:
                (
                    x,
                    g,
                    h,
                    step,
                    info,
                    current,
                    cost,
                    damping,
                    count,
                    rejected,
                    status,
                ) = transition(
                    x,
                    g,
                    h,
                    step,
                    info,
                    current,
                    cost,
                    damping,
                    count,
                    rejected,
                    status,
                    parameters,
                    images,
                    weights,
                    residuals,
                    trial_residuals,
                    jacobian_scratch,
                    singleton_flags,
                    finite_jacobian_output,
                    accepted_jacobian_flags,
                    block,
                    height,
                    width,
                    planes,
                    tiles,
                    extent,
                    max_evaluations,
                    increase,
                    decrease,
                    eps,
                    f_tol,
                    x_tol,
                    g_tol,
                )
        ct.store(parameters, (block, 0), x)
        ct.store(gradients, (block, 0), g)
        ct.store(hessians, (block, 0, 0), ct.reshape(h, (1, 8, 8)))
        ct.store(steps, (block, 0), step)
        ct.store(infos, (block,), ct.reshape(info, (1,)))
        ct.store(
            statuses,
            (block,),
            ct.reshape(status.astype(statuses.dtype), (1,)),
        )
        ct.store(evaluations, (block,), ct.reshape(count, (1,)))
        ct.store(diagnostics_current, (block,), ct.reshape(current, (1,)))
        ct.store(damping_output, (block,), ct.reshape(damping, (1,)))
        ct.store(rejected_output, (block,), ct.reshape(rejected, (1,)))
        if cache_normals:
            ct.store(normal_cost_output, (block,), ct.reshape(cost, (1,)))

    return iteration_kernel


@cache
def _singleton_graph_control(sm100_normals=False):
    """Refresh the bounded graph singleton schedule on the device."""
    _, runtime_ct = _runtime()
    if TYPE_CHECKING:
        import cuda.tile as ct
    else:
        ct = runtime_ct

    @ct.kernel
    def kernel(
        statuses,
        evaluations,
        singleton_flags,
        finite_jacobian_output,
        batch: ct.Constant[int],
        max_evaluations: ct.Constant[int],
        tile_extent: ct.Constant[int],
        diagnostics_current,
    ):
        rows = ct.arange(tile_extent, dtype=ct.int32)
        status = ct.load(
            statuses, (0,), (tile_extent,), padding_mode=ct.PaddingMode.ZERO
        )
        counts = ct.load(
            evaluations,
            (0,),
            (tile_extent,),
            padding_mode=ct.PaddingMode.ZERO,
        )
        eligible = (
            (rows < batch) & (status == _ACTIVE) & (counts < max_evaluations)
        )
        if sm100_normals:
            eligible &= ct.load(
                finite_jacobian_output,
                (0,),
                (tile_extent,),
                padding_mode=ct.PaddingMode.ZERO,
            )
        count = ct.sum(eligible.astype(ct.int32), axis=0)
        updated_flags = eligible & (count == 1)
        if sm100_normals:
            previous_flags = ct.load(
                singleton_flags,
                (0,),
                (tile_extent,),
                padding_mode=ct.PaddingMode.ZERO,
            )
            current = ct.load(
                diagnostics_current,
                (0,),
                (tile_extent,),
                padding_mode=ct.PaddingMode.ZERO,
            )
            active = (
                (rows < batch)
                & (status == _ACTIVE)
                & (counts < max_evaluations)
            )
            current &= ~(active & (previous_flags != updated_flags))
            ct.store(diagnostics_current, (0,), current)
        ct.store(singleton_flags, (0,), updated_flags)

    return kernel


def _gaussian_iterations(
    parameters,
    images,
    weights,
    *,
    image_shape,
    mode,
    singleton_normals=False,
    config=None,
):
    """Run fused Levenberg-Marquardt transitions on the current CUDA stream.

    Inputs must be ready on the current stream. This call completes that
    stream before releasing borrowed inputs and internal scratch buffers.
    The specialized contraction schedule resumes complete steps until all
    rows terminate; the remaining generic path runs a bounded device loop.
    """
    from .solver import LMConfig, _settings

    if (
        parameters.ndim != 2
        or parameters.shape[1] != _PARAMETER_COUNT
        or not len(parameters)
    ):
        raise ValueError("parameters must have shape (batch, 8)")
    if mode not in {"difference", "split"}:
        raise ValueError("mode must be difference or split")
    height, width = image_shape
    planes = 1 if mode == "difference" else 3
    observations = height * width * planes
    if (
        images.shape != (len(parameters), observations)
        or weights.shape != images.shape
    ):
        raise ValueError(
            "flattened images and weights must match the fit dimensions"
        )
    dtype = np.dtype(parameters.dtype)
    if dtype != np.dtype("float64"):
        raise TypeError("Tile/SIMT fitting currently requires float64 inputs")
    selected_config = LMConfig() if config is None else config
    if selected_config.solver_version != "v0.2":
        raise ValueError(
            "Tile/SIMT fitting supports solver_version='v0.2' only"
        )
    settings = _settings(selected_config, dtype, _PARAMETER_COUNT)
    if settings.use_finite_difference:
        raise ValueError(
            "Tile/SIMT Gaussian kernels require analytic derivatives"
        )
    cp, ct = _runtime()
    sm100_normals = False
    if singleton_normals:
        capability = cp.cuda.Device().compute_capability
        sm100_normals = capability == "100"
    stream = cp.cuda.get_current_stream()
    try:
        x = cp.array(parameters, copy=True, order="C")
        data = cp.asarray(images, dtype=dtype, order="C")
        weight = cp.asarray(weights, dtype=dtype, order="C")
        batch = len(x)
        outputs = {
            "parameters": x,
            "residuals": cp.empty_like(data),
            "gradients": cp.empty_like(x),
            "hessians": cp.empty(
                (batch, _PARAMETER_COUNT, _PARAMETER_COUNT), dtype=dtype
            ),
            "steps": cp.empty_like(x),
            "info": cp.empty(batch, dtype=cp.int32),
            "status": cp.empty(batch, dtype=cp.int8),
            "evaluations": cp.empty(batch, dtype=cp.int32),
            "diagnostics_current": cp.empty(batch, dtype=cp.bool_),
            "damping": cp.empty(batch, dtype=dtype),
            "rejected_steps": cp.empty(batch, dtype=cp.int32),
        }
        trial = cp.empty_like(data)
        jacobian_scratch = cp.empty(
            (batch, _PARAMETER_COUNT, observations)
            if singleton_normals
            else (1, 1, 1),
            dtype=dtype,
        )
        singleton_flags = cp.full(
            batch,
            False,
            dtype=cp.bool_,
        )
        # SM100 contraction cardinality excludes invalid-J rows.
        # The flag follows the accepted parameters; it never terminates a row.
        finite_jacobian_output = (
            cp.empty(batch, dtype=cp.bool_)
            if sm100_normals
            else singleton_flags
        )

        # Only graph kernels read this independent scratch-validity flag.
        accepted_jacobian_flags = (
            cp.zeros(batch, dtype=cp.bool_)
            if sm100_normals
            else singleton_flags
        )

        def iteration_kernel(eligible_count):
            options = {}
            if sm100_normals:
                # Match the qualified 51x51 reference's Hessian association:
                # MMA for 2-12 eligible rows; singleton has its own schedule.
                options["normal_recipe"] = (
                    "sm100-mma"
                    if 2 <= eligible_count <= _SM100_MMA_MAX_ELIGIBLE
                    else "sm100-gemmsn"
                )
                # At these boundaries every singleton flag is false. Keep the
                # per-row decision for count one and invalid-J companions.
                options["known_non_singleton_cost"] = eligible_count != 1
            # The qualified SM120 reference changes gradient association at
            # 39 eligible rows. This boundary is provider/shape-specific.
            if (
                not sm100_normals
                and eligible_count >= _GROUP4_GRADIENT_MIN_ELIGIBLE
            ):
                options["group4_gradient"] = True
            return _kernels(singleton_normals, **options)

        if sm100_normals:
            kernel = iteration_kernel(batch)
        else:
            kernel = _kernels(singleton_normals)
        arguments = (
            x,
            data,
            weight,
            outputs["residuals"],
            trial,
            outputs["gradients"],
            outputs["hessians"],
            outputs["steps"],
            outputs["info"],
            outputs["status"],
            outputs["evaluations"],
            outputs["diagnostics_current"],
            outputs["damping"],
            outputs["rejected_steps"],
            jacobian_scratch,
            singleton_flags,
            finite_jacobian_output,
            int(height),
            int(width),
            planes,
            math.ceil(observations / _OBSERVATIONS_PER_TILE),
        )
        controls = (
            settings.max_evaluations,
            *_lm_control_bits(settings, dtype),
            accepted_jacobian_flags,
        )
        # Establish validity before choosing the contraction cardinality.
        initialize_only = singleton_normals
        initial_extent = (
            _ONE_TRANSITION if initialize_only else _PERSISTENT_TRANSITIONS
        )
        ct.launch(
            stream,
            (batch,),
            kernel,
            arguments
            + (initial_extent,)
            + controls
            + (outputs["damping"], False, initialize_only),
        )
        if singleton_normals:
            # Keep control readbacks pinned while this stream completes.
            # Byte 0 holds the bool; bytes 1-7 pad to the int64 at byte 8.
            # All 16 bytes keep the values aligned and disjoint.
            host_control = cp.cuda.alloc_pinned_memory(16)
            host_active = np.ndarray((), dtype=np.bool_, buffer=host_control)
            host_count = np.ndarray(
                (), dtype=np.int64, buffer=host_control, offset=8
            )

            def read_control(value, target):
                value.get(out=target, stream=stream, blocking=True)
                return target.item()

            host_transitions = 0
            graph = None
            capture_stream = None
            while bool(
                read_control(
                    cp.any(outputs["status"] == _ACTIVE), host_active
                )
            ):
                eligible = (outputs["status"] == _ACTIVE) & (
                    outputs["evaluations"] < settings.max_evaluations
                )
                if sm100_normals:
                    eligible &= finite_jacobian_output
                eligible_count = int(
                    read_control(cp.count_nonzero(eligible), host_count)
                )
                if (
                    2 <= batch <= _GRAPH_MAX_BATCH
                    and host_transitions >= _GRAPH_MIN_HOST_TRANSITIONS
                    and (
                        (
                            capability == "120"
                            and 1
                            < eligible_count
                            < _GROUP4_GRADIENT_MIN_ELIGIBLE
                        )
                        or (
                            sm100_normals
                            and 2 <= eligible_count <= _SM100_MMA_MAX_ELIGIBLE
                        )
                    )
                    and graph is None
                ):
                    # Eligible membership cannot grow, so this contraction
                    # recipe stays valid. Keep singleton cost selection live.
                    # Warm real launches before capture to avoid lazy JIT.
                    graph_kernel = (
                        _kernels(
                            singleton_normals,
                            normal_recipe="sm100-mma",
                            known_non_singleton_cost=False,
                            cache_normals=True,
                        )
                        if sm100_normals
                        else kernel
                    )
                    control_kernel = (
                        _singleton_graph_control(True)
                        if sm100_normals
                        else _singleton_graph_control()
                    )
                    control_arguments = (
                        outputs["status"],
                        outputs["evaluations"],
                        singleton_flags,
                        finite_jacobian_output,
                        batch,
                        settings.max_evaluations,
                        # One power-of-two reduction tile covers every row;
                        # the control kernel masks slots beyond the batch.
                        1 << (batch - 1).bit_length(),
                        outputs["diagnostics_current"],
                    )
                    # Non-caching kernels never access this placeholder.
                    normal_cost_output = outputs["damping"]
                    if sm100_normals:
                        normal_cost_output = cp.empty(batch, dtype=dtype)
                        active = (outputs["status"] == _ACTIVE) & (
                            outputs["evaluations"] < settings.max_evaluations
                        )
                        outputs["diagnostics_current"][active] = False
                    step_arguments = (
                        arguments
                        + (_ONE_TRANSITION,)
                        + controls
                        + (normal_cost_output, True, False)
                    )

                    def graph_step(
                        target_stream,
                        control_kernel=control_kernel,
                        control_arguments=control_arguments,
                        step_arguments=step_arguments,
                        graph_kernel=graph_kernel,
                    ):
                        ct.launch(
                            target_stream,
                            (1,),
                            control_kernel,
                            control_arguments,
                        )
                        ct.launch(
                            target_stream,
                            (batch,),
                            graph_kernel,
                            step_arguments,
                        )

                    graph_step(stream)
                    if bool(
                        read_control(
                            cp.any(outputs["status"] == _ACTIVE), host_active
                        )
                    ):
                        # Capture records launches; execution stays on the
                        # caller stream, including legacy/default streams.
                        capture_stream = cp.cuda.Stream(non_blocking=True)
                        capturing = False
                        try:
                            capture_stream.begin_capture(
                                mode=cp.cuda.runtime.streamCaptureModeThreadLocal
                            )
                            capturing = True
                            for _graph_step in range(
                                _GRAPH_TRANSITIONS_PER_REPLAY
                            ):
                                graph_step(capture_stream)
                            graph = capture_stream.end_capture()
                            capturing = False
                        except BaseException:
                            if capturing:
                                try:
                                    capture_stream.end_capture()
                                except BaseException:
                                    pass
                            raise
                        while True:
                            graph.launch(stream=stream)
                            if not bool(
                                read_control(
                                    cp.any(outputs["status"] == _ACTIVE),
                                    host_active,
                                )
                            ):
                                break
                            remaining = (outputs["status"] == _ACTIVE) & (
                                outputs["evaluations"]
                                < settings.max_evaluations
                            )
                            if sm100_normals:
                                remaining &= finite_jacobian_output
                            if (
                                int(
                                    read_control(
                                        cp.count_nonzero(remaining),
                                        host_count,
                                    )
                                )
                                <= 1
                            ):
                                # Resume the persistent singleton loop.
                                break
                    continue
                # Re-select at every completed boundary because compaction can
                # change the reference contraction's active-batch schedule.
                selected_kernel = iteration_kernel(eligible_count)
                singleton_flags[:] = (
                    eligible if eligible_count == 1 else False
                )
                # Eligible membership cannot grow after this boundary.
                extent = (
                    _PERSISTENT_TRANSITIONS
                    if eligible_count == 1
                    else _ONE_TRANSITION
                )
                ct.launch(
                    stream,
                    (batch,),
                    selected_kernel,
                    arguments
                    + (extent,)
                    + controls
                    + (outputs["damping"], True, False),
                )
                host_transitions += 1
        stream.synchronize()
    except BaseException:
        # Setup or control can fail after asynchronous work is enqueued.
        # Retain borrowed buffers and scratch until this stream is drained.
        try:
            stream.synchronize()
        except BaseException:
            # Preserve the original failure, including KeyboardInterrupt.
            pass
        raise
    return outputs


def fit_gaussian_pipeline(
    images, *, model, initial, mask, variance, mode, resolved, config
):
    """Reuse public preparation and final diagnostics around fused LM."""
    from dataclasses import replace

    from .api import _fit_dipoles_backend
    from .backend import Backend
    from .solver import LMConfig, _finalize_lm, _settings

    selected_config = LMConfig() if config is None else config
    if selected_config.solver_version != "v0.2":
        raise ValueError(
            "Tile/SIMT fitting supports solver_version='v0.2' only"
        )
    if selected_config.use_finite_difference:
        raise ValueError("Tile/SIMT fitting requires analytic derivatives")
    cp, _ = _runtime()

    def run(problem, x0, observed, effective, fit_mode, lm_config):
        if np.dtype(x0.dtype) != np.dtype("float64"):
            raise TypeError(
                "Tile/SIMT fitting currently requires float64 inputs"
            )
        flat_images = observed.reshape(len(x0), -1)
        shape = tuple(observed.shape[-2:])
        capability = cp.cuda.Device().compute_capability
        singleton_normals = (
            np.dtype(x0.dtype) == np.dtype("float64")
            and fit_mode == "difference"
            and shape == _SPECIALIZED_IMAGE_SHAPE
            and capability in {"100", "120"}
        )
        out = _gaussian_iterations(
            x0,
            flat_images,
            effective.reshape(len(x0), -1),
            image_shape=shape,
            mode=fit_mode,
            singleton_normals=singleton_normals,
            config=lm_config,
        )
        k, n = x0.shape
        m = flat_images.shape[1]
        return _finalize_lm(
            problem,
            out["parameters"],
            out["residuals"],
            out["status"],
            out["evaluations"],
            cp.full((k, n, m), cp.nan, dtype=x0.dtype),
            cp.zeros(k, dtype=bool),
            out["diagnostics_current"],
            _settings(selected_config, np.dtype(x0.dtype), n),
            cp,
        )

    result = _fit_dipoles_backend(
        images,
        model=model,
        initial=initial,
        mask=mask,
        variance=variance,
        mode=mode,
        resolved=Backend("cutile", resolved.module, resolved.device),
        config=selected_config,
        fusion=True,
        _solver=run,
    )
    return replace(result, backend="tile-simt")
