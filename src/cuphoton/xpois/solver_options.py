# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


"""Shared XPOIS solver-option validation."""

from __future__ import annotations

from .spatial_als import SpatialALSConfig


def resolve_spatial_als_config(
    solver: str,
    *,
    background_degree: int,
    flux_conserve: bool,
    spatial_degree: int | None,
    als_iterations: int | None,
    als_tolerance: float | None,
    als_regularization: float | None,
) -> SpatialALSConfig | None:
    """Build the spatial ALS config, rejecting its options for other solvers.

    ``None`` options resolve to the :class:`SpatialALSConfig` defaults so the
    dataclass remains the single source of those values.
    """

    spatial_options = {
        "spatial_degree": spatial_degree,
        "max_iterations": als_iterations,
        "tolerance": als_tolerance,
        "regularization": als_regularization,
    }
    if solver != "spatial-als":
        if any(value is not None for value in spatial_options.values()):
            raise ValueError(
                "spatial ALS options require solver='spatial-als'"
            )
        return None
    defaults = SpatialALSConfig()
    return SpatialALSConfig(
        background_degree=background_degree,
        flux_conserve=flux_conserve,
        spatial_degree=(
            defaults.spatial_degree
            if spatial_degree is None
            else spatial_degree
        ),
        max_iterations=(
            defaults.max_iterations
            if als_iterations is None
            else als_iterations
        ),
        tolerance=(
            defaults.tolerance if als_tolerance is None else als_tolerance
        ),
        regularization=(
            defaults.regularization
            if als_regularization is None
            else als_regularization
        ),
    )
