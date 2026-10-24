# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Catalog quantities derived from Gaussian dipole fits with uncertainties."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .api import DeviceDipoleFitResult, DipoleFitResult
from .io import _array_module


@dataclass(frozen=True)
class DerivedDipoleQuantities:
    """Per-fit values and standard uncertainties on the fit's backend.

    Positions are in xFit's centered pixel coordinates. ``*_xy_covariance``
    is the covariance between the x and y of the same position.
    """

    center_x: Any
    center_y: Any
    center_x_error: Any
    center_y_error: Any
    center_xy_covariance: Any
    positive_x: Any
    positive_y: Any
    positive_x_error: Any
    positive_y_error: Any
    positive_xy_covariance: Any
    separation: Any
    separation_error: Any
    lobe_flux: Any
    lobe_flux_error: Any
    valid: Any


def derived_dipole_quantities(
    result: DipoleFitResult | DeviceDipoleFitResult,
) -> DerivedDipoleQuantities:
    """Propagate a Gaussian dipole fit's covariance to catalog quantities.

    - ``center``: the midpoint of the two lobes.
    - ``positive``: the position of the lobe with positive flux.
    - ``separation``: the distance between the lobes.
    - ``lobe_flux``: each lobe's total flux, ``2 pi |amplitude| sigma_x
      sigma_y``.

    Each standard uncertainty is ``sqrt(g C g^T)``, with ``C`` the fit's
    parameter covariance and ``g`` the quantity's derivatives with respect
    to the fitted parameters. The cross terms of ``C`` matter: the two lobe
    positions are correlated, and the amplitude and widths are strongly
    anticorrelated. Combining ``standard_errors`` as if independent misstates
    these uncertainties. The propagation is first order, so it holds while
    the uncertainties are small compared with the widths and the
    separation; ``separation_error`` is NaN for coincident lobes.

    Uncertainties are NaN where ``uncertainty_valid`` is false. Values are
    returned for every fit. When available, the fit's covariance factor is
    used to evaluate the same propagation before forming a dense covariance;
    this avoids cancellation between large correlated entries. Results
    constructed without that factor use the dense covariance, whose rounding
    can limit the accuracy of small derived uncertainties.
    """

    if result.model != "gaussian":
        raise ValueError("derived quantities require a Gaussian dipole fit")
    xp = _array_module(result.parameters)
    p = xp.asarray(result.parameters, dtype=xp.float64)
    values, gradients = _values_and_gradients(p, xp)
    factor = getattr(result, "covariance_factor", None)
    if factor is None:
        covariance = xp.asarray(result.covariance, dtype=xp.float64)
        propagated = xp.einsum(
            "bqi,bij,brj->bqr", gradients, covariance, gradients
        )
    else:
        projected = gradients @ xp.asarray(factor, dtype=xp.float64)
        propagated = projected @ projected.swapaxes(-2, -1)
    valid = xp.asarray(result.uncertainty_valid, dtype=bool)
    propagated = xp.where(valid[:, None, None], propagated, xp.nan)
    error = xp.sqrt(xp.diagonal(propagated, axis1=1, axis2=2))
    return DerivedDipoleQuantities(
        center_x=values[:, 0],
        center_y=values[:, 1],
        center_x_error=error[:, 0],
        center_y_error=error[:, 1],
        center_xy_covariance=propagated[:, 0, 1],
        positive_x=values[:, 2],
        positive_y=values[:, 3],
        positive_x_error=error[:, 2],
        positive_y_error=error[:, 3],
        positive_xy_covariance=propagated[:, 2, 3],
        separation=values[:, 4],
        separation_error=error[:, 4],
        lobe_flux=values[:, 5],
        lobe_flux_error=error[:, 5],
        valid=valid,
    )


def _values_and_gradients(p, xp):
    """Quantities ``(batch, 6)`` and their gradients ``(batch, 6, 8)``.

    Quantity order: center x and y, positive-lobe x and y, separation, lobe
    flux. Parameter order: :attr:`GaussianDipoleModel.parameter_names`.
    """

    batch = len(p)
    amplitude, sigma_x, sigma_y = p[:, 0], p[:, 1], p[:, 2]
    dx, dy = p[:, 4] - p[:, 6], p[:, 5] - p[:, 7]
    separation = xp.hypot(dx, dy)
    apart = separation > 0
    safe = xp.where(apart, separation, 1.0)
    positive = amplitude >= 0
    rows = xp.arange(batch)

    values = xp.stack(
        [
            0.5 * (p[:, 4] + p[:, 6]),
            0.5 * (p[:, 5] + p[:, 7]),
            xp.where(positive, p[:, 4], p[:, 6]),
            xp.where(positive, p[:, 5], p[:, 7]),
            separation,
            2.0 * xp.pi * xp.abs(amplitude) * sigma_x * sigma_y,
        ],
        axis=1,
    )
    g = xp.zeros((batch, 6, 8), dtype=xp.float64)
    g[:, 0, 4] = g[:, 0, 6] = 0.5
    g[:, 1, 5] = g[:, 1, 7] = 0.5
    g[rows, 2, xp.where(positive, 4, 6)] = 1.0
    g[rows, 3, xp.where(positive, 5, 7)] = 1.0
    ux = xp.where(apart, dx / safe, xp.nan)
    uy = xp.where(apart, dy / safe, xp.nan)
    g[:, 4, 4], g[:, 4, 5], g[:, 4, 6], g[:, 4, 7] = ux, uy, -ux, -uy
    sign = xp.where(positive, 1.0, -1.0)
    g[:, 5, 0] = 2.0 * xp.pi * sign * sigma_x * sigma_y
    g[:, 5, 1] = 2.0 * xp.pi * xp.abs(amplitude) * sigma_y
    g[:, 5, 2] = 2.0 * xp.pi * xp.abs(amplitude) * sigma_x
    return values, g
