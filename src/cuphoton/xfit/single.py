# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""One-lobe Gaussian fits for single sources.

A source present in only one of the two images leaves one lobe in the
difference image. Fitted with the two-lobe dipole model, the spare lobe
settles on noise and the measured lobe's position inherits its freedom, so
the dipole covariance overstates the position uncertainty. This module fits
the one lobe alone, with the same solver and weighting as
:func:`cuphoton.xfit.fit_dipoles`.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from ._types import COMPUTE_DTYPES, ArrayLike, ComputeDType, FloatDType
from .api import DipoleFitUncertaintyReason, _floating_dtype
from .io import _array_module
from .models import GaussianDipoleModel
from .solver import (
    BatchedLeastSquaresProblem,
    LMConfig,
    batched_levenberg_marquardt,
)

ELLIPTICAL_PARAMETERS = ("amplitude", "sigma_x", "sigma_y", "theta", "x", "y")
CIRCULAR_PARAMETERS = ("amplitude", "x", "y")


@dataclass(frozen=True)
class SingleSourceFitResult:
    """Per-stamp one-lobe fit outputs on the input backend.

    Positions are in xFit's centered pixel coordinates. ``parameters`` and
    ``covariance`` follow ``parameter_names``; ``position_error_x`` and
    ``position_error_y`` are the square roots of the covariance's position
    diagonal and ``position_xy_covariance`` its off-diagonal term. ``flux``
    is ``2 pi |amplitude| sigma_x sigma_y`` with its propagated error.
    ``sign`` is ``+1`` for a brightening and ``-1`` for a fading source.
    ``status_codes`` hold :class:`cuphoton.xfit.LMStatus` values and
    ``uncertainty_reason_codes`` hold
    :class:`cuphoton.xfit.DipoleFitUncertaintyReason` values. Uncertainties
    are NaN where ``uncertainty_valid`` is false; values are returned for
    every fit. ``residuals`` are model minus data in the stamp shape.
    """

    parameters: Any
    parameter_names: tuple[str, ...]
    covariance: Any
    standard_errors: Any
    x: Any
    y: Any
    position_error_x: Any
    position_error_y: Any
    position_xy_covariance: Any
    flux: Any
    flux_error: Any
    sign: Any
    status_codes: Any
    converged: Any
    evaluations: Any
    chi_square: Any
    null_chi_square: Any
    degrees_of_freedom: Any
    reduced_chi_square: Any
    uncertainty_valid: Any
    uncertainty_reason_codes: Any
    residuals: Any
    dtype: FloatDType


def fit_single_sources(
    images: ArrayLike,
    variance: ArrayLike,
    positions: ArrayLike,
    psf_sigma: float | ArrayLike | None = None,
    *,
    mask: ArrayLike | None = None,
    sign: int | ArrayLike | None = None,
    solver: LMConfig | None = None,
    compute_dtype: ComputeDType = "input",
) -> SingleSourceFitResult:
    """Fit one Gaussian lobe per stamp by weighted least squares.

    ``images`` and ``variance`` have shape ``(batch, height, width)`` and
    ``positions`` has shape ``(batch, 2)`` as ``(x, y)`` starting positions
    in xFit's centered pixel coordinates. Without ``psf_sigma`` the lobe is
    an elliptical Gaussian with free ``amplitude, sigma_x, sigma_y, theta,
    x, y``; the widths are optimized in log space and returned positive, and
    ``theta`` is canonicalized to ``[-pi/2, pi/2)``. With ``psf_sigma`` (one
    value or one per stamp, in pixels) the lobe is circular with that fixed
    width and only ``amplitude, x, y`` are fitted.

    ``sign`` is ``+1`` for a brightening source (positive lobe) or ``-1`` for
    a fading one, one value or one per stamp; it sets the sign of the
    starting amplitude, whose size comes from a matched filter at the
    starting position. Without ``sign`` the matched filter's sign is used.
    The fitted amplitude is free, so ``parameters[:, 0]`` may end with the
    other sign when the data demand it.

    Each pixel is weighted by the inverse square root of its variance, as
    :func:`cuphoton.xfit.fit_dipoles` does with a variance. A pixel is
    usable when ``mask`` (if given, same shape as ``images``) is true and
    the image and a positive variance are finite there; other pixels carry
    zero weight. The covariance is the inverse Gauss-Newton Hessian at the
    solution and is not rescaled by the reduced chi-square: it is right when
    ``variance`` describes the noise. Correlated noise between pixels makes
    it too small, as it does for every pixel-weighted fit.

    ``solver`` passes :class:`cuphoton.xfit.LMConfig` controls through.
    ``compute_dtype`` is ``"input"`` (the image's float dtype, float64 for
    non-float input), ``"float32"`` or ``"float64"``.

    Computation follows the backend of ``images``: CuPy images return CuPy
    arrays, NumPy images NumPy arrays.
    """

    if compute_dtype not in COMPUTE_DTYPES:
        raise ValueError(
            "compute_dtype must be 'input', 'float32' or 'float64'"
        )
    xp = _array_module(images)
    data = xp.asarray(images)
    if data.ndim != 3:
        raise ValueError("images must have shape (batch, height, width)")
    dtype = (
        _floating_dtype(data)
        if compute_dtype == "input"
        else np.dtype(compute_dtype)
    )
    data = data.astype(dtype, copy=False)
    batch, height, width = (int(value) for value in data.shape)
    var = xp.asarray(variance)
    if var.shape != data.shape:
        raise ValueError("variance must share the shape of images")
    var = var.astype(dtype, copy=False)
    xy = xp.asarray(positions).astype(dtype, copy=False)
    if xy.shape != (batch, 2):
        raise ValueError("positions must have shape (batch, 2)")
    if not bool(xp.isfinite(xy).all()):
        raise ValueError("positions must contain only finite values")

    good = xp.isfinite(data) & xp.isfinite(var) & (var > 0)
    if mask is not None:
        mask_array = xp.asarray(mask)
        if mask_array.shape != data.shape:
            raise ValueError("mask must share the shape of images")
        good &= mask_array.astype(bool)
    weights = xp.where(
        good, 1.0 / xp.sqrt(xp.where(good, var, 1.0)), 0.0
    ).astype(dtype, copy=False)
    image = xp.where(good, data, 0.0).astype(dtype, copy=False)

    fixed_width = psf_sigma is not None
    if fixed_width:
        fixed_sigma = xp.broadcast_to(
            xp.asarray(psf_sigma).astype(dtype, copy=False), (batch,)
        )
        if not bool((xp.isfinite(fixed_sigma) & (fixed_sigma > 0)).all()):
            raise ValueError("psf_sigma must be finite and positive")
        names = CIRCULAR_PARAMETERS
    else:
        fixed_sigma = xp.full(
            batch, max(1.0, min(height, width) / 8.0), dtype=dtype
        )
        names = ELLIPTICAL_PARAMETERS
    count = len(names)

    model = GaussianDipoleModel(
        (height, width),
        backend="numpy" if xp is np else "cupy",
        dtype=dtype,
    )
    zeros = xp.zeros(batch, dtype=dtype)

    def lobe(amplitude, sigma_x, sigma_y, theta, x, y, *, derivatives):
        return model._star_with_derivatives(
            amplitude, sigma_x, sigma_y, theta, x, y, derivatives=derivatives
        )

    # Matched-filter amplitude at the starting position for the initial guess.
    template = lobe(
        xp.ones(batch, dtype=dtype),
        fixed_sigma,
        fixed_sigma,
        zeros,
        xy[:, 0],
        xy[:, 1],
        derivatives=False,
    )[0]
    squared_weights = weights * weights
    information = (squared_weights * template * template).sum(axis=(1, 2))
    safe_information = xp.where(information > 0, information, 1.0)
    matched = (squared_weights * template * image).sum(
        axis=(1, 2)
    ) / safe_information
    if sign is None:
        sign_array = xp.where(matched < 0, -1, 1).astype(np.int8)
    else:
        sign_array = xp.broadcast_to(xp.asarray(sign), (batch,))
        if not bool(((sign_array == 1) | (sign_array == -1)).all()):
            raise ValueError("sign values must be +1 or -1")
        sign_array = sign_array.astype(np.int8)
    signed = sign_array.astype(dtype)
    amplitude_floor = 1.0 / xp.sqrt(safe_information)
    amplitude0 = signed * xp.maximum(signed * matched, amplitude_floor)

    if fixed_width:
        initial = xp.stack((amplitude0, xy[:, 0], xy[:, 1]), axis=1)
    else:
        log_sigma = xp.log(fixed_sigma)
        initial = xp.stack(
            (amplitude0, log_sigma, log_sigma, zeros, xy[:, 0], xy[:, 1]),
            axis=1,
        )
    initial = initial.astype(dtype, copy=False)

    def error_state():
        if xp is np:
            return np.errstate(over="ignore", invalid="ignore")
        return nullcontext()

    def physical(parameters, indices):
        amplitude = parameters[:, 0]
        if fixed_width:
            sigma = fixed_sigma[indices]
            return (
                amplitude,
                sigma,
                sigma,
                zeros[indices],
                parameters[:, 1],
                parameters[:, 2],
            )
        with error_state():
            sigma_x = xp.exp(parameters[:, 1])
            sigma_y = xp.exp(parameters[:, 2])
        return (
            amplitude,
            sigma_x,
            sigma_y,
            parameters[:, 3],
            parameters[:, 4],
            parameters[:, 5],
        )

    def residual(parameters, *, indices):
        star = lobe(*physical(parameters, indices), derivatives=False)[0]
        value = (star - image[indices]) * weights[indices]
        return value.reshape(parameters.shape[0], -1)

    def jacobian(parameters, *, indices):
        terms = physical(parameters, indices)
        _, d_amplitude, d_sigma_x, d_sigma_y, d_theta, d_x, d_y = lobe(
            *terms, derivatives=True
        )
        if fixed_width:
            columns = (d_amplitude, d_x, d_y)
        else:
            columns = (
                d_amplitude,
                d_sigma_x * terms[1][:, None, None],
                d_sigma_y * terms[2][:, None, None],
                d_theta,
                d_x,
                d_y,
            )
        value = xp.stack(columns, axis=1) * weights[indices, None]
        return value.reshape(parameters.shape[0], count, -1)

    low_level = batched_levenberg_marquardt(
        BatchedLeastSquaresProblem(residual, jacobian),
        initial,
        config=solver,
    )
    all_rows = xp.arange(batch)
    terms = physical(low_level.parameters, all_rows)
    amplitude, sigma_x, sigma_y, theta, x, y = terms
    parameters = xp.stack(
        (amplitude, x, y) if fixed_width else terms, axis=1
    ).astype(dtype, copy=False)
    if not fixed_width:
        half_period = np.pi / 2
        parameters[:, 3] = (
            (parameters[:, 3] + half_period) % (2 * half_period)
        ) - half_period

    residuals = lobe(*terms, derivatives=False)[0] - image
    weighted = residuals * weights
    chi_square = (weighted * weighted).sum(axis=(1, 2))
    null = image * weights
    null_chi_square = (null * null).sum(axis=(1, 2))
    degrees_of_freedom = (weights != 0).sum(axis=(1, 2)).astype(
        np.int64
    ) - count
    positive_dof = degrees_of_freedom > 0
    reduced_chi_square = xp.where(
        positive_dof,
        chi_square / xp.where(positive_dof, degrees_of_freedom, 1),
        xp.nan,
    ).astype(np.float64)

    covariance = low_level.covariance
    if not fixed_width:
        chain = xp.ones_like(parameters)
        chain[:, 1] = sigma_x
        chain[:, 2] = sigma_y
        covariance = covariance * chain[:, :, None] * chain[:, None, :]
    converged = low_level.converged.astype(bool)
    rank = low_level.rank.astype(np.int64)
    jacobian_available = xp.isfinite(low_level.jacobian).all(axis=(1, 2))
    covariance_finite = xp.isfinite(covariance).all(axis=(1, 2))
    diagonal = xp.diagonal(covariance, axis1=1, axis2=2)
    diagonal_valid = xp.isfinite(diagonal).all(axis=1) & ~(diagonal < 0).any(
        axis=1
    )

    reasons = xp.full(batch, int(DipoleFitUncertaintyReason.VALID), np.int8)
    reasons[~diagonal_valid] = int(
        DipoleFitUncertaintyReason.COVARIANCE_DIAGONAL_INVALID
    )
    reasons[~covariance_finite] = int(
        DipoleFitUncertaintyReason.COVARIANCE_NOT_FINITE
    )
    reasons[rank < count] = int(
        DipoleFitUncertaintyReason.RANK_DEFICIENT_JACOBIAN
    )
    reasons[~jacobian_available] = int(
        DipoleFitUncertaintyReason.FINAL_JACOBIAN_UNAVAILABLE
    )
    reasons[~positive_dof] = int(
        DipoleFitUncertaintyReason.INSUFFICIENT_DEGREES_OF_FREEDOM
    )
    reasons[~converged] = int(DipoleFitUncertaintyReason.FIT_NOT_CONVERGED)
    valid = reasons == int(DipoleFitUncertaintyReason.VALID)
    covariance = xp.where(valid[:, None, None], covariance, xp.nan)
    standard_errors = xp.where(
        valid[:, None],
        xp.sqrt(xp.where(diagonal_valid[:, None], diagonal, 0.0)),
        xp.nan,
    )

    x_index, y_index = count - 2, count - 1
    flux = 2.0 * np.pi * xp.abs(amplitude) * sigma_x * sigma_y
    flux_gradient = xp.zeros((batch, count), dtype=dtype)
    flux_gradient[:, 0] = 2.0 * np.pi * xp.sign(amplitude) * sigma_x * sigma_y
    if not fixed_width:
        flux_gradient[:, 1] = 2.0 * np.pi * xp.abs(amplitude) * sigma_y
        flux_gradient[:, 2] = 2.0 * np.pi * xp.abs(amplitude) * sigma_x
    flux_variance = xp.einsum(
        "bi,bij,bj->b", flux_gradient, covariance, flux_gradient
    )

    return SingleSourceFitResult(
        parameters=parameters,
        parameter_names=names,
        covariance=covariance,
        standard_errors=standard_errors,
        x=parameters[:, x_index],
        y=parameters[:, y_index],
        position_error_x=standard_errors[:, x_index],
        position_error_y=standard_errors[:, y_index],
        position_xy_covariance=covariance[:, x_index, y_index],
        flux=flux,
        flux_error=xp.sqrt(flux_variance),
        sign=sign_array,
        status_codes=low_level.status.astype(np.int8),
        converged=converged,
        evaluations=low_level.evaluations.astype(np.int64),
        chi_square=chi_square,
        null_chi_square=null_chi_square,
        degrees_of_freedom=degrees_of_freedom,
        reduced_chi_square=reduced_chi_square,
        uncertainty_valid=valid,
        uncertainty_reason_codes=reasons,
        residuals=residuals,
        dtype=cast(FloatDType, dtype.name),
    )


__all__ = ["SingleSourceFitResult", "fit_single_sources"]
