# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Dipole image models for xFit."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import numpy as np
import numpy.typing as npt

from ._types import (
    FIT_MODES,
    STAMP_EVALUATIONS,
    ArrayLike,
    BackendArray,
    BackendRequest,
    FitMode,
    ModelName,
    StampEvaluation,
)
from .backend import as_numpy, resolve_backend


def _image_shape(value: tuple[int, int]) -> tuple[int, int]:
    if len(value) != 2:
        raise ValueError("image_shape must contain (height, width)")
    height, width = (int(value[0]), int(value[1]))
    if height < 1 or width < 1:
        raise ValueError("image_shape dimensions must be positive")
    return height, width


def _dtype(value: npt.DTypeLike) -> np.dtype[Any]:
    dtype = np.dtype(value)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("xFit models support float32 and float64")
    return dtype


def _split_weights(value: tuple[float, float, float]) -> tuple[float, ...]:
    if len(value) != 3:
        raise ValueError("split_weights must contain three values")
    weights = tuple(float(item) for item in value)
    if not np.isfinite(weights).all() or min(weights) < 0:
        raise ValueError("split_weights must be finite and nonnegative")
    return weights


def _validate_mode(mode: FitMode) -> None:
    if mode not in FIT_MODES:
        raise ValueError("mode must be 'difference' or 'split'")


class GaussianDipoleModel:
    """Rotated elliptical-Gaussian positive/negative dipole model.

    Positions use centered pixel coordinates: the central pixel is ``(0, 0)``
    for odd dimensions, and even-sized images straddle zero symmetrically.
    ``sigma_x`` and ``sigma_y`` are strictly positive standard deviations in
    pixels. :func:`cuphoton.xfit.fit_dipoles` optimizes them in log space and
    always returns positive widths.
    """

    parameter_names = (
        "amplitude",
        "sigma_x",
        "sigma_y",
        "theta",
        "x_pos",
        "y_pos",
        "x_neg",
        "y_neg",
    )
    name: ModelName = "gaussian"

    def __init__(
        self,
        image_shape: tuple[int, int],
        *,
        split_weights: tuple[float, float, float] = (1.0, 0.5, 0.5),
        backend: BackendRequest = "numpy",
        dtype: npt.DTypeLike = np.float64,
    ) -> None:
        resolved = resolve_backend(backend)
        self.image_shape = _image_shape(image_shape)
        self.split_weights = _split_weights(split_weights)
        self.backend = resolved.name
        self.device = resolved.device
        self.dtype = _dtype(dtype)
        self._ap = resolved.module

    def to_backend(
        self, backend: BackendRequest, *, dtype: npt.DTypeLike | None = None
    ) -> GaussianDipoleModel:
        """Return an equivalent model on ``backend``."""

        return GaussianDipoleModel(
            self.image_shape,
            split_weights=self.split_weights,
            backend=backend,
            dtype=self.dtype if dtype is None else dtype,
        )

    def _coordinates(self) -> tuple[BackendArray, BackendArray]:
        height, width = self.image_shape
        ap = self._ap
        x = ap.arange(width, dtype=self.dtype) - (width - 1) / 2
        y = ap.arange(height, dtype=self.dtype) - (height - 1) / 2
        return x[None, None, :], y[None, :, None]

    def _star_with_derivatives(
        self,
        amplitude: BackendArray,
        sigma_x: BackendArray,
        sigma_y: BackendArray,
        theta: BackendArray,
        x_center: BackendArray,
        y_center: BackendArray,
    ) -> tuple[BackendArray, ...]:
        ap = self._ap
        x, y = self._coordinates()
        amplitude = amplitude[..., None, None]
        sigma_x = sigma_x[..., None, None]
        sigma_y = sigma_y[..., None, None]
        theta = theta[..., None, None]
        x_hat = x - x_center[..., None, None]
        y_hat = y - y_center[..., None, None]
        cosine = ap.cos(theta)
        sine = ap.sin(theta)
        x_rotated = cosine * x_hat + sine * y_hat
        y_rotated = cosine * y_hat - sine * x_hat
        valid_sigma_x = ap.isfinite(sigma_x) & (sigma_x > 0)
        valid_sigma_y = ap.isfinite(sigma_y) & (sigma_y > 0)
        sigma_x = ap.where(valid_sigma_x, sigma_x, ap.nan)
        sigma_y = ap.where(valid_sigma_y, sigma_y, ap.nan)
        error_state = (
            np.errstate(
                divide="ignore",
                invalid="ignore",
                over="ignore",
                under="ignore",
            )
            if ap is np
            else nullcontext()
        )
        with error_state:
            inv_x2 = 1.0 / (sigma_x * sigma_x)
            inv_y2 = 1.0 / (sigma_y * sigma_y)
            exponent = -0.5 * (
                x_rotated * x_rotated * inv_x2
                + y_rotated * y_rotated * inv_y2
            )
            unit = ap.exp(exponent)
            star = amplitude * unit
            d_amplitude = unit
            d_sigma_x = star * x_rotated * x_rotated / (sigma_x**3)
            d_sigma_y = star * y_rotated * y_rotated / (sigma_y**3)
            d_theta = -star * x_rotated * y_rotated * (inv_x2 - inv_y2)
            d_x = star * (
                cosine * x_rotated * inv_x2 - sine * y_rotated * inv_y2
            )
            d_y = star * (
                sine * x_rotated * inv_x2 + cosine * y_rotated * inv_y2
            )
        return star, d_amplitude, d_sigma_x, d_sigma_y, d_theta, d_x, d_y

    def _parameters(
        self, parameters: ArrayLike, *, validate: bool
    ) -> BackendArray:
        ap = self._ap
        parameters = ap.asarray(parameters, dtype=self.dtype)
        if parameters.ndim != 2 or parameters.shape[1] != 8:
            raise ValueError("Gaussian parameters must have shape (batch, 8)")
        if validate:
            portable = as_numpy(parameters)
            if not np.isfinite(portable).all():
                raise ValueError(
                    "Gaussian parameters must contain only finite values"
                )
            if np.any(portable[:, 1:3] <= 0):
                raise ValueError("sigma_x and sigma_y must be positive")
        return parameters

    def _evaluate(
        self,
        parameters: BackendArray,
        *,
        mode: FitMode,
        validate: bool,
    ) -> BackendArray:
        _validate_mode(mode)
        ap = self._ap
        parameters = self._parameters(parameters, validate=validate)
        common = parameters[:, :4]
        positive = self._star_with_derivatives(
            common[:, 0],
            common[:, 1],
            common[:, 2],
            common[:, 3],
            parameters[:, 4],
            parameters[:, 5],
        )[0]
        negative = self._star_with_derivatives(
            common[:, 0],
            common[:, 1],
            common[:, 2],
            common[:, 3],
            parameters[:, 6],
            parameters[:, 7],
        )[0]
        difference = positive - negative
        if mode == "difference":
            return difference
        return ap.stack((difference, positive, negative), axis=1)

    def evaluate(
        self, parameters: ArrayLike, *, mode: FitMode = "difference"
    ) -> BackendArray:
        """Evaluate a batch of dipoles in difference or three-plane mode."""

        return self._evaluate(parameters, mode=mode, validate=True)

    def _evaluate_positive_unchecked(
        self, parameters: BackendArray, *, mode: FitMode = "difference"
    ) -> BackendArray:
        return self._evaluate(parameters, mode=mode, validate=False)

    def _jacobian(
        self,
        parameters: BackendArray,
        *,
        mode: FitMode,
        validate: bool,
    ) -> BackendArray:
        _validate_mode(mode)
        ap = self._ap
        parameters = self._parameters(parameters, validate=validate)
        common = parameters[:, :4]
        pos = self._star_with_derivatives(
            common[:, 0],
            common[:, 1],
            common[:, 2],
            common[:, 3],
            parameters[:, 4],
            parameters[:, 5],
        )
        neg = self._star_with_derivatives(
            common[:, 0],
            common[:, 1],
            common[:, 2],
            common[:, 3],
            parameters[:, 6],
            parameters[:, 7],
        )
        batch = parameters.shape[0]
        height, width = self.image_shape
        if mode == "difference":
            result = ap.zeros((batch, 8, height, width), dtype=self.dtype)
            for parameter in range(4):
                result[:, parameter] = pos[parameter + 1] - neg[parameter + 1]
            result[:, 4] = pos[5]
            result[:, 5] = pos[6]
            result[:, 6] = -neg[5]
            result[:, 7] = -neg[6]
            return result

        result = ap.zeros((batch, 8, 3, height, width), dtype=self.dtype)
        for parameter in range(4):
            result[:, parameter, 0] = pos[parameter + 1] - neg[parameter + 1]
            result[:, parameter, 1] = pos[parameter + 1]
            result[:, parameter, 2] = neg[parameter + 1]
        result[:, 4, 0] = pos[5]
        result[:, 4, 1] = pos[5]
        result[:, 5, 0] = pos[6]
        result[:, 5, 1] = pos[6]
        result[:, 6, 0] = -neg[5]
        result[:, 6, 2] = neg[5]
        result[:, 7, 0] = -neg[6]
        result[:, 7, 2] = neg[6]
        return result

    def jacobian(
        self, parameters: ArrayLike, *, mode: FitMode = "difference"
    ) -> BackendArray:
        """Return the analytic parameter-major model Jacobian."""

        return self._jacobian(parameters, mode=mode, validate=True)

    def _jacobian_positive_unchecked(
        self, parameters: BackendArray, *, mode: FitMode = "difference"
    ) -> BackendArray:
        return self._jacobian(parameters, mode=mode, validate=False)


class StampDipoleModel:
    """Dipole model formed by shifting a sampled PSF/stamp basis.

    ``evaluation="bilinear"`` samples a shifted continuous interpolant.
    ``evaluation="bilinear-vignetted"`` renormalizes its visible pixels to
    the requested flux after cropping at the destination boundary.
    ``evaluation="finite-volume"`` integrates the piecewise-constant source
    cells over each destination pixel. Both paths support rectangular source
    and destination grids.
    """

    parameter_names = ("x_pos", "y_pos", "x_neg", "y_neg", "flux")
    name: ModelName = "stamp"

    def __init__(
        self,
        stamp_basis: ArrayLike,
        *,
        image_shape: tuple[int, int],
        evaluation: StampEvaluation = "bilinear",
        scale: float = 1.0,
        basis_weights: ArrayLike | None = None,
        split_weights: tuple[float, float, float] = (1.0, 0.5, 0.5),
        backend: BackendRequest = "numpy",
        dtype: npt.DTypeLike | None = None,
    ) -> None:
        if evaluation not in STAMP_EVALUATIONS:
            raise ValueError(
                "evaluation must be 'bilinear', 'bilinear-vignetted', "
                "or 'finite-volume'"
            )
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("scale must be finite and positive")
        basis_numpy = as_numpy(stamp_basis)
        if basis_numpy.ndim == 2:
            basis_numpy = basis_numpy[None, :, :]
        if basis_numpy.ndim != 3 or min(basis_numpy.shape) < 1:
            raise ValueError(
                "stamp_basis must have shape (y, x) or (modes, y, x)"
            )
        selected_dtype = _dtype(
            basis_numpy.dtype
            if dtype is None and basis_numpy.dtype.kind == "f"
            else (np.float64 if dtype is None else dtype)
        )
        basis_numpy = np.asarray(basis_numpy, dtype=selected_dtype)
        if not np.isfinite(basis_numpy).all():
            raise ValueError("stamp_basis must contain only finite values")
        modes = basis_numpy.shape[0]
        if basis_weights is None:
            if modes != 1:
                raise ValueError(
                    "basis_weights is required when stamp_basis has "
                    "multiple modes"
                )
            weights_numpy = np.ones(1, dtype=selected_dtype)
        else:
            weights_numpy = np.asarray(basis_weights, dtype=selected_dtype)
            if weights_numpy.shape != (modes,):
                raise ValueError(
                    "basis_weights must have one value per basis mode"
                )
            if not np.isfinite(weights_numpy).all():
                raise ValueError(
                    "basis_weights must contain only finite values"
                )
        combined = np.tensordot(weights_numpy, basis_numpy, axes=(0, 0))
        if abs(float(combined.sum())) <= np.finfo(selected_dtype).eps:
            raise ValueError(
                "the weighted stamp basis must have nonzero flux"
            )

        resolved = resolve_backend(backend)
        self.image_shape = _image_shape(image_shape)
        self.evaluation = evaluation
        self.scale = float(scale)
        self.split_weights = _split_weights(split_weights)
        self.backend = resolved.name
        self.device = resolved.device
        self.dtype = selected_dtype
        self._ap = resolved.module
        self.stamp_basis = self._ap.asarray(basis_numpy, dtype=self.dtype)
        self.basis_weights = self._ap.asarray(weights_numpy, dtype=self.dtype)
        self._stamp = self._ap.asarray(combined, dtype=self.dtype)
        self._stamp_flux = float(combined.sum())

    def to_backend(
        self, backend: BackendRequest, *, dtype: npt.DTypeLike | None = None
    ) -> StampDipoleModel:
        """Return an equivalent model on ``backend``."""

        return StampDipoleModel(
            as_numpy(self.stamp_basis),
            image_shape=self.image_shape,
            evaluation=self.evaluation,
            scale=self.scale,
            basis_weights=as_numpy(self.basis_weights),
            split_weights=self.split_weights,
            backend=backend,
            dtype=self.dtype if dtype is None else dtype,
        )

    def _bilinear(
        self,
        x_center: BackendArray,
        y_center: BackendArray,
        flux: BackendArray,
    ) -> BackendArray:
        ap = self._ap
        height, width = self.image_shape
        stamp_height, stamp_width = self._stamp.shape
        x_grid = ap.arange(width, dtype=self.dtype) - (width - 1) / 2
        y_grid = ap.arange(height, dtype=self.dtype) - (height - 1) / 2
        x_source = (x_grid[None, :] - x_center[:, None]) / self.scale + (
            stamp_width - 1
        ) / 2
        y_source = (y_grid[None, :] - y_center[:, None]) / self.scale + (
            stamp_height - 1
        ) / 2
        x_low = ap.floor(x_source).astype(np.int64)
        y_low = ap.floor(y_source).astype(np.int64)
        x_fraction = x_source - x_low
        y_fraction = y_source - y_low
        output = ap.zeros(
            (x_center.shape[0], height, width), dtype=self.dtype
        )
        for y_offset, y_weight in (
            (0, 1.0 - y_fraction),
            (1, y_fraction),
        ):
            y_index = y_low + y_offset
            y_valid = (y_index >= 0) & (y_index < stamp_height)
            y_index = ap.clip(y_index, 0, stamp_height - 1)
            for x_offset, x_weight in (
                (0, 1.0 - x_fraction),
                (1, x_fraction),
            ):
                x_index = x_low + x_offset
                x_valid = (x_index >= 0) & (x_index < stamp_width)
                x_index = ap.clip(x_index, 0, stamp_width - 1)
                samples = self._stamp[
                    y_index[:, :, None], x_index[:, None, :]
                ]
                valid = y_valid[:, :, None] & x_valid[:, None, :]
                output += (
                    samples
                    * valid
                    * y_weight[:, :, None]
                    * x_weight[:, None, :]
                )
        normalization = self._stamp_flux * self.scale * self.scale
        return output * (flux / normalization)[:, None, None]

    def _finite_volume(
        self,
        x_center: BackendArray,
        y_center: BackendArray,
        flux: BackendArray,
    ) -> BackendArray:
        ap = self._ap
        height, width = self.image_shape
        stamp_height, stamp_width = self._stamp.shape
        output_x_edges = ap.arange(width + 1, dtype=self.dtype) - width / 2
        output_y_edges = ap.arange(height + 1, dtype=self.dtype) - height / 2
        output_x_edges = (
            output_x_edges[None, :] - x_center[:, None]
        ) / self.scale
        output_y_edges = (
            output_y_edges[None, :] - y_center[:, None]
        ) / self.scale
        stamp_x_edges = ap.arange(stamp_width + 1, dtype=self.dtype)
        stamp_x_edges -= stamp_width / 2
        stamp_y_edges = ap.arange(stamp_height + 1, dtype=self.dtype)
        stamp_y_edges -= stamp_height / 2
        x_overlap = ap.maximum(
            0,
            ap.minimum(
                stamp_x_edges[None, 1:, None],
                output_x_edges[:, None, 1:],
            )
            - ap.maximum(
                stamp_x_edges[None, :-1, None],
                output_x_edges[:, None, :-1],
            ),
        )
        y_overlap = ap.maximum(
            0,
            ap.minimum(
                output_y_edges[:, 1:, None],
                stamp_y_edges[None, None, 1:],
            )
            - ap.maximum(
                output_y_edges[:, :-1, None],
                stamp_y_edges[None, None, :-1],
            ),
        )
        output = ap.einsum(
            "byi,ij,bjx->byx", y_overlap, self._stamp, x_overlap
        )
        return output * (flux / self._stamp_flux)[:, None, None]

    def _star(
        self,
        x_center: BackendArray,
        y_center: BackendArray,
        flux: BackendArray,
    ) -> BackendArray:
        if self.evaluation in {"bilinear", "bilinear-vignetted"}:
            output = self._bilinear(x_center, y_center, flux)
            if self.evaluation == "bilinear-vignetted":
                ap = self._ap
                visible_flux = ap.sum(output, axis=(-2, -1))
                eps = np.finfo(self.dtype).eps
                factor = ap.where(
                    ap.abs(visible_flux) > eps,
                    flux / visible_flux,
                    0,
                )
                output *= factor[:, None, None]
            return output
        return self._finite_volume(x_center, y_center, flux)

    def evaluate(
        self, parameters: ArrayLike, *, mode: FitMode = "difference"
    ) -> BackendArray:
        """Evaluate a batch of sampled-stamp dipoles."""

        _validate_mode(mode)
        ap = self._ap
        parameters = ap.asarray(parameters, dtype=self.dtype)
        if parameters.ndim != 2 or parameters.shape[1] != 5:
            raise ValueError("stamp parameters must have shape (batch, 5)")
        positive = self._star(
            parameters[:, 0], parameters[:, 1], parameters[:, 4]
        )
        negative = self._star(
            parameters[:, 2], parameters[:, 3], parameters[:, 4]
        )
        difference = positive - negative
        if mode == "difference":
            return difference
        return ap.stack((difference, positive, negative), axis=1)


__all__ = ["GaussianDipoleModel", "StampDipoleModel"]
