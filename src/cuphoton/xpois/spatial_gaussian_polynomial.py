# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Spatially varying Gaussian-polynomial kernel fitting.

This module implements an instrument-neutral reference model for registered
image pairs.  The matching kernel is a linear combination of fixed two-
dimensional Gaussian-polynomial basis kernels whose coefficients vary over the
image.  A unit-sum basis term carries the photometric scale; zero-sum terms
describe kernel-shape changes independently of that scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Literal, Sequence

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .ois import (
    BasisTerm,
    GaussianBasisComponent,
    _coerce_variance,
    _default_fit_mask,
    _validate_image_pair,
    build_gaussian_polynomial_basis,
    triangular_degree_pairs,
)
from .spatial_als import _coerce_fit_mask, _validate_kernel_shape

_DESIGN_CHUNK_SIZE = 4096
# Relative Frobenius/column-norm floor below which a basis kernel or design
# column is treated as numerically null.  A term that cancels to rounding
# noise (for example a duplicate of the unit-sum reference after the
# flux-conserving rewrite) sits near ``eps`` times the cancelled magnitude,
# at most ~1e-13; genuinely distinct model terms sit many orders above.
_NULL_TERM_RELATIVE_NORM = 1.0e-10


@dataclass(frozen=True)
class SpatialKernelDomain:
    """Map array-local pixels into one parent pixel-coordinate domain.

    ``array_origin_yx`` is the parent pixel index of local pixel ``(0, 0)``.
    ``normalization_bbox`` is a half-open parent pixel-index bounding box in
    ``(y0, y1, x0, x1)`` order. Its first and last pixel centers map to -1
    and +1 for the Chebyshev fields; a singleton axis maps to zero, so every
    spatial degree must be 0 along it. The input array must lie inside this
    box.
    """

    array_origin_yx: tuple[int, int]
    normalization_bbox: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.array_origin_yx, tuple)
            or len(self.array_origin_yx) != 2
        ):
            raise ValueError("array_origin_yx must be a 2-tuple")
        if (
            not isinstance(self.normalization_bbox, tuple)
            or len(self.normalization_bbox) != 4
        ):
            raise ValueError("normalization_bbox must be a 4-tuple")
        for name, values in (
            ("array_origin_yx", self.array_origin_yx),
            ("normalization_bbox", self.normalization_bbox),
        ):
            if any(
                not isinstance(value, Integral) or isinstance(value, bool)
                for value in values
            ):
                raise ValueError(f"{name} must contain integers")
        y0, y1, x0, x1 = self.normalization_bbox
        if y1 <= y0 or x1 <= x0:
            raise ValueError("normalization_bbox must have positive extent")


@dataclass(frozen=True, eq=False)
class SpatialGaussianPolynomialKernelFitSamples:
    """Explicit local pixel rows and their optional relative precision.

    Positions use ``(y, x)`` order in the input-array coordinate frame.
    Duplicate pixels are rejected because repeated rows are not independent
    measurements. Use ``relative_precision`` to express intentional weighting.
    Rows are copied into canonical ``(y, x)`` order and made read-only.
    """

    positions_yx: np.ndarray
    relative_precision: np.ndarray | None = None

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions_yx)
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("positions_yx must have shape (row, 2)")
        if positions.shape[0] == 0:
            raise ValueError("positions_yx must not be empty")
        if not (
            np.issubdtype(positions.dtype, np.integer)
            or np.issubdtype(positions.dtype, np.floating)
        ):
            raise ValueError("positions_yx must contain numeric values")
        if not np.isfinite(positions).all():
            raise ValueError("positions_yx must not contain NaN or inf")
        if not np.equal(positions, np.floor(positions)).all():
            raise ValueError("positions_yx must contain integer values")
        integer_limits = np.iinfo(np.int64)
        if np.any(positions < integer_limits.min) or np.any(
            positions > integer_limits.max
        ):
            raise ValueError("positions_yx values exceed the supported range")
        positions = np.array(positions, dtype=np.int64, copy=True)
        sort_order = np.lexsort((positions[:, 1], positions[:, 0]))
        positions = positions[sort_order]
        if np.any(np.all(positions[1:] == positions[:-1], axis=1)):
            raise ValueError(
                "positions_yx must not contain duplicate pixels; aggregate "
                "their relative_precision"
            )
        positions.flags.writeable = False
        object.__setattr__(self, "positions_yx", positions)

        if self.relative_precision is None:
            return
        precision = np.asarray(self.relative_precision)
        if precision.shape != (positions.shape[0],):
            raise ValueError(
                "relative_precision must have one value per fit sample"
            )
        if not (
            np.issubdtype(precision.dtype, np.integer)
            or np.issubdtype(precision.dtype, np.floating)
        ):
            raise ValueError("relative_precision must contain numeric values")
        precision = np.array(
            precision[sort_order],
            dtype=np.float64,
            copy=True,
        )
        if not np.isfinite(precision).all():
            raise ValueError("relative_precision must be finite")
        if np.any(precision <= 0.0):
            raise ValueError("relative_precision must be strictly positive")
        precision.flags.writeable = False
        object.__setattr__(self, "relative_precision", precision)


@dataclass(frozen=True)
class _ResolvedFitSelection:
    sample_indices: np.ndarray
    relative_precision: np.ndarray | None
    explicit_samples: SpatialGaussianPolynomialKernelFitSamples | None
    mask: np.ndarray
    kind: Literal["all_valid", "mask", "samples"]
    requested_count: int
    excluded_requested_count: int


@dataclass(frozen=True)
class SpatialGaussianPolynomialKernelConfig:
    """Configuration for a spatial Gaussian-polynomial kernel fit.

    Parameters
    ----------
    shape_degree
        Maximum total Chebyshev degree for every zero-sum kernel-shape basis
        coefficient.
    photometric_degree
        Maximum total Chebyshev degree for the kernel sum, which is the
        reference-to-target photometric scale.
    background_degree
        Maximum total Chebyshev degree for the differential background.
    condition_limit
        Maximum allowed condition number of the column-scaled design matrix.
        Ill-conditioned fits fail instead of silently changing the model with
        regularization.
    """

    shape_degree: int = 2
    photometric_degree: int = 0
    background_degree: int = 0
    condition_limit: float = 1.0e12

    def __post_init__(self) -> None:
        for name in (
            "shape_degree",
            "photometric_degree",
            "background_degree",
        ):
            value = getattr(self, name)
            if not isinstance(value, Integral) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if not isinstance(self.condition_limit, Real) or isinstance(
            self.condition_limit,
            bool,
        ):
            raise ValueError("condition_limit must be a real number")
        if not np.isfinite(self.condition_limit):
            raise ValueError("condition_limit must be finite")
        if self.condition_limit <= 1.0:
            raise ValueError("condition_limit must be greater than one")


@dataclass
class SpatialGaussianPolynomialKernelFitResult:
    """Result of a spatial Gaussian-polynomial kernel fit.

    ``matched`` models ``target`` from ``source`` and ``residual`` is
    ``target - matched``. Invalid pixels and pixels without a complete source
    footprint are represented by NaN. The kernel is position-dependent; use
    :meth:`kernel_at_local` or :meth:`kernel_at_parent` to evaluate it without
    an implicit coordinate-frame choice. The local methods accept only points
    inside the fitted array; the parent methods accept any point inside
    ``spatial_domain.normalization_bbox`` and extrapolate the fitted fields
    beyond the array.

    ``fit_selection_kind`` records how fit rows were chosen: ``all_valid``
    (every valid kernel-interior pixel), ``mask`` (the caller's ``fit_mask``),
    or ``samples`` (explicit ``fit_samples``). ``requested_fit_pixel_count``
    is the interior pixel count, the mask true-count, or the sample row count
    respectively, and ``excluded_requested_fit_pixel_count`` is how many of
    those were dropped for invalid input (always zero for samples, which fail
    closed). ``fit_pixel_count`` is the number of fitted rows and
    ``unique_fit_pixel_count`` the number of distinct fitted pixels; they are
    equal by construction because duplicate rows are rejected, and the pair
    is kept for comparison with :class:`SpatialALSFitResult`, where repeated
    rows make them differ.

    ``condition_number`` describes the column-scaled design matrix.  The
    nominal ``dof`` does not account for model selection or rejected pixels.
    ``fit_objective`` is the residual sum of squares under ``fit_weighting``;
    it is not necessarily a statistically calibrated chi-square.
    When target variance is supplied, ``marginal_residual_variance`` and
    ``marginal_standardized_residual`` provide per-pixel noise diagnostics.
    Optional source variance is propagated through each realized local kernel,
    assuming independent input pixels; cross-pixel covariance and fit
    uncertainty are not represented.
    """

    matched: np.ndarray
    residual: np.ndarray
    fit_mask: np.ndarray
    background: np.ndarray
    photometric_scale: np.ndarray
    propagated_source_variance: np.ndarray | None
    marginal_residual_variance: np.ndarray | None
    marginal_standardized_residual: np.ndarray | None
    photometric_coefficients: np.ndarray
    shape_coefficients: np.ndarray
    background_coefficients: np.ndarray
    basis_kernels: np.ndarray
    basis_terms: tuple[BasisTerm, ...]
    photometric_terms: tuple[tuple[int, int], ...]
    shape_terms: tuple[tuple[int, int], ...]
    background_terms: tuple[tuple[int, int], ...]
    image_shape: tuple[int, int]
    spatial_domain: SpatialKernelDomain
    explicit_fit_samples: SpatialGaussianPolynomialKernelFitSamples | None
    fit_selection_kind: Literal["all_valid", "mask", "samples"]
    fit_weighting: Literal[
        "uniform",
        "target_variance",
        "relative_precision",
        "target_variance_relative_precision",
    ]
    fit_objective: float
    dof: int
    fit_pixel_count: int
    unique_fit_pixel_count: int
    requested_fit_pixel_count: int
    excluded_requested_fit_pixel_count: int
    condition_number: float
    source_variance_included: bool
    backend: str = "cpu"

    def photometric_scale_at_local(self, y: float, x: float) -> float:
        """Evaluate the kernel sum at local array coordinate ``y,x``."""

        _validate_local_coordinate(y, x, self.image_shape)
        return self._photometric_scale_at_local_unchecked(y, x)

    def photometric_scale_at_parent(self, y: float, x: float) -> float:
        """Evaluate the kernel sum at parent pixel coordinate ``y,x``.

        Any point inside ``spatial_domain.normalization_bbox`` is accepted;
        for a cutout this includes parent positions outside the fitted array,
        where the Chebyshev field is extrapolated.
        """

        _validate_parent_coordinate(y, x, self.spatial_domain)
        origin_y, origin_x = self.spatial_domain.array_origin_yx
        return self._photometric_scale_at_local_unchecked(
            y - origin_y,
            x - origin_x,
        )

    def kernel_at_local(self, y: float, x: float) -> np.ndarray:
        """Evaluate the matching kernel at local array coordinate ``y,x``."""

        _validate_local_coordinate(y, x, self.image_shape)
        return self._kernel_at_local_unchecked(y, x)

    def kernel_at_parent(self, y: float, x: float) -> np.ndarray:
        """Evaluate the matching kernel at parent pixel coordinate ``y,x``.

        Any point inside ``spatial_domain.normalization_bbox`` is accepted;
        for a cutout this includes parent positions outside the fitted array,
        where the Chebyshev fields are extrapolated.
        """

        _validate_parent_coordinate(y, x, self.spatial_domain)
        origin_y, origin_x = self.spatial_domain.array_origin_yx
        return self._kernel_at_local_unchecked(
            y - origin_y,
            x - origin_x,
        )

    def _photometric_scale_at_local_unchecked(
        self,
        y: float,
        x: float,
    ) -> float:
        values = _terms_at(
            y,
            x,
            self.spatial_domain,
            self.photometric_terms,
        )
        return float(self.photometric_coefficients @ values)

    def _kernel_at_local_unchecked(
        self,
        y: float,
        x: float,
    ) -> np.ndarray:
        photometric_scale = self._photometric_scale_at_local_unchecked(y, x)
        shape_values = _terms_at(
            y,
            x,
            self.spatial_domain,
            self.shape_terms,
        )
        shape_weights = self.shape_coefficients @ shape_values
        return photometric_scale * self.basis_kernels[0] + np.einsum(
            "i,iyx->yx",
            shape_weights,
            self.basis_kernels[1:],
            optimize=True,
        )


def solve_spatial_gaussian_polynomial_kernel(
    source: np.ndarray,
    target: np.ndarray,
    components: Sequence[GaussianBasisComponent],
    *,
    kernel_shape: tuple[int, int] = (15, 15),
    variance: np.ndarray | None = None,
    source_variance: np.ndarray | None = None,
    fit_mask: np.ndarray | None = None,
    fit_samples: SpatialGaussianPolynomialKernelFitSamples | None = None,
    spatial_domain: SpatialKernelDomain | None = None,
    config: SpatialGaussianPolynomialKernelConfig | None = None,
) -> SpatialGaussianPolynomialKernelFitResult:
    """Fit a spatial Gaussian-polynomial kernel from ``source`` to ``target``.

    The model at output pixel ``p`` is

    ``target_p ~= sum_q a_q(p) * (source (*) basis_q)_p + background_p``.

    The first basis kernel has unit sum and carries a separately configurable
    photometric-scale field.  Every remaining basis kernel has zero sum and
    carries a kernel-shape field.  Consequently, shape variation cannot
    silently change the photometric scale.

    Parameters
    ----------
    source, target
        Registered two-dimensional images with the same shape. ``source`` is
        convolved to model ``target``.
    components
        Gaussian widths and total polynomial degrees for the fixed 2-D kernel
        basis. Components that duplicate the unit-sum reference leave a
        basis kernel of rounding noise after the flux-conserving rewrite and
        are rejected rather than fitted.
    kernel_shape
        Odd ``(height, width)`` of every realized kernel.
    variance
        Optional positive target variance image. Omitting it requests uniform
        fit weights. This remains the fit objective's variance even when
        ``source_variance`` is supplied.
    source_variance
        Optional non-negative source variance image. It does not change the
        linear fit weights. When supplied with ``variance``, it is propagated
        through the fitted position-dependent kernel and added to the target
        variance for marginal residual-noise diagnostics. It does not describe
        or remove inter-pixel covariance.
    fit_mask
        Optional boolean image selecting fit pixels. Pixels without a complete
        finite source footprint are always excluded.
    fit_samples
        Optional explicit local ``(y, x)`` rows and relative precision.
        Specify this or ``fit_mask``, never both. Explicit rows fail closed if
        any lies outside the kernel interior or references invalid input.
        Duplicate pixels are rejected rather than treated as independent
        measurements.
    spatial_domain
        Optional mapping from array-local pixels to the parent coordinate box
        used to normalize every Chebyshev field. By default the input array is
        the full normalization domain.
    config
        Spatial degrees and numerical condition limit.
    """

    resolved_config = config or SpatialGaussianPolynomialKernelConfig()
    source_arr = np.asarray(source, dtype=np.float64)
    target_arr = np.asarray(target, dtype=np.float64)
    _validate_image_pair(source_arr, target_arr)
    kernel_height, kernel_width = _validate_kernel_shape(
        kernel_shape,
        source_arr.shape,
    )
    source_arr = np.ascontiguousarray(source_arr)
    target_arr = np.ascontiguousarray(target_arr)
    resolved_spatial_domain = _resolve_spatial_domain(
        source_arr.shape,
        spatial_domain,
    )
    if source_variance is not None and variance is None:
        raise ValueError("source_variance requires target variance")
    variance_arr = _coerce_variance(target_arr.shape, variance)
    source_variance_arr = _coerce_source_variance(
        source_arr.shape,
        source_variance,
    )
    boundary_mask = _default_fit_mask(
        target_arr.shape,
        (kernel_height, kernel_width),
    )
    invalid_mask = _invalid_input_mask(
        source_arr,
        target_arr,
        variance_arr,
        (kernel_height, kernel_width),
    )
    resolved_fit = _resolve_fit_samples(
        target_arr.shape,
        boundary_mask=boundary_mask,
        valid_mask=boundary_mask & ~invalid_mask,
        fit_mask=fit_mask,
        fit_samples=fit_samples,
    )
    fit_mask_arr = resolved_fit.mask

    source_model = np.where(np.isfinite(source_arr), source_arr, 0.0)
    target_model = np.where(np.isfinite(target_arr), target_arr, 0.0)
    basis_kernels, basis_terms = build_gaussian_polynomial_basis(
        (kernel_height, kernel_width),
        components,
        flux_conserve=True,
    )
    _validate_basis_kernels(basis_kernels)
    _validate_singleton_axes(
        resolved_spatial_domain,
        resolved_config,
        shape_basis_count=basis_kernels.shape[0] - 1,
    )
    photometric_terms = tuple(
        triangular_degree_pairs(resolved_config.photometric_degree)
    )
    shape_terms = tuple(triangular_degree_pairs(resolved_config.shape_degree))
    background_terms = tuple(
        triangular_degree_pairs(resolved_config.background_degree)
    )

    upper, transformed_rhs, row_count = _accumulate_qr(
        source_model,
        target_model,
        variance_arr,
        resolved_fit.sample_indices,
        resolved_fit.relative_precision,
        resolved_spatial_domain,
        basis_kernels,
        photometric_terms,
        shape_terms,
        background_terms,
    )
    if row_count < upper.shape[1]:
        raise ValueError(
            "fit region is underdetermined for the requested spatial model: "
            f"{row_count} equations for {upper.shape[1]} coefficients"
        )
    coefficients, condition_number = _solve_scaled_qr(
        upper,
        transformed_rhs,
        condition_limit=resolved_config.condition_limit,
    )

    photometric_count = len(photometric_terms)
    shape_count = (basis_kernels.shape[0] - 1) * len(shape_terms)
    photometric_coefficients = coefficients[:photometric_count]
    shape_coefficients = coefficients[
        photometric_count : photometric_count + shape_count
    ].reshape(basis_kernels.shape[0] - 1, len(shape_terms))
    background_coefficients = coefficients[photometric_count + shape_count :]

    photometric_scale = _evaluate_field(
        target_arr.shape,
        photometric_coefficients,
        photometric_terms,
        resolved_spatial_domain,
    )
    background = _evaluate_field(
        target_arr.shape,
        background_coefficients,
        background_terms,
        resolved_spatial_domain,
    )
    matched = _evaluate_matched_image(
        source_model,
        basis_kernels,
        photometric_coefficients,
        shape_coefficients,
        background_coefficients,
        photometric_terms,
        shape_terms,
        background_terms,
        resolved_spatial_domain,
        boundary_mask & ~invalid_mask,
    )
    residual = target_model - matched
    output_invalid_mask = invalid_mask | ~boundary_mask
    matched[output_invalid_mask] = np.nan
    residual[output_invalid_mask] = np.nan

    fit_precision = resolved_fit.relative_precision
    relative_precision_supplied = fit_precision is not None
    fit_objective = _evaluate_fit_objective(
        residual,
        variance_arr,
        resolved_fit.sample_indices,
        fit_precision,
    )
    if not np.isfinite(fit_objective):
        raise ValueError("fit objective is not finite")
    dof = int(row_count) - int(coefficients.size)
    propagated_source_variance: np.ndarray | None = None
    marginal_residual_variance: np.ndarray | None = None
    marginal_standardized_residual: np.ndarray | None = None
    if variance is not None:
        marginal_residual_variance = variance_arr.copy()
        if source_variance_arr is not None:
            source_variance_valid_mask = _finite_footprint_mask(
                source_variance_arr,
                (kernel_height, kernel_width),
            )
            diagnostic_valid_mask = (
                boundary_mask & ~invalid_mask & source_variance_valid_mask
            )
            propagated_source_variance = _propagate_source_variance(
                source_variance_arr,
                basis_kernels,
                photometric_coefficients,
                shape_coefficients,
                photometric_terms,
                shape_terms,
                resolved_spatial_domain,
                diagnostic_valid_mask,
            )
            diagnostic_valid_mask &= np.isfinite(propagated_source_variance)
            propagated_source_variance[~diagnostic_valid_mask] = np.nan
            marginal_residual_variance += propagated_source_variance
        marginal_residual_variance[
            output_invalid_mask
            | ~np.isfinite(marginal_residual_variance)
            | (marginal_residual_variance <= 0.0)
        ] = np.nan
        marginal_standardized_residual = residual / np.sqrt(
            marginal_residual_variance
        )
    return SpatialGaussianPolynomialKernelFitResult(
        matched=matched,
        residual=residual,
        fit_mask=fit_mask_arr,
        background=background,
        photometric_scale=photometric_scale,
        propagated_source_variance=propagated_source_variance,
        marginal_residual_variance=marginal_residual_variance,
        marginal_standardized_residual=marginal_standardized_residual,
        photometric_coefficients=photometric_coefficients,
        shape_coefficients=shape_coefficients,
        background_coefficients=background_coefficients,
        basis_kernels=basis_kernels,
        basis_terms=basis_terms,
        photometric_terms=photometric_terms,
        shape_terms=shape_terms,
        background_terms=background_terms,
        image_shape=target_arr.shape,
        spatial_domain=resolved_spatial_domain,
        explicit_fit_samples=resolved_fit.explicit_samples,
        fit_selection_kind=resolved_fit.kind,
        fit_weighting=_fit_weighting(
            target_variance_supplied=variance is not None,
            relative_precision_supplied=relative_precision_supplied,
        ),
        fit_objective=fit_objective,
        dof=dof,
        fit_pixel_count=int(row_count),
        unique_fit_pixel_count=int(fit_mask_arr.sum()),
        requested_fit_pixel_count=resolved_fit.requested_count,
        excluded_requested_fit_pixel_count=(
            resolved_fit.excluded_requested_count
        ),
        condition_number=condition_number,
        source_variance_included=source_variance_arr is not None,
    )


def _accumulate_qr(
    source: np.ndarray,
    target: np.ndarray,
    variance: np.ndarray,
    sample_indices: np.ndarray,
    relative_precision: np.ndarray | None,
    spatial_domain: SpatialKernelDomain,
    basis_kernels: np.ndarray,
    photometric_terms: tuple[tuple[int, int], ...],
    shape_terms: tuple[tuple[int, int], ...],
    background_terms: tuple[tuple[int, int], ...],
    *,
    chunk_size: int = _DESIGN_CHUNK_SIZE,
) -> tuple[np.ndarray, np.ndarray, int]:
    column_count = (
        len(photometric_terms)
        + (basis_kernels.shape[0] - 1) * len(shape_terms)
        + len(background_terms)
    )
    upper = np.empty((0, column_count), dtype=np.float64)
    transformed_rhs = np.empty(0, dtype=np.float64)
    patches, basis_flat, margin_y, margin_x = _patch_design_inputs(
        source,
        basis_kernels,
    )

    for start in range(0, sample_indices.size, chunk_size):
        stop = min(start + chunk_size, sample_indices.size)
        y_chunk, x_chunk = np.divmod(
            sample_indices[start:stop],
            source.shape[1],
        )
        patch_chunk = patches[
            y_chunk - margin_y,
            x_chunk - margin_x,
        ].reshape(stop - start, -1)
        basis_values = patch_chunk @ basis_flat.T
        design = _model_design(
            basis_values,
            y_chunk,
            x_chunk,
            spatial_domain,
            photometric_terms,
            shape_terms,
            background_terms,
        )
        chunk_precision = (
            None
            if relative_precision is None
            else relative_precision[start:stop]
        )
        row_scale = _fit_row_scale(
            variance[y_chunk, x_chunk],
            chunk_precision,
        )
        weighted_design = design * row_scale[:, None]
        weighted_target = target[y_chunk, x_chunk] * row_scale
        stacked_design = np.concatenate((upper, weighted_design), axis=0)
        stacked_target = np.concatenate(
            (transformed_rhs, weighted_target),
            axis=0,
        )
        orthogonal, upper = np.linalg.qr(stacked_design, mode="reduced")
        transformed_rhs = orthogonal.T @ stacked_target

    return upper, transformed_rhs, int(sample_indices.size)


def _evaluate_fit_objective(
    residual: np.ndarray,
    variance: np.ndarray,
    sample_indices: np.ndarray,
    relative_precision: np.ndarray | None,
    *,
    chunk_size: int = _DESIGN_CHUNK_SIZE,
) -> float:
    objective = 0.0
    for start in range(0, sample_indices.size, chunk_size):
        stop = min(start + chunk_size, sample_indices.size)
        sample_y, sample_x = np.divmod(
            sample_indices[start:stop],
            residual.shape[1],
        )
        chunk_precision = (
            None
            if relative_precision is None
            else relative_precision[start:stop]
        )
        row_scale = _fit_row_scale(
            variance[sample_y, sample_x],
            chunk_precision,
        )
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            weighted_residual = residual[sample_y, sample_x] * row_scale
            contribution = weighted_residual**2
        objective += float(np.sum(contribution))
    return objective


def _fit_row_scale(
    variance: np.ndarray,
    relative_precision: np.ndarray | None,
) -> np.ndarray:
    with np.errstate(
        over="ignore",
        under="ignore",
        divide="ignore",
        invalid="ignore",
    ):
        if relative_precision is None:
            row_scale = 1.0 / np.sqrt(variance)
        else:
            row_scale = np.sqrt(relative_precision) / np.sqrt(variance)
    if np.any(~np.isfinite(row_scale)) or np.any(row_scale <= 0.0):
        raise ValueError(
            "combined fit row scales must be finite and positive"
        )
    return row_scale


def _evaluate_matched_image(
    source: np.ndarray,
    basis_kernels: np.ndarray,
    photometric_coefficients: np.ndarray,
    shape_coefficients: np.ndarray,
    background_coefficients: np.ndarray,
    photometric_terms: tuple[tuple[int, int], ...],
    shape_terms: tuple[tuple[int, int], ...],
    background_terms: tuple[tuple[int, int], ...],
    spatial_domain: SpatialKernelDomain,
    valid_mask: np.ndarray,
    *,
    chunk_size: int = _DESIGN_CHUNK_SIZE,
) -> np.ndarray:
    matched = np.full(source.shape, np.nan, dtype=np.float64)
    coefficients = np.concatenate(
        (
            photometric_coefficients,
            shape_coefficients.ravel(),
            background_coefficients,
        )
    )
    patches, basis_flat, margin_y, margin_x = _patch_design_inputs(
        source,
        basis_kernels,
    )
    ys, xs = np.nonzero(valid_mask)
    for start in range(0, ys.size, chunk_size):
        stop = min(start + chunk_size, ys.size)
        y_chunk = ys[start:stop]
        x_chunk = xs[start:stop]
        patch_chunk = patches[
            y_chunk - margin_y,
            x_chunk - margin_x,
        ].reshape(stop - start, -1)
        basis_values = patch_chunk @ basis_flat.T
        design = _model_design(
            basis_values,
            y_chunk,
            x_chunk,
            spatial_domain,
            photometric_terms,
            shape_terms,
            background_terms,
        )
        matched[y_chunk, x_chunk] = design @ coefficients
    return matched


def _propagate_source_variance(
    source_variance: np.ndarray,
    basis_kernels: np.ndarray,
    photometric_coefficients: np.ndarray,
    shape_coefficients: np.ndarray,
    photometric_terms: tuple[tuple[int, int], ...],
    shape_terms: tuple[tuple[int, int], ...],
    spatial_domain: SpatialKernelDomain,
    valid_mask: np.ndarray,
    *,
    chunk_size: int = _DESIGN_CHUNK_SIZE,
) -> np.ndarray:
    propagated = np.zeros(source_variance.shape, dtype=np.float64)
    kernel_height, kernel_width = basis_kernels.shape[1:]
    margin_y = kernel_height // 2
    margin_x = kernel_width // 2
    variance_patches = sliding_window_view(
        np.where(np.isfinite(source_variance), source_variance, 0.0),
        (kernel_height, kernel_width),
    )
    ys, xs = np.nonzero(valid_mask)
    for start in range(0, ys.size, chunk_size):
        stop = min(start + chunk_size, ys.size)
        y_chunk = ys[start:stop]
        x_chunk = xs[start:stop]
        photometric_values = _chebyshev_design(
            y_chunk,
            x_chunk,
            spatial_domain,
            photometric_terms,
        )
        shape_values = _chebyshev_design(
            y_chunk,
            x_chunk,
            spatial_domain,
            shape_terms,
        )
        photometric_scale = photometric_values @ photometric_coefficients
        shape_weights = shape_values @ shape_coefficients.T
        kernels = photometric_scale[:, None, None] * basis_kernels[
            0
        ] + np.einsum(
            "ri,iyx->ryx",
            shape_weights,
            basis_kernels[1:],
            optimize=True,
        )
        patch_chunk = variance_patches[
            y_chunk - margin_y,
            x_chunk - margin_x,
        ]
        propagated[y_chunk, x_chunk] = np.einsum(
            "rij,rij->r",
            patch_chunk,
            kernels[:, ::-1, ::-1] ** 2,
            optimize=True,
        )
    return propagated


def _patch_design_inputs(
    source: np.ndarray,
    basis_kernels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    kernel_height, kernel_width = basis_kernels.shape[1:]
    patches = sliding_window_view(source, (kernel_height, kernel_width))
    basis_flat = basis_kernels[:, ::-1, ::-1].reshape(
        basis_kernels.shape[0],
        -1,
    )
    return patches, basis_flat, kernel_height // 2, kernel_width // 2


def _model_design(
    basis_values: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    spatial_domain: SpatialKernelDomain,
    photometric_terms: tuple[tuple[int, int], ...],
    shape_terms: tuple[tuple[int, int], ...],
    background_terms: tuple[tuple[int, int], ...],
) -> np.ndarray:
    photometric_values = _chebyshev_design(
        ys,
        xs,
        spatial_domain,
        photometric_terms,
    )
    shape_values = _chebyshev_design(
        ys,
        xs,
        spatial_domain,
        shape_terms,
    )
    background_values = _chebyshev_design(
        ys,
        xs,
        spatial_domain,
        background_terms,
    )
    photometric_design = basis_values[:, :1] * photometric_values
    shape_design = (
        basis_values[:, 1:, None] * shape_values[:, None, :]
    ).reshape(basis_values.shape[0], -1)
    return np.concatenate(
        (photometric_design, shape_design, background_values),
        axis=1,
    )


def _solve_scaled_qr(
    upper: np.ndarray,
    transformed_rhs: np.ndarray,
    *,
    condition_limit: float,
) -> tuple[np.ndarray, float]:
    scales = np.linalg.norm(upper, axis=0)
    if np.any(~np.isfinite(scales)):
        raise ValueError(
            "spatial Gaussian-polynomial design has non-finite column "
            "norms; check the variance and relative_precision scaling"
        )
    if np.any(scales <= _NULL_TERM_RELATIVE_NORM * scales.max()):
        raise ValueError(
            "spatial Gaussian-polynomial design contains an empty or "
            "numerically null column"
        )
    scaled_upper = upper / scales[None, :]
    try:
        condition_number = float(np.linalg.cond(scaled_upper))
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "spatial Gaussian-polynomial design matrix is singular"
        ) from exc
    if (
        not np.isfinite(condition_number)
        or condition_number > condition_limit
    ):
        raise ValueError(
            "spatial Gaussian-polynomial design matrix is ill-conditioned: "
            f"condition {condition_number:.6g} exceeds {condition_limit:.6g}"
        )
    try:
        scaled_coefficients = np.linalg.solve(
            scaled_upper,
            transformed_rhs,
        )
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "spatial Gaussian-polynomial design matrix could not be solved"
        ) from exc
    return scaled_coefficients / scales, condition_number


def _validate_basis_kernels(basis_kernels: np.ndarray) -> None:
    """Fail closed on basis kernels that cancelled to rounding noise.

    ``build_gaussian_polynomial_basis`` rewrites every non-zero-sum term as
    ``kernel - kernel_sum * reference``.  A term proportional to the
    reference (a duplicated component) leaves rounding residue rather than
    an exact zero; the column-scaled condition number cannot see such a
    column, so it is rejected here by its relative Frobenius norm.
    """

    norms = np.linalg.norm(
        basis_kernels.reshape(basis_kernels.shape[0], -1),
        axis=1,
    )
    null = np.flatnonzero(norms <= _NULL_TERM_RELATIVE_NORM * norms.max())
    if null.size:
        raise ValueError(
            f"basis kernel {int(null[0])} is numerically zero after the "
            "flux-conserving rewrite; remove duplicate components"
        )


def _validate_singleton_axes(
    spatial_domain: SpatialKernelDomain,
    config: SpatialGaussianPolynomialKernelConfig,
    *,
    shape_basis_count: int,
) -> None:
    """Reject spatial degrees along a normalization axis of extent one.

    A singleton axis maps to Chebyshev coordinate zero, so odd-degree terms
    vanish and even-degree terms duplicate the constant term; without this
    check the solve fails with an empty or collinear column that blames the
    basis rather than the domain.
    """

    degrees = [config.photometric_degree, config.background_degree]
    if shape_basis_count > 0:
        degrees.append(config.shape_degree)
    if max(degrees) == 0:
        return
    y0, y1, x0, x1 = spatial_domain.normalization_bbox
    for name, lower, upper in (("y", y0, y1), ("x", x0, x1)):
        if upper - lower == 1:
            raise ValueError(
                f"normalization_bbox has extent 1 along {name}; every "
                "spatial degree must be 0 on a singleton axis"
            )


def _evaluate_field(
    image_shape: tuple[int, int],
    coefficients: np.ndarray,
    terms: tuple[tuple[int, int], ...],
    spatial_domain: SpatialKernelDomain,
) -> np.ndarray:
    max_x_degree = max(degree_x for degree_x, _ in terms)
    max_y_degree = max(degree_y for _, degree_y in terms)
    origin_y, origin_x = spatial_domain.array_origin_yx
    y0, y1, x0, x1 = spatial_domain.normalization_bbox
    x_values = np.polynomial.chebyshev.chebvander(
        _normalized_coordinate(
            np.arange(image_shape[1], dtype=np.float64),
            origin_x,
            x0,
            x1,
        ),
        max_x_degree,
    )
    y_values = np.polynomial.chebyshev.chebvander(
        _normalized_coordinate(
            np.arange(image_shape[0], dtype=np.float64),
            origin_y,
            y0,
            y1,
        ),
        max_y_degree,
    )
    result = np.zeros(image_shape, dtype=np.float64)
    for coefficient, (degree_x, degree_y) in zip(
        coefficients,
        terms,
        strict=True,
    ):
        result += coefficient * np.multiply.outer(
            y_values[:, degree_y],
            x_values[:, degree_x],
        )
    return result


def _terms_at(
    y: float,
    x: float,
    spatial_domain: SpatialKernelDomain,
    terms: tuple[tuple[int, int], ...],
) -> np.ndarray:
    return _chebyshev_design(
        np.asarray([y], dtype=np.float64),
        np.asarray([x], dtype=np.float64),
        spatial_domain,
        terms,
    )[0]


def _chebyshev_design(
    ys: np.ndarray,
    xs: np.ndarray,
    spatial_domain: SpatialKernelDomain,
    terms: tuple[tuple[int, int], ...],
) -> np.ndarray:
    max_x_degree = max(degree_x for degree_x, _ in terms)
    max_y_degree = max(degree_y for _, degree_y in terms)
    origin_y, origin_x = spatial_domain.array_origin_yx
    y0, y1, x0, x1 = spatial_domain.normalization_bbox
    x_values = np.polynomial.chebyshev.chebvander(
        _normalized_coordinate(xs, origin_x, x0, x1),
        max_x_degree,
    )
    y_values = np.polynomial.chebyshev.chebvander(
        _normalized_coordinate(ys, origin_y, y0, y1),
        max_y_degree,
    )
    return np.stack(
        [
            x_values[:, degree_x] * y_values[:, degree_y]
            for degree_x, degree_y in terms
        ],
        axis=1,
    )


def _normalized_coordinate(
    values: np.ndarray,
    array_origin: int,
    lower: int,
    upper: int,
) -> np.ndarray:
    values_arr = np.asarray(values, dtype=np.float64)
    length = upper - lower
    if length == 1:
        return np.zeros_like(values_arr)
    parent_values = values_arr + array_origin
    return (2.0 * (parent_values - lower) / float(length - 1)) - 1.0


def _resolve_spatial_domain(
    image_shape: tuple[int, int],
    spatial_domain: SpatialKernelDomain | None,
) -> SpatialKernelDomain:
    if spatial_domain is None:
        return SpatialKernelDomain(
            array_origin_yx=(0, 0),
            normalization_bbox=(0, image_shape[0], 0, image_shape[1]),
        )
    origin_y, origin_x = spatial_domain.array_origin_yx
    y0, y1, x0, x1 = spatial_domain.normalization_bbox
    if (
        origin_y < y0
        or origin_x < x0
        or origin_y + image_shape[0] > y1
        or origin_x + image_shape[1] > x1
    ):
        raise ValueError(
            "input array extent must lie inside normalization_bbox"
        )
    return spatial_domain


def _validate_local_coordinate(
    y: float,
    x: float,
    image_shape: tuple[int, int],
) -> None:
    for name, value, length in (
        ("y", y, image_shape[0]),
        ("x", x, image_shape[1]),
    ):
        if (
            not isinstance(value, Real)
            or isinstance(value, bool)
            or not np.isfinite(value)
            or value < 0.0
            or value > length - 1
        ):
            raise ValueError(
                f"{name} must be finite and inside the local array"
            )


def _validate_parent_coordinate(
    y: float,
    x: float,
    spatial_domain: SpatialKernelDomain,
) -> None:
    y0, y1, x0, x1 = spatial_domain.normalization_bbox
    for name, value, lower, upper in (
        ("y", y, y0, y1),
        ("x", x, x0, x1),
    ):
        if (
            not isinstance(value, Real)
            or isinstance(value, bool)
            or not np.isfinite(value)
            or value < lower
            or value > upper - 1
        ):
            raise ValueError(
                f"{name} must be finite and inside the parent pixel domain"
            )


def _coerce_source_variance(
    image_shape: tuple[int, int],
    source_variance: np.ndarray | None,
) -> np.ndarray | None:
    if source_variance is None:
        return None
    variance_arr = np.asarray(source_variance, dtype=np.float64)
    if variance_arr.shape != image_shape:
        raise ValueError("source_variance must match the image shape")
    if np.any(np.isfinite(variance_arr) & (variance_arr < 0.0)):
        raise ValueError("source_variance must be non-negative")
    return variance_arr


def _resolve_fit_samples(
    image_shape: tuple[int, int],
    *,
    boundary_mask: np.ndarray,
    valid_mask: np.ndarray,
    fit_mask: np.ndarray | None,
    fit_samples: SpatialGaussianPolynomialKernelFitSamples | None,
) -> _ResolvedFitSelection:
    if fit_mask is not None and fit_samples is not None:
        raise ValueError("specify either fit_mask or fit_samples, not both")

    if fit_samples is not None:
        if not isinstance(
            fit_samples, SpatialGaussianPolynomialKernelFitSamples
        ):
            raise TypeError(
                "fit_samples must be a "
                "SpatialGaussianPolynomialKernelFitSamples instance"
            )
        positions = fit_samples.positions_yx
        sample_y = positions[:, 0]
        sample_x = positions[:, 1]
        height, width = image_shape
        if np.any(
            (sample_y < 0)
            | (sample_y >= height)
            | (sample_x < 0)
            | (sample_x >= width)
        ):
            raise ValueError(
                "fit_samples must lie inside the valid kernel interior"
            )
        if not np.all(boundary_mask[sample_y, sample_x]):
            raise ValueError(
                "fit_samples must lie inside the valid kernel interior"
            )
        if not np.all(valid_mask[sample_y, sample_x]):
            raise ValueError(
                "fit_samples must reference finite target and variance "
                "values with finite source footprints"
            )
        result_mask = np.zeros(image_shape, dtype=bool)
        result_mask[sample_y, sample_x] = True
        return _ResolvedFitSelection(
            sample_indices=sample_y * width + sample_x,
            relative_precision=fit_samples.relative_precision,
            explicit_samples=fit_samples,
            mask=result_mask,
            kind="samples",
            requested_count=int(positions.shape[0]),
            excluded_requested_count=0,
        )

    if fit_mask is None:
        requested_mask = boundary_mask
        kind: Literal["all_valid", "mask", "samples"] = "all_valid"
    else:
        requested_mask = _coerce_fit_mask(fit_mask, image_shape)
        kind = "mask"
    result_mask = requested_mask & valid_mask
    if not np.any(result_mask):
        raise ValueError("no finite pixels remain within the fit region")
    return _ResolvedFitSelection(
        sample_indices=np.flatnonzero(result_mask),
        relative_precision=None,
        explicit_samples=None,
        mask=result_mask,
        kind=kind,
        requested_count=int(requested_mask.sum()),
        excluded_requested_count=int(
            requested_mask.sum() - result_mask.sum()
        ),
    )


def _fit_weighting(
    *,
    target_variance_supplied: bool,
    relative_precision_supplied: bool,
) -> Literal[
    "uniform",
    "target_variance",
    "relative_precision",
    "target_variance_relative_precision",
]:
    if target_variance_supplied and relative_precision_supplied:
        return "target_variance_relative_precision"
    if target_variance_supplied:
        return "target_variance"
    if relative_precision_supplied:
        return "relative_precision"
    return "uniform"


def _invalid_input_mask(
    source: np.ndarray,
    target: np.ndarray,
    variance: np.ndarray,
    kernel_shape: tuple[int, int],
) -> np.ndarray:
    center_invalid = ~(np.isfinite(target) & np.isfinite(variance))
    return center_invalid | _invalid_footprint_mask(source, kernel_shape)


def _finite_footprint_mask(
    image: np.ndarray,
    kernel_shape: tuple[int, int],
) -> np.ndarray:
    return ~_invalid_footprint_mask(image, kernel_shape)


def _invalid_footprint_mask(
    image: np.ndarray,
    kernel_shape: tuple[int, int],
) -> np.ndarray:
    """Flag pixels whose centered ``kernel_shape`` footprint is not finite.

    Pixels beyond the image edge count as finite, matching a zero-padded
    convolution.  A boolean maximum filter is exact for any footprint size,
    whereas int16 counting wraps for footprints of 32768 pixels or more.
    """

    from scipy.ndimage import maximum_filter

    invalid = ~np.isfinite(image)
    if not np.any(invalid):
        return invalid
    return maximum_filter(invalid, size=kernel_shape, mode="constant", cval=0)


__all__ = [
    "SpatialKernelDomain",
    "SpatialGaussianPolynomialKernelFitSamples",
    "SpatialGaussianPolynomialKernelConfig",
    "SpatialGaussianPolynomialKernelFitResult",
    "solve_spatial_gaussian_polynomial_kernel",
]
