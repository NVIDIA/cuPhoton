# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Spatially varying separable-kernel alternating least squares.

The solver in this module is camera-neutral. It consumes registered image
arrays and an optional fit region, builds separable Gaussian-polynomial line
bases, and fits Chebyshev coefficient fields for both line profiles. Survey
data preparation, camera calibration, source selection, and PSF measurement
remain caller or adapter responsibilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Sequence

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .ois import (
    GaussianBasisComponent,
    _build_line_basis,
    _coerce_boolean_mask,
    _coerce_variance,
    _validate_image_pair,
    triangular_degree_pairs,
)

_DESIGN_CHUNK_SIZE = 4096
_CONDITION_LIMIT = 1.0e14


@dataclass(frozen=True)
class SpatialALSConfig:
    """Numerical configuration for a spatial ALS fit.

    Parameters
    ----------
    spatial_degree
        Maximum total Chebyshev degree for each line-profile coefficient.
    background_degree
        Maximum total Chebyshev degree for the differential background.
    max_iterations
        Maximum number of complete vertical and horizontal block updates.
        Noisy flux-conserving fits typically need 10 to 25 sweeps to reach
        the default tolerance.
    tolerance
        Relative penalized-objective change used for convergence.
    regularization
        Non-negative ridge penalty applied to spatial line coefficients. The
        global photometric scale and background are not regularized. The
        penalty is added to the normal-equation diagonal in weighted design
        units, so it is not invariant to rescaling the images.
    flux_conserve
        Make all spatial line-basis corrections zero-sum. This keeps the
        signed kernel sum position-independent while fitting one global scale.
        Without it, ``flux_scale`` remains the vertical reference multiplier
        but is not generally the kernel sum. Near dependence between reference
        and correction profiles can cause poor conditioning or slow ALS
        convergence. Prefer ``True`` unless a position-dependent kernel sum
        is required.
    """

    spatial_degree: int = 2
    background_degree: int = 0
    max_iterations: int = 30
    tolerance: float = 1.0e-6
    regularization: float = 1.0e-6
    flux_conserve: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.spatial_degree, Integral) or isinstance(
            self.spatial_degree,
            bool,
        ):
            raise ValueError("spatial_degree must be an integer")
        if self.spatial_degree < 0:
            raise ValueError("spatial_degree must be non-negative")
        if not isinstance(self.background_degree, Integral) or isinstance(
            self.background_degree,
            bool,
        ):
            raise ValueError("background_degree must be an integer")
        if self.background_degree < 0:
            raise ValueError("background_degree must be non-negative")
        if not isinstance(self.max_iterations, Integral) or isinstance(
            self.max_iterations,
            bool,
        ):
            raise ValueError("max_iterations must be an integer")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if not isinstance(self.tolerance, Real) or isinstance(
            self.tolerance,
            bool,
        ):
            raise ValueError("tolerance must be a real number")
        if not np.isfinite(self.tolerance):
            raise ValueError("tolerance must be finite")
        if self.tolerance < 0.0:
            raise ValueError("tolerance must be non-negative")
        if not isinstance(self.regularization, Real) or isinstance(
            self.regularization,
            bool,
        ):
            raise ValueError("regularization must be a real number")
        if not np.isfinite(self.regularization):
            raise ValueError("regularization must be finite")
        if self.regularization < 0.0:
            raise ValueError("regularization must be non-negative")
        if not isinstance(self.flux_conserve, (bool, np.bool_)):
            raise ValueError("flux_conserve must be boolean")


@dataclass
class SpatialALSFitResult:
    """Result of a spatially varying separable-kernel ALS fit.

    ``matched`` models the target from the reference image. ``residual`` is
    always ``target - matched`` wherever the centered kernel footprint and
    inputs are valid. Edge and invalid-input pixels are represented by NaN.
    The realized kernel is position-dependent; use :meth:`kernel_at` rather
    than interpreting either fitted line profile as one global kernel.

    Attributes
    ----------
    matched
        Reference image convolved with the position-dependent kernel plus
        the fitted background, with target shape and pixel units.
    residual
        ``target - matched`` with the target shape and pixel units.
    fit_mask
        Boolean array with the target shape; true pixels entered the solve.
        Repeated sample positions mark a single pixel.
    background
        Differential-background image with target shape and pixel units.
    horizontal_reference, vertical_reference
        Unit-sum reference line profiles with lengths ``kernel_x`` and
        ``kernel_y``; they are not rows of the basis matrices.
    horizontal_basis, vertical_basis
        Correction profiles with shape ``(terms, kernel_x)`` and
        ``(terms, kernel_y)``: the remaining Gaussian-polynomial line
        functions, made zero-sum when ``flux_conserve`` is true and scaled
        to unit L2 norm. Unlike :class:`SeparableKernelFitResult` they
        exclude the reference profile.
    horizontal_coefficients, vertical_coefficients
        Chebyshev coefficient fields with shape ``(terms, spatial_terms)``;
        row ``i`` weights basis row ``i`` and column ``s`` multiplies the
        polynomial named by ``spatial_terms[s]``.
    background_coefficients
        One coefficient per ``background_terms`` entry.
    flux_scale
        Multiplier applied to ``vertical_reference`` in :meth:`vertical_at`;
        :meth:`horizontal_at` carries no such factor. It is the
        position-independent signed kernel sum only when ``flux_conserve``
        is true. Otherwise it is a basis coefficient, not a standalone
        photometric scale; evaluate :meth:`kernel_at` for the local kernel
        and its sum.
    spatial_terms, background_terms
        ``(x_degree, y_degree)`` Chebyshev degree pairs in coefficient
        order.
    image_shape
        Shape of the fitted images. Chebyshev coordinates map its zero-based
        ``(y, x)`` axes onto ``[-1, 1]``, and :meth:`kernel_at` accepts only
        coordinates inside it.
    objective_history
        Penalized objective ``chi2 + regularization * (|H|^2 + |V|^2)``
        after each completed sweep, with ``H`` and ``V`` the two coefficient
        fields; it is not the chi-square alone.
    chi2, dof, fit_pixel_count, unique_fit_pixel_count
        Weighted residual sum over the fitted rows, nominal ``N - p`` with
        ``N`` the row count, the row count including repeated sample
        positions, and the number of distinct fitted pixels. ``dof`` is not
        the ridge-effective degrees of freedom. The chi-square units are
        residual-units squared per variance-unit and are dimensionless when
        variance is supplied in squared target units.
    iterations, converged
        Completed sweeps, and whether the objective met the tolerance or
        the numerical floor before the sweep budget ran out.
    regularization, flux_conserve
        Echoes of the configured ridge penalty and basis rewrite.
    condition_number
        Larger condition number of the two diagonally scaled block normal
        matrices from the final sweep.
    backend
        Backend that ran the fit.
    """

    matched: np.ndarray
    residual: np.ndarray
    fit_mask: np.ndarray
    background: np.ndarray
    horizontal_reference: np.ndarray
    vertical_reference: np.ndarray
    horizontal_basis: np.ndarray
    vertical_basis: np.ndarray
    horizontal_coefficients: np.ndarray
    vertical_coefficients: np.ndarray
    background_coefficients: np.ndarray
    flux_scale: float
    spatial_terms: tuple[tuple[int, int], ...]
    background_terms: tuple[tuple[int, int], ...]
    image_shape: tuple[int, int]
    objective_history: np.ndarray
    chi2: float
    dof: int
    fit_pixel_count: int
    unique_fit_pixel_count: int
    iterations: int
    converged: bool
    regularization: float
    flux_conserve: bool
    condition_number: float
    backend: str = "cpu"

    def horizontal_at(self, y: float, x: float) -> np.ndarray:
        """Evaluate the horizontal profile at image coordinate ``y,x``."""

        terms = _terms_at(
            y,
            x,
            self.image_shape,
            self.spatial_terms,
        )
        return self.horizontal_reference + np.einsum(
            "is,s,iu->u",
            self.horizontal_coefficients,
            terms,
            self.horizontal_basis,
            optimize=True,
        )

    def vertical_at(self, y: float, x: float) -> np.ndarray:
        """Evaluate the vertical line profile at image coordinate ``y,x``."""

        terms = _terms_at(
            y,
            x,
            self.image_shape,
            self.spatial_terms,
        )
        return self.flux_scale * self.vertical_reference + np.einsum(
            "is,s,iv->v",
            self.vertical_coefficients,
            terms,
            self.vertical_basis,
            optimize=True,
        )

    def kernel_at(self, y: float, x: float) -> np.ndarray:
        """Evaluate the two-dimensional convolution kernel at ``y,x``."""

        return np.outer(
            self.vertical_at(y, x),
            self.horizontal_at(y, x),
        )


def solve_spatial_als(
    reference: np.ndarray,
    target: np.ndarray,
    components: Sequence[GaussianBasisComponent],
    *,
    kernel_shape: tuple[int, int] = (15, 15),
    variance: np.ndarray | None = None,
    fit_mask: np.ndarray | None = None,
    sample_positions: np.ndarray | None = None,
    config: SpatialALSConfig | None = None,
) -> SpatialALSFitResult:
    """Fit a spatially varying separable kernel from reference to target.

    Parameters
    ----------
    reference, target
        Registered two-dimensional images with the same shape. ``reference``
        is convolved to model ``target``.
    components
        Gaussian widths and polynomial degrees used to build the horizontal
        and vertical line bases.
    kernel_shape
        Odd ``(height, width)`` of every realized kernel.
    variance
        Optional positive target variance image. Omitting it requests uniform
        weights; the solver never derives hidden weights from image values.
    fit_mask
        Optional boolean image selecting fit pixels.
    sample_positions
        Optional integer ``(row, 2)`` array of ``(y, x)`` fit positions.
        Repeated positions are retained as multiplicity weights. Specify this
        or ``fit_mask``, never both. Fit row counts and degrees of freedom
        include repeated positions. If neither selector is supplied, all valid
        pixels in the centered-kernel interior are fitted.
    config
        Spatial model, convergence, regularization, and flux configuration.

    Returns
    -------
    SpatialALSFitResult
        Position-dependent kernel factors, matched and residual images, and
        numerical diagnostics.

    Notes
    -----
    At output position ``p``, the model kernel is

    ``K_p(v, u) = V_p(v) H_p(u)``.

    Both line profiles have independent Chebyshev coefficient fields, and
    alternating block solves retain both fields during reconstruction.
    """

    resolved_config = config or SpatialALSConfig()
    source_arr = np.asarray(reference, dtype=np.float64)
    target_arr = np.asarray(target, dtype=np.float64)
    _validate_image_pair(source_arr, target_arr)
    # Workflow cutouts can be non-contiguous slice views. Materialize each
    # image once so flattening inside repeated ALS chunks never copies a full
    # crop per chunk.
    source_arr = np.ascontiguousarray(source_arr)
    target_arr = np.ascontiguousarray(target_arr)
    kernel_height, kernel_width = _validate_kernel_shape(
        kernel_shape,
        source_arr.shape,
    )
    variance_arr = np.ascontiguousarray(
        _coerce_variance(target_arr.shape, variance)
    )
    target_flat = target_arr.ravel()
    variance_flat = variance_arr.ravel()

    raw_horizontal = _build_line_basis(kernel_width, components)
    raw_vertical = _build_line_basis(kernel_height, components)
    _validate_line_basis_length(raw_horizontal, kernel_width, axis="width")
    _validate_line_basis_length(raw_vertical, kernel_height, axis="height")
    horizontal_reference, horizontal_basis = _prepare_line_basis(
        raw_horizontal,
        flux_conserve=resolved_config.flux_conserve,
    )
    vertical_reference, vertical_basis = _prepare_line_basis(
        raw_vertical,
        flux_conserve=resolved_config.flux_conserve,
    )

    spatial_terms = tuple(
        triangular_degree_pairs(resolved_config.spatial_degree)
    )
    background_terms = tuple(
        triangular_degree_pairs(resolved_config.background_degree)
    )
    sample_indices, effective_fit_mask = _resolve_fit_samples(
        source_arr,
        target_arr,
        variance_arr,
        (kernel_height, kernel_width),
        fit_mask=fit_mask,
        sample_positions=sample_positions,
    )
    fit_sample_count = int(sample_indices.size)
    unique_fit_pixel_count = int(effective_fit_mask.sum())
    parameter_count = (
        horizontal_basis.shape[0] * len(spatial_terms)
        + vertical_basis.shape[0] * len(spatial_terms)
        + 1
        + len(background_terms)
    )
    # Repeated sample rows add weight but no rank, so the guard must count
    # distinct pixels; otherwise the ridge turns a rank-deficient system into
    # an exact interpolation reported with a large nominal dof.
    if unique_fit_pixel_count < parameter_count:
        raise ValueError(
            "fit region is underdetermined for spatial ALS: "
            f"{unique_fit_pixel_count} unique pixels ({fit_sample_count} "
            f"sample rows) for {parameter_count} coefficients"
        )

    patch_view = sliding_window_view(
        source_arr,
        (kernel_height, kernel_width),
    )
    # Every prediction accumulates one product per kernel pixel plus the
    # background terms. Allow rounding proportional to that length before
    # squaring for the objective. Scale only with weighted data energy: a
    # sample-count baseline would make convergence depend on image units.
    accumulation_length = float(
        kernel_height * kernel_width + len(background_terms)
    )
    base_objective_floor = (
        np.finfo(np.float64).eps ** 2
        * accumulation_length**2
        * _weighted_target_energy(
            target_flat,
            variance_flat,
            sample_indices,
        )
    )

    horizontal_coefficients = np.zeros(
        (horizontal_basis.shape[0], len(spatial_terms)),
        dtype=np.float64,
    )
    vertical_coefficients = np.zeros(
        (vertical_basis.shape[0], len(spatial_terms)),
        dtype=np.float64,
    )
    background_coefficients = np.zeros(
        len(background_terms),
        dtype=np.float64,
    )
    flux_scale = 1.0
    objective_values: list[float] = []
    converged = False
    condition_number = 0.0

    for _ in range(resolved_config.max_iterations):
        # The vertical block carries the global flux scale. Solving it first
        # fits the amplitude before any profile shape is adjusted; starting
        # with the horizontal block instead fits shapes against a model whose
        # amplitude is still 1, which with zero-sum corrections distorts the
        # profiles and costs many sweeps to unwind.
        b_coefficients, b_condition = _solve_vertical_block(
            patch_view,
            sample_indices,
            target_flat,
            variance_flat,
            source_arr.shape,
            spatial_terms,
            background_terms,
            horizontal_reference,
            horizontal_basis,
            horizontal_coefficients,
            vertical_reference,
            vertical_basis,
            regularization=resolved_config.regularization,
            kernel_shape=(kernel_height, kernel_width),
        )
        flux_scale = float(b_coefficients[0])
        vertical_count = vertical_coefficients.size
        vertical_coefficients = b_coefficients[
            1 : 1 + vertical_count
        ].reshape(vertical_coefficients.shape)
        background_coefficients = b_coefficients[1 + vertical_count :]

        a_coefficients, a_condition = _solve_horizontal_block(
            patch_view,
            sample_indices,
            target_flat,
            variance_flat,
            source_arr.shape,
            spatial_terms,
            background_terms,
            horizontal_reference,
            horizontal_basis,
            vertical_reference,
            vertical_basis,
            vertical_coefficients,
            flux_scale,
            regularization=resolved_config.regularization,
            kernel_shape=(kernel_height, kernel_width),
        )
        horizontal_count = horizontal_coefficients.size
        horizontal_coefficients = a_coefficients[:horizontal_count].reshape(
            horizontal_coefficients.shape
        )
        background_coefficients = a_coefficients[horizontal_count:]
        condition_number = max(a_condition, b_condition)

        chi2 = _sample_chi2(
            patch_view,
            sample_indices,
            target_flat,
            variance_flat,
            source_arr.shape,
            spatial_terms,
            background_terms,
            horizontal_reference,
            horizontal_basis,
            horizontal_coefficients,
            vertical_reference,
            vertical_basis,
            vertical_coefficients,
            flux_scale,
            background_coefficients,
            kernel_shape=(kernel_height, kernel_width),
        )
        objective = chi2 + resolved_config.regularization * float(
            np.sum(horizontal_coefficients**2)
            + np.sum(vertical_coefficients**2)
        )
        objective_values.append(objective)
        objective_floor = base_objective_floor * max(condition_number, 1.0)
        if not np.isfinite(objective_floor):
            # An overflowed energy estimate cannot establish convergence.
            objective_floor = 0.0
        if np.isfinite(objective) and objective <= objective_floor:
            converged = True
            break
        if len(objective_values) > 1:
            if _objective_has_converged(
                objective_values[-2],
                objective,
                tolerance=resolved_config.tolerance,
                objective_floor=objective_floor,
            ):
                converged = True
                break

    matched, residual, background = _reconstruct_images(
        source_arr,
        target_arr,
        variance_arr,
        (kernel_height, kernel_width),
        spatial_terms,
        background_terms,
        horizontal_reference,
        horizontal_basis,
        horizontal_coefficients,
        vertical_reference,
        vertical_basis,
        vertical_coefficients,
        flux_scale,
        background_coefficients,
    )
    dof = fit_sample_count - parameter_count

    return SpatialALSFitResult(
        matched=matched,
        residual=residual,
        fit_mask=effective_fit_mask,
        background=background,
        horizontal_reference=horizontal_reference,
        vertical_reference=vertical_reference,
        horizontal_basis=horizontal_basis,
        vertical_basis=vertical_basis,
        horizontal_coefficients=horizontal_coefficients,
        vertical_coefficients=vertical_coefficients,
        background_coefficients=background_coefficients,
        flux_scale=flux_scale,
        spatial_terms=spatial_terms,
        background_terms=background_terms,
        image_shape=source_arr.shape,
        objective_history=np.asarray(objective_values, dtype=np.float64),
        chi2=chi2,
        dof=dof,
        fit_pixel_count=fit_sample_count,
        unique_fit_pixel_count=unique_fit_pixel_count,
        iterations=len(objective_values),
        converged=converged,
        regularization=resolved_config.regularization,
        flux_conserve=resolved_config.flux_conserve,
        condition_number=condition_number,
    )


def _validate_kernel_shape(
    kernel_shape: tuple[int, int],
    image_shape: tuple[int, int],
) -> tuple[int, int]:
    try:
        shape_values = tuple(kernel_shape)
    except TypeError as exc:
        raise ValueError("kernel_shape must be a 2-tuple") from exc
    if len(shape_values) != 2:
        raise ValueError("kernel_shape must be a 2-tuple")
    if any(
        not isinstance(value, Integral) or isinstance(value, bool)
        for value in shape_values
    ):
        raise ValueError("kernel dimensions must be integers")
    height, width = (int(value) for value in shape_values)
    if height <= 0 or width <= 0:
        raise ValueError("kernel dimensions must be positive")
    if height % 2 == 0 or width % 2 == 0:
        raise ValueError("kernel_shape must use odd dimensions")
    if height > image_shape[0] or width > image_shape[1]:
        raise ValueError("kernel_shape leaves no valid interior pixels")
    return height, width


def _validate_line_basis_length(
    raw_basis: np.ndarray,
    length: int,
    *,
    axis: str,
) -> None:
    function_count = int(raw_basis.shape[0])
    if function_count > length:
        raise ValueError(
            f"line basis has {function_count} functions but the kernel "
            f"{axis} is {length}; reduce basis degrees or components, or "
            "enlarge kernel_shape"
        )


def _prepare_line_basis(
    raw_basis: np.ndarray,
    *,
    flux_conserve: bool,
) -> tuple[np.ndarray, np.ndarray]:
    reference_sum = float(raw_basis[0].sum())
    if abs(reference_sum) <= 1.0e-12:
        raise ValueError("first line basis must have a non-zero sum")
    reference = raw_basis[0] / reference_sum
    prepared: list[np.ndarray] = []
    for row in raw_basis[1:]:
        adjusted = np.asarray(row, dtype=np.float64)
        if flux_conserve:
            adjusted = adjusted - float(adjusted.sum()) * reference
        norm = float(np.linalg.norm(adjusted))
        if norm <= 1.0e-12:
            raise ValueError("line basis corrections are rank-deficient")
        prepared.append(adjusted / norm)
    if not prepared:
        return reference, np.empty((0, raw_basis.shape[1]))
    correction_basis = np.stack(prepared, axis=0)
    full_basis = np.vstack((reference, correction_basis))
    if np.linalg.matrix_rank(full_basis) != full_basis.shape[0]:
        raise ValueError("line basis corrections are rank-deficient")
    return reference, correction_basis


def _resolve_fit_samples(
    source: np.ndarray,
    target: np.ndarray,
    variance: np.ndarray,
    kernel_shape: tuple[int, int],
    *,
    fit_mask: np.ndarray | None,
    sample_positions: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if fit_mask is not None and sample_positions is not None:
        raise ValueError(
            "specify either fit_mask or sample_positions, not both"
        )

    height, width = source.shape
    margin_y = kernel_shape[0] // 2
    margin_x = kernel_shape[1] // 2
    patch_finite = sliding_window_view(
        np.isfinite(source),
        kernel_shape,
    ).all(axis=(-2, -1))
    valid = np.zeros(source.shape, dtype=bool)
    interior_slice = (
        slice(margin_y, height - margin_y),
        slice(margin_x, width - margin_x),
    )
    valid[interior_slice] = (
        patch_finite
        & np.isfinite(target[interior_slice])
        & np.isfinite(variance[interior_slice])
    )

    if sample_positions is not None:
        positions = np.asarray(sample_positions)
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("sample_positions must have shape (row, 2)")
        if positions.shape[0] == 0:
            raise ValueError("sample_positions must not be empty")
        if not (
            np.issubdtype(positions.dtype, np.integer)
            or np.issubdtype(positions.dtype, np.floating)
        ):
            raise ValueError("sample_positions must contain numeric values")
        if not np.isfinite(positions).all():
            raise ValueError("sample_positions must not contain NaN or inf")
        if not np.equal(positions, np.floor(positions)).all():
            raise ValueError("sample_positions must contain integer values")
        positions = positions.astype(np.int64)
        sample_y = positions[:, 0]
        sample_x = positions[:, 1]
        if np.any(
            (sample_y < margin_y)
            | (sample_y >= height - margin_y)
            | (sample_x < margin_x)
            | (sample_x >= width - margin_x)
        ):
            raise ValueError(
                "sample_positions must lie inside the valid kernel interior"
            )
        sample_indices = sample_y * width + sample_x
        if not np.all(valid.ravel()[sample_indices]):
            raise ValueError(
                "sample_positions must reference finite target and variance "
                "values with finite source footprints"
            )
    else:
        if fit_mask is None:
            sample_indices = np.flatnonzero(valid)
        else:
            selected = _coerce_fit_mask(fit_mask, source.shape) & valid
            sample_indices = np.flatnonzero(selected)

    if sample_indices.size == 0:
        raise ValueError("no finite samples remain inside the fit region")
    effective_mask = np.zeros(source.shape, dtype=bool)
    effective_mask.ravel()[sample_indices] = True
    return sample_indices, effective_mask


def _coerce_fit_mask(
    fit_mask: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    mask = _coerce_boolean_mask(fit_mask)
    if mask.shape != image_shape:
        raise ValueError("fit_mask must match the image shape")
    return mask


def _chebyshev_design(
    sample_y: np.ndarray,
    sample_x: np.ndarray,
    image_shape: tuple[int, int],
    terms: tuple[tuple[int, int], ...],
) -> np.ndarray:
    height, width = image_shape
    normalized_x = _normalized_coordinate(sample_x, width)
    normalized_y = _normalized_coordinate(sample_y, height)
    max_degree = max(max(pair) for pair in terms)
    tx = _chebyshev_values(normalized_x, max_degree)
    ty = _chebyshev_values(normalized_y, max_degree)
    return np.stack(
        [tx[:, degree_x] * ty[:, degree_y] for degree_x, degree_y in terms],
        axis=1,
    )


def _normalized_coordinate(values: np.ndarray, length: int) -> np.ndarray:
    if length == 1:
        return np.zeros(values.size, dtype=np.float64)
    return 2.0 * values.astype(np.float64) / (length - 1) - 1.0


def _chebyshev_values(values: np.ndarray, degree: int) -> np.ndarray:
    output = np.empty((values.size, degree + 1), dtype=np.float64)
    output[:, 0] = 1.0
    if degree >= 1:
        output[:, 1] = values
    for current in range(2, degree + 1):
        output[:, current] = (
            2.0 * values * output[:, current - 1] - output[:, current - 2]
        )
    return output


def _terms_at(
    y: float,
    x: float,
    image_shape: tuple[int, int],
    terms: tuple[tuple[int, int], ...],
) -> np.ndarray:
    height, width = image_shape
    if not (0.0 <= y <= height - 1 and 0.0 <= x <= width - 1):
        raise ValueError("kernel coordinate lies outside the image")
    return _chebyshev_design(
        np.asarray([y], dtype=np.float64),
        np.asarray([x], dtype=np.float64),
        image_shape,
        terms,
    )[0]


def _profiles(
    reference: np.ndarray,
    basis: np.ndarray,
    coefficients: np.ndarray,
    spatial_design: np.ndarray,
    *,
    reference_scale: float = 1.0,
) -> np.ndarray:
    output = np.broadcast_to(
        reference_scale * reference,
        (spatial_design.shape[0], reference.size),
    ).copy()
    if basis.shape[0]:
        output += np.einsum(
            "is,ps,iu->pu",
            coefficients,
            spatial_design,
            basis,
            optimize=True,
        )
    return output


def _contract_patches(
    patches: np.ndarray,
    vertical: np.ndarray,
    horizontal: np.ndarray,
) -> np.ndarray:
    return np.einsum(
        "pvu,pv,pu->p",
        patches,
        vertical[:, ::-1],
        horizontal[:, ::-1],
        optimize=True,
    )


def _patch_chunk(
    patch_view: np.ndarray,
    sample_indices: np.ndarray,
    start: int,
    stop: int,
    image_shape: tuple[int, int],
    kernel_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    margin_y = kernel_shape[0] // 2
    margin_x = kernel_shape[1] // 2
    sample_y, sample_x = np.divmod(
        sample_indices[start:stop],
        image_shape[1],
    )
    patches = patch_view[
        sample_y - margin_y,
        sample_x - margin_x,
    ]
    return patches, sample_y, sample_x


def _solve_horizontal_block(
    patch_view: np.ndarray,
    sample_indices: np.ndarray,
    target_flat: np.ndarray,
    variance_flat: np.ndarray,
    image_shape: tuple[int, int],
    spatial_terms: tuple[tuple[int, int], ...],
    background_terms: tuple[tuple[int, int], ...],
    horizontal_reference: np.ndarray,
    horizontal_basis: np.ndarray,
    vertical_reference: np.ndarray,
    vertical_basis: np.ndarray,
    vertical_coefficients: np.ndarray,
    flux_scale: float,
    *,
    regularization: float,
    kernel_shape: tuple[int, int],
) -> tuple[np.ndarray, float]:
    horizontal_column_count = horizontal_basis.shape[0] * len(spatial_terms)
    column_count = horizontal_column_count + len(background_terms)
    ridge_mask = np.zeros(column_count, dtype=bool)
    ridge_mask[:horizontal_column_count] = True

    def rows(
        start: int,
        stop: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        patches, sample_y, sample_x = _patch_chunk(
            patch_view,
            sample_indices,
            start,
            stop,
            image_shape,
            kernel_shape,
        )
        spatial = _chebyshev_design(
            sample_y,
            sample_x,
            image_shape,
            spatial_terms,
        )
        background = _chebyshev_design(
            sample_y,
            sample_x,
            image_shape,
            background_terms,
        )
        vertical = _profiles(
            vertical_reference,
            vertical_basis,
            vertical_coefficients,
            spatial,
            reference_scale=flux_scale,
        )
        horizontal_ref = np.broadcast_to(
            horizontal_reference,
            (stop - start, horizontal_reference.size),
        )
        baseline = _contract_patches(
            patches,
            vertical,
            horizontal_ref,
        )
        if horizontal_basis.shape[0]:
            response = np.stack(
                [
                    _contract_patches(
                        patches,
                        vertical,
                        np.broadcast_to(row, horizontal_ref.shape),
                    )
                    for row in horizontal_basis
                ],
                axis=1,
            )
            kernel_rows = (
                response[:, :, None] * spatial[:, None, :]
            ).reshape(stop - start, -1)
        else:
            kernel_rows = np.empty((stop - start, 0))
        design = np.concatenate(
            (kernel_rows, background),
            axis=1,
        )
        flat_indices = sample_indices[start:stop]
        return (
            design,
            target_flat[flat_indices] - baseline,
            variance_flat[flat_indices],
        )

    return _solve_weighted_system(
        sample_indices.size,
        column_count,
        rows,
        ridge_mask,
        regularization=regularization,
    )


def _solve_vertical_block(
    patch_view: np.ndarray,
    sample_indices: np.ndarray,
    target_flat: np.ndarray,
    variance_flat: np.ndarray,
    image_shape: tuple[int, int],
    spatial_terms: tuple[tuple[int, int], ...],
    background_terms: tuple[tuple[int, int], ...],
    horizontal_reference: np.ndarray,
    horizontal_basis: np.ndarray,
    horizontal_coefficients: np.ndarray,
    vertical_reference: np.ndarray,
    vertical_basis: np.ndarray,
    *,
    regularization: float,
    kernel_shape: tuple[int, int],
) -> tuple[np.ndarray, float]:
    vertical_column_count = vertical_basis.shape[0] * len(spatial_terms)
    column_count = 1 + vertical_column_count + len(background_terms)
    ridge_mask = np.zeros(column_count, dtype=bool)
    ridge_mask[1 : 1 + vertical_column_count] = True

    def rows(
        start: int,
        stop: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        patches, sample_y, sample_x = _patch_chunk(
            patch_view,
            sample_indices,
            start,
            stop,
            image_shape,
            kernel_shape,
        )
        spatial = _chebyshev_design(
            sample_y,
            sample_x,
            image_shape,
            spatial_terms,
        )
        background = _chebyshev_design(
            sample_y,
            sample_x,
            image_shape,
            background_terms,
        )
        horizontal = _profiles(
            horizontal_reference,
            horizontal_basis,
            horizontal_coefficients,
            spatial,
        )
        vertical_ref = np.broadcast_to(
            vertical_reference,
            (stop - start, vertical_reference.size),
        )
        reference_response = _contract_patches(
            patches,
            vertical_ref,
            horizontal,
        )
        if vertical_basis.shape[0]:
            response = np.stack(
                [
                    _contract_patches(
                        patches,
                        np.broadcast_to(row, vertical_ref.shape),
                        horizontal,
                    )
                    for row in vertical_basis
                ],
                axis=1,
            )
            kernel_rows = (
                response[:, :, None] * spatial[:, None, :]
            ).reshape(stop - start, -1)
        else:
            kernel_rows = np.empty((stop - start, 0))
        design = np.concatenate(
            (
                reference_response[:, None],
                kernel_rows,
                background,
            ),
            axis=1,
        )
        flat_indices = sample_indices[start:stop]
        return (
            design,
            target_flat[flat_indices],
            variance_flat[flat_indices],
        )

    return _solve_weighted_system(
        sample_indices.size,
        column_count,
        rows,
        ridge_mask,
        regularization=regularization,
    )


def _solve_weighted_system(
    row_count: int,
    column_count: int,
    row_builder,
    ridge_mask: np.ndarray,
    *,
    regularization: float,
) -> tuple[np.ndarray, float]:
    if row_count < column_count:
        raise ValueError(
            "fit region is underdetermined for an ALS block: "
            f"{row_count} equations for {column_count} coefficients"
        )
    gram = np.zeros((column_count, column_count), dtype=np.float64)
    rhs = np.zeros(column_count, dtype=np.float64)
    for start in range(0, row_count, _DESIGN_CHUNK_SIZE):
        stop = min(start + _DESIGN_CHUNK_SIZE, row_count)
        design, values, variance = row_builder(start, stop)
        weights = 1.0 / np.sqrt(variance)
        weighted_design = design * weights[:, None]
        weighted_values = values * weights
        gram += weighted_design.T @ weighted_design
        rhs += weighted_design.T @ weighted_values
    system = gram.copy()
    if regularization:
        diagonal = np.diag_indices_from(system)
        system[diagonal] += regularization * ridge_mask
    # The Gram matrix mixes kernel columns in weighted squared data units
    # with unit-scale Chebyshev columns, so its raw condition number follows
    # the image units. Gate and solve the diagonally scaled system D G D with
    # D = diag(1 / sqrt(diag(G))) instead, then unscale the coefficients.
    diagonal_values = np.diag(system)
    if not np.all(np.isfinite(diagonal_values)) or np.any(
        diagonal_values <= 0.0
    ):
        raise ValueError("spatial ALS design is singular or ill-conditioned")
    scale = 1.0 / np.sqrt(diagonal_values)
    scaled_system = system * scale[:, None] * scale[None, :]
    condition_number = float(np.linalg.cond(scaled_system))
    if (
        not np.isfinite(condition_number)
        or condition_number > _CONDITION_LIMIT
    ):
        raise ValueError("spatial ALS design is singular or ill-conditioned")
    try:
        coefficients = scale * np.linalg.solve(scaled_system, scale * rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError("spatial ALS block could not be solved") from exc
    return coefficients, condition_number


def _weighted_target_energy(
    target_flat: np.ndarray,
    variance_flat: np.ndarray,
    sample_indices: np.ndarray,
) -> float:
    energy = 0.0
    for start in range(0, sample_indices.size, _DESIGN_CHUNK_SIZE):
        stop = min(start + _DESIGN_CHUNK_SIZE, sample_indices.size)
        chunk_indices = sample_indices[start:stop]
        energy += float(
            np.sum(
                target_flat[chunk_indices] ** 2 / variance_flat[chunk_indices]
            )
        )
    return energy


def _objective_has_converged(
    prior: float,
    current: float,
    *,
    tolerance: float,
    objective_floor: float,
) -> bool:
    if not np.isfinite(prior) or not np.isfinite(current):
        return False
    if max(prior, current) <= objective_floor:
        return True
    if current > prior:
        return False
    improvement = prior - current
    return current <= objective_floor or improvement <= tolerance * max(
        abs(prior),
        objective_floor,
    )


def _predict_chunk(
    patches: np.ndarray,
    sample_y: np.ndarray,
    sample_x: np.ndarray,
    image_shape: tuple[int, int],
    spatial_terms: tuple[tuple[int, int], ...],
    background_terms: tuple[tuple[int, int], ...],
    horizontal_reference: np.ndarray,
    horizontal_basis: np.ndarray,
    horizontal_coefficients: np.ndarray,
    vertical_reference: np.ndarray,
    vertical_basis: np.ndarray,
    vertical_coefficients: np.ndarray,
    flux_scale: float,
    background_coefficients: np.ndarray,
) -> np.ndarray:
    spatial = _chebyshev_design(
        sample_y,
        sample_x,
        image_shape,
        spatial_terms,
    )
    background = _chebyshev_design(
        sample_y,
        sample_x,
        image_shape,
        background_terms,
    )
    horizontal = _profiles(
        horizontal_reference,
        horizontal_basis,
        horizontal_coefficients,
        spatial,
    )
    vertical = _profiles(
        vertical_reference,
        vertical_basis,
        vertical_coefficients,
        spatial,
        reference_scale=flux_scale,
    )
    return (
        _contract_patches(
            patches,
            vertical,
            horizontal,
        )
        + background @ background_coefficients
    )


def _sample_chi2(
    patch_view: np.ndarray,
    sample_indices: np.ndarray,
    target_flat: np.ndarray,
    variance_flat: np.ndarray,
    image_shape: tuple[int, int],
    spatial_terms: tuple[tuple[int, int], ...],
    background_terms: tuple[tuple[int, int], ...],
    horizontal_reference: np.ndarray,
    horizontal_basis: np.ndarray,
    horizontal_coefficients: np.ndarray,
    vertical_reference: np.ndarray,
    vertical_basis: np.ndarray,
    vertical_coefficients: np.ndarray,
    flux_scale: float,
    background_coefficients: np.ndarray,
    *,
    kernel_shape: tuple[int, int],
) -> float:
    chi2 = 0.0
    for start in range(0, sample_indices.size, _DESIGN_CHUNK_SIZE):
        stop = min(start + _DESIGN_CHUNK_SIZE, sample_indices.size)
        patches, sample_y, sample_x = _patch_chunk(
            patch_view,
            sample_indices,
            start,
            stop,
            image_shape,
            kernel_shape,
        )
        predictions = _predict_chunk(
            patches,
            sample_y,
            sample_x,
            image_shape,
            spatial_terms,
            background_terms,
            horizontal_reference,
            horizontal_basis,
            horizontal_coefficients,
            vertical_reference,
            vertical_basis,
            vertical_coefficients,
            flux_scale,
            background_coefficients,
        )
        chunk_indices = sample_indices[start:stop]
        residual = target_flat[chunk_indices] - predictions
        chi2 += float(np.sum((residual**2) / variance_flat[chunk_indices]))
    return chi2


def _reconstruct_images(
    source: np.ndarray,
    target: np.ndarray,
    variance: np.ndarray,
    kernel_shape: tuple[int, int],
    spatial_terms: tuple[tuple[int, int], ...],
    background_terms: tuple[tuple[int, int], ...],
    horizontal_reference: np.ndarray,
    horizontal_basis: np.ndarray,
    horizontal_coefficients: np.ndarray,
    vertical_reference: np.ndarray,
    vertical_basis: np.ndarray,
    vertical_coefficients: np.ndarray,
    flux_scale: float,
    background_coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = source.shape
    margin_y = kernel_shape[0] // 2
    margin_x = kernel_shape[1] // 2
    patch_view = sliding_window_view(source, kernel_shape)
    matched = np.full(source.shape, np.nan, dtype=np.float64)
    matched_flat = matched.ravel()
    target_flat = target.ravel()
    variance_flat = variance.ravel()
    interior_height = height - 2 * margin_y
    interior_width = width - 2 * margin_x
    interior_size = interior_height * interior_width
    for start in range(0, interior_size, _DESIGN_CHUNK_SIZE):
        stop = min(start + _DESIGN_CHUNK_SIZE, interior_size)
        interior_offsets = np.arange(start, stop)
        output_y, output_x = np.divmod(interior_offsets, interior_width)
        output_y += margin_y
        output_x += margin_x
        output_indices = output_y * width + output_x
        patches = patch_view[
            output_y - margin_y,
            output_x - margin_x,
        ]
        valid = (
            np.isfinite(patches).all(axis=(-2, -1))
            & np.isfinite(target_flat[output_indices])
            & np.isfinite(variance_flat[output_indices])
        )
        if not np.any(valid):
            continue
        predictions = _predict_chunk(
            patches[valid],
            output_y[valid],
            output_x[valid],
            source.shape,
            spatial_terms,
            background_terms,
            horizontal_reference,
            horizontal_basis,
            horizontal_coefficients,
            vertical_reference,
            vertical_basis,
            vertical_coefficients,
            flux_scale,
            background_coefficients,
        )
        matched_flat[output_indices[valid]] = predictions
    residual = target - matched

    background = np.empty(source.shape, dtype=np.float64)
    background_flat = background.ravel()
    for start in range(0, background.size, _DESIGN_CHUNK_SIZE):
        stop = min(start + _DESIGN_CHUNK_SIZE, background.size)
        output_indices = np.arange(start, stop)
        output_y, output_x = np.divmod(output_indices, width)
        design = _chebyshev_design(
            output_y,
            output_x,
            source.shape,
            background_terms,
        )
        background_flat[start:stop] = design @ background_coefficients
    return matched, residual, background


__all__ = [
    "SpatialALSConfig",
    "SpatialALSFitResult",
    "solve_spatial_als",
]
