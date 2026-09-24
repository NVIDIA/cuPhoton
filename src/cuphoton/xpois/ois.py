# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Alard-Lupton style optimal image subtraction reference solvers.

The implemented solvers cover both the source-backed constant-kernel case and
a rank-one separable-kernel alternating least-squares extension:

- weighted least squares
- Gaussian-times-polynomial kernel bases
- optional polynomial differential background
- optional flux-conserving basis rewrite
- separable horizontal and vertical Gaussian-polynomial profiles
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from numbers import Integral, Real
from typing import Any, Iterable, Literal, Sequence

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

EXPLICIT_BACKENDS = ("cpu", "cupy", "numba-cuda", "cutile")
AUTO_BACKEND_PREFERENCE = ("cupy", "numba-cuda", "cpu")
SUPPORTED_BACKENDS = ("auto", *EXPLICIT_BACKENDS)
_SUPPORTED_BACKENDS_SET = frozenset(SUPPORTED_BACKENDS)
_NUMBA_MAX_COLUMNS = 64
_NUMBA_THREADS_PER_BLOCK = 256
_NUMBA_ROWS_PER_BLOCK = 1024
_CUTILE_MMA_SMALL_ROWS_PER_TILE = 32
_CUTILE_MMA_LARGE_ROWS_PER_TILE = 128
# Fits with at least this many rows use the 128-row tile specialization.
_CUTILE_MMA_LARGE_ROW_THRESHOLD = 1 << 20
_CUTILE_MMA_K_TILE = 16
_NUMBA_ACCUMULATE_KERNEL: Any | None = None
ct: Any | None = None


@dataclass(frozen=True)
class GaussianBasisComponent:
    """One Gaussian-times-polynomial basis family.

    Attributes
    ----------
    sigma
        Gaussian standard deviation in pixels.
    degree
        Maximum total polynomial degree multiplied by the Gaussian.
    """

    sigma: float
    degree: int


@dataclass(frozen=True)
class BasisTerm:
    """Metadata for one concrete two-dimensional basis kernel.

    Attributes
    ----------
    component_index
        Index of the originating :class:`GaussianBasisComponent`.
    sigma
        Gaussian standard deviation in pixels.
    poly_u_degree, poly_v_degree
        Polynomial powers along the kernel coordinate axes.
    zero_sum
        Whether flux-conserving rewriting forced this term to sum to zero.
    """

    component_index: int
    sigma: float
    poly_u_degree: int
    poly_v_degree: int
    zero_sum: bool


@dataclass
class ConstantKernelFitResult:
    """Result of a constant-kernel weighted least-squares fit.

    Attributes
    ----------
    kernel
        Fitted convolution kernel with shape ``(kernel_y, kernel_x)``.
    matched
        Reference image convolved with ``kernel`` plus the fitted background;
        it has the target shape and target pixel units.
    residual
        ``target - matched`` with the target shape and target pixel units.
    fit_mask
        Boolean array with the target shape; true pixels entered the solve.
    background
        Differential-background image with target shape and pixel units.
    kernel_coefficients, background_coefficients
        One-dimensional fitted model coefficients.
    basis_kernels
        Concrete kernels with shape ``(terms, kernel_y, kernel_x)``.
    basis_terms
        Metadata corresponding to ``basis_kernels``.
    chi2, dof, fit_pixel_count
        Weighted residual sum, degrees of freedom, and row count. The
        chi-square units are residual-units squared per variance-unit and are
        dimensionless when variance is supplied in squared target units.
    flux_conserve
        Whether a unit-sum/zero-sum basis rewrite was requested.
    backend
        Backend used to accumulate the normal equations.
    """

    kernel: np.ndarray
    matched: np.ndarray
    residual: np.ndarray
    fit_mask: np.ndarray
    background: np.ndarray
    kernel_coefficients: np.ndarray
    background_coefficients: np.ndarray
    basis_kernels: np.ndarray
    basis_terms: tuple[BasisTerm, ...]
    chi2: float
    dof: int
    fit_pixel_count: int
    flux_conserve: bool
    backend: str = "cpu"


@dataclass(frozen=True, slots=True)
class DeviceConstantKernelFitResult:
    """CuPy-resident result of a constant-kernel least-squares fit.

    The array-valued fields remain on the active CUDA device. Scalar
    diagnostics and basis metadata are host values so callers can record
    compact receipts without materializing any image-sized arrays. The result
    holds strong references to every returned array, but it must remain in the
    producing process and on ``device_id``; it is not a Dragon payload or an
    artifact. Callers should treat its arrays as read-only. Consumers on a
    different CUDA stream are responsible for establishing stream ordering.

    Input arrays are borrowed and are not mutated. Completion timing is not
    part of this contract: callers that measure or materialize results must
    synchronize explicitly.

    Attributes
    ----------
    kernel
        Fitted convolution kernel as a CuPy array.
    matched
        Device-resident reference image convolved with ``kernel`` plus the
        fitted background.
    residual
        Device-resident ``target - matched`` image.
    fit_mask
        Device-resident boolean mask of pixels that entered the solve.
    background
        Device-resident differential-background image.
    kernel_coefficients, background_coefficients
        Device-resident fitted model coefficients.
    basis_kernels
        Device-resident concrete Gaussian-polynomial basis kernels.
    basis_terms
        Host metadata corresponding to ``basis_kernels``.
    chi2, dof, fit_pixel_count
        Compact fit diagnostics copied to host scalars.
    flux_conserve
        Whether a unit-sum/zero-sum basis rewrite was requested.
    backend
        Always ``"cupy"``.
    schema, solver, result_location
        Immutable tags identifying this device-only result contract.
    device_id
        CUDA device ordinal shared by every array-valued field.
    """

    kernel: Any
    matched: Any
    residual: Any
    fit_mask: Any
    background: Any
    kernel_coefficients: Any
    background_coefficients: Any
    basis_kernels: Any
    basis_terms: tuple[BasisTerm, ...]
    chi2: float
    dof: int
    fit_pixel_count: int
    flux_conserve: bool
    device_id: int
    schema: Literal["cuphoton.xpois.device-fit-result/v1"] = field(
        init=False,
        default="cuphoton.xpois.device-fit-result/v1",
    )
    solver: Literal["constant"] = field(init=False, default="constant")
    backend: Literal["cupy"] = field(init=False, default="cupy")
    result_location: Literal["device"] = field(
        init=False,
        default="device",
    )

    def __post_init__(self) -> None:
        if isinstance(self.device_id, bool) or not isinstance(
            self.device_id, (int, np.integer)
        ):
            raise TypeError(
                "device_id must be an integer CUDA device ordinal"
            )
        if self.device_id < 0:
            raise ValueError("device_id must be non-negative")

        cp, _ = _load_cupy()
        arrays = {
            "kernel": self.kernel,
            "matched": self.matched,
            "residual": self.residual,
            "fit_mask": self.fit_mask,
            "background": self.background,
            "kernel_coefficients": self.kernel_coefficients,
            "background_coefficients": self.background_coefficients,
            "basis_kernels": self.basis_kernels,
        }
        for name, value in arrays.items():
            if not isinstance(value, cp.ndarray):
                raise TypeError(f"{name} must be a CuPy array")
            if int(value.device.id) != int(self.device_id):
                raise ValueError(
                    f"{name} is on CUDA device {int(value.device.id)}, "
                    f"expected {int(self.device_id)}"
                )

        image_shape = tuple(int(value) for value in self.matched.shape)
        for name in ("residual", "fit_mask", "background"):
            value = arrays[name]
            if tuple(int(item) for item in value.shape) != image_shape:
                raise ValueError(f"{name} must match the matched image shape")
        if len(image_shape) != 2:
            raise ValueError("matched image must be two-dimensional")
        if self.kernel.ndim != 2:
            raise ValueError("kernel must be two-dimensional")
        if self.basis_kernels.ndim != 3:
            raise ValueError("basis_kernels must be three-dimensional")
        if tuple(
            int(value) for value in self.basis_kernels.shape[1:]
        ) != tuple(int(value) for value in self.kernel.shape):
            raise ValueError("basis_kernels must match the kernel shape")
        if self.kernel_coefficients.ndim != 1:
            raise ValueError("kernel_coefficients must be one-dimensional")
        if self.background_coefficients.ndim != 1:
            raise ValueError(
                "background_coefficients must be one-dimensional"
            )
        basis_count = int(self.basis_kernels.shape[0])
        if int(self.kernel_coefficients.size) != basis_count:
            raise ValueError(
                "kernel_coefficients must align with basis_kernels"
            )
        if len(self.basis_terms) != basis_count:
            raise ValueError("basis_terms must align with basis_kernels")
        if np.dtype(self.fit_mask.dtype) != np.dtype(bool):
            raise TypeError("fit_mask must have boolean dtype")
        for name, value in arrays.items():
            if name == "fit_mask":
                continue
            if np.dtype(value.dtype) != np.dtype(np.float64):
                raise TypeError(f"{name} must have float64 dtype")

        if not isinstance(self.fit_pixel_count, int):
            raise TypeError("fit_pixel_count must be an integer")
        if self.fit_pixel_count < 1:
            raise ValueError("fit_pixel_count must be positive")
        if not isinstance(self.dof, int):
            raise TypeError("dof must be an integer")
        expected_dof = self.fit_pixel_count - int(
            self.kernel_coefficients.size + self.background_coefficients.size
        )
        if self.dof != expected_dof:
            raise ValueError(
                "dof is inconsistent with the fitted coefficients"
            )
        if not np.isfinite(self.chi2) or self.chi2 < 0.0:
            raise ValueError("chi2 must be finite and non-negative")
        if not isinstance(self.flux_conserve, bool):
            raise TypeError("flux_conserve must be boolean")

    def __reduce_ex__(self, protocol: int) -> Any:
        raise TypeError(
            "DeviceConstantKernelFitResult cannot be pickled; keep device "
            "results inside the producing process"
        )


@dataclass
class SeparableKernelFitResult:
    """Result of a separable-kernel alternating least-squares fit.

    Attributes
    ----------
    kernel
        Outer product of the fitted vertical and horizontal profiles, with
        shape ``(kernel_y, kernel_x)``.
    matched
        Convolved reference plus background, with target shape and units.
    residual
        ``target - matched`` with target shape and pixel units.
    fit_mask
        Boolean array with target shape; true pixels entered the solve.
    background
        Differential-background image with target shape and pixel units.
    horizontal_kernel, vertical_kernel
        Fitted profiles with lengths ``kernel_x`` and ``kernel_y``.
    horizontal_coefficients, vertical_coefficients
        Coefficients of the one-dimensional bases.
    horizontal_basis, vertical_basis
        Basis matrices with shape ``(terms, kernel_x)`` and
        ``(terms, kernel_y)``.
    chi2, dof, fit_pixel_count
        Weighted residual sum, degrees of freedom, and row count. The
        chi-square units are residual-units squared per variance-unit and are
        dimensionless when variance is supplied in squared target units.
    iterations, converged
        Alternating-solver termination state.
    flux_conserve
        Whether the fitted two-dimensional kernel is normalized to unit sum.
    """

    kernel: np.ndarray
    matched: np.ndarray
    residual: np.ndarray
    fit_mask: np.ndarray
    background: np.ndarray
    background_coefficients: np.ndarray
    horizontal_kernel: np.ndarray
    vertical_kernel: np.ndarray
    horizontal_coefficients: np.ndarray
    vertical_coefficients: np.ndarray
    horizontal_basis: np.ndarray
    vertical_basis: np.ndarray
    chi2: float
    dof: int
    fit_pixel_count: int
    iterations: int
    converged: bool
    flux_conserve: bool


@dataclass(frozen=True)
class AutoStampMaskResult:
    """Auto-selected compact-source stamp mask and diagnostics.

    Attributes
    ----------
    mask
        Boolean image mask containing selected rectangular stamps.
    centers, stamps
        Selected centers and half-open ``(y0, y1, x0, x1)`` rectangles.
    scores, compact_fractions, radial_moments
        Per-source ranking diagnostics.
    threshold, peak_percentile
        Applied peak threshold and its requested percentile.
    stamp_size, max_stamps
        Stamp geometry and selection limit.
    background_filter_size, peak_filter_size, min_separation
        Source-selection filter sizes and spacing constraint.
    """

    mask: np.ndarray
    centers: tuple[tuple[int, int], ...]
    stamps: tuple[tuple[int, int, int, int], ...]
    scores: tuple[float, ...]
    compact_fractions: tuple[float, ...]
    radial_moments: tuple[float, ...]
    threshold: float
    peak_percentile: float
    stamp_size: int
    max_stamps: int
    background_filter_size: int
    peak_filter_size: int
    min_separation: int

    def to_metadata(self) -> dict[str, object]:
        return {
            "kind": "auto_stamp_mask",
            "stamp_size": self.stamp_size,
            "max_stamps": self.max_stamps,
            "background_filter_size": self.background_filter_size,
            "peak_filter_size": self.peak_filter_size,
            "min_separation": self.min_separation,
            "peak_percentile": self.peak_percentile,
            "peak_threshold": self.threshold,
            "selected_count": len(self.centers),
            "fit_mask_fraction": float(np.mean(self.mask)),
            "centers_yx": [[int(y), int(x)] for y, x in self.centers],
            "stamps_y0y1x0x1": [
                [int(y0), int(y1), int(x0), int(x1)]
                for y0, y1, x0, x1 in self.stamps
            ],
            "scores": [float(value) for value in self.scores],
            "compact_fractions": [
                float(value) for value in self.compact_fractions
            ],
            "radial_moments": [float(value) for value in self.radial_moments],
        }


def triangular_degree_pairs(max_degree: int) -> list[tuple[int, int]]:
    """Return polynomial powers through one total degree.

    Parameters
    ----------
    max_degree
        Maximum non-negative total degree.

    Returns
    -------
    list of tuple of int
        Non-negative ``(i, j)`` pairs satisfying ``i + j <= max_degree``.
    """

    if max_degree < 0:
        raise ValueError("max_degree must be non-negative")

    pairs: list[tuple[int, int]] = []
    for deg_u in range(max_degree + 1):
        for deg_v in range(max_degree + 1 - deg_u):
            pairs.append((deg_u, deg_v))
    return pairs


def make_stamp_mask(
    shape: tuple[int, int],
    stamps: Iterable[tuple[int, int, int, int]],
) -> np.ndarray:
    """Build a boolean mask from rectangular stamps.

    Parameters
    ----------
    shape
        Output ``(height, width)``.
    stamps
        Rectangles expressed as ``(y0, y1, x0, x1)`` with half-open bounds.

    Returns
    -------
    numpy.ndarray
        Boolean mask with pixels inside each rectangle set to true.
    """

    mask = np.zeros(shape, dtype=bool)
    height, width = shape
    for y0, y1, x0, x1 in stamps:
        if not (0 <= y0 < y1 <= height and 0 <= x0 < x1 <= width):
            raise ValueError(
                "stamps must satisfy 0 <= y0 < y1 <= H and 0 <= x0 < x1 <= W"
            )
        mask[y0:y1, x0:x1] = True
    return mask


def build_compact_source_stamp_mask(
    image: np.ndarray,
    *,
    variance: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    stamp_size: int = 31,
    max_stamps: int = 5,
    peak_percentile: float = 99.5,
    background_filter_size: int = 15,
    peak_filter_size: int = 9,
    min_separation: int | None = None,
    compact_core_size: int = 3,
) -> AutoStampMaskResult:
    """Build a sparse fit mask around compact sources.

    Parameters
    ----------
    image
        Two-dimensional image used for peak selection.
    variance
        Optional positive variance image for significance ranking.
    valid_mask
        Optional boolean mask limiting candidate and stamp pixels.
    stamp_size
        Odd side length of each selected square stamp.
    max_stamps
        Maximum number of sources to retain.
    peak_percentile
        Percentile threshold applied to the high-pass image.
    background_filter_size, peak_filter_size
        Odd median and local-maximum filter widths.
    min_separation
        Minimum source-center separation in pixels.
    compact_core_size
        Odd central width used for compactness scoring.

    Returns
    -------
    AutoStampMaskResult
        Selected mask, stamp geometry, and ranking diagnostics.

    Notes
    -----
    The selector is intentionally conservative:

    * Estimate a local background with a median filter.
    * Find local maxima in the background-subtracted image.
    * Rank peaks by local significance and compactness.
    * Keep well-separated, odd-sized rectangular stamps.
    """

    from scipy.ndimage import maximum_filter, median_filter

    stamp_size = _validate_odd_positive(stamp_size, name="stamp_size")
    background_filter_size = _validate_odd_positive(
        background_filter_size,
        name="background_filter_size",
    )
    peak_filter_size = _validate_odd_positive(
        peak_filter_size,
        name="peak_filter_size",
    )
    compact_core_size = _validate_odd_positive(
        compact_core_size,
        name="compact_core_size",
    )
    if compact_core_size > stamp_size:
        raise ValueError("compact_core_size must not exceed stamp_size")
    if max_stamps <= 0:
        raise ValueError("max_stamps must be positive")
    if not 0.0 < peak_percentile < 100.0:
        raise ValueError("peak_percentile must satisfy 0 < p < 100")

    array = np.asarray(image, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("build_compact_source_stamp_mask expects a 2D image")

    base_valid = np.isfinite(array)
    if variance is not None:
        variance_arr = np.asarray(variance, dtype=np.float64)
        if variance_arr.shape != array.shape:
            raise ValueError("variance must match the image shape")
        base_valid &= np.isfinite(variance_arr) & (variance_arr > 0)
    else:
        variance_arr = None
    if valid_mask is not None:
        valid_arr = _coerce_boolean_mask(valid_mask)
        if valid_arr.shape != array.shape:
            raise ValueError("valid_mask must match the image shape")
        base_valid &= valid_arr

    if not np.any(base_valid):
        raise ValueError(
            "no finite pixels remain for compact-source selection"
        )

    fill_value = float(np.median(array[base_valid]))
    filtered_input = np.where(base_valid, array, fill_value)
    background = median_filter(
        filtered_input, size=background_filter_size, mode="nearest"
    )
    highpass = np.where(base_valid, array - background, -np.inf)
    threshold = float(np.percentile(highpass[base_valid], peak_percentile))
    maxima = maximum_filter(highpass, size=peak_filter_size, mode="nearest")
    peaks = np.argwhere(
        (highpass == maxima) & base_valid & (highpass > threshold)
    )
    if peaks.size == 0:
        raise ValueError(
            "no compact-source peaks found for the requested mask"
        )

    half = stamp_size // 2
    min_distance = min_separation
    if min_distance is None:
        min_distance = max(1, half)
    if min_distance <= 0:
        raise ValueError("min_separation must be positive")

    candidates: list[
        tuple[float, float, float, int, int, tuple[int, int, int, int]]
    ] = []
    center_half = compact_core_size // 2
    for y, x in peaks:
        y = int(y)
        x = int(x)
        y0 = y - half
        y1 = y + half + 1
        x0 = x - half
        x1 = x + half + 1
        if y0 < 0 or x0 < 0 or y1 > array.shape[0] or x1 > array.shape[1]:
            continue
        stamp_valid = base_valid[y0:y1, x0:x1]
        if not np.all(stamp_valid):
            continue
        stamp = array[y0:y1, x0:x1]
        local_background = float(np.median(stamp))
        weights = np.clip(stamp - local_background, 0.0, None)
        total_weight = float(np.sum(weights))
        if total_weight <= 0.0:
            continue
        center_weights = weights[
            half - center_half : half + center_half + 1,
            half - center_half : half + center_half + 1,
        ]
        compact_fraction = float(np.sum(center_weights) / total_weight)
        yy, xx = np.mgrid[-half : half + 1, -half : half + 1]
        radial_moment = float(
            np.sum(weights * (xx * xx + yy * yy)) / total_weight
        )
        local_signal = float(stamp[half, half] - local_background)
        if variance_arr is None:
            score = local_signal
        else:
            score = float(local_signal / np.sqrt(variance_arr[y, x]))
        candidates.append(
            (
                score,
                compact_fraction,
                -radial_moment,
                y,
                x,
                (y0, y1, x0, x1),
            )
        )

    if not candidates:
        raise ValueError(
            "no valid compact-source stamps remain after finite-mask "
            "filtering"
        )

    candidates.sort(reverse=True)
    selected: list[
        tuple[float, float, float, int, int, tuple[int, int, int, int]]
    ] = []
    min_distance_sq = min_distance * min_distance
    for candidate in candidates:
        _, _, _, y, x, _ = candidate
        if any(
            (y - y2) ** 2 + (x - x2) ** 2 < min_distance_sq
            for *_, y2, x2, _ in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= max_stamps:
            break

    if not selected:
        raise ValueError(
            "compact-source selection produced no separated stamps"
        )

    stamps = tuple(stamp for *_, stamp in selected)
    mask = make_stamp_mask(array.shape, stamps)
    return AutoStampMaskResult(
        mask=mask,
        centers=tuple((int(y), int(x)) for _, _, _, y, x, _ in selected),
        stamps=stamps,
        scores=tuple(float(score) for score, *_ in selected),
        compact_fractions=tuple(
            float(compact_fraction) for _, compact_fraction, *_ in selected
        ),
        radial_moments=tuple(
            float(-neg_radial) for _, _, neg_radial, *_ in selected
        ),
        threshold=threshold,
        peak_percentile=float(peak_percentile),
        stamp_size=stamp_size,
        max_stamps=max_stamps,
        background_filter_size=background_filter_size,
        peak_filter_size=peak_filter_size,
        min_separation=min_distance,
    )


def build_gaussian_polynomial_basis(
    kernel_shape: tuple[int, int],
    components: Sequence[GaussianBasisComponent],
    *,
    flux_conserve: bool = False,
    flux_reference_index: int = 0,
) -> tuple[np.ndarray, tuple[BasisTerm, ...]]:
    """Build Gaussian-times-polynomial kernel basis images.

    Parameters
    ----------
    kernel_shape
        Odd ``(height, width)`` of each kernel image.
    components
        Gaussian widths and polynomial degrees to expand.
    flux_conserve
        Rewrite the basis into one unit-sum term and zero-sum remainder.
    flux_reference_index
        Raw basis index that carries the flux scale when rewriting.

    Returns
    -------
    kernels
        Array with shape ``(terms, height, width)``.
    terms
        Metadata for each returned basis image.
    """

    kernel_shape = _validate_kernel_shape(kernel_shape)
    raw_kernels, raw_terms = _build_raw_basis_kernels(
        kernel_shape, components
    )
    if not flux_conserve:
        return raw_kernels, raw_terms

    if not 0 <= flux_reference_index < raw_kernels.shape[0]:
        raise ValueError("flux_reference_index is out of range")

    reference = raw_kernels[flux_reference_index]
    reference_sum = float(reference.sum())
    if abs(reference_sum) < 1e-12:
        raise ValueError(
            "flux_reference_index must point to a non-zero-sum basis kernel"
        )

    normalized_reference = reference / reference_sum
    kernels = [normalized_reference]
    terms = [
        BasisTerm(
            component_index=raw_terms[flux_reference_index].component_index,
            sigma=raw_terms[flux_reference_index].sigma,
            poly_u_degree=raw_terms[flux_reference_index].poly_u_degree,
            poly_v_degree=raw_terms[flux_reference_index].poly_v_degree,
            zero_sum=False,
        )
    ]

    for index, (kernel, term) in enumerate(zip(raw_kernels, raw_terms)):
        if index == flux_reference_index:
            continue
        kernel_sum = float(kernel.sum())
        if abs(kernel_sum) < 1e-12:
            adjusted = kernel
        else:
            adjusted = kernel - kernel_sum * normalized_reference
        kernels.append(adjusted)
        terms.append(
            BasisTerm(
                component_index=term.component_index,
                sigma=term.sigma,
                poly_u_degree=term.poly_u_degree,
                poly_v_degree=term.poly_v_degree,
                zero_sum=True,
            )
        )

    return np.stack(kernels, axis=0), tuple(terms)


def solve_constant_kernel(
    reference: np.ndarray,
    target: np.ndarray,
    components: Sequence[GaussianBasisComponent],
    *,
    kernel_shape: tuple[int, int] = (15, 15),
    variance: np.ndarray | None = None,
    fit_mask: np.ndarray | None = None,
    background_degree: int = 0,
    flux_conserve: bool = False,
    flux_reference_index: int = 0,
    backend: str = "auto",
) -> ConstantKernelFitResult:
    """Fit a constant convolution kernel from ``reference`` to ``target``.

    Parameters
    ----------
    reference, target
        Two-dimensional, registered input images.
    components
        Gaussian-polynomial basis families.
    kernel_shape
        Odd convolution-kernel shape.
    variance
        Optional positive target variance image.
    fit_mask
        Optional boolean pixels to include in the solve.
    background_degree
        Total degree of the differential-background polynomial.
    flux_conserve
        Use a flux-conserving kernel basis rewrite.
    flux_reference_index
        Raw basis term carrying the overall flux scale.
    backend
        Normal-equation backend. ``auto`` tries CuPy, then Numba-CUDA, then
        CPU. ``cutile`` is explicit-only.

    Returns
    -------
    ConstantKernelFitResult
        Fitted kernel, images, coefficients, and diagnostics.

    Notes
    -----
    The weighted least-squares model is

    The system solved is a weighted least-squares problem over the selected
    pixels:

    ``target ~= Σ_j a_j (reference ⊗ basis_j) + background(x, y)``
    """

    backend = _normalize_backend(backend)
    reference_arr = np.asarray(reference, dtype=np.float64)
    target_arr = np.asarray(target, dtype=np.float64)
    _validate_image_pair(reference_arr, target_arr)

    reference_model_arr = np.where(
        np.isfinite(reference_arr), reference_arr, 0.0
    )
    target_model_arr = np.where(np.isfinite(target_arr), target_arr, 0.0)
    variance_arr = _coerce_variance(target_arr.shape, variance)
    mask_arr = _coerce_mask(target_arr.shape, fit_mask, kernel_shape)
    boundary_mask = _default_fit_mask(target_arr.shape, kernel_shape)
    invalid_pixel_mask = _invalid_input_mask(
        reference_arr,
        target_arr,
        variance_arr,
        kernel_shape,
    )
    mask_arr &= ~invalid_pixel_mask
    if not np.any(mask_arr):
        raise ValueError("no finite pixels remain within the fit region")

    basis_kernels, basis_terms = build_gaussian_polynomial_basis(
        kernel_shape,
        components,
        flux_conserve=flux_conserve,
        flux_reference_index=flux_reference_index,
    )
    background_terms = background_design(target_arr.shape, background_degree)
    if backend == "cupy":
        gram_matrix, rhs_vector, row_count = (
            _accumulate_normal_equations_cupy(
                reference_model_arr,
                target_model_arr,
                variance_arr,
                mask_arr,
                basis_kernels,
                background_terms,
            )
        )
    elif backend == "numba-cuda":
        gram_matrix, rhs_vector, row_count = (
            _accumulate_normal_equations_numba_cuda(
                reference_model_arr,
                target_model_arr,
                variance_arr,
                mask_arr,
                basis_kernels,
                background_terms,
            )
        )
    elif backend == "cutile":
        gram_matrix, rhs_vector, row_count = (
            _accumulate_normal_equations_cutile(
                reference_model_arr,
                target_model_arr,
                variance_arr,
                mask_arr,
                basis_kernels,
                background_terms,
            )
        )
    else:
        gram_matrix, rhs_vector, row_count = _accumulate_normal_equations(
            reference_model_arr,
            target_model_arr,
            variance_arr,
            mask_arr,
            basis_kernels,
            background_terms,
        )
    column_count = gram_matrix.shape[0]
    if row_count < column_count:
        raise ValueError(
            "fit region is underdetermined for the requested basis: "
            f"{row_count} equations for {column_count} coefficients"
        )
    try:
        condition_number = np.linalg.cond(gram_matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "fit normal equations are singular for the requested basis"
        ) from exc
    if not np.isfinite(condition_number) or condition_number > 1e12:
        raise ValueError(
            "fit normal equations are ill-conditioned for the requested basis"
        )
    try:
        coefficients = np.linalg.solve(gram_matrix, rhs_vector)
    except np.linalg.LinAlgError as exc:
        raise ValueError("fit normal equations could not be solved") from exc

    kernel_coeff_count = basis_kernels.shape[0]
    kernel_coeffs = coefficients[:kernel_coeff_count]
    background_coeffs = coefficients[kernel_coeff_count:]

    kernel = np.tensordot(kernel_coeffs, basis_kernels, axes=(0, 0))
    background = evaluate_background_model(
        target_arr.shape,
        background_coeffs,
        degree=background_degree,
    )
    matched = _matched_image(
        reference_model_arr,
        kernel,
        background,
        backend=backend,
    )
    residual = target_model_arr - matched
    output_invalid_mask = invalid_pixel_mask | ~boundary_mask
    matched[output_invalid_mask] = np.nan
    residual[output_invalid_mask] = np.nan

    residual_fit = residual[mask_arr]
    chi2 = float(np.sum((residual_fit**2) / variance_arr[mask_arr]))
    dof = int(row_count) - int(coefficients.size)

    return ConstantKernelFitResult(
        kernel=kernel,
        matched=matched,
        residual=residual,
        fit_mask=mask_arr,
        background=background,
        kernel_coefficients=kernel_coeffs,
        background_coefficients=background_coeffs,
        basis_kernels=basis_kernels,
        basis_terms=basis_terms,
        chi2=chi2,
        dof=dof,
        fit_pixel_count=int(row_count),
        flux_conserve=flux_conserve,
        backend=backend,
    )


def solve_constant_kernel_device(
    reference: Any,
    target: Any,
    components: Sequence[GaussianBasisComponent],
    *,
    kernel_shape: tuple[int, int] = (15, 15),
    variance: Any | None = None,
    fit_mask: Any | None = None,
    background_degree: int = 0,
    flux_conserve: bool = False,
    flux_reference_index: int = 0,
) -> DeviceConstantKernelFitResult:
    """Fit a constant kernel while keeping array results on the GPU.

    Parameters match :func:`solve_constant_kernel`, except this experimental
    seam is explicitly CuPy-only and therefore has no backend selector.
    NumPy inputs are accepted and copied to the active CUDA device; CuPy
    inputs already on that device are reused when their dtype permits.

    Returns
    -------
    DeviceConstantKernelFitResult
        CuPy-resident fitted arrays and compact host diagnostics.
    """

    cp, cupy_fftconvolve = _load_cupy()
    active_device_id = _active_cupy_device_id(cp)
    for name, value in (
        ("reference", reference),
        ("target", target),
        ("variance", variance),
        ("fit_mask", fit_mask),
    ):
        _validate_cupy_input_device(
            value,
            name=name,
            active_device_id=active_device_id,
            cp=cp,
        )

    reference_arr = cp.asarray(reference, dtype=cp.float64)
    target_arr = cp.asarray(target, dtype=cp.float64)
    if int(reference_arr.device.id) != active_device_id:
        raise ValueError("reference must reside on the active CUDA device")
    if int(target_arr.device.id) != active_device_id:
        raise ValueError("target must reside on the active CUDA device")
    _validate_image_pair(reference_arr, target_arr)

    reference_model_arr = cp.where(
        cp.isfinite(reference_arr), reference_arr, 0.0
    )
    target_model_arr = cp.where(cp.isfinite(target_arr), target_arr, 0.0)
    variance_arr = _coerce_variance_cupy(
        target_arr.shape,
        variance,
        cp=cp,
    )
    mask_arr = _coerce_mask_cupy(
        target_arr.shape,
        fit_mask,
        kernel_shape,
        cp=cp,
    )
    if int(variance_arr.device.id) != active_device_id:
        raise ValueError("variance must reside on the active CUDA device")
    if int(mask_arr.device.id) != active_device_id:
        raise ValueError("fit_mask must reside on the active CUDA device")
    boundary_mask = _default_fit_mask_cupy(
        target_arr.shape,
        kernel_shape,
        cp=cp,
    )
    invalid_pixel_mask = _invalid_input_mask_cupy(
        reference_arr,
        target_arr,
        variance_arr,
        kernel_shape,
        cp=cp,
    )
    mask_arr &= ~invalid_pixel_mask
    if not bool(cp.any(mask_arr).item()):
        raise ValueError("no finite pixels remain within the fit region")

    basis_kernels_host, basis_terms = build_gaussian_polynomial_basis(
        kernel_shape,
        components,
        flux_conserve=flux_conserve,
        flux_reference_index=flux_reference_index,
    )
    basis_kernels = cp.asarray(basis_kernels_host, dtype=cp.float64)
    background_terms = _background_design_cupy(
        target_arr.shape,
        background_degree,
        cp=cp,
    )
    gram_matrix, rhs_vector, row_count = (
        _accumulate_normal_equations_cupy_device(
            reference_model_arr,
            target_model_arr,
            variance_arr,
            mask_arr,
            basis_kernels,
            background_terms,
            cp=cp,
        )
    )
    column_count = int(gram_matrix.shape[0])
    if row_count < column_count:
        raise ValueError(
            "fit region is underdetermined for the requested basis: "
            f"{row_count} equations for {column_count} coefficients"
        )
    try:
        condition_number = float(cp.linalg.cond(gram_matrix).item())
    except cp.linalg.LinAlgError as exc:
        raise ValueError(
            "fit normal equations are singular for the requested basis"
        ) from exc
    if not np.isfinite(condition_number) or condition_number > 1e12:
        raise ValueError(
            "fit normal equations are ill-conditioned for the requested basis"
        )
    try:
        coefficients = cp.linalg.solve(gram_matrix, rhs_vector)
    except cp.linalg.LinAlgError as exc:
        raise ValueError("fit normal equations could not be solved") from exc

    kernel_coeff_count = int(basis_kernels.shape[0])
    kernel_coeffs = coefficients[:kernel_coeff_count]
    background_coeffs = coefficients[kernel_coeff_count:]
    kernel = cp.tensordot(kernel_coeffs, basis_kernels, axes=(0, 0))
    background = cp.tensordot(
        background_coeffs,
        background_terms,
        axes=(0, 0),
    )
    matched = (
        cupy_fftconvolve(reference_model_arr, kernel, mode="same")
        + background
    )
    residual = target_model_arr - matched
    output_invalid_mask = invalid_pixel_mask | ~boundary_mask
    matched[output_invalid_mask] = cp.nan
    residual[output_invalid_mask] = cp.nan

    residual_fit = residual[mask_arr]
    chi2 = float(cp.sum((residual_fit**2) / variance_arr[mask_arr]).item())
    dof = int(row_count) - int(coefficients.size)

    return DeviceConstantKernelFitResult(
        kernel=kernel,
        matched=matched,
        residual=residual,
        fit_mask=mask_arr,
        background=background,
        kernel_coefficients=kernel_coeffs,
        background_coefficients=background_coeffs,
        basis_kernels=basis_kernels,
        basis_terms=basis_terms,
        chi2=chi2,
        dof=dof,
        fit_pixel_count=int(row_count),
        flux_conserve=bool(flux_conserve),
        device_id=active_device_id,
    )


def solve_separable_kernel(
    reference: np.ndarray,
    target: np.ndarray,
    components: Sequence[GaussianBasisComponent],
    *,
    kernel_shape: tuple[int, int] = (15, 15),
    variance: np.ndarray | None = None,
    fit_mask: np.ndarray | None = None,
    background_degree: int = 0,
    flux_conserve: bool = False,
    max_iterations: int = 8,
    tolerance: float = 1e-6,
) -> SeparableKernelFitResult:
    """Fit a separable kernel with alternating least squares.

    Parameters
    ----------
    reference, target
        Two-dimensional, registered input images.
    components
        Gaussian-polynomial basis families for both one-dimensional profiles.
    kernel_shape
        Odd convolution-kernel shape.
    variance
        Optional positive target variance image.
    fit_mask
        Optional boolean pixels to include in the solve.
    background_degree
        Total degree of the differential-background polynomial.
    flux_conserve
        Normalize the two-dimensional kernel to unit sum.
    max_iterations
        Maximum alternating horizontal/vertical updates.
    tolerance
        Relative chi-square improvement used for convergence.

    Returns
    -------
    SeparableKernelFitResult
        Fitted profiles, kernel, images, coefficients, and diagnostics.

    Notes
    -----
    This solves the rank-one model

    ``target ~= reference ⊗ (outer(v, h)) + background(x, y)``

    The horizontal and vertical kernel profiles are each expanded in 1D
    Gaussian-times-polynomial bases and fit by alternating weighted least
    squares steps.
    """

    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")

    reference_arr = np.asarray(reference, dtype=np.float64)
    target_arr = np.asarray(target, dtype=np.float64)
    _validate_image_pair(reference_arr, target_arr)

    reference_model_arr = np.where(
        np.isfinite(reference_arr), reference_arr, 0.0
    )
    target_model_arr = np.where(np.isfinite(target_arr), target_arr, 0.0)
    variance_arr = _coerce_variance(target_arr.shape, variance)
    mask_arr = _coerce_mask(target_arr.shape, fit_mask, kernel_shape)
    boundary_mask = _default_fit_mask(target_arr.shape, kernel_shape)
    invalid_pixel_mask = _invalid_input_mask(
        reference_arr,
        target_arr,
        variance_arr,
        kernel_shape,
    )
    mask_arr &= ~invalid_pixel_mask
    if not np.any(mask_arr):
        raise ValueError("no finite pixels remain within the fit region")

    kernel_height, kernel_width = _validate_kernel_shape(kernel_shape)
    horizontal_basis = _build_line_basis(kernel_width, components)
    vertical_basis = _build_line_basis(kernel_height, components)
    horizontal_coeffs = np.zeros(horizontal_basis.shape[0], dtype=np.float64)
    vertical_coeffs = np.zeros(vertical_basis.shape[0], dtype=np.float64)
    horizontal_coeffs[0] = 1.0
    vertical_coeffs[0] = 1.0
    background_coeffs = np.zeros(
        len(triangular_degree_pairs(background_degree)),
        dtype=np.float64,
    )

    converged = False
    last_chi2: float | None = None
    row_count = int(mask_arr.sum())
    for iteration in range(max_iterations):
        vertical_kernel = np.tensordot(
            vertical_coeffs,
            vertical_basis,
            axes=(0, 0),
        )
        horizontal_step_basis = np.stack(
            [
                np.outer(vertical_kernel, basis_row)
                for basis_row in horizontal_basis
            ],
            axis=0,
        )
        horizontal_coeffs, background_coeffs, row_count = _solve_basis_step(
            reference_model_arr,
            target_model_arr,
            variance_arr,
            mask_arr,
            horizontal_step_basis,
            background_degree=background_degree,
        )
        horizontal_kernel = np.tensordot(
            horizontal_coeffs,
            horizontal_basis,
            axes=(0, 0),
        )
        horizontal_kernel, horizontal_scale = _normalize_line_profile(
            horizontal_kernel
        )
        horizontal_coeffs = horizontal_coeffs / horizontal_scale

        vertical_step_basis = np.stack(
            [
                np.outer(basis_col, horizontal_kernel)
                for basis_col in vertical_basis
            ],
            axis=0,
        )
        vertical_coeffs, background_coeffs, row_count = _solve_basis_step(
            reference_model_arr,
            target_model_arr,
            variance_arr,
            mask_arr,
            vertical_step_basis,
            background_degree=background_degree,
        )
        vertical_kernel = np.tensordot(
            vertical_coeffs,
            vertical_basis,
            axes=(0, 0),
        )
        kernel = np.outer(vertical_kernel, horizontal_kernel)
        if flux_conserve:
            kernel_sum = float(kernel.sum())
            if abs(kernel_sum) < 1e-12:
                raise ValueError(
                    "separable fit produced a near-zero-sum kernel"
                )
            vertical_coeffs = vertical_coeffs / kernel_sum
            vertical_kernel = vertical_kernel / kernel_sum
            kernel = kernel / kernel_sum

        background = evaluate_background_model(
            target_arr.shape,
            background_coeffs,
            degree=background_degree,
        )
        matched = _fftconvolve_same(reference_model_arr, kernel) + background
        residual = target_model_arr - matched
        residual_fit = residual[mask_arr]
        chi2 = float(np.sum((residual_fit**2) / variance_arr[mask_arr]))
        if last_chi2 is not None:
            improvement = abs(last_chi2 - chi2) / max(abs(last_chi2), 1e-12)
            if improvement <= tolerance:
                converged = True
                break
        last_chi2 = chi2

    output_invalid_mask = invalid_pixel_mask | ~boundary_mask
    matched[output_invalid_mask] = np.nan
    residual[output_invalid_mask] = np.nan
    dof = int(row_count) - int(
        horizontal_coeffs.size + vertical_coeffs.size + background_coeffs.size
    )
    return SeparableKernelFitResult(
        kernel=kernel,
        matched=matched,
        residual=residual,
        fit_mask=mask_arr,
        background=background,
        background_coefficients=background_coeffs,
        horizontal_kernel=horizontal_kernel,
        vertical_kernel=vertical_kernel,
        horizontal_coefficients=horizontal_coeffs,
        vertical_coefficients=vertical_coeffs,
        horizontal_basis=horizontal_basis,
        vertical_basis=vertical_basis,
        chi2=chi2,
        dof=dof,
        fit_pixel_count=int(row_count),
        iterations=iteration + 1,
        converged=converged,
        flux_conserve=flux_conserve,
    )


def background_design(
    image_shape: tuple[int, int], degree: int
) -> np.ndarray:
    """Build a centered polynomial background design tensor.

    Parameters
    ----------
    image_shape
        Output ``(height, width)``.
    degree
        Maximum non-negative total polynomial degree.

    Returns
    -------
    numpy.ndarray
        Basis tensor with one leading axis per polynomial term.
    """

    if degree < 0:
        raise ValueError("background degree must be non-negative")

    y_coords, x_coords = _normalized_coordinates(image_shape)
    terms = []
    for deg_x, deg_y in triangular_degree_pairs(degree):
        terms.append((x_coords**deg_x) * (y_coords**deg_y))
    return np.stack(terms, axis=0)


def _background_design_cupy(
    image_shape: tuple[int, int],
    degree: int,
    *,
    cp: Any,
) -> Any:
    if degree < 0:
        raise ValueError("background degree must be non-negative")

    height, width = image_shape
    y = cp.linspace(-1.0, 1.0, num=height, dtype=cp.float64)
    x = cp.linspace(-1.0, 1.0, num=width, dtype=cp.float64)
    y_coords, x_coords = cp.meshgrid(y, x, indexing="ij")
    terms = []
    for deg_x, deg_y in triangular_degree_pairs(degree):
        terms.append((x_coords**deg_x) * (y_coords**deg_y))
    return cp.stack(terms, axis=0)


def evaluate_background_model(
    image_shape: tuple[int, int],
    coefficients: np.ndarray,
    *,
    degree: int,
) -> np.ndarray:
    """Evaluate a polynomial differential-background image.

    Parameters
    ----------
    image_shape
        Output ``(height, width)``.
    coefficients
        Coefficients ordered as :func:`triangular_degree_pairs`.
    degree
        Maximum total polynomial degree.

    Returns
    -------
    numpy.ndarray
        Evaluated two-dimensional background image.
    """

    if coefficients.size == 0:
        return np.zeros(image_shape, dtype=np.float64)
    design = background_design(image_shape, degree)
    return np.tensordot(coefficients, design, axes=(0, 0))


def _build_raw_basis_kernels(
    kernel_shape: tuple[int, int],
    components: Sequence[GaussianBasisComponent],
) -> tuple[np.ndarray, tuple[BasisTerm, ...]]:
    u_coords, v_coords = _kernel_coordinates(kernel_shape)
    kernels = []
    terms: list[BasisTerm] = []
    for component_index, component in enumerate(components):
        if component.sigma <= 0:
            raise ValueError("basis sigma must be positive")
        gaussian = np.exp(
            -((u_coords**2 + v_coords**2) / (2.0 * component.sigma**2))
        )
        for deg_u, deg_v in triangular_degree_pairs(component.degree):
            kernel = gaussian * (u_coords**deg_u) * (v_coords**deg_v)
            kernels.append(kernel)
            terms.append(
                BasisTerm(
                    component_index=component_index,
                    sigma=component.sigma,
                    poly_u_degree=deg_u,
                    poly_v_degree=deg_v,
                    zero_sum=abs(float(kernel.sum())) < 1e-12,
                )
            )
    return np.stack(kernels, axis=0), tuple(terms)


def _build_line_basis(
    length: int,
    components: Sequence[GaussianBasisComponent],
) -> np.ndarray:
    if length <= 0:
        raise ValueError("line basis length must be positive")
    x_coords = np.arange(length, dtype=np.float64) - length // 2
    basis_rows = []
    for component in components:
        if not isinstance(component.sigma, Real) or isinstance(
            component.sigma,
            bool,
        ):
            raise ValueError("basis sigma must be a real number")
        if not np.isfinite(component.sigma) or component.sigma <= 0:
            raise ValueError("basis sigma must be positive")
        if not isinstance(component.degree, Integral) or isinstance(
            component.degree,
            bool,
        ):
            raise ValueError("basis degree must be an integer")
        if component.degree < 0:
            raise ValueError("basis degree must be non-negative")
        gaussian = np.exp(-(x_coords**2) / (2.0 * component.sigma**2))
        for degree in range(component.degree + 1):
            basis_rows.append(gaussian * (x_coords**degree))
    if not basis_rows:
        raise ValueError("components must not be empty")
    return np.stack(basis_rows, axis=0)


def _validate_image_pair(reference: np.ndarray, target: np.ndarray) -> None:
    if reference.ndim != 2 or target.ndim != 2:
        raise ValueError("reference and target must be 2D arrays")
    if reference.shape != target.shape:
        raise ValueError("reference and target must share the same shape")


def _coerce_variance_impl(
    xp: Any, image_shape: tuple[int, int], variance: Any | None
) -> Any:
    if variance is None:
        return xp.ones(image_shape, dtype=xp.float64)
    variance_arr = xp.asarray(variance, dtype=xp.float64)
    if variance_arr.shape != image_shape:
        raise ValueError("variance must match the image shape")
    invalid = xp.isfinite(variance_arr) & (variance_arr <= 0)
    if bool(xp.any(invalid).item()):
        raise ValueError("variance must be strictly positive")
    return variance_arr


def _coerce_variance(
    image_shape: tuple[int, int], variance: np.ndarray | None
) -> np.ndarray:
    return _coerce_variance_impl(np, image_shape, variance)


def _coerce_variance_cupy(
    image_shape: tuple[int, int],
    variance: Any | None,
    *,
    cp: Any,
) -> Any:
    return _coerce_variance_impl(cp, image_shape, variance)


def _coerce_mask(
    image_shape: tuple[int, int],
    fit_mask: np.ndarray | None,
    kernel_shape: tuple[int, int],
) -> np.ndarray:
    valid_mask = _default_fit_mask(image_shape, kernel_shape)
    if not np.any(valid_mask):
        raise ValueError(
            "kernel_shape leaves no valid interior pixels to fit"
        )
    if fit_mask is None:
        return valid_mask
    mask_arr = _coerce_boolean_mask(fit_mask)
    if mask_arr.shape != image_shape:
        raise ValueError("fit_mask must match the image shape")
    mask_arr = mask_arr & valid_mask
    if not np.any(mask_arr):
        raise ValueError("fit_mask must select at least one interior pixel")
    return mask_arr


def _coerce_mask_cupy(
    image_shape: tuple[int, int],
    fit_mask: Any | None,
    kernel_shape: tuple[int, int],
    *,
    cp: Any,
) -> Any:
    valid_mask = _default_fit_mask_cupy(
        image_shape,
        kernel_shape,
        cp=cp,
    )
    if not bool(cp.any(valid_mask).item()):
        raise ValueError(
            "kernel_shape leaves no valid interior pixels to fit"
        )
    if fit_mask is None:
        return valid_mask
    mask_arr = _coerce_boolean_mask_cupy(fit_mask, cp=cp)
    if mask_arr.shape != image_shape:
        raise ValueError("fit_mask must match the image shape")
    mask_arr = mask_arr & valid_mask
    if not bool(cp.any(mask_arr).item()):
        raise ValueError("fit_mask must select at least one interior pixel")
    return mask_arr


def _coerce_boolean_mask(
    mask: np.ndarray, *, name: str = "fit_mask"
) -> np.ndarray:
    mask_arr = np.asarray(mask)
    if mask_arr.dtype == bool:
        return mask_arr
    if not np.issubdtype(mask_arr.dtype, np.number):
        raise ValueError(f"{name} array must be boolean or binary numeric")
    if not np.isfinite(mask_arr).all():
        raise ValueError(f"{name} array must not contain NaN or inf values")
    if not np.all((mask_arr == 0) | (mask_arr == 1)):
        raise ValueError(f"{name} array must contain only 0/1 values")
    return mask_arr.astype(bool)


def _coerce_boolean_mask_cupy(mask: Any, *, cp: Any) -> Any:
    mask_arr = cp.asarray(mask)
    if mask_arr.dtype == cp.bool_:
        return mask_arr
    if not np.issubdtype(mask_arr.dtype, np.number):
        raise ValueError("fit_mask array must be boolean or binary numeric")
    if not bool(cp.isfinite(mask_arr).all().item()):
        raise ValueError("fit_mask array must not contain NaN or inf values")
    is_binary = (mask_arr == 0) | (mask_arr == 1)
    if not bool(cp.all(is_binary).item()):
        raise ValueError("fit_mask array must contain only 0/1 values")
    return mask_arr.astype(cp.bool_)


def _validate_kernel_shape(kernel_shape: tuple[int, int]) -> tuple[int, int]:
    if len(kernel_shape) != 2:
        raise ValueError("kernel_shape must be a 2-tuple")
    height, width = kernel_shape
    if height <= 0 or width <= 0:
        raise ValueError("kernel dimensions must be positive")
    if height % 2 == 0 or width % 2 == 0:
        raise ValueError("kernel_shape must use odd dimensions")
    return int(height), int(width)


def _validate_odd_positive(value: int, *, name: str) -> int:
    number = int(value)
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    if number % 2 == 0:
        raise ValueError(f"{name} must be odd")
    return number


def _kernel_coordinates(
    kernel_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    height, width = kernel_shape
    y = np.arange(height, dtype=np.float64) - height // 2
    x = np.arange(width, dtype=np.float64) - width // 2
    u_coords, v_coords = np.meshgrid(x, y)
    return u_coords, v_coords


def _normalized_coordinates(
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image_shape
    y = np.linspace(-1.0, 1.0, num=height, dtype=np.float64)
    x = np.linspace(-1.0, 1.0, num=width, dtype=np.float64)
    y_coords, x_coords = np.meshgrid(y, x, indexing="ij")
    return y_coords, x_coords


def _default_fit_mask_impl(
    xp: Any, image_shape: tuple[int, int], kernel_shape: tuple[int, int]
) -> Any:
    height, width = image_shape
    kernel_height, kernel_width = kernel_shape
    margin_y = kernel_height // 2
    margin_x = kernel_width // 2
    mask = xp.zeros(image_shape, dtype=xp.bool_)
    mask[
        margin_y : height - margin_y,
        margin_x : width - margin_x,
    ] = True
    return mask


def _default_fit_mask(
    image_shape: tuple[int, int], kernel_shape: tuple[int, int]
) -> np.ndarray:
    return _default_fit_mask_impl(np, image_shape, kernel_shape)


def _default_fit_mask_cupy(
    image_shape: tuple[int, int],
    kernel_shape: tuple[int, int],
    *,
    cp: Any,
) -> Any:
    return _default_fit_mask_impl(cp, image_shape, kernel_shape)


def _invalid_input_mask(
    reference: np.ndarray,
    target: np.ndarray,
    variance: np.ndarray,
    kernel_shape: tuple[int, int],
) -> np.ndarray:
    center_invalid = ~(np.isfinite(target) & np.isfinite(variance))
    reference_patch_invalid = _reference_patch_invalid_mask(
        reference,
        kernel_shape,
    )
    return center_invalid | reference_patch_invalid


def _invalid_input_mask_cupy(
    reference: Any,
    target: Any,
    variance: Any,
    kernel_shape: tuple[int, int],
    *,
    cp: Any,
) -> Any:
    center_invalid = ~(cp.isfinite(target) & cp.isfinite(variance))
    reference_patch_invalid = _reference_patch_invalid_mask_cupy(
        reference,
        kernel_shape,
        cp=cp,
    )
    return center_invalid | reference_patch_invalid


def _reference_patch_invalid_mask(
    reference: np.ndarray,
    kernel_shape: tuple[int, int],
) -> np.ndarray:
    from scipy.signal import convolve2d

    invalid_reference = ~np.isfinite(reference)
    if not np.any(invalid_reference):
        return np.zeros(reference.shape, dtype=bool)

    kernel = np.ones(kernel_shape, dtype=np.int16)
    counts = convolve2d(
        invalid_reference.astype(np.int16),
        kernel,
        mode="same",
    )
    return counts > 0


def _reference_patch_invalid_mask_cupy(
    reference: Any,
    kernel_shape: tuple[int, int],
    *,
    cp: Any,
) -> Any:
    invalid_reference = ~cp.isfinite(reference)
    if not bool(cp.any(invalid_reference).item()):
        return cp.zeros(reference.shape, dtype=cp.bool_)

    try:
        from cupyx.scipy.ndimage import maximum_filter
    except ImportError as exc:
        raise ImportError(
            "solve_constant_kernel_device requires cupyx.scipy.ndimage"
        ) from exc
    counts = maximum_filter(
        invalid_reference.astype(cp.uint8),
        size=kernel_shape,
        mode="constant",
        cval=0,
    )
    return counts > 0


def _accumulate_normal_equations(
    reference: np.ndarray,
    target: np.ndarray,
    variance: np.ndarray,
    mask: np.ndarray,
    basis_kernels: np.ndarray,
    background_terms: np.ndarray,
    *,
    chunk_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray, int]:
    kernel_height, kernel_width = basis_kernels.shape[1:]
    margin_y = kernel_height // 2
    margin_x = kernel_width // 2
    basis_flat = basis_kernels[:, ::-1, ::-1].reshape(
        basis_kernels.shape[0],
        -1,
    )
    patch_view = sliding_window_view(reference, (kernel_height, kernel_width))
    ys, xs = np.nonzero(mask)
    column_count = basis_kernels.shape[0] + background_terms.shape[0]
    gram_matrix = np.zeros((column_count, column_count), dtype=np.float64)
    rhs_vector = np.zeros(column_count, dtype=np.float64)
    row_count = 0

    for start in range(0, ys.size, chunk_size):
        stop = min(start + chunk_size, ys.size)
        y_chunk = ys[start:stop]
        x_chunk = xs[start:stop]
        patch_chunk = patch_view[
            y_chunk - margin_y,
            x_chunk - margin_x,
        ].reshape(stop - start, -1)
        basis_chunk = patch_chunk @ basis_flat.T
        background_chunk = background_terms[:, y_chunk, x_chunk].T
        design_chunk = np.concatenate(
            (basis_chunk, background_chunk),
            axis=1,
        )
        weights = 1.0 / np.sqrt(variance[y_chunk, x_chunk])
        weighted_design = design_chunk * weights[:, None]
        weighted_target = target[y_chunk, x_chunk] * weights
        gram_matrix += weighted_design.T @ weighted_design
        rhs_vector += weighted_design.T @ weighted_target
        row_count += int(y_chunk.size)

    return gram_matrix, rhs_vector, row_count


def _normalize_backend(backend: str) -> str:
    return resolve_backend(backend)


def resolve_backend(backend: str = "auto") -> str:
    """Resolve an xPois backend name.

    Parameters
    ----------
    backend
        ``auto``, ``cupy``, ``numba-cuda``, ``cpu``, or ``cutile``. The
        ``numba_cuda`` spelling is accepted as an input alias.

    Returns
    -------
    str
        Explicit backend name. ``auto`` prefers CuPy, then Numba-CUDA, and
        finally CPU. cuTile is never selected automatically.
    """

    normalized = str(backend).strip().lower()
    if normalized == "numba_cuda":
        normalized = "numba-cuda"
    if normalized not in _SUPPORTED_BACKENDS_SET:
        choices = ", ".join(SUPPORTED_BACKENDS)
        raise ValueError(f"backend must be one of: {choices}")
    if normalized == "auto":
        for candidate in AUTO_BACKEND_PREFERENCE:
            if _backend_available(candidate):
                return candidate
        return "cpu"
    return normalized


def _backend_available(backend: str) -> bool:
    if backend == "cpu":
        return True
    if backend == "cupy":
        try:
            import cupy as cp

            return cp.cuda.runtime.getDeviceCount() > 0
        except Exception:
            return False
    if backend == "numba-cuda":
        try:
            from numba import cuda

            return bool(cuda.is_available())
        except Exception:
            return False
    return False


def _load_cupy() -> tuple[Any, Any]:
    try:
        import cupy as cp
        from cupyx.scipy.signal import fftconvolve as cupy_fftconvolve
    except ImportError as exc:
        raise ImportError(
            "backend='cupy' requires CuPy and cupyx.scipy.signal; "
            "run 'uv sync --extra gpu' for development or install "
            "'cuphoton[gpu]'"
        ) from exc
    return cp, cupy_fftconvolve


def _active_cupy_device_id(cp: Any) -> int:
    message = (
        "solve_constant_kernel_device requires at least one visible "
        "CUDA device"
    )
    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:
        raise RuntimeError(message) from exc
    if device_count < 1:
        raise RuntimeError(message)
    try:
        return int(cp.cuda.runtime.getDevice())
    except Exception as exc:
        raise RuntimeError(message) from exc


def _validate_cupy_input_device(
    value: Any,
    *,
    name: str,
    active_device_id: int,
    cp: Any,
) -> None:
    if value is None or not isinstance(value, cp.ndarray):
        return
    device_id = int(value.device.id)
    if device_id != active_device_id:
        raise ValueError(
            f"{name} is on CUDA device {device_id}; "
            f"active CUDA device is {active_device_id}"
        )


def _accumulate_normal_equations_cupy(
    reference: np.ndarray,
    target: np.ndarray,
    variance: np.ndarray,
    mask: np.ndarray,
    basis_kernels: np.ndarray,
    background_terms: np.ndarray,
    *,
    chunk_size: int = 65536,
) -> tuple[np.ndarray, np.ndarray, int]:
    cp, _ = _load_cupy()
    gram_matrix, rhs_vector, row_count = (
        _accumulate_normal_equations_cupy_device(
            cp.asarray(reference, dtype=cp.float64),
            cp.asarray(target, dtype=cp.float64),
            cp.asarray(variance, dtype=cp.float64),
            cp.asarray(mask, dtype=cp.bool_),
            cp.asarray(basis_kernels, dtype=cp.float64),
            cp.asarray(background_terms, dtype=cp.float64),
            cp=cp,
            chunk_size=chunk_size,
        )
    )
    return cp.asnumpy(gram_matrix), cp.asnumpy(rhs_vector), row_count


def _accumulate_normal_equations_cupy_device(
    reference: Any,
    target: Any,
    variance: Any,
    mask: Any,
    basis_kernels: Any,
    background_terms: Any,
    *,
    cp: Any,
    chunk_size: int = 65536,
) -> tuple[Any, Any, int]:
    try:
        cupy_sliding_window_view = cp.lib.stride_tricks.sliding_window_view
    except AttributeError as exc:
        raise ImportError(
            "solve_constant_kernel_device requires a CuPy build with "
            "cp.lib.stride_tricks.sliding_window_view"
        ) from exc

    kernel_height, kernel_width = basis_kernels.shape[1:]
    margin_y = kernel_height // 2
    margin_x = kernel_width // 2
    basis_flat = basis_kernels[:, ::-1, ::-1].reshape(
        basis_kernels.shape[0],
        -1,
    )
    patch_view = cupy_sliding_window_view(
        reference,
        (kernel_height, kernel_width),
    )
    ys, xs = cp.nonzero(mask)
    column_count = int(basis_kernels.shape[0] + background_terms.shape[0])
    gram_matrix = cp.zeros(
        (column_count, column_count),
        dtype=cp.float64,
    )
    rhs_vector = cp.zeros(column_count, dtype=cp.float64)
    row_count = int(ys.size)

    for start in range(0, row_count, chunk_size):
        stop = min(start + chunk_size, row_count)
        y_chunk = ys[start:stop]
        x_chunk = xs[start:stop]
        patch_chunk = patch_view[
            y_chunk - margin_y,
            x_chunk - margin_x,
        ].reshape(stop - start, -1)
        basis_chunk = patch_chunk @ basis_flat.T
        background_chunk = background_terms[:, y_chunk, x_chunk].T
        design_chunk = cp.concatenate(
            (basis_chunk, background_chunk),
            axis=1,
        )
        weights = 1.0 / cp.sqrt(variance[y_chunk, x_chunk])
        weighted_design = design_chunk * weights[:, None]
        weighted_target = target[y_chunk, x_chunk] * weights
        gram_matrix += weighted_design.T @ weighted_design
        rhs_vector += weighted_design.T @ weighted_target

    return gram_matrix, rhs_vector, row_count


def _load_numba_cuda() -> Any:
    try:
        from numba import cuda
    except ImportError as exc:
        raise ImportError(
            "backend='numba-cuda' requires Numba with CUDA support; "
            "run 'uv sync --extra gpu' for development or install "
            "'cuphoton[gpu]'"
        ) from exc
    if not cuda.is_available():
        raise ImportError(
            "backend='numba-cuda' requires a usable CUDA device"
        )
    return cuda


def _get_numba_accumulate_kernel() -> tuple[Any, Any]:
    global _NUMBA_ACCUMULATE_KERNEL
    cuda = _load_numba_cuda()
    if _NUMBA_ACCUMULATE_KERNEL is not None:
        return cuda, _NUMBA_ACCUMULATE_KERNEL

    from numba import float64

    @cuda.jit
    def kernel(
        reference,
        target,
        variance,
        y_indices,
        x_indices,
        basis_kernels,
        background_terms,
        partial_gram,
        partial_rhs,
        row_count,
        rows_per_block,
        column_count,
        basis_count,
        background_count,
        kernel_height,
        kernel_width,
    ):
        shared_gram = cuda.shared.array(
            shape=4096,
            dtype=float64,
        )
        shared_rhs = cuda.shared.array(
            shape=64,
            dtype=float64,
        )
        design = cuda.local.array(
            shape=64,
            dtype=float64,
        )
        thread_id = cuda.threadIdx.x
        block_start = cuda.blockIdx.x * rows_per_block
        block_stop = block_start + rows_per_block
        if block_stop > row_count:
            block_stop = row_count

        for flat_index in range(
            thread_id,
            column_count * column_count,
            cuda.blockDim.x,
        ):
            shared_gram[flat_index] = 0.0
        for column in range(thread_id, column_count, cuda.blockDim.x):
            shared_rhs[column] = 0.0
        cuda.syncthreads()

        margin_y = kernel_height // 2
        margin_x = kernel_width // 2
        for row_number in range(
            block_start + thread_id,
            block_stop,
            cuda.blockDim.x,
        ):
            y = y_indices[row_number]
            x = x_indices[row_number]
            for column in range(column_count):
                design[column] = 0.0

            for basis_index in range(basis_count):
                value = 0.0
                for ky in range(kernel_height):
                    ref_y = y + ky - margin_y
                    basis_y = kernel_height - 1 - ky
                    for kx in range(kernel_width):
                        ref_x = x + kx - margin_x
                        basis_x = kernel_width - 1 - kx
                        value += (
                            reference[ref_y, ref_x]
                            * basis_kernels[
                                basis_index,
                                basis_y,
                                basis_x,
                            ]
                        )
                design[basis_index] = value

            for background_index in range(background_count):
                design[basis_count + background_index] = background_terms[
                    background_index,
                    y,
                    x,
                ]

            inv_var = 1.0 / variance[y, x]
            target_scaled = target[y, x] * inv_var
            for left in range(column_count):
                left_value = design[left]
                cuda.atomic.add(
                    shared_rhs,
                    left,
                    left_value * target_scaled,
                )
                for right in range(left, column_count):
                    cuda.atomic.add(
                        shared_gram,
                        left * column_count + right,
                        left_value * design[right] * inv_var,
                    )

        cuda.syncthreads()
        block = cuda.blockIdx.x
        for flat_index in range(
            thread_id,
            column_count * column_count,
            cuda.blockDim.x,
        ):
            row = flat_index // column_count
            column = flat_index - row * column_count
            partial_gram[block, row, column] = shared_gram[flat_index]
        for column in range(thread_id, column_count, cuda.blockDim.x):
            partial_rhs[block, column] = shared_rhs[column]

    _NUMBA_ACCUMULATE_KERNEL = kernel
    return cuda, kernel


def _accumulate_normal_equations_numba_cuda(
    reference: np.ndarray,
    target: np.ndarray,
    variance: np.ndarray,
    mask: np.ndarray,
    basis_kernels: np.ndarray,
    background_terms: np.ndarray,
    *,
    rows_per_block: int = _NUMBA_ROWS_PER_BLOCK,
    threads_per_block: int = _NUMBA_THREADS_PER_BLOCK,
) -> tuple[np.ndarray, np.ndarray, int]:
    cuda, kernel = _get_numba_accumulate_kernel()
    column_count = int(basis_kernels.shape[0] + background_terms.shape[0])
    if column_count > _NUMBA_MAX_COLUMNS:
        raise ValueError(
            "backend='numba-cuda' supports at most "
            f"{_NUMBA_MAX_COLUMNS} fit columns; got {column_count}"
        )

    ys, xs = np.nonzero(mask)
    row_count = int(ys.size)
    block_count = math.ceil(row_count / rows_per_block)
    if block_count == 0:
        return (
            np.zeros((column_count, column_count), dtype=np.float64),
            np.zeros(column_count, dtype=np.float64),
            0,
        )

    reference_device = cuda.to_device(
        np.ascontiguousarray(reference, dtype=np.float64)
    )
    target_device = cuda.to_device(
        np.ascontiguousarray(target, dtype=np.float64)
    )
    variance_device = cuda.to_device(
        np.ascontiguousarray(variance, dtype=np.float64)
    )
    y_device = cuda.to_device(np.ascontiguousarray(ys, dtype=np.int64))
    x_device = cuda.to_device(np.ascontiguousarray(xs, dtype=np.int64))
    basis_device = cuda.to_device(
        np.ascontiguousarray(basis_kernels, dtype=np.float64)
    )
    background_device = cuda.to_device(
        np.ascontiguousarray(background_terms, dtype=np.float64)
    )
    partial_gram_device = cuda.device_array(
        (block_count, column_count, column_count),
        dtype=np.float64,
    )
    partial_rhs_device = cuda.device_array(
        (block_count, column_count),
        dtype=np.float64,
    )
    kernel_height, kernel_width = basis_kernels.shape[1:]

    kernel[block_count, threads_per_block](
        reference_device,
        target_device,
        variance_device,
        y_device,
        x_device,
        basis_device,
        background_device,
        partial_gram_device,
        partial_rhs_device,
        row_count,
        rows_per_block,
        column_count,
        int(basis_kernels.shape[0]),
        int(background_terms.shape[0]),
        int(kernel_height),
        int(kernel_width),
    )
    cuda.synchronize()

    gram_matrix = partial_gram_device.copy_to_host().sum(axis=0)
    rhs_vector = partial_rhs_device.copy_to_host().sum(axis=0)
    gram_matrix = np.triu(gram_matrix) + np.triu(gram_matrix, k=1).T
    return gram_matrix, rhs_vector, row_count


def _load_cutile() -> tuple[Any, Any]:
    global ct
    try:
        import cupy as cp

        if ct is None:
            import cuda.tile as cutile

            ct = cutile
    except Exception as exc:
        raise ImportError(
            "backend='cutile' requires CuPy and the cuda-tile package; "
            "run 'uv sync --extra cutile' for development or install "
            "'cuphoton[cutile]'"
        ) from exc
    return cp, ct


@lru_cache(maxsize=1)
def _cutile_mma_normal_equations_kernel() -> Any:
    _, ct = _load_cutile()

    @ct.kernel
    def kernel(
        reference,
        target,
        variance,
        y_indices,
        x_indices,
        padded_basis,
        background_terms,
        partial_normal,
        # A runtime row count lets masks of any size share one compiled
        # kernel; only structural parameters are compile-time constants.
        row_count,
        rows_per_tile: ct.Constant[int],
        k_tile: ct.Constant[int],
        normal_width: ct.Constant[int],
        column_count: ct.Constant[int],
        basis_count: ct.Constant[int],
        kernel_height: ct.Constant[int],
        kernel_width: ct.Constant[int],
        k_tile_count: ct.Constant[int],
    ):
        block = ct.bid(0)
        local_row = ct.arange(rows_per_tile, dtype=ct.int32)
        row_number = block * rows_per_tile + local_row
        valid_row = row_number < row_count
        y = ct.gather(y_indices, row_number, padding_value=0)
        x = ct.gather(x_indices, row_number, padding_value=0)
        y_column = ct.reshape(y, (rows_per_tile, 1))
        x_column = ct.reshape(x, (rows_per_tile, 1))
        valid_row_column = ct.reshape(valid_row, (rows_per_tile, 1))
        margin_y = kernel_height // 2
        margin_x = kernel_width // 2
        kernel_size = kernel_height * kernel_width
        # Project each image patch onto every basis column once. The former
        # kernel repeated this convolution for every Gram-matrix pair.
        design = ct.zeros(
            (rows_per_tile, normal_width),
            dtype=ct.float64,
        )

        for tile_k in range(k_tile_count):
            flat_kernel_index = tile_k * k_tile + ct.arange(
                k_tile,
                dtype=ct.int32,
            )
            kernel_y = flat_kernel_index // kernel_width
            kernel_x = flat_kernel_index - kernel_y * kernel_width
            ref_y = y_column + ct.reshape(kernel_y, (1, k_tile)) - margin_y
            ref_x = x_column + ct.reshape(kernel_x, (1, k_tile)) - margin_x
            valid_kernel = flat_kernel_index < kernel_size
            gather_mask = valid_row_column & ct.reshape(
                valid_kernel,
                (1, k_tile),
            )
            patch = ct.gather(
                reference,
                (ref_y, ref_x),
                mask=gather_mask,
                padding_value=0.0,
            )
            basis_tile = ct.load(
                padded_basis,
                index=(tile_k, 0),
                shape=(k_tile, normal_width),
            )
            design = ct.mma(patch, basis_tile, design)

        columns = ct.arange(normal_width, dtype=ct.int32)
        column_row = ct.reshape(columns, (1, normal_width))
        background_index = column_row - basis_count
        background = ct.gather(
            background_terms,
            (background_index, y_column, x_column),
            padding_value=0.0,
        )
        design = ct.where(column_row < basis_count, design, background)
        target_value = ct.gather(
            target,
            (y_column, x_column),
            padding_value=0.0,
        )
        design = ct.where(column_row == column_count, target_value, design)
        design = ct.where(
            column_row <= column_count,
            design,
            ct.zeros(
                (rows_per_tile, normal_width),
                dtype=ct.float64,
            ),
        )
        variance_value = ct.gather(
            variance,
            (y_column, x_column),
            padding_value=1.0,
        )
        weighted = design / ct.sqrt(variance_value)
        weighted = ct.where(
            valid_row_column,
            weighted,
            ct.zeros(
                (rows_per_tile, normal_width),
                dtype=ct.float64,
            ),
        )
        # Appending the weighted target makes one MMA produce both D.T @ D
        # and D.T @ y. Padded rows and columns are explicitly zero.
        normal = ct.mma(
            ct.transpose(weighted),
            weighted,
            ct.zeros((normal_width, normal_width), dtype=ct.float64),
        )
        normal_rows = ct.reshape(
            ct.arange(normal_width, dtype=ct.int32),
            (normal_width, 1),
        )
        normal_columns = ct.reshape(
            ct.arange(normal_width, dtype=ct.int32),
            (1, normal_width),
        )
        ct.scatter(
            partial_normal,
            (block, normal_rows, normal_columns),
            normal,
            mask=(normal_rows < column_count)
            & (normal_columns <= column_count),
        )

    return kernel


@lru_cache(maxsize=4)
def _cached_cutile_padded_basis(
    device_id: int,
    basis_shape: tuple[int, int, int],
    normal_width: int,
    basis_values: bytes,
) -> Any:
    """Retain at most four immutable padded uploads across bases/devices."""
    cp, _ = _load_cutile()
    basis = np.frombuffer(basis_values, dtype=np.float64).reshape(basis_shape)
    kernel_size = basis_shape[1] * basis_shape[2]
    k_tile = _CUTILE_MMA_K_TILE
    padded_rows = math.ceil(kernel_size / k_tile) * k_tile
    padded = np.zeros((padded_rows, normal_width), dtype=np.float64)
    padded[:kernel_size, : basis_shape[0]] = (
        basis[:, ::-1, ::-1].reshape(basis_shape[0], -1).T
    )
    with cp.cuda.Device(device_id):
        # Finish the miss's upload before publishing it to another stream.
        # Kernels only read this private cached array.
        return cp.asarray(padded, blocking=True)


def _accumulate_normal_equations_cutile(
    reference: np.ndarray,
    target: np.ndarray,
    variance: np.ndarray,
    mask: np.ndarray,
    basis_kernels: np.ndarray,
    background_terms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    cp, ct = _load_cutile()
    reference_gpu = cp.asarray(reference, dtype=cp.float64)
    target_gpu = cp.asarray(target, dtype=cp.float64)
    variance_gpu = cp.asarray(variance, dtype=cp.float64)
    mask_gpu = cp.asarray(mask, dtype=cp.bool_)
    background_gpu = cp.asarray(background_terms, dtype=cp.float64)
    ys, xs = cp.nonzero(mask_gpu)
    ys = ys.astype(cp.int32, copy=False)
    xs = xs.astype(cp.int32, copy=False)
    row_count = int(ys.size)
    column_count = int(basis_kernels.shape[0] + background_terms.shape[0])
    rows_per_tile = (
        _CUTILE_MMA_LARGE_ROWS_PER_TILE
        if row_count >= _CUTILE_MMA_LARGE_ROW_THRESHOLD
        else _CUTILE_MMA_SMALL_ROWS_PER_TILE
    )
    block_count = math.ceil(row_count / rows_per_tile)
    if block_count == 0:
        return (
            np.zeros((column_count, column_count), dtype=np.float64),
            np.zeros(column_count, dtype=np.float64),
            0,
        )

    kernel_height, kernel_width = basis_kernels.shape[1:]
    # cuda.tile MMA dimensions are powers of two. Reserve the next column
    # after the design for the weighted target.
    normal_width = 1 << column_count.bit_length()
    kernel_size = int(kernel_height * kernel_width)
    k_tile = _CUTILE_MMA_K_TILE
    k_tile_count = math.ceil(kernel_size / k_tile)
    padded_basis = _cached_cutile_padded_basis(
        int(cp.cuda.runtime.getDevice()),
        basis_kernels.shape,
        normal_width,
        np.asarray(basis_kernels, dtype=np.float64).tobytes(),
    )
    partial_normal = cp.empty(
        (block_count, column_count, column_count + 1),
        dtype=cp.float64,
    )
    stream = cp.cuda.get_current_stream()
    try:
        ct.launch(
            stream,
            (block_count, 1, 1),
            _cutile_mma_normal_equations_kernel(),
            (
                reference_gpu,
                target_gpu,
                variance_gpu,
                ys,
                xs,
                padded_basis,
                background_gpu,
                partial_normal,
                row_count,
                int(rows_per_tile),
                int(k_tile),
                int(normal_width),
                int(column_count),
                int(basis_kernels.shape[0]),
                int(kernel_height),
                int(kernel_width),
                int(k_tile_count),
            ),
        )
    except Exception as exc:
        raise RuntimeError(
            "backend='cutile' failed to compile or launch its cuda.tile "
            "accumulator; check that cuda-tile and tileiras versions match"
        ) from exc
    normal = partial_normal.sum(axis=0)
    gram_matrix = cp.asnumpy(normal[:, :column_count])
    rhs_vector = cp.asnumpy(normal[:, column_count])
    return gram_matrix, rhs_vector, row_count


def _matched_image(
    reference: np.ndarray,
    kernel: np.ndarray,
    background: np.ndarray,
    *,
    backend: str,
) -> np.ndarray:
    if backend in {"cupy", "cutile"}:
        return _matched_image_cupy(reference, kernel, background)
    return _fftconvolve_same(reference, kernel) + background


def _matched_image_cupy(
    reference: np.ndarray,
    kernel: np.ndarray,
    background: np.ndarray,
) -> np.ndarray:
    cp, cupy_fftconvolve = _load_cupy()
    reference_gpu = cp.asarray(reference, dtype=cp.float64)
    kernel_gpu = cp.asarray(kernel, dtype=cp.float64)
    background_gpu = cp.asarray(background, dtype=cp.float64)
    matched_gpu = (
        cupy_fftconvolve(reference_gpu, kernel_gpu, mode="same")
        + background_gpu
    )
    return cp.asnumpy(matched_gpu)


def _solve_basis_step(
    reference: np.ndarray,
    target: np.ndarray,
    variance: np.ndarray,
    mask: np.ndarray,
    basis_kernels: np.ndarray,
    *,
    background_degree: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    background_terms = background_design(target.shape, background_degree)
    gram_matrix, rhs_vector, row_count = _accumulate_normal_equations(
        reference,
        target,
        variance,
        mask,
        basis_kernels,
        background_terms,
    )
    column_count = gram_matrix.shape[0]
    if row_count < column_count:
        raise ValueError(
            "fit region is underdetermined for the requested basis: "
            f"{row_count} equations for {column_count} coefficients"
        )
    try:
        condition_number = np.linalg.cond(gram_matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "fit normal equations are singular for the requested basis"
        ) from exc
    if not np.isfinite(condition_number) or condition_number > 1e12:
        raise ValueError(
            "fit normal equations are ill-conditioned for the requested basis"
        )
    try:
        coefficients = np.linalg.solve(gram_matrix, rhs_vector)
    except np.linalg.LinAlgError as exc:
        raise ValueError("fit normal equations could not be solved") from exc

    kernel_coeff_count = basis_kernels.shape[0]
    return (
        coefficients[:kernel_coeff_count],
        coefficients[kernel_coeff_count:],
        row_count,
    )


def _normalize_line_profile(profile: np.ndarray) -> tuple[np.ndarray, float]:
    scale = float(np.linalg.norm(profile))
    if scale <= 1e-12:
        raise ValueError("separable fit produced a near-zero line profile")
    return np.asarray(profile / scale, dtype=np.float64), scale


def _fftconvolve_same(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    from scipy.signal import fftconvolve

    return fftconvolve(image, kernel, mode="same")
