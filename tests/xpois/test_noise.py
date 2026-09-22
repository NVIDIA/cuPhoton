# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from dataclasses import asdict

import numpy as np
import pytest
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import fftconvolve

from cuphoton.xpois.noise import (
    ConstantKernelNoiseResult,
    standardize_constant_kernel_residual,
)
from cuphoton.xpois.ois import (
    GaussianBasisComponent,
    _fftconvolve_same,
    build_gaussian_polynomial_basis,
    solve_constant_kernel,
)


def test_delta_kernel_adds_target_and_reference_variance() -> None:
    residual = np.full((4, 5), 6.0)
    target_variance = np.full_like(residual, 5.0)
    reference_variance = np.full_like(residual, 4.0)

    result = standardize_constant_kernel_residual(
        residual,
        kernel=np.asarray([[2.0]]),
        target_variance=target_variance,
        reference_variance=reference_variance,
    )

    assert result.valid_mask.all()
    assert np.allclose(
        result.propagated_reference_variance, 16.0, rtol=1.0e-12, atol=0.0
    )
    assert np.allclose(
        result.marginal_residual_variance, 21.0, rtol=1.0e-12, atol=0.0
    )
    assert np.allclose(result.standardized_residual, 6.0 / np.sqrt(21.0))
    assert result.variance_scope == "marginal_diagonal"
    assert result.kernel_treatment == "fixed"
    assert result.covariance_represented is False
    assert result.fit_parameter_uncertainty_represented is False


def test_asymmetric_kernel_matches_explicit_squared_kernel_oracle() -> None:
    shape = (7, 8)
    residual = np.zeros(shape)
    target_variance = np.linspace(1.0, 2.0, np.prod(shape)).reshape(shape)
    reference_variance = np.arange(1.0, np.prod(shape) + 1.0).reshape(shape)
    kernel = np.asarray(
        [
            [0.02, -0.03, 0.05],
            [0.07, 0.11, 0.13],
            [0.17, 0.19, 0.29],
        ]
    )

    result = standardize_constant_kernel_residual(
        residual,
        kernel=kernel,
        target_variance=target_variance,
        reference_variance=reference_variance,
    )

    windows = sliding_window_view(reference_variance, kernel.shape)
    expected = np.einsum(
        "yxij,ij->yx",
        windows,
        kernel[::-1, ::-1] ** 2,
        optimize=True,
    )
    assert np.allclose(
        result.propagated_reference_variance[1:-1, 1:-1], expected
    )
    assert np.isnan(result.propagated_reference_variance[0]).all()
    assert np.isnan(result.propagated_reference_variance[-1]).all()
    assert np.isnan(result.propagated_reference_variance[:, 0]).all()
    assert np.isnan(result.propagated_reference_variance[:, -1]).all()


@pytest.mark.parametrize("sentinel", [1.0e30, np.finfo(np.float64).max])
def test_large_finite_reference_variance_stays_local(sentinel: float) -> None:
    shape = (40, 44)
    reference_variance = np.arange(1.0, np.prod(shape) + 1.0).reshape(shape)
    reference_variance[20, 22] = sentinel
    kernel = np.random.default_rng(34).uniform(-0.3, 0.3, size=(7, 7))

    result = standardize_constant_kernel_residual(
        np.ones(shape),
        kernel=kernel,
        target_variance=np.ones(shape),
        reference_variance=reference_variance,
    )

    windows = sliding_window_view(reference_variance, kernel.shape)
    expected = np.einsum(
        "yxij,ij->yx",
        windows,
        kernel[::-1, ::-1] ** 2,
        optimize=True,
    )
    assert np.isfinite(expected).all()
    assert result.valid_mask[3:-3, 3:-3].all()
    assert np.allclose(
        result.propagated_reference_variance[3:-3, 3:-3],
        expected,
        rtol=1.0e-12,
        atol=0.0,
    )
    assert np.isfinite(result.standardized_residual[3:-3, 3:-3]).all()


def test_overflowing_reference_footprint_is_excluded_locally() -> None:
    shape = (9, 10)
    reference_variance = np.ones(shape)
    reference_variance[4, 5] = np.finfo(np.float64).max

    result = standardize_constant_kernel_residual(
        np.ones(shape),
        kernel=np.full((3, 3), 1.5),
        target_variance=np.ones(shape),
        reference_variance=reference_variance,
    )

    expected_valid = np.zeros(shape, dtype=bool)
    expected_valid[1:-1, 1:-1] = True
    expected_valid[3:6, 4:7] = False
    assert np.array_equal(result.valid_mask, expected_valid)
    assert np.allclose(
        result.propagated_reference_variance[expected_valid],
        9.0 * 1.5**2,
        rtol=1.0e-12,
        atol=0.0,
    )


def test_invalid_reference_footprint_propagates_fail_closed() -> None:
    shape = (9, 10)
    residual = np.ones(shape)
    target_variance = np.ones(shape)
    reference_variance = np.ones(shape)
    reference_variance[4, 5] = np.nan
    target_variance[6, 7] = 0.0
    selected = np.ones(shape, dtype=bool)
    selected[2, 2] = False

    result = standardize_constant_kernel_residual(
        residual,
        kernel=np.ones((3, 3)) / 9.0,
        target_variance=target_variance,
        reference_variance=reference_variance,
        valid_mask=selected,
    )

    expected_valid = np.zeros(shape, dtype=bool)
    expected_valid[1:-1, 1:-1] = True
    expected_valid[3:6, 4:7] = False
    expected_valid[6, 7] = False
    expected_valid[2, 2] = False
    assert np.array_equal(result.valid_mask, expected_valid)
    assert np.isnan(result.standardized_residual[~expected_valid]).all()
    assert np.isfinite(result.standardized_residual[expected_valid]).all()


def test_reference_footprint_validity_matches_window_oracle() -> None:
    shape = (17, 19)
    kernel = np.ones((5, 3)) / 15.0
    reference_variance = np.ones(shape)
    reference_variance[0, 4] = np.nan
    reference_variance[6, 9] = np.inf
    reference_variance[12, 15] = -np.inf
    reference_variance[16, 18] = np.nan

    result = standardize_constant_kernel_residual(
        np.ones(shape),
        kernel=kernel,
        target_variance=np.ones(shape),
        reference_variance=reference_variance,
    )

    finite_windows = sliding_window_view(
        np.isfinite(reference_variance), kernel.shape
    )
    expected_valid = np.zeros(shape, dtype=bool)
    expected_valid[2:-2, 1:-1] = finite_windows.all(axis=(2, 3))
    assert np.array_equal(result.valid_mask, expected_valid)
    assert np.allclose(
        result.propagated_reference_variance[expected_valid], 1.0 / 15.0
    )


def test_spatially_varying_variance_and_zero_reference_noise() -> None:
    shape = (5, 6)
    target_variance = np.arange(1.0, np.prod(shape) + 1.0).reshape(shape)
    residual = np.sqrt(target_variance)

    result = standardize_constant_kernel_residual(
        residual,
        kernel=np.asarray([[3.0]]),
        target_variance=target_variance,
        reference_variance=np.zeros(shape),
    )

    assert np.all(result.propagated_reference_variance == 0.0)
    assert np.allclose(
        result.marginal_residual_variance,
        target_variance,
        rtol=1.0e-12,
        atol=0.0,
    )
    assert np.allclose(result.standardized_residual, 1.0)


def test_scope_metadata_is_fixed_and_serialized() -> None:
    plane = np.ones((3, 3))
    mask = np.ones((3, 3), dtype=bool)
    scope_fields = {
        "variance_scope": "marginal_diagonal",
        "kernel_treatment": "fixed",
        "covariance_represented": False,
        "fit_parameter_uncertainty_represented": False,
    }

    result = ConstantKernelNoiseResult(plane, plane, plane, mask)

    serialized = asdict(result)
    assert {key: serialized[key] for key in scope_fields} == scope_fields
    with pytest.raises(TypeError):
        ConstantKernelNoiseResult(
            plane, plane, plane, mask, covariance_represented=True
        )


def test_inputs_are_not_mutated_and_outputs_are_read_only() -> None:
    residual = np.ones((5, 5))
    kernel = np.ones((3, 3)) / 9.0
    target_variance = np.ones((5, 5))
    reference_variance = np.ones((5, 5))
    selected = np.ones((5, 5), dtype=bool)
    inputs = [residual, kernel, target_variance, reference_variance, selected]
    before = [value.copy() for value in inputs]

    result = standardize_constant_kernel_residual(
        residual,
        kernel=kernel,
        target_variance=target_variance,
        reference_variance=reference_variance,
        valid_mask=selected,
    )

    for value, original in zip(inputs, before, strict=True):
        assert np.array_equal(value, original)
    for value in (
        result.propagated_reference_variance,
        result.marginal_residual_variance,
        result.standardized_residual,
        result.valid_mask,
    ):
        assert not value.flags.writeable


@pytest.mark.parametrize(
    ("argument", "value", "error", "message"),
    [
        ("residual", np.ones(5), ValueError, "two-dimensional"),
        ("residual", np.zeros((0, 5)), ValueError, "must not be empty"),
        (
            "residual",
            np.ones((5, 5), dtype=bool),
            TypeError,
            "real numeric",
        ),
        ("kernel", np.ones((2, 3)), ValueError, "dimensions must be odd"),
        ("kernel", np.ones((7, 7)), ValueError, "must fit within"),
        (
            "kernel",
            np.ones((3, 3), dtype=complex),
            TypeError,
            "real numeric",
        ),
        *[
            (
                argument,
                np.ones(shape, dtype="timedelta64[s]"),
                TypeError,
                f"{argument} must be a real numeric array",
            )
            for argument, shape in (
                ("residual", (5, 5)),
                ("kernel", (3, 3)),
                ("target_variance", (5, 5)),
                ("reference_variance", (5, 5)),
            )
        ],
        (
            "kernel",
            np.asarray([[np.nan]]),
            ValueError,
            "only finite",
        ),
        (
            "kernel",
            np.asarray([[np.finfo(np.float64).max]]),
            ValueError,
            "squared kernel values",
        ),
        (
            "target_variance",
            np.ones((4, 5)),
            ValueError,
            "must match residual",
        ),
        (
            "reference_variance",
            np.ones((5, 4)),
            ValueError,
            "must match residual",
        ),
        (
            "reference_variance",
            -np.ones((5, 5)),
            ValueError,
            "must be nonnegative",
        ),
        (
            "valid_mask",
            np.ones((4, 5), dtype=bool),
            ValueError,
            "valid_mask must match residual shape",
        ),
        (
            "valid_mask",
            np.full((5, 5), 0.5),
            ValueError,
            "valid_mask array must contain only 0/1",
        ),
        (
            "valid_mask",
            np.full((5, 5), "y"),
            ValueError,
            "valid_mask array must be boolean or binary",
        ),
    ],
)
def test_validation_fails_closed(
    argument: str,
    value: np.ndarray,
    error: type[Exception],
    message: str,
) -> None:
    arguments = {
        "residual": np.ones((5, 5)),
        "kernel": np.ones((3, 3)),
        "target_variance": np.ones((5, 5)),
        "reference_variance": np.ones((5, 5)),
        "valid_mask": np.ones((5, 5), dtype=bool),
    }
    arguments[argument] = value

    with pytest.raises(error, match=message):
        standardize_constant_kernel_residual(
            arguments.pop("residual"), **arguments
        )


def test_binary_valid_mask_matches_boolean_mask() -> None:
    shape = (7, 8)
    selected = np.ones(shape, dtype=bool)
    selected[3, 4] = False
    common = {
        "kernel": np.ones((3, 3)),
        "target_variance": np.ones(shape),
        "reference_variance": np.ones(shape),
    }

    boolean = standardize_constant_kernel_residual(
        np.ones(shape), valid_mask=selected, **common
    )
    binary = standardize_constant_kernel_residual(
        np.ones(shape), valid_mask=selected.astype(np.uint8), **common
    )

    assert np.array_equal(binary.valid_mask, boolean.valid_mask)
    assert not binary.valid_mask[3, 4]
    assert np.count_nonzero(binary.valid_mask) == 29


def test_raises_when_no_valid_output_pixels_remain() -> None:
    with pytest.raises(ValueError, match="no valid output pixels"):
        standardize_constant_kernel_residual(
            np.ones((5, 5)),
            kernel=np.ones((3, 3)),
            target_variance=np.zeros((5, 5)),
            reference_variance=np.ones((5, 5)),
        )


def test_zero_marginal_variance_is_excluded_without_warnings() -> None:
    shape = (5, 5)
    target_variance = np.ones(shape)
    target_variance[2, 2] = 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = standardize_constant_kernel_residual(
            np.ones(shape),
            kernel=np.ones((3, 3)),
            target_variance=target_variance,
            reference_variance=np.zeros(shape),
        )

    assert not result.valid_mask[2, 2]
    assert np.isnan(result.standardized_residual[2, 2])
    assert np.count_nonzero(result.valid_mask) == 8


def test_nonfinite_standardized_value_is_excluded() -> None:
    shape = (3, 4)
    residual = np.ones(shape)
    residual[1, 2] = np.finfo(np.float64).max
    target_variance = np.ones(shape)
    target_variance[1, 2] = np.nextafter(0.0, 1.0)

    result = standardize_constant_kernel_residual(
        residual,
        kernel=np.ones((1, 1)),
        target_variance=target_variance,
        reference_variance=np.zeros(shape),
    )

    assert not result.valid_mask[1, 2]
    assert np.isnan(result.standardized_residual[1, 2])
    assert np.count_nonzero(result.valid_mask) == residual.size - 1


def test_independent_noise_has_unit_marginal_width() -> None:
    rng = np.random.default_rng(20260906)
    shape = (24, 25)
    target_variance = np.full(shape, 1.7)
    reference_variance = np.full(shape, 0.05)
    reference_variance[:, 13:] = 12.0
    reference_variance[12:, :] *= 240.0
    kernel = np.asarray(
        [
            [0.02, 0.04, 0.07],
            [0.09, 0.31, 0.16],
            [0.05, 0.11, 0.15],
        ]
    )
    standardized = []
    residuals = []
    for _ in range(256):
        reference_noise = rng.normal(scale=np.sqrt(reference_variance))
        target_noise = rng.normal(scale=np.sqrt(target_variance))
        residual = target_noise - _fftconvolve_same(reference_noise, kernel)
        result = standardize_constant_kernel_residual(
            residual,
            kernel=kernel,
            target_variance=target_variance,
            reference_variance=reference_variance,
        )
        standardized.append(result.standardized_residual[result.valid_mask])
        residuals.append(residual)

    samples = np.concatenate(standardized)
    assert abs(float(np.mean(samples))) < 0.02
    assert 0.98 < float(np.std(samples)) < 1.02
    # The variance steps make orientation visible: a flipped squared kernel
    # moves the propagated step by one pixel and the pooled ratio along the
    # affected row or column leaves 1 by a factor of 1.2 to 3.9.
    variance_ratio = (
        np.stack(residuals).var(axis=0) / result.marginal_residual_variance
    )[1:-1, 1:-1]
    assert np.allclose(variance_ratio.mean(axis=0), 1.0, atol=0.15)
    assert np.allclose(variance_ratio.mean(axis=1), 1.0, atol=0.15)


def test_diagonal_standardization_does_not_remove_neighbor_correlation() -> (
    None
):
    rng = np.random.default_rng(8675309)
    shape = (18, 19)
    reference_variance = np.ones(shape)
    target_variance = np.full(shape, 1.0e-6)
    kernel_1d = np.asarray([0.25, 0.5, 0.25])
    kernel = np.outer(kernel_1d, kernel_1d)
    left = []
    right = []
    for _ in range(128):
        reference_noise = rng.normal(size=shape)
        target_noise = rng.normal(scale=1.0e-3, size=shape)
        residual = target_noise - fftconvolve(
            reference_noise, kernel, mode="same"
        )
        result = standardize_constant_kernel_residual(
            residual,
            kernel=kernel,
            target_variance=target_variance,
            reference_variance=reference_variance,
        )
        interior = result.standardized_residual[1:-1, 1:-1]
        left.append(interior[:, :-1].ravel())
        right.append(interior[:, 1:].ravel())

    correlation = np.corrcoef(np.concatenate(left), np.concatenate(right))[
        0, 1
    ]
    assert correlation > 0.5


def test_fitted_kernel_residual_has_unit_marginal_width() -> None:
    rng = np.random.default_rng(20260906)
    shape = (128, 128)
    kernel_shape = (7, 7)
    components = [GaussianBasisComponent(sigma=1.5, degree=1)]
    basis, _ = build_gaussian_polynomial_basis(kernel_shape, components)
    true_kernel = 0.08 * basis[0] + 0.015 * basis[1] - 0.01 * basis[2]
    y_coords, x_coords = np.mgrid[: shape[0], : shape[1]]
    clean = 50.0 + 200.0 * np.sin(x_coords / 5.0) * np.cos(y_coords / 6.0)
    target_variance = np.full(shape, 2.0)
    reference_variance = np.full(shape, 1.0)
    reference_variance[:, 64:] = 4.0
    reference_variance[64:, :] *= 2.0
    expected_valid = np.zeros(shape, dtype=bool)
    expected_valid[3:-3, 3:-3] = True
    standardized = []
    for _ in range(8):
        reference = clean + rng.normal(scale=np.sqrt(reference_variance))
        target = (
            fftconvolve(clean, true_kernel, mode="same")
            + 3.0
            + rng.normal(scale=np.sqrt(target_variance))
        )
        fit = solve_constant_kernel(
            reference,
            target,
            components,
            kernel_shape=kernel_shape,
            variance=target_variance,
            backend="cpu",
        )
        noise = standardize_constant_kernel_residual(
            fit.residual,
            kernel=fit.kernel,
            target_variance=target_variance,
            reference_variance=reference_variance,
        )
        assert np.isnan(fit.residual[~expected_valid]).all()
        assert np.array_equal(noise.valid_mask, expected_valid)
        standardized.append(noise.standardized_residual[expected_valid])

    samples = np.concatenate(standardized)
    assert abs(float(np.mean(samples))) < 0.02
    assert 0.97 < float(np.std(samples)) < 1.03


def test_kernel_equal_to_image_leaves_single_centre_pixel() -> None:
    shape = (5, 7)
    reference_variance = np.arange(1.0, np.prod(shape) + 1.0).reshape(shape)
    kernel = np.random.default_rng(7).uniform(-1.0, 1.0, size=shape)

    result = standardize_constant_kernel_residual(
        np.ones(shape),
        kernel=kernel,
        target_variance=np.ones(shape),
        reference_variance=reference_variance,
    )

    expected_valid = np.zeros(shape, dtype=bool)
    expected_valid[2, 3] = True
    assert np.array_equal(result.valid_mask, expected_valid)
    expected = float(np.sum(reference_variance * kernel[::-1, ::-1] ** 2))
    assert np.isclose(
        result.propagated_reference_variance[2, 3], expected, rtol=1.0e-12
    )


def test_fortran_ordered_integer_inputs_are_accepted() -> None:
    shape = (6, 7)
    residual = np.asfortranarray(
        np.arange(np.prod(shape), dtype=np.int32).reshape(shape)
    )
    kernel = np.asfortranarray(
        np.asarray([[1, 2, 0], [0, 1, 1], [2, 0, 1]], dtype=np.int8)
    )
    target_variance = np.asfortranarray(np.full(shape, 2, dtype=np.int16))
    reference_variance = np.asfortranarray(np.full(shape, 3, dtype=np.int64))

    result = standardize_constant_kernel_residual(
        residual,
        kernel=kernel,
        target_variance=target_variance,
        reference_variance=reference_variance,
    )

    valid = result.valid_mask
    assert np.array_equal(valid[1:-1, 1:-1], np.ones((4, 5), dtype=bool))
    assert np.count_nonzero(valid) == 20
    assert np.allclose(result.propagated_reference_variance[valid], 36.0)
    assert np.allclose(result.marginal_residual_variance[valid], 38.0)
    assert np.allclose(
        result.standardized_residual[valid], residual[valid] / np.sqrt(38.0)
    )
    for value in (
        result.propagated_reference_variance,
        result.marginal_residual_variance,
        result.standardized_residual,
    ):
        assert value.dtype == np.float64
        assert value.flags.c_contiguous
