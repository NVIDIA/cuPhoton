# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import get_args, get_type_hints

import numpy as np
import pytest

from cuphoton.xfit import (
    GaussianDipoleModel,
    LMConfig,
    StampDipoleModel,
    fit_dipoles,
)


def _gaussian_truth(dtype=np.float64) -> np.ndarray:
    return np.asarray(
        [
            [5.0, 1.3, 1.8, 0.25, -2.0, 0.5, 2.2, -0.7],
            [3.0, 1.6, 1.1, -0.3, -1.0, -1.0, 2.0, 1.0],
        ],
        dtype=dtype,
    )


@pytest.mark.parametrize("mode", ["difference", "split"])
def test_gaussian_analytic_jacobian_matches_finite_difference(mode) -> None:
    model = GaussianDipoleModel((8, 11), dtype=np.float64)
    parameters = np.asarray([[2.3, 1.2, 1.8, 0.37, 1.1, -0.7, -1.4, 0.55]])
    analytic = np.asarray(model.jacobian(parameters, mode=mode))
    baseline = np.asarray(model.evaluate(parameters, mode=mode))
    finite_difference = np.empty_like(analytic)
    step = 1.0e-6
    for parameter in range(parameters.shape[1]):
        perturbed = parameters.copy()
        perturbed[:, parameter] += step
        finite_difference[:, parameter] = (
            np.asarray(model.evaluate(perturbed, mode=mode)) - baseline
        ) / step

    assert np.allclose(analytic, finite_difference, rtol=3.0e-5, atol=2.0e-6)


@pytest.mark.parametrize("mode", ["difference", "split"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_gaussian_fit_recovers_rectangular_synthetic_batch(
    mode: str, dtype
) -> None:
    model = GaussianDipoleModel((11, 15), dtype=dtype)
    truth = _gaussian_truth(dtype)
    images = model.evaluate(truth, mode=mode)
    offset = np.asarray(
        [0.3, 0.1, -0.1, 0.05, 0.1, -0.1, -0.1, 0.1],
        dtype=dtype,
    )
    result = fit_dipoles(
        images,
        model=model,
        initial=truth + offset,
        mode=mode,
        backend="numpy",
    )

    tolerance = 2.0e-3 if dtype is np.float32 else 2.0e-7
    assert result.converged.all()
    assert np.allclose(
        result.parameters, truth, rtol=tolerance, atol=tolerance
    )
    assert result.residuals.shape == images.shape
    assert result.covariance.shape == (2, 8, 8)
    assert result.standard_errors.shape == (2, 8)
    assert result.backend == "numpy"
    assert result.device == "cpu"
    assert result.dtype == np.dtype(dtype).name


def test_gaussian_default_initialization_is_deterministic() -> None:
    model = GaussianDipoleModel((15, 17), dtype=np.float64)
    truth = np.asarray([[5.0, 1.3, 1.3, 0.0, -3.0, 0.0, 3.0, 0.0]])
    images = model.evaluate(truth)

    first = fit_dipoles(images, model=model, backend="numpy")
    second = fit_dipoles(images, model=model, backend="numpy")

    assert first.converged.all()
    assert np.array_equal(first.status, second.status)
    assert np.array_equal(first.evaluations, second.evaluations)
    assert np.allclose(first.parameters, second.parameters, atol=1.0e-12)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_gaussian_finite_difference_fit_keeps_theta_canonical(dtype) -> None:
    model = GaussianDipoleModel((21, 21), dtype=dtype)
    truth = np.asarray(
        [[5.0, 1.2, 1.8, 0.4, -3.0, 0.5, 3.0, -0.5]], dtype=dtype
    )
    images = np.asarray(model.evaluate(truth))

    result = fit_dipoles(
        images,
        model=model,
        backend="numpy",
        config=LMConfig(use_finite_difference=True),
    )

    assert result.converged.all()
    assert result.uncertainty_valid.all()
    assert np.all(result.parameters[:, 3] >= -np.pi / 2)
    assert np.all(result.parameters[:, 3] < np.pi / 2)
    # Swapping sigma_x/sigma_y and rotating theta by pi/2 is the same ellipse.
    recovered = np.asarray(model.evaluate(result.parameters))
    tolerance = 3.0e-5 if dtype is np.float32 else 2.0e-8
    assert np.allclose(recovered, images, rtol=tolerance, atol=tolerance)


def test_public_fit_option_hints_are_literal_and_cupy_optional() -> None:
    hints = get_type_hints(fit_dipoles)

    assert set(get_args(hints["mode"])) == {"difference", "split"}
    assert set(get_args(hints["backend"])) == {"auto", "numpy", "cupy"}


def test_split_batch_three_auxiliary_precedence_is_per_candidate() -> None:
    model = GaussianDipoleModel((5, 7), dtype=np.float64)
    truth = np.broadcast_to(_gaussian_truth()[:1], (3, 8)).copy()
    images = model.evaluate(truth, mode="split")
    per_candidate_mask = np.ones((3, 5, 7), dtype=bool)
    per_candidate_mask[0] = False
    per_candidate_mask[1, 0, 0] = False

    per_candidate = fit_dipoles(
        images,
        model=model,
        initial=truth,
        mask=per_candidate_mask,
        mode="split",
        backend="numpy",
        config=LMConfig(max_evaluations=1),
    )
    per_plane = fit_dipoles(
        images,
        model=model,
        initial=truth,
        mask=per_candidate_mask[None, ...],
        mode="split",
        backend="numpy",
        config=LMConfig(max_evaluations=1),
    )

    assert per_candidate.valid_pixel_count.tolist() == [0, 102, 105]
    assert per_plane.valid_pixel_count.tolist() == [69, 69, 69]


@pytest.mark.parametrize("outlier_policy", ["mask", "variance"])
def test_gaussian_default_initialization_ignores_downweighted_outlier(
    outlier_policy: str,
) -> None:
    model = GaussianDipoleModel((21, 21), dtype=np.float64)
    truth = np.asarray([[5.0, 1.3, 1.3, 0.0, -3.0, 0.0, 3.0, 0.0]])
    clean = np.asarray(model.evaluate(truth))
    contaminated = clean.copy()
    contaminated[:, 0, 0] = 1.0e3
    mask = np.ones((21, 21), dtype=bool)
    variance = np.ones((21, 21), dtype=np.float64)
    if outlier_policy == "mask":
        mask[0, 0] = False
    else:
        variance[0, 0] = 1.0e16

    result = fit_dipoles(
        contaminated,
        model=model,
        mask=mask,
        variance=variance,
        backend="numpy",
    )
    prediction = np.asarray(model.evaluate(result.parameters))

    assert result.converged.all()
    assert np.allclose(prediction, clean, rtol=2.0e-6, atol=2.0e-6)


def test_fit_accepts_nonfinite_values_only_when_masked() -> None:
    model = GaussianDipoleModel((11, 15), dtype=np.float64)
    truth = _gaussian_truth()[:1]
    images = np.asarray(model.evaluate(truth))
    images[:, 0, 0] = np.nan
    mask = np.ones((11, 15), dtype=bool)

    with pytest.raises(ValueError, match="finite on included pixels"):
        fit_dipoles(
            images,
            model=model,
            initial=truth,
            mask=mask,
            backend="numpy",
        )

    mask[0, 0] = False
    result = fit_dipoles(
        images,
        model=model,
        initial=truth + 0.05,
        mask=mask,
        backend="numpy",
    )

    assert result.converged.all()
    assert np.allclose(
        model.evaluate(result.parameters),
        model.evaluate(truth),
        rtol=1.0e-7,
        atol=1.0e-7,
    )


@pytest.mark.parametrize("widths", [[0.0, 1.0], [-1.3, 1.8]])
def test_gaussian_fit_rejects_nonpositive_initial_widths(widths) -> None:
    model = GaussianDipoleModel((9, 11), dtype=np.float64)
    truth = np.asarray([[5.0, 1.3, 1.8, 0.0, -2.0, 0.0, 2.0, 0.0]])
    images = model.evaluate(truth)
    initial = truth.copy()
    initial[:, 1:3] = widths

    with pytest.raises(ValueError, match="must be positive"):
        fit_dipoles(
            images,
            model=model,
            initial=initial,
            backend="numpy",
        )


@pytest.mark.parametrize("method_name", ["evaluate", "jacobian"])
def test_gaussian_model_rejects_nonpositive_widths(method_name) -> None:
    model = GaussianDipoleModel((9, 11), dtype=np.float64)
    parameters = np.asarray([[5.0, -1.3, 1.8, 0.0, -2.0, 0.0, 2.0, 0.0]])

    with pytest.raises(ValueError, match="must be positive"):
        getattr(model, method_name)(parameters)


def test_gaussian_covariance_is_reported_in_physical_widths() -> None:
    model = GaussianDipoleModel((11, 15), dtype=np.float64)
    truth = _gaussian_truth()[:1]
    images = model.evaluate(truth)

    result = fit_dipoles(
        images,
        model=model,
        initial=truth,
        variance=np.ones((11, 15)),
        backend="numpy",
    )

    physical_jacobian = np.asarray(model.jacobian(truth))[0].reshape(8, -1)
    expected = np.linalg.inv(physical_jacobian @ physical_jacobian.T)
    assert result.uncertainty_valid.all()
    assert np.all(result.parameters[:, 1:3] > 0)
    assert np.allclose(result.covariance[0], expected, rtol=2.0e-12)


def test_split_mask_variance_and_plane_weights_define_chi_square() -> None:
    model = GaussianDipoleModel((5, 7), dtype=np.float64)
    truth = _gaussian_truth()[:1]
    prediction = model.evaluate(truth, mode="split")
    images = prediction + 1.0
    mask = np.ones((1, 5, 7), dtype=bool)
    mask[:, 0, 0] = False
    variance = np.full((5, 7), 4.0)
    variance[0, 0] = np.nan

    result = fit_dipoles(
        images,
        model=model,
        initial=truth,
        mask=mask,
        variance=variance,
        mode="split",
        backend="numpy",
        config=LMConfig(max_evaluations=1),
    )

    included_pixels = 5 * 7 - 1
    expected_chi_square = included_pixels * (1.0 + 0.25 + 0.25) / 4.0
    included = np.broadcast_to(mask[:, None], images.shape)
    plane_weights = np.asarray(model.split_weights)[None, :, None, None]
    expected_null_chi_square = np.sum(
        np.where(included, images * plane_weights / 2.0, 0.0) ** 2
    )
    assert result.chi_square[0] == pytest.approx(expected_chi_square)
    assert result.valid_pixel_count[0] == 3 * included_pixels
    assert result.valid_pixel_fraction[0] == pytest.approx(
        included_pixels / (5 * 7)
    )
    assert result.null_chi_square[0] == pytest.approx(
        expected_null_chi_square
    )
    assert result.delta_chi_square[0] == pytest.approx(
        expected_null_chi_square - expected_chi_square
    )
    assert result.fractional_null_improvement[0] == pytest.approx(
        result.delta_chi_square[0] / expected_null_chi_square
    )
    assert result.degrees_of_freedom[0] == 3 * included_pixels - 8
    assert result.status[0] == "max_evaluations"
    assert not result.uncertainty_valid[0]


def test_fit_quality_statistics_preserve_worse_than_null_fit() -> None:
    model = GaussianDipoleModel((5, 7), dtype=np.float64)
    initial = _gaussian_truth()
    images = np.asarray(model.evaluate(initial))
    images[1] *= 0.01
    mask = np.ones_like(images, dtype=bool)
    mask[0, 0, 0] = False
    mask[1, :2] = False

    result = fit_dipoles(
        images,
        model=model,
        initial=initial,
        mask=mask,
        backend="numpy",
        config=LMConfig(max_evaluations=1),
    )

    assert result.valid_pixel_count.tolist() == [34, 21]
    assert np.allclose(result.valid_pixel_fraction, [34 / 35, 21 / 35])
    assert result.chi_square[0] == pytest.approx(0.0)
    assert result.null_chi_square[0] > 0
    assert result.delta_chi_square[0] == pytest.approx(
        result.null_chi_square[0]
    )
    assert result.fractional_null_improvement[0] == pytest.approx(1.0)
    assert result.chi_square[1] > result.null_chi_square[1] > 0
    assert result.delta_chi_square[1] == pytest.approx(
        result.null_chi_square[1] - result.chi_square[1]
    )
    assert result.delta_chi_square[1] < 0
    assert result.fractional_null_improvement[1] == pytest.approx(
        1 - result.chi_square[1] / result.null_chi_square[1]
    )
    assert result.fractional_null_improvement[1] < 0


def test_valid_pixels_are_distinct_from_zero_weighted_observations() -> None:
    model = GaussianDipoleModel(
        (5, 7),
        split_weights=(1.0, 0.0, 0.5),
        dtype=np.float64,
    )
    initial = _gaussian_truth()[:1]
    images = np.asarray(model.evaluate(initial, mode="split"))
    mask = np.ones((5, 7), dtype=bool)
    mask[0, 0] = False

    result = fit_dipoles(
        images,
        model=model,
        initial=initial,
        mask=mask,
        mode="split",
        backend="numpy",
    )

    included_pixels = 5 * 7 - 1
    assert result.valid_pixel_count[0] == 3 * included_pixels
    assert result.valid_pixel_fraction[0] == pytest.approx(
        included_pixels / (5 * 7)
    )
    assert result.degrees_of_freedom[0] == 2 * included_pixels - 8


def test_all_masked_candidate_has_undefined_fractional_improvement() -> None:
    model = GaussianDipoleModel((5, 7), dtype=np.float64)
    initial = _gaussian_truth()[:1]
    images = np.full((1, 5, 7), np.nan)

    result = fit_dipoles(
        images,
        model=model,
        initial=initial,
        mask=np.zeros_like(images, dtype=bool),
        backend="numpy",
    )

    assert result.valid_pixel_count[0] == 0
    assert result.valid_pixel_fraction[0] == 0
    assert result.chi_square[0] == 0
    assert result.null_chi_square[0] == 0
    assert result.delta_chi_square[0] == 0
    assert np.isnan(result.fractional_null_improvement[0])


def _sampled_basis() -> np.ndarray:
    y, x = np.mgrid[-3:4, -2:3]
    return np.exp(-0.5 * ((x / 1.0) ** 2 + (y / 1.2) ** 2))


@pytest.mark.parametrize(
    ("evaluation", "mode", "dtype"),
    [
        ("bilinear", "difference", np.float32),
        ("bilinear", "difference", np.float64),
        ("bilinear-vignetted", "difference", np.float64),
        ("finite-volume", "difference", np.float64),
        ("finite-volume", "split", np.float64),
    ],
)
def test_stamp_fit_recovers_synthetic_rectangular_images(
    evaluation,
    mode,
    dtype,
) -> None:
    model = StampDipoleModel(
        _sampled_basis().astype(dtype),
        image_shape=(9, 13),
        evaluation=evaluation,
        dtype=dtype,
    )
    truth = np.asarray([[-2.1, 0.6, 2.2, -0.4, 5.0]], dtype=dtype)
    images = model.evaluate(truth, mode=mode)
    initial = truth + np.asarray(
        [[0.15, -0.1, -0.12, 0.1, -0.4]], dtype=dtype
    )
    step = 1.0e-3 if np.dtype(dtype) == np.dtype(np.float32) else 2.0e-5

    result = fit_dipoles(
        images,
        model=model,
        initial=initial,
        mode=mode,
        backend="numpy",
        config=LMConfig(
            max_evaluations=400,
            finite_difference_step=step,
        ),
    )

    tolerance = 3.0e-4 if np.dtype(dtype) == np.dtype(np.float32) else 2.0e-5
    assert result.converged.all()
    assert np.allclose(
        result.parameters,
        truth,
        rtol=tolerance,
        atol=tolerance,
    )


def test_stamp_split_fit_supports_broadcast_mask_and_variance() -> None:
    model = StampDipoleModel(
        _sampled_basis(),
        image_shape=(9, 13),
        evaluation="bilinear",
        dtype=np.float64,
    )
    truth = np.asarray([[-2.1, 0.6, 2.2, -0.4, 5.0]])
    images = model.evaluate(truth, mode="split")
    assert np.allclose(images[:, 0], images[:, 1] - images[:, 2])
    mask = np.ones((1, 9, 13), dtype=bool)
    mask[:, 0, 0] = False
    variance = np.full((9, 13), 2.0)
    variance[0, 0] = np.nan
    initial = truth + np.asarray([[0.15, -0.1, -0.12, 0.1, -0.4]])

    result = fit_dipoles(
        images,
        model=model,
        initial=initial,
        mask=mask,
        variance=variance,
        mode="split",
        backend="numpy",
        config=LMConfig(
            max_evaluations=400,
            finite_difference_step=2.0e-5,
        ),
    )

    assert result.converged.all()
    assert np.allclose(result.parameters, truth, rtol=2.0e-5, atol=2.0e-5)
    assert result.uncertainty_valid.all()


def test_stamp_default_initialization_recovers_synthetic_dipole() -> None:
    model = StampDipoleModel(
        _sampled_basis(),
        image_shape=(9, 13),
        evaluation="bilinear",
        dtype=np.float64,
    )
    truth = np.asarray([[-2.0, 0.0, 2.0, 0.0, 5.0]])
    images = model.evaluate(truth)

    result = fit_dipoles(
        images,
        model=model,
        backend="numpy",
        config=LMConfig(
            max_evaluations=500,
            finite_difference_step=2.0e-5,
        ),
    )

    assert result.converged.all()
    assert np.allclose(result.parameters, truth, rtol=3.0e-5, atol=3.0e-5)


def test_uncertainty_reports_when_final_jacobian_exceeds_budget() -> None:
    model = StampDipoleModel(
        _sampled_basis(),
        image_shape=(9, 13),
        evaluation="bilinear",
        dtype=np.float64,
    )
    truth = np.asarray([[-2.0, 0.0, 2.0, 0.0, 5.0]])
    images = model.evaluate(truth)
    initial = truth + np.asarray([[0.1, 0.0, -0.1, 0.0, -0.2]])

    result = fit_dipoles(
        images,
        model=model,
        initial=initial,
        backend="numpy",
        config=LMConfig(f_tol=1.0, max_evaluations=7),
    )

    assert result.converged.all()
    assert result.evaluations.tolist() == [7]
    assert not result.uncertainty_valid.any()
    assert result.uncertainty_reason == ("final Jacobian is unavailable",)


def test_vignetted_bilinear_preserves_visible_requested_flux() -> None:
    basis = _sampled_basis()
    ordinary = StampDipoleModel(
        basis, image_shape=(9, 9), evaluation="bilinear"
    )
    vignetted = StampDipoleModel(
        basis,
        image_shape=(9, 9),
        evaluation="bilinear-vignetted",
    )
    parameters = np.asarray([[4.0, 0.0, -1.0, 0.0, 7.0]])
    ordinary_split = ordinary.evaluate(parameters, mode="split")
    vignetted_split = vignetted.evaluate(parameters, mode="split")

    assert ordinary_split[0, 1].sum() < 7.0
    assert vignetted_split[0, 1].sum() == pytest.approx(7.0)
    assert vignetted.to_backend("numpy").evaluation == "bilinear-vignetted"


def test_rank_deficient_fit_reports_invalid_public_uncertainty() -> None:
    model = GaussianDipoleModel((7, 9), dtype=np.float64)
    initial = np.asarray([[0.0, 1.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0]])
    images = model.evaluate(initial)

    result = fit_dipoles(
        images,
        model=model,
        initial=initial,
        backend="numpy",
    )

    assert result.converged.all()
    assert not result.uncertainty_valid.any()
    assert result.uncertainty_reason == ("rank-deficient Jacobian",)
    assert np.isnan(result.covariance).all()
    assert np.isnan(result.standard_errors).all()


def test_stamp_multi_mode_basis_requires_explicit_weights() -> None:
    basis = np.stack((_sampled_basis(), _sampled_basis()))
    with pytest.raises(ValueError, match="basis_weights is required"):
        StampDipoleModel(basis, image_shape=(9, 9))


def test_cupy_fit_matches_numpy_when_cuda_is_available() -> None:
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device is unavailable")
    except Exception:
        pytest.skip("CUDA runtime is unavailable")
    truth = _gaussian_truth()
    model = GaussianDipoleModel((11, 15), dtype=np.float64)
    images = model.evaluate(truth)
    initial = truth + np.asarray([0.2, 0.1, -0.1, 0.03, 0.1, -0.1, -0.1, 0.1])

    cpu = fit_dipoles(images, model=model, initial=initial, backend="numpy")
    gpu = fit_dipoles(images, model=model, initial=initial, backend="cupy")

    assert gpu.backend == "cupy"
    assert gpu.device.startswith("cuda:")
    assert np.allclose(gpu.parameters, cpu.parameters, rtol=2e-8, atol=2e-8)
    assert np.array_equal(gpu.valid_pixel_count, cpu.valid_pixel_count)
    for field in (
        "valid_pixel_fraction",
        "null_chi_square",
        "delta_chi_square",
        "fractional_null_improvement",
    ):
        assert np.allclose(
            getattr(gpu, field),
            getattr(cpu, field),
            rtol=2e-8,
            atol=2e-8,
        )


def test_cupy_finite_difference_theta_fix_matches_numpy() -> None:
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device is unavailable")
    except Exception:
        pytest.skip("CUDA runtime is unavailable")
    model = GaussianDipoleModel((21, 21), dtype=np.float64)
    truth = np.asarray([[5.0, 1.2, 1.8, 0.4, -3.0, 0.5, 3.0, -0.5]])
    images = np.asarray(model.evaluate(truth))
    config = LMConfig(use_finite_difference=True)

    cpu = fit_dipoles(images, model=model, backend="numpy", config=config)
    gpu = fit_dipoles(images, model=model, backend="cupy", config=config)

    assert cpu.converged.all()
    assert gpu.converged.all()
    assert np.all(gpu.parameters[:, 3] >= -np.pi / 2)
    assert np.all(gpu.parameters[:, 3] < np.pi / 2)
    cpu_prediction = np.asarray(model.evaluate(cpu.parameters))
    gpu_prediction = np.asarray(model.evaluate(gpu.parameters))
    assert np.allclose(cpu_prediction, images, rtol=2.0e-8, atol=2.0e-8)
    assert np.allclose(gpu_prediction, images, rtol=1.0e-7, atol=1.0e-7)


def test_cupy_stamp_fit_matches_numpy_when_cuda_is_available() -> None:
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device is unavailable")
    except Exception:
        pytest.skip("CUDA runtime is unavailable")
    model = StampDipoleModel(
        _sampled_basis(),
        image_shape=(9, 13),
        evaluation="finite-volume",
        dtype=np.float64,
    )
    truth = np.asarray([[-2.1, 0.6, 2.2, -0.4, 5.0]])
    images = model.evaluate(truth, mode="split")
    initial = truth + np.asarray([[0.15, -0.1, -0.12, 0.1, -0.4]])
    config = LMConfig(
        max_evaluations=400,
        finite_difference_step=2.0e-5,
    )

    cpu = fit_dipoles(
        images,
        model=model,
        initial=initial,
        mode="split",
        backend="numpy",
        config=config,
    )
    gpu = fit_dipoles(
        images,
        model=model,
        initial=initial,
        mode="split",
        backend="cupy",
        config=config,
    )

    assert gpu.backend == "cupy"
    assert gpu.device.startswith("cuda:")
    assert gpu.converged.all()
    assert np.allclose(gpu.parameters, cpu.parameters, rtol=2e-8, atol=2e-8)
