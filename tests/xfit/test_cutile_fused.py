# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
from threading import Barrier

import numpy as np
import pytest

import cuphoton.xfit.api as xfit_api
from cuphoton.xfit import GaussianDipoleModel, LMConfig, fit_dipoles


@pytest.fixture
def cutile_gpu():
    cp = pytest.importorskip("cupy")
    pytest.importorskip("cuda.tile")
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except Exception as exc:
        pytest.skip(f"CUDA runtime is unavailable: {exc}")
    if count < 1:
        pytest.skip("CUDA device is unavailable")
    return cp


def _case(mode, dtype):
    model = GaussianDipoleModel(
        (11, 15), split_weights=(1.0, 0.25, 1.75), dtype=dtype
    )
    truth = np.asarray(
        [
            [5.0, 1.3, 1.8, 0.25, -2.0, 0.5, 2.2, -0.7],
            [3.0, 1.6, 1.1, -0.3, -1.0, -1.0, 2.0, 1.0],
            [4.2, 1.4, 2.0, 0.1, -1.7, 0.8, 1.9, -0.9],
        ],
        dtype=dtype,
    )
    images = np.asarray(model.evaluate(truth, mode=mode))
    initial = truth + np.asarray(
        [0.2, 0.1, -0.1, 0.03, 0.1, -0.1, -0.1, 0.1], dtype=dtype
    )
    # The first row finishes before the other rows, exercising original
    # candidate indices after the active batch has been compacted.
    initial[0] = truth[0]
    mask = np.ones(images.shape, dtype=bool)
    mask.reshape(3, -1)[0, ::17] = False
    mask.reshape(3, -1)[1, 3::19] = False
    mask.reshape(3, -1)[2, 5::23] = False
    variance = np.linspace(0.5, 2.0, images.size, dtype=dtype).reshape(
        images.shape
    )
    images[~mask] = np.nan
    variance[~mask] = np.nan
    return dict(
        images=images,
        model=model,
        initial=initial,
        mode=mode,
        mask=mask,
        variance=variance,
    )


def _assert_fit_equal(actual, expected, dtype):
    tolerance = 3e-5 if dtype is np.float32 else 5e-12
    for name in (
        "status",
        "converged",
        "evaluations",
        "valid_pixel_count",
        "degrees_of_freedom",
        "uncertainty_valid",
    ):
        np.testing.assert_array_equal(
            getattr(actual, name), getattr(expected, name), err_msg=name
        )
    assert actual.uncertainty_reason == expected.uncertainty_reason
    for name in (
        "parameters",
        "residual_norm",
        "chi_square",
        "valid_pixel_fraction",
        "null_chi_square",
        "delta_chi_square",
        "fractional_null_improvement",
        "reduced_chi_square",
        "covariance",
        "standard_errors",
        "residuals",
    ):
        value = getattr(actual, name)
        reference = getattr(expected, name)
        np.testing.assert_array_equal(
            np.isnan(value), np.isnan(reference), err_msg=name
        )
        np.testing.assert_allclose(
            value,
            reference,
            rtol=tolerance,
            atol=tolerance,
            equal_nan=True,
            err_msg=name,
        )


def _fit_and_rank(cp, monkeypatch, **kwargs):
    # Rank is a solver diagnostic; the public result reports its effect on
    # covariance through uncertainty_reason instead of exposing the number.
    solve = xfit_api.batched_levenberg_marquardt
    ranks = []

    def record_rank(*args, **options):
        result = solve(*args, **options)
        ranks.append(cp.asnumpy(result.rank).copy())
        return result

    with monkeypatch.context() as patch:
        patch.setattr(xfit_api, "batched_levenberg_marquardt", record_rank)
        result = fit_dipoles(**kwargs)
    assert len(ranks) == 1
    return result, ranks[0]


@pytest.mark.parametrize("mode", ["difference", "split"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_fused_fit_preserves_weighted_compacted_results(
    cutile_gpu, monkeypatch, mode, dtype
):
    case = _case(mode, dtype)
    fused, fused_rank = _fit_and_rank(
        cutile_gpu, monkeypatch, **case, backend="cutile", fusion=True
    )
    assert fused.evaluations[0] == 1
    assert np.all(fused.evaluations[1:] > 1)
    assert fused.converged.all()
    assert fused.uncertainty_valid.all()
    for backend in ("cupy", "cutile"):
        reference, rank = _fit_and_rank(
            cutile_gpu, monkeypatch, **case, backend=backend
        )
        _assert_fit_equal(fused, reference, dtype)
        np.testing.assert_array_equal(fused_rank, rank)


@pytest.mark.parametrize("scenario", ["rank", "budget", "rejected-step"])
def test_fused_fit_preserves_termination_and_failed_uncertainty(
    cutile_gpu, monkeypatch, scenario
):
    model = GaussianDipoleModel((11, 15), dtype=np.float64)
    truth = np.asarray([[5.0, 1.3, 1.8, 0.25, -2.0, 0.5, 2.2, -0.7]])
    initial = truth.copy()
    config = LMConfig()
    if scenario == "rank":
        truth[:, 0] = 0
        initial = truth.copy()
    elif scenario == "budget":
        initial[:, 0] += 0.2
        config = LMConfig(max_evaluations=1)
    else:
        initial = np.asarray([[1.0, 0.5, 0.7, 0.0, 0.0, 0.0, 0.5, 0.0]])
        config = LMConfig(max_evaluations=2, initial_damping=1e-8)
    case = dict(
        images=model.evaluate(truth),
        model=model,
        initial=initial,
        config=config,
    )
    fused, rank = _fit_and_rank(
        cutile_gpu, monkeypatch, **case, backend="cutile", fusion=True
    )
    unfused, unfused_rank = _fit_and_rank(
        cutile_gpu, monkeypatch, **case, backend="cutile"
    )
    _assert_fit_equal(fused, unfused, np.float64)
    np.testing.assert_array_equal(rank, unfused_rank)
    # Analytic final diagnostics do not consume residual evaluations, even
    # when the normal-equation path needs to refresh its Jacobian.
    reference, reference_rank = _fit_and_rank(
        cutile_gpu, monkeypatch, **case, backend="cupy"
    )
    _assert_fit_equal(fused, reference, np.float64)
    np.testing.assert_array_equal(rank, reference_rank)
    assert not fused.uncertainty_valid.any()
    assert np.isnan(fused.covariance).all()
    if scenario == "rank":
        assert np.all(rank < 8)
        assert fused.uncertainty_reason == ("rank-deficient Jacobian",)
    elif scenario == "budget":
        np.testing.assert_array_equal(fused.evaluations, [1])
        np.testing.assert_array_equal(rank, [-1])
    else:
        np.testing.assert_array_equal(fused.evaluations, [2])
        np.testing.assert_array_equal(fused.status, ["max_evaluations"])
        # A rejected trial must preserve the accepted parameters/residuals.
        np.testing.assert_allclose(
            fused.parameters, initial, rtol=5e-12, atol=5e-12
        )


@pytest.mark.parametrize("mode", ["difference", "split"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_fused_kernel_outputs_match_noisy_strided_inputs(
    cutile_gpu, mode, dtype
):
    from cuphoton.xfit._cutile_fused import GaussianWorkspace

    cp = cutile_gpu
    rng = np.random.default_rng(1219)
    shape = (8, 10)
    physical = cp.asarray(_case(mode, dtype)["initial"])
    model = GaussianDipoleModel(shape, backend="cupy", dtype=dtype)
    prediction = model.evaluate(physical, mode=mode).reshape(3, -1)
    images = prediction + cp.asarray(
        rng.normal(scale=0.2, size=prediction.shape), dtype=dtype
    )
    weights = cp.asarray(
        rng.uniform(0.25, 2.0, size=prediction.shape), dtype=dtype
    )
    weights[:, ::7] = 0
    # Both inputs have non-unit observation strides; the active rows also
    # arrive in a different order from their original image/weight storage.
    images = cp.repeat(images, 2, axis=1)[:, ::2]
    weights = cp.repeat(weights, 2, axis=1)[:, ::2]
    assert not images.flags.c_contiguous
    assert not weights.flags.c_contiguous
    indices = cp.asarray([2, 0], dtype=cp.int64)
    active = physical[indices].copy()
    active[:, 0] *= 1.1
    parameters = active.copy()
    parameters[:, 1:3] = cp.log(parameters[:, 1:3])
    expected_residuals = (
        model.evaluate(active, mode=mode).reshape(2, -1) - images[indices]
    ) * weights[indices]
    jacobian = model.jacobian(active, mode=mode).reshape(2, 8, -1)
    jacobian[:, 1:3] *= active[:, 1:3, None]
    jacobian *= weights[indices, None, :]
    workspace = GaussianWorkspace(
        images, weights, image_shape=shape, mode=mode
    )
    residuals = workspace.residual(parameters, indices=indices)
    actual_jacobian = workspace.jacobian(parameters, indices=indices)
    tolerance = 3e-5 if dtype is np.float32 else 5e-12
    for actual, expected in (
        (residuals, expected_residuals),
        (actual_jacobian, jacobian),
    ):
        np.testing.assert_allclose(
            cp.asnumpy(actual),
            cp.asnumpy(expected),
            rtol=tolerance,
            atol=tolerance,
        )


def test_fused_jacobian_preserves_cancellation_sensitive_hessian(cutile_gpu):
    from cuphoton.xfit._cutile_fused import GaussianWorkspace

    cp = cutile_gpu
    shape = (51, 51)
    parameters = cp.asarray(
        [[2000.0, 1.5, 2.25, 0.4, -2.0, 0.0, 2.0, 0.0]],
        dtype=cp.float64,
    )
    parameters[:, 1:3] = cp.log(parameters[:, 1:3])
    physical = parameters.copy()
    physical[:, 1:3] = cp.exp(physical[:, 1:3])
    model = GaussianDipoleModel(shape, backend="cupy", dtype=np.float64)
    expected_jacobian = model.jacobian(physical).reshape(1, 8, -1)
    expected_jacobian[:, 1:3] *= physical[:, 1:3, None]
    images = cp.zeros((1, shape[0] * shape[1]), dtype=cp.float64)
    workspace = GaussianWorkspace(
        images, cp.ones_like(images), image_shape=shape, mode="difference"
    )
    jacobian = workspace.jacobian(
        parameters, indices=cp.asarray([0], dtype=cp.int64)
    )
    # Large symmetric stars make cancellation in Hessian cross terms
    # sensitive to the separately rounded products in the analytic model.
    expected_hessian = cp.einsum(
        "knm,kpm->knp", expected_jacobian, expected_jacobian
    )
    hessian = cp.einsum("knm,kpm->knp", jacobian, jacobian)
    for actual, expected in (
        (jacobian, expected_jacobian),
        (hessian, expected_hessian),
    ):
        np.testing.assert_allclose(
            cp.asnumpy(actual), cp.asnumpy(expected), rtol=5e-12, atol=5e-12
        )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_fused_invalid_sigma_trials_keep_accepted_residuals(
    cutile_gpu, dtype
):
    from cuphoton.xfit._cutile_fused import GaussianWorkspace

    cp = cutile_gpu
    model = GaussianDipoleModel((7, 9), dtype=dtype)
    physical = np.asarray([[5, 1.3, 1.8, 0.25, -2, 0.5, 2.2, -0.7]], dtype)
    image = cp.asarray(model.evaluate(physical)).reshape(1, -1)
    workspace = GaussianWorkspace(
        image, cp.ones_like(image), image_shape=(7, 9), mode="difference"
    )
    parameters = cp.asarray(physical)
    parameters[:, 1:3] = cp.log(parameters[:, 1:3])
    indices = cp.asarray([0], dtype=cp.int64)
    accepted = workspace.residual(parameters, indices=indices)
    before = cp.asnumpy(accepted)
    for log_sigma in (-1000.0, 1000.0):
        trial = parameters.copy()
        trial[:, 1:3] = log_sigma
        invalid = workspace.residual(trial, indices=indices)
        assert cp.isnan(invalid).all().item()
    np.testing.assert_array_equal(cp.asnumpy(accepted), before)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("fusion", [False, True])
def test_cutile_large_width_keeps_physical_derivative_order(
    cutile_gpu, monkeypatch, dtype, fusion
):
    from cuphoton.xfit._cutile import gaussian_normal_equations
    from cuphoton.xfit._cutile_fused import GaussianWorkspace

    cp = cutile_gpu
    width = 1e13 if dtype is np.float32 else 1e103
    initial = np.asarray([[1, width, 1, 0, -width, 0, width, 0]], dtype)
    model = GaussianDipoleModel((5, 7), split_weights=(1, 1, 1), dtype=dtype)
    images = np.asarray(model.evaluate(initial, mode="split")) * dtype(1.01)
    mask = np.zeros_like(images, dtype=bool)
    mask[..., 2, :] = True
    parameters = cp.asarray(initial)
    parameters[:, 1:3] = cp.log(parameters[:, 1:3])
    physical = parameters.copy()
    physical[:, 1:3] = cp.exp(physical[:, 1:3])
    gpu_model = model.to_backend("cupy")
    weights = cp.asarray(mask, dtype=dtype).reshape(1, -1)
    data = cp.asarray(images).reshape(1, -1)
    expected_residual = (
        gpu_model.evaluate(physical, mode="split").reshape(1, -1) - data
    ) * weights
    jacobian = gpu_model.jacobian(physical, mode="split").reshape(1, 8, -1)
    jacobian[:, 1:3] *= physical[:, 1:3, None]
    jacobian *= weights[:, None, :]
    # The physical sigma**3 denominator overflows, while sigma**2 and
    # every weighted Jacobian entry remain finite. Simplifying the chain
    # rule to star*x*x/sigma**2 would introduce a spurious width direction.
    assert cp.isfinite(jacobian).all().item()
    assert (jacobian[:, 1] == 0).all().item()
    workspace = GaussianWorkspace(
        data, weights, image_shape=(5, 7), mode="split"
    )
    indices = cp.asarray([0], dtype=cp.int64)
    if fusion:
        actual_jacobian = workspace.jacobian(parameters, indices=indices)
        tolerance = 3e-5 if dtype is np.float32 else 5e-12
        np.testing.assert_allclose(
            cp.asnumpy(actual_jacobian),
            cp.asnumpy(jacobian),
            rtol=tolerance,
            atol=tolerance,
        )
        assert cp.isfinite(actual_jacobian).all().item()
        assert (actual_jacobian[:, 1] == 0).all().item()
    else:
        gradient, hessian = gaussian_normal_equations(
            parameters,
            expected_residual,
            weights,
            image_shape=(5, 7),
            mode="split",
        )
        assert cp.isfinite(gradient).all().item()
        assert cp.isfinite(hessian).all().item()
        assert (gradient[:, 1] == 0).all().item()
        assert (hessian[:, 1, :] == 0).all().item()
    case = dict(
        images=images,
        model=model,
        initial=initial,
        mask=mask,
        mode="split",
        config=LMConfig(max_evaluations=2, f_tol=0, x_tol=0, g_tol=0),
    )
    reference, reference_rank = _fit_and_rank(
        cp, monkeypatch, **case, backend="cupy"
    )
    fused, rank = _fit_and_rank(
        cp, monkeypatch, **case, backend="cutile", fusion=fusion
    )
    np.testing.assert_array_equal(reference.status, ["max_evaluations"])
    np.testing.assert_array_equal(reference.evaluations, [2])
    _assert_fit_equal(fused, reference, dtype)
    np.testing.assert_array_equal(rank, reference_rank)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("fusion", [False, True])
def test_cutile_tiny_width_preserves_invalid_derivative(
    cutile_gpu, monkeypatch, dtype, masked, fusion
):
    from cuphoton.xfit._cutile import gaussian_normal_equations
    from cuphoton.xfit._cutile_fused import GaussianWorkspace

    cp = cutile_gpu
    width = 1e-16 if dtype is np.float32 else 1e-110
    initial = np.asarray([[1, width, 1, 0, -1, 0, 1, 0]], dtype)
    images = np.full((1, 2, 2), np.nan if masked else 0, dtype=dtype)
    mask = np.zeros_like(images, dtype=bool) if masked else None
    parameters = cp.asarray(initial)
    parameters[:, 1:3] = cp.log(parameters[:, 1:3])
    # Every model pixel is zero, but the physical width derivative is 0/0
    # because sigma**3 underflows. Zero mask weights preserve that NaN.
    weights = cp.full((1, 4), 0 if masked else 1, dtype=dtype)
    workspace = GaussianWorkspace(
        cp.zeros((1, 4), dtype=dtype),
        weights,
        image_shape=(2, 2),
        mode="difference",
    )
    indices = cp.asarray([0], dtype=cp.int64)
    residual = workspace.residual(parameters, indices=indices)
    assert (residual == 0).all().item()
    if fusion:
        jacobian = workspace.jacobian(parameters, indices=indices)
        physical = parameters.copy()
        physical[:, 1:3] = cp.exp(physical[:, 1:3])
        model = GaussianDipoleModel((2, 2), backend="cupy", dtype=dtype)
        expected_jacobian = model.jacobian(physical).reshape(1, 8, 4)
        expected_jacobian[:, 1:3] *= physical[:, 1:3, None]
        expected_jacobian *= weights[:, None, :]
        assert not cp.isfinite(jacobian).all().item()
        np.testing.assert_array_equal(
            cp.asnumpy(cp.isnan(jacobian)),
            cp.asnumpy(cp.isnan(expected_jacobian)),
        )
        tolerance = 3e-5 if dtype is np.float32 else 5e-12
        np.testing.assert_allclose(
            cp.asnumpy(jacobian),
            cp.asnumpy(expected_jacobian),
            rtol=tolerance,
            atol=tolerance,
            equal_nan=True,
        )
    else:
        gradient, hessian = gaussian_normal_equations(
            parameters,
            residual,
            weights,
            image_shape=(2, 2),
            mode="difference",
        )
        assert not cp.isfinite(gradient).all().item()
        assert not cp.isfinite(hessian).all().item()
    case = dict(
        images=images,
        model="gaussian",
        initial=initial,
        mask=mask,
        config=LMConfig(max_evaluations=2),
    )
    reference, reference_rank = _fit_and_rank(
        cp, monkeypatch, **case, backend="cupy"
    )
    fused, rank = _fit_and_rank(
        cp, monkeypatch, **case, backend="cutile", fusion=fusion
    )
    np.testing.assert_array_equal(reference.status, ["invalid_residual"])
    np.testing.assert_array_equal(reference.evaluations, [1])
    _assert_fit_equal(fused, reference, dtype)
    np.testing.assert_array_equal(rank, reference_rank)


@pytest.mark.parametrize(
    ("dtype", "amplitude"), [(np.float32, 1e25), (np.float64, 1e200)]
)
@pytest.mark.parametrize("nonzero_residual", [False, True])
@pytest.mark.parametrize("fusion", [False, True])
def test_cutile_preserves_finite_jacobian_overflow_results(
    cutile_gpu, monkeypatch, dtype, amplitude, nonzero_residual, fusion
):
    cp = cutile_gpu
    model = GaussianDipoleModel((5, 7), backend="cupy", dtype=dtype)
    # Coincident stars keep the model finite and exactly zero. Only the
    # middle row's finite position derivatives overflow their products.
    initial = np.asarray(
        [
            [5, 1, 1, 0, 0, 0, 0, 0],
            [amplitude, 1, 1, 0, 0, 0, 0, 0],
            [3, 1, 1, 0, 0, 0, 0, 0],
        ],
        dtype=dtype,
    )
    images = np.zeros((3, 5, 7), dtype=dtype)
    if nonzero_residual:
        images[1, 1, 2] = amplitude
    jacobian = model.jacobian(cp.asarray(initial)).reshape(3, 8, -1)
    assert cp.isfinite(jacobian).all().item()
    hessian = cp.einsum("knm,kpm->knp", jacobian, jacobian)
    np.testing.assert_array_equal(
        cp.asnumpy(cp.isfinite(hessian).all(axis=(1, 2))),
        [True, False, True],
    )
    case = dict(images=images, model=model, initial=initial)
    reference, reference_rank = _fit_and_rank(
        cp, monkeypatch, **case, backend="cupy"
    )
    actual, rank = _fit_and_rank(
        cp, monkeypatch, **case, backend="cutile", fusion=fusion
    )
    assert reference.status.tolist() == [
        "converged_g_tol",
        "singular" if nonzero_residual else "converged_g_tol",
        "converged_g_tol",
    ]
    np.testing.assert_array_equal(reference.evaluations, [1, 1, 1])
    np.testing.assert_array_equal(actual.parameters, reference.parameters)
    _assert_fit_equal(actual, reference, dtype)
    np.testing.assert_array_equal(rank, reference_rank)


def test_fused_concurrent_workers_keep_independent_results(cutile_gpu):
    cp = cutile_gpu
    cases = [_case("split", np.float64) for _ in range(2)]
    cases[1]["images"] = cases[1]["images"] * 1.25
    cases[1]["initial"] = cases[1]["initial"].copy()
    cases[1]["initial"][:, 0] *= 1.25
    # Warm all specializations on the default stream before testing worker
    # ownership; compilation concurrency is a separate concern.
    with cp.cuda.Stream.null:
        expected = [fit_dipoles(**case, backend="cupy") for case in cases]
        for case in cases:
            fit_dipoles(**case, backend="cutile", fusion=True)
        cp.cuda.Stream.null.synchronize()
    barrier = Barrier(2)

    def worker(index):
        pool = cp.cuda.MemoryPool()
        stream = cp.cuda.Stream(non_blocking=True)
        retained = []
        with cp.cuda.Device(0), stream, cp.cuda.using_allocator(pool.malloc):
            barrier.wait(timeout=30)
            for _ in range(3):
                result = fit_dipoles(
                    **cases[index], backend="cutile", fusion=True
                )
                snapshots = {
                    field.name: value.copy()
                    for field in fields(result)
                    if isinstance(
                        value := getattr(result, field.name), np.ndarray
                    )
                }
                retained.append((result, snapshots))
            stream.synchronize()
        pool.free_all_blocks()
        for result, snapshots in retained:
            _assert_fit_equal(result, expected[index], np.float64)
            for name, snapshot in snapshots.items():
                np.testing.assert_array_equal(getattr(result, name), snapshot)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(worker, index) for index in range(2)]
        for future in futures:
            future.result(timeout=180)


def test_fused_cli_matches_unfused_artifacts(cutile_gpu, tmp_path):
    import pyarrow.parquet as pq
    import yaml

    case = _case("difference", np.float64)
    input_path = tmp_path / "synthetic.npz"
    np.savez(
        input_path,
        candidate_id=np.asarray(["a", "b", "c"]),
        **{
            name: case[name]
            for name in ("images", "initial", "mask", "variance")
        },
    )
    outputs = []
    for fusion in (False, True):
        output = tmp_path / ("fused" if fusion else "unfused")
        command = [
            sys.executable,
            "-m",
            "cuphoton",
            "xfit",
            "fit-dipoles",
            "--input",
            str(input_path),
            "--output-dir",
            str(output),
            "--model",
            "gaussian",
            "--backend",
            "cutile",
        ]
        if fusion:
            command.append("--fusion")
        completed = subprocess.run(
            command, text=True, capture_output=True, timeout=240, check=False
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert not re.search(
            r"CUDA_ERROR_|Exception ignored (?:in:|while)",
            completed.stdout + completed.stderr,
        ), completed.stdout + completed.stderr
        effective = yaml.safe_load(
            (output / "effective-config.yaml").read_text()
        )
        assert effective["fusion"] is fusion
        outputs.append(output)
    reference, fused = [pq.read_table(p / "fits.parquet") for p in outputs]
    for name in ("candidate_id", "status", "evaluations", "converged"):
        assert reference[name].to_pylist() == fused[name].to_pylist()
    for name in case["model"].parameter_names:
        np.testing.assert_allclose(
            fused[name].to_numpy(),
            reference[name].to_numpy(),
            rtol=5e-12,
            atol=5e-12,
        )
    with (
        np.load(outputs[0] / "fit-arrays.npz", allow_pickle=False) as before,
        np.load(outputs[1] / "fit-arrays.npz", allow_pickle=False) as after,
    ):
        for name in ("covariance", "residuals"):
            np.testing.assert_array_equal(
                np.isnan(after[name]), np.isnan(before[name])
            )
            np.testing.assert_allclose(
                after[name],
                before[name],
                rtol=5e-12,
                atol=5e-12,
                equal_nan=True,
            )
