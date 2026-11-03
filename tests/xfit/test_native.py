# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Scientific and ownership coverage for the optional native Gaussian path."""

from __future__ import annotations

import gc
import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from threading import Barrier, Event, Thread
from types import SimpleNamespace

import numpy as np
import pytest

from cuphoton.core.cli import run_component
from cuphoton.xfit import (
    GaussianDipoleModel,
    LMConfig,
    StampDipoleModel,
    fit_dipoles,
    fit_dipoles_device,
)


def _require_extension():
    from cuphoton.xfit._native import load_extension

    # Missing optional builds must skip without initializing CUDA.
    if importlib.util.find_spec("cuphoton.xfit._native_ext") is None:
        pytest.skip("native xFit extension is not built")
    return load_extension()


def _require_native():
    _require_extension()
    return _require_cupy()


def _require_cupy():
    cp = pytest.importorskip("cupy")
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except Exception as exc:
        pytest.skip(f"CuPy CUDA runtime is not usable: {exc}")
    if count < 1:
        pytest.skip("CUDA device is unavailable")
    return cp


def _fixture(dtype=np.float64, *, shape=(11, 15), mode="difference"):
    truth = np.asarray(
        [
            [5.0, 1.3, 1.8, 0.25, -2.0, 0.5, 2.2, -0.7],
            [3.0, 1.6, 1.1, -0.3, -1.0, -1.0, 2.0, 1.0],
        ],
        dtype=dtype,
    )
    model = GaussianDipoleModel(shape, dtype=dtype)
    initial = truth + np.asarray(
        [0.2, 0.1, -0.1, 0.03, 0.1, -0.1, -0.1, 0.1], dtype=dtype
    )
    return model, truth, model.evaluate(truth, mode=mode), initial


def _native_arguments(cp, stream):
    _, _, images, initial = _fixture()
    images = np.tile(images, (16, 1, 1))
    initial = np.tile(initial, (16, 1))
    with stream:
        x = cp.asarray(initial)
        x[:, 1:3] = cp.log(x[:, 1:3])
        data = cp.asarray(images).reshape(len(images), -1)
        owners = (
            x,
            data,
            cp.ones_like(data),
            cp.empty_like(data),
            cp.empty(len(images), dtype=cp.int8),
            cp.empty(len(images), dtype=cp.int32),
            cp.empty(len(images), dtype=cp.bool_),
        )
    stream.synchronize()
    return owners, (1e-8, 1e-8, 1e-8, 1e-3, 10.0, 0.3, 100), (11, 15, 1)


def _native_fit_arguments(cp, stream):
    """Retain production inputs and outputs for direct capsule calls."""
    from cuphoton.xfit import _native_ext as extension

    model, _, images, initial = _fixture()
    original = extension.fit
    captured = []

    def capture(*arguments):
        captured.append(arguments[1:-1])
        return original(*arguments)

    with pytest.MonkeyPatch.context() as patch, stream:
        patch.setattr(extension, "fit", capture)
        fit_dipoles_device(
            cp.asarray(images),
            model=model,
            initial=cp.asarray(initial),
            backend="native",
        )
    (arguments,) = captured
    return arguments


def _gate_producer(stream):
    release, entered, expired = Event(), Event(), Event()

    def hold(_):
        # CUDA host functions may not call CUDA APIs. These primitives also
        # release Python's GIL while waiting, so the native caller can enter.
        entered.set()
        if not release.wait(timeout=30):
            expired.set()

    stream.launch_host_func(hold, None)
    return release, entered, expired


def _assert_fit_matches(
    actual, expected, dtype=np.float64, *, compare_residuals=True
):
    # Predeclared xFit cross-kernel tolerances, also used by the existing
    # cuTile Gaussian tests. Explicit variance keeps covariance measurable
    # instead of scaling it by rounding-level noiseless residuals.
    tolerance = 3.0e-5 if dtype is np.float32 else 5.0e-12
    for name in (
        "status",
        "evaluations",
        "converged",
        "valid_pixel_count",
        "degrees_of_freedom",
        "uncertainty_valid",
    ):
        np.testing.assert_array_equal(
            getattr(actual, name), getattr(expected, name)
        )
    assert actual.uncertainty_reason == expected.uncertainty_reason
    for name in (
        "parameters",
        "residuals",
        "chi_square",
        "covariance",
        "standard_errors",
        "null_chi_square",
        "delta_chi_square",
    ):
        if name == "residuals" and not compare_residuals:
            continue
        np.testing.assert_allclose(
            getattr(actual, name),
            getattr(expected, name),
            rtol=tolerance,
            atol=tolerance,
            equal_nan=True,
            err_msg=name,
        )


def test_import_and_numpy_fit_do_not_load_native_or_cupy():
    source = textwrap.dedent(
        """
        import importlib.abc
        import sys

        class BlockGPU(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if (
                    fullname in {"cupy", "cuda.pathfinder"}
                    or fullname.endswith("._native_ext")
                ):
                    raise AssertionError(f"unexpected GPU import: {fullname}")

        sys.meta_path.insert(0, BlockGPU())
        gil_before = getattr(sys, "_is_gil_enabled", lambda: True)()
        import cuphoton.xfit
        import cuphoton.xfit._native
        import numpy as np
        from cuphoton.xfit import GaussianDipoleModel, fit_dipoles

        truth = np.asarray([[5., 1.3, 1.8, .25, -2., .5, 2.2, -.7]])
        model = GaussianDipoleModel((11, 15))
        result = fit_dipoles(
            model.evaluate(truth), model=model, initial=truth, backend="numpy"
        )
        assert result.converged.all()
        assert "cupy" not in sys.modules
        assert "cuphoton.xfit._native_ext" not in sys.modules
        assert getattr(sys, "_is_gil_enabled", lambda: True)() == gil_before
        """
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parents[2] / "src")
    completed = subprocess.run(
        [sys.executable, "-c", source],
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("device_result", [False, True])
@pytest.mark.parametrize(
    "unsupported", ["stamp", "finite-difference", "custom"]
)
def test_unsupported_configuration_is_rejected_before_cuda(
    monkeypatch, device_result, unsupported
):
    import cuphoton.xfit.api as api

    class CustomGaussian(GaussianDipoleModel):
        pass

    model = (
        StampDipoleModel(np.ones((3, 3)), image_shape=(7, 9))
        if unsupported == "stamp"
        else (
            CustomGaussian((7, 9))
            if unsupported == "custom"
            else GaussianDipoleModel((7, 9))
        )
    )
    monkeypatch.setattr(
        api, "resolve_backend", lambda *a, **k: pytest.fail("resolved CUDA")
    )
    monkeypatch.setattr(api, "_load_cupy", lambda: pytest.fail("loaded CUDA"))
    fit = fit_dipoles_device if device_result else fit_dipoles
    message = {
        "stamp": "supports only the Gaussian model",
        "finite-difference": "finite-difference",
        "custom": "custom Gaussian models",
    }[unsupported]
    with pytest.raises(ValueError, match=message):
        fit(
            np.zeros((1, 7, 9)),
            model=model,
            backend="native",
            config=LMConfig(use_finite_difference=unsupported != "custom"),
        )


def test_missing_extension_reports_build_requirement_before_cuda(monkeypatch):
    from cuphoton.xfit import _native, backend

    monkeypatch.setattr(_native.importlib.util, "find_spec", lambda _: None)
    monkeypatch.setattr(
        backend, "_cupy_backend", lambda **k: pytest.fail("initialized CUDA")
    )
    with pytest.raises(RuntimeError, match="CUPHOTON_XFIT_BUILD_EXT=1"):
        fit_dipoles(np.zeros((1, 7, 9)), model="gaussian", backend="native")


def test_cli_rejects_native_finite_differences_before_loading_fits(
    monkeypatch, tmp_path, capsys
):
    from cuphoton.xfit import backend

    monkeypatch.setattr(
        backend,
        "resolve_backend",
        lambda *a, **k: pytest.fail("resolved CUDA"),
    )
    input_path = tmp_path / "fits.json"
    input_path.write_text("{}")
    rc = run_component(
        "xfit",
        [
            "fit-dipoles",
            "--input",
            str(input_path),
            "--output-dir",
            str(tmp_path / "fit"),
            "--model",
            "gaussian",
            "--backend",
            "native",
            "--use-finite-difference",
        ],
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "finite-difference" in captured.err
    assert not (tmp_path / "fit").exists()


def test_extension_rejects_negative_stream_before_reading_buffers():
    extension = _require_extension()
    with pytest.raises(OverflowError, match="negative.*unsigned"):
        extension.run(None, None, None, None, -1)


@pytest.mark.parametrize("invalid", ["shape", "readonly"])
def test_extension_rejects_unsafe_buffers_without_a_cuda_workspace(invalid):
    extension = _require_extension()
    shapes = [(1, 8), (1, 63), (1, 63), (1, 63), (1,), (1,), (1,)]
    types = ["<f8"] * 4 + ["|i1", "<i4", "|b1"]
    arrays = tuple(
        SimpleNamespace(
            __cuda_array_interface__={
                "version": 3,
                "shape": shape,
                "typestr": types[index],
                "data": (0x1000 * (index + 1), False),
                "strides": None,
            }
        )
        for index, shape in enumerate(shapes)
    )
    if invalid == "shape":
        arrays[0].__cuda_array_interface__["shape"] = (1, 7)
        message = "floating shape"
    else:
        arrays[0].__cuda_array_interface__["data"] = (0x1000, True)
        message = "must be writable"
    # No workspace and no device pointers exist. Metadata validation must
    # reject these owners before capsule access or any CUDA API call.
    with pytest.raises(ValueError, match=message):
        extension.run(
            None,
            arrays,
            (1e-8, 1e-8, 1e-8, 1e-3, 10.0, 0.3, 100),
            (7, 9, 1),
            0,
        )


@pytest.mark.parametrize("mode", ["difference", "split"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_native_gaussian_fit_matches_cupy(dtype, mode):
    _require_native()
    model, _, images, initial = _fixture(dtype, mode=mode)
    options = dict(
        model=model,
        initial=initial,
        mode=mode,
        variance=np.full(images.shape, 0.5, dtype=dtype),
    )
    expected = fit_dipoles(images, backend="cupy", **options)
    actual = fit_dipoles(images, backend="native", **options)
    assert actual.backend == "native"
    assert actual.dtype == np.dtype(dtype).name
    assert actual.uncertainty_valid.all()
    assert np.abs(expected.covariance).max() > 1.0e-3
    _assert_fit_matches(actual, expected, dtype)


def _native_device_info(cp):
    properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    capability = (properties["major"], properties["minor"])
    version = cp.cuda.cublas.getVersion(cp.cuda.device.get_cublas_handle())
    return capability, version


def _assert_native_dispatch(
    timings, count, capability, version, policy="auto"
):
    """Check dispatch for FP64, one-plane, 51x51 inputs."""
    assert timings["kernel_policy"] == policy
    assert timings["cublas_version"] == version
    tiled = count <= 16 and capability in ((12, 0), (12, 1))
    fused = tiled and capability == (12, 0)
    ordered = tiled and capability == (12, 1) and version == 130101
    assert (timings["tiled_jacobian_launches"] > 0) == tiled
    assert (timings["fused_trial_residual_launches"] > 0) == fused
    if ordered:
        kind = "single" if count == 1 else "batch"
        assert timings[f"hessian_{kind}_launches"] > 0
    else:
        assert timings["hessian_single_launches"] == 0
        assert timings["hessian_batch_launches"] == 0
    if count == 1:
        assert timings["hessian_batch_launches"] == 0


def test_native_kernel_policy_preserves_cublas_version_gate():
    extension = _require_extension()
    # The same pure selector is used by real workspaces, which always pass
    # their actual cuBLAS version. These queries never substitute a provider.
    for policy, major, minor in (
        ("auto", 12, 1),
        ("test_sm121", 8, 9),
    ):
        for version in (130100, 130101, 130102, 130800):
            selected = extension._test_kernel_policy(
                policy, major, minor, version
            )
            assert selected == dict(
                tiled_jacobian=True,
                fused_trial_residual=False,
                small_hessian=version == 130101,
            )
    for capability in ((8, 9), (12, 0)):
        selected = extension._test_kernel_policy("auto", *capability, 130101)
        assert not selected["small_hessian"]


@pytest.mark.parametrize("count", [1, 16, 17])
def test_native_specialized_kernels_match_portable_bits(count):
    cp = _require_native()
    extension = _require_extension()
    capability, version = _native_device_info(cp)
    if capability == (8, 9):
        policies = (("test_sm120", (12, 0)), ("test_sm121", (12, 1)))
    elif capability in ((12, 0), (12, 1)):
        policies = (("auto", capability),)
    else:
        pytest.skip("kernel parity requires SM89, SM120 or SM121")

    model, truth, _, initial = _fixture(shape=(51, 51))
    repeats = ((count + 1) // 2, 1)
    truth = np.tile(truth, repeats)[:count].copy()
    initial = np.tile(initial, repeats)[:count].copy()
    offsets = np.linspace(-0.125, 0.125, count)
    for parameters in (truth, initial):
        parameters[:, 0] += offsets
        parameters[:, 4:] += offsets[:, None] * [1.0, -0.5, 0.25, -1.0]
    images = model.evaluate(truth)
    device = cp.cuda.Device().id
    stream = cp.cuda.get_current_stream()

    def run(workspace, inactive_first=False, evaluations=4):
        x = cp.asarray(initial)
        x[:, 1:3] = cp.log(x[:, 1:3])
        data = cp.asarray(images).reshape(count, -1)
        if inactive_first:
            data[0, 0] = cp.nan
        owners = (
            x,
            data,
            cp.full_like(data, np.sqrt(2.0)),
            cp.empty_like(data),
            cp.empty(count, dtype=cp.int8),
            cp.empty(count, dtype=cp.int32),
            cp.empty(count, dtype=cp.bool_),
        )
        timings = extension.run(
            workspace,
            owners,
            (0.0, 0.0, 0.0, 1e-3, 10.0, 0.3, evaluations),
            (51, 51, 1),
            stream.ptr,
        )
        # Compare final native state before covariance/SVD postprocessing.
        outputs = tuple(cp.asnumpy(owners[i]) for i in (0, 3, 4, 5, 6))
        return timings, outputs

    portable = extension._create_test_workspace(device, "portable")
    for policy, selected_capability in policies:
        workspace = (
            extension.create_workspace(device)
            if policy == "auto"
            else extension._create_test_workspace(device, policy)
        )
        for inactive_first in (False, True) if count == 17 else (False,):
            expected_timings, expected = run(portable, inactive_first)
            timings, actual = run(workspace, inactive_first)
            _assert_native_dispatch(
                expected_timings, count, None, version, "portable"
            )
            _assert_native_dispatch(
                timings, count, selected_capability, version, policy
            )
            assert (actual[3][int(inactive_first) :] > 1).all()
            if inactive_first:
                # Original N remains 17 although only 16 rows remain active.
                assert actual[3][0] == 1
            for reference, result in zip(expected, actual, strict=True):
                if reference.dtype == np.float64:
                    reference = reference.view(np.uint64)
                    result = result.view(np.uint64)
                np.testing.assert_array_equal(result, reference)
        # A reused workspace must not carry its launch counts into a run that
        # exhausts its budget before the first Jacobian/Hessian/trial launch.
        timings, _ = run(workspace, evaluations=1)
        assert all(
            value == 0
            for name, value in timings.items()
            if name.endswith("_launches")
        )


@pytest.mark.parametrize(
    "count,shape,mode",
    [
        (17, (5, 7), "difference"),
        (193, (33, 35), "difference"),
        (769, (17, 19), "split"),
        (16, (51, 51), "difference"),
        (17, (51, 51), "difference"),
        (193, (51, 51), "difference"),
    ],
)
def test_native_varied_gaussian_batches_match_cupy(count, shape, mode):
    cp = _require_native()
    from cuphoton.xfit._native import last_timings

    model, truth, _, initial = _fixture(shape=shape, mode=mode)
    repeats = ((count + 1) // 2, 1)
    truth = np.tile(truth, repeats)[:count].copy()
    initial = np.tile(initial, repeats)[:count].copy()
    # Each row has distinct source positions and amplitudes so comparison
    # against the reference detects misplaced rows throughout the batch.
    offsets = np.linspace(-0.125, 0.125, count)
    for parameters in (truth, initial):
        parameters[:, 0] += offsets
        parameters[:, 4:] += offsets[:, None] * [1.0, -0.5, 0.25, -1.0]
    images = model.evaluate(truth, mode=mode)
    options = dict(
        model=model,
        initial=initial,
        mode=mode,
        variance=np.full(images.shape, 0.5),
        config=LMConfig(max_evaluations=4),
    )
    expected = fit_dipoles(images, backend="cupy", **options)
    actual = fit_dipoles(images, backend="native", **options)
    assert (actual.evaluations > 1).all()
    _assert_fit_matches(actual, expected)
    if count in (16, 17) and shape == (51, 51) and mode == "difference":
        capability, version = _native_device_info(cp)
        _assert_native_dispatch(last_timings(), count, capability, version)


def test_native_fit_with_bright_overlapping_lobes(monkeypatch):
    from cuphoton.xfit import _native, api

    cp = _require_native()
    shape = (51, 51)
    model = GaussianDipoleModel(shape, dtype=np.float64)
    initial = np.asarray(
        [[2**20, 8.0, 4.0, 0.0, -0.25, 0.0, 0.25, 0.0]],
        dtype=np.float64,
    )
    truth = initial.copy()
    truth[:, 1:3] *= [0.99, 1.01]
    truth[:, 3] = 0.02
    truth[:, 4:] += [0.02, -0.01, -0.015, 0.01]
    images = model.evaluate(truth)
    coordinate = np.arange(51, dtype=np.float64) - 25
    radius_squared = coordinate[:, None] ** 2 + coordinate[None, :] ** 2
    weights = 1 + radius_squared[None, ...] / 4096
    # Bright overlapping lobes expose residual errors from small differences
    # in the fitting arithmetic. Two evaluations include one LM trial, so
    # parameter movement checks the backend's solve before final diagnostics.
    options = dict(
        model=model,
        initial=initial,
        variance=1 / (weights * weights),
        config=LMConfig(max_evaluations=2),
    )
    expected = fit_dipoles(images, backend="cupy", **options)
    assert expected.evaluations.tolist() == [2]
    assert np.max(np.abs(expected.parameters - initial)) > 1.0e-4
    actual = fit_dipoles(images, backend="native", **options)
    _assert_fit_matches(actual, expected)

    # Check the damped step itself: tiny solve-rounding differences can move
    # bright-lobe residuals beyond their tolerance at different endpoints.
    jacobian = model.jacobian(initial)[0].reshape(8, -1)
    jacobian[1:3] *= initial[0, 1:3, None]
    jacobian *= weights[0].ravel()
    scale = np.linalg.norm(jacobian, axis=1)
    normalized = jacobian / scale[:, None]
    system = normalized @ normalized.T
    system += options["config"].initial_damping * np.eye(8)
    residual = ((model.evaluate(initial) - images) * weights).ravel()
    gradient = normalized @ residual
    for result in (actual, expected):
        step = result.parameters[0] - initial[0]
        step[1:3] = np.log(result.parameters[0, 1:3] / initial[0, 1:3])
        scaled_step = scale * step
        error = np.linalg.norm(system @ scaled_step + gradient, ord=np.inf)
        magnitude = np.linalg.norm(system, ord=np.inf) * np.linalg.norm(
            scaled_step, ord=np.inf
        ) + np.linalg.norm(gradient, ord=np.inf)
        assert error / magnitude <= 5e-12

    # Public outputs recompute residuals after fitting. Observe the actual
    # LM buffers to compare the two evaluators at identical moved parameters.
    captured = []

    def capture(function):
        def call(*args, **kwargs):
            result = function(*args, **kwargs)
            captured.append(
                (cp.asnumpy(result.parameters), cp.asnumpy(result.residuals))
            )
            return result

        return call

    monkeypatch.setattr(
        api,
        "batched_levenberg_marquardt",
        capture(api.batched_levenberg_marquardt),
    )
    extension = _native.load_extension()
    original_fit = extension.fit

    def capture_native(*args):
        result = original_fit(*args)
        outputs = args[2]
        captured.append((cp.asnumpy(outputs[0]), cp.asnumpy(outputs[3])))
        return result

    monkeypatch.setattr(extension, "fit", capture_native)
    for endpoint in (actual.parameters, expected.parameters):
        captured.clear()
        for backend in ("cupy", "native"):
            fixed = fit_dipoles(
                images,
                **{
                    **options,
                    "initial": endpoint,
                    "config": LMConfig(max_evaluations=1),
                    "backend": backend,
                },
            )
            assert fixed.evaluations.tolist() == [1]
            np.testing.assert_array_equal(fixed.parameters, endpoint)
        assert len(captured) == 2
        np.testing.assert_array_equal(captured[0][0], captured[1][0])
        np.testing.assert_allclose(
            captured[1][1], captured[0][1], rtol=5e-12, atol=5e-12
        )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("backend", ["numpy", "cupy", "native"])
def test_native_gaussian_fit_is_independent_of_flux_units(dtype, backend):
    if backend == "native":
        _require_native()
    elif backend == "cupy":
        _require_cupy()
    exponent = 8 if dtype is np.float32 else 20
    units = np.asarray([10.0**-exponent, 1, 10.0**exponent], dtype=dtype)
    truth = np.tile(
        np.asarray([1, 1.7, 1.1, 0.35, -1.8, 0.7, 2.2, -0.4], dtype=dtype),
        (3, 1),
    )
    truth[:, 0] *= units
    initial = truth.copy()
    initial[:, 0] *= 0.7
    initial[:, 1:3] *= [1.1, 0.9]
    initial[:, 3:] += [0.1, 0.1, -0.1, -0.1, 0.1]
    model = GaussianDipoleModel((25, 25), dtype=dtype)
    images = model.evaluate(truth)
    variance = np.broadcast_to(units[:, None, None] ** 2, images.shape)

    result = fit_dipoles(
        images,
        model=model,
        initial=initial,
        variance=variance,
        backend=backend,
    )

    assert result.converged.all()
    assert result.uncertainty_valid.all()
    assert (result.evaluations > 2).all()
    # Check known truth in noise units, independently of another solver's
    # trajectory. Raw amplitude steps must not set the stopping decision.
    prediction_error = (model.evaluate(result.parameters) - images) / units[
        :, None, None
    ]
    prediction_tolerance = 3e-5 if dtype is np.float32 else 1e-7
    np.testing.assert_allclose(
        prediction_error, 0, atol=prediction_tolerance, rtol=0
    )
    actual = result.parameters.copy()
    expected = truth.copy()
    actual[:, 0] /= units
    expected[:, 0] /= units
    parameter_tolerance = 1e-4 if dtype is np.float32 else 1e-7
    np.testing.assert_allclose(
        actual, expected, atol=parameter_tolerance, rtol=parameter_tolerance
    )


@pytest.mark.parametrize("backend", ["numpy", "cupy", "native"])
def test_fp32_g_tol_rounds_in_residual_precision(backend):
    if backend == "native":
        _require_native()
    elif backend == "cupy":
        _require_cupy()
    model = GaussianDipoleModel((11, 15), dtype=np.float32)
    initial = np.asarray([[5, 1, 2, 0, -2, 0, 2, 0]], dtype=np.float32)
    images = model.evaluate(initial)
    images[0, 5, 7] += 0.25
    # The initial scaled gradient is about 0.03821583012. These two
    # tolerances straddle it in FP64, but round to the same FP32 threshold
    # above it. The residual norm is 0.25, so its threshold factor is one.
    tolerances = (0.0382158296, 0.038215832)
    assert np.float32(tolerances[0]) == np.float32(tolerances[1])
    for tolerance in tolerances:
        result = fit_dipoles(
            images,
            model=model,
            initial=initial,
            backend=backend,
            config=LMConfig(
                g_tol=tolerance, f_tol=0, x_tol=0, max_evaluations=2
            ),
        )
        assert result.status.tolist() == ["converged_g_tol"]
        assert result.evaluations.tolist() == [1]
        np.testing.assert_array_equal(result.parameters, initial)


@pytest.mark.parametrize("backend", ["numpy", "cupy", "native"])
def test_native_f_tol_requires_small_predicted_progress(backend):
    if backend == "native":
        _require_native()
    elif backend == "cupy":
        _require_cupy()
    model = GaussianDipoleModel((11, 15))
    truth = np.asarray([[5, 1.3, 1.8, 0.25, -2, 0.5, 2.2, -0.7]])
    initial = np.asarray(
        [
            [
                2.6057400816822,
                0.6352224510616611,
                1.2229674362952558,
                -0.30532721881901603,
                -1.528461477454861,
                1.5127158178198286,
                2.355429327668466,
                -0.3482435916007965,
            ]
        ]
    )
    images = model.evaluate(truth)
    initial_residual = model.evaluate(initial) - images
    cost = 0.5 * np.sum(initial_residual**2)
    # The damped linear model predicts about 79.34 cost improvement, while
    # the actual first step improves cost by only 0.49. Its cost starts at
    # 102.35, so actual reduction alone incorrectly satisfies f_tol=0.05.
    result = fit_dipoles(
        images,
        model=model,
        initial=initial,
        config=LMConfig(f_tol=0.05, max_evaluations=2),
        backend=backend,
    )

    reduction = cost - 0.5 * np.sum(result.residuals**2)
    assert 0 < reduction < 0.05 * cost
    assert result.status.tolist() == ["max_evaluations"]
    assert result.evaluations.tolist() == [2]


def test_masked_nonfinite_split_data_and_independent_row_convergence():
    _require_native()
    model, truth, _, initial = _fixture(mode="split")
    model = GaussianDipoleModel(
        model.image_shape, split_weights=(1.0, 0.25, 1.75)
    )
    images = model.evaluate(truth, mode="split")
    initial[0] = truth[0]
    mask = np.ones(images.shape, dtype=bool)
    mask[0, 0, 0, :3] = False
    mask[1, 1, 2:4, 5:8] = False
    y, x = np.mgrid[:11, :15]
    variance = np.broadcast_to(
        0.75 + 0.03 * x + 0.05 * y, images.shape
    ).copy()
    images[~mask] = np.nan
    variance[~mask] = np.nan
    options = dict(
        model=model,
        initial=initial,
        mode="split",
        mask=mask,
        variance=variance,
    )
    expected = fit_dipoles(images, backend="cupy", **options)
    actual = fit_dipoles(images, backend="native", **options)
    assert actual.evaluations[0] == 1
    assert actual.evaluations[1] > 1
    _assert_fit_matches(actual, expected)


@pytest.mark.parametrize("case", ["rank-deficient", "budget", "nonfinite"])
@pytest.mark.parametrize("shape", [(11, 15), (51, 51)])
def test_native_failure_statuses_and_diagnostics_match_cupy(case, shape):
    _require_native()
    model, truth, images, initial = _fixture(shape=shape)
    config = LMConfig()
    if case == "rank-deficient":
        truth[:, 0] = 0
        images = model.evaluate(truth)
        initial = truth
    elif case == "budget":
        config = LMConfig(max_evaluations=1)
    else:
        # Finite positive widths can still overflow inverse squared widths;
        # centered source positions make the model residual nonfinite.
        initial[:, 1:3] = np.finfo(np.float64).tiny
        initial[:, 3:8] = 0
    options = dict(model=model, initial=initial, config=config)
    expected = fit_dipoles(images, backend="cupy", **options)
    actual = fit_dipoles(images, backend="native", **options)
    _assert_fit_matches(actual, expected)
    assert np.isnan(actual.covariance).all()
    assert not actual.uncertainty_valid.any()
    if case == "budget":
        assert actual.status.tolist() == ["max_evaluations"] * 2
        assert actual.evaluations.tolist() == [1, 1]
    elif case == "nonfinite":
        assert actual.status.tolist() == ["invalid_residual"] * 2


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("mode", ["difference", "split"])
def test_native_final_diagnostics_preserve_mixed_row_failures(
    dtype, mode, monkeypatch
):
    cp = _require_native()
    from cuphoton.xfit import _native
    from cuphoton.xfit.solver import _factorize_jacobians

    model, truth, _, _ = _fixture(dtype, shape=(7, 9), mode=mode)
    model = GaussianDipoleModel(
        model.image_shape, dtype=dtype, split_weights=(1.0, 0.25, 1.75)
    )
    initial = np.concatenate((truth, truth)).copy()
    initial[1, 0] = 0  # Finite, converged, rank-deficient Jacobian.
    images = model.evaluate(initial, mode=mode)
    initial[2, 1:3] = np.finfo(dtype).tiny
    initial[2, 3:8] = 0  # Nonfinite model, between two healthy rows.
    options = dict(
        model=model,
        initial=initial,
        mode=mode,
        variance=np.full(images.shape, 0.5, dtype=dtype),
    )
    expected = fit_dipoles(images, backend="cupy", **options)
    extension = _native.load_extension()
    original = extension.fit
    captured = []

    def capture(workspace, inputs, outputs, *arguments):
        result = original(workspace, inputs, outputs, *arguments)
        captured.append(outputs)
        return result

    monkeypatch.setattr(extension, "fit", capture)
    actual = fit_dipoles(images, backend="native", **options)
    _assert_fit_matches(actual, expected, dtype)
    np.testing.assert_array_equal(
        actual.uncertainty_valid, [True, False, False, True]
    )
    assert actual.uncertainty_reason[1] == "rank-deficient Jacobian"
    assert actual.status[2] == "invalid_residual"
    outputs = captured[0]
    physical = outputs[0].copy()
    physical[:, 1:3] = cp.exp(physical[:, 1:3])
    reference_model = model.to_backend("cupy")
    jacobian = reference_model._jacobian_positive_unchecked(
        physical, mode=mode
    )
    chain_shape = (len(initial), 2) + (1,) * (jacobian.ndim - 2)
    jacobian[:, 1:3] *= physical[:, 1:3].reshape(chain_shape)
    jacobian = jacobian.reshape(len(initial), 8, -1)
    jacobian *= outputs[2][:, None, :]
    finite = cp.isfinite(jacobian).all(axis=(1, 2))
    selected = jacobian[finite]
    expected_gradient = cp.einsum("knm,km->kn", selected, outputs[3][finite])
    expected_hessian = cp.einsum("knm,kpm->knp", selected, selected)
    _, singular, _ = _factorize_jacobians(selected, cp)
    threshold = (
        max(jacobian.shape[1:]) * np.finfo(dtype).eps * singular.max(axis=1)
    )
    expected_rank = cp.sum(singular > threshold[:, None], axis=1)
    tolerance = 3.0e-5 if dtype is np.float32 else 5.0e-12
    for index, reference in ((26, expected_gradient), (27, expected_hessian)):
        np.testing.assert_allclose(
            cp.asnumpy(outputs[index][finite]),
            cp.asnumpy(reference),
            rtol=tolerance,
            atol=tolerance,
        )
        assert bool(cp.isnan(outputs[index][~finite]).all())
    np.testing.assert_array_equal(
        cp.asnumpy(outputs[28][finite]), cp.asnumpy(expected_rank)
    )
    np.testing.assert_array_equal(cp.asnumpy(outputs[28][~finite]), [-1])
    # Singular vector signs are arbitrary. Propagated variance must agree
    # without subtracting rounded entries of the dense covariance.
    gradient = np.asarray([0.5, -1, 2, 0.25, -1, 0, 1, 0], dtype=dtype)
    for result in (actual, expected):
        assert np.isnan(
            result.covariance_factor[~result.uncertainty_valid]
        ).all()
    np.testing.assert_allclose(
        np.sum((gradient @ actual.covariance_factor) ** 2, axis=1),
        np.sum((gradient @ expected.covariance_factor) ** 2, axis=1),
        rtol=tolerance,
        atol=tolerance,
        equal_nan=True,
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("count", [1, 3])
def test_native_null_improvement_preserves_cancelling_chi_squares(
    dtype, count
):
    cp = _require_native()
    rng = np.random.default_rng(813)
    images = rng.normal(125, 5, (count, 51, 51)).astype(dtype)
    variance = rng.uniform(1, 2, images.shape).astype(dtype)
    initial = np.tile(
        np.asarray([10, 1e30, 1e5, 0.5, 50, 10, -50, 20], dtype=dtype),
        (count, 1),
    )
    actual = fit_dipoles(
        images,
        model=GaussianDipoleModel((51, 51), dtype=dtype),
        initial=initial,
        variance=variance,
        backend="native",
        config=LMConfig(max_evaluations=1),
    )
    weights = 1 / cp.sqrt(cp.asarray(variance))
    weighted_residuals = cp.asarray(actual.residuals) * weights
    weighted_null = cp.asarray(images) * weights
    chi = cp.asnumpy(
        cp.sum(weighted_residuals * weighted_residuals, axis=(1, 2))
    )
    null_chi = cp.asnumpy(cp.sum(weighted_null * weighted_null, axis=(1, 2)))
    # The broad dipole is nearly zero. An ULP in either large sum becomes a
    # material error in their small difference, so match the current-device
    # reduction before comparing the cancellation-sensitive public diagnostic.
    np.testing.assert_array_equal(actual.chi_square, chi)
    np.testing.assert_array_equal(actual.null_chi_square, null_chi)
    np.testing.assert_allclose(
        actual.delta_chi_square,
        null_chi - chi,
        rtol=5e-12,
        atol=5e-12,
    )


@pytest.mark.parametrize("count", [1, 41])
def test_native_final_gradient_preserves_cancelling_products(
    count, monkeypatch
):
    cp = _require_native()
    from cuphoton.xfit import _native

    model, truth, _, _ = _fixture(shape=(51, 51))
    initial = np.tile(truth[:1], (count, 1))
    initial[:, 0] *= 1e6
    jacobian = model.jacobian(initial).reshape(count, 8, -1)
    noise = np.random.default_rng(431).normal(0, 10, (count, 51 * 51))
    for row in range(count):
        orthogonal, _ = np.linalg.qr(jacobian[row].T)
        noise[row] -= orthogonal @ (orthogonal.T @ noise[row])
    images = model.evaluate(initial) + noise.reshape(count, 51, 51)
    if count > 1:
        initial[1, 1:3] = np.finfo(np.float64).tiny
        initial[1, 3:8] = 0

    extension = _native.load_extension()
    original = extension.fit
    captured = []

    def capture(workspace, inputs, outputs, *arguments):
        result = original(workspace, inputs, outputs, *arguments)
        captured.append(outputs)
        return result

    monkeypatch.setattr(extension, "fit", capture)
    fit_dipoles(
        images,
        model=model,
        initial=initial,
        backend="native",
        config=LMConfig(max_evaluations=2, g_tol=1),
    )
    outputs = captured[0]
    physical = outputs[0].copy()
    physical[:, 1:3] = cp.exp(physical[:, 1:3])
    jacobian = model.to_backend("cupy")._jacobian_positive_unchecked(physical)
    jacobian[:, 1:3] *= physical[:, 1:3, None, None]
    jacobian = jacobian.reshape(count, 8, -1) * outputs[2][:, None, :]
    finite = cp.isfinite(jacobian).all(axis=(1, 2))
    assert int(finite.sum()) == (count if count == 1 else count - 1)
    expected = cp.einsum("knm,km->kn", jacobian[finite], outputs[3][finite])
    # Orthogonal noise makes the final gradient cancellation-sensitive despite
    # large individual Jacobian/residual products. Operand layout matters.
    np.testing.assert_allclose(
        cp.asnumpy(outputs[26][finite]),
        cp.asnumpy(expected),
        rtol=5e-12,
        atol=5e-12,
    )
    assert bool(cp.isnan(outputs[26][~finite]).all())
    expected_hessian = cp.einsum(
        "knm,kpm->knp", jacobian[finite], jacobian[finite]
    )
    np.testing.assert_allclose(
        cp.asnumpy(outputs[27][finite]),
        cp.asnumpy(expected_hessian),
        rtol=5e-12,
        atol=5e-12,
    )
    assert bool(cp.isnan(outputs[27][~finite]).all())


@pytest.mark.parametrize(
    "bad_rows",
    [(0,), (1,), (2,), (0, 1, 2)],
    ids=["first", "middle", "last", "all"],
)
@pytest.mark.parametrize("shape", [(11, 15), (51, 51)])
def test_native_filters_nonfinite_jacobians_without_reordering_rows(
    bad_rows, shape
):
    cp = _require_native()
    from cuphoton.xfit._native import last_timings

    model, truth, _, initial = _fixture(shape=shape)
    truth = np.concatenate((truth, truth[:1]))
    initial = np.concatenate((initial, initial[:1]))
    # Distinct amplitudes make a survivor-row permutation observable.
    truth[:, 0] += np.arange(3)
    initial[:, 0] += np.arange(3)
    images = model.evaluate(truth)
    bad = np.zeros(3, dtype=bool)
    bad[list(bad_rows)] = True
    # Inverse squared widths remain finite, while cubed widths underflow.
    # These rows reach Jacobian filtering after valid initial residuals.
    initial[bad, 1:3] = 1.0e-110
    reference = model.to_backend("cupy")
    assert bool(
        cp.isfinite(reference.evaluate(initial) - cp.asarray(images)).all()
    )
    finite_jacobian = cp.asnumpy(
        cp.isfinite(reference.jacobian(initial)).reshape(3, -1).all(axis=1)
    )
    np.testing.assert_array_equal(finite_jacobian, ~bad)
    options = dict(
        model=model,
        initial=initial,
        variance=np.full(images.shape, 0.5),
        config=LMConfig(max_evaluations=2, f_tol=0, x_tol=0, g_tol=0),
    )
    expected = fit_dipoles(images, backend="cupy", **options)
    actual = fit_dipoles(images, backend="native", **options)
    _assert_fit_matches(actual, expected)
    np.testing.assert_array_equal(
        actual.status, np.where(bad, "invalid_residual", "max_evaluations")
    )
    np.testing.assert_array_equal(actual.evaluations, np.where(bad, 1, 2))
    assert last_timings()["iterations"] == (1 if bad.all() else 2)
    if not bad.all():
        assert np.all(
            np.max(np.abs(actual.parameters[~bad] - initial[~bad]), axis=1)
            > 1.0e-4
        )
    assert actual.backend == "native"
    for name in ("parameter_names", "device", "dtype", "model", "mode"):
        assert getattr(actual, name) == getattr(expected, name)


def test_repeated_calls_own_outputs_and_recover_after_invalid_configuration():
    _require_native()
    from cuphoton.xfit._native import release_workspace

    retained = []
    try:
        for shape, dtype in (
            ((7, 9), np.float64),
            ((13, 17), np.float32),
            ((9, 11), np.float64),
        ):
            model, _, images, initial = _fixture(dtype, shape=shape)
            with pytest.raises(ValueError, match="max_evaluations"):
                fit_dipoles(
                    images,
                    model=model,
                    initial=initial,
                    backend="native",
                    config=LMConfig(max_evaluations=0),
                )
            result = fit_dipoles(
                images,
                model=model,
                initial=initial,
                backend="native",
            )
            assert result.converged.all()
            retained.append(
                (result, result.parameters.copy(), result.residuals.copy())
            )
        release_workspace()
        gc.collect()
        for result, parameters, residuals in retained:
            np.testing.assert_array_equal(result.parameters, parameters)
            np.testing.assert_array_equal(result.residuals, residuals)
    finally:
        release_workspace()


@pytest.mark.parametrize("backend", ["cupy", "native"])
@pytest.mark.parametrize(
    "case,message",
    [
        ("images", "images must be finite on included pixels"),
        (
            "variance",
            "variance must be finite and positive on included pixels",
        ),
        ("mask", "mask must contain only finite values"),
    ],
)
def test_native_rejects_invalid_inputs_like_cupy(backend, case, message):
    _require_native()
    model, _, images, initial = _fixture()
    arrays = {
        "images": images.copy(),
        "variance": np.ones_like(images),
        "mask": np.ones_like(images),
    }
    arrays[case][0, 3, 4] = 0.0 if case == "variance" else np.nan
    with pytest.raises(ValueError) as error:
        fit_dipoles(**arrays, model=model, initial=initial, backend=backend)
    assert str(error.value) == message


@pytest.mark.parametrize("overlap", ["outputs", "input"])
def test_native_fit_rejects_overlapping_buffers(overlap):
    cp = _require_native()
    from cuphoton.xfit import _native_ext as extension

    stream = cp.cuda.Stream(non_blocking=True)
    inputs, outputs, settings, dimensions, split = _native_fit_arguments(
        cp, stream
    )
    invalid_outputs = list(outputs)
    if overlap == "outputs":
        invalid_outputs[1] = outputs[2]
        message = "native fit output buffers must not overlap"
    else:
        invalid_outputs[1] = inputs[0].reshape(dimensions[0], -1)
        message = "native fit outputs must not overlap inputs"
    # These pointers are valid, but aliasing must fail before capsule access.
    with stream, pytest.raises(ValueError, match=message):
        extension.fit(
            None,
            inputs,
            tuple(invalid_outputs),
            settings,
            dimensions,
            split,
            stream.ptr,
        )


@pytest.mark.parametrize("weight", [-1.0, np.nan, np.inf])
def test_native_fit_rejects_invalid_split_weights(weight):
    extension = _require_extension()
    with pytest.raises(ValueError, match="invalid native split weights"):
        extension.fit(
            None,
            (None,) * 4,
            (None,) * 29,
            (1e-8, 1e-8, 1e-8, 1e-3, 10.0, 0.3, 100),
            (1, 11, 15, 3),
            (1.0, weight, 1.0),
            0,
        )


def test_invalid_cuda_device_does_not_poison_the_next_native_fit():
    cp = _require_native()
    from cuphoton.xfit import _native_ext as extension

    device = int(cp.cuda.Device().id)
    stream = cp.cuda.Stream(non_blocking=True)
    workspace = extension.create_workspace(device)
    with stream:
        baseline = _native_fit_arguments(cp, stream)
        retry = _native_fit_arguments(cp, stream)
        extension.fit(workspace, *baseline, stream.ptr)
        expected = [
            cp.asnumpy(baseline[1][index]) for index in (8, 9, 4, 11, 10)
        ]
        with pytest.raises(RuntimeError, match="invalid device ordinal"):
            extension.create_workspace(2**31 - 1)
        # No intervening CuPy allocation/copy or last-error query may clear
        # the consumed runtime error before this immediate retry.
        extension.fit(workspace, *retry, stream.ptr)
        for index, reference in zip((8, 9, 4, 11, 10), expected, strict=True):
            actual = cp.asnumpy(retry[1][index])
            assert actual.dtype == reference.dtype
            assert actual.shape == reference.shape
            assert actual.tobytes() == reference.tobytes()


def test_threaded_calls_respect_producer_stream_and_input_lifetime():
    cp = _require_native()
    from cuphoton.xfit._native import release_workspace

    model, _, images, initial = _fixture()
    expected = fit_dipoles(
        images, model=model, initial=initial, backend="cupy"
    )
    device = int(cp.cuda.Device().id)
    barrier = Barrier(4)

    def worker(index):
        with cp.cuda.Device(device):
            stream = cp.cuda.Stream(non_blocking=True)
            try:
                barrier.wait(timeout=30)
                retained = []
                for _ in range(2):
                    with stream:
                        # Asynchronous creation on a nondefault producer
                        # stream must precede the extension's private stream.
                        data = cp.asarray(images)
                        start = cp.asarray(initial)
                        result = fit_dipoles_device(
                            data,
                            model=model,
                            initial=start,
                            backend="native",
                        )
                        del data, start
                        retained.append(result)
                    stream.synchronize()
                release_workspace()
                gc.collect()
                for result in retained:
                    assert result.backend == "native"
                    assert bool(result.converged.all().item())
                    np.testing.assert_allclose(
                        cp.asnumpy(result.parameters),
                        expected.parameters,
                        rtol=5.0e-12,
                        atol=5.0e-12,
                    )
                return index
            finally:
                stream.synchronize()
                release_workspace()

    with ThreadPoolExecutor(max_workers=4) as executor:
        assert sorted(executor.map(worker, range(4))) == list(range(4))


def test_shared_workspace_rejects_collision_and_can_be_reused():
    cp = _require_native()
    from cuphoton.xfit import _native_ext as extension

    device = int(cp.cuda.Device().id)
    stream = cp.cuda.Stream(non_blocking=True)
    workspace = extension.create_workspace(device)

    def run(arguments):
        with cp.cuda.Device(device), stream:
            return extension.fit(workspace, *arguments, stream.ptr)

    run(_native_fit_arguments(cp, stream))
    arguments = [_native_fit_arguments(cp, stream) for _ in range(2)]
    with ThreadPoolExecutor(max_workers=2) as executor:
        release, entered, expired = _gate_producer(stream)
        try:
            assert entered.wait(timeout=10)
            futures = [executor.submit(run, value) for value in arguments]
            done, pending = wait(
                futures, timeout=10, return_when=FIRST_COMPLETED
            )
            # The winner cannot finish until its producer dependency clears.
            # This forces an actual overlap without relying on fit duration.
            assert len(done) == len(pending) == 1
            with pytest.raises(RuntimeError, match="already in use"):
                next(iter(done)).result()
        finally:
            release.set()
            stream.synchronize()
        winner = next(iter(pending))
        assert winner.result(timeout=30)["iterations"] > 0
    assert not expired.is_set()
    successful = arguments[futures.index(winner)][1]
    retry = _native_fit_arguments(cp, stream)
    run(retry)
    np.testing.assert_array_equal(
        cp.asnumpy(retry[1][8]), cp.asnumpy(successful[8])
    )
    assert bool(cp.isin(retry[1][4], cp.asarray([1, 2, 3])).all().item())


def _exercise_native_sigint():
    """Run only in a subprocess: SIGINT must never reach the pytest parent."""
    import signal

    cp = _require_native()
    from cuphoton.xfit import _native_ext as extension

    device = int(cp.cuda.Device().id)
    stream = cp.cuda.Stream(non_blocking=True)
    workspace = extension.create_workspace(device)
    warmup = _native_fit_arguments(cp, stream)
    with stream:
        extension.fit(workspace, *warmup, stream.ptr)
    # If the sender wins the capsule race, release it and retry. A signal is
    # sent only after busy rejection proves the main thread owns native work.
    for _ in range(8):
        arguments = [_native_fit_arguments(cp, stream) for _ in range(2)]
        release, entered, expired = _gate_producer(stream)
        begin, sent = Event(), Event()
        errors = []

        def sender(begin, other_arguments, release, sent, errors):
            try:
                assert begin.wait(timeout=10)
                with cp.cuda.Device(device), stream:
                    extension.fit(workspace, *other_arguments, stream.ptr)
            except RuntimeError as exc:
                if "already in use" not in str(exc):
                    errors.append(exc)
                else:
                    if release.is_set():
                        errors.append(
                            AssertionError(
                                "producer gate opened before SIGINT"
                            )
                        )
                    else:
                        sent.set()
                        os.kill(os.getpid(), signal.SIGINT)
            except BaseException as exc:
                errors.append(exc)
            finally:
                release.set()

        thread = Thread(
            target=sender, args=(begin, arguments[1], release, sent, errors)
        )
        interrupted = False
        try:
            assert entered.wait(timeout=10)
            thread.start()
            with stream:
                begin.set()
                try:
                    extension.fit(workspace, *arguments[0], stream.ptr)
                except KeyboardInterrupt:
                    interrupted = True
                except RuntimeError as exc:
                    assert "already in use" in str(exc)
        finally:
            release.set()
            if thread.ident is not None:
                thread.join(timeout=30)
            stream.synchronize()
        assert not thread.is_alive()
        assert not expired.is_set()
        assert not errors, errors
        if interrupted:
            assert sent.is_set()
            first = cp.asnumpy(arguments[0][1][8])
            retry = _native_fit_arguments(cp, stream)
            with stream:
                extension.fit(workspace, *retry, stream.ptr)
            np.testing.assert_array_equal(cp.asnumpy(retry[1][8]), first)
            assert bool(
                cp.isin(retry[1][4], cp.asarray([1, 2, 3])).all().item()
            )
            return
    raise AssertionError("main thread never acquired the shared workspace")


def test_sigint_drains_native_work_before_raising():
    _require_native()
    source = (
        "import runpy; "
        f"runpy.run_path({str(Path(__file__).resolve())!r})"
        "['_exercise_native_sigint']()"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parents[2] / "src")
    completed = subprocess.run(
        [sys.executable, "-c", source],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_cli_runs_native_fit_and_records_selected_backend(tmp_path, capsys):
    _require_native()
    import pyarrow.parquet as pq
    import yaml

    model, truth, images, initial = _fixture()
    input_path = tmp_path / "gaussian.npz"
    output_dir = tmp_path / "native-output"
    np.savez_compressed(
        input_path,
        candidate_id=np.asarray(["native-a", "native-b"]),
        images=images,
        initial=initial,
        variance=np.ones_like(images),
    )
    argv = [
        "fit-dipoles",
        "--input",
        str(input_path),
        "--output-dir",
        str(output_dir),
        "--model",
        "gaussian",
        "--backend",
        "native",
    ]
    rc = run_component("xfit", argv)
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert captured.err == ""
    summary = json.loads(captured.out)
    assert summary["backend"] == "native"
    assert summary["metrics"]["converged_count"] == 2
    effective = yaml.safe_load(
        (output_dir / "effective-config.yaml").read_text()
    )
    assert effective["backend"] == "native"
    rows = pq.read_table(output_dir / "fits.parquet").to_pylist()
    assert [row["candidate_id"] for row in rows] == ["native-a", "native-b"]
    parameters = np.asarray(
        [[row[name] for name in model.parameter_names] for row in rows]
    )
    np.testing.assert_allclose(parameters, truth, rtol=1e-5, atol=1e-5)
    with np.load(output_dir / "fit-arrays.npz", allow_pickle=False) as arrays:
        assert np.isfinite(arrays["covariance"]).all()
        assert arrays["residuals"].shape == images.shape


def test_native_public_pipeline_disables_cupy_compilation_in_fresh_process(
    tmp_path,
):
    """No on-disk or process-local CuPy kernel cache can hide a dependency."""
    _require_native()
    source = textwrap.dedent(
        """
        import numpy as np
        import cupy as cp
        import cupy.cuda.compiler as compiler
        from cuphoton.xfit import (
            GaussianDipoleModel, fit_dipoles, fit_dipoles_device,
        )
        from cuphoton.xfit._native import last_timings

        def forbidden(*args, **kwargs):
            raise AssertionError("CuPy runtime compilation attempted")

        compiler._compile_module_with_cache = forbidden
        compiler.compile_using_nvrtc = forbidden
        compiler.compile_using_nvcc = forbidden
        for dtype in (np.float32, np.float64):
            truth = np.asarray([
                [5, 1.3, 1.8, .25, -2, .5, 2.2, -.7],
                [3, 1.6, 1.1, -.3, -1, -1, 2, 1],
            ], dtype=dtype)
            model = GaussianDipoleModel((11, 15), dtype=dtype)
            for mode in ("difference", "split"):
                images = model.evaluate(truth, mode=mode)
                for device_result in (False, True):
                    fit = fit_dipoles_device if device_result else fit_dipoles
                    result = fit(
                        images, model=model, initial=truth, mode=mode,
                        variance=np.ones((11, 15), dtype=dtype),
                        backend="native",
                    )
                    assert result.backend == "native"
                    assert last_timings()["native_pipeline"] is True
                    converged = (
                        cp.asnumpy(result.converged) if device_result
                        else result.converged
                    )
                    assert converged.all()
                    assert result.covariance.shape == (2, 8, 8)
                # Exercise native reductions and default initial estimation.
                result = fit_dipoles(
                    images, model=model, mode=mode, backend="native",
                    mask=np.ones((11, 15), dtype=np.uint8),
                )
                assert result.parameters.shape == (2, 8)
        """
    )
    environment = os.environ.copy()
    environment["CUPY_CACHE_DIR"] = str(tmp_path / "cupy-empty")
    environment["CUDA_CACHE_PATH"] = str(tmp_path / "cuda-empty")
    completed = subprocess.run(
        [sys.executable, "-c", source],
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert not list((tmp_path / "cupy-empty").glob("**/*.cubin"))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("mode", ["difference", "split"])
def test_native_initial_estimate_and_strided_broadcast_inputs_match_cupy(
    dtype, mode
):
    cp = _require_native()
    model, _, images, _ = _fixture(dtype, mode=mode)
    # Reverse spatial axes without materializing a contiguous GPU copy.
    images = cp.asarray(images)[..., ::-1, ::-1]
    mask = np.ones(model.image_shape, dtype=np.uint8)
    mask[0, :3] = 0
    variance = np.full(model.image_shape, 0.5, dtype=dtype)
    options = dict(model=model, mask=mask, variance=variance, mode=mode)
    expected = fit_dipoles(images, backend="cupy", **options)
    actual = fit_dipoles(images, backend="native", **options)
    _assert_fit_matches(actual, expected, dtype)
