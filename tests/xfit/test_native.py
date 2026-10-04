# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Scientific and ownership coverage for the optional native Gaussian path."""

from __future__ import annotations

import gc
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


def _require_native():
    # Missing optional builds must skip without initializing CUDA.
    pytest.importorskip("cuphoton.xfit._native_ext")
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


def _assert_fit_matches(actual, expected, dtype=np.float64):
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
                if fullname == "cupy" or fullname.endswith("._native_ext"):
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

    original_import = _native.importlib.import_module

    def without_extension(name, *args, **kwargs):
        if name == "cuphoton.xfit._native_ext":
            raise ModuleNotFoundError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(_native.importlib, "import_module", without_extension)
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
    extension = pytest.importorskip("cuphoton.xfit._native_ext")
    with pytest.raises(OverflowError, match="negative.*unsigned"):
        extension.run(None, None, None, None, -1)


@pytest.mark.parametrize("invalid", ["shape", "readonly"])
def test_extension_rejects_unsafe_buffers_without_a_cuda_workspace(invalid):
    extension = pytest.importorskip("cuphoton.xfit._native_ext")
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


@pytest.mark.parametrize(
    "count,shape,mode",
    [
        (17, (5, 7), "difference"),
        (193, (33, 35), "difference"),
        (769, (17, 19), "split"),
    ],
)
def test_native_varied_gaussian_batches_match_cupy(count, shape, mode):
    _require_native()
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


def test_native_fit_with_bright_overlapping_lobes():
    _require_native()
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
def test_native_failure_statuses_and_diagnostics_match_cupy(case):
    _require_native()
    model, truth, images, initial = _fixture()
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


@pytest.mark.parametrize(
    "bad_rows",
    [(0,), (1,), (2,), (0, 1, 2)],
    ids=["first", "middle", "last", "all"],
)
def test_native_filters_nonfinite_jacobians_without_reordering_rows(bad_rows):
    cp = _require_native()
    from cuphoton.xfit._native import last_timings

    model, truth, _, initial = _fixture()
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


def test_invalid_cuda_device_does_not_poison_the_next_native_fit():
    cp = _require_native()
    from cuphoton.xfit import _native_ext as extension

    device = int(cp.cuda.Device().id)
    stream = cp.cuda.Stream(non_blocking=True)
    workspace = extension.create_workspace(device)
    with stream:
        baseline = _native_arguments(cp, stream)
        retry = _native_arguments(cp, stream)
        extension.run(workspace, *baseline, stream.ptr)
        expected = [
            cp.asnumpy(baseline[0][index]) for index in (0, 3, 4, 5, 6)
        ]
        with pytest.raises(RuntimeError, match="invalid device ordinal"):
            extension.create_workspace(2**31 - 1)
        # No intervening CuPy allocation/copy or last-error query may clear
        # the consumed runtime error before this immediate retry.
        extension.run(workspace, *retry, stream.ptr)
        for index, reference in zip((0, 3, 4, 5, 6), expected, strict=True):
            actual = cp.asnumpy(retry[0][index])
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
            return extension.run(workspace, *arguments, stream.ptr)

    run(_native_arguments(cp, stream))
    arguments = [_native_arguments(cp, stream) for _ in range(2)]
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
    successful = arguments[futures.index(winner)][0]
    retry = _native_arguments(cp, stream)
    run(retry)
    np.testing.assert_array_equal(
        cp.asnumpy(retry[0][0]), cp.asnumpy(successful[0])
    )
    assert bool(cp.isin(retry[0][4], cp.asarray([1, 2, 3])).all().item())


def _exercise_native_sigint():
    """Run only in a subprocess: SIGINT must never reach the pytest parent."""
    import signal

    cp = _require_native()
    from cuphoton.xfit import _native_ext as extension

    device = int(cp.cuda.Device().id)
    stream = cp.cuda.Stream(non_blocking=True)
    workspace = extension.create_workspace(device)
    warmup = _native_arguments(cp, stream)
    with stream:
        extension.run(workspace, *warmup, stream.ptr)
    # If the sender wins the capsule race, release it and retry. A signal is
    # sent only after busy rejection proves the main thread owns native work.
    for _ in range(8):
        arguments = [_native_arguments(cp, stream) for _ in range(2)]
        release, entered, expired = _gate_producer(stream)
        begin, sent = Event(), Event()
        errors = []

        def sender(begin, other_arguments, release, sent, errors):
            try:
                assert begin.wait(timeout=10)
                with cp.cuda.Device(device), stream:
                    extension.run(workspace, *other_arguments, stream.ptr)
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
                    extension.run(workspace, *arguments[0], stream.ptr)
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
            first = cp.asnumpy(arguments[0][0][0])
            retry = _native_arguments(cp, stream)
            with stream:
                extension.run(workspace, *retry, stream.ptr)
            np.testing.assert_array_equal(cp.asnumpy(retry[0][0]), first)
            assert bool(
                cp.isin(retry[0][4], cp.asarray([1, 2, 3])).all().item()
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
