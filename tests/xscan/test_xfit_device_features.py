# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import inspect
import pickle
import subprocess
import sys
import weakref
from dataclasses import FrozenInstanceError, replace
from typing import Any

import numpy as np
import pytest

import cuphoton.xscan.xfit_features as xfit_features_module
from cuphoton.xfit import (
    DeviceDipoleFitResult,
    DipoleFitResult,
    GaussianDipoleModel,
    fit_dipoles_device,
)
from cuphoton.xscan.xfit_features import (
    FEATURE_NAMES,
    DeviceXFitFeatures,
    transform_xfit_result_features,
    transform_xfit_result_features_device,
)

_GAUSSIAN_PARAMETERS = (
    "amplitude",
    "sigma_x",
    "sigma_y",
    "theta",
    "x_pos",
    "y_pos",
    "x_neg",
    "y_neg",
)
_STAMP_PARAMETERS = ("x_pos", "y_pos", "x_neg", "y_neg", "flux")
_IMAGE_SHAPE = (11, 13)


def _require_cupy_device(*, minimum_count: int = 1) -> Any:
    cp = pytest.importorskip("cupy")
    try:
        count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:
        pytest.skip(f"CuPy CUDA runtime is not usable: {exc}")
    if count < minimum_count:
        pytest.skip(
            f"test requires {minimum_count} visible CUDA device(s); "
            f"got {count}"
        )
    return cp


def _source_values(
    *, model: str, dtype: np.dtype[Any]
) -> dict[str, np.ndarray]:
    if model == "gaussian":
        parameters = np.asarray(
            [
                [-10.0, 2.0, 1.0, 0.0, -2.0, 0.0, 2.0, 0.0],
                [10.0, 2.0, 1.0, 0.0, -2.0, 0.0, 2.0, 0.0],
                [10.0, 2.0, 1.0, 0.0, 6.01, 0.0, 2.0, 0.0],
                [10.0, 2.0, 1.0, 0.0, -2.0, 0.0, 2.0, 0.0],
                [10.0, 2.0, 1.0, 0.0, -2.0, 0.0, 2.0, 0.0],
                [10.0, 2.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                [10.0, 5.0, 1.0, 0.0, 6.0, 5.0, -6.0, -5.0],
                [10.0, 0.0, 1.0, 0.0, -2.0, 0.0, 2.0, 0.0],
                [10.0, 1.0, 2.0, 0.25, -2.0, -1.0, 2.0, 1.0],
                [10.0, 2.0, 1.0, 0.0, np.nan, 0.0, 2.0, 0.0],
                [10.0, 2.0, 1.0, 0.0, -2.0, 0.0, 2.0, 0.0],
                [10.0, 2.0, 1.0, 0.0, -2.0, 0.0, 2.0, 0.0],
            ],
            dtype=dtype,
        )
        strength_index = 0
    else:
        parameters = np.asarray(
            [
                [-2.0, -1.0, 2.0, 1.0, -20.0],
                [-2.0, -1.0, 2.0, 1.0, 20.0],
                [6.01, 0.0, 2.0, 1.0, 20.0],
                [-2.0, -1.0, 2.0, 1.0, 20.0],
                [-2.0, -1.0, 2.0, 1.0, 20.0],
                [1.0, 1.0, 1.0, 1.0, 20.0],
                [6.0, 5.0, -6.0, -5.0, 20.0],
                [-2.0, -1.0, 2.0, 1.0, np.nan],
            ],
            dtype=dtype,
        )
        strength_index = 4

    batch_size, parameter_count = parameters.shape
    converged = np.ones(batch_size, dtype=bool)
    converged[1] = False
    degrees_of_freedom = np.full(batch_size, 100, dtype=np.int64)
    if model == "gaussian":
        degrees_of_freedom[7] = 0
    valid_pixel_fraction = np.full(batch_size, 108.0 / 143.0)
    fractional_null_improvement = np.full(batch_size, 0.75)
    delta_chi_square = np.full(batch_size, 75.0, dtype=dtype)
    reduced_chi_square = np.ones(batch_size)
    uncertainty_valid = converged.copy()
    uncertainty_valid[3] = False
    standard_errors = np.ones_like(parameters)
    standard_errors[:, strength_index] = 2.0
    standard_errors[3, strength_index] = np.nan
    covariance = np.repeat(
        np.eye(parameter_count, dtype=dtype)[None, ...],
        batch_size,
        axis=0,
    )

    valid_pixel_fraction[4] = 1.25
    fractional_null_improvement[4] = -2.0
    delta_chi_square[4] = np.nan
    reduced_chi_square[4] = np.nan
    if model == "gaussian":
        covariance[10, 0, 0] = np.nan
        x_pos = _GAUSSIAN_PARAMETERS.index("x_pos")
        covariance[11, x_pos, x_pos] = -2.0

    return {
        "parameters": parameters,
        "converged": converged,
        "degrees_of_freedom": degrees_of_freedom,
        "valid_pixel_fraction": valid_pixel_fraction,
        "fractional_null_improvement": fractional_null_improvement,
        "delta_chi_square": delta_chi_square,
        "reduced_chi_square": reduced_chi_square,
        "uncertainty_valid": uncertainty_valid,
        "standard_errors": standard_errors,
        "covariance": covariance,
    }


def _make_results(
    cp: Any,
    *,
    model: str,
    dtype: type[np.float32] | type[np.float64],
    residual_fill: float = 0.0,
) -> tuple[DipoleFitResult, DeviceDipoleFitResult]:
    numpy_dtype = np.dtype(dtype)
    source = _source_values(model=model, dtype=numpy_dtype)
    parameters = source["parameters"]
    batch_size, parameter_count = parameters.shape
    parameter_names = (
        _GAUSSIAN_PARAMETERS if model == "gaussian" else _STAMP_PARAMETERS
    )
    converged = source["converged"]
    status_codes = np.where(converged, 1, 4).astype(np.int8)
    status = np.where(converged, "converged_f_tol", "max_evaluations")
    evaluations = np.full(batch_size, 12, dtype=np.int64)
    residual_norm = np.full(batch_size, 5.0, dtype=numpy_dtype)
    chi_square = np.full(batch_size, 25.0, dtype=numpy_dtype)
    valid_pixel_count = np.full(batch_size, 108, dtype=np.int64)
    null_chi_square = np.full(batch_size, 100.0, dtype=numpy_dtype)
    uncertainty_reason_codes = np.where(converged, 0, 1).astype(np.int8)
    uncertainty_reason = tuple(
        "" if value else "fit did not converge: max_evaluations"
        for value in converged
    )
    residuals = np.full(
        (batch_size, *_IMAGE_SHAPE),
        residual_fill,
        dtype=numpy_dtype,
    )

    portable = DipoleFitResult(
        parameters=parameters,
        parameter_names=parameter_names,
        status=status,
        converged=converged,
        evaluations=evaluations,
        residual_norm=residual_norm,
        chi_square=chi_square,
        valid_pixel_count=valid_pixel_count,
        valid_pixel_fraction=source["valid_pixel_fraction"],
        null_chi_square=null_chi_square,
        delta_chi_square=source["delta_chi_square"],
        fractional_null_improvement=source["fractional_null_improvement"],
        degrees_of_freedom=source["degrees_of_freedom"],
        reduced_chi_square=source["reduced_chi_square"],
        covariance=source["covariance"],
        standard_errors=source["standard_errors"],
        uncertainty_valid=source["uncertainty_valid"],
        uncertainty_reason=uncertainty_reason,
        residuals=residuals,
        backend="numpy",
        device="cpu",
        dtype=numpy_dtype.name,
        model=model,
        mode="difference",
    )
    device = DeviceDipoleFitResult(
        parameters=cp.asarray(parameters),
        parameter_names=parameter_names,
        status_codes=cp.asarray(status_codes),
        converged=cp.asarray(converged),
        evaluations=cp.asarray(evaluations),
        residual_norm=cp.asarray(residual_norm),
        chi_square=cp.asarray(chi_square),
        valid_pixel_count=cp.asarray(valid_pixel_count),
        valid_pixel_fraction=cp.asarray(source["valid_pixel_fraction"]),
        null_chi_square=cp.asarray(null_chi_square),
        delta_chi_square=cp.asarray(source["delta_chi_square"]),
        fractional_null_improvement=cp.asarray(
            source["fractional_null_improvement"]
        ),
        degrees_of_freedom=cp.asarray(source["degrees_of_freedom"]),
        reduced_chi_square=cp.asarray(source["reduced_chi_square"]),
        covariance=cp.asarray(source["covariance"]),
        standard_errors=cp.asarray(source["standard_errors"]),
        uncertainty_valid=cp.asarray(source["uncertainty_valid"]),
        uncertainty_reason_codes=cp.asarray(uncertainty_reason_codes),
        residuals=cp.asarray(residuals),
        device_id=int(cp.cuda.runtime.getDevice()),
        dtype=numpy_dtype.name,
        model=model,
        mode="difference",
    )
    return portable, device


@pytest.mark.parametrize("model", ["gaussian", "stamp"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("variance_present", [False, True])
def test_device_features_match_host_oracle_without_array_transfers(
    model: str,
    dtype: type[np.float32] | type[np.float64],
    variance_present: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cp = _require_cupy_device()
    portable, device = _make_results(cp, model=model, dtype=dtype)
    expected = transform_xfit_result_features(
        portable,
        image_shape=_IMAGE_SHAPE,
        variance_present=variance_present,
    )
    original_asnumpy = cp.asnumpy

    def reject_transfer(*_args: object, **_kwargs: object) -> None:
        pytest.fail("device feature transform attempted a host transfer")

    with monkeypatch.context() as context:
        context.setattr(cp, "asnumpy", reject_transfer)
        context.setattr(cp, "asarray", reject_transfer)
        features = transform_xfit_result_features_device(
            device,
            image_shape=_IMAGE_SHAPE,
            variance_present=variance_present,
        )

    actual = original_asnumpy(features.values)
    assert np.allclose(actual, expected, rtol=2e-6, atol=2e-7)
    for name in (
        "fit_present",
        "fit_valid",
        "uncertainty_valid",
        "variance_weighted",
        "gaussian_shape_available",
    ):
        column = FEATURE_NAMES.index(name)
        assert np.array_equal(actual[:, column], expected[:, column])
    assert isinstance(features, DeviceXFitFeatures)
    assert features.values.shape == (device.parameters.shape[0], 17)
    assert features.values.dtype == cp.float32
    assert features.values.flags.c_contiguous
    assert int(features.values.device.id) == device.device_id
    assert features.device_id == device.device_id
    assert features.feature_names == FEATURE_NAMES
    assert features.model == model
    assert features.image_shape == _IMAGE_SHAPE
    assert features.variance_present is variance_present
    assert features.schema == "cuphoton.xscan.device-xfit-features/v1"
    assert features.backend == "cupy"
    assert features.result_location == "device"
    fit_valid = FEATURE_NAMES.index("fit_valid")
    valid_fraction = FEATURE_NAMES.index("valid_pixel_fraction")
    improvement = FEATURE_NAMES.index("fractional_null_improvement")
    log_delta = FEATURE_NAMES.index("log_delta_chi_square")
    log_reduced = FEATURE_NAMES.index("log_reduced_chi_square")
    strength = FEATURE_NAMES.index("log_strength_snr")
    separation = FEATURE_NAMES.index("separation_over_stamp")
    separation_significance = FEATURE_NAMES.index("separation_significance")
    assert actual[0, strength] > 0.0
    assert actual[1, fit_valid] == 0.0
    assert actual[2, fit_valid] == 0.0
    assert actual[2, separation] == 0.0
    assert actual[2, valid_fraction] > 0.0
    assert actual[4, valid_fraction] == 1.0
    assert actual[4, improvement] == -1.0
    assert actual[4, log_delta] == 0.0
    assert actual[4, log_reduced] == 0.0
    assert actual[5, separation_significance] == 0.0
    assert actual[6, fit_valid] == 1.0
    if model == "gaussian":
        assert actual[7, fit_valid] == 0.0
        assert actual[9, fit_valid] == 0.0
        assert actual[10, separation_significance] == 0.0
        assert actual[11, separation_significance] == 0.0
    else:
        shape_columns = (
            "gaussian_shape_available",
            "size_over_stamp",
            "axis_ratio",
            "aligned_ellipticity",
        )
        assert not np.any(
            actual[:, [FEATURE_NAMES.index(name) for name in shape_columns]]
        )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_device_features_preserve_subnormal_values(
    dtype: type[np.float32] | type[np.float64],
) -> None:
    cp = _require_cupy_device()
    portable, device = _make_results(cp, model="gaussian", dtype=dtype)
    parameters = portable.parameters.copy()
    parameters[0, _GAUSSIAN_PARAMETERS.index("sigma_x")] = (
        0.0005541826174853382
    )
    parameters[0, _GAUSSIAN_PARAMETERS.index("sigma_y")] = (
        3.704983671705392e-43
    )
    smallest = float(np.finfo(np.float32).smallest_subnormal)
    normal = float(np.finfo(np.float32).tiny)
    # Positive subnormal widths remain valid and can give a normal ratio.
    parameters[8, _GAUSSIAN_PARAMETERS.index("sigma_x")] = 4 * smallest
    parameters[8, _GAUSSIAN_PARAMETERS.index("sigma_y")] = 2 * smallest
    improvement = np.asarray(
        [
            0.0,
            -0.0,
            smallest,
            -smallest,
            smallest / 2,
            -smallest / 2,
            normal - smallest,
            smallest - normal,
            normal,
            -normal,
            normal + smallest,
            -normal - smallest,
        ],
        dtype=np.float64,
    )
    # Keep subnormal deltas observable after the log feature's division by 10.
    delta_chi_square = (10 * improvement).astype(dtype)
    portable = replace(
        portable,
        parameters=parameters,
        fractional_null_improvement=improvement,
        delta_chi_square=delta_chi_square,
    )
    device = replace(
        device,
        parameters=cp.asarray(parameters),
        fractional_null_improvement=cp.asarray(improvement),
        delta_chi_square=cp.asarray(delta_chi_square),
    )
    expected = transform_xfit_result_features(
        portable, image_shape=_IMAGE_SHAPE, variance_present=True
    )
    features = transform_xfit_result_features_device(
        device, image_shape=_IMAGE_SHAPE, variance_present=True
    )
    actual = cp.asnumpy(features.values)
    for name in ("fit_valid", "gaussian_shape_available"):
        column = FEATURE_NAMES.index(name)
        np.testing.assert_array_equal(expected[[0, 8], column], 1.0)
        np.testing.assert_array_equal(actual[[0, 8], column], 1.0)
    axis_ratio = FEATURE_NAMES.index("axis_ratio")
    assert actual[8, axis_ratio] == expected[8, axis_ratio] == 0.5
    assert 0 < expected[0, axis_ratio] < np.finfo(np.float32).tiny
    assert actual[0, axis_ratio].view(np.uint32) == expected[
        0, axis_ratio
    ].view(np.uint32)
    column = FEATURE_NAMES.index("log_delta_chi_square")
    assert 0 < expected[2, column] < np.finfo(np.float32).tiny
    for name in ("fractional_null_improvement", "log_delta_chi_square"):
        column = FEATURE_NAMES.index(name)
        np.testing.assert_array_equal(
            actual[:, column].view(np.uint32),
            expected[:, column].view(np.uint32),
        )


def test_device_features_ignore_residual_values() -> None:
    cp = _require_cupy_device()
    _, first = _make_results(cp, model="gaussian", dtype=np.float64)
    second = replace(first, residuals=cp.full_like(first.residuals, cp.nan))

    first_features = transform_xfit_result_features_device(
        first,
        image_shape=_IMAGE_SHAPE,
        variance_present=True,
    )
    second_features = transform_xfit_result_features_device(
        second,
        image_shape=_IMAGE_SHAPE,
        variance_present=True,
    )

    assert cp.array_equal(
        first_features.values, second_features.values
    ).item()


@pytest.mark.parametrize("model", ["gaussian", "stamp"])
def test_device_features_accept_result_without_residuals(model: str) -> None:
    cp = _require_cupy_device()
    portable, full_result = _make_results(cp, model=model, dtype=np.float64)
    residual_ref = weakref.ref(full_result.residuals)
    result = replace(full_result, residuals=None)
    del full_result
    gc.collect()
    assert residual_ref() is None

    expected = transform_xfit_result_features(
        portable,
        image_shape=_IMAGE_SHAPE,
        variance_present=True,
    )
    features = transform_xfit_result_features_device(
        result,
        image_shape=_IMAGE_SHAPE,
        variance_present=True,
    )

    assert np.allclose(cp.asnumpy(features.values), expected, rtol=2e-6)
    assert features.image_shape == _IMAGE_SHAPE


def test_device_features_accept_actual_device_fit_result() -> None:
    cp = _require_cupy_device()
    truth = np.asarray(
        [[5.0, 1.3, 1.8, 0.25, -2.0, 0.5, 2.2, -0.7]],
        dtype=np.float64,
    )
    model = GaussianDipoleModel(_IMAGE_SHAPE, dtype=np.float64)
    images = cp.asarray(model.evaluate(truth))
    result = fit_dipoles_device(images, model=model, initial=truth)

    features = transform_xfit_result_features_device(
        result,
        image_shape=_IMAGE_SHAPE,
        variance_present=False,
    )

    assert features.values.shape == (1, len(FEATURE_NAMES))
    assert features.values.dtype == cp.float32
    assert features.device_id == int(images.device.id)


def test_device_features_are_frozen_nonpickleable_and_own_storage() -> None:
    cp = _require_cupy_device()
    _, source = _make_results(cp, model="stamp", dtype=np.float32)
    features = transform_xfit_result_features_device(
        source,
        image_shape=_IMAGE_SHAPE,
        variance_present=True,
    )
    feature_sum = float(cp.sum(features.values).item())

    with pytest.raises(FrozenInstanceError):
        features.device_id = features.device_id + 1  # type: ignore[misc]
    with pytest.raises(TypeError, match="cannot be pickled"):
        pickle.dumps(features, protocol=pickle.HIGHEST_PROTOCOL)

    del source
    gc.collect()
    assert float(cp.sum(features.values).item()) == feature_sum


def test_device_feature_result_rejects_invalid_value_storage() -> None:
    cp = _require_cupy_device()
    _, source = _make_results(cp, model="stamp", dtype=np.float32)
    features = transform_xfit_result_features_device(
        source,
        image_shape=_IMAGE_SHAPE,
        variance_present=True,
    )

    with pytest.raises(TypeError, match="float32"):
        replace(features, values=features.values.astype(cp.float64))
    with pytest.raises(ValueError, match="shape"):
        replace(features, values=features.values[:, :-1])
    storage = cp.zeros((features.values.shape[0], 34), dtype=cp.float32)
    with pytest.raises(ValueError, match="C-contiguous"):
        replace(features, values=storage[:, ::2])


@pytest.mark.parametrize(
    "image_shape",
    [[11, 13], (11,), (0, 13), (True, 13), (11, 13.0)],
)
def test_device_feature_transform_rejects_invalid_image_shape(
    image_shape: object,
) -> None:
    cp = _require_cupy_device()
    _, source = _make_results(cp, model="stamp", dtype=np.float64)

    with pytest.raises(ValueError, match="image_shape"):
        transform_xfit_result_features_device(
            source,
            image_shape=image_shape,  # type: ignore[arg-type]
            variance_present=True,
        )


def test_device_feature_transform_rejects_invalid_source_contracts() -> None:
    cp = _require_cupy_device()
    _, source = _make_results(cp, model="stamp", dtype=np.float64)
    split_source = replace(
        source,
        mode="split",
        residuals=cp.zeros(
            (source.parameters.shape[0], 3, *_IMAGE_SHAPE),
            dtype=cp.float64,
        ),
    )
    wrong_names = replace(
        source,
        parameter_names=("flux", "x_pos", "y_pos", "x_neg", "y_neg"),
    )

    with pytest.raises(ValueError, match="difference-mode"):
        transform_xfit_result_features_device(
            split_source,
            image_shape=_IMAGE_SHAPE,
            variance_present=True,
        )
    with pytest.raises(ValueError, match="parameter_names"):
        transform_xfit_result_features_device(
            wrong_names,
            image_shape=_IMAGE_SHAPE,
            variance_present=True,
        )
    with pytest.raises(TypeError, match="variance_present"):
        transform_xfit_result_features_device(
            source,
            image_shape=_IMAGE_SHAPE,
            variance_present=np.bool_(True),  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="DeviceDipoleFitResult"):
        transform_xfit_result_features_device(  # type: ignore[arg-type]
            object(),
            image_shape=_IMAGE_SHAPE,
            variance_present=True,
        )


def test_device_feature_transform_rejects_tampered_source_schema() -> None:
    cp = _require_cupy_device()
    _, source = _make_results(cp, model="stamp", dtype=np.float64)
    object.__setattr__(source, "schema", "unsupported")

    with pytest.raises(ValueError, match="schema"):
        transform_xfit_result_features_device(
            source,
            image_shape=_IMAGE_SHAPE,
            variance_present=True,
        )


def test_device_feature_transform_rejects_wrong_device_before_kernels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cp = _require_cupy_device(minimum_count=2)
    active_device_id = int(cp.cuda.runtime.getDevice())
    other_device_id = next(
        value
        for value in range(int(cp.cuda.runtime.getDeviceCount()))
        if value != active_device_id
    )
    try:
        with cp.cuda.Device(other_device_id):
            _, source = _make_results(cp, model="stamp", dtype=np.float64)
    except Exception as exc:
        pytest.skip(f"second CUDA device is not usable: {exc}")

    monkeypatch.setattr(
        cp,
        "zeros",
        lambda *_args, **_kwargs: pytest.fail(
            "wrong-device source reached feature kernels"
        ),
    )
    with pytest.raises(ValueError, match="active device"):
        transform_xfit_result_features_device(
            source,
            image_shape=_IMAGE_SHAPE,
            variance_present=True,
        )
    assert int(cp.cuda.runtime.getDevice()) == active_device_id


def test_device_feature_transform_reports_no_visible_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeRuntime:
        @staticmethod
        def getDeviceCount() -> int:
            return 0

    class FakeCuda:
        runtime = FakeRuntime()

    class FakeCupy:
        cuda = FakeCuda()

    source = object.__new__(DeviceDipoleFitResult)
    monkeypatch.setattr(
        xfit_features_module, "_load_cupy", lambda: FakeCupy()
    )

    with pytest.raises(
        RuntimeError,
        match="requires at least one visible CUDA device",
    ):
        transform_xfit_result_features_device(
            source,
            image_shape=_IMAGE_SHAPE,
            variance_present=True,
        )


def test_device_feature_transform_does_not_eagerly_import_cupy() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from cuphoton.xscan.xfit_features import "
                "transform_xfit_result_features_device; "
                "assert callable(transform_xfit_result_features_device); "
                "raise SystemExit('cupy' in sys.modules)"
            ),
        ],
        check=False,
    )

    assert result.returncode == 0


def test_device_feature_transform_has_no_row_loop_or_scalar_item() -> None:
    source = inspect.getsource(transform_xfit_result_features_device)

    assert "range(batch_size)" not in source
    assert ".item(" not in source
