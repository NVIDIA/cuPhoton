# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Strict device-tensor inference and CuPy-to-Torch handoff tests."""

from __future__ import annotations

import gc
import pickle
import subprocess
import sys
import weakref
from dataclasses import asdict
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from cuphoton.xscan import training  # noqa: E402
from cuphoton.xscan.config import PerformanceConfig  # noqa: E402
from cuphoton.xscan.training import (  # noqa: E402
    CupyTorchView,
    _temporary_eval_mode,
    cupy_to_torch,
    predict_tensors,
)


class _FusionModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))
        self.register_buffer("offset", torch.zeros(()))
        self.inference_flags: list[bool] = []

    def forward(
        self,
        images: torch.Tensor,
        *,
        xfit_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.inference_flags.append(torch.is_inference_mode_enabled())
        logits = images.mean(dim=(1, 2, 3)) * self.scale + self.offset
        if xfit_features is not None:
            logits = logits + xfit_features.sum(dim=1)
        return logits


class _MixedModeModel(_FusionModel):
    def __init__(self, *, fail: bool = False) -> None:
        super().__init__()
        self.dropout = torch.nn.Dropout(p=0.5)
        self.batchnorm = torch.nn.BatchNorm1d(1)
        self.fail = fail

    def forward(
        self,
        images: torch.Tensor,
        *,
        xfit_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.fail:
            raise RuntimeError("boom")
        logits = super().forward(images, xfit_features=xfit_features)
        return self.dropout(logits[:, None]).squeeze(1)


class _BadOutputModel(torch.nn.Module):
    def __init__(self, kind: str) -> None:
        super().__init__()
        self.kind = kind

    def forward(self, images: torch.Tensor) -> Any:
        if self.kind == "object":
            return "not a tensor"
        if self.kind == "shape":
            return images.mean(dim=(1, 2, 3))[:, None]
        if self.kind == "device":
            return torch.empty(images.shape[0], device="cpu")
        if self.kind.startswith("dtype_"):
            dtype = getattr(torch, self.kind.removeprefix("dtype_"))
            return images.mean(dim=(1, 2, 3)).to(dtype=dtype)
        raise AssertionError("unknown output kind")


class _NonFiniteModel(torch.nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = images.mean(dim=(1, 2, 3))
        return logits * torch.tensor(float("nan"), device=images.device)


class _PartialEvalFailureModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = torch.nn.Dropout()
        self.second = torch.nn.Dropout()

    def train(self, mode: bool = True) -> _PartialEvalFailureModel:
        if mode:
            return super().train(mode)
        self.training = False
        self.first.training = False
        raise RuntimeError("partial eval failure")


def _cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("Torch CUDA is required")
    return torch.device("cuda:0")


def _cupy_on_device(device: torch.device) -> Any:
    cp = pytest.importorskip("cupy")
    try:
        if int(cp.cuda.runtime.getDeviceCount()) <= int(device.index):
            pytest.skip(f"CuPy cannot see {device}")
    except Exception as exc:
        pytest.skip(f"CuPy CUDA is unavailable: {exc}")
    cp.cuda.Device(int(device.index)).use()
    return cp


def _inputs(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    images = torch.arange(
        2 * 2 * 3 * 3,
        dtype=torch.float32,
        device=device,
    ).reshape(2, 2, 3, 3)
    features = torch.arange(
        2 * 17,
        dtype=torch.float32,
        device=device,
    ).reshape(2, 17)
    return images, features


def test_training_import_does_not_eagerly_import_cupy() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import cuphoton.xscan.training; "
                "raise SystemExit('cupy' in sys.modules)"
            ),
        ],
        check=False,
    )
    assert result.returncode == 0


def test_checkpoint_performance_override_replaces_saved_policy(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved_performance = PerformanceConfig(
        amp_dtype="bf16",
        allow_tf32=True,
        cudnn_benchmark=True,
        compile=True,
    )
    checkpoint_model = _FusionModel()
    with torch.no_grad():
        checkpoint_model.scale.fill_(2.5)
        checkpoint_model.offset.fill_(-0.75)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model_config": {},
            "model_state": checkpoint_model.state_dict(),
            "train_config": {"performance": asdict(saved_performance)},
        },
        checkpoint_path,
    )
    checkpoint_bytes = checkpoint_path.read_bytes()
    monkeypatch.setattr(training, "build_model", lambda **_: _FusionModel())
    events: list[tuple[str, PerformanceConfig]] = []

    def record_compile(model: Any, *, performance: PerformanceConfig) -> Any:
        events.append(("compile", performance))
        return model, {"enabled": False}

    monkeypatch.setattr(training, "maybe_compile_model", record_compile)
    monkeypatch.setattr(
        training,
        "configure_runtime",
        lambda *, performance, device: events.append(
            ("runtime", performance)
        ),
    )

    default_model, _, default_policy = training.load_model_from_checkpoint(
        tmp_path, device=torch.device("cpu")
    )
    assert asdict(default_policy) == asdict(saved_performance)
    assert events == [("compile", default_policy)]

    override = PerformanceConfig(pin_memory=True, non_blocking_transfers=True)
    original_override = asdict(override)
    model, checkpoint, policy = training.load_model_from_checkpoint(
        tmp_path,
        device=torch.device("cpu"),
        performance_override=override,
    )
    assert asdict(policy) == asdict(PerformanceConfig())
    assert asdict(override) == original_override
    assert events == [
        ("compile", default_policy),
        ("runtime", policy),
        ("compile", policy),
    ]
    assert checkpoint["train_config"]["performance"] == asdict(
        saved_performance
    )
    assert checkpoint_path.read_bytes() == checkpoint_bytes
    assert model.training is False
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, checkpoint_model.state_dict()[name])
        torch.testing.assert_close(value, default_model.state_dict()[name])
    images = torch.ones((2, 2, 3, 3))
    torch.testing.assert_close(model(images), default_model(images))


@pytest.mark.parametrize(
    ("override", "exception", "message"),
    [
        ({}, TypeError, "performance_override must be a PerformanceConfig"),
        (
            PerformanceConfig(compile_threads=0),
            ValueError,
            "compile_threads must be positive",
        ),
        (
            PerformanceConfig(worker_start_method="invalid"),
            ValueError,
            "worker_start_method must be one of",
        ),
    ],
)
def test_checkpoint_override_rejected_before_loading(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    override: Any,
    exception: type[Exception],
    message: str,
) -> None:
    def unexpected_load(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("invalid override must fail before checkpoint loading")

    monkeypatch.setattr(torch, "load", unexpected_load)
    with pytest.raises(exception, match=message):
        training.load_model_from_checkpoint(
            tmp_path,
            device=torch.device("cpu"),
            performance_override=override,
        )


def test_predict_tensors_is_transfer_free_and_device_resident(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = _cuda_device()
    model = _FusionModel().to(device)
    images, features = _inputs(device)
    expected = images.mean(dim=(1, 2, 3)) + features.sum(dim=1)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise AssertionError(
            "direct inference attempted a forbidden transfer"
        )

    with monkeypatch.context() as scoped:
        scoped.setattr(torch.Tensor, "to", forbidden)
        scoped.setattr(torch.Tensor, "cpu", forbidden)
        scoped.setattr(torch.Tensor, "cuda", forbidden)
        scoped.setattr(torch.Tensor, "numpy", forbidden)
        scoped.setattr(torch.Tensor, "item", forbidden)
        scoped.setattr(torch.Tensor, "contiguous", forbidden)
        scoped.setattr(torch, "as_tensor", forbidden)
        scoped.setattr(torch, "tensor", forbidden)
        scoped.setattr(torch.cuda, "synchronize", forbidden)
        prediction = predict_tensors(
            model=model,
            images=images,
            xfit_features=features,
            device=device,
        )

    assert prediction["logits"].device == device
    assert prediction["probabilities"].device == device
    assert prediction["logits"].dtype == torch.float32
    assert prediction["probabilities"].dtype == torch.float32
    assert prediction["logits"].requires_grad is False
    assert prediction["probabilities"].requires_grad is False
    torch.testing.assert_close(prediction["logits"], expected)
    torch.testing.assert_close(
        prediction["probabilities"], torch.sigmoid(expected)
    )
    assert model.inference_flags == [False]


@pytest.mark.parametrize("device_type", ["cpu", "cuda"])
def test_predict_tensors_outputs_support_downstream_autograd(
    device_type: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if device_type == "cpu":
        device = torch.device("cpu")
        # Exercise the real inference body on CPU; bypass only the public
        # CUDA-availability gate, not tensor/model validation or autograd.
        monkeypatch.setattr(
            training, "_require_explicit_cuda_device", lambda device: device
        )
    else:
        device = _cuda_device()
    model = _FusionModel().to(device)
    images, features = _inputs(device)
    images.requires_grad_()
    prediction = predict_tensors(
        model=model,
        images=images,
        xfit_features=features,
        device=device,
    )

    # A later trainable stage must be able to save these constant outputs
    # for its backward pass, without copying them to remove an inference tag.
    for values in prediction.values():
        assert not values.requires_grad
        weight = torch.nn.Parameter(torch.ones_like(values))
        (weight * values).sum().backward()
        torch.testing.assert_close(weight.grad, values)
    assert images.grad is None
    assert model.scale.grad is None


def test_predict_tensors_retains_amp_output_cast_on_device() -> None:
    device = _cuda_device()
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(4, 1),
        torch.nn.Flatten(0),
    ).to(device)
    images = torch.ones((2, 1, 2, 2), dtype=torch.float32, device=device)

    prediction = predict_tensors(
        model=model,
        images=images,
        device=device,
        performance=PerformanceConfig(amp_dtype="bf16"),
    )

    assert prediction["logits"].device == device
    assert prediction["logits"].dtype == torch.float32
    assert prediction["probabilities"].dtype == torch.float32


def test_temporary_eval_mode_restores_partial_entry_failure() -> None:
    model = _PartialEvalFailureModel()
    model.train()
    model.second.eval()

    with pytest.raises(RuntimeError, match="partial eval failure"):
        with _temporary_eval_mode(model):
            raise AssertionError("unreachable")

    assert model.training is True
    assert model.first.training is True
    assert model.second.training is False


def test_predict_tensors_defers_finite_check_to_terminal_boundary() -> None:
    device = _cuda_device()
    images, _ = _inputs(device)

    prediction = predict_tensors(
        model=_NonFiniteModel(),
        images=images,
        device=device,
    )

    assert prediction["logits"].device == device
    assert torch.isnan(prediction["logits"]).all()


@pytest.mark.parametrize("fail", [False, True])
def test_predict_tensors_restores_mixed_module_modes(fail: bool) -> None:
    device = _cuda_device()
    model = _MixedModeModel(fail=fail).to(device)
    model.train()
    model.batchnorm.eval()
    images, features = _inputs(device)

    if fail:
        with pytest.raises(RuntimeError, match="boom"):
            predict_tensors(
                model=model,
                images=images,
                xfit_features=features,
                device=device,
            )
    else:
        prediction = predict_tensors(
            model=model,
            images=images,
            xfit_features=features,
            device=device,
        )
        assert prediction["logits"].requires_grad is False

    assert model.training is True
    assert model.batchnorm.training is False


@pytest.mark.parametrize(
    ("device", "message"),
    [
        (torch.device("cpu"), "explicit CUDA device"),
        (torch.device("cuda"), "explicit CUDA device"),
    ],
)
def test_predict_tensors_requires_explicit_cuda_ordinal(
    device: torch.device,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        predict_tensors(
            model=_FusionModel(),
            images=torch.ones((2, 2, 3, 3)),
            device=device,
        )


@pytest.mark.parametrize(
    ("mutate", "exception", "message"),
    [
        (lambda value: value.cpu(), ValueError, "exact device"),
        (lambda value: value.to(torch.float64), TypeError, "float32"),
        (lambda value: value[:, :, :, 0], ValueError, "rank 4"),
        (lambda value: value.transpose(2, 3), ValueError, "C-contiguous"),
        (lambda value: value[:0], ValueError, "positive"),
    ],
)
def test_predict_tensors_rejects_invalid_images(
    mutate: Any,
    exception: type[Exception],
    message: str,
) -> None:
    device = _cuda_device()
    model = _FusionModel().to(device)
    images, _ = _inputs(device)
    with pytest.raises(exception, match=message):
        predict_tensors(
            model=model,
            images=mutate(images),
            device=device,
        )


@pytest.mark.parametrize(
    ("mutate", "exception", "message"),
    [
        (lambda value: value.cpu(), ValueError, "exact device"),
        (lambda value: value.to(torch.float64), TypeError, "float32"),
        (lambda value: value[:, 0], ValueError, "rank 2"),
        (lambda value: value[:1], ValueError, "batch size"),
    ],
)
def test_predict_tensors_rejects_invalid_xfit_features(
    mutate: Any,
    exception: type[Exception],
    message: str,
) -> None:
    device = _cuda_device()
    model = _FusionModel().to(device)
    images, features = _inputs(device)
    with pytest.raises(exception, match=message):
        predict_tensors(
            model=model,
            images=images,
            xfit_features=mutate(features),
            device=device,
        )


def test_predict_tensors_accepts_generic_feature_width() -> None:
    device = _cuda_device()
    model = _FusionModel().to(device)
    images, features = _inputs(device)
    features = features[:, :3].contiguous()

    prediction = predict_tensors(
        model=model,
        images=images,
        xfit_features=features,
        device=device,
    )

    expected = images.mean(dim=(1, 2, 3)) + features.sum(dim=1)
    torch.testing.assert_close(prediction["logits"], expected)


@pytest.mark.parametrize("supply_features", [False, True])
def test_predict_tensors_validates_exposed_model_feature_schema(
    supply_features: bool,
) -> None:
    device = _cuda_device()
    model = _FusionModel().to(device)
    model.xfit_feature_names = ("first", "second")
    images, features = _inputs(device)

    with pytest.raises(ValueError, match="xFit features|column count"):
        predict_tensors(
            model=model,
            images=images,
            xfit_features=features if supply_features else None,
            device=device,
        )
    assert model.inference_flags == []


def test_predict_tensors_validates_exposed_model_input_mode() -> None:
    device = _cuda_device()
    model = _FusionModel().to(device)
    model.input_mode = "triplet"
    images, _ = _inputs(device)

    with pytest.raises(ValueError, match="triplet model requires 3"):
        predict_tensors(model=model, images=images, device=device)
    assert model.inference_flags == []


def test_predict_tensors_rejects_noncontiguous_xfit_features() -> None:
    device = _cuda_device()
    model = _FusionModel().to(device)
    images, features = _inputs(device)
    noncontiguous = torch.empty((2, 34), dtype=torch.float32, device=device)[
        :, ::2
    ]
    assert noncontiguous.shape == features.shape
    assert not noncontiguous.is_contiguous()

    with pytest.raises(ValueError, match="C-contiguous"):
        predict_tensors(
            model=model,
            images=images,
            xfit_features=noncontiguous,
            device=device,
        )


@pytest.mark.parametrize("wrong_input", ["images", "xfit_features"])
def test_predict_tensors_rejects_another_cuda_device(
    wrong_input: str,
) -> None:
    device = _cuda_device()
    if torch.cuda.device_count() < 2:
        pytest.skip("two visible CUDA devices are required")
    model = _FusionModel().to(device)
    images, features = _inputs(device)
    if wrong_input == "images":
        images = images.to("cuda:1")
    else:
        features = features.to("cuda:1")

    with pytest.raises(ValueError, match="exact device cuda:0"):
        predict_tensors(
            model=model,
            images=images,
            xfit_features=features,
            device=device,
        )


@pytest.mark.parametrize("collection", ["parameter", "buffer"])
def test_predict_tensors_rejects_model_state_on_another_device(
    collection: str,
) -> None:
    device = _cuda_device()
    images, features = _inputs(device)
    model = _FusionModel()
    if collection == "parameter":
        model.offset = model.offset.to(device)
    else:
        model.scale = torch.nn.Parameter(model.scale.to(device))

    with pytest.raises(ValueError, match=f"model {collection}"):
        predict_tensors(
            model=model,
            images=images,
            xfit_features=features,
            device=device,
        )


@pytest.mark.parametrize("collection", ["parameter", "buffer"])
def test_predict_tensors_rejects_non_float32_model_state(
    collection: str,
) -> None:
    device = _cuda_device()
    images, features = _inputs(device)
    model = _FusionModel().to(device)
    if collection == "parameter":
        model.scale = torch.nn.Parameter(model.scale.double())
    else:
        model.offset = model.offset.double()

    with pytest.raises(TypeError, match=f"model {collection}.*float32"):
        predict_tensors(
            model=model,
            images=images,
            xfit_features=features,
            device=device,
        )


@pytest.mark.parametrize(
    ("kind", "exception", "message"),
    [
        ("object", TypeError, "model output"),
        ("shape", ValueError, "model logits must have shape"),
        ("device", RuntimeError, "model logits are on cpu"),
        ("dtype_float64", TypeError, "unsupported dtype float64"),
        ("dtype_int64", TypeError, "unsupported dtype int64"),
        ("dtype_complex64", TypeError, "unsupported dtype complex64"),
        ("dtype_float16", TypeError, "unsupported dtype float16"),
        ("dtype_bfloat16", TypeError, "unsupported dtype bfloat16"),
    ],
)
def test_predict_tensors_validates_model_output(
    kind: str,
    exception: type[Exception],
    message: str,
) -> None:
    device = _cuda_device()
    images, _ = _inputs(device)
    with pytest.raises(exception, match=message):
        predict_tensors(
            model=_BadOutputModel(kind),
            images=images,
            device=device,
        )


def test_predict_tensors_rejects_nonselected_amp_output_dtype() -> None:
    device = _cuda_device()
    images, _ = _inputs(device)

    with pytest.raises(TypeError, match="unsupported dtype float16"):
        predict_tensors(
            model=_BadOutputModel("dtype_float16"),
            images=images,
            device=device,
            performance=PerformanceConfig(amp_dtype="bf16"),
        )


def test_cupy_bridge_uses_producer_object_and_preserves_pointer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = _cuda_device()
    cp = _cupy_on_device(device)
    values = cp.arange(34, dtype=cp.float32).reshape(2, 17)
    original = torch.utils.dlpack.from_dlpack
    calls: list[tuple[Any, dict[str, Any]]] = []

    def record_producer(producer: Any, **kwargs: Any) -> torch.Tensor:
        calls.append((producer, kwargs))
        return original(producer, **kwargs)

    monkeypatch.setattr(torch.utils.dlpack, "from_dlpack", record_producer)
    view = cupy_to_torch(values, device=device)

    assert isinstance(view, CupyTorchView)
    assert len(calls) == 1
    assert calls[0][0] is values
    assert calls[0][1] == {"copy": False}
    assert view.tensor.data_ptr() == int(values.data.ptr)
    assert view.tensor.device == device
    assert view.tensor.dtype == torch.float32
    assert tuple(view.tensor.shape) == tuple(values.shape)
    assert view.tensor.is_contiguous()
    assert view._owner is values
    with pytest.raises(TypeError, match="cannot be pickled"):
        pickle.dumps(view)


def test_cupy_bridge_retains_owner_until_consumer_completion() -> None:
    device = _cuda_device()
    cp = _cupy_on_device(device)
    values = cp.arange(34, dtype=cp.float32).reshape(2, 17)
    owner_ref = weakref.ref(values)
    view = cupy_to_torch(values, device=device)
    output = view.tensor + 1.0

    del values
    gc.collect()
    assert owner_ref() is not None

    torch.cuda.current_stream(device).synchronize()
    del view
    gc.collect()
    assert owner_ref() is None
    assert output.device == device


def test_device_xfit_features_compose_with_tensor_inference() -> None:
    from cuphoton.xscan.xfit_features import DeviceXFitFeatures

    device = _cuda_device()
    cp = _cupy_on_device(device)
    image_values = cp.arange(2 * 2 * 3 * 3, dtype=cp.float32).reshape(
        2, 2, 3, 3
    )
    feature_values = cp.arange(2 * 17, dtype=cp.float32).reshape(2, 17)
    features = DeviceXFitFeatures(
        values=feature_values,
        device_id=int(device.index),
        model="gaussian",
        image_shape=(3, 3),
        variance_present=True,
    )
    image_view = cupy_to_torch(image_values, device=device)
    feature_view = cupy_to_torch(features.values, device=device)
    model = _FusionModel().to(device)

    prediction = predict_tensors(
        model=model,
        images=image_view.tensor,
        xfit_features=feature_view.tensor,
        device=device,
    )
    completion = torch.cuda.Event()
    completion.record(torch.cuda.current_stream(device))
    completion.synchronize()

    expected = image_view.tensor.mean(dim=(1, 2, 3))
    expected += feature_view.tensor.sum(dim=1)
    torch.testing.assert_close(prediction["logits"], expected)
    assert feature_view.tensor.data_ptr() == int(features.values.data.ptr)


def test_cupy_bridge_orders_nondefault_streams() -> None:
    device = _cuda_device()
    cp = _cupy_on_device(device)
    producer = cp.cuda.Stream(non_blocking=True)
    release = cp.cuda.Stream(non_blocking=True)
    consumer = torch.cuda.Stream(device=device)
    gate = cp.cuda.Event()
    values = cp.zeros((2, 17), dtype=cp.float32)
    initialized = cp.cuda.Event()
    initialized.record()
    initialized.synchronize()
    delay = cp.RawKernel(
        r"""
        extern "C" __global__ void delay(unsigned long long cycles) {
            unsigned long long start = clock64();
            while (clock64() - start < cycles) {}
        }
        """,
        "delay",
    )

    with release:
        delay((1,), (1,), (100_000_000,))
        gate.record()
    assert not gate.done

    with producer:
        producer.wait_event(gate)
        values.fill(7.0)
        with torch.cuda.stream(consumer):
            view = cupy_to_torch(values, device=device)
            output = view.tensor.clone()

    consumer.synchronize()

    assert gate.done
    assert torch.equal(output, torch.full_like(output, 7.0))
    assert view.tensor.data_ptr() == int(values.data.ptr)


def test_cupy_bridge_does_not_normalize_copy_or_synchronize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = _cuda_device()
    cp = _cupy_on_device(device)
    values = cp.ones((2, 17), dtype=cp.float32)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise AssertionError("DLPack bridge attempted a forbidden operation")

    monkeypatch.setattr(cp, "ascontiguousarray", forbidden)
    monkeypatch.setattr(cp, "asnumpy", forbidden)
    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    view = cupy_to_torch(values, device=device)

    assert view.tensor.data_ptr() == int(values.data.ptr)


@pytest.mark.parametrize(
    ("make_values", "exception", "message"),
    [
        (
            lambda cp: cp.ones((2, 17), dtype=cp.float64),
            TypeError,
            "float32",
        ),
        (
            lambda cp: cp.ones((2, 34), dtype=cp.float32)[:, ::2],
            ValueError,
            "C-contiguous",
        ),
        (
            lambda cp: cp.empty((0, 17), dtype=cp.float32),
            ValueError,
            "non-empty",
        ),
    ],
)
def test_cupy_bridge_rejects_invalid_arrays(
    make_values: Any,
    exception: type[Exception],
    message: str,
) -> None:
    device = _cuda_device()
    cp = _cupy_on_device(device)
    with pytest.raises(exception, match=message):
        cupy_to_torch(make_values(cp), device=device)


@pytest.mark.parametrize("wrong_precondition", ["active", "array"])
def test_cupy_bridge_rejects_wrong_device(
    wrong_precondition: str,
) -> None:
    device = _cuda_device()
    cp = _cupy_on_device(device)
    if torch.cuda.device_count() < 2 or cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("two visible CUDA devices are required")
    if wrong_precondition == "active":
        values = cp.ones((2, 17), dtype=cp.float32)
        with cp.cuda.Device(1):
            with pytest.raises(ValueError, match="active CuPy device"):
                cupy_to_torch(values, device=device)
        return
    with cp.cuda.Device(1):
        values = cp.ones((2, 17), dtype=cp.float32)
    cp.cuda.Device(0).use()
    with pytest.raises(ValueError, match="values are on cuda:1"):
        cupy_to_torch(values, device=device)
