# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Direct FITS inference and failure cleanup with CPU boundary adapters."""

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

torch = pytest.importorskip("torch")

from cuphoton.xscan import fits_inference, training  # noqa: E402
from cuphoton.xscan.config import PerformanceConfig  # noqa: E402
from cuphoton.xscan.fits_inference import (  # noqa: E402
    FitsPlane,
    predict_fits,
)


class _Stream:
    def __init__(self, *, fail=False):
        self.synchronized = 0
        self.fail = fail

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def synchronize(self):
        self.synchronized += 1
        if self.fail:
            raise RuntimeError("CUDA completion failed")


class _Model(torch.nn.Module):
    def __init__(self, channels, features=False, fail=False, nonfinite=False):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.01))
        self.input_mode = "pair" if channels == 2 else "triplet"
        self.xfit_feature_names = ["feature"] if features else []
        self.fail = fail
        self.nonfinite = nonfinite

    def forward(self, images, xfit_features=None):
        if self.fail:
            raise RuntimeError("model failed")
        weights = torch.arange(
            1, images.shape[1] + 1, device=images.device
        ).reshape(1, -1, 1, 1)
        value = (images * weights).mean(dim=(1, 2, 3)) * self.scale
        if xfit_features is not None:
            value = value + xfit_features[:, 0]
        return value * float("nan") if self.nonfinite else value


def _write_planes(tmp_path, channels=3):
    path = tmp_path / "planes.fits"
    planes = [
        np.arange(143, dtype="f4").reshape(11, 13) * (i + 1)
        for i in range(channels)
    ]
    fits.HDUList(
        [fits.PrimaryHDU()] + [fits.ImageHDU(array) for array in planes]
    ).writeto(path)
    return path, planes


@pytest.fixture
def cpu_boundaries(monkeypatch):
    """Adapt CUDA boundaries, keeping real FITS reads and Torch inference."""
    producer, consumer = _Stream(), _Stream()
    fake_cp = SimpleNamespace(
        empty=np.empty,
        float32=np.float32,
        isfinite=np.isfinite,
        cuda=SimpleNamespace(
            Device=lambda _: nullcontext(), Stream=lambda **_: producer
        ),
    )
    monkeypatch.setattr(
        training, "_require_explicit_cuda_device", lambda device: device
    )
    monkeypatch.setattr(training, "_load_cupy_for_dlpack", lambda: fake_cp)
    monkeypatch.setattr(
        training,
        "cupy_to_torch",
        lambda values, **_: SimpleNamespace(
            tensor=torch.from_numpy(values), owner=values
        ),
    )
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _: consumer)
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    original_read = fits_inference.read_fits_images
    reads = []

    def host_read(path, hdus, **kwargs):
        reads.append((path, hdus, kwargs.copy()))
        return original_read(path, hdus, reader="astropy", device=False)

    monkeypatch.setattr(fits_inference, "read_fits_images", host_read)
    monkeypatch.setattr(fits_inference, "_FAILED_OWNERS", [])
    return producer, consumer, reads


@pytest.mark.parametrize("channels", [2, 3])
@pytest.mark.parametrize("reader", ["auto", "astropy", "xdr"])
def test_fits_predictions_match_existing_tensor_math(
    tmp_path, cpu_boundaries, channels, reader
):
    path, planes = _write_planes(tmp_path, channels)
    model = _Model(channels, features=True)
    features = torch.tensor([[0.25], [-0.5]])
    centers = [(7, 8), (3, 4)]
    ids = ("later", "earlier")
    args = {"difference": FitsPlane(path, 3)} if channels == 3 else {}
    result = predict_fits(
        model=model,
        search=FitsPlane(path, 1),
        template=FitsPlane(path, 2),
        centers_yx=centers,
        candidate_ids=ids,
        stamp_shape=(3, 5),
        device=torch.device("cpu"),
        performance=PerformanceConfig(),
        xfit_features=features,
        fits_reader=reader,
        **args,
    )
    stamps = np.stack(
        [
            np.stack([p[y - 1 : y + 2, x - 2 : x + 3] for p in planes])
            for y, x in centers
        ]
    )
    expected = training.predict_tensors(
        model=model,
        images=torch.from_numpy(stamps),
        device=torch.device("cpu"),
        xfit_features=features,
        performance=PerformanceConfig(),
    )
    np.testing.assert_array_equal(result.logits, expected["logits"].numpy())
    np.testing.assert_array_equal(
        result.probabilities, expected["probabilities"].numpy()
    )
    assert result.candidate_ids == ids
    assert model.training is True
    reads = cpu_boundaries[2]
    assert len(reads) == 1
    assert reads[0][1] == tuple(range(1, channels + 1))
    assert reads[0][2]["reader"] == reader and reads[0][2]["device"] is True
    assert result.fits_reads[0]["reader"] == "astropy"


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"centers_yx": [(1.5, 4)]}, "integer"),
        ({"centers_yx": [(True, 4)]}, "integer"),
        ({"centers_yx": [(0, 0)]}, "bounds"),
        ({"candidate_ids": []}, "nonempty"),
        ({"candidate_ids": [True]}, "nonempty"),
        ({"stamp_shape": (2, 3)}, "odd"),
    ],
)
def test_invalid_candidate_contract_rejects_before_gpu(
    tmp_path, kwargs, message
):
    path, _ = _write_planes(tmp_path)
    params = dict(centers_yx=[(4, 5)], candidate_ids=[1], stamp_shape=(3, 3))
    params.update(kwargs)
    with pytest.raises(ValueError, match=message):
        fits_inference._plan_inputs(
            (FitsPlane(path, 1), FitsPlane(path, 2)), **params
        )


@pytest.mark.parametrize("inside", [False, True])
def test_nonfinite_pixels_are_rejected_only_inside_candidate_stamps(
    tmp_path, cpu_boundaries, inside
):
    path, planes = _write_planes(tmp_path)
    with fits.open(path, mode="update") as source:
        source[1].data[5 if inside else 0, 5 if inside else 0] = np.nan
    kwargs = dict(
        model=_Model(2),
        search=FitsPlane(path, 1),
        template=FitsPlane(path, 2),
        centers_yx=[(5, 5)],
        candidate_ids=[1],
        stamp_shape=(3, 3),
        device=torch.device("cpu"),
        performance=PerformanceConfig(),
    )
    if inside:
        with pytest.raises(ValueError, match="stamps.*finite"):
            predict_fits(**kwargs)
    else:
        assert np.isfinite(predict_fits(**kwargs).logits).all()


def test_failed_inference_drains_streams_and_retains_failed_owners(
    tmp_path, cpu_boundaries
):
    path, _ = _write_planes(tmp_path)
    producer, consumer, _ = cpu_boundaries
    producer.fail = True
    kwargs = dict(
        model=_Model(2, fail=True),
        search=FitsPlane(path, 1),
        template=FitsPlane(path, 2),
        centers_yx=[(5, 5)],
        candidate_ids=[1],
        stamp_shape=(3, 3),
        device=torch.device("cpu"),
        performance=PerformanceConfig(),
    )
    with pytest.raises(RuntimeError, match="model failed") as caught:
        predict_fits(**kwargs)
    assert producer.synchronized == consumer.synchronized == 1
    assert "could not confirm" in caught.value.__notes__[0]
    assert fits_inference._FAILED_OWNERS[0][1]
    with pytest.raises(RuntimeError, match="restart"):
        predict_fits(**kwargs)


def test_nonfinite_model_outputs_are_rejected(tmp_path, cpu_boundaries):
    path, _ = _write_planes(tmp_path)
    with pytest.raises(ValueError, match="predictions.*nonfinite"):
        predict_fits(
            model=_Model(2, nonfinite=True),
            search=FitsPlane(path, 1),
            template=FitsPlane(path, 2),
            centers_yx=[(5, 5)],
            candidate_ids=[1],
            stamp_shape=(3, 3),
            device=torch.device("cpu"),
            performance=PerformanceConfig(),
        )


@pytest.mark.parametrize("reader", ["astropy", "xdr"])
def test_gpu_fits_predictions_match_tensor_path_on_nondefault_stream(
    tmp_path, reader
):
    if not torch.cuda.is_available():
        pytest.skip("Torch CUDA is required")
    pytest.importorskip("cupy")
    from cuphoton.xdr import gpu_available
    from cuphoton.xdr.nvcomp_batch import get_native_plan_files

    if reader == "xdr" and (
        not gpu_available() or get_native_plan_files(required=False) is None
    ):
        pytest.skip("native xDR dependencies are required")
    device = torch.device("cuda:0")
    path = tmp_path / "compressed.fits"
    planes = [
        np.arange(143, dtype="f4").reshape(11, 13) * i for i in (1, 2, 3)
    ]
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            *(
                fits.CompImageHDU(
                    array, compression_type="GZIP_2", quantize_level=0
                )
                for array in planes
            ),
        ]
    ).writeto(path)
    consumer = torch.cuda.Stream(device=device)
    with torch.cuda.stream(consumer):
        model = _Model(3, features=True).to(device)
        features = torch.tensor([[0.25], [-0.5]], device=device)
        centers = [(7, 8), (3, 4)]
        result = predict_fits(
            model=model,
            search=FitsPlane(path, 1),
            template=FitsPlane(path, 2),
            difference=FitsPlane(path, 3),
            centers_yx=centers,
            candidate_ids=("later", "earlier"),
            stamp_shape=(3, 5),
            device=device,
            performance=PerformanceConfig(),
            xfit_features=features,
            fits_reader=reader,
        )
        stamps = np.stack(
            [
                np.stack([p[y - 1 : y + 2, x - 2 : x + 3] for p in planes])
                for y, x in centers
            ]
        )
        expected = training.predict_tensors(
            model=model,
            images=torch.as_tensor(stamps, device=device),
            device=device,
            xfit_features=features,
            performance=PerformanceConfig(),
        )
        np.testing.assert_array_equal(
            result.logits, expected["logits"].cpu().numpy()
        )
        np.testing.assert_array_equal(
            result.probabilities, expected["probabilities"].cpu().numpy()
        )
    assert result.fits_reads[0]["reader"] == reader
