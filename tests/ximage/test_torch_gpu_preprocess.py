# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cuphoton.ximage.backends.torch_gpu_preprocess import (
    _coadd,
    prepare_torch_reconstruction_inputs,
)
from cuphoton.ximage.config import ReconstructionConfig
from cuphoton.ximage.data import XImageBundle
from cuphoton.ximage.preprocess import (
    coadd,
    pad_with_data,
    prepare_reconstruction_inputs,
    select_cutout,
)

torch = pytest.importorskip("torch")


def _bundle() -> XImageBundle:
    images = np.arange(200, dtype=np.float64).reshape(1, 10, 10, 2)
    masks = np.zeros((1, 10, 10, 12, 2), dtype=np.uint8)
    masks[:, 1, 1, 8, 0] = 1
    masks[:, 2, 2, 0, 1] = 1
    masks[:, 3, 3, 10, 0] = 1
    masks[:, 4, 4, 1, 1] = 1
    variances = np.ones((1, 10, 10, 2), dtype=np.float64)
    psfs = np.zeros((3, 3, 1, 2), dtype=np.float64)
    psfs[1, 1, 0, :] = 1.0
    return XImageBundle(
        store_format="npy",
        store_path=Path("."),
        arrays={
            "images": images,
            "masks": masks,
            "sky": np.ones_like(images, dtype=np.uint8),
            "variances": variances,
            "psfs": psfs,
            "exp_times": np.asarray([1.0, 2.0], dtype=np.float64),
            "mcgauss": np.zeros((2, 3, 3), dtype=np.float64),
        },
        metadata={},
    )


def _config(
    *, mask_mode: str = "non_conservative", dilation_factor: int = 1
) -> ReconstructionConfig:
    return ReconstructionConfig.model_validate(
        {
            "backend": "torch",
            "algorithm": "l2",
            "data": {"base": "data"},
            "output": {"root_dir": "runs", "overwrite": True},
            "roi": {"center": [5, 5], "height": 4, "width": 4},
            "mask": {
                "mode": mask_mode,
                "dilation_factor": dilation_factor,
            },
            "srf": 1,
            "x_initialization": "median",
            "max_iter": 1,
            "padding": "VALID",
        }
    )


def _expected_inputs(bundle: XImageBundle, config: ReconstructionConfig):
    images, masks, variances = prepare_reconstruction_inputs(
        bundle,
        mask_mode=config.mask.mode,
        dilation_factor=config.mask.dilation_factor,
    )
    y, m, v = select_cutout(
        images,
        masks=masks,
        variances=variances,
        center_roi=list(config.roi.center),
        height_roi=config.roi.height,
        width_roi=config.roi.width,
    )
    y0, _, _ = pad_with_data(
        images,
        masks=masks,
        variances=variances,
        center_roi=list(config.roi.center),
        height_roi=config.roi.height,
        width_roi=config.roi.width,
        srf=config.srf,
        psf_size=bundle.psfs.shape[0],
    )
    x0 = coadd(y0, method=config.x_initialization, positive=True)
    return {
        "x0": np.transpose(x0, (0, 3, 1, 2)),
        "y": np.transpose(y, (0, 3, 1, 2)),
        "m": np.transpose(m, (0, 3, 1, 2)),
        "v": np.transpose(v, (0, 3, 1, 2)),
        "w": np.transpose(m / np.maximum(v, 1e-30), (0, 3, 1, 2)),
        "f": np.transpose(bundle.psfs, (3, 2, 0, 1)),
    }


@pytest.mark.parametrize(
    "mask_mode", ["none", "non_conservative", "conservative"]
)
@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)]
)
def test_torch_preprocess_matches_numpy_inputs(
    mask_mode: str, device: str
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    bundle = _bundle()
    config = _config(mask_mode=mask_mode)
    expected = _expected_inputs(bundle, config)

    actual = prepare_torch_reconstruction_inputs(
        bundle,
        config,
        device=torch.device(device),
        dtype=torch.float64,
    )

    np.testing.assert_allclose(
        actual.x0.cpu().numpy(), expected["x0"], atol=1e-12
    )
    np.testing.assert_allclose(
        actual.y.cpu().numpy(), expected["y"], atol=1e-12
    )
    np.testing.assert_allclose(
        actual.m.cpu().numpy(), expected["m"], atol=1e-12
    )
    np.testing.assert_allclose(
        actual.v.cpu().numpy(), expected["v"], atol=1e-12
    )
    np.testing.assert_allclose(
        actual.w.cpu().numpy(), expected["w"], atol=1e-12
    )
    np.testing.assert_allclose(
        actual.f.cpu().numpy(), expected["f"], atol=1e-12
    )
    assert actual.metadata["tensor_shapes"]["y"] == [1, 2, 4, 4]
    assert "unmasked_fraction" in actual.metadata["mask_summary"]


def test_torch_preprocess_rejects_unsupported_conservative_dilation() -> None:
    with pytest.raises(NotImplementedError, match="dilation_factor=1"):
        prepare_torch_reconstruction_inputs(
            _bundle(),
            _config(mask_mode="conservative", dilation_factor=2),
            device=torch.device("cpu"),
            dtype=torch.float64,
        )


def test_torch_preprocess_accepts_default_conservative_dilation() -> None:
    config = _config()
    config.mask = type(config.mask)(mode="conservative")
    prepared = prepare_torch_reconstruction_inputs(
        _bundle(), config, device=torch.device("cpu"), dtype=torch.float64
    )
    assert prepared.m.shape == (1, 2, 4, 4)
    assert prepared.m[0, 0, 0, 0] == 0


@pytest.mark.parametrize("center", [(1, 5), (9, 5), (5, 1), (5, 9)])
def test_torch_same_padding_rejects_roi_beyond_each_image_edge(center):
    config = _config().model_copy(update={"padding": "SAME"})
    config.roi.center = center
    with pytest.raises(ValueError, match="ROI exceeds image bounds"):
        prepare_torch_reconstruction_inputs(
            _bundle(), config, device=torch.device("cpu"), dtype=torch.float64
        )


def test_torch_same_padding_accepts_full_image_at_bounds() -> None:
    config = _config().model_copy(update={"padding": "SAME"})
    config.roi.height = 10
    config.roi.width = 10
    bundle = _bundle()
    prepared = prepare_torch_reconstruction_inputs(
        bundle, config, device=torch.device("cpu"), dtype=torch.float64
    )
    expected = np.transpose(
        bundle.images / bundle.exp_times.reshape(1, 1, 1, -1), (0, 3, 1, 2)
    )
    np.testing.assert_array_equal(prepared.y.numpy(), expected)


@pytest.mark.parametrize("height,width", [(5, 7), (4, 6), (5, 6), (4, 7)])
@pytest.mark.parametrize("srf", [1, 2, 3])
@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)]
)
def test_torch_roi_and_padding_preserve_requested_shape_and_pixels(
    height, width, srf, device
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    from cuphoton.ximage.backends.torch_gpu_preprocess import (
        _pad_with_data,
        _select_cutout,
    )

    images = torch.arange(
        20 * 24, dtype=torch.float64, device=device
    ).reshape(1, 20, 24, 1)
    start_y = 10 - height // 2
    start_x = 12 - width // 2
    cropped, _, _ = _select_cutout(
        torch, images, center_roi=(10, 12), height_roi=height, width_roi=width
    )
    torch.testing.assert_close(
        cropped,
        images[:, start_y : start_y + height, start_x : start_x + width],
    )
    padded, _, _ = _pad_with_data(
        torch,
        images,
        center_roi=(10, 12),
        height_roi=height,
        width_roi=width,
        srf=srf,
        psf_size=3,
    )
    upsampled = images.repeat_interleave(srf, dim=1).repeat_interleave(
        srf, dim=2
    )
    expected = upsampled[
        :,
        start_y * srf - 1 : (start_y + height) * srf + 1,
        start_x * srf - 1 : (start_x + width) * srf + 1,
    ]
    torch.testing.assert_close(padded, expected)


@pytest.mark.parametrize("center", [(2, 4), (6, 4), (4, 2), (4, 6)])
def test_torch_padding_rejects_roi_without_surrounding_pixels(center) -> None:
    from cuphoton.ximage.backends.torch_gpu_preprocess import _pad_with_data

    images = torch.ones((1, 8, 8, 1), dtype=torch.float64)
    with pytest.raises(ValueError, match="Padded ROI exceeds.*bounds"):
        _pad_with_data(
            torch,
            images,
            center_roi=center,
            height_roi=4,
            width_roi=4,
            psf_size=3,
        )


def test_torch_padding_rejects_full_image_with_nontrivial_psf() -> None:
    from cuphoton.ximage.backends.torch_gpu_preprocess import _pad_with_data

    with pytest.raises(ValueError, match="require 4 input pixels"):
        _pad_with_data(torch, torch.ones((1, 64, 64, 1)), psf_size=9)


@pytest.mark.parametrize("srf,psf_size", [(1, 1), (1, 9), (3, 9)])
def test_torch_padding_matches_numpy_at_source_boundary(
    srf, psf_size
) -> None:
    from cuphoton.ximage.backends.torch_gpu_preprocess import _pad_with_data

    images = np.arange(12 * 14, dtype=np.float64).reshape(1, 12, 14, 1)
    halo = (psf_size // 2 + srf - 1) // srf
    kwargs = {
        "height_roi": 12 - 2 * halo,
        "width_roi": 14 - 2 * halo,
        "srf": srf,
        "psf_size": psf_size,
    }
    actual, _, _ = _pad_with_data(torch, torch.from_numpy(images), **kwargs)
    expected, _, _ = pad_with_data(images, **kwargs)
    np.testing.assert_array_equal(actual.numpy(), expected)


@pytest.mark.parametrize("method", ["mean", "median"])
@pytest.mark.parametrize("value", [0.0, -2.0])
def test_positive_torch_coadd_handles_nonpositive_images(method, value):
    images = torch.full((1, 4, 4, 2), value, dtype=torch.float64)
    result = _coadd(torch, images, method=method, positive=True)
    assert bool(torch.all(result > 0))
    np.testing.assert_allclose(
        result.numpy(), np.full((1, 4, 4, 1), 1e-6 * (abs(value) or 1.0))
    )


def test_torch_same_padding_rejects_superresolution() -> None:
    config = _config().model_copy(update={"padding": "SAME", "srf": 2})
    with pytest.raises(ValueError, match="SAME padding requires srf=1"):
        prepare_torch_reconstruction_inputs(
            _bundle(), config, device=torch.device("cpu"), dtype=torch.float64
        )
