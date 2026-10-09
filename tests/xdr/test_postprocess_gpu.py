# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Pixel-level checks of decoded FITS tiles, independent of the codec."""

from __future__ import annotations

import numpy as np
import pytest

from cuphoton.xdr.kernels import scatter_native_tiles_2d, scatter_tiles_2d
from cuphoton.xdr.reader import GpuCompImageReader


@pytest.fixture
def cp():
    module = pytest.importorskip("cupy")
    try:
        if module.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("CUDA device unavailable")
    except module.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA runtime unavailable")
    return module


@pytest.mark.parametrize("postprocess", ["auto", "fused", "separate"])
@pytest.mark.parametrize("dtype", ["u1", "i2", "i4", "i8", "f4", "f8"])
@pytest.mark.parametrize("compression", ["GZIP_1", "GZIP_2"])
@pytest.mark.parametrize("cropped", [False, True])
def test_scatter_restores_fits_pixels(
    cp, dtype, compression, cropped, postprocess
):
    rng = np.random.default_rng(42)
    # Arbitrary bit patterns exercise signed values, NaNs and integer masks.
    pixels = rng.bytes(17 * 19 * np.dtype(dtype).itemsize)
    image = np.frombuffer(pixels, dtype=dtype).reshape(17, 19)
    section = (slice(3, 15), slice(5, 18)) if cropped else np.s_[:, :]
    raw, offsets, plan = _decoded_tiles(image, compression, section)
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        device_raw = cp.asarray(raw)
        out = cp.empty(plan["out_shape"], dtype=dtype)
        result = GpuCompImageReader.postprocess_decoded_tiles(
            device_raw,
            offsets,
            plan,
            out=out,
            stream=stream,
            postprocess=postprocess,
        )
    stream.synchronize()

    assert result is out
    np.testing.assert_array_equal(
        cp.asnumpy(result).view("u1"), image[section].copy().view("u1")
    )
    np.testing.assert_array_equal(cp.asnumpy(device_raw), raw)


@pytest.mark.parametrize("postprocess", ["auto", "separate"])
@pytest.mark.parametrize("compression", ["GZIP_1", "GZIP_2"])
def test_scatter_preserves_nondithered_dequantization(
    cp, compression, postprocess
):
    # FITS quantization stores int32 tiles, including for float64 images.
    # This checks the supported float32 layout, not synthetic int64 tiles.
    image = np.arange(-150, 173, dtype="i4").reshape(17, 19)
    section = np.s_[3:15, 5:18]
    raw, offsets, plan = _decoded_tiles(image, compression, section)
    plan.update(
        quantized=True,
        dtype_char="f4",
        zbitpix=-32,
        sel_zscale=np.full(len(offsets), 0.5),
        sel_zzero=np.full(len(offsets), 2.0),
    )
    result = GpuCompImageReader.postprocess_decoded_tiles(
        cp.asarray(raw), offsets, plan, postprocess=postprocess
    )
    np.testing.assert_array_equal(
        cp.asnumpy(result), image[section].astype("f4") * 0.5 + 2
    )


@pytest.mark.parametrize(
    "scatter", [scatter_tiles_2d, scatter_native_tiles_2d]
)
def test_scatter_rejects_noncontiguous_output_pixels(scatter):
    # Validation must not depend on GPU imports or Python assertions.
    arguments = dict(
        d_tiles_concat=None,
        d_out=np.empty((2, 8), dtype="i4")[:, ::2],
        tile_byte_offsets=None,
        tile_origins_row=None,
        tile_origins_col=None,
        tile_heights=None,
        tile_widths=None,
        tile_src_off_row=None,
        tile_src_off_col=None,
        tile_full_widths=None,
        itemsize=4,
    )
    if scatter is scatter_tiles_2d:
        arguments.update(tile_byte_lengths=None, shuffled=False)
    with pytest.raises(ValueError, match="output pixels must be contiguous"):
        scatter(**arguments)


def test_scatter_respects_output_row_stride(cp):
    image = np.arange(17 * 19, dtype="i4").reshape(17, 19)
    raw, offsets, plan = _decoded_tiles(image, "GZIP_2", np.s_[:, :])
    padded = cp.full((17, 23), -1, dtype="i4")
    out = padded[:, 2:21]
    scatter_tiles_2d(
        cp.asarray(raw),
        out,
        cp.asarray(offsets),
        *(
            cp.asarray(plan[name])
            for name in (
                "out_bytes",
                "origins_r",
                "origins_c",
                "heights",
                "widths",
                "src_off_r",
                "src_off_c",
                "tile_full_w",
            )
        ),
        4,
        shuffled=True,
    )
    actual = cp.asnumpy(padded)
    np.testing.assert_array_equal(actual[:, 2:21], image)
    assert np.all(actual[:, :2] == -1)
    assert np.all(actual[:, 21:] == -1)


def _decoded_tiles(image, compression, section):
    """Encode byte order and byte planes independently of FITS/GPU code."""
    y0, y1, _ = section[0].indices(image.shape[0])
    x0, x1, _ = section[1].indices(image.shape[1])
    plan = {
        name: []
        for name in (
            "origins_r",
            "origins_c",
            "heights",
            "widths",
            "src_off_r",
            "src_off_c",
            "tile_full_w",
            "out_bytes",
        )
    }
    payloads = []
    for r in range(0, image.shape[0], 6):
        for c in range(0, image.shape[1], 8):
            tile = image[r : r + 6, c : c + 8]
            top, bottom = max(y0, r), min(y1, r + tile.shape[0])
            left, right = max(x0, c), min(x1, c + tile.shape[1])
            if top >= bottom or left >= right:
                continue
            big_endian = tile.astype(image.dtype.newbyteorder(">"))
            data = np.frombuffer(big_endian.tobytes(), dtype="u1")
            if compression == "GZIP_2":
                data = data.reshape(-1, image.itemsize).T.ravel()
            payloads.append(data)
            for name, value in zip(
                plan,
                (
                    top - y0,
                    left - x0,
                    bottom - top,
                    right - left,
                    top - r,
                    left - c,
                    tile.shape[1],
                    data.size,
                ),
                strict=True,
            ):
                plan[name].append(value)
    plan = {
        name: np.asarray(values, dtype="i8" if name == "out_bytes" else "i4")
        for name, values in plan.items()
    }
    offsets = np.cumsum(plan["out_bytes"], dtype="i8") - plan["out_bytes"]
    plan.update(
        out_shape=(y1 - y0, x1 - x0),
        itemsize=image.itemsize,
        dtype_char=image.dtype.str[1:],
        quantized=False,
        compression_type=compression,
    )
    return np.concatenate(payloads), offsets, plan
