# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Native FITS planning, device reads, and pooled allocation ownership."""

from __future__ import annotations

import gc

import numpy as np
import pytest
from astropy.io import fits

from cuphoton.xdr import batch_to_device, nvcomp_batch


@pytest.fixture
def native():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("CUDA device unavailable")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA runtime unavailable")
    ext = nvcomp_batch._try_get_cpp_ext()
    if ext is None or not all(
        hasattr(ext, name)
        for name in (
            "plan_native_files",
            "NativeBatchBuilder",
            "acquire_native_device_buffer",
            "batch_deflate_decompress_pooled",
        )
    ):
        pytest.skip("native xDR extension unavailable")
    return cp, ext


def test_native_batch_reads_multiple_image_hdus(native, tmp_path):
    cp, _ = native
    paths, integer_images, float_images = [], [], []
    for index in range(3):
        integers = np.arange(63, dtype="i2").reshape(7, 9) - 31 + index
        floats = integers.astype("f4") * 0.125
        path = tmp_path / f"images-{index}.fits"
        fits.HDUList(
            [fits.PrimaryHDU(integers), fits.ImageHDU(floats)]
        ).writeto(path)
        paths.append(path)
        integer_images.append(integers)
        float_images.append(floats)

    result = batch_to_device(
        paths,
        hdu_indices=(0, 1),
        native_batcher=True,
        decode_batch_files=2,
        native_read_threads=2,
        native_plan_threads=2,
        stream=cp.cuda.Stream(non_blocking=True),
    )

    for actual, expected in zip(
        result, (integer_images, float_images), strict=True
    ):
        np.testing.assert_array_equal(cp.asnumpy(actual), np.stack(expected))


@pytest.mark.parametrize("compression", ["GZIP_1", "GZIP_2"])
def test_native_batch_reads_compressed_section(native, tmp_path, compression):
    cp, _ = native
    paths, expected = [], []
    section = (slice(2, 11), slice(3, 13))
    for index in range(3):
        pixels = np.arange(13 * 15, dtype="i4").reshape(13, 15) * 7 - index
        path = tmp_path / f"compressed-{index}.fits"
        fits.HDUList(
            [
                fits.PrimaryHDU(),
                fits.CompImageHDU(
                    pixels,
                    compression_type=compression,
                    tile_shape=(4, 6),
                ),
            ]
        ).writeto(path)
        paths.append(path)
        expected.append(pixels[section])

    (actual,) = batch_to_device(
        paths,
        section=section,
        native_batcher=True,
        decode_batch_files=2,
        native_read_threads=2,
        native_plan_threads=2,
        stream=cp.cuda.Stream(non_blocking=True),
    )

    np.testing.assert_array_equal(cp.asnumpy(actual), np.stack(expected))


def test_native_pool_retains_views_and_reuses_released_buffer(native):
    cp, ext = native
    device_id = int(cp.cuda.Device().id)
    ext.clear_native_device_pool(device_id)
    try:
        first = nvcomp_batch._native_device_empty_uint8(1024, ext=ext)
        first[:] = cp.arange(1024, dtype=cp.uint8)
        pointer = first.data.ptr
        view = first[13:29]
        del first
        ext.clear_native_device_pool(device_id)

        second = nvcomp_batch._native_device_empty_uint8(1024, ext=ext)
        assert second.data.ptr != pointer
        second.fill(0)
        np.testing.assert_array_equal(
            cp.asnumpy(view), np.arange(13, 29, dtype=np.uint8)
        )
        cp.cuda.get_current_stream().synchronize()
        del second
        ext.clear_native_device_pool(device_id)
        del view
        gc.collect()

        reused = nvcomp_batch._native_device_empty_uint8(512, ext=ext)
        assert reused.data.ptr == pointer
        assert reused.nbytes == 512
        del reused
    finally:
        cp.cuda.get_current_stream().synchronize()
        ext.clear_native_device_pool(device_id)
